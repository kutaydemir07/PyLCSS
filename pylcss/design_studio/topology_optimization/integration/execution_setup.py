# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validate graph inputs and construct a topology optimization problem."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pylcss.input_values import as_bool

from ..configuration.length_scale import (
    LengthScale,
    resolve_physical_length_scale,
)
from ..configuration.presets import migrate_design_goal
from ..manufacturing.cell_material import cell_material_law
from ..manufacturing.lattice_families import normalize_family_key
from ..models.study import ManufacturingConstraints, ThermalBC, VoxelBC
from ..optimization.problem import TopologyOptVoxelProblem
from .boundary_mapping import (
    _add_bc_contact_regions,
    _append_region_once,
    _bc_feature_bboxes,
    _bc_has_nonzero_load,
    _bc_has_support,
)
from .lattice_settings import (
    guided_lattice_voxel_dimensions,
    lattice_dimension_range_text,
    lattice_setup_is_guided,
    node_minimum_feature_size,
    resolve_lattice_structure_options,
)
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
    _thermal_bc_from_graph_payloads,
)
from .voxelization import (
    _cylinder_feature_lengths,
    _guided_voxel_grid,
    _mesh_design_domain_grid,
    _non_design_region_masks,
    _use_automatic_guided_grid,
    lattice_voxels_from_length,
    voxel_size_from_bounds,
)

logger = logging.getLogger(__name__)

Bounds = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PreparedTopologyStudy:
    """Validated solver inputs shared by execution and output recovery."""

    bounds: Bounds
    material: dict[str, Any]
    constraints: list[Any]
    loads: list[Any]
    bc: VoxelBC
    design_domain: np.ndarray
    design_goal: str
    passive_solid_mask: np.ndarray
    passive_void_mask: np.ndarray
    manufacturing: ManufacturingConstraints
    problem: TopologyOptVoxelProblem
    nelx: int
    nely: int
    nelz: int
    automatic_exclusion_scope: str
    automatic_exclusion_thickness: float | None
    lattice_porosity: str
    source_design_domain: Any = None
    non_design_regions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TopologyGraphInputs:
    """Normalized graph payloads and boundary conditions."""

    design_domain_input: Any
    bounds: Bounds
    material: dict[str, Any]
    constraints: list[Any]
    loads: list[Any]
    non_design_regions: list[Any]
    thermal_sinks: list[Any]
    thermal_loads: list[Any]
    bc: VoxelBC


@dataclass(frozen=True)
class TopologyDomainContext:
    """Voxelized domain and physics choices used to construct the problem."""

    graph: TopologyGraphInputs
    nelx: int
    nely: int
    nelz: int
    rmin: float
    filter_radius_is_physical: bool
    minimum_solid_size: float
    minimum_void_size: float
    eta_eroded: float
    eta_dilated: float
    design_domain: np.ndarray
    design_goal: str
    goal_key: str
    minimum_mass_goal: bool
    # Resolved once, including the migration of retired envelope goals, so the
    # solver and the iteration policy read the same decision.
    load_aggregation: str
    worst_case_envelope: bool
    physics_mode: str
    thermal_bc: ThermalBC
    passive_solid_mask: np.ndarray
    passive_void_mask: np.ndarray
    unitx: float
    unity: float
    unitz: float
    # A variable-density lattice needs a graded density field and therefore
    # solves a different continuum problem from solid reconstruction.
    lattice_study: bool
    # The requested structure mode, carried through so the solver can look up
    # that family's homogenized cell law and optimize the real cell.
    lattice_cell_type: str
    # Density ceiling of that cell, which bounds the design variable itself on a
    # lattice study. 1.0 for a solid study.
    lattice_maximum_density: float
    automatic_exclusion_scope: str
    automatic_exclusion_thickness: float | None


def _lattice_cell_physical_size(
    node: Any,
    grid_shape: tuple[int, int, int],
    bounds: Any,
) -> float | None:
    """Manufactured cell pitch in model units, or ``None`` for a solid study.

    Resolved separately from :func:`resolve_lattice_structure_options`, and
    earlier, because the load and support interface pads have to know it: a
    lattice needs a full cell of solid at every interface, and the pads are
    built before the manufacturing options are validated. Deliberately
    tolerant — an invalid lattice setting is reported by the full resolution a
    few lines later, and it must not surface here as a boundary-condition
    error.
    """
    family = normalize_family_key(
        str(node.get_property("structure_mode") or "Solid Envelope")
    )
    if not family or family == "solid":
        return None
    voxel_size = voxel_size_from_bounds(grid_shape, bounds)
    if not voxel_size or not np.isfinite(voxel_size) or voxel_size <= 0.0:
        return None
    try:
        if lattice_setup_is_guided(node.get_property("lattice_settings_mode")):
            cell_voxels, _, _ = guided_lattice_voxel_dimensions(
                family,
                voxel_size=voxel_size,
                minimum_feature_size=node_minimum_feature_size(node),
            )
        else:
            cell_voxels = lattice_voxels_from_length(
                node.get_property("lattice_cell_size_mm"),
                voxel_size,
                node.get_property("structure_cell_size_voxels") or 8.0,
            )
    except (TypeError, ValueError):
        return None
    cell_size = float(cell_voxels) * float(voxel_size)
    return cell_size if np.isfinite(cell_size) and cell_size > 0.0 else None


def _guided_cell_budget(
    structural_case_count: int,
    physics_mode: Any,
    *,
    idealized_joint_count: int = 0,
    base_cell_budget: int = 40_000,
    is_lattice: bool = False,
) -> int:
    """Return an interactive grid budget for the requested solve workload."""
    case_count = max(1, int(structural_case_count))
    base_budget = 80_000 if is_lattice else int(base_cell_budget)
    budget = max(12_000, int(base_budget / np.sqrt(float(case_count))))
    requested_physics = (
        str(physics_mode or "").strip().lower().replace("_", "-")
    )
    if (
        "thermo-mechanical" in requested_physics
        or "coupled structural + thermal" in requested_physics
    ):
        # Each iteration carries structural and thermal state/adjoint solves.
        budget = min(budget, max(12_000, int(0.30 * base_budget)))
    if int(idealized_joint_count) > 0:
        budget = min(budget, max(8_000, int(0.20 * base_budget)))
    return budget


def _bc_joint_count(bc: VoxelBC) -> int:
    """Count global and operating-case-local idealized joints."""
    return len(bc.joints) + sum(
        len(load_case.joints) for load_case in bc.load_cases
    )


def _bc_has_joint(bc: VoxelBC) -> bool:
    """Return whether a global or operating-case-local idealized joint exists."""
    return _bc_joint_count(bc) > 0


def _can_use_planar_extrusion_grid(
    bc: VoxelBC,
    extrusion_axis: int | None,
    physics_mode: Any,
) -> bool:
    """Return whether a four-layer extrusion grid preserves the study physics.

    Exact extrusion removes density variation along one axis.  When every
    support/load interface also spans that axis and no force acts through it,
    additional identical density layers only repeat the same in-plane problem.
    Retaining at least two layers keeps the tested 3-D element and multigrid
    path while avoiding the cost of redundant layers.  Any genuinely 3-D
    boundary condition, connector, point force, or thermal study disables the
    reduction.
    """
    if extrusion_axis not in {0, 1, 2}:
        return False
    physics_key = str(physics_mode or "Structural").strip().lower()
    if "thermal" in physics_key:
        return False
    if bc.point_forces:
        return False

    axis = int(extrusion_axis)
    face_supports = (
        (bc.fixed_left_face_dofs, bc.fixed_right_face_dofs),
        (bc.fixed_bottom_face_dofs, bc.fixed_top_face_dofs),
        (bc.fixed_front_face_dofs, bc.fixed_back_face_dofs),
    )
    if any(face_supports[axis]):
        return False

    def _box_spans_axis(values: Any) -> bool:
        try:
            lo = float(values[2 * axis])
            hi = float(values[2 * axis + 1])
        except (TypeError, ValueError, IndexError):
            return False
        lo, hi = sorted((lo, hi))
        return lo <= 1e-9 and hi >= 1.0 - 1e-9

    def _force_is_in_plane(values: Any) -> bool:
        try:
            vector = [float(values[6]), float(values[7]), float(values[8])]
        except (TypeError, ValueError, IndexError):
            return False
        scale = max(float(np.linalg.norm(vector)), 1.0)
        return abs(vector[axis]) <= 1e-12 * scale

    if any(not _box_spans_axis(box) for box in bc.fixed_boxes):
        return False
    if any(
        not _box_spans_axis(box) or not _force_is_in_plane(box)
        for box in bc.box_forces
    ):
        return False
    extrusion_faces = (
        {"left", "right"},
        {"bottom", "top"},
        {"front", "back"},
    )[axis]
    for face, fx, fy, fz in bc.distributed_forces:
        vector = (float(fx), float(fy), float(fz))
        if str(face).lower() in extrusion_faces or abs(vector[axis]) > 1e-12:
            return False

    for case in bc.load_cases:
        if case.joints or case.point_forces:
            return False
        if any(not _box_spans_axis(box) for box in case.fixed_boxes):
            return False
        if any(
            not _box_spans_axis(box) or not _force_is_in_plane(box)
            for box in case.box_forces
        ):
            return False
        for face, fx, fy, fz in case.distributed_forces:
            vector = (float(fx), float(fy), float(fz))
            if str(face).lower() in extrusion_faces or abs(vector[axis]) > 1e-12:
                return False
    return True


def _validate_multibody_joint_components(
    design_domain: np.ndarray,
    bc: VoxelBC,
    *,
    maximum_anchor_distance_voxels: float,
) -> str | None:
    """Require every multibody joint to connect two distinct source bodies.

    A bearing anchor usually lies at the centre of a bore and is therefore in
    void. Associate it with the nearest design-domain voxel before checking
    which connected source body owns the interface.
    """
    from scipy import ndimage as ndi

    source = np.asarray(design_domain, dtype=bool)
    connectivity = ndi.generate_binary_structure(3, 1)
    labels, component_count = ndi.label(source, structure=connectivity)
    if int(component_count) < 2:
        return (
            "Multi-body TopOpt needs at least two disconnected source bodies "
            "in the design domain. Keep physical gaps between bodies and "
            "connect them with Topology Joint nodes."
        )

    distance, nearest = ndi.distance_transform_edt(
        ~source,
        return_indices=True,
    )
    all_joints = list(bc.joints)
    for load_case in bc.load_cases:
        all_joints.extend(load_case.joints)

    shape = np.asarray(source.shape, dtype=int)
    for joint in all_joints:
        component_ids: list[int] = []
        for anchor_name, anchor in (
            ("A", joint.anchor_a),
            ("B", joint.anchor_b),
        ):
            index = np.rint(
                np.clip(np.asarray(anchor, dtype=float), 0.0, 1.0)
                * np.maximum(shape - 1, 0)
            ).astype(int)
            index = np.clip(index, 0, shape - 1)
            anchor_distance = float(distance[tuple(index)])
            if anchor_distance > float(maximum_anchor_distance_voxels):
                return (
                    f"Topology Joint {joint.name!r} anchor {anchor_name} is "
                    f"{anchor_distance:.1f} voxels from the design domain. "
                    "Select an interface on the intended source body."
                )
            nearest_index = tuple(
                int(nearest[axis][tuple(index)]) for axis in range(3)
            )
            component_ids.append(int(labels[nearest_index]))
        if component_ids[0] == component_ids[1]:
            return (
                f"Topology Joint {joint.name!r} has both anchors on source "
                f"body {component_ids[0]}. Select anchor A and anchor B on two "
                "different design-domain bodies."
            )
    return None


def _collect_graph_inputs(node: Any) -> TopologyGraphInputs | None:
    """Normalize connected graph payloads and validate mapped boundaries."""
    design_domain_input = node.get_input_value("design_domain", None)
    bounds = _mesh_bounds(design_domain_input)
    material = node.get_input_value("material", None)
    material = material if isinstance(material, dict) else {}
    # Every support and every force connected to the study is one scenario:
    # they act together. Independent operating cases, idealized joints, and
    # the thermal study were removed with their nodes.
    constraint_list = _flatten(node.get_input_list("supports"))
    load_list = _flatten(node.get_input_list("loads"))
    non_design_payloads: list[Any] = []
    thermal_sink_payloads: list[Any] = []
    thermal_load_payloads: list[Any] = []
    try:
        bc = node._build_bc()
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt boundary-condition settings: {exc}")
        return None
    if constraint_list:
        bc.fixed_left_face_dofs = []
        bc.fixed_right_face_dofs = []
        bc.fixed_top_face_dofs = []
        bc.fixed_bottom_face_dofs = []
        bc.fixed_front_face_dofs = []
        bc.fixed_back_face_dofs = []
        bc.fixed_boxes = []
    if load_list:
        bc.point_forces = []
        bc.box_forces = []
        bc.distributed_forces = []
    # One scenario per study. Anything a saved project carried in separate
    # operating cases or joints is dropped rather than silently reinterpreted:
    # folding several independent cases into one simultaneous load set is a
    # different structural problem, not the same one written differently.
    bc.joints = []
    bc.load_cases = []
    node._merge_graph_bcs(
        bc,
        design_domain_input,
        constraint_list,
        load_list,
    )
    if constraint_list and not _bc_has_support(bc):
        node.set_error(
            "Connected constraints could not be mapped to voxel supports. "
            "Re-run the geometry selection on the current CAD shape, then "
            "run TopOpt again."
        )
        return None
    if load_list and not _bc_has_nonzero_load(bc):
        node.set_error(
            "Connected loads could not be mapped to a non-zero voxel force. "
            "Check the selected load geometry and force components, then run "
            "TopOpt again."
        )
        return None

    if bounds is None:
        node.set_error(
            "Topology Opt could not read a 3-D design volume. Connect a CAD "
            "solid or watertight imported surface directly to design_domain."
        )
        return None

    return TopologyGraphInputs(
        design_domain_input=design_domain_input,
        bounds=bounds,
        material=material,
        constraints=constraint_list,
        loads=load_list,
        non_design_regions=non_design_payloads,
        thermal_sinks=thermal_sink_payloads,
        thermal_loads=thermal_load_payloads,
        bc=bc,
    )


def _prepare_domain(
    node: Any,
    graph: TopologyGraphInputs,
) -> TopologyDomainContext | None:
    """Resolve a safe voxel grid, physics mode, and passive masks."""
    design_domain_input = graph.design_domain_input
    bounds = graph.bounds
    constraint_list = graph.constraints
    load_list = graph.loads
    non_design_payloads = graph.non_design_regions
    thermal_sink_payloads = graph.thermal_sinks
    thermal_load_payloads = graph.thermal_loads
    bc = graph.bc
    try:
        nelx = int(node.get_property("nelx") or 30)
        nely = int(node.get_property("nely") or 20)
        nelz = int(node.get_property("nelz") or 10)
    except (TypeError, ValueError):
        node.set_error("Topology grid dimensions must be integers.")
        return None
    guided_active = _use_automatic_guided_grid(node.get_property("workflow_mode"))
    try:
        requested_member = float(
            node.get_property("minimum_member_size_mm") or 0.0
        )
        requested_void = float(
            node.get_property("minimum_void_size_mm") or 0.0
        )
    except (TypeError, ValueError):
        node.set_error("Minimum solid and void sizes must be numeric.")
        return None
    extrusion_axis = {
        "x": 0,
        "y": 1,
        "z": 2,
    }.get(str(node.get_property("extrusion") or "none").lower())
    planar_extrusion_axis = (
        extrusion_axis
        if guided_active
        and _can_use_planar_extrusion_grid(
            bc,
            extrusion_axis,
            node.get_property("physics_mode"),
        )
        else None
    )
    setattr(node, "_guided_planar_extrusion_axis", planar_extrusion_axis)
    if guided_active:
        # Guided mode owns its resolution and length scale. Legacy saved
        # element counts and voxel-radius values must never silently change a
        # guided solve. Physical member/void sizes and the selected fidelity
        # are engineering requirements, however, so they remain authoritative.
        # Large studies are solved by multigrid-preconditioned CG, whose
        # memory is linear in the number of voxels, so the budget no longer
        # has to be set by sparse-factorization fill-in. It still shrinks as
        # pose/load cases are added, because each case carries its own solve.
        structural_case_count = max(1, len(bc.load_cases))
        family_mode = normalize_family_key(
            str(node.get_property("structure_mode") or "Solid Envelope")
        )
        is_lattice_study = bool(family_mode and family_mode != "solid")
        guided_cell_budget = _guided_cell_budget(
            structural_case_count,
            node.get_property("physics_mode"),
            is_lattice=is_lattice_study,
        )
        feature_lengths = _cylinder_feature_lengths(
            bounds,
            solid_cylinders=bc.solid_cylinders,
            void_cylinders=bc.void_cylinders,
        )
        feature_lengths.extend(
            value for value in (requested_member, requested_void) if value > 0.0
        )
        # The stated minimum member/void sizes are engineering requirements,
        # not features that are merely nice to resolve: the length scale
        # rejects a grid too coarse to represent them. Give the sizer the
        # binding one so it cannot return a grid that is refused a step later.
        stated_sizes = [
            value for value in (requested_member, requested_void) if value > 0.0
        ]
        guided_grid = _guided_voxel_grid(
            bounds,
            feature_bboxes=_bc_feature_bboxes(constraint_list, load_list),
            feature_lengths=feature_lengths,
            max_total_cells=guided_cell_budget,
            inactive_axes=(
                (planar_extrusion_axis,)
                if planar_extrusion_axis is not None
                else ()
            ),
            required_feature_size=min(stated_sizes) if stated_sizes else None,
        )
        if guided_grid is not None:
            nelx, nely, nelz = guided_grid

    structural_case_count = max(1, len(bc.load_cases))
    safe_cell_cap = max(
        40_000,
        int(160_000 / np.sqrt(float(structural_case_count))),
    )
    if min(nelx, nely, nelz) < 1 or nelx * nely * nelz > safe_cell_cap:
        node.set_error(
            "Topology grid dimensions must be positive and contain no more "
            f"than {safe_cell_cap:,} voxels for this number of structural "
            "load/pose cases. Use Guided mode for safe automatic sizing; "
            "validate the recovered design with an independent refined mesh."
        )
        return None

    rmin_effective = float(node.get_property("rmin") or 1.5)
    filter_radius_is_physical = False
    length_scale = LengthScale(
        filter_radius=rmin_effective,
        minimum_solid_size=0.0,
        minimum_void_size=0.0,
    )
    if guided_active:
        try:
            length_scale = resolve_physical_length_scale(
                bounds,
                (nelx, nely, nelz),
                requested_member if requested_member > 0.0 else None,
                requested_void if requested_void > 0.0 else None,
                inactive_axes=(
                    (extrusion_axis,)
                    if extrusion_axis is not None
                    else ()
                ),
            )
        except ValueError as exc:
            node.set_error(f"Invalid topology length scale: {exc}")
            return None
        rmin_effective = float(length_scale.filter_radius)
        filter_radius_is_physical = True
        setattr(
            node,
            "_resolved_minimum_member_size",
            float(length_scale.minimum_solid_size),
        )
        setattr(
            node,
            "_resolved_minimum_void_size",
            float(length_scale.minimum_void_size),
        )
    design_domain = _mesh_design_domain_grid(
        design_domain_input,
        bounds,
        nelx,
        nely,
        nelz,
    )
    if design_domain is None or not np.any(design_domain):
        node.set_error(
            "Topology Opt could not voxelize the design domain. Repair a "
            "non-watertight imported surface or connect a valid CAD solid."
        )
        return None

    design_goal, legacy_load_aggregation = migrate_design_goal(
        node.get_property("design_goal") or "Lightweight Stiffness"
    )
    goal_key = design_goal.strip().lower()
    load_aggregation_setting = str(
        legacy_load_aggregation
        or node.get_property("load_aggregation")
        or "Weighted Sum"
    )
    # A smooth worst-case (P-norm) envelope converges more slowly than a
    # weighted sum, so the iteration floor follows the aggregation rather than
    # a goal name.
    worst_case_envelope = (
        load_aggregation_setting.strip().lower().replace(" ", "_").replace("-", "_")
        == "worst_case"
    )
    minimum_mass_goal = goal_key == "minimum mass under stress"
    physics_mode = (
        str(node.get_property("physics_mode") or "Structural")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if physics_mode in {
        "thermo_mechanical",
        "coupled structural + thermal",
    }:
        # Public wording is intentionally explicit: this is a weighted
        # structural/conduction objective, not thermal-expansion coupling.
        physics_mode = "thermo_mechanical"
    if str(node.get_property("workflow_mode") or "Guided").lower() == "guided":
        if goal_key == "thermal conduction":
            physics_mode = "thermal"
        elif goal_key in {
            "thermo-mechanical",
            "coupled structural + thermal",
        }:
            physics_mode = "thermo_mechanical"
        elif goal_key in {
            "lightweight stiffness",
            "minimum mass under stress",
        }:
            physics_mode = "structural"
    try:
        if thermal_sink_payloads or thermal_load_payloads:
            thermal_bc = _thermal_bc_from_graph_payloads(
                thermal_sink_payloads,
                thermal_load_payloads,
                bounds,
            )
        else:
            thermal_bc = ThermalBC()
        thermal_bc.convection_coefficient = float(
            node.get_property("convection_coefficient") or 0.0
        )
        # Re-validate: the coefficient is assigned after construction.
        thermal_bc.__post_init__()
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt thermal setup: {exc}")
        return None
    uses_structural = physics_mode in {"structural", "thermo_mechanical"}
    uses_thermal = physics_mode in {"thermal", "thermo_mechanical"}
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    if not is_infill_study and uses_structural and not _bc_has_support(bc):
        node.set_error(
            "Structural TopOpt needs a connected Topology Support, or a "
            "Topology Load Case containing one."
        )
        return None
    try:
        (
            explicit_passive_solid_mask,
            explicit_passive_void_mask,
        ) = _non_design_region_masks(
            non_design_payloads,
            bounds,
            nelx,
            nely,
            nelz,
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid non-design region: {exc}")
        return None
    if not is_infill_study and uses_structural and not _bc_has_nonzero_load(bc):
        node.set_error(
            "Structural TopOpt needs a connected Topology Force, or a "
            "Topology Load Case containing one."
        )
        return None
    if not is_infill_study and uses_thermal and not (thermal_bc.fixed_faces or thermal_bc.fixed_boxes):
        node.set_error(
            "Thermal TopOpt needs at least one connected Reference Temperature."
        )
        return None
    if not is_infill_study and uses_thermal and not thermal_bc.load_cases:
        node.set_error(
            "Thermal TopOpt needs at least one connected Topology Heat Load."
        )
        return None
    unitx = unity = unitz = 1.0
    if bounds is not None:
        mins, maxs = bounds
        span = np.maximum(maxs[:3] - mins[:3], 1e-12)
        unitx = float(span[0] / max(nelx, 1))
        unity = float(span[1] / max(nely, 1))
        unitz = float(span[2] / max(nelz, 1))
    else:
        span = np.ones(3, dtype=float)

    # Resolved before the exclusion pads, because a lattice study's interface
    # pad has to be at least one cell deep — see `_add_bc_contact_regions`.
    lattice_interface_cell_size = _lattice_cell_physical_size(
        node, (nelx, nely, nelz), bounds
    )

    exclusion_scope = str(
        node.get_property("exclusion_scope") or "All Loads and Supports"
    )
    exclusion_mode = str(
        node.get_property("exclusion_thickness_mode") or "Program Controlled"
    )
    try:
        manual_exclusion = None
        if exclusion_mode.strip().lower() == "manual":
            manual_exclusion = float(
                node.get_property("exclusion_thickness_mm") or 0.0
            )
            if manual_exclusion <= 0.0:
                raise ValueError(
                    "Manual exclusion thickness must be greater than 0 mm."
                )
        elif exclusion_mode.strip().lower() != "program controlled":
            raise ValueError(
                "Exclusion thickness mode must be Program Controlled or Manual."
            )
        effective_exclusion = _add_bc_contact_regions(
            bc,
            span,
            (nelx, nely, nelz),
            scope=exclusion_scope,
            manual_thickness=manual_exclusion,
            design_domain=design_domain,
            lattice_cell_size=lattice_interface_cell_size,
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt support/load exclusion settings: {exc}")
        return None

    structure_mode_value = str(node.get_property("structure_mode") or "Solid Envelope")
    # Ask the registry, not the display string. "Gyroid Network (Skeletal)" is
    # every bit as much a lattice study as "Gyroid Lattice", and a substring
    # test on the word "lattice" would quietly send it down the solid path.
    structure_family = normalize_family_key(structure_mode_value)
    lattice_maximum_density = 1.0
    if structure_family and structure_family != "solid":
        voxel_size = voxel_size_from_bounds(
            (nelx, nely, nelz),
            bounds,
        )
        setattr(node, "_resolved_lattice_voxel_size", voxel_size)
        try:
            # Validate manufacturing dimensions before starting the costly
            # optimization. Output recovery resolves them again in case a
            # convergence study finishes on a refined grid.
            resolved_options = resolve_lattice_structure_options(
                node,
                (nelx, nely, nelz),
                bounds,
            )
            # The cell's density ceiling bounds the *design variable*, not just
            # the reconstruction: see `TopologyOptVoxelProblem`. Resolving it
            # here keeps one source for the number the solve and the build both
            # use, so a delivered lattice cannot be denser than the cell the
            # macro tensor was measured from.
            lattice_maximum_density = float(
                resolved_options.maximum_relative_density
            )
            if callable(getattr(node, "resolve_infill_cell_size", None)):
                # Lattice infill does not optimize, so there is no design
                # variable to bound. Its density field is the constant 1.0
                # meaning "this is all part", and the lattice density it wants
                # is carried by the target instead.
                lattice_maximum_density = 1.0
        except (TypeError, ValueError) as exc:
            hint = ""
            if not lattice_setup_is_guided(
                node.get_property("lattice_settings_mode")
            ):
                hint = " " + lattice_dimension_range_text(
                    structure_mode_value,
                    voxel_size,
                    cell_size=node.get_property("lattice_cell_size_mm"),
                )
            node.set_error(
                f"Invalid lattice manufacturing settings: {exc}{hint}"
            )
            return None

    return TopologyDomainContext(
        graph=graph,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        rmin=rmin_effective,
        filter_radius_is_physical=filter_radius_is_physical,
        minimum_solid_size=float(length_scale.minimum_solid_size),
        minimum_void_size=float(length_scale.minimum_void_size),
        eta_eroded=float(length_scale.eta_eroded),
        eta_dilated=float(length_scale.eta_dilated),
        design_domain=design_domain,
        design_goal=design_goal,
        goal_key=goal_key,
        minimum_mass_goal=minimum_mass_goal,
        load_aggregation=load_aggregation_setting,
        worst_case_envelope=worst_case_envelope,
        physics_mode=physics_mode,
        thermal_bc=thermal_bc,
        passive_solid_mask=explicit_passive_solid_mask,
        passive_void_mask=explicit_passive_void_mask,
        unitx=unitx,
        unity=unity,
        unitz=unitz,
        lattice_study=bool(structure_family) and structure_family != "solid",
        lattice_cell_type=structure_family,
        lattice_maximum_density=lattice_maximum_density,
        automatic_exclusion_scope=exclusion_scope,
        automatic_exclusion_thickness=effective_exclusion,
    )


def _build_problem(
    node: Any,
    domain: TopologyDomainContext,
) -> PreparedTopologyStudy | None:
    """Validate material/optimizer settings and instantiate the solver problem."""
    graph = domain.graph
    bounds = graph.bounds
    material = graph.material
    constraint_list = graph.constraints
    load_list = graph.loads
    bc = graph.bc
    nelx, nely, nelz = domain.nelx, domain.nely, domain.nelz
    rmin_effective = domain.rmin
    design_domain = domain.design_domain
    design_goal = domain.design_goal
    goal_key = domain.goal_key
    minimum_mass_goal = domain.minimum_mass_goal
    load_aggregation_setting = domain.load_aggregation
    worst_case_envelope = domain.worst_case_envelope
    physics_mode = domain.physics_mode
    thermal_bc = domain.thermal_bc
    explicit_passive_solid_mask = domain.passive_solid_mask
    explicit_passive_void_mask = domain.passive_void_mask
    unitx, unity, unitz = domain.unitx, domain.unity, domain.unitz
    try:
        mc = ManufacturingConstraints(
            symmetry=(str(node.get_property("symmetry") or "None")).lower(),
            extrusion=(str(node.get_property("extrusion") or "None")).lower(),
            overhang_build_axis=(
                str(node.get_property("overhang_build_axis") or "None")
            ).lower(),
            overhang_angle_deg=float(
                node.get_property("overhang_angle_deg") or 45.0
            ),
            pull_out_axis=(
                str(node.get_property("pull_out_axis") or "None")
            ).lower(),
            max_member_size_voxels=float(
                node.get_property("max_member_size_voxels") or 0.0
            ),
            max_member_size_physical=float(
                node.get_property("maximum_member_size_mm") or 0.0
            ),
            pattern_repeat=int(node.get_property("pattern_repeat") or 1),
            pattern_axis=(str(node.get_property("pattern_axis") or "Y")).lower(),
        )
    except (TypeError, ValueError):
        node.set_error(
            "Topology manufacturing settings must be numeric where required."
        )
        return None

    stress_enabled = (
        as_bool(node.get_property("stress_constraint")) or minimum_mass_goal
    )
    optimizer = str(node.get_property("optimizer") or "Auto")
    if optimizer.strip().lower() == "projected gradient":
        optimizer = "PGD"
    if stress_enabled and optimizer.strip().upper() not in {"MMA", "GCMMA"}:
        optimizer = "GCMMA"
    try:
        yield_stress = float(node.get_property("yield_stress") or 0.0)
        if yield_stress <= 0.0:
            yield_stress = float(material.get("yield_strength") or 0.0)
    except (TypeError, ValueError):
        node.set_error("Topology allowable/yield stress must be numeric.")
        return None
    if stress_enabled and yield_stress <= 0.0:
        node.set_error(
            "Stress-constrained TopOpt needs a positive allowable/yield stress in MPa."
        )
        return None
    try:
        youngs_modulus = float(material.get("E", node.get_property("E0") or 1.0))
        minimum_stiffness = max(
            float(node.get_property("Emin") or 1e-9),
            youngs_modulus * 1e-9,
        )
        thermal_conductivity = float(
            material.get(
                "thermal_conductivity",
                node.get_property("thermal_conductivity") or 1.0,
            )
        )
        minimum_thermal_conductivity = max(
            float(node.get_property("thermal_conductivity_min") or 1e-6),
            thermal_conductivity * 1e-9,
        )
        # A solid study and a lattice study are different design problems, not
        # two renderings of one result.
        #
        # SIMP penalization exists to drive the density to 0/1, but a graded
        # lattice *is* the intermediate density — so the two work against each
        # other. With penal=3 and the Heaviside projection on, a converged
        # design reaches rho ~ 1 throughout the structure, every voxel clears
        # `solid_transition_density`, and the "lattice" comes out fully solid:
        # measured on a converged 58x38x16 field, 100% of the envelope became
        # solid and no lattice was generated at all.
        #
        # Lattice studies retain intermediate density so that it can size the
        # explicit cells. Families with a homogenized cell law no longer need a
        # chosen exponent at all: the macro operator uses that cell's measured
        # anisotropic tensor, so the density-to-stiffness relation is whatever
        # the cell actually is. The porosity exponents below remain for the
        # families that have no law (honeycomb), where they are still an
        # isotropic continuum assumption. Explicit recovered-geometry
        # reanalysis remains mandatory for every option.
        lattice_study = domain.lattice_study
        lattice_porosity = (
            "Conservative"
            if lattice_study
            and lattice_setup_is_guided(
                node.get_property("lattice_settings_mode")
            )
            else str(
                node.get_property("lattice_porosity") or "Conservative"
            ).strip()
        )
        material_penal = float(node.get_property("penal") or 3.0)
        study_poisson = float(material.get("nu", node.get_property("nu") or 0.3))
        # Mirrors TopologyOptVoxelProblem.homogenized_cell_law: a stress
        # constraint keeps the study on the isotropic surrogate, so the
        # exponent below is still the physics and must stay reported as such.
        homogenized_law = (
            cell_material_law(domain.lattice_cell_type, poisson=study_poisson)
            if lattice_study and not stress_enabled
            else None
        )
        if lattice_study:
            porosity_penalties = {
                "maximum porosity (concept)": 1.0,
                "balanced (concept)": 1.25,
                "conservative": 1.8,
                # Backward compatibility for previously saved studies.
                "high": 1.0,
                "medium": 1.25,
                "low": 1.8,
                "low (natural)": 1.8,
            }
            porosity_key = lattice_porosity.lower()
            if porosity_key not in porosity_penalties:
                raise ValueError(
                    "Lattice continuum assumption must be Conservative, "
                    "Balanced (Concept), or Maximum Porosity (Concept)."
                )
            material_penal = porosity_penalties[porosity_key]
            if homogenized_law is not None:
                # The exponent is unused on this path; say so rather than
                # leaving a stale value in the report to be read as physics.
                lattice_porosity = (
                    f"Homogenized {homogenized_law.cell_type} cell law "
                    "(continuum exponent not used)"
                )
        else:
            lattice_porosity = "Not applicable"
        guided_mode = (
            str(node.get_property("workflow_mode") or "Guided")
            .strip()
            .lower()
            == "guided"
        )
        if guided_mode:
            # Hidden legacy solver controls must not silently turn a guided
            # study into a long expert run. Convergence is not a user-facing
            # preference either: a solver should converge rather than be told
            # to stop early, so guided studies always use the strict budget.
            solver_max_iter, solver_tol, solver_patience = (120, 0.0035, 6)
            if goal_key == "minimum mass under stress":
                solver_max_iter = max(solver_max_iter, 120)
            elif goal_key == "coupled structural + thermal" or worst_case_envelope:
                solver_max_iter = max(solver_max_iter, 100)
        else:
            solver_max_iter = int(node.get_property("max_iter") or 80)
            solver_tol = float(node.get_property("tol") or 0.01)
            solver_patience = int(
                node.get_property("convergence_patience") or 5
            )
        problem = TopologyOptVoxelProblem(
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            E0=youngs_modulus,
            Emin=minimum_stiffness,
            nu=study_poisson,
            penal=material_penal,
            lattice_cell_type=domain.lattice_cell_type if lattice_study else "",
            lattice_maximum_density=(
                domain.lattice_maximum_density if lattice_study else 1.0
            ),
            volfrac=float(node.get_property("volfrac") or 0.5),
            rmin=rmin_effective,
            filter_radius_is_physical=domain.filter_radius_is_physical,
            minimum_solid_size=domain.minimum_solid_size,
            minimum_void_size=domain.minimum_void_size,
            # The requested sizes only bind through the robust formulation, so
            # a study that resolved a physical length scale is solved with it.
            # A lattice study is excluded with the projection it depends on:
            # its graded intermediate density is the design, not a defect to
            # be pushed to 0/1.
            robust_length_scale=bool(
                domain.filter_radius_is_physical and not lattice_study
            ),
            eta_eroded=domain.eta_eroded,
            eta_dilated=domain.eta_dilated,
            unitx=unitx,
            unity=unity,
            unitz=unitz,
            optimizer=optimizer,
            formulation="density",
            max_iter=solver_max_iter,
            tol=solver_tol,
            patience=solver_patience,
            bc=bc,
            mc=mc,
            design_domain=design_domain,
            passive_solid_mask=explicit_passive_solid_mask,
            passive_void_mask=explicit_passive_void_mask,
            objective_mode="minimum_mass" if minimum_mass_goal else "compliance",
            physics_mode=physics_mode,
            thermal_bc=thermal_bc,
            thermal_conductivity=thermal_conductivity,
            thermal_conductivity_min=minimum_thermal_conductivity,
            thermal_penal=float(node.get_property("thermal_penal") or 3.0),
            structural_weight=float(
                node.get_property("structural_weight")
                if node.get_property("structural_weight") is not None
                else 1.0
            ),
            thermal_weight=float(
                node.get_property("thermal_weight")
                if node.get_property("thermal_weight") is not None
                else 1.0
            ),
            load_aggregation=load_aggregation_setting,
            load_pnorm_p=float(node.get_property("load_pnorm_p") or 8.0),
            stress_constraint_enabled=stress_enabled,
            yield_stress=yield_stress,
            # The projection drives the field to 0/1, which is what a solid
            # study wants and exactly what a graded lattice must not have.
            heaviside_enabled=not lattice_study,
            # Numerical hyperparameters use the dataclass defaults (industrial
            # values: q=0.5 Bruggi qp-approach, p=8.0 PNorm aggregation,
            # Heaviside three-field SIMP on with β: 1 → 16 and physically
            # calibrated dilated/nominal/eroded thresholds. These are not
            # fragile numerical knobs in the guided workflow.
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt solver settings: {exc}")
        return None

    return PreparedTopologyStudy(
        bounds=bounds,
        material=material,
        constraints=constraint_list,
        loads=load_list,
        bc=bc,
        design_domain=design_domain,
        design_goal=design_goal,
        passive_solid_mask=explicit_passive_solid_mask,
        passive_void_mask=explicit_passive_void_mask,
        manufacturing=mc,
        problem=problem,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        automatic_exclusion_scope=domain.automatic_exclusion_scope,
        automatic_exclusion_thickness=domain.automatic_exclusion_thickness,
        lattice_porosity=lattice_porosity,
        source_design_domain=graph.design_domain_input,
        non_design_regions=tuple(graph.non_design_regions),
    )


def prepare_topology_study(node: Any) -> PreparedTopologyStudy | None:
    """Map graph inputs to a validated, solver-ready study."""
    graph = _collect_graph_inputs(node)
    if graph is None:
        return None
    domain = _prepare_domain(node, graph)
    if domain is None:
        return None
    return _build_problem(node, domain)
