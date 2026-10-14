# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validate graph inputs and construct a topology optimization problem."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pylcss.input_values import as_bool

from ..configuration.length_scale import (
    LengthScale,
    resolve_physical_length_scale,
)
from ..configuration.presets import migrate_design_goal
from ..configuration.voxel_meshing import (
    VoxelMeshPlan,
    voxel_mesh_plan,
    available_voxel_memory_bytes,
)
from ..manufacturing.lattice_families import normalize_family_key
from ..models.study import ManufacturingConstraints, VoxelBC
from ..optimization.problem import TopologyOptVoxelProblem
from .boundary_mapping import (
    _bc_feature_bboxes,
    _bc_has_nonzero_load,
    _bc_has_support,
)
from .lattice_settings import (
    lattice_dimension_range_text,
    lattice_setup_is_guided,
    resolve_lattice_structure_options,
)
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
)
from .voxelization import (
    _cylinder_feature_lengths,
    _mesh_design_domain_grid,
    _non_design_region_masks,
    voxel_size_from_bounds,
)
from .voxel_grid_mapping import rebase_voxel_boundary_conditions

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
    lattice_porosity: str
    source_design_domain: Any = None
    non_design_regions: tuple[Any, ...] = ()
    voxel_pitch: float | None = None
    max_voxel_elements: int = 0


@dataclass(frozen=True)
class TopologyGraphInputs:
    """Normalized graph payloads and boundary conditions."""

    design_domain_input: Any
    bounds: Bounds
    material: dict[str, Any]
    constraints: list[Any]
    loads: list[Any]
    non_design_regions: list[Any]
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
    passive_solid_mask: np.ndarray
    passive_void_mask: np.ndarray
    unitx: float
    unity: float
    unitz: float




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
    boundary condition, connector, or point force disables the
    reduction.
    """
    if extrusion_axis not in {0, 1, 2}:
        return False
    physics_key = str(physics_mode or "Structural").strip().lower()
    if physics_key != "structural":
        return False
    if bc.point_forces or bc.boundary_forces or bc.boundary_supports:
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
        if case.joints or case.point_forces or case.boundary_forces or case.boundary_supports:
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
    # they act together. Independent operating cases and idealized joints
    # were removed with their nodes.
    constraint_list = _flatten(node.get_input_list("supports"))
    load_list = _flatten(node.get_input_list("loads"))
    from ..geometry.non_design import non_design_payloads

    try:
        non_design_regions = non_design_payloads(node, design_domain_input)
    except ValueError as exc:
        node.set_error(f"Invalid non-design region: {exc}")
        return None
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
        bc.boundary_supports = []
    if load_list:
        bc.point_forces = []
        bc.box_forces = []
        bc.distributed_forces = []
        bc.boundary_forces = []
    # One scenario per study. Anything a saved project carried in separate
    # operating cases or joints is dropped rather than silently reinterpreted:
    # folding several independent cases into one simultaneous load set is a
    # different structural problem, not the same one written differently.
    bc.joints = []
    bc.load_cases = []
    try:
        node._merge_graph_bcs(
            bc,
            design_domain_input,
            constraint_list,
            load_list,
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid connected TopOpt boundary condition: {exc}")
        return None
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
        non_design_regions=non_design_regions,
        bc=bc,
    )


def plan_topology_voxel_mesh(node: Any, bounds: Bounds, *, bc=None, constraints=None, loads=None) -> VoxelMeshPlan:
    """Shared cheap inspector/backend plan; no geometry rasterization occurs."""
    lengths = _cylinder_feature_lengths(
        bounds, solid_cylinders=bc.solid_cylinders if bc else [],
        void_cylinders=bc.void_cylinders if bc else [],
    )
    if as_bool(node.get_property("non_design_enabled")):
        from ..geometry.non_design import read_non_design_regions
        lengths.extend(float(region["thickness_mm"]) for region in
                       read_non_design_regions(node.get_property("non_design_regions") or "[]"))
    automatic = node.get_property("automatic_grid")
    available_memory = available_voxel_memory_bytes()
    return voxel_mesh_plan(
        bounds, node.get_property("voxel_mesh_quality"),
        voxel_size=float(node.get_property("voxel_size") or 0),
        automatic=automatic is None or as_bool(automatic),
        manual_shape=tuple(node.get_property(key) or default for key, default in
                           zip(("nelx", "nely", "nelz"), (30, 20, 10))),
        max_elements=node.get_property("max_voxel_elements"),
        feature_bboxes=_bc_feature_bboxes(
            constraints if constraints is not None else _flatten(node.get_input_list("supports")),
            loads if loads is not None else _flatten(node.get_input_list("loads")),
        ),
        feature_lengths=lengths,
        available_memory_bytes=available_memory,
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
    bc = graph.bc
    try:
        plan = plan_topology_voxel_mesh(node, bounds, bc=bc, constraints=constraint_list, loads=load_list)
    except (TypeError, ValueError, OverflowError) as exc:
        node.set_error(f"Invalid voxel meshing: {exc}")
        return None
    nelx, nely, nelz = plan.shape
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    node._resolved_voxel_meshing = {
        "mode": plan.mode, "quality": plan.quality,
        "requested_voxel_size": float(node.get_property("voxel_size") or 0),
        "voxel_size": plan.pitch, "grid": list(plan.shape),
        "voxel_edges": plan.voxel_edges, "elements": plan.elements,
        "max_elements": plan.element_limit,
        "estimated_memory_bytes": plan.estimated_memory_bytes,
        "source_bounds": [value.tolist() for value in plan.source_bounds],
        "grid_bounds": [value.tolist() for value in plan.bounds],
        "automatic_refinement": as_bool(node.get_property("automatic_mesh_refinement")),
        "geometric_target_pitch": plan.geometric_target_pitch,
        "automatic_element_budget": plan.automatic_element_budget,
        "resource_limited": plan.resource_limited,
    }
    if not is_infill_study and plan.resource_error():
        node.set_error(plan.resource_error())
        return None
    if plan.pitch is not None:
        bc = rebase_voxel_boundary_conditions(bc, graph.bounds, plan.bounds)
        bounds = plan.bounds
        graph = replace(graph, bounds=bounds, bc=bc)
    requested_member = requested_void = 0.0
    extrusion_axis = None
    setattr(node, "_guided_planar_extrusion_axis", None)

    rmin_effective = float(node.get_property("rmin") or 1.5)
    filter_radius_is_physical = False
    length_scale = LengthScale(
        filter_radius=rmin_effective,
        minimum_solid_size=0.0,
        minimum_void_size=0.0,
    )
    if node.get_property("automatic_filter_radius") is not False:
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
    # Retired project values cannot switch this structural node's analysis.
    physics_mode = "structural"
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    if not is_infill_study and not _bc_has_support(bc):
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
    if not is_infill_study and not _bc_has_nonzero_load(bc):
        node.set_error(
            "Structural TopOpt needs a connected Topology Force, or a "
            "Topology Load Case containing one."
        )
        return None
    unitx = unity = unitz = 1.0
    if bounds is not None:
        mins, maxs = bounds
        span = np.maximum(maxs[:3] - mins[:3], 1e-12)
        unitx = float(span[0] / max(nelx, 1))
        unity = float(span[1] / max(nely, 1))
        unitz = float(span[2] / max(nelz, 1))
    structure_mode_value = str(node.get_property("structure_mode") or "Solid Envelope")
    # Ask the registry, not the display string. "Gyroid Network (Skeletal)" is
    # every bit as much a lattice study as "Gyroid Lattice", and a substring
    # test on the word "lattice" would quietly send it down the solid path.
    structure_family = normalize_family_key(structure_mode_value)
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
            resolve_lattice_structure_options(
                node,
                (nelx, nely, nelz),
                bounds,
            )
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
                f"Invalid lattice geometry: {exc}{hint}"
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
        passive_solid_mask=explicit_passive_solid_mask,
        passive_void_mask=explicit_passive_void_mask,
        unitx=unitx,
        unity=unity,
        unitz=unitz,
    )


def _build_problem(
    node: Any,
    domain: TopologyDomainContext,
) -> PreparedTopologyStudy | None:
    """Validate material/optimizer settings and instantiate the solver problem."""
    graph = domain.graph
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    bounds = graph.bounds
    material = graph.material
    constraint_list = graph.constraints
    load_list = graph.loads
    bc = graph.bc
    nelx, nely, nelz = domain.nelx, domain.nely, domain.nelz
    rmin_effective = domain.rmin
    design_domain = domain.design_domain
    design_goal = domain.design_goal
    minimum_mass_goal = domain.minimum_mass_goal
    load_aggregation_setting = domain.load_aggregation
    physics_mode = domain.physics_mode
    explicit_passive_solid_mask = domain.passive_solid_mask
    explicit_passive_void_mask = domain.passive_void_mask
    unitx, unity, unitz = domain.unitx, domain.unity, domain.unitz
    # Density studies use an unrestricted design domain. Old project
    # manufacturing controls cannot impose hidden constraints.
    mc = ManufacturingConstraints()

    stress_enabled = (
        as_bool(node.get_property("stress_constraint")) or minimum_mass_goal
    )
    optimizer = str(node.get_property("optimizer") or "Auto")
    if optimizer.strip().lower() == "projected gradient":
        optimizer = "PGD"
    if stress_enabled and optimizer.strip().upper() not in {"MMA", "GCMMA"}:
        optimizer = "GCMMA"
    try:
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
        # Cellular geometry is generated afterwards; every actual solve uses
        # the solid SIMP material and its projection.
        lattice_porosity = "Not applicable"
        material_penal = float(node.get_property("penal") or 3.0)
        study_poisson = float(material.get("nu", node.get_property("nu") or 0.3))
        solver_max_iter = int(node.get_property("max_iter") or 120)
        solver_tol = float(node.get_property("tol") or 0.0035)
        solver_patience = int(node.get_property("convergence_patience") or 6)
        problem = TopologyOptVoxelProblem(
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            # A lattice infill never reaches the solver: this instance exists
            # to carry the grid, units and cell family to the shared output
            # stage. See `TopologyOptVoxelProblem.solve_enabled`.
            solve_enabled=not is_infill_study,
            max_voxel_elements=getattr(node, "_resolved_voxel_meshing", {}).get("max_elements", 0),
            E0=youngs_modulus,
            Emin=minimum_stiffness,
            nu=study_poisson,
            penal=material_penal,
            lattice_cell_type="",
            lattice_maximum_density=1.0,
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
                domain.filter_radius_is_physical
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
            load_aggregation=load_aggregation_setting,
            load_pnorm_p=float(node.get_property("load_pnorm_p") or 8.0),
            stress_constraint_enabled=stress_enabled,
            yield_stress=yield_stress,
            # The projection drives the field to 0/1, which is what a solid
            # study wants and exactly what a graded lattice must not have.
            heaviside_enabled=True,
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
        lattice_porosity=lattice_porosity,
        source_design_domain=graph.design_domain_input,
        non_design_regions=tuple(graph.non_design_regions),
        voxel_pitch=getattr(node, "_resolved_voxel_meshing", {}).get("voxel_size"),
        max_voxel_elements=getattr(node, "_resolved_voxel_meshing", {}).get("max_elements", 0),
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
