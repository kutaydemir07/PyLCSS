# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validate graph inputs and construct a topology optimization problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pylcss.input_values import as_bool

from ..models.study import ManufacturingConstraints, ThermalBC, VoxelBC
from ..optimization.problem import TopologyOptVoxelProblem
from .boundary_mapping import (
    _add_bc_contact_regions,
    _append_region_once,
    _bc_feature_bboxes,
    _bc_has_nonzero_load,
    _bc_has_support,
)
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
    _thermal_bc_from_graph_payloads,
)
from .voxelization import (
    _cylinder_feature_lengths,
    _guided_rmin,
    _guided_voxel_grid,
    _mesh_design_domain_grid,
    _non_design_region_masks,
    _use_automatic_guided_grid,
)

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
    design_domain: np.ndarray
    design_goal: str
    goal_key: str
    minimum_mass_goal: bool
    physics_mode: str
    thermal_bc: ThermalBC
    passive_solid_mask: np.ndarray
    passive_void_mask: np.ndarray
    unitx: float
    unity: float
    unitz: float
    level_set_formulation: bool


def _collect_graph_inputs(node: Any) -> TopologyGraphInputs | None:
    """Normalize connected graph payloads and validate mapped boundaries."""
    design_domain_input = node.get_input_value("design_domain", None)
    bounds = _mesh_bounds(design_domain_input)
    material = node.get_input_value("material", None)
    material = material if isinstance(material, dict) else {}
    constraint_list = _flatten(node.get_input_list("supports"))
    load_list = _flatten(node.get_input_list("loads"))
    non_design_payloads = _flatten(node.get_input_list("non_design_regions"))
    joint_list = _flatten(node.get_input_list("joints"))
    operating_case_payloads = _flatten(node.get_input_list("load_cases"))
    thermal_sink_payloads = _flatten(node.get_input_list("thermal_sinks"))
    thermal_load_payloads = _flatten(node.get_input_list("thermal_loads"))
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
        bc.load_cases = []
    if joint_list:
        bc.joints = []
    if operating_case_payloads:
        bc.load_cases = []
    node._merge_graph_bcs(
        bc,
        design_domain_input,
        constraint_list,
        load_list,
    )
    try:
        if bounds is not None:
            node._merge_graph_joints(bc, bounds, joint_list)
            graph_cases = [
                node._graph_operating_case(
                    payload,
                    design_domain_input,
                    bounds,
                )
                for payload in operating_case_payloads
            ]
            bc.load_cases.extend(graph_cases)
            for case in graph_cases:
                for pin in case.joint_pin_cylinders:
                    _append_region_once(bc.joint_pin_cylinders, pin)
            for payload in operating_case_payloads:
                if isinstance(payload, dict):
                    constraint_list.extend(_flatten(payload.get("supports")))
                    load_list.extend(_flatten(payload.get("loads")))
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt joint/operating-case setup: {exc}")
        return None
    try:
        _add_bc_contact_regions(bc)
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt support/load settings: {exc}")
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
    guided_active = _use_automatic_guided_grid(
        node.get_property("workflow_mode"),
        node.get_property("quality_preset"),
    )
    if guided_active:
        # Each structural pose/load case owns a separate sparse
        # factorization in the current pyMOTO network.  Grid size therefore
        # has to shrink as cases are added; otherwise the curated
        # multi-body example can consume several GB before iteration one.
        structural_case_count = max(1, len(bc.load_cases))
        guided_cell_budget = max(
            5_000,
            int(13_000 / np.sqrt(float(structural_case_count))),
        )
        guided_grid = _guided_voxel_grid(
            bounds,
            "Automatic",
            feature_bboxes=_bc_feature_bboxes(constraint_list, load_list),
            feature_lengths=_cylinder_feature_lengths(
                bounds,
                solid_cylinders=bc.solid_cylinders,
                void_cylinders=bc.void_cylinders,
            ),
            max_total_cells=guided_cell_budget,
        )
        if guided_grid is not None:
            nelx, nely, nelz = guided_grid

    structural_case_count = max(1, len(bc.load_cases))
    safe_cell_cap = max(
        8_000,
        int(24_000 / np.sqrt(float(structural_case_count))),
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
    if guided_active:
        rmin_effective = _guided_rmin(nelx, nely, nelz)
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

    design_goal = str(node.get_property("design_goal") or "Lightweight Stiffness")
    goal_key = design_goal.strip().lower()
    minimum_mass_goal = goal_key == "minimum mass under stress"
    physics_mode = (
        str(node.get_property("physics_mode") or "Structural")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if str(node.get_property("workflow_mode") or "Guided").lower() == "guided":
        if goal_key == "thermal conduction":
            physics_mode = "thermal"
        elif goal_key == "thermo-mechanical":
            physics_mode = "thermo_mechanical"
        elif goal_key in {
            "lightweight stiffness",
            "minimum mass under stress",
            "multibody load envelope",
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
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid TopOpt thermal setup: {exc}")
        return None
    uses_structural = physics_mode in {"structural", "thermo_mechanical"}
    uses_thermal = physics_mode in {"thermal", "thermo_mechanical"}
    if uses_structural and not _bc_has_support(bc):
        node.set_error(
            "Structural TopOpt needs a connected TopOpt Support, or an "
            "Operating Case containing one."
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
    if uses_structural and not _bc_has_nonzero_load(bc):
        node.set_error(
            "Structural TopOpt needs a connected TopOpt Force, or an "
            "Operating Case containing one."
        )
        return None
    if uses_thermal and not (thermal_bc.fixed_faces or thermal_bc.fixed_boxes):
        node.set_error(
            "Thermal TopOpt needs at least one connected TopOpt Thermal Sink."
        )
        return None
    if uses_thermal and not thermal_bc.load_cases:
        node.set_error("Thermal TopOpt needs at least one connected TopOpt Heat Load.")
        return None
    unitx = unity = unitz = 1.0
    if bounds is not None:
        mins, maxs = bounds
        span = np.maximum(maxs[:3] - mins[:3], 1e-12)
        unitx = float(span[0] / max(nelx, 1))
        unity = float(span[1] / max(nely, 1))
        unitz = float(span[2] / max(nelz, 1))

    structure_mode_value = str(node.get_property("structure_mode") or "Solid Envelope")
    formulation = str(node.get_property("formulation") or "Density (SIMP)").strip()
    level_set_formulation = formulation.lower().startswith("level set")
    if (
        level_set_formulation
        and structure_mode_value.strip().lower() != "solid envelope"
    ):
        node.set_error(
            "Level-set optimization produces a crisp solid boundary. "
            "Select Solid Envelope, or use Density (SIMP) for ribs or lattices."
        )
        return None

    return TopologyDomainContext(
        graph=graph,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        rmin=rmin_effective,
        design_domain=design_domain,
        design_goal=design_goal,
        goal_key=goal_key,
        minimum_mass_goal=minimum_mass_goal,
        physics_mode=physics_mode,
        thermal_bc=thermal_bc,
        passive_solid_mask=explicit_passive_solid_mask,
        passive_void_mask=explicit_passive_void_mask,
        unitx=unitx,
        unity=unity,
        unitz=unitz,
        level_set_formulation=level_set_formulation,
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
    physics_mode = domain.physics_mode
    thermal_bc = domain.thermal_bc
    explicit_passive_solid_mask = domain.passive_solid_mask
    explicit_passive_void_mask = domain.passive_void_mask
    unitx, unity, unitz = domain.unitx, domain.unity, domain.unitz
    level_set_formulation = domain.level_set_formulation
    try:
        mc = ManufacturingConstraints(
            symmetry=(str(node.get_property("symmetry") or "None")).lower(),
            extrusion=(str(node.get_property("extrusion") or "None")).lower(),
            overhang_build_axis=(
                str(node.get_property("overhang_build_axis") or "None")
            ).lower(),
            max_member_size_voxels=float(
                node.get_property("max_member_size_voxels") or 0.0
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
        # The structure choice is a recovery/manufacturing choice. It must
        # not silently change the continuum topology problem; otherwise
        # switching only the display/manufacturing output produces a
        # different load path and a different iteration count.
        material_penal = float(node.get_property("penal") or 3.0)
        problem = TopologyOptVoxelProblem(
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            E0=youngs_modulus,
            Emin=minimum_stiffness,
            nu=float(material.get("nu", node.get_property("nu") or 0.3)),
            penal=material_penal,
            volfrac=float(node.get_property("volfrac") or 0.5),
            rmin=rmin_effective,
            unitx=unitx,
            unity=unity,
            unitz=unitz,
            optimizer=optimizer,
            formulation=("level_set" if level_set_formulation else "density"),
            max_iter=int(node.get_property("max_iter") or 80),
            tol=float(node.get_property("tol") or 0.01),
            patience=int(node.get_property("convergence_patience") or 5),
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
            load_aggregation=(
                "worst_case"
                if goal_key == "multibody load envelope"
                else str(node.get_property("load_aggregation") or "Weighted Sum")
            ),
            load_pnorm_p=float(node.get_property("load_pnorm_p") or 8.0),
            projected_gradient_step=float(
                node.get_property("projected_gradient_step") or 0.15
            ),
            stress_constraint_enabled=stress_enabled,
            yield_stress=yield_stress,
            heaviside_enabled=True,
            # Numerical hyperparameters use the dataclass defaults (industrial
            # values: q=0.5 Bruggi qp-approach, p=8.0 PNorm aggregation,
            # Heaviside three-field SIMP on with β: 1 → 16 stepping every 30
            # iters, η=0.5).  These are NOT user knobs in industrial codes.
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
