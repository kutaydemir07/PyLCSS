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
from ..configuration.presets import normalize_design_goal
from ..configuration.voxel_meshing import (
    VoxelMeshPlan,
    voxel_mesh_plan,
    available_voxel_memory_bytes,
)
from ..models.study import VoxelBC
from ..optimization.problem import TopologyOptVoxelProblem
from .boundary_mapping import (
    _bc_has_nonzero_load,
    _bc_has_support,
)
from .lattice_settings import resolve_lattice_structure_options
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
)
from .voxelization import (
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
    problem: TopologyOptVoxelProblem
    nelx: int
    nely: int
    nelz: int
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
    minimum_mass_goal: bool
    # Resolved once, including the migration of retired envelope goals, so the
    # solver and the iteration policy read the same decision.
    load_aggregation: str
    physics_mode: str
    passive_solid_mask: np.ndarray
    passive_void_mask: np.ndarray
    unitx: float
    unity: float
    unitz: float


def _collect_graph_inputs(node: Any) -> TopologyGraphInputs | None:
    """Normalize connected graph payloads and validate mapped boundaries."""
    design_domain_input = node.get_input_value("design_domain", None)
    bounds = _mesh_bounds(design_domain_input)
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    material = {} if is_infill_study else node.get_input_value("material", None)
    material = material if isinstance(material, dict) else {}
    constraint_list = (
        [] if is_infill_study else _flatten(node.get_input_list("supports"))
    )
    load_list = [] if is_infill_study else _flatten(node.get_input_list("loads"))
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
        if not is_infill_study:
            node._merge_graph_bcs(bc, design_domain_input, constraint_list, load_list)
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


def plan_topology_voxel_mesh(node: Any, bounds: Bounds) -> VoxelMeshPlan:
    """Shared cheap inspector/backend plan; no geometry rasterization occurs."""
    if callable(getattr(node, "resolve_infill_cell_size", None)):
        pitch = float(node._resolved_infill_voxel_size)
        shape = tuple(int(node.get_property(key)) for key in ("nelx", "nely", "nelz"))
        from .lattice_infill_node import BUILD_VOXEL_BUDGET, DEFAULT_BUILD_QUALITY

        quality = str(
            node.get_property("infill_build_quality") or DEFAULT_BUILD_QUALITY
        )
        return VoxelMeshPlan(
            shape,
            (bounds[0], bounds[0] + np.asarray(shape) * pitch),
            bounds,
            pitch,
            quality,
            "Lattice build",
            BUILD_VOXEL_BUDGET[quality],
        )
    automatic = node.get_property("automatic_grid")
    available_memory = available_voxel_memory_bytes()
    return voxel_mesh_plan(
        bounds,
        node.get_property("voxel_mesh_quality"),
        voxel_size=float(node.get_property("voxel_size") or 0),
        automatic=automatic is None or as_bool(automatic),
        manual_shape=tuple(
            node.get_property(key) or default
            for key, default in zip(("nelx", "nely", "nelz"), (30, 20, 10))
        ),
        max_elements=node.get_property("max_voxel_elements"),
        available_memory_bytes=available_memory,
        memory_fraction=node.get_property("voxel_ram_fraction"),
    )


def _prepare_domain(
    node: Any,
    graph: TopologyGraphInputs,
) -> TopologyDomainContext | None:
    """Resolve a safe voxel grid, physics mode, and passive masks."""
    design_domain_input = graph.design_domain_input
    bounds = graph.bounds
    non_design_payloads = graph.non_design_regions
    bc = graph.bc
    try:
        plan = plan_topology_voxel_mesh(node, bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        node.set_error(f"Invalid voxel meshing: {exc}")
        return None
    nelx, nely, nelz = plan.shape
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    node._resolved_voxel_meshing = {
        "mode": plan.mode,
        "quality": plan.quality,
        "requested_voxel_size": float(node.get_property("voxel_size") or 0),
        "voxel_size": plan.pitch,
        "grid": list(plan.shape),
        "voxel_edges": plan.voxel_edges,
        "elements": plan.elements,
        "max_elements": plan.element_limit,
        "estimated_memory_bytes": plan.estimated_memory_bytes,
        "source_bounds": [value.tolist() for value in plan.source_bounds],
        "grid_bounds": [value.tolist() for value in plan.bounds],
        "automatic_refinement": as_bool(node.get_property("automatic_mesh_refinement")),
        "preset_element_budget": plan.preset_element_budget,
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
    rmin_effective = 1.5 if is_infill_study else float(node.get_property("rmin") or 1.5)
    filter_radius_is_physical = False
    length_scale = LengthScale(
        filter_radius=rmin_effective,
        minimum_solid_size=0.0,
        minimum_void_size=0.0,
    )
    automatic_filter = node.get_property("automatic_filter_radius")
    if not is_infill_study and (automatic_filter is None or as_bool(automatic_filter)):
        try:
            length_scale = resolve_physical_length_scale(
                bounds,
                (nelx, nely, nelz),
                node.get_property("topology_minimum_solid_size_mm"),
                node.get_property("topology_minimum_void_size_mm"),
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

    try:
        design_goal = (
            "Lattice Infill"
            if is_infill_study
            else normalize_design_goal(node.get_property("design_goal"))
        )
    except ValueError as exc:
        node.set_error(str(exc))
        return None
    goal_key = design_goal.strip().lower()
    load_aggregation_setting = str(
        node.get_property("load_aggregation") or "Weighted Sum"
    )
    minimum_mass_goal = goal_key == "minimum mass under stress"
    # Retired project values cannot switch this structural node's analysis.
    physics_mode = "structural"
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    if not is_infill_study and not _bc_has_support(bc):
        node.set_error("Structural TopOpt needs a connected Topology Support.")
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
        node.set_error("Structural TopOpt needs a connected Topology Force.")
        return None
    unitx = unity = unitz = 1.0
    if bounds is not None:
        mins, maxs = bounds
        span = np.maximum(maxs[:3] - mins[:3], 1e-12)
        unitx = float(span[0] / max(nelx, 1))
        unity = float(span[1] / max(nely, 1))
        unitz = float(span[2] / max(nelz, 1))
    if is_infill_study:
        setattr(
            node,
            "_resolved_lattice_voxel_size",
            voxel_size_from_bounds((nelx, nely, nelz), bounds),
        )
        try:
            resolve_lattice_structure_options(node, (nelx, nely, nelz), bounds)
        except (TypeError, ValueError) as exc:
            node.set_error(f"Invalid lattice geometry: {exc}")
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
        minimum_mass_goal=minimum_mass_goal,
        load_aggregation=load_aggregation_setting,
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
    if callable(getattr(node, "resolve_infill_cell_size", None)):
        # Shared recovery needs grid/units metadata, but no solver choices are read.
        problem = TopologyOptVoxelProblem(
            nelx=domain.nelx,
            nely=domain.nely,
            nelz=domain.nelz,
            solve_enabled=False,
            volfrac=1.0,
            unitx=domain.unitx,
            unity=domain.unity,
            unitz=domain.unitz,
            bc=graph.bc,
            design_domain=domain.design_domain,
            passive_solid_mask=domain.passive_solid_mask,
            passive_void_mask=domain.passive_void_mask,
        )
        return _prepared_study(node, domain, problem)
    material = graph.material
    bc = graph.bc
    nelx, nely, nelz = domain.nelx, domain.nely, domain.nelz
    rmin_effective = domain.rmin
    design_domain = domain.design_domain
    minimum_mass_goal = domain.minimum_mass_goal
    load_aggregation_setting = domain.load_aggregation
    physics_mode = domain.physics_mode
    explicit_passive_solid_mask = domain.passive_solid_mask
    explicit_passive_void_mask = domain.passive_void_mask
    unitx, unity, unitz = domain.unitx, domain.unity, domain.unitz
    stress_enabled = (
        as_bool(node.get_property("stress_constraint")) or minimum_mass_goal
    )
    optimizer = str(node.get_property("optimizer") or "Auto")
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
        material_penal = float(node.get_property("penal") or 3.0)
        study_poisson = float(material.get("nu", node.get_property("nu") or 0.3))
        solver_max_iter = int(node.get_property("max_iter") or 120)
        solver_tol = float(node.get_property("tol") or 0.0035)
        solver_patience = int(node.get_property("convergence_patience") or 6)
        problem = TopologyOptVoxelProblem(
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            max_voxel_elements=getattr(node, "_resolved_voxel_meshing", {}).get(
                "max_elements", 0
            ),
            E0=youngs_modulus,
            Emin=minimum_stiffness,
            nu=study_poisson,
            penal=material_penal,
            volfrac=float(node.get_property("volfrac") or 0.5),
            rmin=rmin_effective,
            filter_radius_is_physical=domain.filter_radius_is_physical,
            minimum_solid_size=domain.minimum_solid_size,
            minimum_void_size=domain.minimum_void_size,
            # The requested sizes only bind through the robust formulation, so
            # a study that resolved a physical length scale is solved with it.
            robust_length_scale=bool(domain.filter_radius_is_physical),
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
            design_domain=design_domain,
            passive_solid_mask=explicit_passive_solid_mask,
            passive_void_mask=explicit_passive_void_mask,
            objective_mode="minimum_mass" if minimum_mass_goal else "compliance",
            physics_mode=physics_mode,
            load_aggregation=load_aggregation_setting,
            load_pnorm_p=float(node.get_property("load_pnorm_p") or 8.0),
            stress_constraint_enabled=stress_enabled,
            yield_stress=yield_stress,
            # The solid SIMP study uses a projected material boundary.
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

    return _prepared_study(node, domain, problem)


def _prepared_study(node, domain, problem) -> PreparedTopologyStudy:
    graph = domain.graph
    mesh = getattr(node, "_resolved_voxel_meshing", {})
    return PreparedTopologyStudy(
        bounds=graph.bounds,
        material=graph.material,
        constraints=graph.constraints,
        loads=graph.loads,
        bc=graph.bc,
        design_domain=domain.design_domain,
        design_goal=domain.design_goal,
        passive_solid_mask=domain.passive_solid_mask,
        passive_void_mask=domain.passive_void_mask,
        problem=problem,
        nelx=domain.nelx,
        nely=domain.nely,
        nelz=domain.nelz,
        source_design_domain=graph.design_domain_input,
        non_design_regions=tuple(graph.non_design_regions),
        voxel_pitch=mesh.get("voxel_size"),
        max_voxel_elements=mesh.get("max_elements", 0),
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
