# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for voxel topology optimization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from pylcss.cae.design_studio.core.base_node import CadQueryNode
from ..models.study import (
    VoxelBC,
    LoadCase,
    BoundaryMesh,
)
from ..configuration.presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
)
from ..configuration.voxel_meshing import (
    DEFAULT_VOXEL_MESH_QUALITY,
    DEFAULT_MAX_VOXEL_ELEMENTS,
    VOXEL_MESH_RESOLUTIONS,
)
from ..configuration.fea_meshing import FEA_MESH_DEFAULTS, fea_mesh_options_from_node
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
    _joint_from_graph_payload,
)
from .boundary_mapping import (
    _pressure_load_patches,
    _cylinder_void_region_from_face,
    _append_region_once,
    _joint_pin_from_anchor_cylinders,
)
from . import boundary_mapping as _boundary_mapping
from . import geometry_mapping as _geometry_mapping
from . import voxelization as _voxelization
from .selected_boundary import selection_regions, region_measure, boundary_with_force
from .node_execution import _TopologyExecutionMixin

logger = logging.getLogger(__name__)

_HELPER_MODULES = (
    _geometry_mapping,
    _boundary_mapping,
    _voxelization,
)


def __getattr__(name: str) -> object:
    """Resolve helpers moved out of this formerly monolithic module."""
    for module in _HELPER_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    helper_names = {
        name
        for module in _HELPER_MODULES
        for name in dir(module)
        if name.startswith("_")
    }
    return sorted(set(globals()) | helper_names)


class TopologyOptVoxelNode(_TopologyExecutionMixin, CadQueryNode):
    """3-D SIMP topology optimisation using pyMOTO on a structured voxel grid.

    The graph-facing inputs describe the topology study directly: a CAD/mesh
    design domain, material, and selected-region structural supports/loads.
    The structured analysis grid is generated
    internally; a prebuilt finite-element mesh is not required.

    The solver produces a solid envelope. The inspector can subsequently
    generate a density-graded lattice from the saved result.
    """

    __identifier__ = "com.cad.sim.topopt_voxel"
    NODE_NAME = "Topology Solver"

    def __init__(self) -> None:
        super().__init__()
        self.add_output("results", color=(180, 255, 180))
        legacy_result = self.add_output(
            "result", display_name=False, color=(180, 255, 180)
        )
        legacy_result.view.setVisible(False)
        self.add_output("recovered_shape", color=(100, 255, 100))
        self.add_input("design_domain", color=(100, 255, 100))
        self.add_input("material", color=(200, 200, 200))
        # Both accept any number of connections, so several supports and
        # several forces are wired straight in. That is what made the separate
        # Load Case node redundant: grouping supports and forces into a case
        # node was only ever a way to get more than one of each.
        self.add_input("supports", color=(255, 100, 100), multi_input=True)
        self.add_input("loads", color=(255, 255, 0), multi_input=True)
        # Non-design material is authored on design-domain faces in the inspector.
        self.create_property("non_design_enabled", True, widget_type="bool")
        self.create_property("non_design_regions", "[]", widget_type="string")
        self.create_property("automatic_grid", True, widget_type="bool")
        self.create_property(
            "voxel_mesh_quality", DEFAULT_VOXEL_MESH_QUALITY,
            widget_type="combo", items=list(VOXEL_MESH_RESOLUTIONS),
        )
        self.create_property("voxel_size", 0.0, widget_type="float")
        self.create_property("max_voxel_elements", DEFAULT_MAX_VOXEL_ELEMENTS, widget_type="int")
        self.create_property("automatic_mesh_refinement", False, widget_type="bool")
        self.create_property("automatic_filter_radius", True, widget_type="bool")
        self.create_property("advanced_overrides", "[]", widget_type="string")
        self.create_property(
            "design_goal",
            "Lightweight Stiffness",
            widget_type="combo",
            items=list(INDUSTRIAL_DESIGN_GOALS),
        )
        self.create_property(
            "physics_mode",
            "Structural",
            widget_type="combo",
            items=["Structural"],
        )
        self.create_property(
            "formulation",
            "Density (SIMP)",
            widget_type="combo",
            items=["Density (SIMP)"],
        )
        # The one length-scale control worth exposing in guided mode: the
        # thinnest member the design may contain, in model units. It replaces
        # reasoning about `rmin` in voxels, and is what a process capability or
        # a drawing note is actually stated in. Zero means program controlled.
        self.create_property("advanced_settings_visible", False, widget_type="bool")
        self.create_property("validate_after_optimize", False, widget_type="bool")
        # Retained for older projects; CAD FEA uses one user-configured mesh.
        self.create_property(
            "validation_quality",
            "Standard",
            widget_type="combo",
            items=["Standard"],
        )
        # These controls mesh the final CAD for independent FEA. They do not
        # select the density optimizer's structured voxel grid.
        for name, default in FEA_MESH_DEFAULTS.items():
            if name == "fea_element_order":
                self.create_property(
                    name, default, widget_type="combo", items=["C3D10", "C3D4"]
                )
            else:
                self.create_property(
                    name, default, widget_type="int" if isinstance(default, int) else "float"
                )
        self.create_property(
            "validation_displacement_limit_mm",
            0.0,
            widget_type="float",
        )
        self.create_property(
            "validation_yield_safety_factor",
            1.0,
            widget_type="float",
        )
        self.create_property(
            "topology_convergence_enabled",
            False,
            widget_type="bool",
        )
        self.create_property(
            "topology_convergence_levels",
            3,
            widget_type="int",
        )
        # STL and the surface preview use the recovered mesh. Build CAD when
        # CAD view / STEP export is requested. The old property is retained
        # only for project compatibility.
        self.create_property("generate_cad_after_optimize", False, widget_type="bool")
        self.create_property(
            "cad_export_filename", "topology_optimized.step", widget_type="string"
        )
        self.create_property(
            "cad_export_format", "STEP", widget_type="string"
        )
        # Domain and recovery settings.
        self.create_property("nelx", 30, widget_type="int")
        self.create_property("nely", 20, widget_type="int")
        self.create_property("nelz", 10, widget_type="int")
        self.create_property("volfrac", 0.5, widget_type="float")
        self.create_property("rmin", 1.5, widget_type="float")
        self.create_property("penal", 3.0, widget_type="float")
        self.create_property("density_cutoff", 0.30, widget_type="float")
        self.create_property(
            "visualization",
            "CAD",
            widget_type="combo",
            items=[
                "CAD",
                "Density",
            ],
        )
        self.create_property("E0", 1.0, widget_type="float")
        self.create_property("Emin", 1e-9, widget_type="float")
        self.create_property("nu", 0.3, widget_type="float")

        # Post-processing choices must never change the continuum solve.
        from ..manufacturing import PUBLIC_FAMILIES, PUBLIC_FAMILY_KEYS
        self.create_property(
            "post_lattice_type", "Gyroid Lattice", widget_type="combo",
            items=[PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS],
        )
        self.create_property("post_lattice_cell_size_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_min_thickness_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_skin_thickness_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_min_density", 0.15, widget_type="float")
        self.create_property("post_lattice_max_density", 0.60, widget_type="float")

        # Solver settings.
        self.create_property(
            "optimizer",
            "Auto",
            widget_type="combo",
            items=["Auto", "OC", "MMA", "GCMMA"],
        )
        self.create_property(
            "load_aggregation",
            "Weighted Sum",
            widget_type="combo",
            items=["Weighted Sum", "Worst Case"],
        )
        self.create_property("load_pnorm_p", 8.0, widget_type="float")
        self.create_property("max_iter", 120, widget_type="int")
        # `tol` is the relative-compliance-change convergence threshold; the run
        # stops once it holds for `convergence_patience` consecutive iterations.
        self.create_property("tol", 0.0035, widget_type="float")
        self.create_property("convergence_patience", 6, widget_type="int")

        # Manufacturing constraints (casting draw, extrusion, symmetry, AM overhang, member sizing)

        # Stress constraint (engineering surface only).
        # User sees: Enable + Yield Stress.  The qp-approach exponent q,
        # PNorm exponent, and multi-LC aggregation are silent industrial
        # defaults inside the solver (q=0.5 Bruggi 2008, p=8.0 PNorm).
        self.create_property("stress_constraint", False, widget_type="bool")
        self.create_property("yield_stress", 250.0, widget_type="float")

    # Property helpers.

    def apply_guided_defaults(self) -> dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_topopt_defaults(
            self.get_property("design_goal"),
            nelx=self.get_property("nelx") or 30,
            nely=self.get_property("nely") or 20,
            nelz=self.get_property("nelz") or 10,
        )
        for key, value in settings.items():
            if self.has_property(key):
                self.set_property(key, value)
        return settings

    def request_stop(self) -> None:
        """Request a clean stop at the next optimizer iteration boundary."""
        solver = getattr(self, "_active_solver", None)
        if solver is not None:
            solver.stop()

    def _run_embedded_validation(
        self,
        topo_output: dict[str, Any],
        material: dict[str, Any],
        constraint_list: list[Any],
        load_list: list[Any],
        cancel_callback: Callable[[], bool] | None = None,
    ) -> Optional[dict[str, Any]]:
        """Run the validation stage as an internal TopOpt study action."""
        from ..verification.fea_analysis import run_topopt_fea

        return run_topopt_fea(
            topo_output,
            material,
            constraint_list,
            load_list,
            quality="Standard",
            mesh_options=fea_mesh_options_from_node(self),
            cancel_callback=cancel_callback,
        )

    def _run_embedded_cad_reconstruction(
        self,
        topo_output: dict[str, Any],
        progress_callback: Any = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> Any:
        """Build a CAD body as an internal TopOpt study action."""
        from ..geometry.cad_reconstruction import normalize_cad_strategy
        from pylcss.cae.user_interface.cad.cad_background import reconstruct_cad_in_background

        strategy = 'auto'
        topo_payload = dict(topo_output or {})
        export_steps_dir = getattr(self, "_export_steps_dir", None) or topo_payload.get("export_steps_dir")
        shape, report, preview_mesh = reconstruct_cad_in_background(
            topo_payload,
            reconstruction_strategy=strategy,
            sew_tolerance=1e-4,
            merge_angle_deg=0.0,
            # Strut families use the solved analytic members. Other families
            # fit shared splines to the actual recovered boundary.
            structure_options=getattr(self, "_last_structure_options", None),
            member_plan=getattr(self, "_last_member_plan", None),
            export_steps_dir=export_steps_dir,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        topo_output["cad_preview_mesh"] = preview_mesh
        self._last_cad_reconstruction_report = report
        return shape

    def _build_bc(self) -> VoxelBC:
        """Start an empty study; connected graph nodes provide every input."""
        return VoxelBC()

    def _merge_graph_bcs(
        self,
        bc: VoxelBC,
        design_domain: Any,
        constraints: list[Any],
        loads: list[Any],
    ) -> None:
        """Map selected-region support and load nodes onto the voxel domain."""
        bounds = _mesh_bounds(design_domain)
        if bounds is None:
            if constraints or loads:
                raise ValueError("Cannot map connected supports and loads without design-domain bounds.")
            return

        for index, constraint in enumerate(constraints, start=1):
            if not isinstance(constraint, dict):
                raise ValueError(f"Connected support {index} did not produce a support definition.")
            dofs = constraint.get("fixed_dofs") or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                raise ValueError(f"Connected support {index} has no valid constrained DOFs.")
            regions = selection_regions(constraint, bounds)
            if not regions:
                raise ValueError(f"Connected support {index} has no selected geometry or bounding-box metadata.")
            for region in regions:
                if isinstance(region[0], BoundaryMesh):
                    boundary = region[0]
                    boundary.fixed_dofs = list(dofs)
                    bc.boundary_supports.append(boundary)
                else:
                    bc.fixed_boxes.append((*region, dofs))

        bc.load_cases = []

        for index, load in enumerate(loads, start=1):
            if not isinstance(load, dict):
                raise ValueError(f"Connected load {index} did not produce a load definition.")
            load_type = str(load.get("type") or "").lower()
            local_boxes = []
            if load_type in ("force", "bearing_load"):
                vector = load.get("vector")
                try:
                    fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
                except Exception:
                    raise ValueError(f"Connected load {index} needs three finite force components.")
                if not np.all(np.isfinite([fx, fy, fz])):
                    raise ValueError(f"Connected load {index} needs three finite force components.")
                regions = selection_regions(load, bounds)
                if not regions:
                    raise ValueError(f"Connected load {index} has no selected geometry or bounding-box metadata.")
                measures = [region_measure(region, bounds) for region in regions]
                for region, measure in zip(regions, measures):
                    vector = np.asarray([fx, fy, fz]) * measure / sum(measures)
                    if isinstance(region[0], BoundaryMesh):
                        bc.boundary_forces.append(boundary_with_force(region, vector))
                    else:
                        local_boxes.append((*region, *vector))
            elif load_type == "pressure":
                if load.get("geometries") or load.get("geometry") is not None:
                    pressure = float(load.get("pressure", load.get("magnitude", 0.0)))
                    if not np.isfinite(pressure):
                        raise ValueError(f"Connected pressure load {index} needs a finite pressure.")
                    regions = selection_regions(load, bounds, exact_boxes=False)
                    for region in regions:
                        bc.boundary_forces.append(boundary_with_force(region, pressure=pressure))
                    continue
                selection_regions(load, bounds)  # Validate metadata before mapping pressure.
                pressure_patches = _pressure_load_patches(load, bounds)
                if not pressure_patches:
                    raise ValueError(f"Connected pressure load {index} could not be mapped to voxel patch forces.")
                for frac_box, fx, fy, fz in pressure_patches:
                    local_boxes.append((*frac_box, fx, fy, fz))
            else:
                raise ValueError(f"Connected load {index} has unsupported type {load_type!r}.")

            if not local_boxes:
                continue

            # One study is one simultaneous operating scenario. Alternative
            # scenarios belong in separate study/project copies.
            bc.box_forces.extend(local_boxes)

    def _merge_graph_joints(
        self,
        bc: VoxelBC,
        bounds: tuple[np.ndarray, np.ndarray],
        joints: list[Any],
    ) -> None:
        """Convert graph joints and infer their separate pin hardware."""
        for payload in joints:
            joint = _joint_from_graph_payload(payload, bounds)
            bc.joints.append(joint)
            if not isinstance(payload, dict):
                continue
            # Joint hardware is separate from the optimized body. Preserved
            # material and keep-outs come only from connected CAD bodies.
            anchor_cylinders: dict[str, list[tuple[Any, ...]]] = {
                "anchor_a_geometries": [],
                "anchor_b_geometries": [],
            }
            for key in anchor_cylinders:
                for geometry in _flatten(payload.get(key)):
                    cylinder = _cylinder_void_region_from_face(geometry, bounds)
                    if cylinder is not None:
                        anchor_cylinders[key].append(cylinder)
            pin = _joint_pin_from_anchor_cylinders(
                anchor_cylinders["anchor_a_geometries"],
                anchor_cylinders["anchor_b_geometries"],
            )
            if pin is not None:
                _append_region_once(bc.joint_pin_cylinders, pin)

    def _graph_operating_case(
        self,
        payload: Any,
        design_domain: Any,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> LoadCase:
        """Convert one TopOpt Load Case node to the typed solver model."""
        if (
            not isinstance(payload, dict)
            or str(payload.get("type") or "").lower() != "topology_operating_case"
        ):
            raise ValueError(
                "The load_cases input accepts TopOpt Load Case nodes only."
            )
        local_bc = VoxelBC()
        self._merge_graph_bcs(
            local_bc,
            design_domain,
            _flatten(payload.get("supports")),
            _flatten(payload.get("loads")),
        )
        self._merge_graph_joints(
            local_bc,
            bounds,
            _flatten(payload.get("joints")),
        )
        return LoadCase(
            name=str(payload.get("name") or "Load Case"),
            weight=float(payload.get("weight") or 1.0),
            point_forces=list(local_bc.point_forces),
            box_forces=list(local_bc.box_forces),
            distributed_forces=list(local_bc.distributed_forces),
            boundary_forces=list(local_bc.boundary_forces),
            boundary_supports=list(local_bc.boundary_supports),
            fixed_boxes=list(local_bc.fixed_boxes),
            replace_supports=bool(payload.get("replace_supports", True)),
            joints=list(local_bc.joints),
            replace_joints=bool(payload.get("replace_joints", False)),
            joint_pin_cylinders=list(local_bc.joint_pin_cylinders),
        )
