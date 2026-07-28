# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for voxel topology optimization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from pylcss.design_studio.core.base_node import CadQueryNode
from ..models.study import (
    VoxelBC,
    LoadCase,
)
from ..configuration.presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_WORKFLOW_MODES,
    INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
)
from .geometry_mapping import (
    _flatten,
    _mesh_bounds,
    _entry_bboxes,
    _fraction_box,
    _joint_from_graph_payload,
)
from .boundary_mapping import (
    _iter_load_faces,
    _pressure_load_patches,
    _cylinder_void_region_from_face,
    _append_region_once,
    _add_cylindrical_contact_region,
    _joint_pin_from_anchor_cylinders,
)
from . import boundary_mapping as _boundary_mapping
from . import geometry_mapping as _geometry_mapping
from . import voxelization as _voxelization
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
    design domain, material, selected-region supports/loads, joints, operating
    cases, and thermal conditions. The structured analysis grid is generated
    internally; a prebuilt finite-element mesh is not required.
    """

    __identifier__ = "com.cad.sim.topopt_voxel"
    NODE_NAME = "Topology Opt (3D Voxel)"

    def __init__(self) -> None:
        super().__init__()
        self.add_output("result", color=(180, 255, 180))
        self.add_output("recovered_shape", color=(100, 255, 100))
        self.add_input("design_domain", color=(100, 255, 100))
        self.add_input("material", color=(200, 200, 200))
        self.add_input("supports", color=(255, 100, 100), multi_input=True)
        self.add_input("loads", color=(255, 255, 0), multi_input=True)
        self.add_input(
            "non_design_regions",
            color=(220, 180, 80),
            multi_input=True,
        )
        self.add_input("joints", color=(255, 170, 80), multi_input=True)
        self.add_input("load_cases", color=(120, 220, 255), multi_input=True)
        self.add_input(
            "thermal_sinks",
            color=(80, 190, 255),
            multi_input=True,
        )
        self.add_input(
            "thermal_loads",
            color=(255, 110, 50),
            multi_input=True,
        )
        # Guided workflow
        self.create_property(
            "workflow_mode",
            "Guided",
            widget_type="combo",
            items=list(INDUSTRIAL_WORKFLOW_MODES),
        )
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
            items=["Structural", "Thermal", "Thermo-Mechanical"],
        )
        self.create_property(
            "formulation",
            "Density (SIMP)",
            widget_type="combo",
            items=[
                "Density (SIMP)",
                "Level Set (Reaction-Diffusion)",
            ],
        )
        # Legacy saved studies may still carry a quality_preset. Guided mode
        # now chooses a single automatic feature-aware grid instead.
        self.create_property("quality_preset", "Automatic", widget_type="string")
        self.create_property(
            "manufacturing_process",
            "None",
            widget_type="combo",
            items=list(INDUSTRIAL_MANUFACTURING_PROCESSES),
        )
        self.create_property("advanced_settings_visible", False, widget_type="bool")
        self.create_property("validate_after_optimize", False, widget_type="bool")
        self.create_property(
            "validation_quality",
            "Standard",
            widget_type="combo",
            items=["Standard", "Mesh Convergence"],
        )
        self.create_property("generate_cad_after_optimize", False, widget_type="bool")
        self.create_property(
            "cad_reconstruction_method",
            "Recovered Shape",
            widget_type="combo",
            items=["Recovered Shape"],
        )
        self.create_property(
            "cad_export_filename", "topology_optimized.step", widget_type="string"
        )
        # Domain and recovery settings.
        self.create_property("nelx", 30, widget_type="int")
        self.create_property("nely", 20, widget_type="int")
        self.create_property("nelz", 10, widget_type="int")
        self.create_property("volfrac", 0.5, widget_type="float")
        self.create_property("rmin", 1.5, widget_type="float")
        self.create_property("penal", 3.0, widget_type="float")
        self.create_property("density_cutoff", 0.45, widget_type="float")
        self.create_property(
            "visualization",
            "Density",
            widget_type="combo",
            items=["Density", "Recovered Shape"],
        )
        # STL post-processing.
        # When True, applies the trimesh print-ready pipeline (hole fill +
        # light Humphrey smoothing + optional decimation) on top of marching cubes.
        self.create_property("print_ready_mesh", False, widget_type="bool")
        # 1.0 = no decimation; lower values reduce face count (requires
        # `fast_simplification` to be installed — silently skipped otherwise).
        self.create_property("mesh_decimate_ratio", 1.0, widget_type="float")
        self.create_property(
            "surface_recovery_method",
            "Volume-Preserving SDF (VTK)",
            widget_type="combo",
            items=["Volume-Preserving SDF (VTK)", "Legacy Marching Cubes"],
        )
        self.create_property(
            "structure_mode",
            "Solid Envelope",
            widget_type="combo",
            items=[
                "Solid Envelope",
                "Topology-Following Ribs",
                "Gyroid Lattice",
                "Diamond Lattice",
                "Honeycomb Lattice",
                "Cubic Lattice",
                "Octet Truss Lattice",
            ],
        )
        self.create_property("structure_cell_size_voxels", 8.0, widget_type="float")
        self.create_property(
            "structure_member_thickness_voxels",
            1.0,
            widget_type="float",
        )
        self.create_property(
            "structure_skin_thickness_voxels", 0.75, widget_type="float"
        )
        self.create_property("lattice_variable_density", True, widget_type="bool")
        self.create_property("lattice_min_relative_density", 0.15, widget_type="float")
        self.create_property("lattice_max_relative_density", 0.60, widget_type="float")
        self.create_property(
            "lattice_solid_transition_density", 0.92, widget_type="float"
        )
        self.create_property("E0", 1.0, widget_type="float")
        self.create_property("Emin", 1e-9, widget_type="float")
        self.create_property("nu", 0.3, widget_type="float")

        # Solver settings.
        self.create_property(
            "optimizer",
            "Auto",
            widget_type="combo",
            items=["Auto", "OC", "MMA", "GCMMA", "Projected Gradient"],
        )
        self.create_property(
            "load_aggregation",
            "Weighted Sum",
            widget_type="combo",
            items=["Weighted Sum", "Worst Case"],
        )
        self.create_property("load_pnorm_p", 8.0, widget_type="float")
        self.create_property("projected_gradient_step", 0.15, widget_type="float")
        self.create_property("max_iter", 80, widget_type="int")
        # `tol` is the relative-compliance-change convergence threshold; the run
        # stops once it holds for `convergence_patience` consecutive iterations.
        self.create_property("tol", 0.01, widget_type="float")
        self.create_property("convergence_patience", 5, widget_type="int")

        # Stress constraint (engineering surface only).
        # User sees: Enable + Yield Stress.  The qp-approach exponent q,
        # PNorm exponent, and multi-LC aggregation are silent industrial
        # defaults inside the solver (q=0.5 Bruggi 2008, p=8.0 PNorm).
        self.create_property("stress_constraint", False, widget_type="bool")
        self.create_property("yield_stress", 250.0, widget_type="float")

        # Steady-state thermal and coupled thermo-structural controls.
        self.create_property("thermal_conductivity", 1.0, widget_type="float")
        self.create_property("thermal_conductivity_min", 1e-6, widget_type="float")
        self.create_property("thermal_penal", 3.0, widget_type="float")
        self.create_property("structural_weight", 1.0, widget_type="float")
        self.create_property("thermal_weight", 1.0, widget_type="float")
        # Manufacturing constraints.
        # `symmetry`: subset of {X,Y,Z} meaning "axes to mirror across the
        # domain centre".  E.g. 'Y' = mirror plane XZ (left-right symmetric);
        # 'XY' = mirror both X and Y.
        self.create_property(
            "symmetry",
            "None",
            widget_type="combo",
            items=["None", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"],
        )
        self.create_property(
            "extrusion", "None", widget_type="combo", items=["None", "X", "Y", "Z"]
        )
        # AM build direction.  '+Y' means parts grow upward in +Y; voxels are
        # only allowed if supported by a 3×3 neighbourhood of denser voxels
        # in the layer below (45° self-supporting rule).
        self.create_property(
            "overhang_build_axis",
            "None",
            widget_type="combo",
            items=["None", "+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        )
        # Max member size — radius in voxels (0 = disabled); local-mean
        # threshold defaults to 0.6 and isn't user-exposed (industry default).
        self.create_property("max_member_size_voxels", 0.0, widget_type="float")
        # N-fold rotational pattern. 1 = disabled.
        self.create_property("pattern_repeat", 1, widget_type="int")
        self.create_property(
            "pattern_axis", "Y", widget_type="combo", items=["X", "Y", "Z"]
        )

    # Property helpers.

    def apply_guided_defaults(self) -> dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_topopt_defaults(
            self.get_property("design_goal"),
            "Automatic",
            self.get_property("manufacturing_process"),
            nelx=self.get_property("nelx") or 30,
            nely=self.get_property("nely") or 20,
            nelz=self.get_property("nelz") or 10,
        )
        for key, value in settings.items():
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
        from ..verification.calculix_validation import run_topopt_validation

        quality = str(self.get_property("validation_quality") or "Standard")
        return run_topopt_validation(
            topo_output,
            material,
            constraint_list,
            load_list,
            convergence_levels=2 if quality == "Mesh Convergence" else 1,
            max_validation_elements=500000,
            run_external_solver=True,
            deck_only=False,
            analysis_type="Linear",
            visualization="Von Mises Stress",
            deformation_scale="Auto",
            cancel_callback=cancel_callback,
        )

    def _run_embedded_cad_reconstruction(
        self,
        topo_output: dict[str, Any],
    ) -> Any:
        """Build a CAD body as an internal TopOpt study action."""
        from ..geometry.cad_reconstruction import reconstruct_topopt_cad

        self.set_property("cad_reconstruction_method", "Recovered Shape")
        topo_payload = dict(topo_output or {})
        return reconstruct_topopt_cad(
            topo_payload,
            source_geometry="Recovered Shape",
            sew_tolerance=1e-4,
            merge_angle_deg=0.0,
        )

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
                logger.warning(
                    "TopologyOptVoxelNode: cannot map graph BCs without mesh bounds."
                )
            return

        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            dofs = constraint.get("fixed_dofs") or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                continue
            for face in constraint.get("geometries") or []:
                cylinder = _cylinder_void_region_from_face(face, bounds)
                if cylinder is not None:
                    _add_cylindrical_contact_region(bc, cylinder)
            bboxes = _entry_bboxes(constraint)
            if not bboxes:
                logger.warning(
                    "TopologyOptVoxelNode: connected constraint did not "
                    "include selected geometry or bounding-box metadata; it "
                    "cannot be "
                    "mapped to voxel supports."
                )
                continue
            for bbox in bboxes:
                frac_box = _fraction_box(bbox, bounds)
                bc.fixed_boxes.append((*frac_box, dofs))

        for load in loads:
            if not isinstance(load, dict):
                continue
            load_type = str(load.get("type") or "").lower()
            if load_type == "force":
                vector = load.get("vector")
                try:
                    fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
                except Exception:
                    continue
                for face in _iter_load_faces(load):
                    cylinder = _cylinder_void_region_from_face(face, bounds)
                    if cylinder is not None:
                        _add_cylindrical_contact_region(bc, cylinder)
                bboxes = _entry_bboxes(load)
                if not bboxes:
                    logger.warning(
                        "TopologyOptVoxelNode: connected force load did not "
                        "include selected geometry or bounding-box metadata; "
                        "it cannot be "
                        "mapped to voxel forces."
                    )
                    continue
                scale = 1.0 / max(1, len(bboxes))
                for bbox in bboxes:
                    frac_box = _fraction_box(bbox, bounds)
                    # Graph-selected load faces become distributed patch loads,
                    # not mathematical point loads.  The same patch is also kept
                    # local so CAD face loads stay tied to their source face.
                    bc.box_forces.append(
                        (*frac_box, fx * scale, fy * scale, fz * scale)
                    )
            elif load_type == "pressure":
                for face in _iter_load_faces(load):
                    cylinder = _cylinder_void_region_from_face(face, bounds)
                    if cylinder is not None:
                        _add_cylindrical_contact_region(bc, cylinder)
                pressure_patches = _pressure_load_patches(load, bounds)
                if not pressure_patches:
                    logger.warning(
                        "TopologyOptVoxelNode: pressure load could not be "
                        "mapped to voxel patch forces."
                    )
                    continue
                for frac_box, fx, fy, fz in pressure_patches:
                    bc.box_forces.append((*frac_box, fx, fy, fz))
            else:
                logger.debug(
                    "TopologyOptVoxelNode: load type %r is not supported by "
                    "the voxel optimiser.",
                    load_type,
                )

    def _merge_graph_joints(
        self,
        bc: VoxelBC,
        bounds: tuple[np.ndarray, np.ndarray],
        joints: list[Any],
    ) -> None:
        """Convert graph joints and preserve selected pin/bore interfaces."""
        for payload in joints:
            joint = _joint_from_graph_payload(payload, bounds)
            bc.joints.append(joint)
            if not isinstance(payload, dict):
                continue
            # A penalty coupling transfers load but does not itself create
            # material or void. Preserve cylindrical anchor faces explicitly
            # as a solid sleeve around the original bore so recovery and STEP
            # export cannot fill the hole or erase its load-transfer interface.
            anchor_cylinders: dict[str, list[tuple[Any, ...]]] = {
                "anchor_a_geometries": [],
                "anchor_b_geometries": [],
            }
            for key in anchor_cylinders:
                for geometry in _flatten(payload.get(key)):
                    cylinder = _cylinder_void_region_from_face(geometry, bounds)
                    if cylinder is not None:
                        anchor_cylinders[key].append(cylinder)
                        _add_cylindrical_contact_region(bc, cylinder)
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
        """Convert one TopOpt Operating Case node to the typed solver model."""
        if (
            not isinstance(payload, dict)
            or str(payload.get("type") or "").lower() != "topology_operating_case"
        ):
            raise ValueError(
                "The load_cases input accepts TopOpt Operating Case nodes only."
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
            name=str(payload.get("name") or "Operating Case"),
            weight=float(payload.get("weight") or 1.0),
            point_forces=list(local_bc.point_forces),
            box_forces=list(local_bc.box_forces),
            distributed_forces=list(local_bc.distributed_forces),
            fixed_boxes=list(local_bc.fixed_boxes),
            replace_supports=bool(payload.get("replace_supports", True)),
            joints=list(local_bc.joints),
            replace_joints=bool(payload.get("replace_joints", False)),
            joint_pin_cylinders=list(local_bc.joint_pin_cylinders),
        )
