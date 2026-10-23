# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for voxel topology optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry_node import GeometryStudyNode
from ..models.study import (
    VoxelBC,
    BoundaryMesh,
)
from ..configuration.presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_DESIGN_GOALS,
)
from ..configuration.voxel_meshing import (
    DEFAULT_VOXEL_MESH_QUALITY,
    DEFAULT_MAX_VOXEL_ELEMENTS,
    VOXEL_MESH_ELEMENT_BUDGETS,
)
from ..configuration.fea_meshing import FEA_MESH_DEFAULTS, FEA_QUALITY_OPTIONS
from .geometry_mapping import (
    _mesh_bounds,
)
from .boundary_mapping import (
    _pressure_load_patches,
)
from .selected_boundary import selection_regions, region_measure, boundary_with_force
from .node_execution import _TopologyExecutionMixin


class TopologyOptVoxelNode(_TopologyExecutionMixin, GeometryStudyNode):
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
        self.add_input("material", color=(200, 200, 200))
        # Both accept any number of connections, so several supports and
        # several forces are wired straight in. That is what made the separate
        # Load Case node redundant: grouping supports and forces into a case
        # node was only ever a way to get more than one of each.
        self.add_input("supports", color=(255, 100, 100), multi_input=True)
        self.add_input("loads", color=(255, 255, 0), multi_input=True)
        # Non-design material is authored on design-domain faces in the inspector.
        self.create_property("automatic_grid", True, widget_type="bool")
        self.create_property(
            "voxel_mesh_quality",
            DEFAULT_VOXEL_MESH_QUALITY,
            widget_type="combo",
            items=list(VOXEL_MESH_ELEMENT_BUDGETS),
        )
        self.create_property("voxel_size", 0.0, widget_type="float")
        self.create_property(
            "max_voxel_elements", DEFAULT_MAX_VOXEL_ELEMENTS, widget_type="int"
        )
        self.create_property("automatic_mesh_refinement", False, widget_type="bool")
        self.create_property("automatic_filter_radius", True, widget_type="bool")
        self.create_property("topology_minimum_solid_size_mm", 0.0, widget_type="float")
        self.create_property("topology_minimum_void_size_mm", 0.0, widget_type="float")
        self.create_property("voxel_ram_fraction", 0.5, widget_type="float")
        self.create_property("topology_objective_tolerance", 0.05, widget_type="float")
        self.create_property("topology_overlap_tolerance", 0.90, widget_type="float")
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
        self.create_property("advanced_settings_visible", False, widget_type="bool")
        # These controls mesh the final CAD for independent FEA. They do not
        # select the density optimizer's structured voxel grid.
        for name, default in FEA_MESH_DEFAULTS.items():
            if name == "fea_element_order":
                self.create_property(
                    name, default, widget_type="combo", items=["C3D10", "C3D4"]
                )
            elif name == 'fea_solver_backend':
                self.create_property(name, default, widget_type='combo', items=['Automatic', 'Sparse Direct', 'CalculiX'])
            else:
                self.create_property(
                    name,
                    default,
                    widget_type="int" if isinstance(default, int) else "float",
                )
        self.create_property(
            "fea_quality",
            "Standard",
            widget_type="combo",
            items=list(FEA_QUALITY_OPTIONS),
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
        # Domain and recovery settings.
        self.create_property("nelx", 30, widget_type="int")
        self.create_property("nely", 20, widget_type="int")
        self.create_property("nelz", 10, widget_type="int")
        self.create_property("volfrac", 0.5, widget_type="float")
        self.create_property("rmin", 1.5, widget_type="float")
        self.create_property("penal", 3.0, widget_type="float")
        self.create_property("E0", 1.0, widget_type="float")
        self.create_property("Emin", 1e-9, widget_type="float")
        self.create_property("nu", 0.3, widget_type="float")

        # Post-processing choices must never change the continuum solve.
        from ..manufacturing import PUBLIC_FAMILIES, PUBLIC_FAMILY_KEYS

        self.create_property(
            "post_lattice_type",
            "Gyroid Lattice",
            widget_type="combo",
            items=[PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS],
        )
        from .lattice_fit import DEFAULT_POST_LATTICE_QUALITY, POST_LATTICE_QUALITY_BUDGETS
        self.create_property(
            "post_lattice_quality", DEFAULT_POST_LATTICE_QUALITY,
            widget_type="combo", items=list(POST_LATTICE_QUALITY_BUDGETS),
        )
        self.create_property("post_lattice_cell_size_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_adaptive_cells", True, widget_type="bool")
        self.create_property("post_lattice_min_thickness_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_skin_thickness_mm", 0.0, widget_type="float")
        self.create_property("post_lattice_min_density", 0.15, widget_type="float")
        self.create_property("post_lattice_max_density", 0.60, widget_type="float")
        self.create_property("post_lattice_use_fea", False, widget_type="bool")
        self.create_property("post_lattice_sizing_iterations", 3, widget_type="int")

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

        # Stress limits are resolved from the connected material.
        self.create_property("stress_constraint", False, widget_type="bool")

    # Property helpers.

    def is_execution_property(self, name: str) -> bool:
        """FEA and lattice post-processing settings reuse the continuum solve."""
        from ..configuration.fea_meshing import is_topopt_fea_setting

        return (
            super().is_execution_property(name)
            and not name.startswith("post_lattice_")
            and not is_topopt_fea_setting(self, name)
        )

    def apply_guided_defaults(self) -> dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_topopt_defaults(
            self.get_property("design_goal"),
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
                raise ValueError(
                    "Cannot map connected supports and loads without design-domain bounds."
                )
            return

        for index, constraint in enumerate(constraints, start=1):
            if not isinstance(constraint, dict):
                raise ValueError(
                    f"Connected support {index} did not produce a support definition."
                )
            dofs = constraint.get("fixed_dofs") or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                raise ValueError(
                    f"Connected support {index} has no valid constrained DOFs."
                )
            regions = selection_regions(constraint, bounds)
            if not regions:
                raise ValueError(
                    f"Connected support {index} has no selected geometry or bounding-box metadata."
                )
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
                raise ValueError(
                    f"Connected load {index} did not produce a load definition."
                )
            load_type = str(load.get("type") or "").lower()
            local_boxes = []
            if load_type in ("force", "bearing_load"):
                vector = load.get("vector")
                try:
                    fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
                except Exception:
                    raise ValueError(
                        f"Connected load {index} needs three finite force components."
                    )
                if not np.all(np.isfinite([fx, fy, fz])):
                    raise ValueError(
                        f"Connected load {index} needs three finite force components."
                    )
                regions = selection_regions(load, bounds)
                if not regions:
                    raise ValueError(
                        f"Connected load {index} has no selected geometry or bounding-box metadata."
                    )
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
                        raise ValueError(
                            f"Connected pressure load {index} needs a finite pressure."
                        )
                    regions = selection_regions(load, bounds, exact_boxes=False)
                    for region in regions:
                        bc.boundary_forces.append(
                            boundary_with_force(region, pressure=pressure)
                        )
                    continue
                selection_regions(
                    load, bounds
                )  # Validate metadata before mapping pressure.
                pressure_patches = _pressure_load_patches(load, bounds)
                if not pressure_patches:
                    raise ValueError(
                        f"Connected pressure load {index} could not be mapped to voxel patch forces."
                    )
                for frac_box, fx, fy, fz in pressure_patches:
                    local_boxes.append((*frac_box, fx, fy, fz))
            else:
                raise ValueError(
                    f"Connected load {index} has unsupported type {load_type!r}."
                )

            if not local_boxes:
                continue

            # One study is one simultaneous operating scenario. Alternative
            # scenarios belong in separate study/project copies.
            bc.box_forces.extend(local_boxes)
