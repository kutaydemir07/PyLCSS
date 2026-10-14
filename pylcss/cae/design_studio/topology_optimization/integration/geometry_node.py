# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Graph ports and geometry properties shared by topology and direct infill."""

from pylcss.cae.design_studio.core.base_node import CadQueryNode

from ..models.study import VoxelBC


class GeometryStudyNode(CadQueryNode):
    """Provide geometry inputs/outputs without adding optimization controls."""

    def __init__(self) -> None:
        super().__init__()
        self.add_output("results", color=(180, 255, 180))
        self.add_output("recovered_shape", color=(100, 255, 100))
        self.add_input("design_domain", color=(100, 255, 100))
        self.create_property("non_design_enabled", True, widget_type="bool")
        self.create_property("non_design_regions", "[]", widget_type="string")
        self.create_property("density_cutoff", 0.30, widget_type="float")
        self.create_property(
            "visualization",
            "STL Shape",
            widget_type="combo",
            items=["CAD", "STL Shape", "Density", "Lattice"],
        )
        self.create_property(
            "cad_export_filename", "topology_optimized.step", widget_type="string"
        )
        self.create_property("cad_export_format", "STEP", widget_type="string")

    def _build_bc(self) -> VoxelBC:
        return VoxelBC()
