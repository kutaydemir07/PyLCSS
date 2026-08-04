# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEA component definitions for multi-body, multi-material studies."""

from __future__ import annotations

from pylcss.design_studio.core.base_node import CadQueryNode


class FEAComponentNode(CadQueryNode):
    """Pair one analysis mesh with its material and engineering identity."""

    __identifier__ = "com.cad.sim.component"
    NODE_NAME = "Body"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("mesh", color=(200, 100, 200))
        self.add_input("material", color=(200, 200, 200))
        self.add_output("component", color=(120, 190, 255))
        self.create_property("component_name", "Body 1", widget_type="text")

    def run(self):
        self.clear_error()
        mesh = self.get_input_value("mesh", None)
        material = self.get_input_value("material", None)
        name = str(self.get_property("component_name") or "").strip()
        if mesh is None:
            self.set_error("Connect an analysis mesh to this FEA component.")
            return None
        if material is None:
            self.set_error("Connect a material to this FEA component.")
            return None
        if not name:
            self.set_error("Component name cannot be empty.")
            return None
        return {
            "type": "fea_component",
            "name": name,
            "mesh": mesh,
            "material": material,
        }


__all__ = ["FEAComponentNode"]
