# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Interactive graph nodes used to define topology-optimization studies.

The node graph is the engineering model, so supports and loads are first-class
nodes rather than JSON strings embedded in the optimizer inspector.

Both connect to multi-input ports on the study node, so a study takes any
number of each. The design domain arrives on the study node's own
``design_domain`` port, and regions to preserve or exclude are declared on the
study itself rather than as separate nodes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pylcss.design_studio.core.base_node import CadQueryNode
from pylcss.input_values import as_bool


def _flatten(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        flattened: list[Any] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [values]


def _region_geometries(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        entities = (
            payload.get("entities") or payload.get("faces") or payload.get("geometries")
        )
        if entities:
            return [entity for entity in entities if entity is not None]
        entity = payload.get("entity") or payload.get("face") or payload.get("geometry")
        return [entity] if entity is not None else []
    if hasattr(payload, "vals"):
        try:
            return [item for item in payload.vals() if item is not None]
        except Exception:
            return []
    return [payload] if payload is not None else []


def _region_payload(node: CadQueryNode, port_name: str) -> list[Any] | None:
    geometries = _region_geometries(node.get_input_value(port_name, None))
    if not geometries:
        # Both selection nodes emit the same payload.
        node.set_error(
            f"Connect a selection to '{port_name}'."
        )
        return None
    return geometries


def _region_entity_type(node: CadQueryNode, port_name: str) -> str:
    """Return the declared geometric dimension of a connected selection."""
    payload = node.get_input_value(port_name, None)
    if isinstance(payload, dict):
        entity_type = str(payload.get("entity_type") or "Face").title()
        if entity_type in {"Face", "Edge", "Vertex"}:
            return entity_type
    geometries = _region_geometries(payload)
    if not geometries:
        return "Unknown"
    try:
        entity_type = str(geometries[0].ShapeType()).title()
    except Exception:
        entity_type = "Unknown"
    return entity_type if entity_type in {"Face", "Edge", "Vertex"} else "Unknown"


class TopologySupportNode(CadQueryNode):
    """A structural support applied to selected topology-study geometry."""

    __identifier__ = "com.cad.topopt.support"
    NODE_NAME = "Fixed Support"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("target_region", color=(100, 200, 255))
        self.add_output("supports", color=(255, 100, 100))
        self.create_property(
            "support_type",
            "Fixed",
            widget_type="combo",
            items=[
                "Fixed",
                "Block X Translation",
                "Block Y Translation",
                "Block Z Translation",
                # Two-axis supports. A pin or bore that carries radial load but
                # lets the shaft float axially blocks exactly two translations,
                # and a sliding mount blocks the two axes it is not free along.
                # Both are ordinary suspension/linkage boundary conditions, and
                # the solver's fixed_boxes already accept any DOF combination.
                "Block XY Translation",
                "Block YZ Translation",
                "Block XZ Translation",
            ],
        )

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        geometries = _region_payload(self, "target_region")
        if geometries is None:
            return None
        support_type = str(self.get_property("support_type") or "Fixed")
        mapping = {
            "Fixed": [0, 1, 2],
            "Block X Translation": [0],
            "Block Y Translation": [1],
            "Block Z Translation": [2],
            "Block XY Translation": [0, 1],
            "Block YZ Translation": [1, 2],
            "Block XZ Translation": [0, 2],
            # Labels used by the combo before v2.2.0 renamed it. Studies saved
            # against those releases must keep resolving to the same DOFs.
            "Roller X": [0],
            "Roller Y": [1],
            "Roller Z": [2],
            "Symmetry X": [0],
            "Symmetry Y": [1],
            "Symmetry Z": [2],
        }
        return {
            "type": "topology_support",
            "support_type": support_type,
            "fixed_dofs": mapping.get(support_type, [0, 1, 2]),
            "geometries": geometries,
            "selection_entity_type": _region_entity_type(
                self, "target_region"
            ),
        }


class TopologyLoadNode(CadQueryNode):
    """A resultant force on selected topology-study geometry."""

    __identifier__ = "com.cad.topopt.load"
    NODE_NAME = "Force"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("target_region", color=(100, 200, 255))
        self.add_output("loads", color=(255, 255, 0))
        self.create_property("force_x", 0.0, widget_type="float")
        self.create_property("force_y", -1000.0, widget_type="float")
        self.create_property("force_z", 0.0, widget_type="float")

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        geometries = _region_payload(self, "target_region")
        if geometries is None:
            return None
        try:
            vector = [
                float(self.get_property("force_x") or 0.0),
                float(self.get_property("force_y") or 0.0),
                float(self.get_property("force_z") or 0.0),
            ]
        except (TypeError, ValueError):
            self.set_error("Force components must be numeric.")
            return None
        if not all(math.isfinite(value) for value in vector):
            self.set_error("Force components must be finite.")
            return None
        if float(np.linalg.norm(vector)) <= 1e-12:
            self.set_error("Enter at least one non-zero force component.")
            return None
        return {
            "type": "force",
            "vector": vector,
            "geometries": geometries,
            "selection_entity_type": _region_entity_type(
                self, "target_region"
            ),
        }
