# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Interactive graph nodes used to define topology-optimization studies.

The node graph is the engineering model. Supports, loads, joints, operating
cases, heat sinks, and heat inputs are therefore first-class nodes rather than
JSON strings embedded in the optimizer inspector.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from pylcss.design_studio.core.base_node import CadQueryNode
from pylcss.solver_backends.common import as_bool


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
        faces = payload.get("faces") or payload.get("geometries")
        if faces:
            return [face for face in faces if face is not None]
        face = payload.get("face") or payload.get("geometry")
        return [face] if face is not None else []
    if hasattr(payload, "vals"):
        try:
            return [item for item in payload.vals() if item is not None]
        except Exception:
            return []
    return [payload] if payload is not None else []


def _region_payload(node: CadQueryNode, port_name: str) -> list[Any] | None:
    geometries = _region_geometries(node.get_input_value(port_name, None))
    if not geometries:
        node.set_error(
            f"Connect a Select Face or Select Face (Interactive) node to "
            f"'{port_name}'."
        )
        return None
    return geometries


class TopologySupportNode(CadQueryNode):
    """A selected-face structural support for a topology study."""

    __identifier__ = "com.cad.topopt.support"
    NODE_NAME = "TopOpt Support"

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
                "Roller X",
                "Roller Y",
                "Roller Z",
                "Symmetry X",
                "Symmetry Y",
                "Symmetry Z",
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
            "fixed_dofs": mapping[support_type],
            "geometries": geometries,
        }


class TopologyLoadNode(CadQueryNode):
    """A selected-face resultant force for a topology operating condition."""

    __identifier__ = "com.cad.topopt.load"
    NODE_NAME = "TopOpt Force"

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
        }


class TopologyJointNode(CadQueryNode):
    """A translational penalty joint between two interactively selected regions."""

    __identifier__ = "com.cad.topopt.joint"
    NODE_NAME = "TopOpt Joint"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("anchor_a", color=(100, 200, 255))
        self.add_input("anchor_b", color=(100, 200, 255))
        self.add_output("joints", color=(255, 170, 80))
        self.create_property("joint_name", "Joint", widget_type="string")
        self.create_property(
            "joint_type",
            "Spherical",
            widget_type="combo",
            items=["Fixed", "Revolute", "Spherical", "Prismatic"],
        )
        self.create_property(
            "axis",
            "X",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property(
            "relative_stiffness",
            100.0,
            widget_type="float",
        )

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        anchor_a = _region_payload(self, "anchor_a")
        anchor_b = _region_payload(self, "anchor_b")
        if anchor_a is None or anchor_b is None:
            return None
        try:
            relative_stiffness = float(
                self.get_property("relative_stiffness") or 100.0
            )
        except (TypeError, ValueError):
            self.set_error("Joint stiffness must be numeric.")
            return None
        if not math.isfinite(relative_stiffness) or relative_stiffness <= 0.0:
            self.set_error("Joint stiffness must be finite and greater than zero.")
            return None
        return {
            "type": "topology_joint",
            "name": str(self.get_property("joint_name") or "Joint"),
            "joint_type": str(
                self.get_property("joint_type") or "Spherical"
            ).lower(),
            "axis": str(self.get_property("axis") or "X").lower(),
            "relative_stiffness": relative_stiffness,
            "anchor_a_geometries": anchor_a,
            "anchor_b_geometries": anchor_b,
        }


class TopologyOperatingCaseNode(CadQueryNode):
    """Group supports, loads, and pose-local joints into one operating case."""

    __identifier__ = "com.cad.topopt.operating_case"
    NODE_NAME = "TopOpt Operating Case"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("supports", color=(255, 100, 100), multi_input=True)
        self.add_input("loads", color=(255, 255, 0), multi_input=True)
        self.add_input("joints", color=(255, 170, 80), multi_input=True)
        self.add_output("load_case", color=(120, 220, 255))
        self.create_property("case_name", "Operating Case 1", widget_type="string")
        self.create_property("weight", 1.0, widget_type="float")
        self.create_property("replace_base_supports", True, widget_type="bool")
        self.create_property("replace_global_joints", False, widget_type="bool")

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        supports = _flatten(self.get_input_list("supports"))
        loads = _flatten(self.get_input_list("loads"))
        joints = _flatten(self.get_input_list("joints"))
        if not supports:
            self.set_error("Connect at least one TopOpt Support to this operating case.")
            return None
        if not loads:
            self.set_error("Connect at least one TopOpt Force to this operating case.")
            return None
        try:
            weight = float(self.get_property("weight") or 1.0)
        except (TypeError, ValueError):
            self.set_error("Operating-case weight must be numeric.")
            return None
        if not math.isfinite(weight) or weight <= 0.0:
            self.set_error("Operating-case weight must be finite and greater than zero.")
            return None
        return {
            "type": "topology_operating_case",
            "name": str(self.get_property("case_name") or "Operating Case"),
            "weight": weight,
            "supports": supports,
            "loads": loads,
            "joints": joints,
            "replace_supports": as_bool(
                self.get_property("replace_base_supports")
            ),
            "replace_joints": as_bool(
                self.get_property("replace_global_joints")
            ),
        }


class TopologyThermalSinkNode(CadQueryNode):
    """A selected region held at the thermal reference temperature."""

    __identifier__ = "com.cad.topopt.thermal_sink"
    NODE_NAME = "TopOpt Thermal Sink"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("target_region", color=(100, 200, 255))
        self.add_output("thermal_sinks", color=(80, 190, 255))

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        geometries = _region_payload(self, "target_region")
        if geometries is None:
            return None
        return {
            "type": "topology_thermal_sink",
            "geometries": geometries,
        }


class TopologyHeatLoadNode(CadQueryNode):
    """Total heat input applied to a selected region."""

    __identifier__ = "com.cad.topopt.heat_load"
    NODE_NAME = "TopOpt Heat Load"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("target_region", color=(100, 200, 255))
        self.add_output("thermal_loads", color=(255, 110, 50))
        self.create_property("case_name", "Thermal Case 1", widget_type="string")
        self.create_property("total_heat", 100.0, widget_type="float")
        self.create_property("weight", 1.0, widget_type="float")

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        geometries = _region_payload(self, "target_region")
        if geometries is None:
            return None
        try:
            total_heat = float(self.get_property("total_heat") or 0.0)
            weight = float(self.get_property("weight") or 1.0)
        except (TypeError, ValueError):
            self.set_error("Heat input and case weight must be numeric.")
            return None
        if (
            not math.isfinite(total_heat)
            or total_heat <= 0.0
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            self.set_error(
                "Heat input and case weight must be finite and greater than zero."
            )
            return None
        return {
            "type": "topology_heat_load",
            "name": str(self.get_property("case_name") or "Thermal Case"),
            "total_heat": total_heat,
            "weight": weight,
            "geometries": geometries,
        }


__all__ = [
    "TopologySupportNode",
    "TopologyLoadNode",
    "TopologyJointNode",
    "TopologyOperatingCaseNode",
    "TopologyThermalSinkNode",
    "TopologyHeatLoadNode",
]
