# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""GUI-native parametric solid-modeling nodes.

The nodes expose ordinary engineering dimensions and choices in the inspector.
They provide a reproducible no-code authoring path for Design Studio studies.
"""
from __future__ import annotations

import math
from typing import Any

import cadquery as cq

from pylcss.cae.design_studio.core.base_node import (
    CadQueryNode,
    is_shape,
    resolve_numeric_input,
)
from pylcss.input_values import as_bool


_SHAPE_COLOR = (100, 255, 100)
_NUMBER_COLOR = (180, 220, 255)


def _positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return number


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _shape_value(value: Any):
    if isinstance(value, cq.Assembly):
        return value.toCompound()
    if hasattr(value, "val"):
        return value.val()
    return value


def _workplane_from_shape(shape: Any) -> cq.Workplane:
    return cq.Workplane("XY").newObject([_shape_value(shape)])


def _axis_vector(axis: str) -> cq.Vector:
    return {
        "X": cq.Vector(1.0, 0.0, 0.0),
        "Y": cq.Vector(0.0, 1.0, 0.0),
        "Z": cq.Vector(0.0, 0.0, 1.0),
    }[str(axis or "Z").upper()]


class BoxNode(CadQueryNode):
    """Create a dimensioned rectangular design space."""

    __identifier__ = "com.cad.geometry.box"
    NODE_NAME = "Box"

    def __init__(self):
        super().__init__()
        for port in ("length", "width", "height"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("length_x", 100.0, widget_type="float")
        self.create_property("width_y", 40.0, widget_type="float")
        self.create_property("height_z", 20.0, widget_type="float")
        self.create_property("centered", True, widget_type="bool")

    def run(self):
        self.clear_error()
        try:
            raw_len = self.get_property("length_x")
            if raw_len is None and self.has_property("length"):
                raw_len = self.get_property("length")
            length = _positive(
                "Length",
                resolve_numeric_input(self.get_input("length"), raw_len),
            )
            raw_width = self.get_property("width_y")
            width = _positive(
                "Width",
                resolve_numeric_input(self.get_input("width"), raw_width),
            )
            raw_height = self.get_property("height_z")
            height = _positive(
                "Height",
                resolve_numeric_input(self.get_input("height"), raw_height),
            )
            centered = as_bool(self.get_property("centered"))
            return cq.Workplane("XY").box(
                length,
                width,
                height,
                centered=(centered, centered, centered),
            )
        except Exception as exc:
            self.set_error(f"Box creation failed: {exc}")
            return None


class CylinderNode(CadQueryNode):
    """Create a solid cylinder aligned with a principal axis."""

    __identifier__ = "com.cad.geometry.cylinder"
    NODE_NAME = "Cylinder"

    def __init__(self):
        super().__init__()
        for port in ("diameter", "length"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("diameter", 20.0, widget_type="float")
        self.create_property("length", 40.0, widget_type="float")
        self.create_property(
            "axis",
            "Z",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property("centered", True, widget_type="bool")

    def run(self):
        self.clear_error()
        try:
            diameter = _positive(
                "Diameter",
                resolve_numeric_input(
                    self.get_input("diameter"), self.get_property("diameter")
                ),
            )
            length = _positive(
                "Length",
                resolve_numeric_input(self.get_input("length"), self.get_property("length")),
            )
            direction = _axis_vector(str(self.get_property("axis") or "Z"))
            start = (
                direction.multiply(-0.5 * length)
                if as_bool(self.get_property("centered"))
                else cq.Vector(0.0, 0.0, 0.0)
            )
            solid = cq.Solid.makeCylinder(0.5 * diameter, length, start, direction)
            return _workplane_from_shape(solid)
        except Exception as exc:
            self.set_error(f"Cylinder creation failed: {exc}")
            return None


class BooleanNode(CadQueryNode):
    """Union, subtract, or intersect two connected solids."""

    __identifier__ = "com.cad.geometry.boolean"
    NODE_NAME = "Boolean"

    def __init__(self):
        super().__init__()
        self.add_input("base", color=_SHAPE_COLOR)
        self.add_input("tool", color=_SHAPE_COLOR, multi_input=True)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property(
            "operation",
            "Union",
            widget_type="combo",
            items=["Union", "Subtract", "Intersect"],
        )

    def run(self):
        self.clear_error()
        base = self.get_input_shape("base")
        if base is None:
            self.set_error("Connect a Base shape to Boolean.")
            return None

        connected_tools = self.get_input_list("tool")
        tool_shapes = []
        for item in connected_tools:
            if item is None:
                continue
            if is_shape(item):
                tool_shapes.append(_shape_value(item))
            elif hasattr(item, "val"):
                tool_shapes.append(item.val())
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if is_shape(sub):
                        tool_shapes.append(_shape_value(sub))
        if not tool_shapes:
            single = self.get_input_shape("tool")
            if single is not None:
                tool_shapes = [_shape_value(single)]

        if not tool_shapes:
            self.set_error("Connect at least one Tool shape to Boolean.")
            return None

        try:
            result = _shape_value(base)
            operation = str(self.get_property("operation") or "Union")
            for t in tool_shapes:
                if operation == "Subtract":
                    result = result.cut(t)
                elif operation == "Intersect":
                    result = result.intersect(t)
                else:
                    result = result.fuse(t)
                if result is None or result.isNull():
                    raise ValueError("The selected operation produced an empty solid.")
            return _workplane_from_shape(result)
        except Exception as exc:
            self.set_error(f"Boolean operation failed: {exc}")
            return None


class TransformNode(CadQueryNode):
    """Translate and rotate a connected solid."""

    __identifier__ = "com.cad.geometry.transform"
    NODE_NAME = "Transform"

    def __init__(self):
        super().__init__()
        self.add_input("shape", color=_SHAPE_COLOR)
        for port in ("translate_x", "translate_y", "translate_z", "rotation_angle_deg"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("translate_x", 0.0, widget_type="float")
        self.create_property("translate_y", 0.0, widget_type="float")
        self.create_property("translate_z", 0.0, widget_type="float")
        self.create_property(
            "rotation_axis",
            "Z",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property("rotation_angle_deg", 0.0, widget_type="float")

    def run(self):
        self.clear_error()
        source = self.get_input_shape("shape")
        if source is None:
            self.set_error("Connect a solid to Transform.")
            return None
        try:
            dx = _finite(
                "Translation X",
                resolve_numeric_input(self.get_input("translate_x"), self.get_property("translate_x")),
            )
            dy = _finite(
                "Translation Y",
                resolve_numeric_input(self.get_input("translate_y"), self.get_property("translate_y")),
            )
            dz = _finite(
                "Translation Z",
                resolve_numeric_input(self.get_input("translate_z"), self.get_property("translate_z")),
            )
            angle = _finite(
                "Rotation angle",
                resolve_numeric_input(self.get_input("rotation_angle_deg"), self.get_property("rotation_angle_deg")),
            )
            result = _workplane_from_shape(source)
            if abs(angle) > 1e-12:
                vector = _axis_vector(str(self.get_property("rotation_axis") or "Z"))
                result = result.rotate(
                    (0.0, 0.0, 0.0),
                    (vector.x, vector.y, vector.z),
                    angle,
                )
            if any(abs(v) > 1e-12 for v in (dx, dy, dz)):
                result = result.translate((dx, dy, dz))
            return result
        except Exception as exc:
            self.set_error(f"Transform failed: {exc}")
            return None


class LinearPatternNode(CadQueryNode):
    """Create a linear array of a connected part."""

    __identifier__ = "com.cad.geometry.linear_pattern"
    NODE_NAME = "Linear Pattern"

    def __init__(self):
        super().__init__()
        self.add_input("shape", color=_SHAPE_COLOR)
        for port in ("count", "spacing"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("count", 2, widget_type="int")
        self.create_property("spacing", 20.0, widget_type="float")
        self.create_property(
            "axis",
            "X",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property("fuse", False, widget_type="bool")

    def run(self):
        self.clear_error()
        source = self.get_input_shape("shape")
        if source is None:
            self.set_error("Connect a solid to Linear Pattern.")
            return None
        try:
            raw_count = resolve_numeric_input(self.get_input("count"), self.get_property("count"))
            count = int(raw_count or 0)
            if not 1 <= count <= 1000:
                raise ValueError("Count must be between 1 and 1000.")
            spacing = _finite(
                "Spacing",
                resolve_numeric_input(self.get_input("spacing"), self.get_property("spacing")),
            )
            vector = _axis_vector(str(self.get_property("axis") or "X"))
            original = _shape_value(source)
            parts = [
                original.moved(
                    cq.Location(
                        cq.Vector(
                            vector.x * spacing * index,
                            vector.y * spacing * index,
                            vector.z * spacing * index,
                        )
                    )
                )
                for index in range(count)
            ]
            if as_bool(self.get_property("fuse")):
                result = parts[0]
                for part in parts[1:]:
                    result = result.fuse(part)
                return _workplane_from_shape(result)
            return _workplane_from_shape(cq.Compound.makeCompound(parts))
        except Exception as exc:
            self.set_error(f"Linear pattern failed: {exc}")
            return None
