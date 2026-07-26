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

from pylcss.design_studio.core.base_node import CadQueryNode, resolve_numeric_input
from pylcss.solver_backends.common import as_bool


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
            length = _positive(
                "Length",
                resolve_numeric_input(
                    self.get_input("length"), self.get_property("length_x")
                ),
            )
            width = _positive(
                "Width",
                resolve_numeric_input(
                    self.get_input("width"), self.get_property("width_y")
                ),
            )
            height = _positive(
                "Height",
                resolve_numeric_input(
                    self.get_input("height"), self.get_property("height_z")
                ),
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


class TubeNode(CadQueryNode):
    """Create a hollow circular tube without scripted subtraction."""

    __identifier__ = "com.cad.geometry.tube"
    NODE_NAME = "Tube"

    def __init__(self):
        super().__init__()
        for port in ("outer_diameter", "wall_thickness", "length"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("outer_diameter", 40.0, widget_type="float")
        self.create_property("wall_thickness", 2.0, widget_type="float")
        self.create_property("length", 100.0, widget_type="float")
        self.create_property(
            "axis",
            "X",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property("centered", True, widget_type="bool")

    def run(self):
        self.clear_error()
        try:
            outer = _positive(
                "Outer diameter",
                resolve_numeric_input(
                    self.get_input("outer_diameter"),
                    self.get_property("outer_diameter"),
                ),
            )
            wall = _positive(
                "Wall thickness",
                resolve_numeric_input(
                    self.get_input("wall_thickness"),
                    self.get_property("wall_thickness"),
                ),
            )
            length = _positive(
                "Length",
                resolve_numeric_input(self.get_input("length"), self.get_property("length")),
            )
            inner = outer - 2.0 * wall
            if inner <= 0.0:
                raise ValueError("Wall thickness must be less than half the diameter.")
            direction = _axis_vector(str(self.get_property("axis") or "X"))
            start = (
                direction.multiply(-0.5 * length)
                if as_bool(self.get_property("centered"))
                else cq.Vector(0.0, 0.0, 0.0)
            )
            outside = cq.Solid.makeCylinder(0.5 * outer, length, start, direction)
            inside = cq.Solid.makeCylinder(0.5 * inner, length, start, direction)
            return _workplane_from_shape(outside.cut(inside))
        except Exception as exc:
            self.set_error(f"Tube creation failed: {exc}")
            return None


class CylindricalShellNode(CadQueryNode):
    """Create a cylindrical midsurface for shell-element analysis."""

    __identifier__ = "com.cad.geometry.cylindrical_shell"
    NODE_NAME = "Cylindrical Shell"

    def __init__(self):
        super().__init__()
        for port in ("diameter", "length"):
            self.add_input(port, color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("diameter", 70.0, widget_type="float")
        self.create_property("length", 180.0, widget_type="float")
        self.create_property(
            "axis",
            "X",
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
            direction = _axis_vector(str(self.get_property("axis") or "X"))
            start = (
                direction.multiply(-0.5 * length)
                if as_bool(self.get_property("centered"))
                else cq.Vector(0.0, 0.0, 0.0)
            )
            temporary = cq.Solid.makeCylinder(
                0.5 * diameter, length, start, direction
            )
            lateral_faces = [
                face
                for face in temporary.Faces()
                if str(face.geomType()).upper() == "CYLINDER"
            ]
            if len(lateral_faces) != 1:
                raise ValueError("Could not isolate the cylindrical midsurface.")
            return _workplane_from_shape(lateral_faces[0])
        except Exception as exc:
            self.set_error(f"Cylindrical shell creation failed: {exc}")
            return None


class BooleanNode(CadQueryNode):
    """Union, subtract, or intersect two connected solids."""

    __identifier__ = "com.cad.geometry.boolean"
    NODE_NAME = "Boolean"

    def __init__(self):
        super().__init__()
        self.add_input("base", color=_SHAPE_COLOR)
        self.add_input("tool", color=_SHAPE_COLOR)
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
        tool = self.get_input_shape("tool")
        if base is None or tool is None:
            self.set_error("Connect both Base and Tool shapes to Boolean.")
            return None
        try:
            a, b = _shape_value(base), _shape_value(tool)
            operation = str(self.get_property("operation") or "Union")
            if operation == "Subtract":
                result = a.cut(b)
            elif operation == "Intersect":
                result = a.intersect(b)
            else:
                result = a.fuse(b)
            if result is None or result.isNull():
                raise ValueError("The selected operation produced an empty solid.")
            return _workplane_from_shape(result)
        except Exception as exc:
            self.set_error(f"Boolean operation failed: {exc}")
            return None


class ThroughHoleNode(CadQueryNode):
    """Cut a dimensioned through-hole along a principal axis."""

    __identifier__ = "com.cad.geometry.through_hole"
    NODE_NAME = "Through Hole"

    def __init__(self):
        super().__init__()
        self.add_input("shape", color=_SHAPE_COLOR)
        self.add_input("diameter", color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("diameter", 10.0, widget_type="float")
        self.create_property(
            "axis",
            "Z",
            widget_type="combo",
            items=["X", "Y", "Z"],
        )
        self.create_property("center_x", 0.0, widget_type="float")
        self.create_property("center_y", 0.0, widget_type="float")
        self.create_property("center_z", 0.0, widget_type="float")

    def run(self):
        self.clear_error()
        source = self.get_input_shape("shape")
        if source is None:
            self.set_error("Connect a solid to Through Hole.")
            return None
        try:
            diameter = _positive(
                "Diameter",
                resolve_numeric_input(
                    self.get_input("diameter"), self.get_property("diameter")
                ),
            )
            center = cq.Vector(
                _finite("Center X", self.get_property("center_x")),
                _finite("Center Y", self.get_property("center_y")),
                _finite("Center Z", self.get_property("center_z")),
            )
            solid = _shape_value(source)
            bounds = solid.BoundingBox()
            length = max(bounds.xlen, bounds.ylen, bounds.zlen) * 3.0 + diameter
            direction = _axis_vector(str(self.get_property("axis") or "Z"))
            start = center - direction.multiply(0.5 * length)
            cutter = cq.Solid.makeCylinder(0.5 * diameter, length, start, direction)
            return _workplane_from_shape(solid.cut(cutter))
        except Exception as exc:
            self.set_error(f"Through-hole operation failed: {exc}")
            return None


class FilletNode(CadQueryNode):
    """Apply a constant-radius fillet to a reproducible edge family."""

    __identifier__ = "com.cad.geometry.fillet"
    NODE_NAME = "Fillet"

    def __init__(self):
        super().__init__()
        self.add_input("shape", color=_SHAPE_COLOR)
        self.add_input("radius", color=_NUMBER_COLOR)
        self.add_output("shape", color=_SHAPE_COLOR)
        self.create_property("radius", 2.0, widget_type="float")
        self.create_property(
            "edges",
            "All",
            widget_type="combo",
            items=["All", "Parallel X", "Parallel Y", "Parallel Z"],
        )

    def run(self):
        self.clear_error()
        source = self.get_input_shape("shape")
        if source is None:
            self.set_error("Connect a solid to Fillet.")
            return None
        try:
            radius = _positive(
                "Radius",
                resolve_numeric_input(self.get_input("radius"), self.get_property("radius")),
            )
            selector = {
                "Parallel X": "|X",
                "Parallel Y": "|Y",
                "Parallel Z": "|Z",
            }.get(str(self.get_property("edges") or "All"))
            workplane = _workplane_from_shape(source)
            edges = workplane.edges(selector) if selector else workplane.edges()
            return edges.fillet(radius)
        except Exception as exc:
            self.set_error(f"Fillet failed: {exc}")
            return None


class TransformNode(CadQueryNode):
    """Translate and rotate a connected solid."""

    __identifier__ = "com.cad.geometry.transform"
    NODE_NAME = "Transform"

    def __init__(self):
        super().__init__()
        self.add_input("shape", color=_SHAPE_COLOR)
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
            dx = _finite("Translation X", self.get_property("translate_x"))
            dy = _finite("Translation Y", self.get_property("translate_y"))
            dz = _finite("Translation Z", self.get_property("translate_z"))
            angle = _finite(
                "Rotation angle", self.get_property("rotation_angle_deg")
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
            count = int(self.get_property("count") or 0)
            if not 1 <= count <= 1000:
                raise ValueError("Count must be between 1 and 1000.")
            spacing = _finite("Spacing", self.get_property("spacing"))
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
