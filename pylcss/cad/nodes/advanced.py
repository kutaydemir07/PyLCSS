# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Advanced CAD nodes for professional workflows:
    - ImportStepNode: Import STEP/IGES/STL/OBJ files into the graph
    - ImportStlNode: Import STL mesh files
    - ThickenNode: Thicken a shell/surface into a solid
    - SplitNode: Split a solid by a plane
    - TextNode: 3D text geometry
    - PipeNode: Pipe/tube from path curve
    - ThreadNode: Helical thread feature
    - MathExpressionNode: Evaluate math expressions
"""

import logging
import math
import os
from typing import Optional

import numpy as np

from pylcss.cad.core.base_node import (
    CadQueryNode,
    resolve_shape_input,
    resolve_numeric_input,
    resolve_any_input,
)

logger = logging.getLogger(__name__)


class ImportStepNode(CadQueryNode):
    """Import a STEP/IGES file as geometry."""

    __identifier__ = "com.cad"
    NODE_NAME = "Import STEP"

    def __init__(self):
        super().__init__()
        self.add_text_input("filepath", "File Path", text="")
        self.add_output("shape_out")
        self.set_color(100, 160, 100)

    def run(self, **kwargs):
        self.clear_error()
        filepath = self.get_property("filepath") or ""
        if not filepath or not os.path.isfile(filepath):
            self.set_error("No valid file path")
            return None
        try:
            from pylcss.io_manager.cad_io import CADImporter
            result = CADImporter.import_file(filepath)
            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class ImportStlNode(CadQueryNode):
    """Import an STL/OBJ mesh file."""

    __identifier__ = "com.cad"
    NODE_NAME = "Import STL"

    def __init__(self):
        super().__init__()
        self.add_text_input("filepath", "File Path", text="")
        self.add_output("mesh_out")
        self.set_color(100, 160, 100)

    def run(self, **kwargs):
        self.clear_error()
        filepath = self.get_property("filepath") or ""
        if not filepath or not os.path.isfile(filepath):
            self.set_error("No valid file path")
            return None
        try:
            from pylcss.io_manager.cad_io import CADImporter
            result = CADImporter.import_file(filepath)
            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class ThickenNode(CadQueryNode):
    """Thicken a surface/shell into a solid by offsetting."""

    __identifier__ = "com.cad"
    NODE_NAME = "Thicken"

    def __init__(self):
        super().__init__()
        self.add_input("shape_in")
        self.add_output("shape_out")
        self.add_text_input("thickness", "Thickness", text="2.0")
        self.set_color(80, 130, 200)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq

            shape = resolve_shape_input(self.input(0))
            if shape is None:
                self.set_error("No input shape")
                return None

            thickness = float(self.get_property("thickness") or "2.0")

            if isinstance(shape, cq.Workplane):
                result = shape.shell(thickness)
            else:
                result = cq.Workplane("XY").newObject([shape]).shell(thickness)

            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class SplitNode(CadQueryNode):
    """Split a solid body along a plane."""

    __identifier__ = "com.cad"
    NODE_NAME = "Split Body"

    def __init__(self):
        super().__init__()
        self.add_input("shape_in")
        self.add_output("shape_out")
        self.add_combo_menu(
            "plane", "Split Plane", items=["XY", "XZ", "YZ"]
        )
        self.add_text_input("offset", "Offset", text="0.0")
        self.add_combo_menu("keep", "Keep", items=["Both", "Positive", "Negative"])
        self.set_color(200, 100, 100)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
            from OCP.gp import gp_Pln, gp_Pnt, gp_Dir

            shape = resolve_shape_input(self.input(0))
            if shape is None:
                self.set_error("No input shape")
                return None

            plane = self.get_property("plane") or "XY"
            offset = float(self.get_property("offset") or "0.0")
            keep = self.get_property("keep") or "Both"

            if isinstance(shape, cq.Workplane):
                wp = shape
            else:
                wp = cq.Workplane("XY").newObject([shape])

            # Use CadQuery's cut with a large box as splitting tool
            bb = wp.val().BoundingBox()
            size = max(bb.xlen, bb.ylen, bb.zlen) * 2

            if plane == "XY":
                tool = cq.Workplane("XY").transformed(offset=(0, 0, offset)).box(size, size, size, centered=(True, True, False))
            elif plane == "XZ":
                tool = cq.Workplane("XZ").transformed(offset=(0, 0, offset)).box(size, size, size, centered=(True, True, False))
            else:
                tool = cq.Workplane("YZ").transformed(offset=(0, 0, offset)).box(size, size, size, centered=(True, True, False))

            if keep == "Positive":
                result = wp.cut(tool)
            elif keep == "Negative":
                result = wp.intersect(tool)
            else:
                result = wp  # Return the full shape for "Both"

            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class PipeNode(CadQueryNode):
    """Create a pipe/tube along a path."""

    __identifier__ = "com.cad"
    NODE_NAME = "Pipe"

    def __init__(self):
        super().__init__()
        self.add_input("path_in")
        self.add_output("shape_out")
        self.add_text_input("outer_radius", "Outer Radius", text="5.0")
        self.add_text_input("inner_radius", "Inner Radius", text="3.0")
        self.set_color(80, 160, 180)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq

            path = resolve_shape_input(self.input(0))
            if path is None:
                self.set_error("No path input")
                return None

            outer_r = float(self.get_property("outer_radius") or "5.0")
            inner_r = float(self.get_property("inner_radius") or "3.0")

            # Create annular profile
            profile = (
                cq.Workplane("YZ")
                .circle(outer_r)
                .circle(inner_r)
            )

            if isinstance(path, cq.Workplane):
                result = profile.sweep(path)
            else:
                result = profile.sweep(cq.Workplane("XY").newObject([path]))

            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class TextNode(CadQueryNode):
    """Create 3D text geometry."""

    __identifier__ = "com.cad"
    NODE_NAME = "3D Text"

    def __init__(self):
        super().__init__()
        self.add_output("shape_out")
        self.add_text_input("text_content", "Text", text="pylcss")
        self.add_text_input("font_size", "Font Size", text="10.0")
        self.add_text_input("depth", "Depth", text="2.0")
        self.set_color(180, 140, 80)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq

            text = self.get_property("text_content") or "pylcss"
            font_size = float(self.get_property("font_size") or "10.0")
            depth = float(self.get_property("depth") or "2.0")

            result = (
                cq.Workplane("XY")
                .text(text, fontsize=font_size, distance=depth)
            )

            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class MathExpressionNode(CadQueryNode):
    """Evaluate a mathematical expression with input variables."""

    __identifier__ = "com.cad"
    NODE_NAME = "Math Expression"

    def __init__(self):
        super().__init__()
        self.add_input("x", multi_input=False)
        self.add_input("y", multi_input=False)
        self.add_input("z", multi_input=False)
        self.add_output("result")
        self.add_text_input("expression", "Expression", text="x + y")
        self.set_color(160, 120, 200)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import math as _math

            # Resolve inputs
            x_val = resolve_any_input(self.input(0))
            y_val = resolve_any_input(self.input(1))
            z_val = resolve_any_input(self.input(2))

            expression = self.get_property("expression") or "0"

            # Build a safe namespace for evaluation
            ns = {
                "np": np, "math": _math,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
                "abs": np.abs, "pi": np.pi, "e": np.e,
            }
            if x_val is not None:
                ns["x"] = float(x_val) if not isinstance(x_val, (list, np.ndarray)) else x_val
            if y_val is not None:
                ns["y"] = float(y_val) if not isinstance(y_val, (list, np.ndarray)) else y_val
            if z_val is not None:
                ns["z"] = float(z_val) if not isinstance(z_val, (list, np.ndarray)) else z_val

            result = eval(expression, {"__builtins__": {}}, ns)

            self._last_result = result
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class MeasureDistanceNode(CadQueryNode):
    """Measure distance between two shapes."""

    __identifier__ = "com.cad"
    NODE_NAME = "Measure Distance"

    def __init__(self):
        super().__init__()
        self.add_input("shape_a")
        self.add_input("shape_b")
        self.add_output("distance")
        self.set_color(120, 180, 120)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq
            from OCP.BRepExtrema import BRepExtrema_DistShapeShape

            shape_a = resolve_shape_input(self.input(0))
            shape_b = resolve_shape_input(self.input(1))

            if shape_a is None or shape_b is None:
                self.set_error("Two input shapes required")
                return None

            # Get OCC shapes
            if isinstance(shape_a, cq.Workplane):
                occ_a = shape_a.val().wrapped
            else:
                occ_a = shape_a.wrapped if hasattr(shape_a, "wrapped") else shape_a

            if isinstance(shape_b, cq.Workplane):
                occ_b = shape_b.val().wrapped
            else:
                occ_b = shape_b.wrapped if hasattr(shape_b, "wrapped") else shape_b

            dist_calc = BRepExtrema_DistShapeShape(occ_a, occ_b)
            if dist_calc.IsDone():
                distance = dist_calc.Value()
                self._last_result = distance
                return distance
            else:
                self.set_error("Distance calculation failed")
                return None
        except Exception as e:
            self.set_error(str(e))
            return None


class SurfaceAreaNode(CadQueryNode):
    """Compute surface area of a shape."""

    __identifier__ = "com.cad"
    NODE_NAME = "Surface Area"

    def __init__(self):
        super().__init__()
        self.add_input("shape_in")
        self.add_output("area_out")
        self.set_color(120, 180, 120)

    def run(self, **kwargs):
        self.clear_error()
        try:
            import cadquery as cq
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp

            shape = resolve_shape_input(self.input(0))
            if shape is None:
                self.set_error("No input shape")
                return None

            if isinstance(shape, cq.Workplane):
                occ_shape = shape.val().wrapped
            else:
                occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape

            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(occ_shape, props)
            area = props.Mass()

            self._last_result = area
            return area
        except Exception as e:
            self.set_error(str(e))
            return None
