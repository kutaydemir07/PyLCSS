# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Advanced CAD nodes for professional workflows:
    - ImportStepNode: Import STEP/IGES files into the graph
    - ImportStlNode: Import STL mesh files
    - MathExpressionNode: Evaluate math expressions
    - MeasureDistanceNode: Measure distance between two shapes
    - SurfaceAreaNode: Compute surface area of a shape
"""

import ast
import os

import numpy as np

from pylcss.design_studio.core.base_node import (
    CadQueryNode,
    resolve_shape_input,
    resolve_any_input,
)


def _resolve_data_path(filepath: str, project_dir: str | None = None) -> str:
    """Return an existing absolute path for ``filepath`` or the original.

    Saved projects historically stored absolute paths (e.g. the shipped
    sample .cad files), which break the moment another machine clones
    the repo. New saves prefer repo-relative paths like
    ``data/cantilever.stl``. This resolver lets both work: tries the
    path as given first (absolute and cwd-relative), then the directory that
    owns the saved project, and finally the PyLCSS repo root (parent of the
    pylcss package).
    """
    if not filepath:
        return filepath
    if os.path.isfile(filepath):
        return filepath
    if project_dir:
        candidate = os.path.join(project_dir, filepath)
        if os.path.isfile(candidate):
            return candidate
    try:
        from pylcss.config import BASE_DIR
        repo_root = os.path.dirname(BASE_DIR)
        candidate = os.path.join(repo_root, filepath)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return filepath


class ImportStepNode(CadQueryNode):
    """Import a STEP/IGES file as geometry."""

    __identifier__ = "com.cad.import_step"
    NODE_NAME = "Import STEP"

    def __init__(self):
        super().__init__()
        self.add_text_input("filepath", "File Path", text="")
        self.add_output("shape_out")
        self.set_color(100, 160, 100)

    def run(self, **kwargs):
        self.clear_error()
        filepath = _resolve_data_path(
            self.get_property("filepath") or "", getattr(self, "_project_dir", None)
        )
        if not filepath or not os.path.isfile(filepath):
            self.set_error("No valid file path")
            return None
        try:
            from pylcss.io_manager.cad_io import CADImporter
            result = CADImporter.import_file(filepath)
            return result
        except Exception as e:
            self.set_error(str(e))
            return None


class ImportStlNode(CadQueryNode):
    """Import an STL/OBJ mesh file."""

    __identifier__ = "com.cad.import_stl"
    NODE_NAME = "Import STL"

    def __init__(self):
        super().__init__()
        self.add_text_input("filepath", "File Path", text="")
        self.add_input("filepath_in")
        self.add_output("mesh_out")
        self.set_color(100, 160, 100)

    def run(self, **kwargs):
        self.clear_error()
        incoming = resolve_any_input(self.get_input("filepath_in"))
        if isinstance(incoming, dict):
            incoming = incoming.get("file") or incoming.get("path")
        filepath = _resolve_data_path(
            str(incoming or self.get_property("filepath") or ""),
            getattr(self, "_project_dir", None),
        )
        if not filepath or not os.path.isfile(filepath):
            self.set_error("No valid file path")
            return None
        try:
            from pylcss.io_manager.cad_io import CADImporter
            result = CADImporter.import_file(filepath)
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
            # Resolve inputs
            x_val = resolve_any_input(self.input(0))
            y_val = resolve_any_input(self.input(1))
            z_val = resolve_any_input(self.input(2))

            expression = self.get_property("expression") or "0"

            # Only arithmetic and calls to the explicit function
            # list are accepted.  Exposing numpy/math modules to eval allowed
            # arbitrary attribute traversal from a project file.
            ns = {
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
                "abs": np.abs, "min": np.minimum, "max": np.maximum,
                "floor": np.floor, "ceil": np.ceil,
                "pi": np.pi, "e": np.e,
            }
            if x_val is not None:
                ns["x"] = float(x_val) if not isinstance(x_val, (list, np.ndarray)) else x_val
            if y_val is not None:
                ns["y"] = float(y_val) if not isinstance(y_val, (list, np.ndarray)) else y_val
            if z_val is not None:
                ns["z"] = float(z_val) if not isinstance(z_val, (list, np.ndarray)) else z_val

            tree = ast.parse(str(expression), mode="eval")
            _validate_math_expression(tree, set(ns))
            result = eval(compile(tree, "<math-expression>", "eval"), {"__builtins__": {}}, ns)
            numeric = np.asarray(result, dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError("Expression result must be finite.")

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

            return area
        except Exception as e:
            self.set_error(str(e))
            return None


_ALLOWED_MATH_AST = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
    ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.UAdd, ast.USub,
)


def _validate_math_expression(tree: ast.AST, allowed_names: set[str]) -> None:
    """Reject attributes, indexing, comprehensions, and unknown identifiers."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_MATH_AST):
            raise ValueError(f"Unsupported expression feature: {type(node).__name__}.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown or unconnected expression variable: {node.id}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                raise ValueError("Only the listed math functions may be called.")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in math expressions.")
