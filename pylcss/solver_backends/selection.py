# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CAD geometry and coordinate-expression selection for solver decks."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.mesh import mesh_to_tet4


IntArray: TypeAlias = NDArray[np.int_]


def _as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [item for item in value.reshape(-1).tolist() if item is not None]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    return [value]


def normalize_geometries(value: Any) -> list[Any]:
    """Normalize node output aliases into CAD entities or mesh selections."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        if (
            value.get("mesh_selection")
            or value.get("node_ids") is not None
            or value.get("condition")
        ):
            return [value]
        for key in ("geometries", "entities", "faces"):
            if key in value and value[key] is not None:
                return _as_items(value[key])
        for key in ("geometry", "entity", "face"):
            if key in value and value[key] is not None:
                return [value[key]]
        return []
    return _as_items(value)


def dict_geometries(data: Mapping[str, Any]) -> list[Any]:
    """Extract every supported geometry/entity alias from a load or constraint."""
    for key in (
        "geometries",
        "entities",
        "faces",
        "geometry",
        "entity",
        "face",
    ):
        if key in data and data[key] is not None:
            return normalize_geometries(data[key])
    return []


_ALLOWED_CONDITION_FUNCS: dict[str, Any] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "isclose": np.isclose,
}
_ALLOWED_NP_ATTRS: dict[str, Any] = {
    **_ALLOWED_CONDITION_FUNCS,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_not": np.logical_not,
    "pi": np.pi,
}


class _SafeNumpy:
    """Read-only namespace for approved NumPy expression functions."""

    def __getattr__(self, name: str) -> Any:
        try:
            return _ALLOWED_NP_ATTRS[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _ConditionValidator(ast.NodeVisitor):
    """Validate the small expression language supported by saved CAD studies."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Attribute,
        ast.And,
        ast.Or,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    _ALLOWED_NAMES = {"x", "y", "z", "np", *_ALLOWED_CONDITION_FUNCS}

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(
                f"unsupported expression element: {node.__class__.__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self._ALLOWED_NAMES:
            raise ValueError(f"unsupported name: {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise ValueError("only np.<function> attributes are allowed")
        if node.attr not in _ALLOWED_NP_ATTRS:
            raise ValueError(f"unsupported numpy function: np.{node.attr}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_CONDITION_FUNCS:
                raise ValueError(f"unsupported function: {node.func.id!r}")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            raise ValueError("unsupported function call")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("condition expressions do not support **kwargs")
            self.visit(keyword.value)


def _mesh_points(mesh: Any, *, label: str) -> NDArray[np.float64]:
    if mesh is None or not hasattr(mesh, "p"):
        raise SolverBackendError(f"{label} expected a mesh with node coordinates.")
    points = np.asarray(mesh.p, dtype=float)
    if points.ndim != 2 or points.shape[0] != 3 or points.shape[1] == 0:
        raise SolverBackendError(
            f"{label} expected mesh.p with shape (3, N); got {points.shape!r}."
        )
    if not np.all(np.isfinite(points)):
        raise SolverBackendError(f"{label} mesh contains non-finite coordinates.")
    return points


def nodes_matching_condition(
    mesh: Any,
    condition: str,
    warnings: list[str] | None = None,
    label: str = "condition",
) -> IntArray:
    """Return zero-based mesh nodes selected by an ``x/y/z`` expression."""
    expression = str(condition or "").strip()
    if not expression:
        return np.array([], dtype=int)

    points = _mesh_points(mesh, label="Condition-based selection")
    x, y, z = points
    try:
        expression_ast = ast.parse(expression, mode="eval")
        _ConditionValidator().visit(expression_ast)
        raw_mask = eval(  # noqa: S307 - AST and namespace are strictly whitelisted.
            compile(expression_ast, "<pylcss-condition>", "eval"),
            {"__builtins__": {}},
            {
                "x": x,
                "y": y,
                "z": z,
                "np": _SafeNumpy(),
                **_ALLOWED_CONDITION_FUNCS,
            },
        )
    except Exception as exc:
        raise SolverBackendError(
            f"{label} expression {expression!r} could not be evaluated: {exc}"
        ) from exc

    mask = np.asarray(raw_mask, dtype=bool)
    if mask.shape == ():
        mask = np.full(x.shape, bool(mask), dtype=bool)
    if mask.shape != x.shape:
        raise SolverBackendError(
            f"{label} expression {expression!r} returned shape {mask.shape}, "
            f"expected {x.shape}."
        )

    node_ids = np.flatnonzero(mask).astype(int)
    if warnings is not None and node_ids.size == 0:
        warnings.append(f"{label} expression {expression!r} matched no mesh nodes.")
    return node_ids


def nodes_matching_geometries(
    mesh: Any,
    geometries: Sequence[Any],
    tolerance: float = 1.5,
) -> IntArray:
    """Return zero-based nodes close to CAD entities or stored selections."""
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            "Geometry-selection tolerance must be numeric."
        ) from exc
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise SolverBackendError(
            "Geometry-selection tolerance must be finite and non-negative."
        )

    entities = [geometry for geometry in geometries if geometry is not None]
    if not entities:
        return np.array([], dtype=int)
    points = _mesh_points(mesh, label="Geometry-based selection")

    selected: list[int] = []
    cad_entities: list[Any] = []
    for entity in entities:
        if isinstance(entity, Mapping):
            if entity.get("node_ids") is not None:
                ids = np.asarray(entity["node_ids"], dtype=int).reshape(-1)
                ids = ids[(ids >= 0) & (ids < points.shape[1])]
                selected.extend(int(value) for value in ids)
                continue
            condition = str(entity.get("condition") or "").strip()
            if condition:
                selected.extend(
                    int(value) for value in nodes_matching_condition(mesh, condition)
                )
                continue
            nested = dict_geometries(entity)
            if nested:
                cad_entities.extend(nested)
                continue
        cad_entities.append(entity)

    if not cad_entities:
        return np.asarray(sorted(set(selected)), dtype=int)

    vector_type: Any = None
    try:
        from cadquery import Vector

        vector_type = Vector
    except ImportError:  # pragma: no cover - CadQuery is an app dependency.
        pass

    for entity in cad_entities:
        candidates: Sequence[int] = range(points.shape[1])
        try:
            bounds = entity.BoundingBox()
            within_bounds = (
                (points[0] >= bounds.xmin - tolerance_value)
                & (points[0] <= bounds.xmax + tolerance_value)
                & (points[1] >= bounds.ymin - tolerance_value)
                & (points[1] <= bounds.ymax + tolerance_value)
                & (points[2] >= bounds.zmin - tolerance_value)
                & (points[2] <= bounds.zmax + tolerance_value)
            )
            candidates = np.flatnonzero(within_bounds).tolist()
        except Exception:
            bounds = None

        for node_index in candidates:
            xyz = (
                float(points[0, node_index]),
                float(points[1, node_index]),
                float(points[2, node_index]),
            )
            matched = False
            if vector_type is not None:
                try:
                    matched = (
                        float(entity.distanceTo(vector_type(*xyz))) <= tolerance_value
                    )
                except Exception:
                    matched = False
            if not matched and bounds is not None:
                matched = (
                    bounds.xmin - tolerance_value
                    <= xyz[0]
                    <= bounds.xmax + tolerance_value
                    and bounds.ymin - tolerance_value
                    <= xyz[1]
                    <= bounds.ymax + tolerance_value
                    and bounds.zmin - tolerance_value
                    <= xyz[2]
                    <= bounds.zmax + tolerance_value
                )
            if matched:
                selected.append(int(node_index))

    return np.asarray(sorted(set(selected)), dtype=int)


def tet_face_sets_for_geometries(
    mesh: Any,
    geometries: Sequence[Any],
    tolerance: float = 1.5,
) -> list[tuple[int, int]]:
    """Return 1-based CalculiX element/face ids on selected CAD boundaries."""
    if mesh is None or not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        return []
    try:
        points, cells = mesh_to_tet4(mesh, [])
    except SolverBackendError:
        return []

    # CalculiX C3D4 S1..S4 local face-node positions.
    local_faces = np.asarray(
        ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)),
        dtype=int,
    )
    owners: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for element_index, cell in enumerate(cells):
        for face_index, local_nodes in enumerate(local_faces):
            sorted_nodes = sorted(int(value) for value in cell[local_nodes])
            key = (sorted_nodes[0], sorted_nodes[1], sorted_nodes[2])
            owners.setdefault(key, []).append((element_index, face_index))

    selected_nodes = set(
        int(value)
        for value in nodes_matching_geometries(
            mesh,
            geometries,
            tolerance=tolerance,
        )
    )
    result: list[tuple[int, int]] = []
    for node_key, face_owners in owners.items():
        if len(face_owners) != 1 or not set(node_key).issubset(selected_nodes):
            continue
        element_index, face_index = face_owners[0]
        result.append((element_index + 1, face_index + 1))
    return result
