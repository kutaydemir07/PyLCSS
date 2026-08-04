# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Extract normalized meshes and nodal fields from PyLCSS CAD results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CadGeometry:
    """Mesh, nodal fields, and scalar results for one CAD evaluation."""

    points: np.ndarray
    cells: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, object] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.cells = np.asarray(self.cells, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not len(self.points):
            raise ValueError(
                f"points must have shape (n_nodes, 3); received {self.points.shape}."
            )
        if not np.isfinite(self.points).all():
            raise ValueError("points contains NaN or infinite coordinates.")
        if self.cells.ndim != 2 or self.cells.shape[1] < 3 or not len(self.cells):
            raise ValueError(
                "cells must have shape (n_cells, at_least_3_corners); "
                f"received {self.cells.shape}."
            )
        if np.any(self.cells < 0) or np.any(self.cells >= len(self.points)):
            raise ValueError("cells contains node indices outside the points array.")
        self.fields = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in self.fields.items()
        }

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(minimum, maximum)``."""
        return np.min(self.points, axis=0), np.max(self.points, axis=0)


def cad_evaluate_geometry(
    cad_path: str,
    kind: str,
    params: Mapping[str, float],
    field_name: str | None = None,
) -> CadGeometry:
    """Run a CAD graph and normalize its mesh and optional nodal field."""
    from pylcss.design_studio import runtime as cad_runtime

    runners = {
        "fea": cad_runtime.fea,
        "impact": cad_runtime.impact,
        "crash": cad_runtime.crash,
        "topopt": cad_runtime.topopt,
    }
    try:
        runner = runners[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver kind {kind!r}; expected fea, impact, or topopt."
        ) from exc

    result = runner(cad_path, **params)
    raw = result.raw()
    standard = result.standard()
    if not isinstance(raw, Mapping) or not isinstance(standard, Mapping):
        raise TypeError("CAD result raw() and standard() must return mappings.")

    points = _coerce_points(raw)
    cells = _coerce_cells(raw)
    if points is None or cells is None:
        raise RuntimeError(
            "CAD result has no usable mesh data. "
            f"Raw keys: {sorted(str(key) for key in raw)}"
        )
    fields = _extract_nodal_fields(raw, points.shape[0], only=field_name)
    return CadGeometry(
        points=points,
        cells=cells,
        fields=fields,
        scalars={str(name): value for name, value in standard.items()},
        params={str(name): float(value) for name, value in params.items()},
    )


def _coerce_points(raw: Mapping[str, Any]) -> np.ndarray | None:
    """Pull node coordinates from skfem or common array keys."""
    mesh = raw.get("mesh")
    if mesh is not None and hasattr(mesh, "p"):
        points = np.asarray(mesh.p, dtype=np.float64)
        if points.ndim == 2:
            if points.shape[0] == 3:
                return points.T
            if points.shape[1] == 3:
                return points
            if points.shape[0] == 2:
                return np.column_stack([points.T, np.zeros(points.shape[1])])

    for key in ("points", "nodes", "node_coords", "vertices", "coords"):
        value = raw.get(key)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 3:
            return points
        if points.ndim == 2 and points.shape[1] == 2:
            return np.column_stack([points, np.zeros(points.shape[0])])
    return None


def _coerce_cells(raw: Mapping[str, Any]) -> np.ndarray | None:
    """Pull element connectivity from skfem or common array keys."""
    mesh = raw.get("mesh")
    if mesh is not None and hasattr(mesh, "t"):
        cells = np.asarray(mesh.t, dtype=np.int64)
        if cells.ndim == 2:
            if cells.shape[0] in (3, 4, 6, 8, 10):
                return cells.T
            if cells.shape[1] in (3, 4, 6, 8, 10):
                return cells
            return cells.T

    for key in (
        "cells",
        "elements",
        "tris",
        "triangles",
        "tets",
        "tetrahedra",
        "faces",
    ):
        value = raw.get(key)
        if value is None:
            continue
        cells = np.asarray(value, dtype=np.int64)
        if cells.ndim == 2 and cells.shape[1] >= 3:
            return cells
    return None


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "von_mises": (
        "von_mises",
        "vonmises",
        "stress_vm",
        "node_stress",
        "nodal_stress",
        "stress",
    ),
    "stress": (
        "stress",
        "von_mises",
        "vonmises",
        "stress_vm",
        "node_stress",
        "nodal_stress",
    ),
    "displacement": ("displacement", "node_disp", "u"),
    "energy": ("ener_nodal", "energy", "node_energy"),
}

_FIELD_CANDIDATES = (
    "von_mises",
    "vonmises",
    "stress_vm",
    "node_stress",
    "nodal_stress",
    "stress",
    "displacement",
    "node_disp",
    "u",
    "u_x",
    "u_y",
    "u_z",
    "density",
    "strain",
    "temperature",
    "ener_nodal",
    "energy",
)


def _extract_nodal_fields(
    raw: Mapping[str, Any],
    n_nodes: int,
    only: str | None = None,
) -> dict[str, np.ndarray]:
    """Extract arrays whose first logical dimension matches the mesh nodes."""
    if only is not None:
        for key in _FIELD_ALIASES.get(only, (only,)):
            value = raw.get(key)
            if value is None:
                continue
            field = _normalize_nodal_array(np.asarray(value, dtype=np.float64), n_nodes)
            if field is not None:
                return {only: field}
        return {}

    fields: dict[str, np.ndarray] = {}
    for key in _FIELD_CANDIDATES:
        value = raw.get(key)
        if value is None:
            continue
        field = _normalize_nodal_array(np.asarray(value, dtype=np.float64), n_nodes)
        if field is not None:
            fields[key] = field
    return fields


def _normalize_nodal_array(
    values: np.ndarray,
    n_nodes: int,
) -> np.ndarray | None:
    if values.ndim == 1:
        if values.shape[0] == n_nodes:
            return values.reshape(-1, 1)
        if values.shape[0] == 3 * n_nodes:
            return values.reshape(n_nodes, 3)
        return None
    if values.ndim == 2:
        if values.shape[0] == n_nodes:
            return values
        if values.shape[1] == n_nodes and values.shape[0] in (1, 3, 6, 9):
            return values.T
    return None


__all__ = ["CadGeometry", "cad_evaluate_geometry"]
