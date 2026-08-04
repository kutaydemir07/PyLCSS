# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Mesh validation and connectivity conversion for external solver decks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pylcss.solver_backends.base import SolverBackendError


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]


def _points(mesh: Any, *, label: str) -> FloatArray:
    if mesh is None or not hasattr(mesh, "p"):
        raise SolverBackendError(f"{label} expected a mesh with node coordinates.")
    try:
        points = np.asarray(mesh.p, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(f"{label} node coordinates are not numeric.") from exc
    if points.ndim != 2 or points.shape[0] != 3 or points.shape[1] == 0:
        raise SolverBackendError(
            f"{label} mesh.p must have shape (3, N) with N > 0; got {points.shape!r}."
        )
    if not np.all(np.isfinite(points)):
        raise SolverBackendError(f"{label} mesh contains non-finite node coordinates.")
    return points.T


def _connectivity(mesh: Any, *, rows: int, label: str) -> IntArray:
    if mesh is None or not hasattr(mesh, "t"):
        raise SolverBackendError(f"{label} expected a mesh with element connectivity.")
    try:
        numeric = np.asarray(mesh.t, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            f"{label} element connectivity is not integral."
        ) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise SolverBackendError(
            f"{label} element connectivity must contain finite integer indices."
        )
    connectivity = numeric.astype(int)
    if (
        connectivity.ndim != 2
        or connectivity.shape[0] < rows
        or connectivity.shape[1] == 0
    ):
        raise SolverBackendError(
            f"{label} mesh.t must have at least {rows} rows and one element; "
            f"got {connectivity.shape!r}."
        )
    return connectivity


def _validate_indices(
    cells: IntArray,
    *,
    node_count: int,
    label: str,
) -> None:
    if np.any(cells < 0) or np.any(cells >= node_count):
        raise SolverBackendError(
            f"{label} connectivity contains a node index outside [0, {node_count - 1}]."
        )


def _volume_tolerance(points: FloatArray) -> float:
    """Return a scale-aware tolerance for six times tetrahedron volume."""
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    scale = max(diagonal, float(np.finfo(float).tiny))
    return float(128.0 * np.finfo(float).eps * scale**3)


def mesh_to_tet4(
    mesh: Any,
    warnings: list[str],
) -> tuple[FloatArray, IntArray]:
    """Return validated, positively oriented four-node tetrahedra."""
    points = _points(mesh, label="Tetrahedral deck writer")
    connectivity = _connectivity(
        mesh,
        rows=4,
        label="Tetrahedral deck writer",
    )
    cells = connectivity[:4].T.copy()
    _validate_indices(
        cells,
        node_count=points.shape[0],
        label="Tetrahedral mesh",
    )
    if connectivity.shape[0] > 4:
        warnings.append(
            "Higher-order tetrahedra were downgraded to their four corner "
            "nodes for this external deck."
        )

    v0, v1, v2, v3 = (points[cells[:, index]] for index in range(4))
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(v1 - v0, v2 - v0),
        v3 - v0,
    )
    degenerate = np.abs(signed_six_volume) <= _volume_tolerance(points)
    if np.any(degenerate):
        raise SolverBackendError(
            f"Tetrahedral mesh contains {int(np.count_nonzero(degenerate))} "
            "degenerate element(s); repair or remesh before solving."
        )

    flipped = signed_six_volume < 0.0
    if np.any(flipped):
        cells[flipped, 0], cells[flipped, 1] = (
            cells[flipped, 1],
            cells[flipped, 0].copy(),
        )
    return points, cells


def tet10_connectivity(mesh: Any) -> IntArray | None:
    """Return ``(10, N)`` quadratic tetrahedron connectivity when available."""
    if mesh is None or not hasattr(mesh, "t"):
        return None
    direct = np.asarray(mesh.t, dtype=int)
    if direct.ndim == 2 and direct.shape[0] == 10:
        return direct
    try:
        element_dofs = np.asarray(mesh.dofs.element_dofs, dtype=int)
    except (AttributeError, TypeError, ValueError):
        return None
    if element_dofs.ndim == 2 and element_dofs.shape[0] == 10:
        return element_dofs
    return None


def mesh_to_tet10(
    mesh: Any,
    warnings: list[str],
) -> tuple[FloatArray, IntArray]:
    """Return validated, positively oriented ten-node C3D10 tetrahedra."""
    del warnings  # Kept for API symmetry with the linear converter.
    points = _points(mesh, label="C3D10 deck writer")
    connectivity = tet10_connectivity(mesh)
    if connectivity is None or connectivity.shape[1] == 0:
        raise SolverBackendError(
            "C3D10 deck writer expected ten-node connectivity in mesh.t or "
            "mesh.dofs.element_dofs."
        )
    cells = connectivity.T.copy()
    _validate_indices(cells, node_count=points.shape[0], label="C3D10 mesh")

    v0, v1, v2, v3 = (points[cells[:, index]] for index in range(4))
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(v1 - v0, v2 - v0),
        v3 - v0,
    )
    degenerate = np.abs(signed_six_volume) <= _volume_tolerance(points)
    if np.any(degenerate):
        raise SolverBackendError(
            f"C3D10 mesh contains {int(np.count_nonzero(degenerate))} "
            "degenerate element(s); repair or remesh before solving."
        )

    flipped = signed_six_volume < 0.0
    if np.any(flipped):
        cells[flipped, 0], cells[flipped, 1] = (
            cells[flipped, 1],
            cells[flipped, 0].copy(),
        )
        cells[flipped, 5], cells[flipped, 6] = (
            cells[flipped, 6],
            cells[flipped, 5].copy(),
        )
        cells[flipped, 7], cells[flipped, 8] = (
            cells[flipped, 8],
            cells[flipped, 7].copy(),
        )
    return points, cells


def is_shell_mesh(mesh: Any) -> bool:
    """Return whether a mesh exposes triangle surface connectivity."""
    if mesh is None or not hasattr(mesh, "t"):
        return False
    connectivity = np.asarray(mesh.t)
    return connectivity.ndim == 2 and connectivity.shape[0] == 3


def mesh_to_shell(
    mesh: Any,
    warnings: list[str],
) -> tuple[FloatArray, IntArray]:
    """Return validated three-node shell triangles, dropping zero-area cells."""
    points = _points(mesh, label="Shell deck writer")
    connectivity = _connectivity(mesh, rows=3, label="Shell deck writer")
    if connectivity.shape[0] != 3:
        raise SolverBackendError(
            f"Shell mesh.t must have shape (3, N); got {connectivity.shape!r}."
        )
    triangles = connectivity.T.copy()
    _validate_indices(
        triangles,
        node_count=points.shape[0],
        label="Shell mesh",
    )

    v0 = points[triangles[:, 0]]
    twice_area = np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - v0,
            points[triangles[:, 2]] - v0,
        ),
        axis=1,
    )
    diagonal = max(
        float(np.linalg.norm(np.ptp(points, axis=0))),
        float(np.finfo(float).tiny),
    )
    area_tolerance = 128.0 * np.finfo(float).eps * diagonal**2
    keep = twice_area > area_tolerance
    dropped = int(np.count_nonzero(~keep))
    if dropped:
        warnings.append(
            f"Dropped {dropped}/{triangles.shape[0]} degenerate shell "
            "triangle(s) before writing the OpenRadioss deck."
        )
        triangles = triangles[keep]
    if triangles.shape[0] == 0:
        raise SolverBackendError("Shell mesh contains no non-degenerate elements.")
    return points, triangles


def id_lines(ids: Iterable[int], per_line: int = 12) -> list[str]:
    """Format solver ids into comma-separated deck lines."""
    if per_line <= 0:
        raise ValueError("per_line must be greater than zero")
    values = [int(value) for value in ids]
    return [
        ", ".join(str(value) for value in values[index : index + per_line])
        for index in range(0, len(values), per_line)
    ]


def load_vector(load: dict[str, Any]) -> FloatArray:
    """Return a validated finite three-vector from a load dictionary."""
    try:
        vector = np.asarray(
            load.get("vector", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError("Load vector must contain three numbers.") from exc
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise SolverBackendError(
            "Load vector must contain exactly three finite numbers."
        )
    return vector
