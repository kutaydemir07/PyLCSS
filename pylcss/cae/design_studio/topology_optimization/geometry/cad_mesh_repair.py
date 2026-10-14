# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Repair and validate recovered triangle meshes before STEP conversion."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

def _extract_recovered_mesh(payload: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(vertices, triangle_faces)`` from a topology result payload."""
    if not isinstance(payload, dict):
        return None

    mesh = payload.get("recovered_shape")
    if mesh is None and "vertices" in payload and "faces" in payload:
        mesh = payload
    if not isinstance(mesh, dict):
        return None

    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    if vertices is None or faces is None:
        return None

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return None
    if faces.ndim != 2 or faces.shape[1] < 3:
        return None
    if len(vertices) < 4 or len(faces) < 4:
        return None
    return vertices[:, :3], faces[:, :3]


def _bbox_diag(vertices: np.ndarray) -> float:
    """Return the model bounding-box diagonal, clamped to a non-zero value."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    lo = np.nanmin(vertices[:, :3], axis=0)
    hi = np.nanmax(vertices[:, :3], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return diag if np.isfinite(diag) and diag > 1e-12 else 1.0


def _effective_tolerance(
    vertices: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Combine user absolute tolerance with model-size-relative tolerance."""
    abs_tol = float(absolute_tolerance or 0.0)
    rel_tol = float(relative_tolerance or 0.0) * _bbox_diag(vertices)
    return max(abs_tol, rel_tol, 1e-9)
