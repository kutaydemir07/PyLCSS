# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Small, dependency-free finite-element mesh quality checks.

The reported ``mean_ratio`` is dimensionless and equals one for an ideal
equilateral triangle/tetrahedron and approaches zero for a collapsed element.
It is intended as an immediate pre-solver diagnostic, not as a replacement for
an element formulation's Jacobian checks or a mesh-convergence study.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import ArrayLike


class MeshLike(Protocol):
    """Structural mesh interface used by quality checks and exporters."""

    p: ArrayLike
    t: ArrayLike


class MeshQualityReport(TypedDict):
    """Serializable mesh-quality result shared by meshing workflows."""

    element_type: str
    node_count: int
    element_count: int
    valid_element_count: int
    degenerate_element_count: int
    min_mean_ratio: float
    p05_mean_ratio: float
    mean_mean_ratio: float
    max_edge_ratio: float
    solver_ready: bool
    assessment: str


_TET_EDGES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    dtype=int,
)
_TRI_EDGES = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=int)


def _empty_report(kind: str, nodes: int, elements: int) -> MeshQualityReport:
    return {
        "element_type": kind,
        "node_count": int(nodes),
        "element_count": int(elements),
        "valid_element_count": 0,
        "degenerate_element_count": int(elements),
        "min_mean_ratio": 0.0,
        "p05_mean_ratio": 0.0,
        "mean_mean_ratio": 0.0,
        "max_edge_ratio": float("inf"),
        "solver_ready": False,
        "assessment": "No valid elements",
    }


def assess_mesh_quality(mesh: MeshLike) -> MeshQualityReport:
    """Return robust first-order triangle or tetrahedron quality statistics."""
    points = np.asarray(getattr(mesh, "p", None), dtype=float)
    connectivity = np.asarray(getattr(mesh, "t", None), dtype=int)
    if points.ndim != 2 or connectivity.ndim != 2:
        raise ValueError("Mesh quality requires two-dimensional p and t arrays.")
    if points.shape[0] < 3:
        raise ValueError("Mesh quality requires three-dimensional node coordinates.")

    nodes = int(points.shape[1])
    elements = int(connectivity.shape[1])
    if connectivity.shape[0] >= 4:
        kind = "tetrahedron"
        width = 4
        edges = _TET_EDGES
    elif connectivity.shape[0] == 3:
        kind = "triangle"
        width = 3
        edges = _TRI_EDGES
    else:
        raise ValueError("Only triangle and tetrahedron meshes are supported.")
    if nodes < width or elements == 0:
        return _empty_report(kind, nodes, elements)
    if np.min(connectivity[:width]) < 0 or np.max(connectivity[:width]) >= nodes:
        raise ValueError("Mesh connectivity references a node outside the point array.")

    xyz = points[:3, connectivity[:width]].transpose(2, 1, 0)
    # xyz has shape (n_elements, element_width, 3).
    edge_vectors = xyz[:, edges[:, 1], :] - xyz[:, edges[:, 0], :]
    edge_sq = np.einsum("eij,eij->ei", edge_vectors, edge_vectors)
    max_edge = np.sqrt(np.max(edge_sq, axis=1))
    min_edge = np.sqrt(np.maximum(np.min(edge_sq, axis=1), 0.0))
    edge_ratio = np.divide(
        max_edge,
        min_edge,
        out=np.full_like(max_edge, np.inf),
        where=min_edge > 0.0,
    )

    span = np.ptp(points[:3], axis=1)
    length_scale = max(float(np.linalg.norm(span)), 1.0)
    if kind == "tetrahedron":
        triple = np.einsum(
            "ei,ei->e",
            xyz[:, 1, :] - xyz[:, 0, :],
            np.cross(
                xyz[:, 2, :] - xyz[:, 0, :],
                xyz[:, 3, :] - xyz[:, 0, :],
            ),
        )
        measure = np.abs(triple) / 6.0
        tolerance = np.finfo(float).eps * length_scale**3 * 100.0
        mean_ratio = 12.0 * np.power(3.0 * measure, 2.0 / 3.0)
        mean_ratio = np.divide(
            mean_ratio,
            np.sum(edge_sq, axis=1),
            out=np.zeros_like(mean_ratio),
            where=np.sum(edge_sq, axis=1) > 0.0,
        )
    else:
        cross = np.cross(
            xyz[:, 1, :] - xyz[:, 0, :],
            xyz[:, 2, :] - xyz[:, 0, :],
        )
        measure = 0.5 * np.linalg.norm(cross, axis=1)
        tolerance = np.finfo(float).eps * length_scale**2 * 100.0
        mean_ratio = 4.0 * np.sqrt(3.0) * measure
        mean_ratio = np.divide(
            mean_ratio,
            np.sum(edge_sq, axis=1),
            out=np.zeros_like(mean_ratio),
            where=np.sum(edge_sq, axis=1) > 0.0,
        )

    valid = (
        np.isfinite(measure)
        & np.isfinite(mean_ratio)
        & np.isfinite(edge_ratio)
        & (measure > tolerance)
    )
    valid_quality = np.clip(mean_ratio[valid], 0.0, 1.0)
    degenerate = int(elements - np.count_nonzero(valid))
    if valid_quality.size == 0:
        return _empty_report(kind, nodes, elements)

    minimum = float(np.min(valid_quality))
    p05 = float(np.percentile(valid_quality, 5.0))
    mean = float(np.mean(valid_quality))
    maximum_edge_ratio = float(np.max(edge_ratio[valid]))
    solver_ready = degenerate == 0 and minimum > 1.0e-6
    if not solver_ready:
        assessment = "Rejected: collapsed or invalid elements"
    elif p05 < 0.05:
        assessment = "Poor: inspect and refine before production analysis"
    elif p05 < 0.15:
        assessment = "Marginal: perform a mesh-convergence study"
    else:
        assessment = "Acceptable initial mesh; convergence still required"
    return {
        "element_type": kind,
        "node_count": nodes,
        "element_count": elements,
        "valid_element_count": int(valid_quality.size),
        "degenerate_element_count": degenerate,
        "min_mean_ratio": minimum,
        "p05_mean_ratio": p05,
        "mean_mean_ratio": mean,
        "max_edge_ratio": maximum_edge_ratio,
        "solver_ready": bool(solver_ready),
        "assessment": assessment,
    }


def attach_mesh_quality(mesh: MeshLike) -> MeshQualityReport:
    """Assess ``mesh`` and attach the serialisable report for downstream UI."""
    report = assess_mesh_quality(mesh)
    try:
        mesh.quality_report = report
    except Exception:
        pass
    return report
