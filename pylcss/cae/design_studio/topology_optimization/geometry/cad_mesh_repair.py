# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Repair and validate recovered triangle meshes before STEP conversion."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _extract_supplied_quad_mesh(payload):
    """Read explicit quad/layout inputs for the paper's intermediate entry points."""
    if payload.get('amrto_layouts_dir'):
        from .amrto_layout_partition import quad_mesh_from_layout_folder
        vertices, quads = quad_mesh_from_layout_folder(payload['amrto_layouts_dir'])
    elif isinstance(payload.get('amrto_quad_mesh'), dict):
        supplied = payload['amrto_quad_mesh']
        vertices, quads = supplied['vertices'], supplied['quads']
    else:
        return None
    vertices, quads = np.asarray(vertices, dtype=float), np.asarray(quads, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or quads.ndim != 2 or quads.shape[1] != 4 or quads.size == 0
            or quads.min() < 0 or quads.max() >= len(vertices)):
        raise ValueError('Invalid supplied quad mesh')
    return vertices, quads

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


def _extract_amrto_design_mesh(payload: Any, *, validate_geometry: bool = True) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read the certified recovery's fitting surface, keeping public STL intact.

    CAD assembly must still restore the original analytic regions. Fingerprints
    bind the auxiliary surface to the delivered recovery so stale cached arrays
    cannot silently supply a different topology result.
    """
    recovered = _extract_recovered_mesh(payload)
    record = payload.get('recovered_shape') if isinstance(payload, dict) else None
    design = record.get('amrto_design_surface') if isinstance(record, dict) else None
    if design is None:
        return recovered
    from .amrto_surface import geometry_fingerprint
    from .preserved_surface import _closed_mesh
    from .mesh_screening import screen_mesh_intersections

    certificate = record.get('preserved_geometry', {}).get('surface_restoration', {})
    if (recovered is None or not isinstance(design, dict)
            or certificate.get('certified') is not True or certificate.get('applied') is not True
            or design.get('requires_exact_cad_restoration') is not True
            or design.get('restored_geometry_sha256') != geometry_fingerprint(*recovered)):
        raise ValueError('AMRTO design surface is not bound to the certified recovered geometry')
    selected = _extract_recovered_mesh({'recovered_shape': design})
    if selected is None or design.get('geometry_sha256') != geometry_fingerprint(*selected):
        raise ValueError('AMRTO design surface changed after recovery')
    # UI cache lookup checks identities only. The background build validates
    # closure and all intersections before invoking either native stage.
    if not validate_geometry:
        return selected
    mesh = _closed_mesh(*selected, 'AMRTO design surface')
    screening = screen_mesh_intersections(mesh.vertices, mesh.faces, include_boundary_contacts=False)
    if screening.get('screening_scope') != 'full' or screening.get('self_intersecting_faces') != 0:
        raise ValueError('AMRTO design surface has unresolved triangle intersections')
    return mesh.vertices, mesh.faces


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
