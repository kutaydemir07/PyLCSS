# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Independent triangle screening of the actual trimmed CAD boundary.

Kernel validity and BOP checks can miss two regular branches of the same
B-spline face crossing. This is a finite tessellation check, not a proof of
global spline injectivity. It never heals, caps, or changes the CAD body.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .cad_tessellation import tessellate_cad_absolute
from .mesh_screening import screen_mesh_intersections


def screen_cad_boundary_intersections(shape, *, tolerance: float) -> dict:
    """Screen every triangle, rejecting missing tessellations and crossings."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('CAD boundary screening requires a positive absolute tolerance')
    vertices, triangles, ranges = tessellate_cad_absolute(
        shape, tolerance, .3, return_face_ranges=True)
    report = {
        'method': 'Actual trimmed CAD boundary; full non-adjacent triangle screening',
        'absolute_deflection': float(tolerance),
        'continuous_injectivity_certified': False,
        'geometry_changed': False,
        'passed': False,
        'missing_face_count': sum(end == start for _, _, start, end in ranges),
    }
    if not len(triangles) or report['missing_face_count']:
        report['reason'] = 'CAD boundary tessellation omitted one or more faces'
        return report
    # OCCT emits a separate vertex array for each face. Merge coincident seam
    # vertices before the adjacency test; leave all triangles in the screen.
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.merge_vertices()
    repeated = np.any(np.diff(np.sort(mesh.faces, axis=1), axis=1) == 0, axis=1)
    report.update(vertex_count=len(mesh.vertices), triangle_count=len(mesh.faces),
                  collapsed_triangle_count=int(repeated.sum()),
                  is_watertight=bool(mesh.is_watertight),
                  is_winding_consistent=bool(mesh.is_winding_consistent))
    if repeated.any():
        report['reason'] = 'CAD boundary contains collapsed tessellation triangles'
        return report
    screening = screen_mesh_intersections(mesh.vertices, mesh.faces, include_boundary_contacts=False)
    report.update(screening)
    report['passed'] = (screening['screening_scope'] == 'full'
                        and screening['self_intersecting_faces'] == 0)
    if not report['passed']:
        report['reason'] = 'The CAD boundary has intersecting triangles or could not be fully screened'
    return report
