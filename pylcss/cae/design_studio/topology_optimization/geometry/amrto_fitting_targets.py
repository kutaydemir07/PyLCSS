# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Optional bounded harmonic transport of a separate fitting target mesh.

The public source layout is never mutated. Shared CAD curves can differ
slightly from the piecewise-linear source boundary; positive-weight graph
harmonic transport carries only that prescribed difference into a copy of
the fitting targets. This is an engineering extension, not a paper step.
"""
from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def harmonic_fitting_targets(layout: Any, *, maximum_displacement: float):
    """Return ``(fitting_layout_copy, report)`` or reject without mutation.

    UVs must already agree with the four shared boundary clocks. Uniform
    positive graph weights give each interior displacement as a convex
    combination of Dirichlet displacements. The resulting pointwise bound
    also holds on every affine source triangle. Every transported physical
    triangle must retain positive area and its original normal orientation.
    These local checks do not certify global mesh/spline injectivity.
    """
    if not np.isfinite(maximum_displacement) or maximum_displacement <= 0:
        raise ValueError('Fitting-target displacement budget must be finite and positive')
    vertices = np.asarray(layout.vertices, dtype=float)
    uv = np.asarray(layout.parameter_coords, dtype=float)
    triangles = np.asarray(layout.triangles)
    paths = getattr(layout, 'boundary_paths', None)
    curves = getattr(layout, 'boundary_splines', None)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or uv.shape != (len(vertices), 2) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv > 1) or triangles.ndim != 2
            or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0
            or triangles.max() >= len(vertices) or paths is None or len(paths) != 4
            or curves is None or len(curves) != 4):
        raise ValueError('Harmonic targets require finite indexed source geometry, UVs and four shared curves')
    source_triangles = vertices[triangles]
    original_normals = np.cross(source_triangles[:, 1] - source_triangles[:, 0],
                                source_triangles[:, 2] - source_triangles[:, 0])
    original_areas = np.linalg.norm(original_normals, axis=1)
    if np.any(original_areas <= 0):
        raise ValueError('Harmonic targets require nondegenerate physical source triangles')
    prescribed, disagreements = {}, []
    for side, path in enumerate(paths):
        indices = np.asarray(path)
        if (indices.ndim != 1 or len(indices) < 2 or not np.issubdtype(indices.dtype, np.integer)
                or indices.min() < 0 or indices.max() >= len(vertices)):
            raise ValueError('Fitting-target boundary paths must contain valid indexed vertices')
        parameters = uv[indices, 0 if side % 2 == 0 else 1]
        if side >= 2:
            parameters = 1 - parameters
        targets = np.asarray(curves[side](parameters), dtype=float)
        if targets.shape != (len(indices), 3) or not np.isfinite(targets).all():
            raise ValueError('Shared boundary curve returned invalid fitting targets')
        changes = targets - vertices[indices]
        for index, change in zip(indices, changes):
            if int(index) in prescribed:
                disagreements.append(float(np.linalg.norm(change - prescribed[int(index)])))
            else:
                prescribed[int(index)] = change
    corner_tolerance = max(1e-10, float(maximum_displacement) * 1e-8)
    if max(disagreements, default=0.0) > corner_tolerance:
        raise ValueError('Shared curves prescribe incompatible fitting-target corner displacements')
    edges = np.unique(np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                         triangles[:, [2, 0]])), axis=1), axis=0)
    count = len(vertices)
    graph = coo_matrix((np.ones(2 * len(edges)),
                        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(count, count)).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    # Unused source storage is not a physical graph component; keep it fixed.
    for index in np.flatnonzero(degree == 0):
        prescribed.setdefault(int(index), np.zeros(3))
    boundary = np.asarray(sorted(prescribed), dtype=int)
    interior = np.setdiff1d(np.arange(count), boundary)
    _, labels = connected_components(graph, directed=False)
    if not set(labels[interior]).issubset(set(labels[boundary])):
        raise ValueError('A physical source component has no Dirichlet fitting-target boundary')
    laplacian = diags(degree) - graph
    displacement = np.zeros_like(vertices)
    displacement[boundary] = [prescribed[int(index)] for index in boundary]
    boundary_maximum = float(np.linalg.norm(displacement[boundary], axis=1).max(initial=0.0))
    arithmetic_tolerance = max(1e-10, float(maximum_displacement) * 1e-8)
    if boundary_maximum > maximum_displacement + arithmetic_tolerance:
        raise ValueError('Shared boundary displacement exceeds the fitting-target fidelity budget')
    if len(interior):
        displacement[interior] = spsolve(laplacian[interior][:, interior].tocsc(),
                                          -laplacian[interior][:, boundary] @ displacement[boundary])
    maximum = float(np.linalg.norm(displacement, axis=1).max(initial=0.0))
    if not np.isfinite(displacement).all() or maximum > boundary_maximum + arithmetic_tolerance:
        raise ValueError('Positive-weight harmonic fitting targets violated the maximum principle')
    residual = laplacian[interior] @ displacement if len(interior) else np.zeros((0, 3))
    residual_maximum = float(np.linalg.norm(residual, axis=1).max(initial=0.0))
    if residual_maximum > arithmetic_tolerance:
        raise ValueError('Harmonic fitting-target solve did not meet its independent equation residual')
    transported = vertices + displacement
    physical = transported[triangles]
    normals = np.cross(physical[:, 1] - physical[:, 0], physical[:, 2] - physical[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    cosine = np.divide(np.einsum('ij,ij->i', normals, original_normals), areas * original_areas,
                       out=np.full(len(triangles), -np.inf), where=areas > 0)
    ratio = areas / original_areas
    if np.any(areas <= 0) or np.any(cosine <= 0) or not np.isfinite(cosine).all():
        raise ValueError('Harmonic fitting targets reversed or degenerated a physical source triangle')
    working = copy(layout)
    working.vertices = transported
    working.parameter_coords = uv.copy()
    working.boundary_splines = list(curves)
    working.nurbs_surface = None
    report = {'method': 'Positive uniform-edge graph harmonic transport of separate fitting targets',
              'accepted': True, 'engineering_extension': True, 'paper_algorithm': False,
              'layout_id': int(layout.layout_id), 'public_source_XYZ_unchanged': True,
              'fitting_target_XYZ_changed': bool(maximum > 0), 'source_triangles_unchanged': True,
              'source_UV_unchanged': True, 'positive_graph_edge_weights': True,
              'weight_policy': 'One for each unique undirected physical triangle edge',
              'maximum_displacement_budget': float(maximum_displacement),
              'maximum_boundary_displacement': boundary_maximum, 'maximum_target_displacement': maximum,
              'maximum_principle_verified': True, 'maximum_laplacian_residual': residual_maximum,
              'maximum_corner_disagreement': max(disagreements, default=0.0),
              'boundary_vertex_count': len(boundary), 'interior_vertex_count': len(interior),
              'source_triangle_count': len(triangles), 'reversed_source_triangle_count': 0,
              'minimum_source_triangle_normal_cosine': float(cosine.min()),
              'minimum_source_triangle_area_ratio': float(ratio.min()),
              'pointwise_displacement_bound_on_affine_source_triangles': maximum,
              'global_injectivity_certified': False, 'surface_fits': 0}
    return working, report
