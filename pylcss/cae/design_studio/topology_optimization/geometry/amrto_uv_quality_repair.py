# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Measure varying chart distortion and optionally relax only interior UVs.

A long thin physical chart mapped onto a unit square has a large raw stretch
ratio even when it is perfectly affine. A frozen area-weighted affine metric
removes that constant scale/skew before measuring symmetric Dirichlet energy.
This engineering extension preserves source XYZs and all four boundary clocks.
It is not a substitute for source fitting or final native solid validation.
"""
from __future__ import annotations

from copy import copy

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _chart_data(layout, reference_affine_metric=None):
    vertices = np.asarray(layout.vertices, dtype=float)
    triangles = np.asarray(layout.triangles)
    uv = np.asarray(layout.parameter_coords, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or uv.shape != (len(vertices), 2) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv > 1) or triangles.ndim != 2
            or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0
            or triangles.max() >= len(vertices)):
        raise ValueError('UV quality requires finite indexed source triangles and a chart in [0, 1]^2')
    xyz, planar = vertices[triangles], uv[triangles]
    first, second = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    area2 = np.linalg.norm(np.cross(first, second), axis=1)
    lengths = np.linalg.norm(first, axis=1)
    if np.any(area2 <= 0) or np.any(lengths <= 0):
        raise ValueError('UV quality requires nondegenerate physical source triangles')
    edges = np.stack((planar[:, 1] - planar[:, 0], planar[:, 2] - planar[:, 0]), axis=2)
    determinants = np.linalg.det(edges)
    if np.any(np.abs(determinants) <= 128 * np.finfo(float).eps) or np.any(determinants * determinants[0] <= 0):
        raise ValueError('UV quality requires consistently oriented, noncollapsed source UV triangles')
    intrinsic = np.zeros((len(triangles), 2, 2))
    intrinsic[:, 0, 0] = lengths
    intrinsic[:, 0, 1] = np.einsum('ij,ij->i', first, second) / lengths
    intrinsic[:, 1, 1] = area2 / lengths
    inverse = np.linalg.inv(intrinsic)
    weights = area2 / area2.sum()
    if reference_affine_metric is None:
        jacobian = np.stack((first, second), axis=2) @ np.linalg.inv(edges)
        metric = np.einsum('t,tji,tjk->ik', weights, jacobian, jacobian)
    else:
        metric = np.asarray(reference_affine_metric, dtype=float)
    if metric.shape != (2, 2) or not np.isfinite(metric).all() or not np.allclose(metric, metric.T):
        raise ValueError('Reference UV affine metric must be a finite symmetric 2x2 matrix')
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.any(eigenvalues <= 0):
        raise ValueError('Reference UV affine metric must be positive definite')
    affine = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    paths = getattr(layout, 'boundary_paths', ())
    if len(paths) != 4 or any(np.asarray(path).ndim != 1 or len(path) < 2
                              or not np.issubdtype(np.asarray(path).dtype, np.integer) for path in paths):
        raise ValueError('UV quality requires four valid indexed boundary clock paths')
    boundary = np.unique(np.concatenate([np.asarray(path, dtype=int) for path in paths]))
    if boundary.min() < 0 or boundary.max() >= len(vertices):
        raise ValueError('UV quality boundary paths contain an invalid source index')
    return {'vertices': vertices, 'triangles': triangles, 'uv': uv, 'inverse': inverse,
            'weights': weights, 'affine': affine, 'metric': metric, 'determinants': determinants,
            'boundary': boundary, 'interior': np.setdiff1d(np.unique(triangles), boundary)}


def _quality(uv, data):
    points = uv[data['triangles']]
    edges = np.stack((points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=2)
    raw = np.linalg.svd(edges @ data['inverse'], compute_uv=False)
    normalized = np.linalg.svd(data['affine'] @ edges @ data['inverse'], compute_uv=False)
    ratios = normalized[:, 0] / normalized[:, 1]
    determinants = np.linalg.det(edges)
    relative_areas = np.abs(determinants) / np.abs(determinants).sum() / data['weights']
    return {'raw_physical_UV_aspect_maximum': float(np.max(raw[:, 0] / raw[:, 1])),
            'affine_normalized_aspect_maximum': float(ratios.max()),
            'affine_normalized_aspect_p95': float(np.percentile(ratios, 95)),
            'affine_normalized_symmetric_Dirichlet_energy': float(data['weights'] @
                np.sum(normalized ** 2 + normalized ** -2, axis=1)),
            'affine_normalized_symmetric_conformal_energy': float(data['weights'] @ (ratios + 1 / ratios)),
            'minimum_absolute_UV_triangle_determinant': float(np.abs(determinants).min()),
            'minimum_source_area_normalized_UV_ratio': float(relative_areas.min())}


def measure_affine_normalized_uv_quality(layout, *, reference_affine_metric=None):
    """Return diagnostics; use the returned metric unchanged for comparisons."""
    data = _chart_data(layout, reference_affine_metric)
    return {'layout_id': int(layout.layout_id), **_quality(data['uv'], data),
            'normalization': 'Frozen physical-area-weighted mean of J_transpose_J for source UV to XYZ maps',
            'reference_affine_metric': data['metric'].tolist(),
            'ideal_symmetric_Dirichlet_energy': 4., 'ideal_symmetric_conformal_energy': 2.,
            'source_vertex_count': len(data['vertices']), 'source_triangle_count': len(data['triangles']),
            'fixed_boundary_vertex_count': len(data['boundary']), 'interior_vertex_count': len(data['interior']),
            'constant_chart_aspect_and_skew_removed': True, 'source_geometry_and_UV_unchanged': True}


def _energy_gradient(values, uv, data):
    candidate = uv.copy()
    candidate[data['interior']] = values.reshape(-1, 2)
    points = candidate[data['triangles']]
    edges = np.stack((points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=2)
    determinants = np.linalg.det(edges)
    if (candidate.min() < 0 or candidate.max() > 1 or
            np.any(determinants * data['determinants'] <= 0)
            or np.any(np.abs(determinants) <= 128 * np.finfo(float).eps)):
        return np.inf, None
    relative = np.abs(determinants) / np.abs(determinants).sum() / data['weights']
    if relative.min() < data['minimum_relative_area'] * (1 - 1e-10):
        return np.inf, None
    jacobian = data['affine'] @ edges @ data['inverse']
    inverse = np.linalg.inv(jacobian)
    energy = float(data['weights'] @ np.sum(jacobian ** 2 + inverse ** 2, axis=(1, 2)))
    transposed = np.swapaxes(inverse, 1, 2)
    gradient = 2 * jacobian - 2 * transposed @ inverse @ transposed
    local = (data['affine'].T @ (data['weights'][:, None, None] * gradient)
             @ np.swapaxes(data['inverse'], 1, 2))
    assembled = np.zeros_like(uv)
    np.add.at(assembled, data['triangles'][:, 0], -local.sum(axis=2))
    np.add.at(assembled, data['triangles'][:, 1], local[:, :, 0])
    np.add.at(assembled, data['triangles'][:, 2], local[:, :, 1])
    return energy, assembled[data['interior']].ravel()


def repair_layout_uv_distortion(layout, *, maximum_iterations=32, maximum_backtracks=16,
                                 minimum_relative_improvement=1e-3,
                                 normalized_tail_threshold=3., energy_threshold=5.,
                                 maximum_source_vertices=20000, maximum_source_triangles=40000):
    """Try bounded fixed-boundary descent, publishing only strict improvements.

    Admission requires reduced energy AND p95 distortion, no worse maximum
    distortion, and positive noncollapsed source triangles. Every comparison
    uses the same frozen affine normalization. Failed proposals leave the
    original chart and its samples intact. Source fitting/native CAD checks
    remain the caller's responsibility after an accepted UV-only proposal.
    """
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (maximum_iterations, maximum_backtracks))
            or not 1 <= maximum_iterations <= 32 or not 1 <= maximum_backtracks <= 32
            or not 0 < minimum_relative_improvement < .1
            or not np.isfinite(normalized_tail_threshold) or normalized_tail_threshold < 1
            or not np.isfinite(energy_threshold) or energy_threshold < 4):
        raise ValueError('UV repair requires 1..32 iterations/backtracks and a relative improvement in (0, .1)')
    data = _chart_data(layout)
    uv, interior = data['uv'], data['interior']
    before = _quality(uv, data)
    report = {'layout_id': int(layout.layout_id), 'method': 'Fixed-boundary preconditioned symmetric-Dirichlet descent',
              'accepted': False, 'before': before, 'after': before,
              'reference_affine_metric': data['metric'].tolist(), 'iterations': 0, 'objective_evaluations': 0,
              'maximum_iterations': maximum_iterations, 'maximum_backtracks_per_iteration': maximum_backtracks,
              'interior_vertex_count': len(interior), 'source_XYZ_unchanged': True,
              'fixed_boundary_clocks_unchanged': True, 'source_topology_unchanged': True,
              'requires_source_fit_and_native_solid_checks': True}
    def retain(reason):
        if report['after'] is not before:
            report['rejected_candidate_quality'] = report['after']
            report['after'] = before
        report['reason'] = reason
        return report
    if not len(interior):
        return retain('No interior source UV degrees of freedom; all source vertices lie on fixed boundaries')
    if (before['affine_normalized_aspect_p95'] <= normalized_tail_threshold
            and before['affine_normalized_symmetric_Dirichlet_energy'] <= energy_threshold):
        return retain('Affine-normalized distortion does not warrant interior UV optimization')
    if len(uv) > maximum_source_vertices or len(data['triangles']) > maximum_source_triangles:
        return retain('Source chart exceeds the bounded interior UV optimization work policy')
    data['minimum_relative_area'] = min(before['minimum_source_area_normalized_UV_ratio'], 1e-3)
    inverse = data['inverse']
    gradients = np.stack((-inverse.sum(axis=1), inverse[:, 0, :], inverse[:, 1, :]), axis=1)
    local = np.einsum('t,tik,tjk->tij', data['weights'], gradients, gradients)
    triangles = data['triangles']
    laplacian = coo_matrix((local.ravel(), (np.repeat(triangles, 3, axis=1).ravel(),
        np.tile(triangles, (1, 3)).ravel())), shape=(len(uv), len(uv))).tocsr()
    try:
        factor = splu(laplacian[interior][:, interior].tocsc())
    except RuntimeError:
        return retain('Interior source graph has no invertible fixed-boundary preconditioner')
    metric_inverse = np.linalg.inv(data['metric'])
    values, best = uv[interior].ravel().copy(), None
    best_energy = np.inf
    for iteration in range(maximum_iterations):
        energy, gradient = _energy_gradient(values, uv, data)
        report['objective_evaluations'] += 1
        if gradient is None or not np.isfinite(energy):
            break
        direction = -(factor.solve(gradient.reshape(-1, 2)) @ metric_inverse).ravel()
        slope = float(gradient @ direction)
        if not np.isfinite(slope) or slope >= -1e-20:
            break
        step, stepped = .25, False
        for _ in range(maximum_backtracks):
            trial = values + step * direction
            candidate_energy, _ = _energy_gradient(trial, uv, data)
            report['objective_evaluations'] += 1
            if candidate_energy < energy + 1e-4 * step * slope:
                values, stepped = trial, True
                break
            step *= .5
        if not stepped:
            break
        report['iterations'] = iteration + 1
        candidate = uv.copy()
        candidate[interior] = values.reshape(-1, 2)
        quality = _quality(candidate, data)
        if (quality['affine_normalized_symmetric_Dirichlet_energy'] < before['affine_normalized_symmetric_Dirichlet_energy'] * (1 - minimum_relative_improvement)
                and quality['affine_normalized_aspect_p95'] < before['affine_normalized_aspect_p95'] * (1 - minimum_relative_improvement)
                and quality['affine_normalized_aspect_maximum'] <= before['affine_normalized_aspect_maximum'] * (1 + 1e-10)
                and candidate_energy < best_energy):
            best, best_energy = candidate.copy(), candidate_energy
            report['after'] = quality
        if energy - candidate_energy <= 1e-11 * max(energy, 1.):
            break
    if best is None:
        return retain('No bounded candidate improved energy and p95 distortion without worsening maximum distortion')
    samples = getattr(layout, 'sampled_grid_3d', None)
    if samples is not None:
        shape = getattr(layout, 'sampled_shape', (0, 0))
        if (len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in shape)
                or np.asarray(samples).shape != (int(np.prod(shape)), 3)):
            return retain('Existing source samples have no valid tensor shape for atomic UV refresh')
        from .amrto_parameterization import sample_parameter_grid
        working = copy(layout)
        working.parameter_coords = best
        try:
            samples = sample_parameter_grid(working, np.linspace(0., 1., shape[0]),
                                             np.linspace(0., 1., shape[1])).reshape(-1, 3)
        except ValueError as error:
            return retain(f'Atomic source sampling refresh rejected the UV proposal: {error}')
    layout.parameter_coords = best
    if getattr(layout, 'sampled_grid_3d', None) is not None:
        layout.sampled_grid_3d = samples
    report.update(accepted=True, reason='Energy, p95 and maximum distortion admission passed',
                  maximum_UV_vertex_shift=float(np.linalg.norm(best - uv, axis=1).max()),
                  existing_source_grid_refreshed=samples is not None)
    return report


def repair_network_uv_distortion(layouts, *, maximum_charts=3, maximum_iterations=32):
    """Rank genuinely distorted charts; try at most three bounded proposals."""
    if isinstance(maximum_charts, bool) or not isinstance(maximum_charts, int) or not 1 <= maximum_charts <= 3:
        raise ValueError('Network UV repair admits a work budget of one to three charts')
    diagnostics = [measure_affine_normalized_uv_quality(layout) for layout in layouts]
    eligible = [(index, row) for index, row in enumerate(diagnostics) if row['interior_vertex_count']
                and (row['affine_normalized_aspect_p95'] > 3.
                     or row['affine_normalized_symmetric_Dirichlet_energy'] > 5.)]
    eligible.sort(key=lambda item: (-item[1]['affine_normalized_symmetric_Dirichlet_energy'],
                                   -item[1]['affine_normalized_aspect_p95'], item[1]['layout_id']))
    proposals = [repair_layout_uv_distortion(layouts[index], maximum_iterations=maximum_iterations)
                 for index, row in eligible[:maximum_charts]]
    return {'method': 'Selective affine-normalized fixed-boundary interior UV proposals',
            'maximum_chart_attempts': maximum_charts, 'eligible_chart_count': len(eligible),
            'attempted_chart_count': len(proposals), 'accepted_chart_count': sum(row['accepted'] for row in proposals),
            'charts': diagnostics, 'proposals': proposals}
