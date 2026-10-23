# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded normal fitting at smooth shared CAD seams.

This is an engineering extension, inspired by point/normal surface fitting
(Seiler, Grossmann and Juettler, CAGD 64, 2018, 37-49). It is not an
implementation of their nonlinear objective, or a global G1 certificate.
The source charts, knots and common boundary poles remain fixed. Only free
interior poles can move, with a complete convex-hull displacement bound.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, eye, vstack


def _basis(surface, uv):
    weights = getattr(surface, 'weights', None)
    if weights is not None and not np.all(np.asarray(weights) == 1.):
        raise ValueError('Seam normal fitting currently supports unit-weight B-splines only')
    controls = np.asarray(surface.ctrlpts).reshape(
        surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    u = BSpline(surface.knotvector_u, np.eye(controls.shape[0]), surface.degree_u)
    v = BSpline(surface.knotvector_v, np.eye(controls.shape[1]), surface.degree_v)
    bu, bv = u(uv[:, 0]), v(uv[:, 1])
    du = np.einsum('ni,nj->nij', u(uv[:, 0], nu=1), bv).reshape(len(uv), -1)
    dv = np.einsum('ni,nj->nij', bu, v(uv[:, 1], nu=1)).reshape(len(uv), -1)
    position = np.einsum('ni,nj,ijc->nc', bu, bv, controls)
    return position, du, dv, controls


def _normal(du, dv, controls):
    flat = controls.reshape(-1, 3)
    cross = np.cross(du @ flat, dv @ flat)
    length = np.linalg.norm(cross, axis=1)
    if np.any(length <= 1e-14):
        raise ValueError('A seam sample has a degenerate surface tangent')
    return cross / length[:, None]


def _seams(layouts, crease_angle):
    """Use original shared mesh segments, including T-junction subsegments."""
    owners = defaultdict(list)
    protected = set()
    key = lambda point: tuple(map(float, point))
    for index, layout in enumerate(layouts):
        for edge in getattr(layout, 'protected_source_crease_edges', ()):
            protected.add(tuple(sorted(map(key, edge))))
        edge_triangles = defaultdict(list)
        xyz = layout.vertices[layout.triangles]
        normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        for triangle, normal in zip(layout.triangles, normals):
            for a, b in zip(triangle, np.roll(triangle, -1)):
                edge_triangles[tuple(sorted((int(a), int(b))))].append(normal)
        for path in layout.boundary_paths:
            for a, b in zip(path, path[1:]):
                endpoints = (key(layout.vertices[a]), key(layout.vertices[b]))
                ids = (a, b) if endpoints[0] < endpoints[1] else (b, a)
                physical = tuple(sorted(endpoints))
                incident = edge_triangles[tuple(sorted((int(a), int(b))))]
                normal = np.sum(incident, axis=0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    owners[physical].append((index, np.asarray(ids), normal / norm))
    seams, skipped = [], defaultdict(int)
    for edge, sides in owners.items():
        if len(sides) != 2 or sides[0][0] == sides[1][0]:
            skipped['unpaired_or_nonmanifold'] += 1
            continue
        if edge in protected:
            skipped['protected'] += 1
            continue
        if abs(sides[0][2] @ sides[1][2]) < np.cos(np.deg2rad(crease_angle)):
            skipped['source_crease'] += 1
            continue
        seams.append((edge, sides))
    return seams, dict(skipped)


def fair_shared_seam_normals(layouts, maximum_displacement, *, normal_weight=25.,
                             source_crease_angle=30., maximum_controls=4096,
                             progress_callback=None):
    """Propose one bounded fit; admit only improving, source-valid patches.

    Normal-plane residuals are linear in the poles. Pole displacement is the
    fidelity term. Each coordinate is bounded by budget/sqrt(3), hence every
    surface point moves at most the supplied physical budget by the positive
    B-spline convex-hull property. Sampled source/orientation tests still run
    independently. This does not smooth source ripples or change partitions.
    """
    from .amrto_fit_orientation import assess_fit_orientation
    from .amrto_nurbs_fitting import _surface_map_residual
    from .amrto_shared_stock import control_hull_in_box

    if (not np.isfinite(maximum_displacement) or maximum_displacement <= 0
            or not np.isfinite(normal_weight) or normal_weight <= 0
            or not 0 < source_crease_angle < 90):
        raise ValueError('Require a positive physical displacement budget and valid normal settings')
    seams, skipped = _seams(layouts, source_crease_angle)
    rows = defaultdict(list)
    pairs = []
    for edge, sides in seams:
        evaluated = []
        try:
            for index, ids, _ in sides:
                layout = layouts[index]
                uv = np.asarray(layout.parameter_coords)[ids]
                t = np.array([.211324865405187, .5, .788675134594813])
                sites = uv[0] + t[:, None] * (uv[1] - uv[0])
                position, du, dv, controls = _basis(layout.nurbs_surface, sites)
                normal = _normal(du, dv, controls)
                tangent = ((uv[1, 0] - uv[0, 0]) * du
                           + (uv[1, 1] - uv[0, 1]) * dv) @ controls.reshape(-1, 3)
                tangent /= np.linalg.norm(tangent, axis=1)[:, None]
                evaluated.append((index, sites, position, normal, tangent))
            if np.max(np.linalg.norm(evaluated[0][2] - evaluated[1][2], axis=1)) > 1e-7:
                # Retained native clocks can have different source-to-curve
                # parameter maps. Never pretend different XYZ sites coincide.
                skipped['noncoincident_parameter_samples'] = skipped.get('noncoincident_parameter_samples', 0) + 1
                continue
            first, second = evaluated
            sign = np.where(np.einsum('ij,ij->i', first[3], second[3]) >= 0, 1., -1.)
            target = first[3] + sign[:, None] * second[3]
            target -= np.einsum('ij,ij->i', target, first[4])[:, None] * first[4]
            target /= np.linalg.norm(target, axis=1)[:, None]
            if not np.isfinite(target).all():
                raise ValueError('Incompatible normal consensus')
            for current in evaluated:
                rows[current[0]].append((current[1], target))
            pairs.append((first[0], first[1], second[0], second[1]))
        except ValueError:
            skipped['degenerate_sample'] = skipped.get('degenerate_sample', 0) + 1

    def jumps(surfaces):
        result = []
        for ia, ua, ib, ub in pairs:
            _, au, av, ac = _basis(surfaces[ia], ua)
            _, bu, bv, bc = _basis(surfaces[ib], ub)
            dot = np.abs(np.einsum('ij,ij->i', _normal(au, av, ac), _normal(bu, bv, bc)))
            result.extend(np.rad2deg(np.arccos(np.clip(dot, 0, 1))))
        return np.asarray(result)

    original = [layout.nurbs_surface for layout in layouts]
    candidates = list(original)
    reports = []
    for index, observations in rows.items():
        layout, surface = layouts[index], original[index]
        if callable(progress_callback):
            progress_callback(f'Fitting smooth CAD seam normals {len(reports) + 1}/{len(rows)}')
        item = {'layout_id': int(layout.layout_id), 'accepted': False}
        reports.append(item)
        if not getattr(layout, 'fitting_report', {}).get('source_fit_acceptance', {}).get('accepted', False):
            item['reason'] = 'Initial source fit was not accepted'
            continue
        uv = np.vstack([row[0] for row in observations])
        normals = np.vstack([row[1] for row in observations])
        _, du, dv, controls = _basis(surface, uv)
        if controls.shape[0] * controls.shape[1] > maximum_controls:
            item['reason'] = 'Explicit control-count budget exceeded'
            continue
        interior = np.zeros(controls.shape[:2], dtype=bool)
        interior[1:-1, 1:-1] = True
        free = np.flatnonzero(interior.ravel())
        flat = controls.reshape(-1, 3)
        # Normalize derivative equations by their pole sensitivity. A row's
        # residual then has model-length units, like pole displacement.
        coefficients, targets = [], []
        for derivative in (du, dv):
            sensitivity = np.linalg.norm(derivative[:, free], axis=1)
            keep = sensitivity > 1e-10
            if not np.any(keep):
                continue
            basis = derivative[keep][:, free] / sensitivity[keep, None]
            n = normals[keep]
            block = (basis[:, :, None] * n[:, None, :]).reshape(len(basis), -1)
            coefficients.append(block)
            targets.append(-np.einsum('ij,ij->i', derivative[keep] @ flat, n) / sensitivity[keep])
        if not sum(len(block) for block in coefficients):
            item['reason'] = 'No free transverse boundary derivatives'
            continue
        matrix = coo_matrix(np.vstack(coefficients)).tocsr()
        target = np.concatenate(targets)
        weight = np.sqrt(normal_weight / len(target))
        system = vstack((eye(3 * len(free), format='csr') / np.sqrt(len(free)), weight * matrix), format='csr')
        right = np.r_[np.zeros(3 * len(free)), weight * target]
        bound = maximum_displacement / np.sqrt(3.)
        solution = lsq_linear(system, right, bounds=(-bound, bound), method='trf',
                              lsq_solver='lsmr', tol=1e-8, max_iter=50)
        if not solution.success or not np.isfinite(solution.x).all():
            item['reason'] = 'Bounded linear solve did not converge'
            continue
        poles = flat.copy()
        poles[free] += solution.x.reshape(-1, 3)
        candidate = deepcopy(surface)
        candidate.set_ctrlpts(poles.tolist(), *controls.shape[:2])
        orientation = assess_fit_orientation(layout, candidate)
        error = _surface_map_residual(layout, candidate, grid_size=73)
        limits = getattr(layout, 'source_box_bounds', None)
        stock = (control_hull_in_box(poles, limits) if limits is not None else {'accepted': True})
        limit = getattr(layout, 'solid_source_fit_limit',
                        layout.fitting_report['source_fit_acceptance']['maximum_deviation_mm'])
        accepted = bool(orientation['accepted'] and error <= limit and stock['accepted'])
        item.update(accepted=accepted, measured_source_deviation_mm=error,
                    surface_orientation=orientation, stock_hull=stock,
                    maximum_pole_displacement_mm=float(np.linalg.norm(poles - flat, axis=1).max()),
                    boundary_poles_unchanged=bool(np.array_equal(poles[~interior.ravel()], flat[~interior.ravel()])))
        if accepted:
            candidates[index] = candidate
        else:
            item['reason'] = 'Independent source, orientation or stock admission rejected the proposal'
    before, after = jumps(original), jumps(candidates)
    # Atomic publication prevents a local normal improvement from hiding a
    # worse overall shared-seam fit after independent patch admissions.
    accepted = bool(len(before) and np.mean(after ** 2) < np.mean(before ** 2) - 1e-12
                    and after.max() <= before.max() + 1e-7)
    if accepted:
        for index, layout in enumerate(layouts):
            layout.nurbs_surface = candidates[index]
            if candidates[index] is not original[index]:
                row = next(row for row in reports if row['layout_id'] == int(layout.layout_id))
                layout.fitting_report['seam_normal_fit'] = row
                layout.fitting_report['source_fit_acceptance'].update(
                    measured_deviation_mm=row['measured_source_deviation_mm'],
                    surface_orientation=row['surface_orientation'], stock_hull=row['stock_hull'])
                layout.fitting_report['maximum_source_map_residual'] = row['measured_source_deviation_mm']
                layout.fitting_report['source_orientation'] = row['surface_orientation']
    def summary(values):
        return ({'sample_count': len(values), 'rms_degrees': float(np.sqrt(np.mean(values ** 2))),
                 'median_degrees': float(np.median(values)), 'p95_degrees': float(np.percentile(values, 95)),
                 'maximum_degrees': float(values.max())} if len(values) else {'sample_count': 0})
    return {'accepted': accepted, 'method': 'One bounded linear normal-plane fit with fixed shared boundary poles',
            'engineering_extension': True, 'maximum_surface_displacement_bound_mm': float(maximum_displacement),
            'normal_weight': float(normal_weight), 'source_crease_angle_degrees': float(source_crease_angle),
            'patch_partition_unchanged': True, 'shared_boundary_geometry_unchanged': True,
            'global_G1_continuity_certified': False, 'source_ripples_removed': False,
            'angle_definition': 'Acute angle between unoriented tangent planes at coincident seam samples',
            'before': summary(before), 'proposed': summary(after),
            'changed_patch_count': sum(a is not b for a, b in zip(original, candidates)) if accepted else 0,
            'skipped_segments': skipped, 'patches': reports}
