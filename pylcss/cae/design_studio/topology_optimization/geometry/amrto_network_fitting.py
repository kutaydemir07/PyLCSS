# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded coupled point/normal/bending fitting of a spline patch network.

This engineering extension uses sparse Gauss--Newton normal agreement terms,
not independent fits to frozen consensus normals. Common boundary poles stay
exact. Physical XYZ inversion pairs different boundary clocks before fitting;
noncoincident curves never create a constraint. Source, stock and orientation
checks independently admit a complete connected component or none of it.
Finite samples and a bending proxy do not certify global G1 continuity.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize, minimize_scalar

from .amrto_seam_fairing import _seams
from .amrto_surface_fairing import _pole_normals, fairness_matrix


def _axis_basis(sites, knots, degree, derivative=False):
    """Compact support evaluation, including repeated-knot derivatives."""
    if not derivative:
        return sparse.csr_matrix(BSpline.design_matrix(sites, knots, degree))
    # The lower-degree auxiliary basis retains the original clamped knot
    # multiplicity. Its exact endpoint convention has an extra zero slot;
    # evaluate the analytic one-sided derivative limit instead.
    lo, hi = knots[degree], knots[-degree-1]
    sites = np.clip(sites, np.nextafter(lo, hi), np.nextafter(hi, lo))
    count = len(knots)-degree-1
    i = np.arange(count)
    first = np.asarray(knots)[i+degree]-np.asarray(knots)[i]
    second = np.asarray(knots)[i+degree+1]-np.asarray(knots)[i+1]
    a = np.divide(degree, first, out=np.zeros(count), where=first > 0)
    b = np.divide(-degree, second, out=np.zeros(count), where=second > 0)
    conversion = sparse.coo_matrix((np.r_[a, b], (np.r_[i, i+1], np.r_[i, i])),
                                   shape=(count+1, count)).tocsr()
    return sparse.csr_matrix(BSpline.design_matrix(sites, knots, degree-1)) @ conversion


def _tensor_rows(first, second):
    # Row-wise tensor products have only (p+1)^2 entries. Build all CSR
    # products in arrays; constructing two temporary sparse rows per site
    # otherwise dominates batched interior-normal evaluation.
    first_count, second_count = np.diff(first.indptr), np.diff(second.indptr)
    count = first_count*second_count
    indptr = np.r_[0, np.cumsum(count)]
    row = np.repeat(np.arange(first.shape[0]), count)
    slot = np.arange(indptr[-1])-np.repeat(indptr[:-1], count)
    a = first.indptr[row]+slot//second_count[row]
    b = second.indptr[row]+slot % second_count[row]
    columns = first.indices[a]*second.shape[1]+second.indices[b]
    values = first.data[a]*second.data[b]
    result = sparse.csr_matrix((values, columns, indptr),
        shape=(first.shape[0], first.shape[1]*second.shape[1]))
    result.sort_indices()
    result.eliminate_zeros()
    return result


def _position_basis(surface, uv):
    u = _axis_basis(uv[:, 0], surface.knotvector_u, surface.degree_u)
    v = _axis_basis(uv[:, 1], surface.knotvector_v, surface.degree_v)
    return _tensor_rows(u, v)


def _boundary_axis(layout, ids):
    uv = np.asarray(layout.parameter_coords, float)[ids]
    for fixed in range(2):
        if (abs(uv[0, fixed] - uv[1, fixed]) <= 1e-9
                and min(abs(uv[0, fixed]), abs(uv[0, fixed] - 1)) <= 1e-9):
            return fixed, float(round(uv[0, fixed]))
    raise ValueError('Source segment does not lie on one rectangular spline edge')


def _boundary_sites(axis, value, parameters):
    result = np.empty((len(np.atleast_1d(parameters)), 2))
    result[:, axis] = value
    result[:, 1-axis] = np.atleast_1d(parameters)
    return result


def _closest_boundary_parameter(surface, axis, value, xyz, bounds=(0., 1.)):
    """Bounded global span search; one minimizer cannot jump a looping edge."""
    knots = np.asarray(surface.knotvector_v if axis == 0 else surface.knotvector_u)
    degree = surface.degree_v if axis == 0 else surface.degree_u
    controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    side = 0 if value == 0 else -1
    curve = BSpline(knots, controls[side] if axis == 0 else controls[:, side], degree)
    lo, hi = sorted(map(float, bounds))
    if hi-lo <= 1e-14:
        return lo
    cuts = np.unique(np.r_[lo, knots[(knots > lo) & (knots < hi)], hi])
    if len(cuts) > 257:
        raise ValueError('Boundary inversion exceeds its explicit knot-span budget')
    def objective(parameter):
        delta = curve(parameter) - xyz
        return float(delta @ delta)
    candidates = [(objective(float(site)), float(site)) for site in cuts]
    for start, end in zip(cuts[:-1], cuts[1:]):
        answer = minimize_scalar(objective, bounds=(float(start), float(end)), method='bounded',
                                 options={'xatol': 1e-13, 'maxiter': 80})
        if answer.success and np.isfinite(answer.fun):
            candidates.append((float(answer.fun), float(answer.x)))
    return min(candidates)[1]


def _normal_data(surface, uv, directions=None, *, sparse_derivatives=False):
    """Analytic unit normals and their differential for normal pole motion."""
    uv = np.asarray(uv)
    u = _axis_basis(uv[:, 0], surface.knotvector_u, surface.degree_u)
    v = _axis_basis(uv[:, 1], surface.knotvector_v, surface.degree_v)
    du = _tensor_rows(_axis_basis(uv[:, 0], surface.knotvector_u, surface.degree_u, True), v)
    dv = _tensor_rows(u, _axis_basis(uv[:, 1], surface.knotvector_v, surface.degree_v, True))
    flat = np.asarray(surface.ctrlpts)
    position = _tensor_rows(u, v) @ flat
    a, b = du @ flat, dv @ flat
    cross = np.cross(a, b)
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths < 1e-13):
        raise ValueError('A coupled normal sample has a degenerate tangent plane')
    normal = cross / lengths[:, None]
    if directions is None:
        return position, normal
    weighted_u = [du @ sparse.diags(directions[:, axis]) for axis in range(3)]
    weighted_v = [dv @ sparse.diags(directions[:, axis]) for axis in range(3)]
    raw = []
    for axis in range(3):
        j, k = (axis+1) % 3, (axis+2) % 3
        raw.append(weighted_u[j].multiply(b[:, k, None])-weighted_u[k].multiply(b[:, j, None])
                   + weighted_v[k].multiply(a[:, j, None])-weighted_v[j].multiply(a[:, k, None]))
    projection = sum((raw[axis].multiply(normal[:, axis, None]) for axis in range(3)),
                     sparse.csr_matrix(du.shape))
    derivative = [(raw[axis]-projection.multiply(normal[:, axis, None])).multiply(1/lengths[:, None]).tocsr()
                  for axis in range(3)]
    if sparse_derivatives:
        return position, normal, derivative
    return position, normal, np.stack([item.toarray() for item in derivative], axis=2)


def _t_junction_segments(layouts, crease_angle):
    """Overlay unpaired collinear mesh segments without rounding adjacency.

    Some source meshes retain a long edge opposite two shorter boundary
    edges. The ordinary exact edge map cannot see those T-junction owners.
    Only positive-length collinear overlaps create a pairing; mere point
    contacts, nearby parallel curves and nonmanifold triples are excluded.
    """
    owners, protected = defaultdict(list), set()
    for index, layout in enumerate(layouts):
        key = lambda p: tuple(map(float, p))
        protected.update(tuple(sorted(map(key, edge)))
                         for edge in getattr(layout, 'protected_source_crease_edges', ()))
        incident = defaultdict(list)
        for face in np.asarray(layout.triangles):
            xyz = np.asarray(layout.vertices)[face]
            normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
            for a, b in zip(face, np.roll(face, -1)):
                incident[tuple(sorted((int(a), int(b))))].append(normal)
        for path in layout.boundary_paths:
            for a, b in zip(path[:-1], path[1:]):
                first, second = key(layout.vertices[a]), key(layout.vertices[b])
                edge = tuple(sorted((first, second)))
                normal = np.sum(incident[tuple(sorted((int(a), int(b))))], axis=0)
                length = np.linalg.norm(normal)
                if length > 0:
                    owners[edge].append((index, np.array([a, b] if first < second else [b, a]), normal/length))
    segments = [(np.asarray(edge), side[0], edge in protected)
                for edge, side in owners.items() if len(side) == 1]
    scale = max((np.linalg.norm(edge[1]-edge[0]) for edge, _, _ in segments), default=1.)
    epsilon = max(scale*1e-10, 1e-10)
    segments.sort(key=lambda item: (float(item[0][:, 0].min()), item[1][0]))
    proposals = []
    for k, (a, owner_a, protected_a) in enumerate(segments):
        low_a, high_a = a.min(axis=0), a.max(axis=0)
        direction = a[1]-a[0]; length = np.linalg.norm(direction)
        if length <= epsilon: continue
        direction /= length
        for b, owner_b, protected_b in segments[k+1:]:
            low_b, high_b = b.min(axis=0), b.max(axis=0)
            if low_b[0] > high_a[0]+epsilon: break
            if owner_a[0] == owner_b[0] or np.any(high_a+epsilon < low_b) or np.any(high_b+epsilon < low_a): continue
            if np.max(np.linalg.norm(np.cross(b-a[0], direction), axis=1)) > epsilon: continue
            parameters = (b-a[0])@direction
            lo, hi = max(0., float(parameters.min())), min(length, float(parameters.max()))
            if hi-lo <= epsilon: continue
            edge = tuple(map(tuple, (a[0]+lo*direction, a[0]+hi*direction)))
            reason = ('protected' if protected_a or protected_b else 'source_crease'
                      if abs(owner_a[2]@owner_b[2]) < np.cos(np.deg2rad(crease_angle)) else None)
            proposals.append((edge, [owner_a, owner_b], reason))
    # Any overlapping three-owner segment is nonmanifold, not a smooth seam.
    result, skipped = [], defaultdict(int)
    for edge, sides, reason in proposals:
        midpoint = np.mean(edge, axis=0)
        direction = np.asarray(edge[1])-edge[0]; length = np.linalg.norm(direction); direction /= length
        covering = set()
        for source, owner, _ in segments:
            vector = midpoint-source[0]; source_direction = source[1]-source[0]
            source_length = np.linalg.norm(source_direction)
            if source_length <= epsilon: continue
            clock = vector@source_direction/(source_length**2)
            if -1e-10 <= clock <= 1+1e-10 and np.linalg.norm(vector-clock*source_direction) <= epsilon:
                covering.add(owner[0])
        if len(covering) != 2:
            skipped['nonmanifold_T_junction'] += 1
        elif reason:
            skipped[reason] += 1
        else:
            result.append((edge, sides))
    return result, dict(skipped)


def coincident_smooth_seam_samples(layouts, *, source_crease_angle=30., coincidence_tolerance=None):
    """Pair physical sites on existing common curves, retaining T subsegments.

    The mesh supplies edge ownership and source crease classification only.
    Its UV clocks do not identify matching physical sites after curve retiming.
    """
    if not layouts:
        return [], {}
    span = max(float(np.ptp(np.asarray(item.vertices), axis=0).max()) for item in layouts)
    tolerance = (max(1e-8*span, 1e-7) if coincidence_tolerance is None
                 else float(coincidence_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Require a finite positive physical coincidence tolerance')
    seams, counts = _seams(layouts, source_crease_angle)
    skipped = defaultdict(int, counts)
    junctions, junction_skips = _t_junction_segments(layouts, source_crease_angle)
    seams.extend(junctions)
    for name, count in junction_skips.items():
        skipped[name] += count
    if junctions:
        skipped['T_junction_overlaps_resolved'] += len(junctions)
    pairs, endpoint_cache, seen = [], {}, set()
    for edge, sides in seams:
        try:
            endpoints = np.asarray(edge)
            t = np.array([.211324865405187, .5, .788675134594813])
            fast = []
            for index, ids, _ in sides:
                layout = layouts[index]
                axis, value = _boundary_axis(layout, ids)
                xyz = np.asarray(layout.vertices)[ids]
                vector = xyz[1]-xyz[0]
                denominator = float(vector@vector)
                if denominator <= 1e-24:
                    raise ValueError('Zero-length source boundary edge')
                alpha = (endpoints-xyz[0])@vector/denominator
                source_uv = np.asarray(layout.parameter_coords)[ids]
                end_uv = source_uv[0]+alpha[:, None]*(source_uv[1]-source_uv[0])
                uv = end_uv[0]+t[:, None]*(end_uv[1]-end_uv[0])
                if abs(end_uv[1, 1-axis]-end_uv[0, 1-axis]) <= 1e-12:
                    raise ValueError('Degenerate source boundary parameter segment')
                positions = _position_basis(layout.nurbs_surface, uv) @ np.asarray(layout.nurbs_surface.ctrlpts)
                fast.append((index, uv, positions))
            fast_gap = np.linalg.norm(fast[0][2]-fast[1][2], axis=1)
            if fast_gap.max() <= tolerance:
                first, second = fast
                key = (first[0], second[0], tuple(np.round(first[1].ravel(), 12)),
                       tuple(np.round(second[1].ravel(), 12)))
                if key not in seen:
                    seen.add(key)
                    pairs.append({'first': first[0], 'uv_first': first[1],
                        'second': second[0], 'uv_second': second[1], 'kind': 'shared_seam',
                        'maximum_position_gap_mm': float(fast_gap.max()),
                        'pairing_method': 'Source-clock sites independently verified in physical XYZ'})
                    skipped['physically_verified_source_clock_segments'] += 1
                continue
            # Different native clocks still need a physical inversion. A fast
            # mismatch never means that an otherwise valid seam is skipped.
            skipped['physical_clock_inversion_segments'] += 1
            projected = []
            for index, ids, _ in sides:
                layout = layouts[index]
                axis, value = _boundary_axis(layout, ids)
                parameters = []
                for point in endpoints:
                    key = (index, axis, value, tuple(point))
                    if key not in endpoint_cache:
                        endpoint_cache[key] = _closest_boundary_parameter(
                            layout.nurbs_surface, axis, value, point)
                    parameters.append(endpoint_cache[key])
                projected.append((index, axis, value, np.asarray(parameters)))
            first, second = projected
            if min(abs(np.diff(first[3])[0]), abs(np.diff(second[3])[0])) <= 1e-12:
                raise ValueError('Source segment collapses on a common boundary')
            parameters = first[3][0] + t * (first[3][1]-first[3][0])
            ua = _boundary_sites(first[1], first[2], parameters)
            pa = _position_basis(layouts[first[0]].nurbs_surface, ua) @ np.asarray(layouts[first[0]].nurbs_surface.ctrlpts)
            ub_values = [_closest_boundary_parameter(layouts[second[0]].nurbs_surface,
                second[1], second[2], point, second[3]) for point in pa]
            ub = _boundary_sites(second[1], second[2], ub_values)
            pb = _position_basis(layouts[second[0]].nurbs_surface, ub) @ np.asarray(layouts[second[0]].nurbs_surface.ctrlpts)
            gaps = np.linalg.norm(pa-pb, axis=1)
            if np.max(gaps) > tolerance:
                skipped['noncoincident_physical_curves'] += 1
                continue
            # Monotonic pairing disallows folds or arbitrary nearby branches.
            if not np.all(np.diff(ub_values) * (second[3][1]-second[3][0]) > 0):
                skipped['nonmonotone_boundary_pairing'] += 1
                continue
            key = (first[0], second[0], tuple(np.round(ua.ravel(), 12)), tuple(np.round(ub.ravel(), 12)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({'first': first[0], 'uv_first': ua, 'second': second[0],
                          'uv_second': ub, 'kind': 'shared_seam',
                          'maximum_position_gap_mm': float(gaps.max()),
                          'pairing_method': 'Bounded physical boundary-curve inversion'})
        except (ValueError, FloatingPointError):
            skipped['unusable_boundary_segment'] += 1
    return pairs, dict(skipped)


def _source_normals(layout, uv):
    """Incident source normals in the chart's orientation; no nearest-face guess."""
    result = np.zeros((len(uv), 3)); found = np.zeros(len(uv), bool)
    coordinates = np.asarray(layout.parameter_coords)
    for face in np.asarray(layout.triangles):
        a, b, c = coordinates[face]
        ba, ca = b-a, c-a
        determinant = ba[0]*ca[1]-ba[1]*ca[0]
        if abs(float(determinant)) <= 1e-15:
            continue
        query = uv-a
        beta = (query[:, 0]*ca[1]-query[:, 1]*ca[0])/determinant
        gamma = (ba[0]*query[:, 1]-ba[1]*query[:, 0])/determinant
        keep = (~found) & (beta >= -1e-10) & (gamma >= -1e-10) & (beta+gamma <= 1+1e-10)
        if np.any(keep):
            xyz = np.asarray(layout.vertices)[face]
            normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]) * np.sign(determinant)
            length = np.linalg.norm(normal)
            if length > 1e-14:
                result[keep] = normal/length; found[keep] = True
    return result, found


def _internal_pairs(layout, index, crease_angle):
    surface = layout.nurbs_surface
    axes = [(np.asarray(surface.knotvector_u), surface.degree_u),
            (np.asarray(surface.knotvector_v), surface.degree_v)]
    pairs, protected = [], 0
    for axis, (knots, degree) in enumerate(axes):
        unique, multiplicity = np.unique(knots, return_counts=True)
        other = np.unique(axes[1-axis][0])
        sites = np.concatenate([a+(b-a)*np.array([.2113248654, .5, .7886751346])
                                for a, b in zip(other[:-1], other[1:])])
        for k in np.flatnonzero((multiplicity >= degree) & (unique > 0) & (unique < 1)):
            epsilon = min(unique[k]-unique[k-1], unique[k+1]-unique[k])*1e-6
            ua = _boundary_sites(axis, unique[k]-epsilon, sites)
            ub = _boundary_sites(axis, unique[k]+epsilon, sites)
            na, fa = _source_normals(layout, ua); nb, fb = _source_normals(layout, ub)
            smooth = fa & fb & (np.abs(np.einsum('nc,nc->n', na, nb)) >= np.cos(np.deg2rad(crease_angle)))
            if not np.all(smooth):
                # A protected physical crease makes this patch immutable in
                # this stage, including its bending term.
                protected += int(np.count_nonzero(~smooth))
                continue
            pairs.append({'first': index, 'uv_first': ua, 'second': index,
                          'uv_second': ub, 'kind': 'internal_C0'})
    return pairs, protected


def _protected_interior(layout):
    lookup = {tuple(map(float, p)): i for i, p in enumerate(layout.vertices)}
    for edge in getattr(layout, 'protected_source_crease_edges', ()):
        ids = [lookup.get(tuple(map(float, p))) for p in edge]
        if any(value is None for value in ids):
            return True
        uv = np.asarray(layout.parameter_coords)[ids]
        if not any(np.all(np.abs(uv[:, axis]-side) <= 1e-9)
                   for axis in range(2) for side in (0., 1.)):
            return True
    return False


def _summary(values):
    values = np.asarray(values)
    return ({'sample_count': int(len(values)), 'rms_degrees': float(np.sqrt(np.mean(values**2))),
             'maximum_degrees': float(values.max()), 'p95_degrees': float(np.percentile(values, 95))}
            if len(values) else {'sample_count': 0, 'rms_degrees': 0., 'maximum_degrees': 0.})


def _network_normal_data(surfaces, pairs, directions=None):
    """Evaluate each patch in one compact-support batch, not once per edge."""
    sites, counts, references = defaultdict(list), defaultdict(int), []
    for pair in pairs:
        refs = []
        for side in ('first', 'second'):
            index, uv = pair[side], pair['uv_'+side]
            start = counts[index]; counts[index] += len(uv)
            sites[index].append(uv); refs.append((index, slice(start, counts[index])))
        references.append(refs)
    values = {index: _normal_data(surfaces[index], np.vstack(uv),
              None if directions is None else directions.get(index), sparse_derivatives=True)
              for index, uv in sites.items()}
    results = []
    for refs in references:
        result = []
        for index, section in refs:
            item = values[index]
            current = (item[0][section], item[1][section])
            if len(item) == 3:
                current += ([matrix[section] for matrix in item[2]],)
            result.append(current)
        results.append(result)
    return results


def _angles(surfaces, pairs):
    values = []
    for first, second in _network_normal_data(surfaces, pairs):
        a, b = first[1], second[1]
        values.extend(np.rad2deg(np.arccos(np.clip(np.abs(np.einsum('nc,nc->n', a, b)), 0, 1))))
    return np.asarray(values)


def _normal_admission_by_kind(before, after, kinds):
    """An internal-line gain cannot compensate for a worse physical seam."""
    checks = {}
    for kind in ('shared_seam', 'internal_C0'):
        mask = kinds == kind
        if not np.any(mask): continue
        first, second = before[mask], after[mask]
        checks[kind] = {
            'rms_not_worse': bool(np.mean(second**2) <= np.mean(first**2)+1e-9),
            'maximum_not_worse': bool(second.max() <= first.max()+1e-5)}
    return checks


def fair_surface_network(layouts, maximum_displacement, *, normal_weight=100.,
                         bending_weight=5., source_weight=1., pole_weight=.01,
                         interior_normal_weight=25., normal_field_iterations=5,
                         normal_field_relaxation=.5, normal_field_spatial_sigma_mm=None,
                         source_crease_angle=30., maximum_controls=65536,
                         maximum_patch_controls=8192, maximum_iterations=3,
                         maximum_linear_iterations=120, point_grid_size=13,
                         maximum_admission_scales=4, maximum_patch_feasibility_scales=4,
                         coincidence_tolerance=None, fixed_layout_ids=(), progress_callback=None):
    """Fit connected neighbors together and atomically publish admitted sets.

    ``maximum_displacement`` is physical mm: a scalar, a sequence in layout
    order, or a mapping keyed by layout_id. It bounds *total* pole movement
    from entry to exit, including all bounded Gauss--Newton iterations. Unit
    positive B-spline bases extend this bound to every surface point. Source
    samples and analytic normal agreement are separate objective terms;
    source-map/orientation/stock admission remains independent of the solve.
    """
    from .amrto_fit_orientation import assess_fit_orientation
    from .amrto_nurbs_fitting import _surface_map_residual
    from .amrto_parameterization import sample_parameter_grid
    from .amrto_shared_stock import control_hull_in_box
    started = perf_counter()
    if (not all(np.isfinite(value) and value > 0 for value in
                (normal_weight, bending_weight, source_weight, pole_weight))
            or not np.isfinite(interior_normal_weight) or interior_normal_weight < 0
            or not 0 < source_crease_angle < 90 or maximum_iterations not in range(1, 5)
            or not 1 <= maximum_linear_iterations <= 500 or point_grid_size not in range(3, 34)
            or maximum_admission_scales not in range(1, 6)
            or maximum_patch_feasibility_scales not in range(0, 5)
            or maximum_controls < 1 or maximum_patch_controls < 1):
        raise ValueError('Require finite positive weights and explicit finite fitting budgets')
    if np.isscalar(maximum_displacement):
        budgets = np.full(len(layouts), float(maximum_displacement))
    elif isinstance(maximum_displacement, dict):
        budgets = np.array([float(maximum_displacement.get(item.layout_id, 0.)) for item in layouts])
    else:
        budgets = np.asarray(maximum_displacement, float)
    if budgets.shape != (len(layouts),) or not np.isfinite(budgets).all() or np.any(budgets < 0):
        raise ValueError('Every patch requires a finite nonnegative displacement budget')
    original = [layout.nurbs_surface for layout in layouts]
    layout_ids = [int(item.layout_id) for item in layouts]
    fixed_ids = frozenset(int(value) for value in fixed_layout_ids)
    if len(set(layout_ids)) != len(layout_ids) or not fixed_ids.issubset(layout_ids):
        raise ValueError('Fixed network charts require known unique layout IDs')
    for surface in original:
        weights = getattr(surface, 'weights', None)
        if weights is not None and not np.all(np.asarray(weights) == 1.):
            raise ValueError('Coupled network fitting currently requires unit-weight B-splines')
    pairs, skipped = coincident_smooth_seam_samples(layouts,
        source_crease_angle=source_crease_angle, coincidence_tolerance=coincidence_tolerance)
    pairing_seconds = perf_counter()-started
    normal_field = None
    if interior_normal_weight > 0:
        from .amrto_normal_field import build_source_normal_field
        normal_field = build_source_normal_field(layouts, iterations=normal_field_iterations,
            relaxation=normal_field_relaxation, source_crease_angle=source_crease_angle,
            spatial_sigma_mm=normal_field_spatial_sigma_mm)
    data, unavailable, internal = {}, {}, []
    for index, (layout, surface, budget) in enumerate(zip(layouts, original, budgets)):
        acceptance = getattr(layout, 'fitting_report', {}).get('source_fit_acceptance', {})
        if int(layout.layout_id) in fixed_ids:
            unavailable[index] = 'Chart locked at its entry geometry for native contact ownership'; continue
        if not acceptance.get('accepted', False) or budget <= 0:
            unavailable[index] = 'Initial source fit unaccepted or no physical displacement available'; continue
        count = surface.ctrlpts_size_u*surface.ctrlpts_size_v
        if count > maximum_patch_controls:
            unavailable[index] = 'Per-patch control budget exceeded'; continue
        local_pairs, protected = _internal_pairs(layout, index, source_crease_angle)
        if protected or _protected_interior(layout):
            unavailable[index] = 'Protected or true internal source crease retained'; continue
        controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
        mask = np.zeros(controls.shape[:2], bool); mask[1:-1, 1:-1] = True
        free = np.flatnonzero(mask.ravel())
        if not len(free):
            unavailable[index] = 'No free interior poles'; continue
        directions = _pole_normals(surface, controls).reshape(-1, 3)
        if not np.isfinite(directions).all():
            unavailable[index] = 'Invalid reference pole normals'; continue
        flat = controls.reshape(-1, 3)
        lower, upper = np.full(len(free), -budget), np.full(len(free), budget)
        bounds = getattr(layout, 'source_box_bounds', None)
        if bounds is not None:
            low, high = np.asarray(bounds)
            for axis in range(3):
                velocity = directions[free, axis]
                moving = np.abs(velocity) > 1e-14
                a = np.divide(low[axis]-flat[free, axis], velocity, out=np.zeros(len(free)), where=moving)
                b = np.divide(high[axis]-flat[free, axis], velocity, out=np.zeros(len(free)), where=moving)
                lower[moving] = np.maximum(lower[moving], np.minimum(a, b)[moving])
                upper[moving] = np.minimum(upper[moving], np.maximum(a, b)[moving])
        if np.any(lower > 0) or np.any(upper < 0):
            unavailable[index] = 'Initial control hull lies outside stock'; continue
        data[index] = dict(free=free, flat=flat, directions=directions, lower=lower,
                           upper=upper, count=count, bounds=bounds)
        internal.extend(local_pairs)
    pairs.extend(internal)
    adjacency = {index: set() for index in data}
    for pair in pairs:
        a, b = pair['first'], pair['second']
        if a in data and b in data:
            adjacency[a].add(b); adjacency[b].add(a)
    components, remaining = [], set(data)
    while remaining:
        queue = [min(remaining)]; group = []
        while queue:
            index = queue.pop()
            if index not in remaining: continue
            remaining.remove(index); group.append(index)
            queue.extend(sorted(adjacency[index] & remaining, reverse=True))
        components.append(sorted(group))
    reports, accepted_ids = [], []
    total_controls = sum(item['count'] for item in data.values())
    for component_number, group in enumerate(components):
        began = perf_counter()
        report = {'layout_ids': [int(layouts[index].layout_id) for index in group],
                  'accepted': False, 'iterations': [], 'control_count': sum(data[index]['count'] for index in group)}
        reports.append(report)
        if total_controls > maximum_controls:
            report['reason'] = 'Whole-network total control budget exceeded'; continue
        if callable(progress_callback):
            progress_callback(f'Joint CAD point/normal/bending fit {component_number+1}/{len(components)}')
        group_set = set(group)
        local_pairs = [pair for pair in pairs if pair['first'] in group_set or pair['second'] in group_set]
        offset, size = {}, 0
        for index in group:
            offset[index] = size; size += len(data[index]['free'])
        hessian = sparse.lil_matrix((size, size)); right = np.zeros(size)
        lower, upper = np.empty(size), np.empty(size)
        for index in group:
            datum, layout, surface = data[index], layouts[index], original[index]
            free, flat, directions = datum['free'], datum['flat'], datum['directions']
            section = slice(offset[index], offset[index]+len(free))
            lower[section], upper[section] = datum['lower'], datum['upper']
            knot_u, knot_v = np.unique(surface.knotvector_u), np.unique(surface.knotvector_v)
            def training_axis(knots):
                # Objective quadrature is bounded; validation below still
                # inspects every span and source vertex independently.
                mids = .5*(knots[:-1]+knots[1:])
                count = min(len(mids), 33-point_grid_size)
                ids = np.unique(np.round(np.linspace(0, len(mids)-1, count)).astype(int)) if count else []
                return np.unique(np.r_[np.linspace(0, 1, point_grid_size), mids[ids]])
            u, v = training_axis(knot_u), training_axis(knot_v)
            datum['source_training_sites'] = int(len(u)*len(v))
            datum['source_training_knot_spans_available'] = [int(len(knot_u)-1), int(len(knot_v)-1)]
            uu, vv = np.meshgrid(u, v, indexing='ij'); uv = np.column_stack((uu.ravel(), vv.ravel()))
            source = sample_parameter_grid(layout, u, v).reshape(-1, 3)
            if normal_field is not None:
                # Normal targets are interior observations. Exact chart
                # corners may legitimately have a zero tensor tangent while
                # the trimmed physical surface remains regular; the position
                # objective and independent orientation checker still retain
                # those boundary sites under their own explicit rules.
                normal_u, normal_v = u[(u > 0) & (u < 1)], v[(v > 0) & (v < 1)]
                gu, gv = np.meshgrid(normal_u, normal_v, indexing='ij')
                normal_uv = np.column_stack((gu.ravel(), gv.ravel()))
                target_normals = normal_field.sample(layout, index, normal_u, normal_v).reshape(-1, 3)
                _, initial_normals = _normal_data(surface, normal_uv)
                target_normals *= np.where(np.einsum('nc,nc->n', target_normals, initial_normals) >= 0, 1., -1.)[:, None]
                datum['interior_uv'] = normal_uv
                datum['interior_normal_targets'] = target_normals
                datum['interior_normal_initial_error'] = np.rad2deg(np.arccos(np.clip(
                    np.einsum('nc,nc->n', initial_normals, target_normals), -1, 1)))
            basis = sparse.csr_matrix(_position_basis(surface, uv))
            residual = source - basis @ flat
            local_hessian = sparse.eye(len(free), format='csr')*(pole_weight/len(free))
            local_right = np.zeros(len(free))
            bending = fairness_matrix(surface)
            datum['bending'] = bending
            for axis in range(3):
                direction = sparse.diags(directions[free, axis])
                jacobian = basis[:, free] @ direction
                local_hessian += source_weight*(jacobian.T @ jacobian)/len(uv)
                local_right += source_weight*(jacobian.T @ residual[:, axis])/len(uv)
                local_hessian += bending_weight*(direction @ bending[free][:, free] @ direction)/len(flat)
                local_right -= bending_weight*(direction @ (bending[free] @ flat[:, axis]))/len(flat)
            hessian[section, section] = local_hessian
            right[section] = local_right
        else:
            hessian = hessian.tocsr(); q = np.zeros(size); candidate_surfaces = list(original)
            report['base_objective_assembly_seconds'] = perf_counter()-began
            before = _angles(original, local_pairs)
            local_kinds = (np.concatenate([np.full(len(pair['uv_first']), pair['kind']) for pair in local_pairs])
                           if local_pairs else np.array([]))
            initial_energy = 0.
            for index in group:
                flat = data[index]['flat']; matrix = data[index]['bending']
                initial_energy += float(np.sum(flat*(matrix@flat)))
            report['before'] = _summary(before)
            for iteration in range(maximum_iterations):
                jacobians, residuals = [], []
                normal_data = _network_normal_data(candidate_surfaces, local_pairs,
                    {index: data[index]['directions'] for index in group})
                for pair, (aa, bb) in zip(local_pairs, normal_data):
                    a, b = pair['first'], pair['second']
                    sign = np.where(np.einsum('nc,nc->n', aa[1], bb[1]) >= 0, 1., -1.)
                    residual = aa[1]-sign[:, None]*bb[1]
                    block = sparse.csr_matrix((3*len(residual), size))
                    for index, values, factor in ((a, aa, np.ones(len(sign))), (b, bb, -sign)):
                        if index not in group_set: continue
                        datum = data[index]
                        global_columns = np.full(len(datum['flat']), -1, dtype=int)
                        global_columns[datum['free']] = offset[index]+np.arange(len(datum['free']))
                        rr, cc, dd = [], [], []
                        for axis in range(3):
                            local = values[2][axis].tocoo()
                            columns = global_columns[local.col]
                            keep = columns >= 0
                            rr.extend((3*local.row[keep]+axis).tolist())
                            cc.extend(columns[keep].tolist())
                            dd.extend((local.data[keep]*factor[local.row[keep]]).tolist())
                        block += sparse.coo_matrix((dd, (rr, cc)), shape=block.shape).tocsr()
                    # Same-patch C0 limits have already summed common unknowns.
                    norm = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).ravel().reshape(-1, 3).sum(axis=1))
                    inverse = np.divide(1., norm, out=np.zeros_like(norm), where=norm > 1e-13)
                    jacobians.append(block.multiply(np.repeat(inverse, 3)[:, None]).tocsr())
                    residuals.extend((residual*inverse[:, None]).ravel().tolist())
                jacobian = sparse.vstack(jacobians, format='csr') if jacobians else sparse.csr_matrix((0, size))
                row = jacobian.shape[0]
                weight = normal_weight/max(1, row)
                system = hessian + weight*(jacobian.T@jacobian)
                target = right + weight*(jacobian.T@(jacobian@q-np.asarray(residuals)))
                interior_rows, interior_errors = 0, []
                if normal_field is not None:
                    physical_scale = normal_field.report['median_source_edge_mm']
                    blocks, interior_right = [], np.zeros(size)
                    for index in group:
                        datum = data[index]; free = datum['free']
                        _, normals, derivatives = _normal_data(candidate_surfaces[index],
                            datum['interior_uv'], datum['directions'], sparse_derivatives=True)
                        desired = datum['interior_normal_targets']
                        residual = (normals-desired).T.ravel()*physical_scale
                        matrix = sparse.vstack([item[:, free] for item in derivatives], format='csr')*physical_scale
                        section = slice(offset[index], offset[index]+len(free))
                        blocks.append((matrix.T@matrix).tocsr())
                        interior_right[section] = matrix.T@(matrix@q[section]-residual)
                        interior_rows += matrix.shape[0]
                        interior_errors.extend(np.rad2deg(np.arccos(np.clip(np.einsum('nc,nc->n', normals, desired), -1, 1))))
                    interior_weight = interior_normal_weight/max(1, interior_rows)
                    system += interior_weight*sparse.block_diag(blocks, format='csr')
                    target += interior_weight*interior_right
                diagonal = np.sqrt(np.maximum(system.diagonal(), 1e-15))
                scaled = sparse.diags(1/diagonal) @ system @ sparse.diags(1/diagonal)
                scaled_target = target/diagonal
                def objective(value):
                    product = scaled@value
                    return .5*float(value@product)-float(scaled_target@value), product-scaled_target
                solution = minimize(objective, q*diagonal, method='L-BFGS-B', jac=True,
                    bounds=np.column_stack((lower*diagonal, upper*diagonal)),
                    options={'maxiter': maximum_linear_iterations, 'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-8})
                report['iterations'].append({'iteration': iteration+1, 'linear_iterations': int(solution.nit),
                    'linear_success': bool(solution.success), 'linear_message': str(solution.message),
                    'normal_equation_rows': int(row), 'sparse_system_nonzeros': int(system.nnz),
                    'interior_normal_equation_rows': int(interior_rows),
                    'interior_normal_error_before_step': _summary(interior_errors)})
                if not np.isfinite(solution.x).all():
                    report['reason'] = 'Nonfinite bounded sparse solution'; break
                q = np.clip(solution.x/diagonal, lower, upper)
                for index in group:
                    datum, surface = data[index], original[index]
                    free, poles = datum['free'], datum['flat'].copy()
                    poles[free] += q[offset[index]:offset[index]+len(free), None]*datum['directions'][free]
                    candidate = deepcopy(surface)
                    candidate.set_ctrlpts(poles.tolist(), surface.ctrlpts_size_u, surface.ctrlpts_size_v)
                    candidate_surfaces[index] = candidate
                after = _angles(candidate_surfaces, local_pairs)
                report['iterations'][-1]['measured_normal_jumps'] = _summary(after)
                if callable(progress_callback):
                    progress_callback(f'Joint CAD fitting component {component_number+1}: normal iteration {iteration+1}/{maximum_iterations}')
            if 'reason' not in report:
                after = _angles(candidate_surfaces, local_pairs)
                report['unscaled_proposed'] = _summary(after)
                report['admission_attempts'] = []
                def candidate_at(index, displacement):
                    datum, surface = data[index], original[index]
                    free, poles = datum['free'], datum['flat'].copy()
                    poles[free] += displacement[:, None]*datum['directions'][free]
                    candidate = deepcopy(surface)
                    candidate.set_ctrlpts(poles.tolist(), surface.ctrlpts_size_u, surface.ctrlpts_size_v)
                    return candidate
                def patch_admission(index, candidate):
                    layout, datum = layouts[index], data[index]
                    poles = np.asarray(candidate.ctrlpts)
                    energy = float(np.sum(poles*(datum['bending']@poles)))
                    orientation = assess_fit_orientation(layout, candidate)
                    deviation = float(_surface_map_residual(layout, candidate, grid_size=73))
                    limit = float(getattr(layout, 'solid_source_fit_limit',
                        layout.fitting_report['source_fit_acceptance']['maximum_deviation_mm']))
                    stock = control_hull_in_box(poles, datum['bounds']) if datum['bounds'] is not None else {'accepted': True}
                    movement = float(np.linalg.norm(poles-datum['flat'], axis=1).max(initial=0.))
                    fixed = np.ones(len(poles), bool); fixed[datum['free']] = False
                    boundary_exact = bool(np.array_equal(poles[fixed], datum['flat'][fixed]))
                    good = bool(orientation['accepted'] and deviation <= limit and stock['accepted']
                                and movement <= budgets[index]+1e-12 and boundary_exact)
                    field_info, first_errors, last_errors = {}, [], []
                    if normal_field is not None:
                        _, measured_normals = _normal_data(candidate, datum['interior_uv'])
                        desired = datum['interior_normal_targets']
                        last_errors = np.rad2deg(np.arccos(np.clip(np.einsum('nc,nc->n', measured_normals, desired), -1, 1)))
                        first_errors = datum['interior_normal_initial_error']
                        field_info = {'interior_normal_error_before': _summary(first_errors),
                                      'interior_normal_error_after': _summary(last_errors)}
                    row = {'layout_id': int(layout.layout_id), 'accepted': good, **field_info,
                        'source_deviation_mm': deviation, 'source_limit_mm': limit,
                        'maximum_pole_displacement_mm': movement,
                        'maximum_surface_displacement_bound_mm': float(budgets[index]),
                        'source_training_sites': datum['source_training_sites'],
                        'source_training_knot_spans_available': datum['source_training_knot_spans_available'],
                        'boundary_poles_unchanged': boundary_exact, 'orientation': orientation, 'stock': stock}
                    return row, energy, first_errors, last_errors
                # Project only geometrically infeasible local motions through
                # a fixed set of smaller factors. This publishes nothing. All
                # join/field metrics and the entire component are checked again
                # after projection, so one patch cannot silently degrade peers.
                feasible_q = q.copy(); patch_factors = {index: 1. for index in group}
                admission_cache, projection_attempts = {}, []
                for index in group:
                    section = slice(offset[index], offset[index]+len(data[index]['free']))
                    result = patch_admission(index, candidate_surfaces[index])
                    if not result[0]['accepted']:
                        initial_result = result
                        projected = False
                        factors = [.5**(k+1) for k in range(maximum_patch_feasibility_scales)]
                        if factors: factors[-1] = 0.
                        attempts = [{'factor': 1., 'accepted': False}]
                        for local_factor in factors:
                            candidate = candidate_at(index, local_factor*q[section])
                            result = patch_admission(index, candidate)
                            attempts.append({'factor': local_factor, 'accepted': result[0]['accepted']})
                            if result[0]['accepted']:
                                candidate_surfaces[index] = candidate
                                feasible_q[section] = local_factor*q[section]
                                patch_factors[index] = local_factor
                                projected = True
                                break
                        if not projected:
                            result = initial_result
                        projection_attempts.append({'layout_id': int(layouts[index].layout_id), 'attempts': attempts})
                    admission_cache[index] = result
                report['patch_feasibility_attempts'] = projection_attempts
                report['patch_feasibility_displacement_factors'] = {
                    str(int(layouts[index].layout_id)): factor for index, factor in patch_factors.items()}
                # Common component backoff remains a finite fallback if the
                # projected network does not meet independent normal checks.
                for scale_index in range(maximum_admission_scales):
                    factor = .5**scale_index
                    if scale_index:
                        for index in group:
                            section = slice(offset[index], offset[index]+len(data[index]['free']))
                            candidate_surfaces[index] = candidate_at(index, factor*feasible_q[section])
                    after = _angles(candidate_surfaces, local_pairs)
                    patches, final_energy, field_before, field_after = [], 0., [], []
                    for index in group:
                        result = admission_cache[index] if not scale_index else patch_admission(index, candidate_surfaces[index])
                        item, energy, first_errors, last_errors = result
                        item = {**item, 'patch_feasibility_displacement_factor': patch_factors[index],
                                'component_displacement_factor': factor,
                                'total_displacement_factor': factor*patch_factors[index]}
                        patches.append(item); final_energy += energy
                        field_before.extend(first_errors); field_after.extend(last_errors)
                    normal_improved = (bool(len(before)) and float(np.mean(after**2)) < float(np.mean(before**2))-1e-10)
                    maximum_ok = not len(before) or float(after.max()) <= float(before.max())+1e-5
                    kind_checks = _normal_admission_by_kind(before, after, local_kinds)
                    kinds_ok = all(all(checks.values()) for checks in kind_checks.values())
                    field_ok = (not field_before or np.mean(np.asarray(field_after)**2)
                                <= np.mean(np.asarray(field_before)**2)+1e-9)
                    bending_improved = final_energy < initial_energy-1e-10*max(1., initial_energy)
                    accepted = bool(all(item['accepted'] for item in patches) and maximum_ok and kinds_ok and field_ok
                                   and (normal_improved or (not len(before) and bending_improved)))
                    report['admission_attempts'].append({'component_displacement_factor': factor,
                        'accepted': accepted, 'failed_layout_ids': [item['layout_id'] for item in patches if not item['accepted']],
                        'normal_jumps': _summary(after), 'normal_admission_by_kind': kind_checks,
                        'interior_normal_error_before': _summary(field_before),
                        'interior_normal_error_after': _summary(field_after),
                        'interior_normal_rms_not_worse': bool(field_ok)})
                    if accepted: break
                report.update(patches=patches, proposed=_summary(after), bending_proxy_before=initial_energy,
                    bending_proxy_after=final_energy, accepted=accepted,
                    component_displacement_factor=factor)
                if report['accepted']:
                    # Publish only after all independent checks finish.
                    for index, item in zip(group, patches):
                        layout = layouts[index]; layout.nurbs_surface = candidate_surfaces[index]
                        layout.fitting_report['network_fit'] = item
                        layout.fitting_report['source_fit_acceptance'].update(
                            measured_deviation_mm=item['source_deviation_mm'],
                            surface_orientation=item['orientation'], stock_hull=item['stock'])
                        layout.fitting_report['maximum_source_map_residual'] = item['source_deviation_mm']
                        layout.fitting_report['source_orientation'] = item['orientation']
                        accepted_ids.append(int(layout.layout_id))
                else:
                    report['reason'] = 'Whole component retained: admission or measured normal/bending improvement failed'
        report['seconds'] = perf_counter()-began
    current = [layout.nurbs_surface for layout in layouts]
    before_all, after_all = _angles(original, pairs), _angles(current, pairs)
    kinds = np.concatenate([np.full(len(pair['uv_first']), pair['kind']) for pair in pairs]) if pairs else np.array([])
    return {'accepted': bool(accepted_ids), 'method': 'Coupled sparse bounded point/normal/bending fitting',
        'engineering_extension': True, 'component_count': len(components),
        'attempted_component_count': sum(bool(item.get('iterations')) for item in reports),
        'accepted_component_count': sum(item['accepted'] for item in reports),
        'accepted_layout_ids': accepted_ids, 'components': reports,
        'before': _summary(before_all), 'after': _summary(after_all),
        'normal_jumps_by_kind': {kind: {'before': _summary(before_all[kinds == kind]),
                                      'after': _summary(after_all[kinds == kind])}
                                 for kind in ('shared_seam', 'internal_C0')},
        'seam_sample_pair_count': sum(item['kind'] == 'shared_seam' for item in pairs),
        'internal_C0_pair_count': sum(item['kind'] == 'internal_C0' for item in pairs),
        'skipped_segments': skipped,
        'physical_seam_pairing_seconds': pairing_seconds,
        'immutable_layouts': [{'layout_id': int(layouts[index].layout_id), 'reason': reason}
                              for index, reason in sorted(unavailable.items())],
        'fixed_layout_ids': sorted(fixed_ids),
        'weights': {'source_position': source_weight, 'normal_agreement': normal_weight,
                    'bending': bending_weight, 'pole_displacement': pole_weight,
                    'interior_source_normal_field': interior_normal_weight},
        'source_normal_field': None if normal_field is None else normal_field.report,
        'maximum_outer_iterations': maximum_iterations,
        'maximum_linear_iterations_per_outer_iteration': maximum_linear_iterations,
        'maximum_component_admission_scales': maximum_admission_scales,
        'maximum_patch_feasibility_scales': maximum_patch_feasibility_scales,
        'maximum_total_controls': maximum_controls, 'total_active_controls': total_controls,
        'source_objective_quadrature': 'At most 33 by 33 uniform and deterministically selected knot-span sites per chart',
        'source_admission_quadrature': 'Independent 73 by 73 grid plus every knot span and source vertex',
        'shared_boundary_geometry_unchanged': True, 'patch_partition_unchanged': True,
        'tangential_reparameterization_excluded': True, 'global_G1_continuity_certified': False,
        'seconds': perf_counter()-started}
