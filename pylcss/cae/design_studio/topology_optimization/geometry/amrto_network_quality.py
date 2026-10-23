# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded, geometry-based diagnostics for a fitted tensor-spline network.

These are sampled diagnostics, not a smoothness or geometric-error certificate.
Curvature uses analytic physical first/second fundamental forms, rather than
UV second differences. Shared boundaries are paired in physical space so a
different boundary clock is not mistaken for a geometric gap or tangent jump.
Explicitly protected edges are reported separately from inferred source creases.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from .amrto_continuity import inspect_internal_spline_continuity


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'sample_count': int(len(values)),
            'minimum': float(values.min()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p95': float(np.quantile(values, .95)) if len(values) else None,
            'maximum': float(values.max()) if len(values) else None,
            'rms': float(np.sqrt(np.mean(values ** 2))) if len(values) else None}


def _evaluate(surface, uv, second=True):
    """Evaluate polynomial or rational tensor splines, including derivatives."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    controls = np.asarray(surface.ctrlpts, dtype=float).reshape(
        surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    weights = getattr(surface, 'weights', None)
    weights = (np.ones(controls.shape[:2]) if weights is None else
               np.asarray(weights, dtype=float).reshape(controls.shape[:2]))
    if not np.isfinite(controls).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Diagnostics require finite control points and positive rational weights')
    bases = [BSpline(surface.knotvector_u, np.eye(controls.shape[0]), surface.degree_u),
             BSpline(surface.knotvector_v, np.eye(controls.shape[1]), surface.degree_v)]
    degrees = [surface.degree_u, surface.degree_v]
    derivative = [[basis(uv[:, axis], nu=order) if order <= degrees[axis] else
                   np.zeros((len(uv), controls.shape[axis]))
                   for order in range(3 if second else 2)]
                  for axis, basis in enumerate(bases)]
    weighted = controls * weights[:, :, None]
    def numerator(a, b):
        return np.einsum('ni,nj,ijc->nc', derivative[0][a], derivative[1][b], weighted)
    def denominator(a, b):
        return np.einsum('ni,nj,ij->n', derivative[0][a], derivative[1][b], weights)
    w = denominator(0, 0)[:, None]
    point = numerator(0, 0) / w
    wu, wv = denominator(1, 0)[:, None], denominator(0, 1)[:, None]
    du = (numerator(1, 0) - wu * point) / w
    dv = (numerator(0, 1) - wv * point) / w
    if not second:
        return point, du, dv
    duu = (numerator(2, 0) - 2 * wu * du - denominator(2, 0)[:, None] * point) / w
    dvv = (numerator(0, 2) - 2 * wv * dv - denominator(0, 2)[:, None] * point) / w
    duv = (numerator(1, 1) - wu * dv - wv * du - denominator(1, 1)[:, None] * point) / w
    return point, du, dv, duu, duv, dvv


def _normals(du, dv):
    cross = np.cross(du, dv)
    lengths = np.linalg.norm(cross, axis=1)
    scale = np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1)
    valid = (lengths > np.maximum(np.finfo(float).tiny, 1e-12 * scale)) & np.isfinite(lengths)
    normal = np.full_like(cross, np.nan)
    np.divide(cross, lengths[:, None], out=normal, where=valid[:, None])
    return normal, lengths, valid


def inspect_patch_quality(layout, *, maximum_samples=225, maximum_controls=8192):
    """Sample physical curvature and source-triangle UV distortion of a chart.

    Curvature variation is a diagnostic, not inherently a defect: a designed
    fillet can have changing curvature. Reported units assume XYZ is in mm.
    Dimensionless curvature variation is invariant under uniform XYZ scaling.
    Raw UV anisotropy includes legitimate global chart aspect ratio and is not
    by itself a distortion error. Area-scale variation is normalized by its
    median to remove global UV scale, but not global affine anisotropy.
    """
    if maximum_samples < 4:
        raise ValueError('Require at least four patch samples')
    surface = layout.nurbs_surface
    if surface.ctrlpts_size_u * surface.ctrlpts_size_v > maximum_controls:
        raise ValueError('Explicit diagnostic control-count budget exceeded')
    count = max(2, int(np.sqrt(maximum_samples)))
    axes = []
    for knots, degree in ((surface.knotvector_u, surface.degree_u),
                          (surface.knotvector_v, surface.degree_v)):
        knots = np.asarray(knots, dtype=float)
        lo, hi = knots[degree], knots[-degree-1]
        values = lo + (np.arange(count) + .5) / count * (hi - lo)
        # Avoid evaluating an unspecified side at internal knots.
        for knot in np.unique(knots[degree+1:-degree-1]):
            values[values == knot] = np.nextafter(knot, hi)
        axes.append(values)
    uv = np.array([(u, v) for u in axes[0] for v in axes[1]])
    point, du, dv, duu, duv, dvv = _evaluate(surface, uv)
    normal, area, valid = _normals(du, dv)
    e1, f1, g1 = (np.einsum('ij,ij->i', a, b) for a, b in ((du, du), (du, dv), (dv, dv)))
    e2, f2, g2 = (np.einsum('ij,ij->i', normal, d) for d in (duu, duv, dvv))
    determinant = e1 * g1 - f1 ** 2
    valid &= determinant > np.maximum(np.finfo(float).tiny, 1e-12 * e1 * g1)
    mean = np.full(len(uv), np.nan)
    gaussian = np.full(len(uv), np.nan)
    np.divide(e2 * g1 - 2 * f2 * f1 + g2 * e1, 2 * determinant, out=mean, where=valid)
    np.divide(e2 * g2 - f2 ** 2, determinant, out=gaussian, where=valid)
    principal = np.sqrt(np.maximum(0., mean ** 2 - gaussian))
    max_curvature = np.maximum(np.abs(mean + principal), np.abs(mean - principal))
    span = float(np.linalg.norm(np.ptp(np.asarray(layout.vertices, dtype=float), axis=0)))
    absolute_mean = np.abs(mean).reshape(count, count)
    xyz = point.reshape(count, count, 3)
    variation = []
    for axis in range(2):
        difference = np.abs(np.diff(absolute_mean, axis=axis))
        distance = np.linalg.norm(np.diff(xyz, axis=axis), axis=2)
        keep = np.isfinite(difference) & (distance > max(np.finfo(float).tiny, 1e-12 * span))
        variation.extend((difference[keep] / distance[keep]).tolist())
    # Source mesh UV distortion is separately measured before spline fitting.
    triangles = np.asarray(layout.triangles, dtype=int)
    vertex = np.asarray(layout.vertices, dtype=float)
    source_uv = np.asarray(layout.parameter_coords, dtype=float)
    # Deterministic bounded subset; every triangle is counted, omissions explicit.
    maximum_triangles = max(64, 4 * maximum_samples)
    chosen = (np.arange(len(triangles)) if len(triangles) <= maximum_triangles else
              np.unique(np.linspace(0, len(triangles)-1, maximum_triangles).astype(int)))
    cells = triangles[chosen]
    source_edge = np.stack((vertex[cells[:, 1]] - vertex[cells[:, 0]],
                            vertex[cells[:, 2]] - vertex[cells[:, 0]]), axis=2)
    uv_edge = np.stack((source_uv[cells[:, 1]] - source_uv[cells[:, 0]],
                        source_uv[cells[:, 2]] - source_uv[cells[:, 0]]), axis=2)
    determinant_uv = np.linalg.det(uv_edge)
    invertible = np.abs(determinant_uv) > np.finfo(float).eps * np.linalg.norm(uv_edge, axis=(1, 2)) ** 2
    singular = np.linalg.svd(source_edge[invertible] @ np.linalg.inv(uv_edge[invertible]), compute_uv=False)
    positive = singular[:, 1] > np.finfo(float).eps * singular[:, 0]
    singular = singular[positive]
    anisotropy = singular[:, 0] / singular[:, 1]
    area_scale = np.prod(singular, axis=1)
    median_area = float(np.median(area_scale)) if len(area_scale) else 0.
    area_ratio = area_scale / median_area if median_area > 0 else np.array([])
    variation = np.asarray(variation)
    curvature = max_curvature[np.isfinite(max_curvature)]
    peak = int(np.nanargmax(max_curvature)) if len(curvature) else None
    peak_location = (None if peak is None else
                     {'position_mm': point[peak].tolist(), 'UV': uv[peak].tolist(),
                      'maximum_absolute_principal_curvature_per_mm': float(max_curvature[peak])})
    return {'layout_id': int(layout.layout_id), 'sampling_budget': int(maximum_samples),
            'control_count_budget': int(maximum_controls),
            'physical_curvature': {'method': 'Analytic first and second fundamental forms at bounded UV interior sites',
                'sample_count': int(len(uv)), 'undefined_sample_count': int(np.count_nonzero(~valid)),
                'maximum_absolute_principal_curvature_per_mm': _summary(curvature),
                'maximum_sampled_curvature_location': peak_location,
                'absolute_mean_curvature_neighbor_gradient_per_mm_squared': _summary(variation),
                'dimensionless_curvature_variation': float(np.std(curvature) * span) if len(curvature) else None,
                'dimensionless_mean_curvature_gradient_rms': float(np.sqrt(np.mean(variation ** 2)) * span ** 2) if len(variation) else None,
                'source_chart_diagonal_mm': span,
                'warning': 'Curvature variation can be intentional; small unsampled knot spans and peaks can be missed'},
            'source_uv_distortion': {'source_triangle_count': int(len(triangles)),
                'sampled_triangle_count': int(len(chosen)),
                'singular_or_degenerate_sampled_triangles': int(len(chosen) - len(singular)),
                'UV_to_XYZ_singular_value_ratio': _summary(anisotropy),
                'area_scale_divided_by_sample_median': _summary(area_ratio),
                'raw_aspect_ratio_includes_legitimate_global_chart_anisotropy': True,
                'warning': 'Raw singular-value ratio is not a distortion defect; global affine chart aspect has not been removed',
                'method': 'SVD of piecewise affine UV-to-source-XYZ triangle Jacobians'}}


def _source_segments(layouts):
    owners, protected = defaultdict(list), set()
    key = lambda point: tuple(map(float, point))
    for index, layout in enumerate(layouts):
        for edge in getattr(layout, 'protected_source_crease_edges', ()):
            protected.add(tuple(sorted(map(key, edge))))
        incident = defaultdict(list)
        xyz = np.asarray(layout.vertices)[np.asarray(layout.triangles)]
        face_normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        for face, normal in zip(layout.triangles, face_normals):
            for a, b in zip(face, np.roll(face, -1)):
                incident[tuple(sorted((int(a), int(b))))].append(normal)
        for path in layout.boundary_paths:
            for a, b in zip(path, path[1:]):
                endpoints = (key(layout.vertices[a]), key(layout.vertices[b]))
                ids = (int(a), int(b)) if endpoints[0] < endpoints[1] else (int(b), int(a))
                normal = np.sum(incident[tuple(sorted((int(a), int(b))))], axis=0)
                length = np.linalg.norm(normal)
                owners[tuple(sorted(endpoints))].append((index, ids, normal / length if length > 0 else None))
    return owners, protected


def inspect_shared_seams(layouts, *, maximum_samples=1200, samples_per_segment=3,
                         position_tolerance=None, source_crease_angle=30.):
    """Measure shared seam tangent planes at physically coincident sites.

    Each sample on the first UV segment is projected onto the paired UV
    segment with a bounded scalar search. A position gap above tolerance is
    excluded from the normal statistic and explicitly counted. Coincidence
    alone does not prove that trimming, solid topology or thickness is valid.
    """
    if maximum_samples < 1 or samples_per_segment < 1 or not 0 < source_crease_angle < 90:
        raise ValueError('Require positive sample budgets and a crease angle between 0 and 90 degrees')
    coordinates = np.vstack([layout.vertices for layout in layouts]) if layouts else np.zeros((1, 3))
    scale = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    tolerance = max(1e-9, 1e-7 * scale) if position_tolerance is None else float(position_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Require a positive physical coincidence tolerance')
    owners, protected = _source_segments(layouts)
    categories = {name: {'segment_count': 0, 'sampled_segment_count': 0, 'angles': [], 'gaps': [],
                         'noncoincident_samples': 0, 'undefined_normal_samples': 0}
                  for name in ('artificial_seam', 'protected_mechanical_crease', 'inferred_source_crease')}
    candidates, unpaired, worst_artificial = [], 0, []
    for edge, sides in sorted(owners.items()):
        if len(sides) != 2 or sides[0][0] == sides[1][0]:
            unpaired += 1
            continue
        source_dot = None if any(side[2] is None for side in sides) else abs(sides[0][2] @ sides[1][2])
        kind = ('protected_mechanical_crease' if edge in protected else
                'inferred_source_crease' if source_dot is not None and source_dot < np.cos(np.deg2rad(source_crease_angle)) else
                'artificial_seam')
        categories[kind]['segment_count'] += 1
        candidates.append((kind, sides))
    segment_budget = max(1, maximum_samples // samples_per_segment)
    chosen = (np.arange(len(candidates)) if len(candidates) <= segment_budget else
              np.unique(np.linspace(0, len(candidates)-1, segment_budget).astype(int)))
    remaining = maximum_samples
    for selection in chosen:
        kind, sides = candidates[selection]
        row = categories[kind]
        row['sampled_segment_count'] += 1
        count = min(samples_per_segment, remaining)
        remaining -= count
        first, second = sides
        surfaces = [layouts[side[0]].nurbs_surface for side in sides]
        endpoints = [np.asarray(layouts[side[0]].parameter_coords)[np.asarray(side[1])] for side in sides]
        sites = (np.arange(count) + .5) / count
        first_uv = endpoints[0][0] + sites[:, None] * np.diff(endpoints[0], axis=0)
        point, du, dv = _evaluate(surfaces[0], first_uv, second=False)
        first_normal, _, first_valid = _normals(du, dv)
        # Coarse bracketing prevents a nonmonotone boundary from silently using
        # a remote local minimum; the final result still has a position gate.
        coarse_t = np.linspace(0., 1., 9)
        second_direction = endpoints[1][1] - endpoints[1][0]
        coarse_uv = endpoints[1][0] + coarse_t[:, None] * second_direction
        coarse_point = _evaluate(surfaces[1], coarse_uv, second=False)[0]
        for index, target in enumerate(point):
            def error(t):
                current = _evaluate(surfaces[1], endpoints[1][0] + float(t) * second_direction, second=False)[0][0]
                return float(np.sum((current - target) ** 2))
            closest = int(np.argmin(np.linalg.norm(coarse_point - target, axis=1)))
            lower, upper = coarse_t[max(0, closest-1)], coarse_t[min(8, closest+1)]
            result = minimize_scalar(error, bounds=(lower, upper), method='bounded',
                                     options={'xatol': 1e-12, 'maxiter': 40})
            best = min((float(result.x), lower, upper), key=error)
            paired_uv = endpoints[1][0] + best * second_direction
            paired_point, paired_du, paired_dv = _evaluate(surfaces[1], paired_uv, second=False)
            gap = float(np.linalg.norm(paired_point[0] - target))
            row['gaps'].append(gap)
            second_normal, _, second_valid = _normals(paired_du, paired_dv)
            if not first_valid[index] or not second_valid[0]:
                row['undefined_normal_samples'] += 1
            elif gap > tolerance:
                row['noncoincident_samples'] += 1
            else:
                angle = float(np.degrees(np.arccos(np.clip(abs(first_normal[index] @ second_normal[0]), 0., 1.))))
                row['angles'].append(angle)
                if kind == 'artificial_seam':
                    worst_artificial.append({'layout_ids': [int(layouts[side[0]].layout_id) for side in sides],
                        'position_mm': target.tolist(), 'paired_position_mm': paired_point[0].tolist(),
                        'UV_by_layout': [first_uv[index].tolist(), paired_uv.tolist()],
                        'physical_gap_mm': gap, 'tangent_plane_angle_degrees': angle})
                    worst_artificial.sort(key=lambda item: item['tangent_plane_angle_degrees'], reverse=True)
                    del worst_artificial[8:]
    for row in categories.values():
        angles = row.pop('angles')
        row['sampled_tangent_plane_jump_degrees'] = _summary(angles)
        row['samples_over_1_degree'] = int(np.count_nonzero(np.asarray(angles) > 1.))
        row['samples_over_5_degrees'] = int(np.count_nonzero(np.asarray(angles) > 5.))
        row['closest_position_gap_mm'] = _summary(row.pop('gaps'))
    return {'method': 'Analytic tangent planes; bounded physical closest-point pairing of shared source boundary segments',
            'angle_definition': 'Acute angle between unoriented physical tangent planes',
            'physical_coincidence_tolerance_mm': tolerance,
            'sample_budget': int(maximum_samples), 'sampled_count': int(maximum_samples - remaining),
            'unpaired_or_nonmanifold_source_segment_count': unpaired,
            'protected_edge_classification': 'Explicit protected source crease edges only',
            'source_crease_classification': 'Inferred from adjacent source triangle normals; can include source mesh noise',
            'inferred_source_crease_angle_degrees': float(source_crease_angle),
            'worst_artificial_seam_samples': worst_artificial,
            'categories': categories,
            'warning': 'Finite sampling and local 1D projection can miss peaks or other coincident branches; no global G1 proof'}


def inspect_network_quality(layouts, *, maximum_patch_samples=225, maximum_seam_samples=1200,
                            seam_samples_per_segment=3, seam_position_tolerance=None,
                            source_crease_angle=30., maximum_patch_controls=8192):
    """Return bounded, serializable waviness diagnostics without changing CAD."""
    patches, errors, internal, unsupported_internal = [], [], [], []
    for layout in layouts:
        try:
            report = inspect_patch_quality(layout, maximum_samples=maximum_patch_samples,
                                           maximum_controls=maximum_patch_controls)
            try:
                continuity = inspect_internal_spline_continuity(layout.nurbs_surface)
                # Keep compact line statistics; the established detailed inspector
                # remains available for locating each individual C0 knot line.
                lines = continuity.pop('lines')
                angles = [line['maximum_sampled_tangent_plane_jump_degrees'] for line in lines
                          if line['maximum_sampled_tangent_plane_jump_degrees'] is not None]
                report['internal_spline_lines'] = continuity
                report['internal_spline_lines']['per_line_maximum_angle_degrees'] = _summary(angles)
                internal.extend(angles)
            except ValueError as exc:
                report['internal_spline_lines'] = {'unsupported_or_invalid': str(exc)}
                unsupported_internal.append({'layout_id': int(layout.layout_id), 'reason': str(exc)})
            patches.append(report)
        except (ValueError, TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as exc:
            errors.append({'layout_id': int(layout.layout_id), 'reason': str(exc)})
    try:
        shared = inspect_shared_seams(layouts, maximum_samples=maximum_seam_samples,
            samples_per_segment=seam_samples_per_segment, position_tolerance=seam_position_tolerance,
            source_crease_angle=source_crease_angle)
    except (ValueError, TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as exc:
        shared = {'unsupported_or_invalid': str(exc)}
    return {'method': 'Sampled physical spline-network diagnostics', 'geometry_changed': False,
            'maximum_patch_controls': int(maximum_patch_controls),
            'global_G1_or_G2_certified': False, 'patch_count': int(len(layouts)),
            'measured_patch_count': len(patches), 'invalid_or_unsupported_patches': errors,
            'internal_C0_representation_lines': {'line_count': len(internal),
                'unsupported_or_invalid_charts': unsupported_internal,
                'per_line_maximum_physical_angle_degrees': _summary(internal),
                'lines_over_5_degrees': int(np.count_nonzero(np.asarray(internal) > 5.)),
                'meaning': 'C0 knot multiplicity permits a crease; only sampled nonzero physical angles indicate one'},
            'shared_seams': shared, 'patches': patches,
            'interpretation': 'Protected edges excluded from artificial-seam statistics; curvature variation alone is not a defect'}


def inspect_mesh_normal_thickness(vertices, triangles, *, maximum_rays=256):
    """Sample inward normal chord lengths of a closed, oriented surface mesh.

    This is not minimum wall thickness: finite area-distributed rays can miss
    narrow necks, oblique walls, small cavities and thickness away from sampled
    points. Results also depend on tessellation. No strength claim is implied.
    Duplicate CAD face vertices are welded at a small scale-relative tolerance;
    the method refuses an open, inconsistent or negatively oriented mesh.
    """
    if maximum_rays < 1:
        raise ValueError('Require a positive ray budget')
    import trimesh
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkOBBTree

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    span = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 0.
    report = {'method': 'Area-distributed face-centroid inward rays on a welded triangle mesh',
              'maximum_rays': int(maximum_rays), 'global_minimum_thickness_certified': False,
              'warning': 'Sampled normal chord length, not minimum wall thickness or a strength verification; tessellation dependent'}
    if not len(triangles) or not np.isfinite(vertices).all() or span <= 0:
        return dict(report, accepted_for_sampling=False, reason='Empty or invalid mesh geometry')
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=triangles.copy(), process=False)
    weld = max(1e-12, span*1e-10)
    digits = max(0, int(np.ceil(-np.log10(weld))))
    mesh.merge_vertices(digits_vertex=digits)
    report.update(weld_rounding_decimal_places=digits, triangle_count=int(len(mesh.faces)),
                  closed_mesh=bool(mesh.is_watertight), consistent_winding=bool(mesh.is_winding_consistent))
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        return dict(report, accepted_for_sampling=False, reason='Require a closed, consistently outward-oriented positive-volume mesh')
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    cells = vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(
        np.c_[np.full(len(mesh.faces), 3), mesh.faces].astype(np.int64).ravel(), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    locator = vtkOBBTree()
    locator.SetDataSet(poly)
    locator.SetTolerance(max(1e-12, span*1e-10))
    locator.BuildLocator()
    cumulative = np.cumsum(mesh.area_faces)
    count = min(maximum_rays, len(mesh.faces))
    selected = (np.arange(len(mesh.faces)) if len(mesh.faces) <= maximum_rays else
                np.searchsorted(cumulative, (np.arange(count)+.5) * cumulative[-1]/count))
    # Area stratification can select a large triangle repeatedly; count once.
    selected = np.unique(selected)
    epsilon = max(1e-10, span*1e-8)
    thickness, missing = [], 0
    samples = []
    for index in selected:
        origin = mesh.triangles_center[index]
        direction = -mesh.face_normals[index]
        hits, hit_ids = vtkPoints(), vtkIdList()
        locator.IntersectWithLine(origin+epsilon*direction, origin+2*span*direction, hits, hit_ids)
        distances = [float(np.dot(np.asarray(hits.GetPoint(i))-origin, direction)) for i in range(hits.GetNumberOfPoints())]
        distances = [distance for distance in distances if distance > 2*epsilon]
        if not distances:
            missing += 1
            continue
        distance = min(distances)
        thickness.append(distance)
        samples.append({'source_face_index': int(index), 'position_mm': origin.tolist(),
                        'inward_normal_chord_mm': distance})
    return dict(report, accepted_for_sampling=True, sampled_ray_count=int(len(selected)),
                rays_without_an_opposite_hit=missing,
                sampled_normal_chord_length_mm=_summary(thickness), samples=samples,
                inward_offset_mm=epsilon)


def _fixed_source_samples(layout, count):
    """Area-stratified XYZ barycentric sites, independent of UV or spline knots."""
    xyz = np.asarray(layout.vertices, dtype=float)
    faces = np.asarray(layout.triangles, dtype=int)
    cells = xyz[faces]
    cross = np.cross(cells[:, 1]-cells[:, 0], cells[:, 2]-cells[:, 0])
    areas = .5*np.linalg.norm(cross, axis=1)
    valid = np.flatnonzero(areas > 0.)
    if not len(valid):
        raise ValueError('Source chart has no positive-area triangle')
    cumulative = np.cumsum(areas[valid])
    triangle_ids = valid[np.searchsorted(cumulative, (np.arange(count)+.5)*cumulative[-1]/count)]
    # Fixed irrational sequences prevent repeated selections of a large source
    # triangle from repeatedly probing its centroid. No random generator/state.
    first = np.mod((np.arange(count)+.5)*.6180339887498949, 1.)
    second = np.mod((np.arange(count)+.5)*.4142135623730951, 1.)
    root = np.sqrt(first)
    barycentric = np.column_stack((1.-root, root*(1.-second), root*second))
    positions = np.einsum('ni,nic->nc', barycentric, cells[triangle_ids])
    normals = cross[triangle_ids]/(2.*areas[triangle_ids, None])
    return triangle_ids, barycentric, positions, normals, float(cumulative[-1])


def _curvature_and_normals(surface, uv, source_normals):
    point, du, dv, duu, duv, dvv = _evaluate(surface, uv)
    normal, _, valid = _normals(du, dv)
    normal *= np.where(np.einsum('ij,ij->i', normal, source_normals) >= 0., 1., -1.)[:, None]
    e1, f1, g1 = (np.einsum('ij,ij->i', a, b) for a, b in ((du, du), (du, dv), (dv, dv)))
    e2, f2, g2 = (np.einsum('ij,ij->i', normal, d) for d in (duu, duv, dvv))
    determinant = e1*g1-f1*f1
    valid &= determinant > np.maximum(np.finfo(float).tiny, 1e-12*e1*g1)
    mean, gaussian = np.full(len(uv), np.nan), np.full(len(uv), np.nan)
    np.divide(e2*g1-2*f2*f1+g2*e1, 2*determinant, out=mean, where=valid)
    np.divide(e2*g2-f2*f2, determinant, out=gaussian, where=valid)
    root = np.sqrt(np.maximum(0., mean*mean-gaussian))
    curvature = np.maximum(np.abs(mean+root), np.abs(mean-root))
    return point, normal, curvature, valid


def _native_sample_visibility(boundary, point, normal, distance_tolerance, normal_tolerance):
    """Use trimmed final faces, not solid distance, and verify the tangent."""
    import cadquery as cq
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_IsInFace
    vertex = cq.Vertex.makeVertex(*point)
    distance = BRepExtrema_DistShapeShape(boundary.wrapped, vertex.wrapped)
    if not distance.IsDone() or distance.NbSolution() == 0:
        return {'visible_with_matching_tangent': False, 'reason': 'Native boundary distance failed'}
    gap = float(distance.Value())
    if gap > distance_tolerance:
        return {'visible_with_matching_tangent': False, 'native_boundary_distance_mm': gap,
                'reason': 'Fitted support point is covered or trimmed away from the final boundary'}
    angles = []
    for index in range(1, distance.NbSolution()+1):
        if distance.SupportTypeShape1(index) != BRepExtrema_IsInFace:
            continue
        face = cq.Shape.cast(distance.SupportOnShape1(index))
        closest = distance.PointOnShape1(index)
        native_normal = np.asarray(face.normalAt(cq.Vector(closest.X(), closest.Y(), closest.Z())).toTuple())
        angles.append(float(np.degrees(np.arccos(np.clip(abs(native_normal@normal), 0., 1.)))))
    aligned = bool(angles and min(angles) <= normal_tolerance)
    return {'visible_with_matching_tangent': aligned, 'native_boundary_distance_mm': gap,
            'minimum_native_tangent_plane_difference_degrees': min(angles) if angles else None,
            'reason': ('Final native face and tangent agree' if aligned else
                       'Coincident point has a different final tangent or lies only on an edge/vertex')}


def _project_source_sites(surface, seed, targets, maximum_iterations):
    """Bounded local Gauss-Newton projection; no global closest-point claim."""
    uv = np.asarray(seed, dtype=float).copy()
    lower = np.array([surface.knotvector_u[surface.degree_u], surface.knotvector_v[surface.degree_v]])
    upper = np.array([surface.knotvector_u[-surface.degree_u-1], surface.knotvector_v[-surface.degree_v-1]])
    initial_error = np.linalg.norm(_evaluate(surface, uv, second=False)[0]-targets, axis=1)
    unchanged = np.zeros(len(uv), dtype=bool)
    for _ in range(maximum_iterations):
        point, du, dv = _evaluate(surface, uv, second=False)
        residual = targets-point
        e, f, g = (np.einsum('ij,ij->i', a, b) for a, b in ((du, du), (du, dv), (dv, dv)))
        regularizer = 1e-10*np.maximum(e+g, np.finfo(float).tiny)
        e, g = e+regularizer, g+regularizer
        determinant = e*g-f*f
        ru, rv = np.einsum('ij,ij->i', residual, du), np.einsum('ij,ij->i', residual, dv)
        step = np.zeros_like(uv)
        valid = determinant > np.finfo(float).tiny
        np.divide(g*ru-f*rv, determinant, out=step[:, 0], where=valid)
        np.divide(e*rv-f*ru, determinant, out=step[:, 1], where=valid)
        step = np.clip(step, -.125*(upper-lower), .125*(upper-lower))
        old_error = np.einsum('ij,ij->i', residual, residual)
        best, best_error = uv.copy(), old_error.copy()
        for fraction in (1., .5, .25):
            proposal = np.clip(uv+fraction*step, lower, upper)
            difference = _evaluate(surface, proposal, second=False)[0]-targets
            error = np.einsum('ij,ij->i', difference, difference)
            improves = error < best_error
            best[improves], best_error[improves] = proposal[improves], error[improves]
        unchanged = np.linalg.norm(best-uv, axis=1) <= 1e-10*np.linalg.norm(upper-lower)
        uv = best
        if unchanged.all():
            break
    final_error = np.linalg.norm(_evaluate(surface, uv, second=False)[0]-targets, axis=1)
    return uv, {'method': 'Seeded local analytic Gauss-Newton with bounded UV steps and three-way decrease search',
        'maximum_iterations': maximum_iterations, 'global_closest_point_certified': False,
        'initial_distance_to_fixed_source_mm': _summary(initial_error),
        'final_distance_to_fixed_source_mm': _summary(final_error),
        'sites_with_final_UV_step_below_tolerance': int(np.count_nonzero(unchanged))}


def inspect_knot_span_curvature(surface, *, maximum_samples=128):
    """Independent bounded span probes, including small spans UV grids miss.

    One quarter of the cell budget prioritizes the smallest parametric cells;
    the rest follows a deterministic area-stratified subset. Three diagonal
    interior sites probe each chosen cell. This can still miss unsampled peaks.
    Different knot bases have different sites, so this is not a paired quality
    comparison and its peak alone must not reject a refit. It also describes
    untrimmed support geometry, which may be hidden in a completed native CAD.
    """
    if maximum_samples < 3:
        raise ValueError('Require at least three knot-span samples')
    axes = [np.unique(surface.knotvector_u), np.unique(surface.knotvector_v)]
    cells = [(a, b, c, d) for a, b in zip(axes[0][:-1], axes[0][1:])
             for c, d in zip(axes[1][:-1], axes[1][1:]) if b > a and d > c]
    if not cells:
        raise ValueError('Surface has no nonempty knot-span cells')
    cells = np.asarray(cells, dtype=float)
    area = (cells[:, 1]-cells[:, 0])*(cells[:, 3]-cells[:, 2])
    budget = max(1, maximum_samples//3)
    if len(cells) <= budget:
        chosen = np.arange(len(cells))
    else:
        small_count = max(1, budget//4)
        small = np.argsort(area, kind='stable')[:small_count]
        remaining = np.setdiff1d(np.arange(len(cells)), small)
        cumulative = np.cumsum(area[remaining])
        sites = (np.arange(budget-small_count)+.5)*cumulative[-1]/(budget-small_count)
        other = remaining[np.searchsorted(cumulative, sites)]
        chosen = np.unique(np.r_[small, other])
    values = np.array([.2113248654051871, .5, .7886751345948129])
    intervals = cells[chosen]
    u = intervals[:, 0, None]+values*(intervals[:, 1]-intervals[:, 0])[:, None]
    v = intervals[:, 2, None]+values*(intervals[:, 3]-intervals[:, 2])[:, None]
    uv = np.column_stack((u.ravel(), v.ravel()))
    result = _curvature_and_normals(surface, uv, np.tile([0., 0., 1.], (len(uv), 1)))
    valid = result[3] & np.isfinite(result[2])
    peak = int(np.nanargmax(np.where(valid, result[2], np.nan))) if valid.any() else None
    return {'method': 'Three analytic curvature probes per selected knot-span cell; small-span priority plus UV-area stratification',
        'maximum_samples': maximum_samples, 'sample_count': len(uv), 'total_span_cell_count': len(cells),
        'sampled_span_cell_count': len(chosen), 'undefined_sample_count': int(np.count_nonzero(~valid)),
        'maximum_absolute_principal_curvature_per_mm': _summary(result[2][valid]),
        'maximum_sampled_curvature_location': (None if peak is None else
            {'position_mm': result[0][peak].tolist(), 'UV': uv[peak].tolist(),
             'maximum_absolute_principal_curvature_per_mm': float(result[2][peak])}),
        'untrimmed_support_geometry_only': True, 'used_for_paired_admission': False,
        'warning': 'Knot-dependent sample sites are independent peak warnings; narrow unsampled wiggles can remain and peaks may be covered in final native CAD'}


def compare_network_ripple_quality(reference_layouts, candidate_layouts, *,
        maximum_samples_per_patch=256, maximum_patch_controls=8192, neighbors=24,
        maximum_source_normal_spread_degrees=35., oscillation_growth_factor=1.75,
        minimum_oscillation_increase_degrees=1., minimum_regressed_patch_fraction=.1,
        maximum_regressed_source_area_fraction=.005, severe_oscillation_degrees=8.,
        candidate_native_shape=None, native_boundary_tolerance_mm=1e-5,
        native_normal_tolerance_degrees=5., maximum_native_visibility_samples=128,
        passive_solid_mask=None, passive_solid_bounds=None, maximum_projection_iterations=6,
        maximum_knot_span_samples_per_patch=128):
    """Compare sampled normal oscillation at matching physical source regions.

    Both surfaces are evaluated at the same source-triangle barycentric sites,
    using their respective source-to-UV maps. This avoids comparing unrelated
    uniform UV grids after a clock change. It is a source correspondence, not
    an assertion that the two surfaces pass through identical XYZ positions.

    Local oscillation is a leave-one-out quadratic normal-field prediction
    residual in a fixed physical source neighborhood. Smooth planes, spheres
    and cylindrical bends have little residual; short-wavelength ripples can
    have a large residual. This is a scale-dependent engineering diagnostic,
    not a mathematical classification of design intent. Protected source edges,
    sharp source neighborhoods, degenerate and insufficient samples are omitted
    explicitly. Physical curvature changes alone never reject a candidate.

    A pre-integration support chart can be covered by an exact restored collar
    or trimmed later. Consequently a detected regression is only preliminary
    unless its worst sites are verified on the final trimmed native boundary
    with the same tangent plane. Without native geometry, ``accepted`` is None
    for a preliminary regression, not a hard rejection. An optional passive
    solid grid conservatively masks protected source regions before integration.

    Candidate-only uniform/knot probes should still be inspected separately:
    matched samples can miss a narrow new ripple. Default thresholds are
    conservative engineering choices, exposed here rather than shape-specific.
    """
    from scipy.spatial import cKDTree

    if (maximum_samples_per_patch < 32 or neighbors < 12
            or neighbors > maximum_samples_per_patch or oscillation_growth_factor <= 1.
            or minimum_oscillation_increase_degrees <= 0.
            or not 0. < minimum_regressed_patch_fraction <= 1.
            or not 0. <= maximum_regressed_source_area_fraction <= 1.
            or not 0. < maximum_source_normal_spread_degrees < 90.
            or maximum_projection_iterations < 1 or maximum_native_visibility_samples < 0
            or native_boundary_tolerance_mm <= 0. or not 0. < native_normal_tolerance_degrees < 90.
            or severe_oscillation_degrees <= 0.):
        raise ValueError('Require valid positive sampling budgets and explicit ripple thresholds')
    references = {int(layout.layout_id): layout for layout in reference_layouts}
    candidates = {int(layout.layout_id): layout for layout in candidate_layouts}
    rows, unsupported = [], []
    total_area, regressed_area = 0., 0.
    all_before, all_after = [], []
    for layout_id, reference in references.items():
        candidate = candidates.get(layout_id)
        item = {'layout_id': layout_id}
        try:
            if candidate is None:
                raise ValueError('Candidate does not contain this source chart')
            if (not np.array_equal(reference.vertices, candidate.vertices)
                    or not np.array_equal(reference.triangles, candidate.triangles)):
                raise ValueError('Require identical source XYZ vertices and triangle indexing for physical correspondence')
            if any(layout.nurbs_surface.ctrlpts_size_u*layout.nurbs_surface.ctrlpts_size_v > maximum_patch_controls
                   for layout in (reference, candidate)):
                raise ValueError('Explicit diagnostic control-count budget exceeded')
            ids, bary, source, source_normal, area = _fixed_source_samples(reference, maximum_samples_per_patch)
            passive = np.zeros(len(source), dtype=bool)
            if passive_solid_mask is not None:
                mask = np.asarray(passive_solid_mask, dtype=bool)
                bounds = passive_solid_bounds
                if mask.ndim != 3 or bounds is None:
                    raise ValueError('A passive solid grid requires 3D mask and explicit XYZ bounds')
                low, high = (bounds['min'], bounds['max']) if isinstance(bounds, dict) else bounds
                low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
                if not np.isfinite([low, high]).all() or np.any(high <= low):
                    raise ValueError('Passive solid bounds must have finite positive XYZ spans')
                coordinates = (source-low)/(high-low)
                inside = np.all((coordinates >= 0.) & (coordinates <= 1.), axis=1)
                cells = np.minimum(np.floor(np.clip(coordinates, 0., 1.)*np.asarray(mask.shape)).astype(int), np.asarray(mask.shape)-1)
                passive[inside] = mask[tuple(cells[inside].T)]
            source_faces = np.asarray(reference.triangles, dtype=int)[ids]
            seeds = [np.einsum('ni,nic->nc', bary, np.asarray(layout.parameter_coords)[source_faces])
                     for layout in (reference, candidate)]
            projected = [_project_source_sites(layout.nurbs_surface, seed, source, maximum_projection_iterations)
                         for layout, seed in zip((reference, candidate), seeds)]
            uv = [item[0] for item in projected]
            evaluated = [_curvature_and_normals(layout.nurbs_surface, sites, source_normal)
                         for layout, sites in zip((reference, candidate), uv)]
            gaps = np.linalg.norm(evaluated[0][0]-evaluated[1][0], axis=1)
            source_radius, nearest = cKDTree(source).query(source, k=neighbors)
            # Remove the center sample; the prediction must not fit itself.
            nearest = nearest[:, 1:]
            radius = source_radius[:, -1]
            before, after = np.full(len(source), np.nan), np.full(len(source), np.nan)
            exclusions = defaultdict(int)
            protected = list(getattr(reference, 'protected_source_crease_edges', ()))
            protected += list(getattr(candidate, 'protected_source_crease_edges', ()))
            for site, adjacent in enumerate(nearest):
                if passive[site] and candidate_native_shape is None:
                    exclusions['passive_solid_region_without_native_visibility'] += 1
                    continue
                if not all(result[3][np.r_[site, adjacent]].all() for result in evaluated):
                    exclusions['undefined_surface_derivatives'] += 1
                    continue
                source_dot = np.clip(source_normal[adjacent]@source_normal[site], -1., 1.)
                if np.max(np.degrees(np.arccos(source_dot))) > maximum_source_normal_spread_degrees:
                    exclusions['sharp_or_folded_source_neighborhood'] += 1
                    continue
                blocked = False
                for edge in protected:
                    a, b = np.asarray(edge, dtype=float)
                    direction = b-a
                    length_squared = float(direction@direction)
                    t = np.clip(float((source[site]-a)@direction)/length_squared, 0., 1.) if length_squared else 0.
                    if np.linalg.norm(source[site]-(a+t*direction)) < .5*radius[site]:
                        blocked = True
                        break
                if blocked:
                    exclusions['near_explicit_protected_source_edge'] += 1
                    continue
                offsets = source[adjacent]-source[site]
                _, singular, frame = np.linalg.svd(offsets, full_matrices=False)
                if singular[1] <= max(np.finfo(float).tiny, 1e-7*singular[0]) or radius[site] <= 0:
                    exclusions['degenerate_physical_neighborhood'] += 1
                    continue
                local = offsets@frame[:2].T/radius[site]
                x, y = local.T
                design = np.column_stack((np.ones(len(local)), x, y, x*x, x*y, y*y))
                weight = np.exp(-np.linalg.norm(offsets, axis=1)**2/(radius[site]**2))
                design_weighted = design*np.sqrt(weight[:, None])
                if np.linalg.matrix_rank(design_weighted) < 6:
                    exclusions['rank_deficient_normal_field_fit'] += 1
                    continue
                for result, output in zip(evaluated, (before, after)):
                    coefficient = np.linalg.lstsq(design_weighted,
                        result[1][adjacent]*np.sqrt(weight[:, None]), rcond=None)[0]
                    prediction = coefficient[0]
                    norm = np.linalg.norm(prediction)
                    if norm > 1e-12:
                        output[site] = np.degrees(np.arccos(np.clip(
                            float(prediction@result[1][site])/norm, -1., 1.)))
            valid = np.isfinite(before)&np.isfinite(after)
            bad = valid & (after > before*oscillation_growth_factor+minimum_oscillation_increase_degrees)
            eligible_count, bad_count = int(np.count_nonzero(valid)), int(np.count_nonzero(bad))
            fraction = bad_count/max(eligible_count, 1)
            first, second = _summary(before[valid]), _summary(after[valid])
            p95_regression = bool(eligible_count >= 24 and second['p95'] >
                                 first['p95']*oscillation_growth_factor+minimum_oscillation_increase_degrees)
            regressed = bool(p95_regression and fraction >= minimum_regressed_patch_fraction)
            severe = bool(regressed and second['p95'] > severe_oscillation_degrees)
            sampled_regressed_area = area*bad_count/maximum_samples_per_patch if regressed else 0.
            total_area += area
            regressed_area += sampled_regressed_area
            all_before.extend(before[valid])
            all_after.extend(after[valid])
            worst = np.flatnonzero(bad)
            worst = worst[np.argsort(after[worst]-before[worst])[::-1]][:4]
            item.update(source_area_mm_squared=area, matched_source_site_count=len(source),
                candidate_independent_knot_span_probe=inspect_knot_span_curvature(candidate.nurbs_surface,
                    maximum_samples=maximum_knot_span_samples_per_patch),
                reference_source_projection=projected[0][1], candidate_source_projection=projected[1][1],
                passive_solid_region_sample_count=int(np.count_nonzero(passive)),
                eligible_oscillation_sample_count=eligible_count, excluded_samples=dict(exclusions),
                source_neighborhood_radius_mm=_summary(radius),
                evaluated_surface_position_difference_mm=_summary(gaps),
                reference_normal_oscillation_degrees=first, candidate_normal_oscillation_degrees=second,
                regressed_sample_count=bad_count, regressed_eligible_sample_fraction=fraction,
                sampled_ripple_regression=regressed, severe_sampled_ripple_regression=severe,
                sampled_regressed_source_area_mm_squared=sampled_regressed_area,
                reference_matched_absolute_principal_curvature_per_mm=_summary(evaluated[0][2]),
                candidate_matched_absolute_principal_curvature_per_mm=_summary(evaluated[1][2]),
                worst_regressions=[{'source_position_mm': source[index].tolist(),
                    'reference_position_mm': evaluated[0][0][index].tolist(),
                    'candidate_position_mm': evaluated[1][0][index].tolist(),
                    'reference_UV': uv[0][index].tolist(), 'candidate_UV': uv[1][index].tolist(),
                    'candidate_unit_normal': evaluated[1][1][index].tolist(),
                    'reference_oscillation_degrees': float(before[index]),
                    'candidate_oscillation_degrees': float(after[index])} for index in worst])
            rows.append(item)
        except (ValueError, TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as exc:
            unsupported.append(dict(item, reason=str(exc)))
    fraction = regressed_area/max(total_area, np.finfo(float).tiny)
    severe_ids = [row['layout_id'] for row in rows if row['severe_sampled_ripple_regression']]
    preliminary_ids = [row['layout_id'] for row in rows if row['sampled_ripple_regression']]
    preliminary_accepted = bool(not unsupported and rows and not severe_ids and fraction <= maximum_regressed_source_area_fraction)
    checked = 0
    if candidate_native_shape is not None:
        import cadquery as cq
        shape = candidate_native_shape.val() if hasattr(candidate_native_shape, 'val') else candidate_native_shape
        boundary = cq.Compound.makeCompound(shape.Faces())
        for row in sorted(rows, key=lambda item: item['candidate_normal_oscillation_degrees']['p95'] or 0., reverse=True):
            if not row['sampled_ripple_regression']:
                continue
            visibility = []
            for site in row['worst_regressions']:
                if checked >= maximum_native_visibility_samples:
                    break
                value = _native_sample_visibility(boundary, site['candidate_position_mm'],
                    np.asarray(site['candidate_unit_normal']), native_boundary_tolerance_mm,
                    native_normal_tolerance_degrees)
                visibility.append(dict(source_position_mm=site['source_position_mm'], **value))
                checked += 1
            row['final_native_visibility'] = visibility
            row['confirmed_visible_regression_sample_count'] = sum(item['visible_with_matching_tangent'] for item in visibility)
    visible_ids = [row['layout_id'] for row in rows
                   if row.get('confirmed_visible_regression_sample_count', 0) >= 2]
    visible_severe_ids = [layout_id for layout_id in severe_ids if layout_id in visible_ids]
    visible_area = sum(row['sampled_regressed_source_area_mm_squared'] for row in rows if row['layout_id'] in visible_ids)
    unchecked = [row['layout_id'] for row in rows if row['sampled_ripple_regression']
                 and candidate_native_shape is not None and not row.get('final_native_visibility')]
    eligible_total = sum(row['eligible_oscillation_sample_count'] for row in rows)
    if unsupported or not rows or eligible_total == 0:
        accepted, status = None, 'insufficient_or_unsupported_source_correspondence'
    elif not preliminary_ids:
        accepted, status = True, 'no_sampled_ripple_regression'
    elif candidate_native_shape is None:
        accepted, status = None, 'preliminary_regression_requires_final_native_visibility'
    elif unchecked:
        accepted, status = None, 'native_visibility_budget_exhausted_for_regressed_regions'
    else:
        accepted = bool(not visible_severe_ids and visible_area/max(total_area, np.finfo(float).tiny) <= maximum_regressed_source_area_fraction)
        status = 'sampled_visible_regression_rejected' if not accepted else 'no_rejecting_visible_regression_in_bounded_check'
    return {'method': 'Fixed source-XYZ barycentric correspondence and leave-one-out physical quadratic normal-field residual',
        'accepted': accepted, 'admission_status': status, 'preliminary_source_scope_accepted': preliminary_accepted,
        'geometry_changed': False, 'no_global_smoothness_or_design_intent_certificate': True,
        'thresholds': {'oscillation_growth_factor': oscillation_growth_factor,
            'minimum_oscillation_increase_degrees': minimum_oscillation_increase_degrees,
            'minimum_regressed_patch_fraction': minimum_regressed_patch_fraction,
            'maximum_regressed_source_area_fraction': maximum_regressed_source_area_fraction,
            'severe_oscillation_degrees': severe_oscillation_degrees,
            'maximum_source_normal_spread_degrees': maximum_source_normal_spread_degrees},
        'sampling': {'maximum_sites_per_patch': maximum_samples_per_patch,
            'maximum_controls_per_patch': maximum_patch_controls, 'physical_nearest_neighbors': neighbors,
            'maximum_source_projection_iterations': maximum_projection_iterations,
            'maximum_independent_knot_span_probes_per_patch': maximum_knot_span_samples_per_patch},
        'measured_patch_count': len(rows), 'unsupported_source_correspondence_or_patches': unsupported,
        'source_area_mm_squared': total_area, 'sampled_regressed_source_area_mm_squared': regressed_area,
        'sampled_regressed_source_area_fraction': fraction,
        'severe_regression_layout_ids': severe_ids,
        'preliminary_regression_layout_ids': preliminary_ids,
        'confirmed_visible_regression_layout_ids': visible_ids,
        'confirmed_visible_severe_regression_layout_ids': visible_severe_ids,
        'visibility_unchecked_regression_layout_ids': unchecked,
        'native_visibility': {'provided': candidate_native_shape is not None, 'checked_sample_count': checked,
            'maximum_checks': maximum_native_visibility_samples, 'boundary_tolerance_mm': native_boundary_tolerance_mm,
            'tangent_plane_tolerance_degrees': native_normal_tolerance_degrees,
            'method': 'Distance to compound of final trimmed faces plus closest-face tangent agreement',
            'limitation': 'Only bounded worst regressed sites are checked; this is not a global visibility or ripple certificate'},
        'reference_normal_oscillation_degrees': _summary(all_before),
        'candidate_normal_oscillation_degrees': _summary(all_after), 'patches': rows,
        'warning': 'Matched finite samples and source-neighborhood scale can miss narrow wiggles; curvature change alone does not reject; source sharpness exclusions can hide a ripple near a real crease'}


def confirm_network_ripple_visibility(ripple_report, candidate_native_shape, *,
        native_boundary_tolerance_mm=1e-5, native_normal_tolerance_degrees=5.,
        maximum_native_visibility_samples=128):
    """Admit a precomputed matched report against its completed native shape.

    This only rechecks bounded worst-site visibility, not the expensive source
    correspondence, derivatives or local normal-field predictions. The report
    must describe the same private fitted proposal used to build this shape;
    callers must not reuse it after changing that proposal. The returned copy
    retains all sampling limits and preliminary source-region observations.
    """
    from copy import deepcopy
    import cadquery as cq
    if (maximum_native_visibility_samples < 0 or native_boundary_tolerance_mm <= 0.
            or not 0. < native_normal_tolerance_degrees < 90.):
        raise ValueError('Require valid native visibility tolerances and budget')
    report = deepcopy(ripple_report)
    rows = report.get('patches', [])
    shape = candidate_native_shape.val() if hasattr(candidate_native_shape, 'val') else candidate_native_shape
    boundary = cq.Compound.makeCompound(shape.Faces())
    checked = 0
    for row in rows:
        row.pop('final_native_visibility', None)
        row.pop('confirmed_visible_regression_sample_count', None)
    for row in sorted(rows, key=lambda item: item['candidate_normal_oscillation_degrees']['p95'] or 0., reverse=True):
        if not row['sampled_ripple_regression']:
            continue
        visibility = []
        for site in row['worst_regressions']:
            if checked >= maximum_native_visibility_samples:
                break
            value = _native_sample_visibility(boundary, site['candidate_position_mm'],
                np.asarray(site['candidate_unit_normal']), native_boundary_tolerance_mm,
                native_normal_tolerance_degrees)
            visibility.append(dict(source_position_mm=site['source_position_mm'], **value))
            checked += 1
        row['final_native_visibility'] = visibility
        row['confirmed_visible_regression_sample_count'] = sum(item['visible_with_matching_tangent'] for item in visibility)
    visible_ids = [row['layout_id'] for row in rows if row.get('confirmed_visible_regression_sample_count', 0) >= 2]
    visible_severe_ids = [layout_id for layout_id in report['severe_regression_layout_ids'] if layout_id in visible_ids]
    visible_area = sum(row['sampled_regressed_source_area_mm_squared'] for row in rows if row['layout_id'] in visible_ids)
    unchecked = [row['layout_id'] for row in rows if row['sampled_ripple_regression'] and not row.get('final_native_visibility')]
    if (report.get('unsupported_source_correspondence_or_patches') or not rows
            or not sum(row['eligible_oscillation_sample_count'] for row in rows)):
        accepted, status = None, 'insufficient_or_unsupported_source_correspondence'
    elif not report['preliminary_regression_layout_ids']:
        accepted, status = True, 'no_sampled_ripple_regression'
    elif unchecked:
        accepted, status = None, 'native_visibility_budget_exhausted_for_regressed_regions'
    else:
        accepted = bool(not visible_severe_ids and visible_area/max(report['source_area_mm_squared'], np.finfo(float).tiny)
                        <= report['thresholds']['maximum_regressed_source_area_fraction'])
        status = 'sampled_visible_regression_rejected' if not accepted else 'no_rejecting_visible_regression_in_bounded_check'
    report.update(accepted=accepted, admission_status=status,
        confirmed_visible_regression_layout_ids=visible_ids,
        confirmed_visible_severe_regression_layout_ids=visible_severe_ids,
        visibility_unchecked_regression_layout_ids=unchecked,
        estimated_source_region_area_associated_with_visible_worst_sites_mm_squared=visible_area,
        native_visibility={'provided': True, 'checked_sample_count': checked,
            'maximum_checks': maximum_native_visibility_samples, 'boundary_tolerance_mm': native_boundary_tolerance_mm,
            'tangent_plane_tolerance_degrees': native_normal_tolerance_degrees,
            'method': 'Distance to compound of final trimmed faces plus closest-face tangent agreement',
            'reused_matched_correspondence_and_normal_field_report': True,
            'limitation': 'Only bounded worst regressed sites are checked; report must describe this same fitted proposal; no global visibility certificate'})
    return report


# Short alias retained for standalone diagnostic scripts written with the first
# helper name; reconstruction uses the explicit network-scoped public name.
confirm_ripple_visibility = confirm_network_ripple_visibility
