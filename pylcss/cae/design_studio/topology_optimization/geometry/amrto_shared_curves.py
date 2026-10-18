"""Common cubic boundary curves for adjacent sparse CAD patches."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq


def reconcile_roundoff_boundary_knots(curves, *, maximum_geometry_error=1e-9):
    """Reconcile only machine-near opposing knots, with physical error checks.

    Reversing ``1 - t`` twice can produce adjacent floating-point values for
    one mathematical knot. OCCT cannot store such intervals. Distinct knots
    in any one input curve are never merged, and an independently sampled
    physical-curve comparison must pass before using a changed representation.
    """
    curves = list(curves)
    tolerance = 64 * np.finfo(float).eps
    values = sorted(set(float(t) for curve in curves for t in curve.t))
    groups = []
    for value in values:
        if groups and value - groups[-1][0] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    mapping = {}
    for group in groups:
        if len(group) < 2:
            continue
        if any(len(set(curve.t).intersection(group)) > 1 for curve in curves):
            continue  # A real input interval cannot be erased by this treatment.
        representative = 0. if 0. in group else 1. if 1. in group else group[0]
        mapping.update((value, representative) for value in group)
    results, changed, maximum_error = [], 0, 0.
    for curve in curves:
        knots = np.array([mapping.get(float(t), float(t)) for t in curve.t])
        if np.array_equal(knots, curve.t):
            results.append(curve)
            continue
        candidate = BSpline(knots, np.asarray(curve.c).copy(), curve.k)
        sites = np.unique(np.r_[np.linspace(0, 1, 1001), curve.t, knots])
        span = np.unique(np.r_[curve.t, knots])
        sites = np.unique(np.concatenate([sites] + [span[:-1] + fraction * np.diff(span)
                                                   for fraction in (.25, .5, .75)]))
        error = float(np.linalg.norm(candidate(sites) - curve(sites), axis=1).max())
        if not np.isfinite(error) or error > maximum_geometry_error:
            raise ValueError("Machine-near boundary knot reconciliation exceeds its physical geometry limit")
        maximum_error = max(maximum_error, error)
        changed += int(np.count_nonzero(knots != curve.t))
        results.append(candidate)
    return results, {"method": "Machine-roundoff knot reconciliation with independent physical-curve checks",
                     "parameter_tolerance": tolerance, "changed_knot_entry_count": changed,
                     "maximum_sampled_geometry_error": maximum_error,
                     "maximum_allowed_geometry_error": maximum_geometry_error,
                     "distinct_input_knot_intervals_preserved": True}


def _curve_distance_to_polyline(points, polyline):
    distance = np.full(len(points), np.inf)
    for start, end in zip(polyline, polyline[1:]):
        direction = end - start
        squared = float(direction @ direction)
        if squared == 0:
            continue
        position = np.clip((points - start) @ direction / squared, 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - start - position[:, None] * direction, axis=1))
    return distance


def _polyline_curve(points, parameters):
    """Represent every physical polyline turn exactly as cubic Bezier pieces."""
    directions = np.diff(points, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Collinear source samples need no extra C0 knot. Their chord parameters
    # already follow the same affine segment, including unequal edge lengths.
    turns = np.linalg.norm(np.diff(directions, axis=0), axis=1) > 128 * np.finfo(float).eps
    breaks = parameters[1:-1][turns]
    knots = np.r_[np.zeros(4), np.repeat(breaks, 3), np.ones(4)]
    count = len(knots) - 4
    greville = np.array([np.mean(knots[i + 1:i + 4]) for i in range(count)])
    controls = np.column_stack([np.interp(greville, parameters, points[:, i]) for i in range(3)])
    return BSpline(knots, controls, 3)


def simplify_shared_curve(curve, points, tolerance, *, bounds=None, maximum_attempts=64,
                          preserve_endpoint_tangents=False):
    """Remove boundary knots once per shared chain under the source error budget.

    Both incident charts subsequently receive the same curve. Endpoints and
    protected polylines stay exact; patch-level compression still freezes borders.
    """
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    parameters = np.r_[0., np.cumsum(lengths)] / lengths.sum()
    attempts = removed = 0
    current = curve
    while attempts < maximum_attempts:
        changed = False
        for knot in np.unique(current.t[current.k + 1:-current.k - 1]):
            if attempts >= maximum_attempts:
                break
            attempts += 1
            knots = np.delete(current.t, np.flatnonzero(current.t == knot)[0])
            count = len(knots) - current.k - 1
            insertion = BSpline(knots, np.eye(count), current.k).insert_knot(float(knot)).c
            # A fixed corner's surface normal depends on these endpoint
            # tangents. Knot reduction must preserve them as well as positions;
            # an interior surface solve cannot repair an inverted fixed corner.
            controls = np.zeros((count, 3))
            controls[[0, -1]] = current.c[[0, -1]]
            fixed = [0, count - 1]
            if preserve_endpoint_tangents:
                controls[1] = controls[0] + current.derivative()(0.) * (knots[4] - knots[3]) / 3.
                controls[-2] = controls[-1] - current.derivative()(1.) * (knots[-4] - knots[-5]) / 3.
                fixed = [0, 1, count - 2, count - 1]
            free = sorted(set(range(count)) - set(fixed))
            right = current.c - insertion[:, fixed] @ controls[fixed]
            if free:
                controls[free] = lstsq(insertion[:, free], right, lapack_driver='gelsy')[0]
            if bounds is not None and (np.any(controls < bounds[0]) or np.any(controls > bounds[1])):
                continue
            candidate = BSpline(knots, controls, current.k)
            sites = np.unique(np.r_[parameters, current.t, knots,
                                    np.linspace(0., 1., max(40, 5 * len(points))),
                                    np.linspace(0., 1., max(80, 8 * len(points)))])
            source = np.column_stack([np.interp(sites, parameters, points[:, axis]) for axis in range(3)])
            distance = max(float(np.linalg.norm(candidate(sites) - source, axis=1).max()),
                           float(_curve_distance_to_polyline(candidate(sites), points).max()))
            if distance <= tolerance:
                if bounds is not None:
                    from .amrto_shared_stock import CanonicalChainCertificationError, project_canonical_chain
                    try:
                        project_canonical_chain(candidate, points, bounds, tolerance)
                    except CanonicalChainCertificationError:
                        continue
                current = candidate
                removed += 1
                changed = True
        if not changed:
            break
    return current, {'removed_knot_count': removed, 'attempts': attempts,
                     'endpoint_tangents_preserved': bool(preserve_endpoint_tangents),
                     'boundary_simplified_jointly': True, 'control_points': len(current.c)}


def _endpoint_tangents_follow_source(curve, points):
    """Check both directed one-sided source tangents after any stock clipping."""
    edges = np.asarray([points[1] - points[0], points[-1] - points[-2]])
    tangents = np.asarray([curve.derivative()(0.), curve.derivative()(1.)])
    lengths = np.linalg.norm(edges, axis=1) * np.linalg.norm(tangents, axis=1)
    if not np.isfinite(tangents).all() or np.any(lengths <= 0):
        return False
    cosine = np.einsum('ij,ij->i', edges, tangents) / lengths
    return bool(np.all(cosine >= 1. - 1e-10))


def _fit_curve(points, tolerance, *, preserve_polyline=False, bounds=None,
               preserve_endpoint_tangents=False):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (lengths <= 0).any():
        raise ValueError("Shared boundary chain has zero-length edges")
    parameters = np.r_[0, np.cumsum(lengths)]
    parameters /= parameters[-1]
    if preserve_polyline:
        curve = _polyline_curve(points, parameters)
        return curve, {"maximum_sampled_error": 0.0, "control_points": len(curve.c),
                       "endpoint_tangent_directions_preserved": True,
                       "smooth_interior_knots": False, "exact_protected_chain": True,
                       "reason": "Protected source boundary retained by exact polyline degree elevation"}
    sample_t = np.unique(np.r_[parameters, np.linspace(0, 1, max(40, 5 * len(points)))])
    samples = np.column_stack([np.interp(sample_t, parameters, points[:, i]) for i in range(3)])
    best = None
    candidates = sorted(set([4, 6, 10, 16, 24, 40, min(64, 2 * len(points) + 2)]))
    for count in candidates:
        if count > max(4, min(64, 2 * len(points) + 2)):
            continue
        knots = np.r_[np.zeros(4), np.linspace(0, 1, count - 2)[1:-1], np.ones(4)]
        basis = BSpline.design_matrix(sample_t, knots, 3).toarray()
        controls = np.zeros((count, 3))
        controls[0], controls[-1] = points[0], points[-1]
        # Preserve both one-sided source tangent directions. Position-only
        # curve fitting can reverse the corner normal even within its distance
        # budget. Keep adjacent poles on their actual first/last source edges;
        # the positive lengths also avoid creating a stationary endpoint.
        fixed = [0, count - 1]
        if preserve_endpoint_tangents:
            first = min(float(lengths.sum()) * float(knots[4] - knots[3]) / 3., float(lengths[0]))
            last = min(float(lengths.sum()) * float(knots[-4] - knots[-5]) / 3., float(lengths[-1]))
            controls[1] = points[0] + first * (points[1] - points[0]) / lengths[0]
            controls[-2] = points[-1] - last * (points[-1] - points[-2]) / lengths[-1]
            fixed = [0, 1, count - 2, count - 1]
        free = sorted(set(range(count)) - set(fixed))
        residual = samples - basis[:, fixed] @ controls[fixed]
        if free:
            controls[free] = lstsq(basis[:, free], residual, lapack_driver="gelsy")[0]
        if bounds is not None:
            controls = np.clip(controls, bounds[0], bounds[1])
        curve = BSpline(knots, controls, 3)
        if preserve_endpoint_tangents and not _endpoint_tangents_follow_source(curve, points):
            continue
        dense_t = np.linspace(0, 1, max(80, 8 * len(points)))
        distance = max(float(np.linalg.norm(curve(sample_t) - samples, axis=1).max()),
                       float(_curve_distance_to_polyline(curve(dense_t), points).max()))
        best = (curve, distance)
        if distance <= tolerance:
            if bounds is not None:
                from .amrto_shared_stock import CanonicalChainCertificationError, project_canonical_chain
                try:
                    project_canonical_chain(curve, points, bounds, tolerance)
                except CanonicalChainCertificationError:
                    continue
            curve, reduction = simplify_shared_curve(curve, points, tolerance, bounds=bounds,
                preserve_endpoint_tangents=preserve_endpoint_tangents)
            distance = max(float(np.linalg.norm(curve(sample_t) - samples, axis=1).max()),
                           float(_curve_distance_to_polyline(curve(dense_t), points).max()))
            return curve, {"maximum_sampled_error": distance, "control_points": len(curve.c),
                           "endpoint_tangent_directions_preserved": bool(preserve_endpoint_tangents),
                           "smooth_interior_knots": True, 'shared_curve_reduction': reduction}
    # True sharp turns cannot always be approximated within tolerance by a small
    # smooth curve. Preserve their polyline instead of rounding material away.
    curve = _polyline_curve(points, parameters)
    return curve, {"maximum_sampled_error": 0.0,
        "endpoint_tangent_directions_preserved": True,
        "control_points": len(curve.c), "smooth_interior_knots": False,
        "reason": "Smooth curve exceeded source-boundary tolerance",
        "rejected_smooth_error": best[1] if best else None}


def reparameterize_shared_curve(curve, chord_parameters, native_parameters, tolerance=1e-8):
    """Compose a common cubic curve with an exact piecewise affine UV change.

    Changed slopes receive C0 knots before their values are transformed. Each
    polynomial piece therefore keeps its original physical Bezier controls.
    This preserves common curve geometry but can add three controls per source
    boundary breakpoint. It is an extension for retained native rectangles,
    separate from harmonic chord-length boundary parameterization.
    """
    chord = np.asarray(chord_parameters, dtype=float)
    native = np.asarray(native_parameters, dtype=float)
    if (chord.ndim != 1 or native.shape != chord.shape or len(chord) < 2
            or not np.isfinite(chord).all() or not np.isfinite(native).all()
            or (np.diff(chord) <= 0).any() or (np.diff(native) <= 0).any()):
        raise ValueError("Shared-curve reparameterization requires finite strictly increasing side parameters")
    if (curve.k != 3 or abs(chord[0]) > 1e-12 or abs(native[0]) > 1e-12
            or abs(chord[-1] - 1) > 1e-12 or abs(native[-1] - 1) > 1e-12):
        raise ValueError("Shared cubic side parameters must span the unit interval")
    mismatch = float(np.max(np.abs(chord - native)))
    report = {"method": "Exact piecewise affine common-curve reparameterization into native rectangle UV",
              "native_boundary_progress_maximum_difference": mismatch,
              "original_control_count": len(curve.c), "added_slope_breakpoints": 0,
              "applied": False, "production_boundary_extension": True}
    if mismatch <= tolerance:
        return curve, {**report, "control_count": len(curve.c), "skip_tolerance": tolerance}
    slopes = np.diff(native) / np.diff(chord)
    scale = np.maximum(np.maximum(np.abs(slopes[:-1]), np.abs(slopes[1:])), 1.)
    breakpoints = chord[1:-1][np.abs(np.diff(slopes)) > 128 * np.finfo(float).eps * scale]
    refined = curve
    for parameter in breakpoints:
        multiplicity = int(np.count_nonzero(refined.t == parameter))
        if multiplicity < 3:
            refined = refined.insert_knot(float(parameter), 3 - multiplicity)
    knots = np.interp(refined.t, chord, native)
    result = BSpline(knots, np.asarray(refined.c).copy(), 3)
    return result, {**report, "applied": True, "added_slope_breakpoints": len(breakpoints),
                    "control_count": len(result.c),
                    "added_control_count": len(result.c) - len(curve.c)}


def prepare_shared_boundary_curves(layouts, tolerance, *, bounds=None, blocked_junction_positions=(),
                                   certify_stock_hull=False, preserve_endpoint_tangents=False,
                                   tangent_preserving_edges=()):
    """Fit seams once; retain protected source creases and junctions exactly."""
    def key(point):
        return tuple(map(float, point))
    tangent_edges = {tuple(sorted((key(a), key(b)))) for a, b in tangent_preserving_edges}
    tangent_chain_count = 0
    protected_edges = {tuple(sorted((key(edge[0]), key(edge[1]))))
                       for layout in layouts for edge in getattr(layout, "protected_source_crease_edges", ())}
    protected_edges.update(
        tuple(sorted((key(layout.vertices[a]), key(layout.vertices[b]))))
        for layout in layouts if getattr(layout, "require_exact_boundaries", False)
        for path in layout.boundary_paths for a, b in zip(path, path[1:])
    )
    anchors, corner_positions, graph, owners = set(), set(), defaultdict(set), defaultdict(set)
    for owner, layout in enumerate(layouts):
        for path in layout.boundary_paths:
            points = layout.vertices[path]
            endpoints = (key(points[0]), key(points[-1]))
            anchors.update(endpoints)
            corner_positions.update(endpoints)
            for a, b in zip(map(key, points), map(key, points[1:])):
                graph[a].add(b)
                graph[b].add(a)
                owners[tuple(sorted((a, b)))].add(owner)
    cornerless_junctions = set()
    for vertex, neighbors in graph.items():
        multi_patch = (len(neighbors) != 2
                       or len({frozenset(owners[tuple(sorted((vertex, n)))]) for n in neighbors}) != 1)
        if multi_patch or len({tuple(sorted((vertex, n))) in protected_edges for n in neighbors}) != 1:
            anchors.add(vertex)
            if multi_patch and vertex not in corner_positions:
                cornerless_junctions.add(vertex)
        elif len(neighbors) == 2 and any(
                tuple(sorted((vertex, neighbor))) in protected_edges for neighbor in neighbors):
            # Ordinary graph seams follow the faceted quad mesh and can zigzag
            # even on a smooth surface. Anchoring every such turn forces C0
            # knots into the tensor surface; opposing sides then multiply the
            # mandatory control net. Only protected creases need these fixed
            # turn anchors. Other chains still obey _fit_curve's source-distance
            # tolerance, with an exact polyline fallback if smoothing fails.
            a, b = (np.asarray(n) - vertex for n in neighbors)
            cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            if cosine > -np.cos(np.deg2rad(30)):
                anchors.add(vertex)
    if certify_stock_hull and bounds is None:
        raise ValueError('Strict canonical-chain stock certification requires supplied source box bounds')
    cache, reports, native_reports, stock_reports = {}, [], [], []
    records, record_indices, side_specs = [], {}, []
    for layout in layouts:
        layout.cornerless_junction_positions = [
            [pos for vertex in path[1:-1] if (pos := key(layout.vertices[vertex])) in cornerless_junctions]
            for path in layout.boundary_paths
        ]
        for side, path in enumerate(layout.boundary_paths):
            cuts = [i for i, vertex in enumerate(path) if key(layout.vertices[vertex]) in anchors]
            chain_points = [layout.vertices[path[left:right + 1]] for left, right in zip(cuts, cuts[1:])]
            lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in chain_points]
            total = sum(lengths)
            if total <= 0:
                raise ValueError("Shared boundary side has zero length")
            pieces = []
            for points, length in zip(chain_points, lengths):
                reverse = key(points[-1]) < key(points[0])
                canonical = points[::-1] if reverse else points
                identifier = tuple(map(key, canonical))
                if identifier not in cache:
                    protected = any(tuple(sorted((a, b))) in protected_edges
                                    for a, b in zip(identifier, identifier[1:]))
                    retain_tangents = preserve_endpoint_tangents or any(
                        tuple(sorted((a, b))) in tangent_edges for a, b in zip(identifier, identifier[1:]))
                    curve, report = _fit_curve(canonical, tolerance, preserve_polyline=protected, bounds=bounds,
                        preserve_endpoint_tangents=retain_tangents)
                    tangent_chain_count += int(retain_tangents)
                    if certify_stock_hull:
                        from .amrto_shared_stock import project_canonical_chain
                        curve, stock = project_canonical_chain(curve, canonical, bounds, tolerance)
                        stock_reports.append(stock)
                        report['canonical_chain_stock_certificate'] = stock
                    cache[identifier] = curve
                    reports.append(report)
                    record_indices[identifier] = len(records)
                    records.append({"points": canonical.copy(), "length": float(length),
                                    "protected": protected or not report["smooth_interior_knots"],
                                    "baseline": curve, "baseline_report": report})
                pieces.append((record_indices[identifier], reverse))
            side_specs.append(pieces)

    # Couple ordinary through-junction tangents once for the whole shared
    # graph. Protected or failed components retain the original C0 curves.
    # Do this before native UV changes, whose exact geometry-preserving
    # reparameterization remains necessary for retained native rectangles.
    from .amrto_shared_network import fit_shared_boundary_network
    side_curves, network_report = fit_shared_boundary_network(
        records, side_specs, tolerance, bounds=bounds, blocked_junction_positions=blocked_junction_positions)
    if certify_stock_hull and network_report['active_component_count']:
        raise ValueError('Strict canonical-chain certificate requires the fixed-junction boundary policy')
    side_index = 0
    for layout in layouts:
        sides = []
        influencing_positions = set()
        retained_native = getattr(layout, "parameterization_report", {}).get("branch") == "scaffold_rectangle"
        layout_native_reports = []
        for side, path in enumerate(layout.boundary_paths):
            side_curve = side_curves[side_index]
            influencing_positions.update(network_report["active_junction_positions_by_side"][side_index])
            side_index += 1
            if retained_native:
                chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(layout.vertices[path], axis=0), axis=1))]
                chord /= chord[-1]
                axis = 0 if side % 2 == 0 else 1
                native = np.asarray(layout.parameter_coords)[path, axis]
                if side >= 2:
                    native = 1. - native
                side_curve, native_report = reparameterize_shared_curve(side_curve, chord, native)
                native_report.update(layout_id=layout.layout_id, side=side)
                layout_native_reports.append(native_report)
                native_reports.append(native_report)
            sides.append(side_curve)
            if certify_stock_hull:
                from .amrto_shared_stock import control_hull_in_box
                if not control_hull_in_box(side_curve.c, bounds)['accepted']:
                    raise ValueError('Composed shared side escaped its certified canonical-chain stock hull')
        layout.boundary_splines = sides
        layout._active_shared_network_junction_positions = tuple(sorted(influencing_positions))
        layout.shared_boundary_parameterization_report = {
            "retained_native_uv": retained_native, "sides": layout_native_reports}
    return {"method": "Common cubic B-spline graph chains with fixed junctions",
            "endpoint_tangent_preservation_extension": bool(preserve_endpoint_tangents or tangent_edges),
            "endpoint_tangent_preserving_chain_count": tangent_chain_count,
            "chain_count": len(cache), "smooth_chain_count": sum(r["smooth_interior_knots"] for r in reports),
            "polyline_fallback_chain_count": sum(not r["smooth_interior_knots"] and not r.get("exact_protected_chain", False)
                                                   for r in reports),
            "exact_protected_chain_count": sum(r.get("exact_protected_chain", False) for r in reports),
            "protected_source_crease_edge_count": len(protected_edges),
            "maximum_sampled_chain_error": max(
                max((r["maximum_sampled_error"] for r in reports), default=0),
                network_report["maximum_sampled_chain_source_error"],
                network_report["maximum_final_composed_side_source_error"]),
            "native_reparameterized_side_count": sum(r["applied"] for r in native_reports),
            "native_added_boundary_control_count": sum(r.get("added_control_count", 0) for r in native_reports),
            "native_boundary_parameterization_extension": bool(native_reports),
            "joint_boundary_network": network_report,
            "canonical_chain_stock_certificate": {
                'enabled': bool(certify_stock_hull), 'chain_count': len(stock_reports),
                'all_composed_boundary_paths_certified': bool(certify_stock_hull),
                'projection_before_side_composition': bool(certify_stock_hull),
                'changed_chain_count': sum(row['changed_pole_count'] > 0 for row in stock_reports),
                'changed_pole_count': sum(row['changed_pole_count'] for row in stock_reports),
                'maximum_complete_source_curve_error_bound': max(
                    (row['complete_source_curve_error_bound'] for row in stock_reports), default=0.0),
                'chains': stock_reports,
            },
            "tolerance": float(tolerance), "paper_multiresolution": False}
