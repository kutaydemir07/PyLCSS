# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Joint ordinary seam tangents before fitting shared-boundary CAD surfaces.

This production extension changes source approximations only within their
existing boundary budget. Protected, incompatible and unsuccessful components
keep their original curves; it never changes an already fitted CAD surface.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq


_JOIN_GEOMETRY_TOLERANCE = 1e-9


def _away_derivative(record, endpoint):
    parameter = 0. if endpoint == 0 else 1.
    return record["baseline"].derivative()(parameter) * (1. if endpoint == 0 else -1.) / record["length"]


def _continuation_components(records, side_specs):
    graph, joins = defaultdict(set), set()
    skipped = 0
    for pieces in side_specs:
        for first, second in zip(pieces, pieces[1:]):
            a, b = (first[0], 0 if first[1] else 1), (second[0], 1 if second[1] else 0)
            if (a, b) in joins:
                continue
            joins.add((a, b))
            if records[a[0]]["protected"] or records[b[0]]["protected"]:
                skipped += 1
                continue
            graph[a].add(b)
            graph[b].add(a)
    components, assigned = [], {}
    for seed in graph:
        if seed in assigned:
            continue
        index = len(components)
        signs, pending, odd = {seed: 1}, [seed], False
        while pending:
            node = pending.pop()
            assigned[node] = index
            for other in sorted(graph[node]):
                expected = -signs[node]
                if other in signs:
                    odd |= signs[other] != expected
                else:
                    signs[other] = expected
                    pending.append(other)
        values = np.asarray([sign * _away_derivative(records[node[0]], node[1])
                             for node, sign in signs.items()])
        common = values.mean(axis=0)
        magnitude = float(np.linalg.norm(common))
        compatible = (not odd and magnitude > 1e-12 and np.isfinite(values).all()
                      and np.all(np.linalg.norm(values, axis=1) > 1e-12)
                      and np.all(values @ common > 0.))
        if compatible:
            common /= magnitude
        components.append({"signs": signs, "direction": common, "enabled": bool(compatible),
                           "compatible": bool(compatible), "odd_cycle": bool(odd)})
    return components, assigned, len(joins), skipped


def _source_parameters(points):
    parameters = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    parameters /= parameters[-1]
    return parameters


def _source_error(curve, points, *, distance_to_polyline):
    parameters = _source_parameters(points)
    spans = np.unique(curve.t)
    sites = np.unique(np.r_[parameters, spans,
        np.linspace(0., 1., max(80, 8 * len(points), 8 * len(curve.c))),
        *(spans[:-1] + fraction * np.diff(spans) for fraction in (.25, .5, .75))])
    source = np.column_stack([np.interp(sites, parameters, points[:, axis]) for axis in range(3)])
    sampled = curve(sites)
    return max(float(np.linalg.norm(sampled - source, axis=1).max()),
               float(distance_to_polyline(sampled, points).max()))


def _constrained_chain(record, directions, tolerance, bounds, *, distance_to_polyline):
    if not directions:
        return record["baseline"], {"unchanged": True}
    points, length = record["points"], record["length"]
    parameters = _source_parameters(points)
    sites = np.unique(np.r_[parameters, np.linspace(0., 1., max(40, 5 * len(points)))])
    source = np.column_stack([np.interp(sites, parameters, points[:, axis]) for axis in range(3)])
    error = np.inf
    # This is the existing shared-curve work policy, not a model acceptance
    # limit: unsuccessful attempts revert the entire coupled component to C0.
    for count in (4, 6, 10, 16, 24, 40, 64):
        knots = np.r_[np.zeros(4), np.linspace(0., 1., count - 2)[1:-1], np.ones(4)]
        controls = np.zeros((count, 3))
        controls[0], controls[-1] = points[0], points[-1]
        fixed = {0, count - 1}
        for endpoint, away in directions.items():
            if endpoint == 0:
                controls[1] = points[0] + away * length * knots[4] / 3.
                fixed.add(1)
            else:
                controls[-2] = points[-1] + away * length * (1. - knots[-5]) / 3.
                fixed.add(count - 2)
        free, fixed = sorted(set(range(count)) - fixed), sorted(fixed)
        basis = BSpline.design_matrix(sites, knots, 3).toarray()
        if free:
            controls[free] = lstsq(basis[:, free], source - basis[:, fixed] @ controls[fixed], lapack_driver="gelsy")[0]
        if bounds is not None and (np.any(controls < bounds[0]) or np.any(controls > bounds[1])):
            continue
        candidate = BSpline(knots, controls, 3)
        error = _source_error(candidate, points, distance_to_polyline=distance_to_polyline)
        if np.isfinite(error) and error <= tolerance:
            return candidate, {"unchanged": False, "maximum_sampled_source_error": error}
    return None, {"maximum_sampled_source_error": error}


def _remove_redundant_join(curve, knot):
    knots = np.delete(curve.t, np.flatnonzero(curve.t == knot)[0])
    count = len(knots) - 4
    insertion = BSpline(knots, np.eye(count), 3).insert_knot(knot).c
    controls = np.zeros((count, 3))
    controls[[0, -1]] = curve.c[[0, -1]]
    rhs = curve.c - insertion[:, [0, -1]] @ controls[[0, -1]]
    controls[1:-1] = lstsq(insertion[:, 1:-1], rhs, lapack_driver="gelsy")[0]
    candidate = BSpline(knots, controls, 3)
    # Nonnegative, partition-of-unity basis functions give a uniform error
    # bound over the WHOLE curve, including arbitrarily narrow knot spans.
    bound = float(np.linalg.norm(insertion @ controls - curve.c, axis=1).max())
    spans = np.unique(curve.t)
    sites = np.unique(np.r_[np.linspace(0., 1., 1001), spans,
        *(spans[:-1] + fraction * np.diff(spans) for fraction in (.125, .25, .5, .75, .875))])
    error = float(np.linalg.norm(candidate(sites) - curve(sites), axis=1).max())
    return candidate, bound, error


def _compose_side(records, pieces, curves, active, endpoint_components):
    total = sum(records[index]["length"] for index, reverse in pieces)
    knots, controls, offset, removable = [], [], 0., []
    for piece_index, (index, reverse) in enumerate(pieces):
        curve = curves[index]
        local_knots = 1. - curve.t[::-1] if reverse else curve.t
        local_controls = curve.c[::-1] if reverse else curve.c
        scaled = offset + local_knots * records[index]["length"] / total
        if piece_index:
            previous, previous_reverse = pieces[piece_index - 1]
            a, b = (previous, 0 if previous_reverse else 1), (index, 1 if reverse else 0)
            component = endpoint_components.get(a)
            if component is not None and component == endpoint_components.get(b) and component in active:
                removable.append(offset)
            knots = knots[:-4] + [offset] * 3 + scaled[4:].tolist()
            controls.extend(local_controls[1:].tolist())
        else:
            knots, controls = scaled.tolist(), local_controls.tolist()
        offset += records[index]["length"] / total
    knots[-4:] = [1.] * 4
    original = curve = BSpline(np.asarray(knots), np.asarray(controls), 3)
    removed_knots, accumulated, sampled_error = [], 0., 0.
    for knot in removable:
        try:
            candidate, bound, error = _remove_redundant_join(curve, knot)
        except (ValueError, np.linalg.LinAlgError):
            continue  # Retaining the C0 representation preserves this curve.
        if (np.isfinite(bound + error) and error <= _JOIN_GEOMETRY_TOLERANCE
                and accumulated + bound <= _JOIN_GEOMETRY_TOLERANCE):
            curve = candidate
            accumulated += bound
            sampled_error = max(sampled_error, error)
            removed_knots.append(knot)
    restored = curve
    for knot in removed_knots:
        restored = restored.insert_knot(knot)
    bound = (float(np.linalg.norm(restored.c - original.c, axis=1).max())
             if np.array_equal(restored.t, original.t) else np.inf)
    if not np.isfinite(bound) or bound > _JOIN_GEOMETRY_TOLERANCE:
        return original, {"removed_c0_join_knots": 0, "uniform_common_basis_error_bound": 0.,
                          "accumulated_removal_error_bound": 0., "maximum_sampled_removal_error": 0.}
    return curve, {"removed_c0_join_knots": len(removed_knots), "uniform_common_basis_error_bound": bound,
                   "accumulated_removal_error_bound": accumulated, "maximum_sampled_removal_error": sampled_error}


def fit_shared_boundary_network(records, side_specs, tolerance, *, bounds=None,
                                blocked_junction_positions=()):
    """Return shared side curves with compatible ordinary continuations joined.

    ``records`` holds canonical points, lengths and baseline cubic curves.
    Every ``side_specs`` entry traverses (chain index, reversed) pairs. A bad
    chain disables its entire endpoint component before any side is accepted.
    Native rectangle UV changes must be applied afterwards by the caller.
    Blocked source-coordinate keys survive layout renumbering and subdivision.
    """
    from .amrto_shared_curves import _curve_distance_to_polyline

    components, endpoint_components, joins, skipped = _continuation_components(records, side_specs)
    blocked = {tuple(map(float, position)) for position in blocked_junction_positions}
    blocked_components = set()
    for index, component in enumerate(components):
        positions = {tuple(map(float, records[chain]["points"][0 if endpoint == 0 else -1]))
                     for chain, endpoint in component["signs"]}
        if positions & blocked:
            component["enabled"] = False
            blocked_components.add(index)
    cache, fallback_components, passes = {}, set(), 0
    while True:
        passes += 1
        active = {index for index, component in enumerate(components) if component["enabled"]}
        curves, chain_reports, failing = {}, {}, set()
        for index, record in enumerate(records):
            directions = {}
            for endpoint in (0, 1):
                node = (index, endpoint)
                component_id = endpoint_components.get(node)
                if component_id in active:
                    component = components[component_id]
                    directions[endpoint] = component["signs"][node] * component["direction"]
            signature = (index, tuple((endpoint, value.tobytes()) for endpoint, value in directions.items()))
            if signature not in cache:
                try:
                    cache[signature] = _constrained_chain(record, directions, tolerance, bounds,
                                                         distance_to_polyline=_curve_distance_to_polyline)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    cache[signature] = (None, {})
            candidate, report = cache[signature]
            chain_reports[index] = report
            if candidate is None:
                failing.update(endpoint_components[(index, endpoint)] for endpoint in directions)
            else:
                curves[index] = candidate
        if not failing:
            sides, side_reports = [], []
            for pieces in side_specs:
                curve, report = _compose_side(records, pieces, curves, active, endpoint_components)
                changed = any(curves[index] is not records[index]["baseline"] for index, reverse in pieces)
                source_error, junction_error = 0., 0.
                if changed:
                    points = [records[index]["points"][::-1] if reverse else records[index]["points"]
                              for index, reverse in pieces]
                    source = np.concatenate([points[0], *(part[1:] for part in points[1:])])
                    source_error = _source_error(curve, source, distance_to_polyline=_curve_distance_to_polyline)
                    total, offset = sum(records[index]["length"] for index, reverse in pieces), 0.
                    for index, reverse in pieces:
                        point = records[index]["points"][-1 if reverse else 0]
                        junction_error = max(junction_error, float(np.linalg.norm(curve(offset) - point)))
                        offset += records[index]["length"] / total
                    junction_error = max(junction_error, float(np.linalg.norm(curve(1.) - points[-1][-1])))
                    if (not np.isfinite(source_error + junction_error) or source_error > tolerance
                            or junction_error > _JOIN_GEOMETRY_TOLERANCE):
                        failing.update(endpoint_components[(index, endpoint)] for index, reverse in pieces
                                       for endpoint in (0, 1) if endpoint_components.get((index, endpoint)) in active)
                report.update(maximum_final_source_error=source_error, maximum_junction_error=junction_error)
                sides.append(curve)
                side_reports.append(report)
        if not failing:
            break
        if not failing.issubset(active):
            raise RuntimeError("Shared boundary continuation fallback did not identify active source constraints")
        for index in failing:
            components[index]["enabled"] = False
        fallback_components.update(failing)
    # Even a side containing only ONE chain inherits its constrained endpoint
    # tangents from neighboring layouts. Expose every active influencing key,
    # not just joins that occur inside this side's own traversal.
    active_positions_by_side = [sorted({
        tuple(map(float, records[index]["points"][0 if endpoint == 0 else -1]))
        for index, reverse in pieces for endpoint in (0, 1)
        if endpoint_components.get((index, endpoint)) in active}) for pieces in side_specs]
    return sides, {
        "method": "Joint shared-chain chord-length tangents with certified redundant C0 knot removal",
        "production_boundary_extension": True, "pre_fit_source_approximation_only": True,
        "source_tolerance": float(tolerance), "junction_geometry_tolerance": _JOIN_GEOMETRY_TOLERANCE,
        "chain_count": len(records), "continuation_count": joins, "continuation_component_count": len(components),
        "odd_component_count": sum(component["odd_cycle"] for component in components),
        "incompatible_component_count": sum(not component["compatible"] for component in components),
        "protected_or_nonsmooth_continuations_skipped": skipped,
        "active_component_count": len(active), "source_guard_fallback_component_count": len(fallback_components),
        "blocked_junction_count": len(blocked), "blocked_component_count": len(blocked_components),
        "active_junction_positions_by_side": active_positions_by_side,
        "fallback_passes": passes, "changed_chain_count": sum(not row.get("unchanged", False) for row in chain_reports.values()),
        "removed_c0_join_occurrences": sum(row["removed_c0_join_knots"] for row in side_reports),
        "maximum_sampled_chain_source_error": max((row.get("maximum_sampled_source_error", 0.) for row in chain_reports.values()), default=0.),
        "maximum_final_composed_side_source_error": max((row["maximum_final_source_error"] for row in side_reports), default=0.),
        "maximum_graph_junction_position_error": max((row["maximum_junction_error"] for row in side_reports), default=0.),
        "maximum_uniform_common_basis_error_bound": max((row["uniform_common_basis_error_bound"] for row in side_reports), default=0.),
        "maximum_accumulated_removal_error_bound": max((row["accumulated_removal_error_bound"] for row in side_reports), default=0.),
        "uniform_curve_geometry_tolerance": _JOIN_GEOMETRY_TOLERANCE,
        "protected_and_failed_components_retain_original_c0_curves": True,
        "closed_body_and_final_step_checks_required": True,
    }
