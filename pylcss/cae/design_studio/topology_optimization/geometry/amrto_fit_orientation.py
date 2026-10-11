# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Independent finite orientation checks against the actual source UV map.

Position error alone does not exclude spline folds. This acceptance extension
compares exact first derivatives with oriented source triangles at knot-span
sites and immediately on both sides of internal knots. It does not impose G1
continuity or smooth intentional source creases, and is not an injectivity proof.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import BSpline


def _surface_data(surface: Any):
    def get(name):
        return surface[name] if isinstance(surface, dict) else getattr(surface, name)

    if isinstance(surface, dict):
        shape = tuple(get("control_shape"))
        controls = np.asarray(get("control_points"), dtype=float).reshape(*shape, 3)
    else:
        shape = (surface.ctrlpts_size_u, surface.ctrlpts_size_v)
        # geomdl NURBS.Surface.ctrlpts2d contains homogeneous 4D controls;
        # ctrlpts is its Euclidean 3D view, while weights remain separate.
        controls_value = getattr(surface, "ctrlpts", None)
        controls = (np.asarray(surface.ctrlpts2d, dtype=float) if controls_value is None
                    else np.asarray(controls_value, dtype=float).reshape(*shape, 3))
    if controls.shape != (*shape, 3) or not np.isfinite(controls).all():
        raise ValueError("Orientation validation requires finite Euclidean control points")
    weights_value = surface.get("weights") if isinstance(surface, dict) else getattr(surface, "weights", None)
    weights = None if weights_value is None else np.asarray(weights_value, dtype=float).reshape(shape)
    if weights is not None and (not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Orientation validation requires finite positive rational weights")
    axes = []
    for index, axis in enumerate(("u", "v")):
        degree, knots = int(get("degree_" + axis)), np.asarray(get("knotvector_" + axis), dtype=float)
        if (degree < 1 or degree >= shape[index] or len(knots) != shape[index] + degree + 1
                or not np.isfinite(knots).all() or np.any(np.diff(knots) < 0)
                or knots[degree] >= knots[-degree - 1]):
            raise ValueError("Orientation validation requires a valid nonzero spline domain")
        axes.append((degree, knots, (float(knots[degree]), float(knots[-degree - 1]))))
    return controls, weights, axes


def _axis_sites(knots, degree, uniform_samples, knot_probe_fraction):
    lo, hi = knots[degree], knots[-degree - 1]
    unique = np.unique(knots[(knots >= lo) & (knots <= hi)])
    quarter = [a + (b - a) * fraction for a, b in zip(unique[:-1], unique[1:])
               for fraction in (.25, .5, .75)]
    paired = []
    for index in range(1, len(unique) - 1):
        knot = unique[index]
        offset = min(knot - unique[index - 1], unique[index + 1] - knot) * knot_probe_fraction
        # nextafter retains a one-sided site even for a span a few ULPs wide.
        paired.extend((max(unique[index - 1], min(knot - offset, np.nextafter(knot, -np.inf))),
                       min(unique[index + 1], max(knot + offset, np.nextafter(knot, np.inf)))))
    # Internal knots themselves are excluded: derivatives may be one-sided at
    # C0 creases. Both limits and all span quarter points remain represented.
    uniform = np.linspace(lo, hi, uniform_samples)
    uniform = uniform[~np.isin(uniform, unique[1:-1])]
    return np.unique(np.r_[uniform, quarter, paired]), len(paired)


def assess_fit_orientation(layout, surface: Any, *, uniform_samples: int = 33,
                           knot_probe_fraction: float = 1e-6,
                           minimum_tangent_sine: float = 1e-10,
                           maximum_sites: int = 1_000_000,
                           maximum_reported_sites: int = 32) -> dict:
    """Return an independent acceptance report without mutating either input.

    Source normals are divided by the sign of each triangle's UV determinant,
    so reversed triangle ordering and UV axis conventions are handled locally.
    At a shared source edge, alignment with any incident oriented source normal
    is sufficient; this preserves legitimate source creases. Only exact chart
    corners are exempt from tangent degeneracy, never from an actual inversion.
    All knot-span cross-pairs are checked. Excessive grids fail explicitly;
    sampling is never silently reduced to meet the bounded resource budget.
    """
    if (uniform_samples < 3 or not 0 < knot_probe_fraction < .25
            or not 0 < minimum_tangent_sine < 1 or maximum_sites < 1
            or maximum_reported_sites < 1):
        raise ValueError("Require positive bounded orientation sampling settings")
    controls, weights, axes = _surface_data(surface)
    vertices, triangles = np.asarray(layout.vertices, dtype=float), np.asarray(layout.triangles)
    uv = np.asarray(layout.parameter_coords, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or uv.shape != (len(vertices), 2) or not np.isfinite(uv).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("Orientation validation requires actual finite indexed source geometry and UVs")
    sites, probe_counts = [], []
    for degree, knots, _ in axes:
        parameters, count = _axis_sites(knots, degree, uniform_samples, knot_probe_fraction)
        sites.append(parameters)
        probe_counts.append(count)
    count = len(sites[0]) * len(sites[1])
    if count > maximum_sites:
        raise ValueError(f"Orientation grid has {count} sites, exceeding explicit budget {maximum_sites}")
    basis = [BSpline(knots, np.eye(controls.shape[index]), degree)
             for index, (degree, knots, _) in enumerate(axes)]
    values = [spline(parameters) for spline, parameters in zip(basis, sites)]
    derivatives = [spline(parameters, nu=1) for spline, parameters in zip(basis, sites)]
    tensor = (controls if weights is None else
              np.concatenate((controls * weights[..., None], weights[..., None]), axis=2))
    points, normals = np.empty((count, 3)), np.empty((count, 3))
    magnitude, sine = np.empty(count), np.empty(count)
    nu, nv = len(sites[0]), len(sites[1])
    tile = max(1, 8192 // nv)
    for start in range(0, nu, tile):
        end = min(start + tile, nu)
        position = np.einsum("ui,ijc,vj->uvc", values[0][start:end], tensor, values[1], optimize=True)
        du = np.einsum("ui,ijc,vj->uvc", derivatives[0][start:end], tensor, values[1], optimize=True)
        dv = np.einsum("ui,ijc,vj->uvc", values[0][start:end], tensor, derivatives[1], optimize=True)
        if weights is not None:
            denominator = position[..., 3:]
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ValueError("Rational derivative evaluation has a nonpositive denominator")
            position = position[..., :3] / denominator
            du = (du[..., :3] - du[..., 3:] * position) / denominator
            dv = (dv[..., :3] - dv[..., 3:] * position) / denominator
        cross = np.cross(du, dv)
        length = np.linalg.norm(cross, axis=2)
        product = np.linalg.norm(du, axis=2) * np.linalg.norm(dv, axis=2)
        tangent_sine = np.divide(length, product, out=np.zeros_like(length), where=product > 0)
        normal = np.divide(cross, length[..., None], out=np.zeros_like(cross), where=length[..., None] > 0)
        section = slice(start * nv, end * nv)
        points[section], normals[section] = position.reshape(-1, 3), normal.reshape(-1, 3)
        magnitude[section], sine[section] = length.ravel(), tangent_sine.ravel()
    if not all(np.isfinite(array).all() for array in (points, normals, magnitude, sine)):
        raise ValueError("Spline orientation evaluation produced nonfinite derivatives")
    regular = (magnitude > np.finfo(float).tiny) & (sine > minimum_tangent_sine)
    grid_u, grid_v = np.meshgrid(*sites, indexing="ij")
    flat_uv = np.column_stack((grid_u.ravel(), grid_v.ravel()))
    corner = ((flat_uv[:, 0] == axes[0][2][0]) | (flat_uv[:, 0] == axes[0][2][1])) & (
        (flat_uv[:, 1] == axes[1][2][0]) | (flat_uv[:, 1] == axes[1][2][1]))
    best_dot = np.full(count, -np.inf)
    source_ids = np.full(count, -1, dtype=np.int64)
    interior_owners = np.zeros(count, dtype=np.int32)
    bary_tolerance = 1e-9
    for index, triangle in enumerate(triangles):
        a, b, c = uv[triangle]
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        physical = vertices[triangle]
        normal = np.cross(physical[1] - physical[0], physical[2] - physical[0])
        length = np.linalg.norm(normal)
        if determinant == 0 or not np.isfinite(determinant) or length <= 0:
            raise ValueError("Source orientation requires nondegenerate physical and UV triangles")
        normal *= np.sign(determinant) / length
        lower, upper = np.min(uv[triangle], axis=0), np.max(uv[triangle], axis=0)
        pad = (upper - lower) * bary_tolerance + 16 * np.finfo(float).eps
        ui = np.flatnonzero((sites[0] >= lower[0] - pad[0]) & (sites[0] <= upper[0] + pad[0]))
        vi = np.flatnonzero((sites[1] >= lower[1] - pad[1]) & (sites[1] <= upper[1] + pad[1]))
        if not len(ui) or not len(vi):
            continue
        first, second = np.meshgrid(ui, vi, indexing="ij")
        ids = (first * nv + second).ravel()
        query = flat_uv[ids]
        beta = ((query[:, 0] - a[0]) * (c[1] - a[1]) -
                (query[:, 1] - a[1]) * (c[0] - a[0])) / determinant
        gamma = ((b[0] - a[0]) * (query[:, 1] - a[1]) -
                 (b[1] - a[1]) * (query[:, 0] - a[0])) / determinant
        alpha = 1 - beta - gamma
        valid = (alpha >= -bary_tolerance) & (beta >= -bary_tolerance) & (gamma >= -bary_tolerance)
        local_ids = ids[valid]
        dots = normals[local_ids] @ normal
        better = dots > best_dot[local_ids]
        best_dot[local_ids[better]] = dots[better]
        source_ids[local_ids[better]] = index
        interior = (alpha > bary_tolerance) & (beta > bary_tolerance) & (gamma > bary_tolerance)
        interior_owners[ids[interior]] += 1
    mapped = source_ids >= 0
    inverted = regular & mapped & (best_dot < 0)
    degenerate = ~regular & ~corner
    unmapped = ~mapped
    overlapping = interior_owners > 1
    bad = inverted | degenerate | unmapped | overlapping
    bad_ids = np.flatnonzero(bad)
    # Keep the most strongly inverted sites first, then degeneracy/coverage.
    ordered = sorted(bad_ids, key=lambda i: (not inverted[i], best_dot[i] if mapped[i] else 0., int(i)))
    failures = []
    for index in ordered[:maximum_reported_sites]:
        reasons = [name for name, flag in (("inverted_source_orientation", inverted[index]),
                    ("degenerate_spline_tangents", degenerate[index]),
                    ("source_UV_not_covered", unmapped[index]),
                    ("overlapping_source_UV_interiors", overlapping[index])) if flag]
        source_normal = None
        if mapped[index]:
            triangle = triangles[source_ids[index]]
            p, t = vertices[triangle], uv[triangle]
            cross = np.cross(p[1] - p[0], p[2] - p[0])
            sign = np.sign((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) -
                           (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0]))
            source_normal = (cross * sign / np.linalg.norm(cross)).tolist()
        failures.append({"uv": flat_uv[index].tolist(), "physical_point": points[index].tolist(),
                         "source_triangle": int(source_ids[index]), "source_normal": source_normal,
                         "spline_normal": normals[index].tolist(), "reasons": reasons,
                         "normal_dot": float(best_dot[index]) if mapped[index] else None,
                         "tangent_cross_magnitude": float(magnitude[index]), "tangent_sine": float(sine[index])})
    return {"method": "Exact spline first derivatives versus actual piecewise-linear source UV orientation",
            "accepted": not bool(bad.any()), "layout_id": int(layout.layout_id),
            "sample_count": count, "axis_sample_counts": [nu, nv],
            "internal_knot_one_sided_axis_site_counts": probe_counts,
            "knot_probe_fraction": knot_probe_fraction, "rational_derivatives_supported": True,
            "rational_surface": weights is not None, "minimum_tangent_sine": minimum_tangent_sine,
            "inverted_site_count": int(inverted.sum()), "degenerate_site_count": int(degenerate.sum()),
            "exempt_degenerate_exact_chart_corner_count": int((~regular & corner).sum()),
            "unmapped_site_count": int(unmapped.sum()), "overlapping_source_interior_site_count": int(overlapping.sum()),
            "minimum_source_normal_dot": float(best_dot[regular & mapped].min()) if (regular & mapped).any() else None,
            "failed_sites": failures, "reported_sites_truncated": len(ordered) > maximum_reported_sites,
            "source_crease_policy": "Accept orientation against any incident source normal at shared triangle edges",
            "certified_orientation_or_injectivity_bound": False}
