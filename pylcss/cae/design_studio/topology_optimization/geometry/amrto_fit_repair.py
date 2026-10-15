# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Bounded accuracy extension for folded polynomial spline interiors.

The paper's position fit does not constrain orientation. This extension fits
the actual source map while retaining the complete basis and exact canonical
borders. Linearized signed-area inequalities use source triangle normals,
without generic fairing. Independent acceptance remains finite sampling; it
does not certify global injectivity or a continuous orientation bound.
"""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import factorized, spsolve

from .amrto_fit_orientation import _axis_sites, assess_fit_orientation
from .amrto_parameterization import sample_parameter_grid


def _axis_basis(parameters, knots, degree, derivative=0):
    """CSR values/first derivatives with at most degree+1 entries per row."""
    knots = np.asarray(knots, dtype=float)
    if not derivative:
        result = BSpline.design_matrix(parameters, knots, degree).tocsr()
    else:
        count = len(knots) - degree - 1
        # The lower-degree auxiliary basis has an extra zero-width end span.
        # Evaluate its left limit at the final endpoint to obtain the derivative.
        points = np.asarray(parameters, dtype=float).copy()
        end = knots[-degree - 1]
        points[points == end] = np.nextafter(end, knots[degree])
        lower = BSpline.design_matrix(points, knots, degree - 1).tocsr()
        indices = np.arange(count)
        first = knots[indices + degree] - knots[indices]
        second = knots[indices + degree + 1] - knots[indices + 1]
        a, b = first > 0, second > 0
        transform = sparse.csr_matrix(
            (
                np.r_[degree / first[a], -degree / second[b]],
                (np.r_[indices[a], indices[b] + 1], np.r_[indices[a], indices[b]]),
            ),
            shape=(count + 1, count),
        )
        result = (lower @ transform).tocsr()
    result.eliminate_zeros()
    return result


def _tensor_basis(uv, knots, degrees, du=0, dv=0):
    """Sparse rowwise tensor products; cubic support is at most 16 entries."""
    axes = [
        _axis_basis(uv[:, axis], knots[axis], degrees[axis], derivative)
        for axis, derivative in enumerate((du, dv))
    ]
    padded = []
    for basis in axes:
        counts = np.diff(basis.indptr)
        width = int(counts.max(initial=0))
        columns = np.zeros((len(uv), width), dtype=np.int64)
        values = np.zeros((len(uv), width))
        rows = np.repeat(np.arange(len(uv)), counts)
        local = np.arange(basis.nnz) - np.repeat(basis.indptr[:-1], counts)
        columns[rows, local], values[rows, local] = basis.indices, basis.data
        padded.append((columns, values))
    columns = (
        padded[0][0][:, :, None] * axes[1].shape[1] + padded[1][0][:, None, :]
    ).reshape(len(uv), -1)
    values = (padded[0][1][:, :, None] * padded[1][1][:, None, :]).reshape(len(uv), -1)
    rows = np.repeat(np.arange(len(uv)), columns.shape[1])
    keep = values.ravel() != 0
    return sparse.csr_matrix(
        (values.ravel()[keep], (rows[keep], columns.ravel()[keep])),
        shape=(len(uv), axes[0].shape[1] * axes[1].shape[1]),
    )


def _source_jacobians(layout, query, deadline):
    result = np.full((len(query), 2, 3), np.nan)
    for index, triangle in enumerate(layout.triangles):
        if index % 32 == 0 and time.monotonic() > deadline:
            raise TimeoutError("source Jacobian time limit")
        uv, points = layout.parameter_coords[triangle], layout.vertices[triangle]
        lower, upper = uv.min(axis=0) - 1e-10, uv.max(axis=0) + 1e-10
        ids = np.flatnonzero(np.all((query >= lower) & (query <= upper), axis=1))
        if not len(ids):
            continue
        edges = np.array([uv[1] - uv[0], uv[2] - uv[0]]).T
        inverse = np.linalg.inv(edges)
        local = (query[ids] - uv[0]) @ inverse.T
        inside = (
            (1 - local.sum(axis=1) >= -1e-10)
            & (local[:, 0] >= -1e-10)
            & (local[:, 1] >= -1e-10)
        )
        result[ids[inside]] = (
            np.array([points[1] - points[0], points[2] - points[0]]).T @ inverse
        ).T
    if not np.isfinite(result).all():
        raise ValueError("Source UV map does not cover derivative constraint sites")
    return result


def _projected_derivative(matrix, coefficients):
    matrix = matrix.tocoo()
    return sparse.csr_matrix(
        (
            (matrix.data[:, None] * coefficients[matrix.row]).ravel(),
            (
                np.repeat(matrix.row, 3),
                (matrix.col[:, None] * 3 + np.arange(3)).ravel(),
            ),
        ),
        shape=(matrix.shape[0], matrix.shape[1] * 3),
    )


def repair_surface_orientation(
    layout,
    surface,
    *,
    position_limit,
    maximum_seconds=60.0,
    maximum_iterations=8,
    maximum_control_points=4096,
    maximum_training_sites=40_000,
    maximum_constraint_sites=50_000,
    maximum_verification_sites=150_000,
    reference_layout=None,
    box_bounds=None,
):
    """Return ``(accepted_copy, report)`` or ``(None, rejection_report)``.

    Inputs are never mutated. Borders/basis stay bitwise equal. Acceptance
    requires source-grid73 accuracy and complete independent 129-uniform
    orientation checks at five one-sided knot offsets; no grid is coarsened.
    Sparse matrix solves and validation are bounded by explicit size budgets.
    The time limit is checked between indivisible SciPy/validation operations.
    """
    started = time.monotonic()
    from .amrto_nurbs_fitting import _surface_map_residual

    deadline = started + maximum_seconds
    reference = layout if reference_layout is None else reference_layout
    report = {
        "method": "Frozen canonical borders with source signed-area constrained interior LS",
        "paper_algorithm": False,
        "production_accuracy_extension": True,
        "accepted": False,
        "layout_id": int(layout.layout_id),
        "iterations": [],
        "certified_orientation_or_injectivity_bound": False,
        "resource_limits": {
            "seconds": maximum_seconds,
            "iterations": maximum_iterations,
            "control_points": maximum_control_points,
            "training_sites": maximum_training_sites,
            "constraint_sites": maximum_constraint_sites,
            "verification_sites": maximum_verification_sites,
        },
    }

    def reject(reason, status="rejected", error=None):
        report.update(
            status=status, reason=reason, wall_seconds=time.monotonic() - started
        )
        if error is not None:
            report["error"] = str(error)
        return None, report

    if (
        not np.isfinite(position_limit)
        or position_limit <= 0
        or maximum_seconds <= 0
        or min(
            maximum_iterations,
            maximum_control_points,
            maximum_training_sites,
            maximum_constraint_sites,
            maximum_verification_sites,
        )
        < 1
    ):
        return reject("invalid_limits")
    if getattr(surface, "weights", None) is not None:
        return reject("rational_surface_not_supported", "skipped")
    if time.monotonic() > deadline:
        return reject("time_limit", "resource_limit")
    try:
        if (not np.array_equal(reference.parameter_coords, layout.parameter_coords)
                or not np.array_equal(reference.triangles, layout.triangles)):
            return reject('reference_chart_mismatch')
        baseline = np.asarray(surface.ctrlpts2d, dtype=float)
        nu, nv = baseline.shape[:2]
        if nu * nv > min(maximum_control_points, 4096):
            return reject("control_point_limit", "resource_limit")
        if len(layout.triangles) > 20_000:
            return reject("source_triangle_limit_20000", "resource_limit")
        degrees = [surface.degree_u, surface.degree_v]
        knots = [surface.knotvector_u, surface.knotvector_v]
        if min(degrees) < 1 or max(degrees) > 3:
            return reject("only_polynomial_degrees_1_to_3_supported", "skipped")
        training_axes = []
        for axis in range(2):
            unique = np.unique(knots[axis])
            spans = np.concatenate(
                [
                    unique[:-1] + f * np.diff(unique)
                    for f in (0.125, 0.25, 0.5, 0.75, 0.875)
                ]
            )
            training_axes.append(
                np.unique(
                    np.r_[
                        (np.arange(49) + 0.5) / 49,
                        spans,
                        _axis_sites(np.asarray(knots[axis]), degrees[axis], 9, 1e-6)[0],
                    ]
                )
            )
        if len(training_axes[0]) * len(training_axes[1]) > maximum_training_sites:
            return reject("training_site_limit", "resource_limit")
        for fraction in (1e-6, 1e-4, 0.005, 0.01, 0.02):
            count = np.prod(
                [
                    len(_axis_sites(np.asarray(knots[a]), degrees[a], 129, fraction)[0])
                    for a in range(2)
                ]
            )
            if count > maximum_verification_sites:
                return reject("verification_site_limit", "resource_limit")
        if time.monotonic() > deadline:
            return reject("time_limit", "resource_limit")
        candidate = deepcopy(surface)
        report["baseline_position_maximum_error"] = _surface_map_residual(
            reference, surface, grid_size=73
        )
        report["baseline_orientation"] = assess_fit_orientation(
            reference, surface, maximum_sites=maximum_verification_sites
        )
        if time.monotonic() > deadline:
            return reject("time_limit", "resource_limit")
        # Start with a small signed-area system and every reported baseline
        # failure. Full independent knot/uniform verification below remains
        # unchanged and adds newly found inversions before the next solve.
        # Constraining the complete dense verification grid up front can use
        # the entire repair budget before a single candidate is checked.
        seed = np.linspace(0., 1., 9)
        uu, vv = np.meshgrid(seed, seed, indexing="ij")
        failed_uv = np.asarray([site["uv"] for site in report["baseline_orientation"]["failed_sites"]],
                               dtype=float).reshape(-1, 2)
        query = np.unique(np.vstack([np.column_stack((uu.ravel(), vv.ravel())), failed_uv]), axis=0)
        query = query[
            ~(
                ((query[:, 0] == 0) | (query[:, 0] == 1))
                & ((query[:, 1] == 0) | (query[:, 1] == 1))
            )
        ]
        if len(query) > maximum_constraint_sites:
            return reject("constraint_site_limit", "resource_limit")
        report["constraint_strategy"] = "9x9 seed and reported failures; full independent verification adds constraints"
        report["initial_constraint_site_count"] = len(query)
        uu, vv = np.meshgrid(*training_axes, indexing="ij")
        training = np.column_stack((uu.ravel(), vv.ravel()))
        matrix = sparse.kron(
            _tensor_basis(training, knots, degrees), sparse.eye(3), format="csr"
        )
        values = sample_parameter_grid(layout, *training_axes).ravel()
        boundary = np.zeros((nu, nv), dtype=bool)
        boundary[[0, -1], :], boundary[:, [0, -1]] = True, True
        known = np.repeat(boundary.ravel(), 3)
        interior, original = ~known, baseline.ravel().copy()
        lower_box = upper_box = None
        if box_bounds is not None:
            limits = np.asarray(box_bounds, dtype=float)
            if limits.shape != (2, 3) or not np.isfinite(limits).all() or np.any(limits[0] >= limits[1]):
                return reject('invalid_source_box')
            lower_box = np.broadcast_to(limits[0], baseline.shape).ravel()[interior]
            upper_box = np.broadcast_to(limits[1], baseline.shape).ravel()[interior]
            report['source_box_interior_constraints'] = True
        rhs = values - matrix[:, known] @ original[known]
        matrix = matrix[:, interior]
        normal, target = (matrix.T @ matrix).tocsc(), matrix.T @ rhs
        x = spsolve(normal + sparse.eye(normal.shape[0], format="csc") * 1e-12, target)
        if lower_box is not None:
            x = np.clip(x, lower_box, upper_box)
        report["training_site_count"] = len(training)
        for outer in range(maximum_iterations):
            if time.monotonic() > deadline:
                return reject("time_limit", "resource_limit")
            if len(query) > maximum_constraint_sites:
                return reject("constraint_site_limit", "resource_limit")
            jac = _source_jacobians(reference, query, deadline)
            normals = np.cross(jac[:, 0], jac[:, 1])
            area = np.linalg.norm(normals, axis=1)
            normals /= area[:, None]
            du, dv = (
                _tensor_basis(query, knots, degrees, du=1),
                _tensor_basis(query, knots, degrees, dv=1),
            )
            controls = original.copy()
            controls[interior] = x
            tangent_u, tangent_v = (
                du @ controls.reshape(-1, 3),
                dv @ controls.reshape(-1, 3),
            )
            signed = np.einsum("ij,ij->i", np.cross(tangent_u, tangent_v), normals)
            full = _projected_derivative(
                du, np.cross(tangent_v, normals)
            ) + _projected_derivative(dv, np.cross(normals, tangent_u))
            lower = area * 0.05 + signed - full[:, known] @ original[known]
            constraint = full[:, interior]
            scale = np.maximum(
                np.sqrt(np.asarray(constraint.power(2).sum(axis=1)).ravel()), 1e-12
            )
            constraint = sparse.diags(1 / scale) @ constraint
            lower /= scale
            if lower_box is not None:
                identity = sparse.eye(len(x), format='csr')
                constraint = sparse.vstack((constraint, identity, -identity), format='csr')
                lower = np.r_[lower, lower_box, -upper_box]
            rho, proximity = 100.0, 0.001
            solve = factorized(
                normal
                + rho * (constraint.T @ constraint).tocsc()
                + sparse.eye(len(x), format="csc") * proximity
            )
            local_target = target + proximity * x
            z, dual = np.maximum(constraint @ x, lower), np.zeros(len(lower))
            for inner in range(500):
                x = solve(local_target + rho * constraint.T @ (z - dual))
                projected = constraint @ x
                previous, z = z, np.maximum(projected + dual, lower)
                dual += projected - z
                primal = float(np.max(np.abs(projected - z)))
                dual_error = float(np.max(np.abs(rho * constraint.T @ (z - previous))))
                if primal < 1e-8 and dual_error < 1e-7:
                    break
                if time.monotonic() > deadline:
                    return reject("time_limit", "resource_limit")
            if lower_box is not None:
                # Remove solver roundoff at a stock plane before independent
                # orientation/position verification; never clamp afterward.
                x = np.clip(x, lower_box, upper_box)
            controls[interior] = x
            candidate.set_ctrlpts(controls.reshape(-1, 3).tolist(), nu, nv)
            checks, failures = [], []
            for fraction in (1e-6, 1e-4, 0.005, 0.01, 0.02):
                if time.monotonic() > deadline:
                    return reject("time_limit", "resource_limit")
                checked = assess_fit_orientation(
                    reference,
                    candidate,
                    uniform_samples=129,
                    knot_probe_fraction=fraction,
                    maximum_sites=maximum_verification_sites,
                    maximum_reported_sites=128,
                )
                checks.append(
                    {
                        "uniform_samples": 129,
                        "knot_probe_fraction": fraction,
                        "report": checked,
                    }
                )
                failures.extend(
                    [
                        site["uv"]
                        for site in checked["failed_sites"]
                        if "inverted_source_orientation" in site["reasons"]
                    ]
                )
            error = _surface_map_residual(reference, candidate, grid_size=73)
            report["final_position_maximum_error"] = error
            report["iterations"].append(
                {
                    "outer_iteration": outer,
                    "inner_iterations": inner + 1,
                    "constraint_site_count": len(query),
                    "constraint_primal_inf": primal,
                    "constraint_dual_inf": dual_error,
                    "position_maximum_error": error,
                    "orientation_checks": checks,
                }
            )
            report["canonical_boundary_controls_bitwise_equal"] = bool(
                np.array_equal(controls[known], original[known])
            )
            report["canonical_boundary_control_maximum_difference"] = float(
                np.max(np.abs(controls[known] - original[known]))
            )
            report["maximum_pole_displacement"] = float(
                np.linalg.norm(
                    controls.reshape(-1, 3) - baseline.reshape(-1, 3), axis=1
                ).max()
            )
            report["basis_bitwise_equal"] = bool(
                candidate.knotvector_u == surface.knotvector_u
                and candidate.knotvector_v == surface.knotvector_v
                and candidate.degree_u == surface.degree_u
                and candidate.degree_v == surface.degree_v
            )
            if (
                all(check["report"]["accepted"] for check in checks)
                and error <= position_limit
                and report["canonical_boundary_controls_bitwise_equal"]
                and report["basis_bitwise_equal"]
            ):
                if time.monotonic() > deadline:
                    return reject("time_limit", "resource_limit")
                report.update(
                    accepted=True,
                    status="accepted",
                    wall_seconds=time.monotonic() - started,
                )
                return candidate, report
            if failures:
                query = np.unique(np.vstack([query, np.asarray(failures)]), axis=0)
            elif not all(check["report"]["accepted"] for check in checks):
                return reject("source_UV_or_tangent_validation_failed")
        return reject("orientation_or_source_accuracy_not_satisfied")
    except TimeoutError as error:
        return reject("time_limit", "resource_limit", error)
    except Exception as error:
        return reject("invalid_source_or_candidate", error=error)
