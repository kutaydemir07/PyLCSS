# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compact spline B-rep reconstruction for extrusion-constrained topology.

An arbitrary topology surface cannot be turned into a useful smooth CAD model
by assigning one planar B-rep face to every mesh triangle, and freeform patch
fitting produced bodies that no longer described the load path.  Extrusion-
constrained topology is the exact subset that does convert, because its final
surface is a planar profile swept along one axis.  This module recovers that
profile, fits sparse periodic B-spline wires, and builds a small, editable
prismatic B-rep.  Everything else is delivered as its recovered surface.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .brep_validation import protected_point_coverage

logger = logging.getLogger(__name__)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_CURVATURE_FAIRNESS_WEIGHTS = (2.0, 2.0, 1.0)
# Fairing is allowed to move the profile by this fraction of the fit tolerance,
# measured pointwise. It is a fraction rather than the whole tolerance because
# the initial approximation has already spent most of that budget getting off
# the voxel staircase; fairing only gets the remainder.
_CURVATURE_FAIRNESS_DEVIATION_BAND_FRACTION = 0.5
# Halvings of the retreat blend before fairing is abandoned entirely. Twenty
# steps take the perturbation below a millionth of the band, which is the
# unfaired curve for every practical purpose.
_CURVATURE_FAIRNESS_RETREAT_STEPS = 20
# Halvings of the fit tolerance the reconstruction may spend to satisfy its
# acceptance gates. Four attempts reach one eighth of the requested tolerance,
# by which point the profile is tracing the section closely enough that any
# remaining rejection is a real defect in the recovered surface, not a fit that
# was allowed to wander.
_FIT_TOLERANCE_RETRY_STEPS = 4


def _extrusion_axis_index(value: Any) -> Optional[int]:
    """Return a Cartesian axis index for an explicit extrusion constraint."""
    key = str(value or "").strip().lower()
    return _AXIS_INDEX.get(key)


def _signed_area(points_2d: np.ndarray) -> float:
    """Return the signed area of a closed 2-D polygon."""
    p = np.asarray(points_2d, dtype=float)
    return 0.5 * float(
        np.sum(p[:, 0] * np.roll(p[:, 1], -1)) - np.sum(p[:, 1] * np.roll(p[:, 0], -1))
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Even/odd point-in-polygon test for non-intersecting section loops."""
    point = np.asarray(point, dtype=float)[:2]
    poly = np.asarray(polygon, dtype=float)[:, :2]
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi = float(poly[i, 1])
        yj = float(poly[j, 1])
        if (yi > point[1]) != (yj > point[1]):
            x_cross = (float(poly[j, 0]) - float(poly[i, 0])) * (
                float(point[1]) - yi
            ) / (yj - yi) + float(poly[i, 0])
            if float(point[0]) < x_cross:
                inside = not inside
        j = i
    return inside


def _loop_nesting_depths(loops_2d: list[np.ndarray]) -> list[int]:
    """Return containment depth for each disjoint section loop."""
    depths: list[int] = []
    areas = [abs(_signed_area(loop)) for loop in loops_2d]
    for index, loop in enumerate(loops_2d):
        # A point on this loop cannot lie on a different non-intersecting loop,
        # so it is a stable containment probe even for strongly concave shapes.
        probe = loop[0]
        depth = sum(
            1
            for other_index, other in enumerate(loops_2d)
            if other_index != index
            and areas[other_index] > areas[index]
            and _point_in_polygon(probe, other)
        )
        depths.append(depth)
    return depths


def _profile_wire(
    points: np.ndarray,
    *,
    fit_tolerance: float,
) -> tuple[Any, int, dict[str, float]]:
    """Create one closed C2 B-spline curve with four stable OCCT trims."""
    import cadquery as cq

    fair, control_count, target_area, fairness_report = _periodic_fair_profile(
        points,
        fit_tolerance=fit_tolerance,
    )

    def make_wire(values: np.ndarray) -> tuple[Any, Any]:
        vectors = [
            cq.Vector(*(float(value) for value in point))
            for point in values
        ]
        periodic_edge = cq.Edge.makeSpline(
            vectors,
            periodic=True,
            tol=max(float(fit_tolerance) * 1.0e-4, 1.0e-9),
        )
        adaptor = periodic_edge._geomAdaptor()
        boundaries = np.linspace(
            adaptor.FirstParameter(),
            adaptor.LastParameter(),
            5,
        )
        # A complex periodic OCCT edge is geometrically closed but is not a
        # reliable single-edge boundary for planar face/prism construction.
        # Four trims of the *same* C2 curve avoid that kernel ambiguity; they
        # are parameter spans, not independent fitted patches or a fallback.
        edges = [
            periodic_edge.trim(boundaries[index], boundaries[index + 1])
            for index in range(4)
        ]
        return periodic_edge, cq.Wire.assembleEdges(edges)

    edge, wire = make_wire(fair)
    normal_axis = int(np.argmin(np.ptp(fair, axis=0)))
    unit_sweep = np.zeros(3, dtype=float)
    unit_sweep[normal_axis] = 1.0
    kernel_area = float(
        cq.Solid.extrudeLinear(
            wire,
            [],
            cq.Vector(*(float(value) for value in unit_sweep)),
        ).Volume()
    )
    if not np.isfinite(kernel_area) or kernel_area <= 1.0e-20:
        raise RuntimeError("Periodic B-spline profile has no usable section area.")

    # OCCT interpolates the already-faired construction samples into its own
    # B-spline basis. Correct the small area drift on that actual kernel curve,
    # not just on the sampled SciPy curve used above.
    area_scale = float(np.sqrt(float(target_area) / kernel_area))
    if abs(area_scale - 1.0) > 1.0e-10:
        in_plane = np.argsort(np.ptp(fair, axis=0))[-2:]
        centre = np.mean(fair[:, in_plane], axis=0)
        fair[:, in_plane] = centre + area_scale * (
            fair[:, in_plane] - centre
        )
        edge, wire = make_wire(fair)
    if not edge.isValid() or not wire.isValid() or not wire.IsClosed():
        raise RuntimeError("Periodic B-spline profile did not produce a valid closed wire.")
    return wire, control_count, fairness_report


def _uniform_closed_samples(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed polyline uniformly in arclength."""
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1.0e-12:
        points = points[:-1]
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    usable = lengths > 1.0e-12
    points = points[usable]
    segments = segments[usable]
    lengths = lengths[usable]
    if len(points) < 4 or float(np.sum(lengths)) <= 1.0e-12:
        raise RuntimeError("A profile loop collapsed during arclength resampling.")

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], int(count), endpoint=False)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    fractions = (targets - cumulative[indices]) / lengths[indices]
    return points[indices] + fractions[:, None] * segments[indices]


def _unfaired_report(*, sample_count: int = 1) -> dict[str, float]:
    """The fairness report for a profile that was left as it was fitted."""
    return {
        "curvature_rms_before": 0.0,
        "curvature_rms_after": 0.0,
        "curvature_variation_rms_before": 0.0,
        "curvature_variation_rms_after": 0.0,
        "fit_rms": 0.0,
        "fit_max_correspondence_error": 0.0,
        "fairing_deviation_band": 0.0,
        "fairing_max_deviation": 0.0,
        "fairing_blend": 0.0,
        "fairing_applied": 0.0,
        "fairing_iterations": 0,
        "fairing_sample_count": int(max(sample_count, 1)),
        "fairing_verify_sample_count": 0,
    }


def _curvature_fair_periodic_spline(
    spline: tuple[Any, Any, int],
    source_points: np.ndarray,
    *,
    in_plane: np.ndarray,
    target_area: float,
    fit_tolerance: float,
) -> tuple[tuple[Any, Any, int], dict[str, float]]:
    """Apply the Wang-Ma curvature fairness energy to a periodic 2-D profile.

    Equation 9 of Wang and Ma (CAGD 127, 2026, 102557) balances surface
    curvature, neighbouring curvature variation, and least-squares fitting.
    An extruded side wall has zero Gaussian curvature regardless of its planar
    outline, so its correct dimensional reduction is curve curvature and the
    arc-length variation of curve curvature. The paper's SQP scheme and
    illustrated 2:2:1 penalty ratio are retained here.

    Fidelity is held pointwise, not on average. The classical constrained
    fairing statement bounds the maximum permissible error between
    parametrically corresponding points of the initial and faired curve,
    ``rho >= ||Q0(u) - Qf(u)||`` for every ``u``, and that is what is imposed
    below. An aggregate mean-square budget is the wrong dual for the acceptance
    gate this feeds: coverage is decided one cell centre at a time, so an
    optimizer that only has to keep an average can -- and does -- pay for a
    large local excursion with a great many near-zero ones. Curvature
    minimization concentrates its correction exactly where curvature is
    highest, which on a topology profile is the strut junction roots where the
    protected centres sit. Measured on a four-strut cantilever section: the
    mean-square form drove its budget to the bound (fit RMS 0.1242 against a
    0.1241 limit) while placing a 0.7579 pointwise excursion, 6.1x the RMS and
    1.38x the entire coverage envelope it had to live inside.

    The band is enforced twice: as a per-sample SQP constraint so the optimizer
    shapes a feasible curve rather than being clipped afterwards, and again as
    a postcondition on a denser sample set after exact area restoration. If the
    postcondition fails, the perturbation retreats toward the unfaired curve
    until it holds. Fairing therefore cannot make the profile worse than the
    approximation it started from, and never raises: a curve that will not fair
    within tolerance is returned unfaired.
    """
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize

    knots, coefficients_raw, degree = spline
    coefficients = np.asarray(coefficients_raw, dtype=float)
    degree = int(degree)
    unique_count = int(coefficients.shape[1] - degree)
    if unique_count < degree + 1:
        raise RuntimeError("Periodic spline has too few independent control points.")

    # FITPACK stores the first `degree` periodic coefficients again at the end.
    # This matrix exposes only the independent controls while retaining exact
    # C2 periodicity throughout optimization.
    wrapping = np.zeros((coefficients.shape[1], unique_count), dtype=float)
    wrapping[:unique_count, :] = np.eye(unique_count)
    wrapping[unique_count:, :degree] = np.eye(degree)
    basis_spline = BSpline(knots, wrapping, degree, extrapolate=False)

    sample_count = max(128, min(512, 8 * unique_count))
    parameters = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    basis = np.asarray(basis_spline(parameters), dtype=float)
    basis_first = np.asarray(basis_spline.derivative(1)(parameters), dtype=float)
    basis_second = np.asarray(basis_spline.derivative(2)(parameters), dtype=float)
    target = _uniform_closed_samples(source_points, sample_count)[:, in_plane]
    controls = coefficients[in_plane, :unique_count].T.copy()

    perimeter = float(
        np.sum(np.linalg.norm(np.roll(target, -1, axis=0) - target, axis=1))
    )
    if perimeter <= 1.0e-12:
        raise RuntimeError("Periodic profile has no usable perimeter for fairing.")
    characteristic_radius = float(np.sqrt(abs(float(target_area)) / np.pi))
    arc_step = perimeter / float(sample_count)
    tolerance_sq = max(float(fit_tolerance), 1.0e-9) ** 2

    def signed_area(values: np.ndarray) -> float:
        return _signed_area(values)

    # Start SQP on the exact signed-area manifold. Uniform scaling preserves
    # curve topology and gives the equality constraint a numerically clean
    # initial point.
    initial_curve = basis @ controls
    initial_area = signed_area(initial_curve)
    if initial_area * float(target_area) <= 0.0 or abs(initial_area) <= 1.0e-20:
        raise RuntimeError("Initial periodic spline has invalid signed area.")
    initial_centre = np.mean(initial_curve, axis=0)
    initial_scale = float(np.sqrt(abs(float(target_area) / initial_area)))
    controls = initial_centre + initial_scale * (controls - initial_centre)

    # Everything downstream measures against this curve, so it is captured once
    # the controls sit on the area manifold and never recomputed.
    initial_controls = controls.copy()
    reference_curve = basis @ initial_controls

    def energies_and_gradient(
        flat_controls: np.ndarray,
        *,
        with_gradient: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        control_points = np.asarray(flat_controls, dtype=float).reshape(
            unique_count,
            2,
        )
        curve = basis @ control_points
        first = basis_first @ control_points
        second = basis_second @ control_points
        x_first, y_first = first[:, 0], first[:, 1]
        x_second, y_second = second[:, 0], second[:, 1]
        speed_sq = np.maximum(
            x_first * x_first + y_first * y_first,
            1.0e-18,
        )
        numerator = x_first * y_second - y_first * x_second
        curvature = numerator / np.power(speed_sq, 1.5)
        curvature_delta = np.roll(curvature, -1) - curvature
        residual = curve - target

        energies = np.asarray(
            (
                np.mean((curvature * characteristic_radius) ** 2),
                np.mean(
                    (
                        curvature_delta
                        * characteristic_radius**2
                        / arc_step
                    )
                    ** 2
                ),
                np.mean(np.sum(residual * residual, axis=1)) / tolerance_sq,
            ),
            dtype=float,
        )
        if not with_gradient:
            return energies, None, curvature, curve

        inverse_speed_cubed = np.power(speed_sq, -1.5)
        denominator_term = -1.5 * numerator * np.power(speed_sq, -2.5)
        curvature_gradient_x = (
            inverse_speed_cubed[:, None]
            * (
                basis_first * y_second[:, None]
                - basis_second * y_first[:, None]
            )
            + denominator_term[:, None]
            * (2.0 * x_first[:, None] * basis_first)
        )
        curvature_gradient_y = (
            inverse_speed_cubed[:, None]
            * (
                basis_second * x_first[:, None]
                - basis_first * x_second[:, None]
            )
            + denominator_term[:, None]
            * (2.0 * y_first[:, None] * basis_first)
        )

        curvature_x = (
            2.0
            * characteristic_radius**2
            / sample_count
            * (curvature_gradient_x.T @ curvature)
        )
        curvature_y = (
            2.0
            * characteristic_radius**2
            / sample_count
            * (curvature_gradient_y.T @ curvature)
        )
        delta_gradient_x = (
            np.roll(curvature_gradient_x, -1, axis=0) - curvature_gradient_x
        )
        delta_gradient_y = (
            np.roll(curvature_gradient_y, -1, axis=0) - curvature_gradient_y
        )
        variation_factor = (
            2.0
            * characteristic_radius**4
            / (sample_count * arc_step**2)
        )
        variation_x = variation_factor * (
            delta_gradient_x.T @ curvature_delta
        )
        variation_y = variation_factor * (
            delta_gradient_y.T @ curvature_delta
        )
        fitting_x = (
            2.0
            / (sample_count * tolerance_sq)
            * (basis.T @ residual[:, 0])
        )
        fitting_y = (
            2.0
            / (sample_count * tolerance_sq)
            * (basis.T @ residual[:, 1])
        )
        gradients = np.vstack(
            (
                np.column_stack((curvature_x, curvature_y)).reshape(-1),
                np.column_stack((variation_x, variation_y)).reshape(-1),
                np.column_stack((fitting_x, fitting_y)).reshape(-1),
            )
        )
        return energies, gradients, curvature, curve

    initial_flat = controls.reshape(-1)
    initial_energies, _, initial_curvature, initial_curve = (
        energies_and_gradient(initial_flat, with_gradient=False)
    )
    normalization = np.asarray(
        (
            max(float(initial_energies[0]), 1.0e-8),
            max(float(initial_energies[1]), 1.0e-8),
            1.0,
        ),
        dtype=float,
    )
    penalty_weights = (
        np.asarray(_CURVATURE_FAIRNESS_WEIGHTS, dtype=float) / normalization
    )

    def objective(flat_controls: np.ndarray) -> float:
        energies, _, _, _ = energies_and_gradient(
            flat_controls,
            with_gradient=False,
        )
        return float(penalty_weights @ energies)

    def objective_jacobian(flat_controls: np.ndarray) -> np.ndarray:
        _, gradients, _, _ = energies_and_gradient(
            flat_controls,
            with_gradient=True,
        )
        assert gradients is not None
        return penalty_weights @ gradients

    area_scale = max(abs(float(target_area)), 1.0e-12)

    def area_constraint(flat_controls: np.ndarray) -> float:
        curve = basis @ np.asarray(flat_controls).reshape(unique_count, 2)
        return (signed_area(curve) - float(target_area)) / area_scale

    def area_constraint_jacobian(flat_controls: np.ndarray) -> np.ndarray:
        curve = basis @ np.asarray(flat_controls).reshape(unique_count, 2)
        area_gradient_x = 0.5 * (
            np.roll(curve[:, 1], -1) - np.roll(curve[:, 1], 1)
        )
        area_gradient_y = 0.5 * (
            np.roll(curve[:, 0], 1) - np.roll(curve[:, 0], -1)
        )
        return (
            np.column_stack(
                (
                    basis.T @ area_gradient_x,
                    basis.T @ area_gradient_y,
                )
            ).reshape(-1)
            / area_scale
        )

    deviation_band = max(
        _CURVATURE_FAIRNESS_DEVIATION_BAND_FRACTION
        * max(float(fit_tolerance), 1.0e-9),
        1.0e-12,
    )
    band_sq = deviation_band**2

    # One inequality per sample, not one for the whole curve. `minimize` accepts
    # a vector-valued constraint, so this stays a single SLSQP constraint block
    # whose Jacobian is assembled in closed form.
    def band_constraint(flat_controls: np.ndarray) -> np.ndarray:
        curve = basis @ np.asarray(flat_controls).reshape(unique_count, 2)
        offset = curve - reference_curve
        return (band_sq - np.sum(offset * offset, axis=1)) / band_sq

    def band_constraint_jacobian(flat_controls: np.ndarray) -> np.ndarray:
        curve = basis @ np.asarray(flat_controls).reshape(unique_count, 2)
        offset = curve - reference_curve
        jacobian = np.empty((sample_count, unique_count, 2), dtype=float)
        jacobian[:, :, 0] = -2.0 * offset[:, 0:1] * basis
        jacobian[:, :, 1] = -2.0 * offset[:, 1:2] * basis
        return jacobian.reshape(sample_count, 2 * unique_count) / band_sq

    result = minimize(
        objective,
        initial_flat,
        jac=objective_jacobian,
        method="SLSQP",
        constraints=(
            {
                "type": "eq",
                "fun": area_constraint,
                "jac": area_constraint_jacobian,
            },
            {
                "type": "ineq",
                "fun": band_constraint,
                "jac": band_constraint_jacobian,
            },
        ),
        options={"maxiter": 300, "ftol": 1.0e-8, "disp": False},
    )
    # The SQP band is imposed at `sample_count` parameters; between them the
    # curve is free to bulge. Verification runs on a four-times denser set, and
    # on the geometry that will actually be built -- after the signed area is
    # restored exactly, since that restoration is itself a displacement.
    verify_count = int(max(4 * sample_count, 512))
    verify_parameters = np.linspace(0.0, 1.0, verify_count, endpoint=False)
    verify_basis = np.asarray(basis_spline(verify_parameters), dtype=float)

    def area_restored(
        control_points: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Scale controls about the curve centroid onto the exact target area.

        A B-spline is affine invariant and its basis is a partition of unity,
        so scaling the controls about a point scales the curve about that same
        point. The area correction is therefore exact rather than iterative.
        """
        curve = verify_basis @ control_points
        if not np.all(np.isfinite(curve)):
            return None, None
        area = _signed_area(curve)
        if not np.isfinite(area) or area * float(target_area) <= 0.0:
            return None, None
        centre = np.mean(curve, axis=0)
        scale = float(np.sqrt(abs(float(target_area) / area)))
        return (
            centre + scale * (control_points - centre),
            centre + scale * (curve - centre),
        )

    verify_initial_controls, verify_initial_curve = area_restored(initial_controls)
    if verify_initial_curve is None:
        raise RuntimeError("Initial periodic spline could not be area-normalized.")

    solver_controls = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(solver_controls)):
        logger.debug("SQP fairing returned non-finite controls; leaving profile unfaired.")
        solver_controls = initial_controls.reshape(-1)
    solver_controls = solver_controls.reshape(unique_count, 2)

    # Retreat along the segment from the unfaired controls to the SQP result
    # until the dense band postcondition holds. The endpoint at blend zero is
    # the unfaired curve, whose excursion is zero by construction, so this
    # terminates. Fairing can only ever be given back, never forced through.
    blend = 1.0
    optimized_controls = verify_initial_controls
    band_excursion = 0.0
    for _retreat in range(_CURVATURE_FAIRNESS_RETREAT_STEPS):
        trial = initial_controls + blend * (solver_controls - initial_controls)
        scaled_controls, scaled_curve = area_restored(trial)
        if scaled_controls is not None:
            excursion = float(
                np.max(np.linalg.norm(scaled_curve - verify_initial_curve, axis=1))
            )
            if excursion <= deviation_band:
                optimized_controls = scaled_controls
                band_excursion = excursion
                break
        blend *= 0.5
    else:
        blend = 0.0
        logger.debug(
            "SQP fairing stayed outside its %g deviation band after %d retreats; "
            "profile left unfaired.",
            deviation_band,
            _CURVATURE_FAIRNESS_RETREAT_STEPS,
        )

    final_energies, _, final_curvature, final_curve = energies_and_gradient(
        optimized_controls.reshape(-1),
        with_gradient=False,
    )
    # A retreated iterate is a different point from the one SLSQP evaluated, so
    # whether it is actually fairer is re-checked here rather than inherited
    # from the solver. Neither curvature term may grow.
    fairness_improved = bool(
        np.all(np.isfinite(final_energies))
        and final_energies[0] <= initial_energies[0] * (1.0 + 1.0e-6)
        and final_energies[1] <= initial_energies[1] * (1.0 + 1.0e-6)
    )
    if blend > 0.0 and not fairness_improved:
        logger.debug(
            "SQP fairing did not reduce either curvature term; profile left unfaired."
        )
        blend = 0.0
    if blend <= 0.0:
        optimized_controls = verify_initial_controls
        band_excursion = 0.0
        final_energies, _, final_curvature, final_curve = energies_and_gradient(
            optimized_controls.reshape(-1),
            with_gradient=False,
        )
    elif not result.success:
        logger.debug(
            "Accepted validated SQP fairing iterate after solver status %d: %s.",
            result.status,
            result.message,
        )

    optimized_coefficients = coefficients.copy()
    for local_axis, model_axis in enumerate(in_plane):
        optimized_coefficients[model_axis, :unique_count] = (
            optimized_controls[:, local_axis]
        )
        optimized_coefficients[model_axis, unique_count:] = (
            optimized_controls[:degree, local_axis]
        )

    initial_delta = np.roll(initial_curvature, -1) - initial_curvature
    final_delta = np.roll(final_curvature, -1) - final_curvature
    final_residual = final_curve - target
    report = {
        "curvature_rms_before": float(
            np.sqrt(np.mean(initial_curvature**2))
        ),
        "curvature_rms_after": float(np.sqrt(np.mean(final_curvature**2))),
        "curvature_variation_rms_before": float(
            np.sqrt(np.mean((initial_delta / arc_step) ** 2))
        ),
        "curvature_variation_rms_after": float(
            np.sqrt(np.mean((final_delta / arc_step) ** 2))
        ),
        "fit_rms": float(
            np.sqrt(np.mean(np.sum(final_residual**2, axis=1)))
        ),
        "fit_max_correspondence_error": float(
            np.max(np.linalg.norm(final_residual, axis=1))
        ),
        "fairing_deviation_band": float(deviation_band),
        "fairing_max_deviation": float(band_excursion),
        "fairing_blend": float(blend),
        "fairing_applied": float(1.0 if blend > 0.0 else 0.0),
        "fairing_iterations": int(result.nit),
        "fairing_sample_count": int(sample_count),
        "fairing_verify_sample_count": int(verify_count),
    }
    optimized_spline = (
        knots,
        [optimized_coefficients[index] for index in range(coefficients.shape[0])],
        degree,
    )
    return optimized_spline, report


def _periodic_fair_profile(
    points: np.ndarray,
    *,
    fit_tolerance: float,
) -> tuple[np.ndarray, int, float, dict[str, float]]:
    """Fit one sparse periodic cubic spline with curvature-constrained fairing.

    Plane/triangle intersection samples are uneven and contain voxel-frequency
    noise. Interpolating those samples keeps the staircase in the CAD topology.
    The initial arclength-parameterized approximation establishes sparse knots.
    Its controls are then optimized for curve curvature, curvature variation,
    and contour fidelity before one periodic OCCT B-spline is constructed.
    """
    from scipy.interpolate import splev, splprep

    original = np.asarray(points, dtype=float)
    source_count = max(64, min(1024, 2 * len(original)))
    source = _uniform_closed_samples(original, source_count)
    closed_source = np.vstack((source, source[0]))
    parameters = np.linspace(0.0, 1.0, len(closed_source))

    # FITPACK bounds the summed squared residual. Scaling by the sample count
    # gives a geometric RMS tolerance independent of the contour resolution.
    fairing_distance = max(float(fit_tolerance), 1.0e-7)
    smoothing_bound = float(len(closed_source)) * fairing_distance**2
    spline, _ = splprep(
        closed_source.T,
        u=parameters,
        k=3,
        s=smoothing_bound,
        per=True,
    )

    in_plane = np.argsort(np.ptp(original, axis=0))[-2:]
    area_before = _signed_area(original[:, in_plane])
    if abs(area_before) <= 1.0e-20:
        raise RuntimeError("Periodic profile has no usable signed area.")
    # Fairing is an improvement on the approximation above, not a precondition
    # for it. A profile the fairing cannot handle -- too few controls, a
    # degenerate perimeter, a solver that will not run -- is still a perfectly
    # buildable profile, so a failure here costs smoothness, never the body.
    try:
        spline, fairness_report = _curvature_fair_periodic_spline(
            spline,
            original,
            in_plane=in_plane,
            target_area=area_before,
            fit_tolerance=fit_tolerance,
        )
    except Exception:
        logger.debug(
            "Curvature fairing unavailable for this profile loop; "
            "using the fitted approximation.",
            exc_info=True,
        )
        fairness_report = _unfaired_report(sample_count=len(original))

    dense_count = max(256, min(2048, 4 * len(original)))
    dense_parameters = np.linspace(0.0, 1.0, dense_count, endpoint=False)
    fair = np.column_stack(splev(dense_parameters, spline))

    # Fairing can shrink a closed contour. Restore signed section area so it
    # cannot silently remove material or close holes.
    area_after = _signed_area(fair[:, in_plane])
    if (
        abs(area_before) <= 1.0e-20
        or abs(area_after) <= 1.0e-20
        or area_before * area_after <= 0.0
    ):
        raise RuntimeError("Periodic spline fitting reversed or collapsed a profile loop.")
    scale = float(np.sqrt(abs(area_before / area_after)))
    centre = np.mean(fair[:, in_plane], axis=0)

    # Uniform construction samples prevent global interpolation ringing and
    # keep the CAD edge count independent of the source mesh density.
    construction_count = max(48, min(256, int(np.ceil(len(original) / 2.0))))
    construction_parameters = np.linspace(
        0.0,
        1.0,
        construction_count,
        endpoint=False,
    )
    construction_points = np.column_stack(splev(construction_parameters, spline))
    construction_points[:, in_plane] = centre + scale * (
        construction_points[:, in_plane] - centre
    )
    control_count = max(0, int(len(spline[1][0]) - 3))
    return (
        construction_points,
        control_count,
        abs(float(area_before)),
        fairness_report,
    )


def _section_profile_loops(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    axis: int,
    plane_coordinate: float,
    closure_tolerance: float,
) -> list[np.ndarray]:
    """Intersect a triangle surface with a plane and return closed 3-D loops."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    origin = np.zeros(3, dtype=float)
    origin[axis] = float(plane_coordinate)
    segments = trimesh.intersections.mesh_plane(
        mesh,
        plane_normal=normal,
        plane_origin=origin,
    )
    if segments is None or len(segments) == 0:
        return []

    path = trimesh.load_path(np.asarray(segments, dtype=float))
    loops: list[np.ndarray] = []
    for discrete in path.discrete:
        loop = np.asarray(discrete, dtype=float)
        if len(loop) < 4:
            continue
        if np.linalg.norm(loop[0] - loop[-1]) > float(closure_tolerance):
            continue
        loop = loop[:-1]
        if len(loop) < 3:
            continue
        delta = np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1)
        loop = loop[delta > max(float(closure_tolerance) * 0.05, 1e-12)]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _sampled_surface_deviation(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    candidate: Any,
    *,
    tessellation_tolerance: float,
    sample_limit: int = 2500,
) -> float:
    """Estimate symmetric surface deviation between mesh and fitted B-rep."""
    import trimesh

    source = trimesh.Trimesh(
        vertices=np.asarray(source_vertices, dtype=float),
        faces=np.asarray(source_faces, dtype=np.int64),
        process=False,
    )
    candidate_vertices, candidate_faces = candidate.tessellate(
        max(float(tessellation_tolerance), 1e-6)
    )
    candidate_mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [[point.x, point.y, point.z] for point in candidate_vertices],
            dtype=float,
        ),
        faces=np.asarray(candidate_faces, dtype=np.int64),
        process=False,
    )

    def bounded_samples(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= sample_limit:
            return points
        indices = np.linspace(0, len(points) - 1, sample_limit, dtype=int)
        return points[indices]

    def distances(mesh: Any, points: np.ndarray) -> np.ndarray:
        try:
            _, values, _ = trimesh.proximity.closest_point(mesh, points)
        except Exception:
            _, values, _ = trimesh.proximity.closest_point_naive(mesh, points)
        return np.asarray(values, dtype=float)

    forward = distances(candidate_mesh, bounded_samples(source.vertices))
    reverse = distances(source, bounded_samples(candidate_mesh.vertices))
    return float(max(np.max(forward, initial=0.0), np.max(reverse, initial=0.0)))


def _sampled_extruded_profile_deviation(
    source_loops: list[np.ndarray],
    candidate_wires: list[Any],
    *,
    in_plane: list[int],
    sample_spacing: float,
) -> float:
    """Estimate symmetric deviation directly on exact extrusion profiles.

    Both source and candidate are constant sections swept over the same two
    end planes, so their 3-D surface Hausdorff distance is the 2-D section
    distance. Testing 50k triangulated side-wall faces with general 3-D nearest
    triangles made a compact spline CAD validation take tens of seconds. A
    densely sampled profile and cKDTree provide the same conservative sampled
    gate in a fraction of that time.
    """
    from scipy.spatial import cKDTree

    spacing = max(float(sample_spacing), 1.0e-5)

    def _densify(points: np.ndarray, *, closed: bool) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            return values
        pairs = list(zip(values[:-1], values[1:]))
        if closed:
            pairs.append((values[-1], values[0]))
        samples: list[np.ndarray] = []
        for start, stop in pairs:
            length = float(np.linalg.norm(stop - start))
            count = max(1, int(np.ceil(length / spacing)))
            fractions = np.arange(count, dtype=float) / float(count)
            samples.append(
                start[None, :]
                + fractions[:, None] * (stop - start)[None, :]
            )
        return np.vstack(samples) if samples else values

    source_samples = [
        _densify(np.asarray(loop, dtype=float)[:, :2], closed=True)
        for loop in source_loops
        if len(loop) >= 2
    ]
    candidate_samples: list[np.ndarray] = []
    for wire in candidate_wires:
        edge_polylines: list[np.ndarray] = []
        for edge in wire.Edges():
            points, _parameters = edge.sample(max(spacing * 0.25, 1.0e-5))
            edge_points = np.asarray(
                [
                    [float(point.x), float(point.y), float(point.z)]
                    for point in points
                ],
                dtype=float,
            )
            if len(edge_points):
                edge_polylines.append(edge_points[:, in_plane])
        if edge_polylines:
            ordered = [edge_polylines.pop(0)]
            while edge_polylines:
                endpoint = ordered[-1][-1]
                candidates = [
                    (
                        min(
                            float(np.linalg.norm(polyline[0] - endpoint)),
                            float(np.linalg.norm(polyline[-1] - endpoint)),
                        ),
                        index,
                    )
                    for index, polyline in enumerate(edge_polylines)
                ]
                _distance, index = min(candidates)
                next_polyline = edge_polylines.pop(index)
                if np.linalg.norm(next_polyline[-1] - endpoint) < np.linalg.norm(
                    next_polyline[0] - endpoint
                ):
                    next_polyline = next_polyline[::-1]
                ordered.append(next_polyline)
            candidate_samples.append(
                _densify(np.vstack(ordered), closed=True)
            )

    if not source_samples or not candidate_samples:
        return float("inf")
    source = np.vstack(source_samples)
    candidate = np.vstack(candidate_samples)
    source_to_candidate = cKDTree(candidate).query(source, k=1)[0]
    candidate_to_source = cKDTree(source).query(candidate, k=1)[0]
    return float(
        max(
            np.max(source_to_candidate, initial=0.0),
            np.max(candidate_to_source, initial=0.0),
        )
    )


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> Optional[float]:
    """Return summed absolute component volume for a watertight source mesh."""
    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        parts = mesh.split(only_watertight=False)
        volume = float(sum(abs(float(part.volume)) for part in parts))
        return volume if np.isfinite(volume) and volume > 1e-12 else None
    except Exception:
        return None


def _extruded_spline_brep(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    extrusion_axis: Any,
    absolute_fit_tolerance: float,
    relative_fit_tolerance: float,
    crease_angle_deg: float,
    maximum_volume_delta: float,
    maximum_relative_deviation: float,
    protected_points: Optional[np.ndarray] = None,
) -> tuple[Any, dict[str, Any]]:
    """Build and validate a compact spline B-rep for a prismatic result."""
    import cadquery as cq

    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    axis = _extrusion_axis_index(extrusion_axis)
    if axis is None:
        raise RuntimeError(
            "Smooth spline reconstruction currently requires an explicit "
            "X, Y, or Z extrusion constraint."
        )

    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    requested_fit_tolerance = max(
        float(absolute_fit_tolerance or 0.0),
        float(relative_fit_tolerance or 0.0) * diagonal,
        1e-7,
    )
    height = float(upper[axis] - lower[axis])
    if not np.isfinite(height) or height <= requested_fit_tolerance:
        raise RuntimeError("Extrusion-constrained result has no usable sweep height.")

    plane_coordinate = 0.5 * float(lower[axis] + upper[axis])
    # The section loops are source data, not a fitted result, so their closure
    # tolerance stays at the requested value across every attempt below. Letting
    # it move would change the profile topology mid-retry.
    loops = _section_profile_loops(
        vertices,
        faces,
        axis=axis,
        plane_coordinate=plane_coordinate,
        closure_tolerance=max(requested_fit_tolerance * 0.25, 1e-8),
    )
    if not loops:
        raise RuntimeError("No closed mid-plane profile was found for spline fitting.")

    in_plane = [index for index in range(3) if index != axis]
    loops_2d = [loop[:, in_plane] for loop in loops]
    depths = _loop_nesting_depths(loops_2d)
    source_volume = _mesh_volume(vertices, faces)
    sweep = np.zeros(3, dtype=float)
    sweep[axis] = height

    def _attempt(fit_tolerance: float) -> dict[str, Any]:
        """Fit, build and gate one candidate body at a given fit tolerance."""
        wires: list[Any] = []
        profile_control_points = 0
        fairness_reports: list[dict[str, float]] = []
        for loop in loops:
            profile = loop.copy()
            profile[:, axis] = float(lower[axis])
            wire, control_count, fairness_report = _profile_wire(
                profile,
                fit_tolerance=fit_tolerance,
            )
            wires.append(wire)
            profile_control_points += int(control_count)
            fairness_reports.append(fairness_report)

        solids: list[Any] = []
        for index, depth in enumerate(depths):
            if depth % 2:
                continue
            hole_indices = [
                hole_index
                for hole_index, hole_depth in enumerate(depths)
                if hole_depth == depth + 1
                and _point_in_polygon(loops_2d[hole_index][0], loops_2d[index])
            ]
            material = cq.Solid.extrudeLinear(
                wires[index],
                [],
                cq.Vector(*(float(value) for value in sweep)),
            )
            if hole_indices:
                hole_solids = [
                    cq.Solid.extrudeLinear(
                        wires[hole_index],
                        [],
                        cq.Vector(*(float(value) for value in sweep)),
                    )
                    for hole_index in hole_indices
                ]
                material = material.cut(cq.Compound.makeCompound(hole_solids))
            solids.append(material)

        if not solids:
            raise RuntimeError("Spline profiles produced no extrudable material region.")
        candidate: Any = (
            solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
        )
        if not candidate.isValid():
            raise RuntimeError("Spline profile extrusion produced an invalid CAD body.")

        candidate_volume = float(candidate.Volume())
        volume_delta = None
        if source_volume is not None:
            volume_delta = (candidate_volume - source_volume) / source_volume
            if not np.isfinite(volume_delta) or abs(float(volume_delta)) > float(
                maximum_volume_delta
            ):
                raise RuntimeError(
                    "Spline reconstruction changed enclosed volume by "
                    f"{float(volume_delta):+.2%}; the limit is "
                    f"{float(maximum_volume_delta):.2%}."
                )

        sampled_deviation = _sampled_extruded_profile_deviation(
            loops_2d,
            wires,
            in_plane=in_plane,
            sample_spacing=max(fit_tolerance * 0.35, diagonal * 5.0e-5),
        )
        deviation_limit = max(
            fit_tolerance * 4.0,
            float(maximum_relative_deviation) * diagonal,
        )
        if not np.isfinite(sampled_deviation) or sampled_deviation > deviation_limit:
            raise RuntimeError(
                "Spline reconstruction sampled surface deviation is "
                f"{sampled_deviation:g}, above the {deviation_limit:g} limit."
            )

        coverage = protected_point_coverage(
            candidate,
            protected_points,
            fit_tolerance=fit_tolerance,
            description="Spline reconstruction",
        )
        return {
            "candidate": candidate,
            "wires": wires,
            "fairness_reports": fairness_reports,
            "profile_control_points": profile_control_points,
            "candidate_volume": candidate_volume,
            "volume_delta": volume_delta,
            "sampled_deviation": sampled_deviation,
            "deviation_limit": deviation_limit,
            "coverage": coverage,
            "fit_tolerance": float(fit_tolerance),
        }

    # Every stage that shapes the profile -- FITPACK's smoothing bound, the
    # curvature fairing, the area corrections -- is controlled in a mean-square
    # sense, while acceptance is decided pointwise, one cell centre at a time.
    # A mean-square budget says nothing about its own worst point, so the
    # reconstruction negotiates instead of predicting: build at the requested
    # tolerance, and if a gate rejects the body, halve the tolerance and build
    # again. Tightening converges by construction -- as the tolerance falls the
    # profile approaches an interpolation of the section itself, whose coverage
    # is exact -- and it is only ever paid for on a body that would otherwise
    # have been thrown away. The cost is a denser profile, which is a far better
    # outcome than no CAD body at all.
    attempts: list[str] = []
    outcome: Optional[dict[str, Any]] = None
    for step in range(_FIT_TOLERANCE_RETRY_STEPS):
        fit_tolerance = requested_fit_tolerance * (0.5**step)
        try:
            outcome = _attempt(fit_tolerance)
            break
        except RuntimeError as error:
            attempts.append(f"{fit_tolerance:g}: {error}")
            if step + 1 >= _FIT_TOLERANCE_RETRY_STEPS:
                raise RuntimeError(
                    "Spline reconstruction could not satisfy its acceptance gates "
                    f"at any fit tolerance down to {fit_tolerance:g}. "
                    + " | ".join(attempts)
                ) from error
            logger.info(
                "Topology CAD: retrying spline reconstruction at fit tolerance "
                "%g after: %s",
                requested_fit_tolerance * (0.5 ** (step + 1)),
                error,
            )

    assert outcome is not None
    candidate = outcome["candidate"]
    wires = outcome["wires"]
    fairness_reports = outcome["fairness_reports"]
    profile_control_points = outcome["profile_control_points"]
    candidate_volume = outcome["candidate_volume"]
    volume_delta = outcome["volume_delta"]
    sampled_deviation = outcome["sampled_deviation"]
    deviation_limit = outcome["deviation_limit"]
    coverage = outcome["coverage"]
    fit_tolerance = outcome["fit_tolerance"]

    fairing_sample_count = int(
        sum(report["fairing_sample_count"] for report in fairness_reports)
    )

    def combined_rms(key: str) -> float:
        return float(
            np.sqrt(
                sum(
                    report[key] ** 2 * report["fairing_sample_count"]
                    for report in fairness_reports
                )
                / fairing_sample_count
            )
        )

    curvature_before = combined_rms("curvature_rms_before")
    curvature_after = combined_rms("curvature_rms_after")
    variation_before = combined_rms("curvature_variation_rms_before")
    variation_after = combined_rms("curvature_variation_rms_after")
    report = {
        "method": "Smooth",
        "representation": "trimmed spline/analytic B-rep",
        "editable": True,
        "smooth": True,
        "fallback_used": False,
        "profile_fairing_method": (
            "curvature-constrained periodic B-spline SQP"
        ),
        "profile_fairing_weights": {
            "curvature": _CURVATURE_FAIRNESS_WEIGHTS[0],
            "curvature_variation": _CURVATURE_FAIRNESS_WEIGHTS[1],
            "fitting": _CURVATURE_FAIRNESS_WEIGHTS[2],
        },
        "profile_fairing_deviation_band": float(
            max(report["fairing_deviation_band"] for report in fairness_reports)
        ),
        "profile_fairing_max_deviation": float(
            max(report["fairing_max_deviation"] for report in fairness_reports)
        ),
        "profile_fairing_loops_faired": int(
            sum(int(report["fairing_applied"]) for report in fairness_reports)
        ),
        "profile_curvature_rms_before": curvature_before,
        "profile_curvature_rms_after": curvature_after,
        "profile_curvature_reduction_pct": (
            100.0 * (1.0 - curvature_after / curvature_before)
            if curvature_before > 0.0
            else 0.0
        ),
        "profile_curvature_variation_rms_before": variation_before,
        "profile_curvature_variation_rms_after": variation_after,
        "profile_curvature_variation_reduction_pct": (
            100.0 * (1.0 - variation_after / variation_before)
            if variation_before > 0.0
            else 0.0
        ),
        "profile_fairing_fit_rms": combined_rms("fit_rms"),
        "profile_fairing_max_correspondence_error": float(
            max(
                report["fit_max_correspondence_error"]
                for report in fairness_reports
            )
        ),
        "profile_fairing_iterations": int(
            sum(report["fairing_iterations"] for report in fairness_reports)
        ),
        "profile_fairing_sample_count": fairing_sample_count,
        "extrusion_axis": "xyz"[axis].upper(),
        "profile_loops": len(loops),
        "profile_spline_edges": int(sum(len(wire.Edges()) for wire in wires)),
        "profile_spline_control_points": int(profile_control_points),
        "source_triangle_count": int(len(faces)),
        "cad_face_count": int(len(candidate.Faces())),
        "fit_tolerance": float(fit_tolerance),
        "requested_fit_tolerance": float(requested_fit_tolerance),
        "fit_tolerance_attempts": int(len(attempts) + 1),
        "max_sampled_surface_deviation": float(sampled_deviation),
        "maximum_allowed_surface_deviation": float(deviation_limit),
        **coverage,
        "source_mesh_volume": source_volume,
        "cad_volume_before_feature_healing": candidate_volume,
        "volume_delta_pct": (
            float(volume_delta * 100.0) if volume_delta is not None else None
        ),
    }
    logger.info(
        "Topology CAD: fitted %d section loop(s) into %d spline/analytic "
        "B-rep face(s) at fit tolerance %g (%d attempt(s)), max sampled "
        "deviation=%g, volume delta=%s.",
        len(loops),
        len(candidate.Faces()),
        fit_tolerance,
        len(attempts) + 1,
        sampled_deviation,
        f"{volume_delta:+.2%}" if volume_delta is not None else "unknown",
    )
    return candidate, report


__all__ = ["_extruded_spline_brep", "_extrusion_axis_index"]
