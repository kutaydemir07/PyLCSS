# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Numerical convergence, repeatability, and crash curve comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from pylcss.cae.crash_qualification import (
    BENCHMARK_CHANNELS,
    QualificationPolicy,
    validate_benchmark_metadata,
)
from pylcss.cae.crash_utils import as_array as _array, assessment_status, json_safe


DEFAULT_VALIDATION_METRICS = (
    "peak_crushing_force_kN",
    "mean_crushing_force_kN",
    "absorbed_energy_kJ",
    "useful_crush_stroke_mm",
    "peak_acceleration_g",
)


@dataclass(frozen=True)
class CurveComparisonPolicy:
    """Project thresholds for the custom PyLCSS engineering screening score."""

    pass_score: float = 0.80
    warning_score: float = 0.65
    peak_error_tolerance: float = 0.15
    integral_error_tolerance: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "pass_score",
            "warning_score",
            "peak_error_tolerance",
            "integral_error_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        if not 0 <= self.warning_score <= self.pass_score <= 1:
            raise ValueError(
                "Scores must satisfy 0 <= warning_score <= pass_score <= 1."
            )


def _relative_error(value: float, reference: float) -> float:
    scale = max(abs(value), abs(reference), 1.0e-12)
    return abs(value - reference) / scale


def _validate_tolerances(pass_tolerance: float, warning_tolerance: float) -> None:
    if (
        not math.isfinite(pass_tolerance)
        or not math.isfinite(warning_tolerance)
        or not 0 <= pass_tolerance <= warning_tolerance
    ):
        raise ValueError("Tolerances must be finite and satisfy 0 <= pass <= warning.")


def compare_time_histories(
    simulation_time_ms: Iterable[float],
    simulation_values: Iterable[float],
    reference_time_ms: Iterable[float],
    reference_values: Iterable[float],
    corridor_relative: float = 0.10,
    corridor_absolute: float = 0.0,
    sample_count: int = 1000,
    *,
    policy: CurveComparisonPolicy | None = None,
    axis_name: str = "time",
    axis_unit: str = "ms",
) -> dict[str, object]:
    """Compare curves with finite values and distinct coordinates on their overlap.

    Coordinates are sorted before interpolation. Each coordinate/value pair must
    have matching lengths. Repeated identical coordinate/value pairs are collapsed;
    conflicting values at a shared coordinate are rejected. Use
    axis_name='displacement', axis_unit='mm' for a crush curve. Values must already
    have compatible units and sign conventions.

    The weighted amplitude, shape, corridor, and integral score is a custom
    engineering screen. This function does not calculate an ISO rating or export
    aligned histories.
    """
    policy = policy if policy is not None else CurveComparisonPolicy()
    for value in (corridor_relative, corridor_absolute):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Corridor tolerances must be finite and nonnegative.")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count < 3
    ):
        raise ValueError("sample_count must be an integer of at least three.")
    if (
        not isinstance(axis_name, str)
        or not axis_name.strip()
        or not isinstance(axis_unit, str)
        or not axis_unit.strip()
    ):
        raise ValueError("Provide an axis name and unit.")
    sim_t = _array(simulation_time_ms)
    sim_y = _array(simulation_values)
    ref_t = _array(reference_time_ms)
    ref_y = _array(reference_values)
    if sim_t.size != sim_y.size or ref_t.size != ref_y.size:
        return {"status": "fail", "reason": "Coordinate and value lengths must match."}
    if sim_t.size < 3 or ref_t.size < 3:
        return {
            "status": "fail",
            "reason": "Both curves require at least three samples.",
        }
    if any(not np.all(np.isfinite(values)) for values in (sim_t, sim_y, ref_t, ref_y)):
        return {
            "status": "fail",
            "reason": "Curve coordinates and values must be finite.",
        }
    curves = []
    for coordinates, values in ((sim_t, sim_y), (ref_t, ref_y)):
        unique, first, inverse = np.unique(
            coordinates, return_index=True, return_inverse=True
        )
        if np.any(values != values[first][inverse]):
            return {
                "status": "fail",
                "reason": "Duplicate curve coordinates have conflicting values.",
            }
        if unique.size < 3:
            return {
                "status": "fail",
                "reason": "Both curves require at least three distinct coordinates.",
            }
        curves.append((unique, values[first]))
    (sim_t, sim_y), (ref_t, ref_y) = curves

    start = max(float(sim_t[0]), float(ref_t[0]))
    stop = min(float(sim_t[-1]), float(ref_t[-1]))
    if stop <= start:
        return {
            "status": "fail",
            "reason": "Curves do not overlap on the selected axis.",
        }
    grid = np.linspace(start, stop, int(sample_count))
    sim = np.interp(grid, sim_t, sim_y)
    ref = np.interp(grid, ref_t, ref_y)
    residual = sim - ref
    reference_range = max(float(np.ptp(ref)), float(np.max(np.abs(ref))), 1.0e-12)
    nrmse = float(np.sqrt(np.mean(residual**2)) / reference_range)
    if np.std(sim) > 1.0e-15 and np.std(ref) > 1.0e-15:
        correlation = float(np.corrcoef(sim, ref)[0, 1])
    else:
        correlation = 1.0 if np.allclose(sim, ref) else 0.0
    peak_error = _relative_error(float(np.max(np.abs(sim))), float(np.max(np.abs(ref))))
    sim_integral = float(np.trapezoid(sim, grid))
    ref_integral = float(np.trapezoid(ref, grid))
    integral_error = _relative_error(sim_integral, ref_integral)
    corridor = corridor_absolute + corridor_relative * np.maximum(
        np.abs(ref), 0.05 * reference_range
    )
    corridor_fraction = float(np.mean(np.abs(residual) <= corridor))
    amplitude_score = float(np.exp(-3.0 * nrmse))
    shape_score = max(min((correlation + 1.0) * 0.5, 1.0), 0.0)
    integral_score = max(0.0, 1.0 - integral_error)
    engineering_score = (
        0.35 * amplitude_score
        + 0.30 * shape_score
        + 0.20 * corridor_fraction
        + 0.15 * integral_score
    )
    status = (
        "pass"
        if engineering_score >= policy.pass_score
        and peak_error <= policy.peak_error_tolerance
        and integral_error <= policy.integral_error_tolerance
        else "warning"
        if engineering_score >= policy.warning_score
        else "fail"
    )
    return {
        "status": status,
        "overlap": {"axis": axis_name, "unit": axis_unit, "start": start, "stop": stop},
        "sample_count": int(grid.size),
        "normalized_rmse": nrmse,
        "pearson_correlation": correlation,
        "peak_relative_error": peak_error,
        "integral_relative_error": integral_error,
        "corridor_fraction": corridor_fraction,
        "engineering_score": engineering_score,
        "policy": asdict(policy),
    }


def assess_convergence(
    cases: Sequence[Mapping[str, object]],
    metrics: Sequence[str] = DEFAULT_VALIDATION_METRICS,
    pass_tolerance: float = 0.05,
    warning_tolerance: float = 0.10,
) -> dict[str, object]:
    """Compare the final two ordered cases; absent evidence remains unavailable."""
    _validate_tolerances(pass_tolerance, warning_tolerance)
    if not metrics or len(cases) < 2:
        return {
            "status": "not_evaluated",
            "reason": "Select metrics and provide at least two ordered cases.",
            "case_count": len(cases),
            "comparisons": {},
        }
    previous = cases[-2].get("metrics", {})
    reference = cases[-1].get("metrics", {})
    if not isinstance(previous, Mapping):
        previous = {}
    if not isinstance(reference, Mapping):
        reference = {}
    comparisons = {}
    for metric in metrics:
        if previous.get(metric) is None or reference.get(metric) is None:
            comparisons[metric] = {
                "status": "not_evaluated",
                "reason": "metric missing",
            }
            continue
        try:
            coarse_value = float(previous[metric])
            reference_value = float(reference[metric])
            if not math.isfinite(coarse_value) or not math.isfinite(reference_value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            comparisons[metric] = {
                "status": "fail",
                "reason": "metric must be finite and numeric",
            }
            continue
        error = _relative_error(coarse_value, reference_value)
        status = (
            "pass"
            if error <= pass_tolerance
            else "warning"
            if error <= warning_tolerance
            else "fail"
        )
        comparisons[metric] = {
            "status": status,
            "relative_change": error,
            "previous": coarse_value,
            "reference": reference_value,
        }
    return {
        "status": assessment_status(item["status"] for item in comparisons.values()),
        "case_count": len(cases),
        "reference_case": cases[-1].get("case_id", f"case_{len(cases) - 1}"),
        "comparisons": comparisons,
        "pass_tolerance": pass_tolerance,
        "warning_tolerance": warning_tolerance,
    }


def assess_repeatability(
    repeated_metrics: Sequence[Mapping[str, float]],
    metrics: Sequence[str] = DEFAULT_VALIDATION_METRICS,
    pass_cv: float = 0.01,
    warning_cv: float = 0.03,
) -> dict[str, object]:
    """Measure scatter and report how many runs supplied each metric.

    Missing runs leave otherwise passing scatter unverified. Invalid numeric
    values fail input validation. Neither condition is silently discarded.
    """
    _validate_tolerances(pass_cv, warning_cv)
    if not metrics or len(repeated_metrics) < 2:
        return {
            "status": "not_evaluated",
            "reason": "Select metrics and provide at least two independent runs.",
            "repeat_count": len(repeated_metrics),
            "metrics": {},
        }
    output = {}
    for metric in metrics:
        samples = []
        missing_count = 0
        invalid_count = 0
        for case in repeated_metrics:
            value = case.get(metric)
            if value is None:
                missing_count += 1
                continue
            try:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                invalid_count += 1
            else:
                samples.append(value)
        result = {
            "sample_count": len(samples),
            "missing_count": missing_count,
            "invalid_count": invalid_count,
        }
        if len(samples) < 2:
            result.update(
                status="fail" if invalid_count else "not_evaluated",
                reason="At least two finite metric values are required.",
            )
        else:
            values = np.asarray(samples, dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            cv = std / max(abs(mean), 1.0e-12)
            scatter_status = (
                "pass" if cv <= pass_cv else "warning" if cv <= warning_cv else "fail"
            )
            statuses = [scatter_status]
            if missing_count:
                statuses.append("not_evaluated")
            if invalid_count:
                statuses.append("fail")
            result.update(
                status=assessment_status(statuses),
                scatter_status=scatter_status,
                mean=mean,
                standard_deviation=std,
                coefficient_of_variation=cv,
            )
        output[metric] = result
    return {
        "status": assessment_status(item["status"] for item in output.values()),
        "repeat_count": len(repeated_metrics),
        "metrics": output,
        "pass_cv": pass_cv,
        "warning_cv": warning_cv,
    }


def correlate_crash_benchmark(
    simulation: Mapping[str, object],
    benchmark: Mapping[str, object],
    *,
    qualification: bool = False,
    required_channels: Sequence[str] | None = None,
    minimum_replicates: int = 3,
    qualification_policy: QualificationPolicy | None = None,
    comparison_policy: CurveComparisonPolicy | None = None,
) -> dict[str, object]:
    """Compare selected curves and optionally qualify their benchmark metadata.

    Screening never evaluates qualification metadata. Prescribed-wall comparisons
    omit the inertial acceleration pulse by default. Policies configure project
    thresholds and documentation requirements without changing curve data.
    """
    sim_processed = simulation.get("processed", {})
    measurement = simulation.get("measurement", {})
    if not isinstance(sim_processed, Mapping):
        sim_processed = {}
    if not isinstance(measurement, Mapping):
        measurement = {}
    prescribed_wall = measurement.get("scenario") == "prescribed_moving_wall"
    channels = (
        tuple(required_channels)
        if required_channels is not None
        else ("force_time", "force_displacement")
        if prescribed_wall
        else BENCHMARK_CHANNELS
    )
    if not channels or set(channels) - set(BENCHMARK_CHANNELS):
        raise ValueError("Select at least one supported benchmark channel.")
    traceability = benchmark.get("traceability", {})
    if not isinstance(traceability, Mapping):
        traceability = {}
    metadata_validation = (
        validate_benchmark_metadata(
            simulation,
            benchmark,
            required_channels=channels,
            minimum_replicates=minimum_replicates,
            policy=qualification_policy,
        )
        if qualification
        else {"status": "not_evaluated", "failures": []}
    )
    channel_inputs = {
        "force_time": (
            "time_ms", "rigid_wall_force_kN", "time_ms", "force_kN", "time", "ms"
        ),
        "crash_pulse": (
            "time_ms", "acceleration_g", "time_ms", "acceleration_g", "time", "ms"
        ),
        "force_displacement": (
            "crush_displacement_mm",
            "rigid_wall_force_kN",
            "displacement_mm",
            "force_displacement_force_kN",
            "displacement",
            "mm",
        ),
    }
    comparisons = {}
    for name in channels:
        (
            sim_axis, sim_values, ref_axis, ref_values, axis_name, axis_unit
        ) = channel_inputs[name]
        if benchmark.get(ref_axis) is None or benchmark.get(ref_values) is None:
            comparisons[name] = {
                "status": "fail",
                "reason": "Required physical benchmark channel is missing.",
            }
        else:
            comparisons[name] = compare_time_histories(
                sim_processed.get(sim_axis, []),
                sim_processed.get(sim_values, []),
                benchmark[ref_axis],
                benchmark[ref_values],
                policy=comparison_policy,
                axis_name=axis_name,
                axis_unit=axis_unit,
            )
    statuses = [item["status"] for item in comparisons.values()]
    if qualification:
        statuses.append(metadata_validation["status"])
    overall = assessment_status(statuses)
    return {
        "status": overall,
        "benchmark_id": benchmark.get("benchmark_id"),
        "assessment_mode": "qualification" if qualification else "screening",
        "qualification_status": overall if qualification else "not_evaluated",
        "required_channels": list(channels),
        "metadata_validation": metadata_validation,
        "comparisons": comparisons,
        "traceability": traceability,
    }


def summarize_solver_quality(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require every solver case to pass its numerical quality gate."""
    case_results = []
    statuses = []
    for index, case in enumerate(cases):
        quality = case.get("quality", {})
        status = str(
            quality.get(
                "numerical_status",
                case.get("quality_status", "fail"),
            )
        )
        failed_checks = [
            check_id
            for check_id in quality.get("failed_checks", [])
            if check_id != "material_validation"
        ]
        warning_checks = list(quality.get("warning_checks", []))
        case_results.append(
            {
                "case_id": case.get("case_id", f"case_{index}"),
                "status": status,
                "external_status": case.get("external_status"),
                "failed_checks": failed_checks,
                "warning_checks": warning_checks,
                "run_id": case.get("provenance", {}).get("run_id"),
            }
        )
        statuses.append(status)
    overall = assessment_status(statuses)
    return {
        "status": overall,
        "case_count": len(case_results),
        "cases": case_results,
    }


def write_validation_report(
    path: str | Path,
    *,
    solver_quality: Mapping[str, object],
    mesh_convergence: Mapping[str, object],
    timestep_convergence: Mapping[str, object],
    repeatability: Mapping[str, object],
    material_validation: Mapping[str, object],
    benchmark_correlation: Mapping[str, object],
) -> Path:
    """Persist a machine-readable qualification dossier."""
    sections = {
        "solver_quality": dict(solver_quality),
        "mesh_convergence": dict(mesh_convergence),
        "timestep_convergence": dict(timestep_convergence),
        "repeatability": dict(repeatability),
        "material_validation": dict(material_validation),
        "benchmark_correlation": dict(benchmark_correlation),
    }
    # A screening curve match cannot establish physical qualification.
    benchmark_status = benchmark_correlation.get(
        "qualification_status", "not_evaluated"
    )
    section_statuses = {
        name: section.get("status", "not_evaluated")
        for name, section in sections.items()
    }
    section_statuses["benchmark_correlation"] = benchmark_status
    overall = assessment_status(section_statuses.values())
    numerical_status = assessment_status(
        section_statuses[name]
        for name in (
            "solver_quality",
            "mesh_convergence",
            "timestep_convergence",
            "repeatability",
        )
    )
    physical_validation_status = assessment_status(
        section_statuses[name]
        for name in ("material_validation", "benchmark_correlation")
    )
    payload = {
        "schema": "pylcss.crash.validation",
        "schema_version": "2.0.0",
        "status": overall,
        "numerical_status": numerical_status,
        "physical_validation_status": physical_validation_status,
        "ml_ground_truth_eligible": overall == "pass",
        "sections": sections,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return target
