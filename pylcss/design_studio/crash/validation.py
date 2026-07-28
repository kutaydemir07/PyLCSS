# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Numerical convergence, repeatability, and benchmark-correlation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_VALIDATION_METRICS = (
    "peak_crushing_force_kN",
    "mean_crushing_force_kN",
    "absorbed_energy_kJ",
    "useful_crush_stroke_mm",
    "peak_acceleration_g",
)


def _array(value: object) -> np.ndarray:
    try:
        return np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _relative_error(value: float, reference: float) -> float:
    scale = max(abs(value), abs(reference), 1.0e-12)
    return abs(value - reference) / scale


def compare_time_histories(
    simulation_time_ms: Iterable[float],
    simulation_values: Iterable[float],
    reference_time_ms: Iterable[float],
    reference_values: Iterable[float],
    corridor_relative: float = 0.10,
    corridor_absolute: float = 0.0,
    sample_count: int = 1000,
) -> dict[str, object]:
    """Objectively compare two unambiguous time histories on their overlap.

    The returned primitives (amplitude, phase/correlation, corridor, peak, and
    integral errors) are suitable inputs to an ISO/TS 18571 assessment tool.
    ``engineering_score`` is a transparent PyLCSS screening score and is not
    presented as the proprietary ISO rating itself.
    """
    sim_t = _array(simulation_time_ms)
    sim_y = _array(simulation_values)
    ref_t = _array(reference_time_ms)
    ref_y = _array(reference_values)
    n_sim = min(sim_t.size, sim_y.size)
    n_ref = min(ref_t.size, ref_y.size)
    if n_sim < 3 or n_ref < 3:
        return {
            "status": "fail",
            "reason": "Both histories require at least three samples.",
        }
    sim_t, sim_y = sim_t[:n_sim], sim_y[:n_sim]
    ref_t, ref_y = ref_t[:n_ref], ref_y[:n_ref]
    sim_order = np.argsort(sim_t)
    ref_order = np.argsort(ref_t)
    sim_t, sim_y = sim_t[sim_order], sim_y[sim_order]
    ref_t, ref_y = ref_t[ref_order], ref_y[ref_order]

    start = max(float(sim_t[0]), float(ref_t[0]))
    stop = min(float(sim_t[-1]), float(ref_t[-1]))
    if stop <= start:
        return {"status": "fail", "reason": "Histories do not overlap in time."}
    grid = np.linspace(start, stop, max(int(sample_count), 50))
    sim = np.interp(grid, sim_t, sim_y)
    ref = np.interp(grid, ref_t, ref_y)
    residual = sim - ref
    reference_range = max(float(np.ptp(ref)), float(np.max(np.abs(ref))), 1.0e-12)
    nrmse = float(np.sqrt(np.mean(residual ** 2)) / reference_range)
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
        if engineering_score >= 0.80
        and peak_error <= 0.15
        and integral_error <= 0.15
        else "warning"
        if engineering_score >= 0.65
        else "fail"
    )
    return {
        "status": status,
        "overlap_ms": [start, stop],
        "sample_count": int(grid.size),
        "normalized_rmse": nrmse,
        "pearson_correlation": correlation,
        "peak_relative_error": peak_error,
        "integral_relative_error": integral_error,
        "corridor_fraction": corridor_fraction,
        "engineering_score": engineering_score,
        "iso_ts_18571": {
            "status": "ready_for_external_rating",
            "note": (
                "PyLCSS exports aligned, unit-labelled histories and objective "
                "primitives but does not label its screening score as the "
                "ISO/TS 18571 proprietary rating."
            ),
        },
    }


def assess_convergence(
    cases: Sequence[Mapping[str, object]],
    metrics: Sequence[str] = DEFAULT_VALIDATION_METRICS,
    pass_tolerance: float = 0.05,
    warning_tolerance: float = 0.10,
) -> dict[str, object]:
    """Compare the two finest/most conservative cases in an ordered study."""
    if len(cases) < 2:
        return {
            "status": "fail",
            "reason": "A convergence study requires at least two ordered cases.",
            "comparisons": {},
        }
    previous = cases[-2].get("metrics", {})
    reference = cases[-1].get("metrics", {})
    comparisons = {}
    statuses = []
    for metric in metrics:
        try:
            coarse_value = float(previous[metric])
            reference_value = float(reference[metric])
        except (KeyError, TypeError, ValueError):
            comparisons[metric] = {
                "status": "fail",
                "reason": "metric missing",
            }
            statuses.append("fail")
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
        statuses.append(status)
    overall = "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"
    return {
        "status": overall,
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
    """Measure numerical scatter using coefficient of variation."""
    if len(repeated_metrics) < 2:
        return {
            "status": "fail",
            "reason": "Repeatability requires at least two independent runs.",
            "metrics": {},
        }
    output = {}
    statuses = []
    for metric in metrics:
        values = np.asarray(
            [case.get(metric, np.nan) for case in repeated_metrics],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size < 2:
            output[metric] = {"status": "fail", "reason": "metric missing"}
            statuses.append("fail")
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        cv = std / max(abs(mean), 1.0e-12)
        status = (
            "pass"
            if cv <= pass_cv
            else "warning"
            if cv <= warning_cv
            else "fail"
        )
        output[metric] = {
            "status": status,
            "mean": mean,
            "standard_deviation": std,
            "coefficient_of_variation": cv,
        }
        statuses.append(status)
    overall = "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"
    return {
        "status": overall,
        "repeat_count": len(repeated_metrics),
        "metrics": output,
        "pass_cv": pass_cv,
        "warning_cv": warning_cv,
    }


def correlate_crash_benchmark(
    simulation: Mapping[str, object],
    benchmark: Mapping[str, object],
) -> dict[str, object]:
    """Compare force-displacement and crash-pulse benchmark channels."""
    sim_processed = simulation.get("processed", {})
    measurement = simulation.get("measurement", {})
    processing = simulation.get("processing", {})
    traceability = benchmark.get("traceability", {})

    def present(value: object) -> bool:
        text = str(value or "").strip()
        return bool(text) and text.upper() not in {
            "REQUIRED",
            "REPLACE_WITH_TEST_ID",
        }

    metadata_failures = []
    required_metadata = (
        "benchmark_id",
        "specimen_id",
        "test_date",
        "test_laboratory",
        "geometry_revision",
        "material_lot_id",
        "thickness_mm",
        "test_replicate_count",
        "boundary_condition",
        "trigger_definition",
        "time_zero_definition",
        "source_document",
        "source_document_sha256",
        "csv_sha256",
        "metadata_sha256",
        "license_or_permission",
        "approved_by",
        "approval_date",
    )
    if str(traceability.get("status", "")).lower() != "pass":
        metadata_failures.append("traceability status is not pass")
    for field in required_metadata:
        if not present(traceability.get(field)):
            metadata_failures.append(f"missing or placeholder {field}")

    force_channel = traceability.get("force_channel", {})
    acceleration_channel = traceability.get("acceleration_channel", {})
    displacement_channel = traceability.get("displacement_channel", {})
    channel_requirements = {
        "force_channel": (
            force_channel,
            (
                "sensor_id",
                "calibration_id",
                "sample_rate_hz",
                "filter_cfc",
                "positive_convention",
            ),
        ),
        "acceleration_channel": (
            acceleration_channel,
            (
                "sensor_id",
                "location",
                "axis",
                "calibration_id",
                "sample_rate_hz",
                "filter_cfc",
                "positive_convention",
            ),
        ),
        "displacement_channel": (
            displacement_channel,
            (
                "sensor_id",
                "calibration_id",
                "sample_rate_hz",
                "positive_convention",
            ),
        ),
    }
    for channel_name, (channel, fields) in channel_requirements.items():
        if not isinstance(channel, Mapping):
            metadata_failures.append(f"missing {channel_name}")
            continue
        for field in fields:
            if not present(channel.get(field)):
                metadata_failures.append(
                    f"missing or placeholder {channel_name}.{field}"
                )

    expected_force_cfc = (
        processing.get("force_filter", {}).get("cfc")
        if isinstance(processing, Mapping)
        else None
    )
    expected_acceleration_cfc = (
        processing.get("acceleration_filter", {}).get("cfc")
        if isinstance(processing, Mapping)
        else None
    )
    if (
        expected_force_cfc is not None
        and force_channel.get("filter_cfc") != expected_force_cfc
    ):
        metadata_failures.append("force CFC does not match simulation")
    if (
        expected_acceleration_cfc is not None
        and acceleration_channel.get("filter_cfc")
        != expected_acceleration_cfc
    ):
        metadata_failures.append("acceleration CFC does not match simulation")
    expected_conventions = (
        (force_channel, "compression_positive", "force"),
        (acceleration_channel, "deceleration_positive", "acceleration"),
        (displacement_channel, "crush_positive", "displacement"),
    )
    for channel, expected, name in expected_conventions:
        if channel.get("positive_convention") != expected:
            metadata_failures.append(
                f"{name} positive convention must be {expected}"
            )

    for name, channel in (
        ("force_channel", force_channel),
        ("acceleration_channel", acceleration_channel),
    ):
        try:
            sample_rate = float(channel.get("sample_rate_hz"))
            cfc = float(channel.get("filter_cfc"))
        except (TypeError, ValueError):
            continue
        if sample_rate < 10.0 * cfc:
            metadata_failures.append(
                f"{name} sample rate is below the 10x-CFC policy"
            )

    for field, simulation_field in (
        ("impact_velocity_m_s", "initial_speed_m_s"),
        ("impactor_mass_kg", "impactor_mass_kg"),
    ):
        try:
            test_value = float(traceability[field])
            simulation_value = float(measurement[simulation_field])
        except (KeyError, TypeError, ValueError):
            metadata_failures.append(f"missing numeric {field}")
            continue
        if _relative_error(test_value, simulation_value) > 0.01:
            metadata_failures.append(
                f"{field} differs from simulation by more than 1%"
            )
    try:
        if float(traceability["thickness_mm"]) <= 0.0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        metadata_failures.append("thickness_mm must be positive")
    try:
        if int(traceability["test_replicate_count"]) < 3:
            metadata_failures.append(
                "test_replicate_count must be at least 3"
            )
    except (KeyError, TypeError, ValueError):
        metadata_failures.append("test_replicate_count must be an integer")

    metadata_validation = {
        "status": "fail" if metadata_failures else "pass",
        "failures": metadata_failures,
        "impact_condition_tolerance": 0.01,
        "sample_rate_policy": "at least 10 times the declared CFC",
    }
    comparisons = {}
    if benchmark.get("force_kN") is not None:
        comparisons["force_time"] = compare_time_histories(
            sim_processed.get("time_ms", []),
            sim_processed.get("rigid_wall_force_kN", []),
            benchmark.get("time_ms", []),
            benchmark.get("force_kN", []),
        )
    if benchmark.get("acceleration_g") is not None:
        comparisons["crash_pulse"] = compare_time_histories(
            sim_processed.get("time_ms", []),
            sim_processed.get("acceleration_g", []),
            benchmark.get("time_ms", []),
            benchmark.get("acceleration_g", []),
        )
    if (
        benchmark.get("force_displacement_force_kN") is not None
        and benchmark.get("displacement_mm") is not None
    ):
        comparisons["force_displacement"] = compare_time_histories(
            sim_processed.get("crush_displacement_mm", []),
            sim_processed.get("rigid_wall_force_kN", []),
            benchmark.get("displacement_mm", []),
            benchmark.get("force_displacement_force_kN", []),
        )
    for required in ("force_time", "crash_pulse", "force_displacement"):
        if required not in comparisons:
            comparisons[required] = {
                "status": "fail",
                "reason": "Required physical benchmark channel is missing.",
            }
    statuses = [
        metadata_validation["status"],
        *(item.get("status", "fail") for item in comparisons.values()),
    ]
    overall = (
        "fail"
        if not statuses or "fail" in statuses
        else "warning"
        if "warning" in statuses
        else "pass"
    )
    return {
        "status": overall,
        "benchmark_id": benchmark.get("benchmark_id"),
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
    overall = (
        "fail"
        if not statuses or "fail" in statuses
        else "warning"
        if "warning" in statuses
        else "pass"
    )
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
    statuses = [section.get("status", "fail") for section in sections.values()]
    overall = (
        "fail"
        if "fail" in statuses
        else "warning"
        if "warning" in statuses
        else "pass"
    )
    numerical_statuses = [
        sections[name].get("status", "fail")
        for name in (
            "solver_quality",
            "mesh_convergence",
            "timestep_convergence",
            "repeatability",
        )
    ]
    numerical_status = (
        "fail"
        if "fail" in numerical_statuses
        else "warning"
        if "warning" in numerical_statuses
        else "pass"
    )
    physical_statuses = [
        sections[name].get("status", "fail")
        for name in ("material_validation", "benchmark_correlation")
    ]
    physical_validation_status = (
        "fail"
        if "fail" in physical_statuses
        else "warning"
        if "warning" in physical_statuses
        else "pass"
    )
    payload = {
        "schema": "pylcss.crash.validation",
        "schema_version": "1.1.0",
        "status": overall,
        "numerical_status": numerical_status,
        "physical_validation_status": physical_validation_status,
        "ml_ground_truth_eligible": overall == "pass",
        "sections": sections,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
