# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Automated numerical and physical consistency gates for crash results."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


_SEVERITY = {"pass": 0, "warning": 1, "fail": 2, "not_evaluated": -1}


def _array(value: object) -> np.ndarray:
    try:
        return np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _check(
    check_id: str,
    status: str,
    value: object,
    limit: object,
    message: str,
    basis: str,
) -> Dict[str, object]:
    return {
        "id": check_id,
        "status": status,
        "value": value,
        "limit": limit,
        "message": message,
        "basis": basis,
    }


def _threshold_status(
    value: float,
    pass_limit: float,
    warning_limit: float,
) -> str:
    if value <= pass_limit:
        return "pass"
    if value <= warning_limit:
        return "warning"
    return "fail"


def _energy_error_percent(raw: Mapping[str, object]) -> tuple[float | None, str]:
    relative = _array(raw.get("delta_total_energy_relative"))
    relative = relative[np.isfinite(relative)]
    if relative.size:
        max_abs = float(np.max(np.abs(relative)))
        # OpenRadioss post-processors have emitted this channel as both a
        # fraction and a percentage across versions. Values within [-1, 1]
        # are treated as a fraction and normalised to percent.
        return (
            max_abs * 100.0 if max_abs <= 1.0 else max_abs,
            "OpenRadioss DTE_REL",
        )

    ke = _array(raw.get("kinetic_energy_kj"))
    ie = _array(raw.get("internal_energy_kj"))
    rke = _array(raw.get("rotational_kinetic_energy_kj"))
    external = _array(raw.get("external_work_kj"))
    sizes = [arr.size for arr in (ke, ie) if arr.size]
    if not sizes:
        return None, "unavailable"
    n = min(sizes)
    total = ke[:n] + ie[:n]
    if rke.size >= n:
        total = total + rke[:n]
    reference = float(total[0])
    if abs(reference) <= 1.0e-12:
        reference = max(float(np.max(np.abs(total))), 1.0e-12)
    expected = np.full(n, float(total[0]))
    if external.size >= n:
        expected = expected + external[:n] - external[0]
    error = 100.0 * (total - expected) / abs(reference)
    return float(np.max(np.abs(error))), "reconstructed KE+RKE+IE balance"


def evaluate_crash_quality(
    measurements: Mapping[str, object],
    external_status: str,
    end_time_ms: float,
    required_history_samples: int = 200,
) -> Dict[str, object]:
    """Evaluate a result against conservative explicit-crash quality gates.

    Thresholds are intentionally visible in the returned checks.  They are
    defaults for model screening, not universal homologation limits; a project
    validation plan may tighten them.
    """
    if external_status == "deck_written":
        return {
            "schema": "pylcss.crash.quality",
            "schema_version": "1.1.0",
            "status": "not_evaluated",
            "numerical_status": "not_evaluated",
            "physical_validation_status": "not_evaluated",
            "ml_eligible": False,
            "checks": [],
            "summary": "Deck generated; no solver result exists to qualify.",
        }

    raw = measurements.get("raw", {}) if isinstance(measurements, Mapping) else {}
    processed = (
        measurements.get("processed", {})
        if isinstance(measurements, Mapping)
        else {}
    )
    metrics = (
        measurements.get("metrics", {})
        if isinstance(measurements, Mapping)
        else {}
    )
    measurement_meta = measurements.get("measurement", {})
    scenario = str(measurement_meta.get("scenario") or "")
    prescribed_wall = scenario == "prescribed_moving_wall"
    checks = []

    terminated = external_status == "engine_completed"
    checks.append(
        _check(
            "normal_termination",
            "pass" if terminated else "fail",
            external_status,
            "engine_completed",
            "OpenRadioss Engine must exit successfully.",
            "OpenRadioss model debugging/result-checking guidance",
        )
    )

    diagnostics = measurement_meta.get("solver_diagnostics", {})
    starter_diagnostics = diagnostics.get("starter", {})
    engine_diagnostics = diagnostics.get("engine", {})
    diagnostic_errors = [
        value
        for value in (
            starter_diagnostics.get("error_count"),
            engine_diagnostics.get("error_count"),
        )
        if value is not None
    ]
    diagnostic_warnings = [
        value
        for value in (
            starter_diagnostics.get("warning_count"),
            engine_diagnostics.get("warning_count"),
        )
        if value is not None
    ]
    diagnostics_complete = (
        starter_diagnostics.get("available") is True
        and engine_diagnostics.get("available") is True
        and starter_diagnostics.get("normal_termination") is True
        and engine_diagnostics.get("normal_termination") is True
        and len(diagnostic_errors) == 2
        and len(diagnostic_warnings) == 2
    )
    if not diagnostics_complete or any(
        count > 0 for count in diagnostic_errors
    ):
        diagnostic_status = "fail"
    elif any(count > 0 for count in diagnostic_warnings):
        diagnostic_status = "warning"
    else:
        diagnostic_status = "pass"
    checks.append(
        _check(
            "solver_model_check",
            diagnostic_status,
            {
                "starter_errors": starter_diagnostics.get("error_count"),
                "starter_warnings": starter_diagnostics.get("warning_count"),
                "engine_errors": engine_diagnostics.get("error_count"),
                "engine_warnings": engine_diagnostics.get("warning_count"),
            },
            "normal termination with 0 errors and 0 warnings",
            (
                "Starter and Engine OUT summaries must be present; warnings "
                "remain visible and prevent an unconditional qualification pass."
            ),
            "OpenRadioss Starter model-check guidance",
        )
    )

    material_validation = measurement_meta.get("material_validation", {})
    material_status = str(material_validation.get("status") or "missing")
    material_check_status = (
        "pass"
        if material_status == "pass"
        else "warning"
        if material_status == "unverified"
        else "fail"
    )
    checks.append(
        _check(
            "material_validation",
            material_check_status,
            material_status,
            "pass with traceable lot/rate dossier",
            (
                material_validation.get("reason")
                or "The exact material lot and strain-rate range are not validated."
            ),
            "PyLCSS material traceability policy",
        )
    )

    time_ms = _array(raw.get("time_ms"))
    finite_time = time_ms[np.isfinite(time_ms)]
    completion = (
        float(finite_time[-1] / end_time_ms)
        if finite_time.size and end_time_ms > 0.0
        else 0.0
    )
    checks.append(
        _check(
            "analysis_duration",
            "pass" if completion >= 0.98 else "fail",
            completion,
            ">= 0.98",
            "The imported history must reach the requested termination time.",
            "PyLCSS trace completeness policy",
        )
    )

    sample_count = int(finite_time.size)
    sample_status = (
        "pass"
        if sample_count >= required_history_samples
        else "warning"
        if sample_count >= 50
        else "fail"
    )
    checks.append(
        _check(
            "history_resolution",
            sample_status,
            sample_count,
            f">= {required_history_samples} samples (50 minimum)",
            "Force and acceleration peaks require a resolved physical time axis.",
            "SAE J211/ISO 6487 measurement contract",
        )
    )

    force = _array(processed.get("rigid_wall_force_kN"))
    pulse = _array(processed.get("acceleration_g"))
    force_source = str(measurement_meta.get("force_source") or "")
    physical_wall_force = (
        (
            "rigid-wall" in force_source.lower()
            or "/th/rwall" in force_source.lower()
        )
        and "fallback" not in force_source.lower()
    )
    for check_id, values, label in (
        ("rigid_wall_force_channel", force, "rigid-wall force"),
        ("crash_pulse_channel", pulse, "reference acceleration"),
    ):
        finite = values[np.isfinite(values)]
        available = finite.size == sample_count and finite.size > 0
        nonzero = available and float(np.max(np.abs(finite))) > 1.0e-12
        acceleration_not_required = (
            check_id == "crash_pulse_channel" and prescribed_wall
        )
        qualified = acceleration_not_required or (
            nonzero
            and (
                physical_wall_force
                if check_id == "rigid_wall_force_channel"
                else True
            )
        )
        checks.append(
            _check(
                check_id,
                "pass" if qualified else "fail",
                (
                    {"samples": int(finite.size), "source": force_source}
                    if check_id == "rigid_wall_force_channel"
                    else (
                        {
                            "samples": int(finite.size),
                            "required": False,
                            "reason": "massless prescribed-speed wall",
                        }
                        if acceleration_not_required
                        else int(finite.size)
                    )
                ),
                (
                    f"{sample_count} finite, non-zero samples from a "
                    "native rigid-wall channel"
                    if check_id == "rigid_wall_force_channel"
                    else (
                        "not required for a massless prescribed-speed wall"
                        if acceleration_not_required
                        else f"{sample_count} finite, non-zero samples"
                    )
                ),
                (
                    "A prescribed-speed platen has no inertial crash pulse."
                    if acceleration_not_required
                    else f"A qualified crash result requires a physical {label} channel."
                ),
                "PyLCSS crash measurement contract",
            )
        )

    processing = measurements.get("processing", {})
    units = measurements.get("units", {})
    force_filter = processing.get("force_filter", {})
    acceleration_filter = processing.get("acceleration_filter", {})
    required_units = {
        "time_ms": "ms",
        "rigid_wall_force_kN": "kN",
        "crush_displacement_mm": "mm",
        "acceleration_g": "g",
    }
    unit_contract_valid = all(
        units.get(channel) == unit
        for channel, unit in required_units.items()
    )
    signal_processing_valid = (
        processing.get("raw_preserved") is True
        and force_filter.get("applied") is True
        and acceleration_filter.get("applied") is True
        and unit_contract_valid
    )
    checks.append(
        _check(
            "signal_processing_contract",
            "pass" if signal_processing_valid else "fail",
            {
                "raw_preserved": bool(processing.get("raw_preserved")),
                "force_filter_applied": bool(force_filter.get("applied")),
                "force_cfc": force_filter.get("cfc"),
                "acceleration_filter_applied": bool(
                    acceleration_filter.get("applied")
                ),
                "acceleration_cfc": acceleration_filter.get("cfc"),
                "unit_contract_valid": unit_contract_valid,
            },
            "raw preserved; both configured CFC profiles applied; SI-derived units labelled",
            (
                "Processed peaks must be derived from resolved, labelled "
                "signals while the original solver channels remain available."
            ),
            "SAE J211/ISO 6487-aligned PyLCSS measurement contract",
        )
    )

    impactor_mass_kg = float(
        measurement_meta.get("impactor_mass_kg") or 0.0
    )
    independent_required = (
        scenario in {
            "fixed_specimen_moving_impactor",
            "moving_body_fixed_wall",
        }
        and impactor_mass_kg > 0.0
    )
    independent_kinematics = (
        processing.get("independent_impactor_kinematics") is True
    )
    checks.append(
        _check(
            "independent_impactor_kinematics",
            (
                "pass"
                if independent_kinematics or not independent_required
                else "fail"
            ),
            independent_kinematics,
            (
                "required for a finite-mass moving impactor"
                if independent_required
                else "not required for this scenario"
            ),
            (
                "Work-energy and impulse-momentum closure must use independent "
                "solver motion (rigid-wall main-node motion or moving-body "
                "global momentum), not motion reconstructed from the same "
                "force history."
            ),
            "Independent measurement-channel consistency policy",
        )
    )

    if prescribed_wall:
        # A massless prescribed wall supplies external work instead of carrying
        # initial kinetic energy. OpenRadioss' global-energy histories do not
        # include that wall work as an external-work channel, so close the
        # balance against the independently integrated wall force-displacement
        # history. This is the physically meaningful balance for a controlled
        # platen test.
        kinetic = _array(raw.get("kinetic_energy_kj"))
        internal = _array(raw.get("internal_energy_kj"))
        rotational = _array(raw.get("rotational_kinetic_energy_kj"))
        available_sizes = [arr.size for arr in (kinetic, internal) if arr.size]
        wall_work = abs(
            float(metrics.get("force_displacement_energy_kJ") or 0.0)
        )
        if available_sizes and wall_work > 1.0e-12:
            count = min(available_sizes)
            total = kinetic[:count] + internal[:count]
            if rotational.size >= count:
                total = total + rotational[:count]
            system_energy_change = abs(float(total[-1] - total[0]))
            energy_error_pct = (
                100.0
                * abs(system_energy_change - wall_work)
                / max(system_energy_change, wall_work, 1.0e-12)
            )
            energy_source = "prescribed-wall work versus system-energy change"
        else:
            energy_error_pct = None
            energy_source = "prescribed-wall work unavailable"
    else:
        energy_error_pct, energy_source = _energy_error_percent(raw)
    if energy_error_pct is None:
        energy_status = "fail"
        energy_value = None
    else:
        energy_status = _threshold_status(energy_error_pct, 5.0, 15.0)
        energy_value = energy_error_pct
    checks.append(
        _check(
            "energy_balance",
            energy_status,
            energy_value,
            "<= 5% pass; <= 15% warning",
            f"Maximum absolute energy-balance error ({energy_source}).",
            "Explicit-solver computation checks",
        )
    )

    ke = _array(raw.get("kinetic_energy_kj"))
    ie = _array(raw.get("internal_energy_kj"))
    rke = _array(raw.get("rotational_kinetic_energy_kj"))
    ce = _array(raw.get("contact_energy_kj"))
    he = _array(raw.get("hourglass_energy_kj"))
    available_sizes = [arr.size for arr in (ke, ie) if arr.size]
    denominator = np.asarray([], dtype=float)
    if available_sizes:
        n = min(available_sizes)
        denominator = np.abs(ke[:n] + ie[:n])
        if rke.size >= n:
            denominator += np.abs(rke[:n])
        denominator = np.maximum(denominator, 1.0e-12)

    if denominator.size and he.size >= denominator.size:
        hourglass_ratio = 100.0 * float(
            np.max(np.abs(he[: denominator.size]) / denominator)
        )
        hg_status = _threshold_status(hourglass_ratio, 10.0, 15.0)
    else:
        hourglass_ratio = None
        hg_status = "fail"
    checks.append(
        _check(
            "hourglass_energy",
            hg_status,
            hourglass_ratio,
            "<= 10% pass; <= 15% warning",
            "Hourglass energy must remain a small part of system energy.",
            "Explicit-solver computation checks",
        )
    )

    if (
        denominator.size
        and he.size >= denominator.size
        and ce.size >= denominator.size
    ):
        artificial_ratio = 100.0 * float(
            np.max(
                (np.abs(he[: denominator.size]) + np.abs(ce[: denominator.size]))
                / denominator
            )
        )
        artificial_status = (
            "pass" if artificial_ratio <= 15.0 else "fail"
        )
    else:
        artificial_ratio = None
        artificial_status = "fail"
    checks.append(
        _check(
            "hourglass_plus_contact_energy",
            artificial_status,
            artificial_ratio,
            "<= 15%",
            "Combined hourglass and contact energy is checked against total energy.",
            "Explicit-solver computation checks",
        )
    )

    mass_kg = _array(raw.get("mass_kg"))
    mass_kg = mass_kg[np.isfinite(mass_kg)]
    if mass_kg.size and abs(mass_kg[0]) > 1.0e-12:
        mass_error_pct = 100.0 * float(
            np.max(np.maximum(mass_kg - mass_kg[0], 0.0)) / abs(mass_kg[0])
        )
        mass_status = _threshold_status(mass_error_pct, 1.0, 3.0)
    else:
        mass_error_pct = None
        mass_status = "fail"
    checks.append(
        _check(
            "added_mass",
            mass_status,
            mass_error_pct,
            "<= 1% pass; <= 3% warning",
            "Mass added by explicit time-step control must remain limited.",
            "Explicit-solver nodal time-step and results-checking practice",
        )
    )

    timestep = _array(raw.get("timestep_ms"))
    positive_dt = timestep[np.isfinite(timestep) & (timestep > 0.0)]
    if positive_dt.size:
        reference_dt = max(float(positive_dt[0]), 1.0e-15)
        dt_ratio = float(np.min(positive_dt) / reference_dt)
        dt_status = "pass" if dt_ratio >= 0.1 else "warning"
    else:
        dt_ratio = None
        dt_status = "fail"
    checks.append(
        _check(
            "time_step_evolution",
            dt_status,
            dt_ratio,
            "minimum / initial >= 0.10",
            "A severe time-step collapse can indicate unstable element distortion.",
            "Explicit-solver computation checks",
        )
    )

    fd_energy = float(metrics.get("force_displacement_energy_kJ") or 0.0)
    absorbed = float(metrics.get("absorbed_energy_kJ") or 0.0)
    transferred = float(
        metrics.get("impactor_kinetic_energy_loss_kJ") or absorbed
    )
    if max(abs(fd_energy), abs(transferred)) > 1.0e-9:
        energy_consistency_pct = (
            100.0
            * abs(fd_energy - transferred)
            / max(abs(fd_energy), abs(transferred))
        )
        consistency_status = _threshold_status(
            energy_consistency_pct, 10.0, 20.0
        )
    else:
        energy_consistency_pct = None
        consistency_status = "fail"
    checks.append(
        _check(
            "work_energy_consistency",
            consistency_status,
            energy_consistency_pct,
            "<= 10% pass; <= 20% warning",
            (
                "Area under force-displacement is compared with the impactor "
                "kinetic-energy loss (or internal energy when no finite mass exists)."
            ),
            "Work-energy theorem and PyLCSS consistency policy",
        )
    )

    impulse = float(metrics.get("force_impulse_N_s") or 0.0)
    delta_v = float(metrics.get("delta_v_m_s") or 0.0)
    impactor_mass = float(
        measurement_meta.get("impactor_mass_kg") or 0.0
    )
    if prescribed_wall:
        momentum_error_pct = None
        momentum_status = "pass"
        momentum_limit = "not applicable to a massless prescribed-speed wall"
    elif impactor_mass > 0.0 and max(abs(impulse), impactor_mass * abs(delta_v)) > 1.0e-9:
        momentum_delta = impactor_mass * abs(delta_v)
        momentum_error_pct = (
            100.0
            * abs(impulse - momentum_delta)
            / max(abs(impulse), abs(momentum_delta))
        )
        momentum_status = _threshold_status(momentum_error_pct, 5.0, 15.0)
        momentum_limit = "<= 5% pass; <= 15% warning"
    else:
        momentum_error_pct = None
        momentum_status = "warning"
        momentum_limit = "<= 5% pass; <= 15% warning"
    checks.append(
        _check(
            "impulse_momentum_consistency",
            momentum_status,
            momentum_error_pct,
            momentum_limit,
            (
                "Momentum closure is not applicable to a massless "
                "prescribed-speed wall."
                if prescribed_wall
                else "Rigid-wall impulse is compared with impactor momentum change."
            ),
            "Impulse-momentum theorem",
        )
    )

    overall = max(
        (str(check["status"]) for check in checks),
        key=lambda value: _SEVERITY.get(value, 2),
    )
    numerical_checks = [
        check for check in checks if check["id"] != "material_validation"
    ]
    numerical_status = max(
        (str(check["status"]) for check in numerical_checks),
        key=lambda value: _SEVERITY.get(value, 2),
    )
    physical_validation_status = material_status
    failed = [check["id"] for check in checks if check["status"] == "fail"]
    warned = [check["id"] for check in checks if check["status"] == "warning"]
    return {
        "schema": "pylcss.crash.quality",
        "schema_version": "1.1.0",
        "status": overall,
        "numerical_status": numerical_status,
        "physical_validation_status": physical_validation_status,
        "ml_eligible": overall == "pass",
        "checks": checks,
        "failed_checks": failed,
        "warning_checks": warned,
        "energy_balance_max_error": (
            energy_error_pct / 100.0 if energy_error_pct is not None else None
        ),
        "hourglass_energy_ratio": (
            hourglass_ratio / 100.0 if hourglass_ratio is not None else None
        ),
        "contact_hourglass_energy_ratio": (
            artificial_ratio / 100.0 if artificial_ratio is not None else None
        ),
        "added_mass_ratio": (
            mass_error_pct / 100.0 if mass_error_pct is not None else None
        ),
        "summary": (
            f"{overall.upper()}: {len(failed)} failed, {len(warned)} warning "
            f"checks out of {len(checks)}."
        ),
    }
