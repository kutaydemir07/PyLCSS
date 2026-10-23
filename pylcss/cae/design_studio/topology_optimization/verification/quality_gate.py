# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Reevaluate independent FEA checks without rerunning topology optimization."""

from __future__ import annotations

import math

import numpy as np


VERIFICATION_DEFAULTS = {
    "validation_yield_safety_factor": 1.0,
    "validation_displacement_limit_mm": 0.0,
}
_VALIDATION_CHECKS = {
    "independent_validation_not_run",
    "explicit_structure_external_reanalysis_required",
    "validation_missing",
    "validation_failed",
    "validation_result_invalid",
    "validation_settings_invalid",
    "mesh_convergence_not_demonstrated",
    "validated_stress_exceeds_yield",
    "validated_displacement_exceeds_limit",
    "lattice_analysis_geometry_mismatch",
    "lattice_equilibrium_not_balanced",
    "lattice_yield_strength_missing",
    "cad_analysis_geometry_mismatch",
    "equilibrium_not_balanced",
    "yield_strength_missing",
}
_LEGACY_VALIDATION_MESSAGES = (
    "Independent analysis of the manufactured geometry was not run.",
    "The built-in voxel check does not resolve local lattice/rib",
    "This is generated geometry, not an analyzed result:",
    "Independent CalculiX validation exceeds",
    "The independent CalculiX mesh-convergence study did not converge.",
    "Validation was requested but did not produce",
    "Validation skipped/failed:",
)


def _peak(analysis, name):
    if name == "stress":
        scalar = analysis.get("peak_stress_nodal", analysis.get("max_stress_gauss"))
    else:
        scalar = analysis.get("peak_displacement")
    if scalar is not None:
        value = float(scalar)
        return value if math.isfinite(value) and value >= 0 else None
    values = np.asarray(analysis.get(name), dtype=float)
    if not values.size or not np.all(np.isfinite(values)):
        return None
    if name == "displacement":
        values = np.linalg.norm(values.reshape(-1, 3), axis=1)
    peak = float(np.max(values))
    return peak if peak >= 0 else None


def update_verification_status(
    output, *, material=None, safety_factor=None, displacement_limit=None
):
    """Combine existing geometry/optimization checks with current independent FEA.

    Completing a solver is insufficient: usable response values and configured
    allowable/displacement checks must pass. Old validation messages are replaced
    on reevaluation, while unrelated warnings and base checks are preserved.
    """
    gate = dict(output.get("quality_gate") or {})
    failed = list(
        gate.get(
            "base_failed_checks",
            [c for c in gate.get("failed_checks", []) if c not in _VALIDATION_CHECKS],
        )
    )
    base_failed = bool(failed)
    advisory = list(
        gate.get(
            "base_advisory_checks",
            [
                c
                for c in gate.get("advisory_checks", [])
                if c != "infill_is_geometry_only"
            ],
        )
    )
    previous_messages = set(gate.get("verification_warnings") or [])
    warnings = [
        str(message)
        for message in output.get("warnings") or []
        if message not in previous_messages
    ]
    if "verification_warnings" not in gate and (
        _VALIDATION_CHECKS.intersection(gate.get("failed_checks") or [])
        or "infill_is_geometry_only" in (gate.get("advisory_checks") or [])
    ):
        warnings = [
            message
            for message in warnings
            if not message.startswith(_LEGACY_VALIDATION_MESSAGES)
        ]
    messages = []
    criteria = dict(output.get("verification_criteria") or {})
    if material is not None:
        criteria["yield_strength"] = (
            material.get("yield_strength", material.get("yield", 0.0)) or 0.0
        )
    if safety_factor is not None:
        criteria["safety_factor"] = safety_factor
    if displacement_limit is not None:
        criteria["displacement_limit_mm"] = displacement_limit
    output["verification_criteria"] = criteria

    def block(check, message):
        failed.append(check)
        messages.append(message)

    analysis = output.get("validation")
    current = output.get("fea_analysis_status") in (None, "completed")
    completed = (
        isinstance(analysis, dict)
        and analysis.get("source") in {'topopt_cad_fea', 'topopt_lattice_fea'}
        and current
    )
    lattice = bool(output.get('generated_lattice') or output.get('study_kind') == 'post_topopt_lattice')
    if completed and lattice:
        from ..geometry.lattice_tet_mesh import lattice_surface_fingerprint
        surface = output.get('recovered_shape')
        try:
            matching = (analysis.get('source') == 'topopt_lattice_fea' and isinstance(surface, dict)
                        and analysis.get('analysis_geometry_sha256') == lattice_surface_fingerprint(surface))
        except (KeyError, TypeError, ValueError):
            matching = False
        if not matching:
            block('lattice_analysis_geometry_mismatch', 'The FEA result does not describe the current lattice surface.')
            completed = False
    if completed and not lattice:
        current_cad=output.get('cad_shape')
        if current_cad is None:current_cad=output.get('shape')
        if current_cad is not None:
            from ..geometry.cad_fingerprint import cad_shape_fingerprint
            try:
                matching=analysis.get('analysis_geometry_sha256')==cad_shape_fingerprint(current_cad)
            except (TypeError,ValueError,RuntimeError):matching=False
            if not matching:
                block('cad_analysis_geometry_mismatch','The FEA result does not describe the current CAD solid.')
                completed=False
    validation_attempted = completed
    infill = output.get("study_kind") == "lattice_infill"
    if completed:
        try:
            stress = _peak(analysis, "stress")
            displacement = _peak(analysis, "displacement")
            factor = float(criteria.get("safety_factor", 1.0))
            limit = float(criteria.get("displacement_limit_mm", 0.0))
            allowable = float(criteria.get("yield_strength", 0.0))
            if (
                not all(math.isfinite(value) for value in (factor, limit, allowable))
                or factor < 1
                or limit < 0
                or allowable < 0
            ):
                raise ValueError(
                    "Use a finite safety factor >= 1 and non-negative limits."
                )
        except (TypeError, ValueError, OverflowError) as exc:
            stress = displacement = None
            block("validation_settings_invalid", str(exc))
        if stress is None or displacement is None:
            block(
                "validation_result_invalid",
                "Independent FEA did not provide finite peak stress and displacement.",
            )
            completed = False
        else:
            convergence = analysis.get("convergence_study") or {}
            output["validation_summary"] = {
                "max_stress": stress,
                "max_displacement": displacement,
                "compliance": analysis.get("compliance"),
                "converged": convergence.get("converged"),
            }
            if convergence.get('converged') is not True:
                block(
                    "mesh_convergence_not_demonstrated",
                    "The independent FEA mesh-convergence check did not converge.",
                )
            if analysis.get('equilibrium_balanced') is not True:
                block('lattice_equilibrium_not_balanced' if lattice else 'equilibrium_not_balanced',
                      'The independent FEA did not demonstrate force equilibrium.')
            if allowable <= 0:
                block('lattice_yield_strength_missing' if lattice else 'yield_strength_missing',
                      'Yield verification requires a positive material yield strength.')
            if allowable > 0 and stress > allowable / factor:
                block(
                    "validated_stress_exceeds_yield",
                    f"Independent FEA stress {stress:.4g} MPa exceeds the allowable {allowable / factor:.4g} MPa after safety factor {factor:g}.",
                )
            if limit > 0 and displacement > limit:
                block(
                    "validated_displacement_exceeds_limit",
                    f"Independent FEA displacement {displacement:.4g} mm exceeds the {limit:.4g} mm limit.",
                )
    elif infill:
        advisory.append("infill_is_geometry_only")
        messages.append(
            "This infill is generated geometry. Independently analyse it before load-bearing use."
        )
    else:
        block(
            "independent_validation_not_run",
            "Independent analysis of the current geometry is not complete. The topology result remains suitable for concept screening.",
        )

    passed = completed and not failed
    if passed:
        status = "verification passed"
    elif infill:
        status = (
            "geometry checks failed"
            if base_failed
            else "verification failed"
            if validation_attempted
            else "geometry built"
        )
    else:
        status = "concept only"
    gate.update(
        status=status,
        release_ready=passed,
        verification_passed=passed,
        failed_checks=list(dict.fromkeys(failed)),
        advisory_checks=list(dict.fromkeys(advisory)),
        independent_validation_completed=completed,
        verification_scope=('Resolved linear-elastic lattice yield and displacement; buckling and fatigue not assessed'
                            if lattice else 'Independent CAD linear-elastic checks'),
        verification_warnings=messages,
    )
    output["quality_gate"] = gate
    output["release_ready"] = output["verification_passed"] = passed
    output["warnings"] = list(dict.fromkeys(warnings + messages))
    if completed and lattice:
        output['warnings'] = [warning for warning in output['warnings'] if not str(warning).startswith(
            'Mechanical results refer to the original solid TopOpt solve.')]
    return output
