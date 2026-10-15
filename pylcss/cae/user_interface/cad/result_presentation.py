# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Short result labels and observed states shared by CAE result views."""

from __future__ import annotations

from typing import Any

import numpy as np


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def analysis_label(data: dict[str, Any]) -> str:
    kind = str(data.get("type") or "").lower()
    if kind == "topopt_voxel":
        if data.get("study_kind") == "lattice_infill":
            return "Lattice Infill"
        if data.get("generated_lattice"):
            return "Generated Lattice"
        return "Topology Optimization"
    return {"fea": "Structural FEA", "external_solver": "Structural FEA",
            "modal": "Modal Analysis", "crash": "Crash / Impact"}.get(kind, "Results")


def run_status(data: dict[str, Any]) -> str:
    if data.get("generated_lattice") or data.get("study_kind") == "lattice_infill":
        return "Generated"
    status = str(data.get("external_status") or data.get("engine_status") or "").lower()
    if data.get("error") or any(word in status for word in ("fail", "error", "abort")):
        return "Failed"
    if "timeout" in status or "timed_out" in status:
        return "Timed out"
    if "cancel" in status:
        return "Cancelled"
    if status in {"deck_written", "deck_staged", "prepared"}:
        return "Input prepared"
    if status in {"queued", "running", "not_run", "not_started"}:
        return {"queued": "Queued", "running": "Running", "not_run": "Not run", "not_started": "Not started"}[status]
    if "missing" in status or "unavailable" in status:
        return "Results unavailable"
    if data.get("converged") is not None:
        return "Converged" if data["converged"] else "Not converged"
    return "Completed"


def result_state_rows(data: dict[str, Any]) -> list[tuple[str, str]]:
    rows = [("Status", run_status(data))]
    if data.get("type") == "topopt_voxel":
        status = str(data.get("fea_analysis_status") or "").lower()
        analysis = fea_result(data)
        if data.get("fea_analysis_error") or data.get("validation_error"):
            rows.append(("FEA analysis", "Failed"))
        elif status in {"running", "cancelled", "failed"}:
            rows.append(("FEA analysis", status.title()))
        elif analysis:
            rows.append(("FEA analysis", run_status(analysis)))
        else:
            rows.append(("FEA analysis", "Not run"))
    cad = _mapping(data.get("cad_reconstruction"))
    if data.get("cad_error") or cad.get("failed"):
        rows.append(("CAD", "Conversion failed"))
    elif cad.get("paper_pipeline") and cad.get("is_solid") is False:
        rows.append(("CAD", "Surface model — solid construction incomplete"))
    elif cad.get("valid") is False:
        rows.append(("CAD", "Built — review topology" if cad.get("paper_pipeline") else "Invalid solid"))
    elif cad.get("valid") is True:
        rows.append(("CAD", "Complete"))
    cad_checks = str(cad.get("validation_status") or "").lower()
    if cad_checks in {"not_run", "reported", "unverified"}:
        rows.append(("CAD checks", {
            "not_run": "Not run", "reported": "Reported", "unverified": "Not run",
        }[cad_checks]))
    if cad.get("warnings"):
        warnings = cad["warnings"]
        count = len(warnings) if isinstance(warnings, (list, tuple)) else 1
        rows.append(("CAD notes", f"{count} to review"))
    manufacturing = _mapping(data.get("manufacturing"))
    surface = _mapping(manufacturing.get("surface_quality") or _mapping(data.get("recovered_shape")).get("surface_quality"))
    if (surface.get("is_watertight", surface.get("watertight")) is False
            or surface.get("self_intersecting_faces") or surface.get("nonmanifold_edges")):
        rows.append(("Geometry", "Invalid surface"))
    quality = _mapping(data.get("quality"))
    quality_status = str(quality.get("numerical_status") or data.get("quality_status") or quality.get("status") or "").lower()
    if quality_status in {"pass", "warning", "fail", "unverified", "not_evaluated"}:
        rows.append(("Numerical checks", {"pass": "Passed", "warning": "Review", "fail": "Failed", "unverified": "Incomplete", "not_evaluated": "Not evaluated"}[quality_status]))
    validation = _mapping(data.get("validation_summary"))
    if validation.get("error"):
        rows.append(("Validation", "Failed"))
    elif validation.get("converged") is False:
        rows.append(("FEA mesh convergence", "Not converged"))
    return rows


def fea_result(data: dict[str, Any]) -> dict[str, Any]:
    """Keep the optimization solve and the independent FEA payload distinct."""
    if data.get("type") == "topopt_voxel":
        return _mapping(data.get("validation"))
    if data.get("type") in {"fea", "external_solver"}:
        return data
    return {}


def is_lattice_result(data: dict[str, Any]) -> bool:
    options = data.get("structure_options")
    mode = options.get("mode", "solid") if isinstance(options, dict) else getattr(options, "mode", "solid")
    return data.get("type") == "topopt_voxel" and bool(
        data.get("generated_lattice") or data.get("study_kind") == "lattice_infill"
        or mode != "solid"
    )


def metric_value(value: Any, unit: str = "", scale: float = 1.0) -> str:
    if value is None:
        return "—"
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if unit == "%":
        return f"{number:.1f}%"
    return f"{number:.4g} {unit}".strip()


def _peak(data: dict, key: str, *scalar_keys: str, vector: bool = False):
    for name in scalar_keys:
        if data.get(name) is not None:
            return data[name]
    values = data.get(key)
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if vector and array.size:
        array = np.linalg.norm(array.reshape(-1, 3), axis=1)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else None


def result_metric_sections(data: dict[str, Any]) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Outcome metrics with units and provenance for overview, tables and copy."""
    sections = []

    def add(rows, label, value, unit="", meaning="", scale=1.0):
        if value is not None:
            rows.append((label, metric_value(value, unit, scale), meaning))

    topology = data.get("type") == "topopt_voxel"
    lattice = is_lattice_result(data)
    optimization = topology and data.get("study_kind") != "lattice_infill" and not data.get("generated_lattice")
    if topology:
        material = []
        manufacturing = _mapping(data.get("manufacturing"))
        fraction = data.get("final_vol_frac")
        add(material, "Material retained", fraction, "%", "Fraction of the original design domain occupied by material.", 100)
        add(material, "Material removed", 1 - float(fraction) if fraction is not None else None, "%", "Compared with a fully solid design domain.", 100)
        add(material, "Target material", data.get("target_vol_frac") if optimization else None, "%", "Requested material budget for the optimization.", 100)
        volume = manufacturing.get("volume") if lattice else None
        mass = manufacturing.get("mass") if lattice else None
        add(material, "Lattice volume" if lattice else "Simulated material volume", volume if volume is not None else data.get("volume"), "mm³", "Built lattice geometry." if lattice else "Density-equivalent volume used by the optimization.")
        add(material, "Lattice mass" if lattice else "Simulated material mass", mass if mass is not None else data.get("mass"), "kg", "Built lattice geometry." if lattice else "Density-equivalent mass used by the optimization.", 1000)
        if material:
            sections.append(("Material use", material))

    if optimization:
        rows = []
        cases = [c for c in data.get("structural_cases") or [] if isinstance(c, dict)]
        displacements = [c["maximum_displacement"] for c in cases if c.get("maximum_displacement") is not None]
        add(rows, "Peak displacement", max(displacements) if displacements else data.get("peak_displacement"), "mm", "Response of the optimized density model; maximum across structural cases.")
        add(rows, "Structural compliance", data.get("compliance"), "N mm", "Work done by the applied loads; lower means a stiffer design for the same loads.")
        add(rows, "Aggregated stress", data.get("stress_pnorm"), "MPa", "Optimization stress constraint measure; this is not the peak CAD FEA stress.")
        add(rows, "Allowable aggregated stress", _mapping(data.get("stress_aggregation")).get("allowable_stress"), "MPa", "Limit applied to the optimizer's stress measure.")
        add(rows, "Objective improvement", data.get("objective_reduction_pct"), "%", "Reduction within the final continuation stage; negative means an increase.")
        add(rows, "Iterations", data.get("iterations"), meaning="Number of optimization iterations completed.")
        if data.get("converged") is not None:
            rows.append(("Optimization convergence", "Converged" if data["converged"] else "Not converged", "Whether the optimizer met its stopping criteria."))
        if rows:
            sections.append(("Optimization performance", rows))

    if lattice:
        manufacturing = _mapping(data.get("manufacturing"))
        recovered = _mapping(data.get("recovered_shape"))
        infill = _mapping(data.get("lattice_infill"))
        generated = _mapping(data.get("generated_lattice"))
        sizing = _mapping(generated.get("lattice_sizing") or manufacturing.get("lattice_sizing") or recovered.get("lattice_sizing"))
        connectivity = _mapping(manufacturing.get("lattice_connectivity") or recovered.get("lattice_connectivity"))
        rows = []
        add(rows, "Cell pattern", infill.get("pattern") or generated.get("cell_family") or manufacturing.get("structure"))
        add(rows, "Cell size", infill.get("cell_size") or generated.get("cell_size_mm") or manufacturing.get("cell_size_physical"), "mm", "Cell pitch used in the delivered lattice.")
        add(rows, "Member thickness", infill.get("member_thickness_model_units") or sizing.get("member_thickness_model_units") or manufacturing.get("member_thickness_physical"), "mm", "Thickness of the built struts or walls.")
        add(rows, "Skin thickness", infill.get("skin_thickness", manufacturing.get("skin_thickness_physical")), "mm")
        add(rows, "Target relative density", sizing.get("target_relative_density", infill.get("requested_relative_density")), "%", "Requested material fraction inside the lattice envelope.", 100)
        add(rows, "Built relative density", sizing.get("achieved_relative_density", infill.get("achieved_relative_density")), "%", "Material fraction inside the lattice envelope; distinct from the whole design domain.", 100)
        add(rows, "Envelope filled", connectivity.get("envelope_reach", infill.get("envelope_reach")), "%", "How much of the intended envelope the lattice reaches.", 100)
        surface = _mapping(manufacturing.get("surface_quality") or recovered.get("surface_quality"))
        add(rows, "Connected bodies", surface.get("connected_components", manufacturing.get("meaningful_component_count")), meaning="Separate connected bodies in the delivered geometry.")
        if rows:
            sections.append(("Lattice geometry", rows))

    analysis = fea_result(data)
    if analysis:
        rows = []
        add(rows, "Peak von Mises stress", _peak(analysis, "stress", "peak_stress_nodal", "max_stress_gauss"), "MPa", "Peak nodal stress from the FEA solve.")
        add(rows, "Peak displacement", _peak(analysis, "displacement", "peak_displacement", vector=True), "mm", "Maximum displacement magnitude from the FEA solve.")
        add(rows, "Minimum yield safety factor", analysis.get("minimum_yield_safety_factor", analysis.get("min_safety_factor")), meaning="Yield strength divided by stress; values below 1 exceed yield.")
        add(rows, "Structural compliance", analysis.get("compliance"), "N mm", "Work done by the applied loads in the FEA model.")
        add(rows, "Strain energy", analysis.get("strain_energy"), "N mm")
        if not topology:
            add(rows, "Volume", analysis.get("volume"), "mm³")
            add(rows, "Mass", analysis.get("mass"), "kg", scale=1000)
        else:
            target = analysis.get("analysis_target") or ("CAD solid" if analysis.get("source") == "topopt_cad_fea" else "Voxel-derived geometry")
            rows.insert(0, ("Analyzed geometry", str(target), "Geometry re-meshed for this independent analysis."))
        study = _mapping(analysis.get("convergence_study"))
        if study:
            convergence = study.get("converged")
            rows.append(("Mesh convergence", "Not assessed" if convergence is None else "Converged" if convergence else "Not converged", "Convergence of FEA compliance across mesh refinements."))
            add(rows, "Compliance change on refinement", study.get("compliance_rel_change"), "%", "Difference between the two finest FEA meshes.", 100)
        if rows:
            sections.append(("FEA analysis" if topology else "Structural results", rows))

    if topology and not lattice:
        rows = []
        add(rows, "Recovered geometry volume", data.get("recovered_design_volume"), "mm³", "Volume of the recovered design surface, excluding assembly hardware.")
        add(rows, "Recovered geometry mass", data.get("recovered_design_mass"), "kg", scale=1000)
        add(rows, "Volume change after recovery", data.get("recovery_volume_delta_pct"), "%", "Difference from the simulated density-equivalent volume.")
        if rows:
            sections.append(("Recovered geometry", rows))
    return sections


def result_notices(data: dict[str, Any]) -> list[str]:
    notices = []
    if data.get("type") == "topopt_voxel":
        if is_lattice_result(data) and not fea_result(data):
            notices.append("Lattice geometry is available. Stress, displacement and stiffness require a separate FEA of the generated lattice.")
        elif not fea_result(data):
            notices.append("Optimization responses describe the density model. Use Analyse FEA in Setup to check the recovered CAD solid.")
        error = data.get("fea_analysis_error") or data.get("validation_error")
        if error:
            notices.append(f"FEA analysis: {error}")
    for warning in data.get("warnings") or []:
        if isinstance(warning, str) and warning and warning not in notices:
            notices.append(warning)
    return notices


def crash_check_percent(data: dict[str, Any], check_id: str, fallback_key: str | None = None) -> float | None:
    checks = _mapping(data.get("quality")).get("checks") or []
    for check in checks:
        if isinstance(check, dict) and check.get("id") == check_id and check.get("value") is not None:
            return float(check["value"])
    value = data.get(fallback_key) if fallback_key else None
    return float(value) * 100.0 if value is not None else None
