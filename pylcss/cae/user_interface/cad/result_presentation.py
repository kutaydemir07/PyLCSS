# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Short result labels and observed states shared by CAE result views."""

from __future__ import annotations

from typing import Any


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def analysis_label(data: dict[str, Any]) -> str:
    kind = str(data.get("type") or "").lower()
    if kind == "topopt_voxel":
        if data.get("study_kind") == "lattice_infill":
            return "Lattice Infill"
        if data.get("lattice_optimization"):
            return "Lattice Optimization"
        return "Topology Optimization"
    return {"fea": "Structural FEA", "external_solver": "Structural FEA",
            "modal": "Modal Analysis", "crash": "Crash / Impact"}.get(kind, "Results")


def run_status(data: dict[str, Any]) -> str:
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
    cad = _mapping(data.get("cad_reconstruction"))
    if data.get("cad_error"):
        rows.append(("CAD", "Conversion failed"))
    elif cad.get("valid") is False:
        rows.append(("CAD", "Invalid solid"))
    elif cad.get("valid") is True:
        rows.append(("CAD", "Complete"))
    manufacturing = _mapping(data.get("manufacturing"))
    surface = _mapping(manufacturing.get("surface_quality") or _mapping(data.get("recovered_shape")).get("surface_quality"))
    if (surface.get("is_watertight", surface.get("watertight")) is False
            or surface.get("self_intersecting_faces") or surface.get("nonmanifold_edges")):
        rows.append(("Geometry", "Invalid surface"))
    quality = _mapping(data.get("quality"))
    quality_status = str(quality.get("numerical_status") or data.get("quality_status") or quality.get("status") or "").lower()
    if quality_status in {"pass", "warning", "fail"}:
        rows.append(("Numerical checks", {"pass": "Passed", "warning": "Review", "fail": "Failed"}[quality_status]))
    validation = _mapping(data.get("validation_summary"))
    if validation.get("converged") is False or validation.get("error"):
        rows.append(("Validation", "Failed"))
    return rows


def crash_check_percent(data: dict[str, Any], check_id: str, fallback_key: str | None = None) -> float | None:
    checks = _mapping(data.get("quality")).get("checks") or []
    for check in checks:
        if isinstance(check, dict) and check.get("id") == check_id and check.get("value") is not None:
            return float(check["value"])
    value = data.get(fallback_key) if fallback_key else None
    return float(value) * 100.0 if value is not None else None
