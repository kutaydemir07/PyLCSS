# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Run a reproducible PyLCSS crash numerical-qualification matrix.

Examples
--------
Baseline smoke run:
    python experiments/crash_validation/run_qualification.py --mode baseline

Reprocess a saved baseline with the current measurement/QC contract:
    python experiments/crash_validation/run_qualification.py --mode baseline-report

Mesh, time-step, and repeatability study:
    python experiments/crash_validation/run_qualification.py --mode numerical

Replace time-step/repeatability cases on an already converged mesh:
    python experiments/crash_validation/run_qualification.py --mode temporal \
        --reference-mesh-size 3.5 --time-step-scales 0.9 0.67 0.5

Add physical benchmark correlation:
    python experiments/crash_validation/run_qualification.py --mode report \
        --benchmark-csv path/to/instrumented_component_test.csv \
        --material-validation path/to/material_coupon_validation.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYLCSS_KEEP_SOLVER_RUNS", "100")

from pylcss.design_studio.crash.validation import (  # noqa: E402
    assess_convergence,
    assess_repeatability,
    correlate_crash_benchmark,
    summarize_solver_quality,
    write_validation_report,
)
from pylcss.design_studio.crash.quality import (  # noqa: E402
    evaluate_crash_quality,
)
from pylcss.design_studio.crash.materials import (  # noqa: E402
    validate_material_dossier,
)
from pylcss.design_studio.crash.signals import (  # noqa: E402
    build_crash_measurements,
)
from pylcss.solver_backends.radioss_reader import (  # noqa: E402
    read_animation_frames,
    resolve_anim_to_vtk,
)
from pylcss.solver_backends.radioss_time_history import (  # noqa: E402
    read_openradioss_time_history,
)
from pylcss.solver_backends.openradioss import (  # noqa: E402
    _read_radioss_diagnostics,
)
from pylcss.design_studio.runtime import (  # noqa: E402
    clear_cache,
    crash,
    discover_override_controls,
)


DEFAULT_CAD = (
    REPO_ROOT
    / "data"
    / "cad_environment"
    / "02_crash"
    / "01_shell_tube_barrier_impact.cad"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "crash_validation" / "results"


def _setting_keys(cad_path: Path) -> dict[str, str]:
    session = json.loads(cad_path.read_text(encoding="utf-8"))
    controls = discover_override_controls(session)
    by_property = {}
    for control in controls:
        prop = str(control.get("property") or "")
        by_property.setdefault(prop, str(control["key"]))
    required = (
        "element_size",
        "refinement_size",
        "enable_mass_scaling",
        "time_step_scale",
    )
    missing = [name for name in required if name not in by_property]
    if missing:
        raise RuntimeError(
            "Crash qualification could not discover settings: "
            + ", ".join(missing)
        )
    return by_property


def _compact_result(result, case_id: str, settings: Mapping[str, float]) -> dict:
    raw = result.raw()
    return {
        "case_id": case_id,
        "settings": dict(settings),
        "external_status": raw.get("external_status"),
        "quality_status": raw.get("quality_status"),
        "ml_eligible": bool(raw.get("ml_eligible")),
        "metrics": dict(raw.get("crash_metrics") or {}),
        "quality": dict(raw.get("quality") or {}),
        "provenance": dict(raw.get("provenance") or {}),
        "manifest_file": raw.get("manifest_file"),
        "work_dir": raw.get("work_dir"),
        "histories": dict(raw.get("histories") or {}),
        "warnings": list(raw.get("warnings") or []),
    }


def _run_case(
    cad_path: Path,
    case_id: str,
    settings: Mapping[str, float],
) -> dict:
    clear_cache()
    print(f"[qualification] START {case_id}: {dict(settings)}", flush=True)
    result = crash(str(cad_path), _settings=settings)
    compact = _compact_result(result, case_id, settings)
    print(
        f"[qualification] END {case_id}: "
        f"status={compact['external_status']} quality={compact['quality_status']}",
        flush=True,
    )
    return compact


def _read_benchmark(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Benchmark CSV is empty: {path}")

    def column(name: str):
        values = []
        for row in rows:
            text = str(row.get(name, "")).strip()
            if text:
                values.append(float(text))
            else:
                values.append(float("nan"))
        arr = np.asarray(values, dtype=float)
        return arr.tolist() if np.isfinite(arr).any() else None

    metadata_path = path.with_suffix(".json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    source_document = str(metadata.get("source_document") or "").strip()
    source_path = Path(source_document) if source_document else None
    if source_path is not None and not source_path.is_absolute():
        source_path = metadata_path.parent / source_path

    def sha256(file_path: Path | None) -> str | None:
        if file_path is None or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return {
        "benchmark_id": metadata.get("benchmark_id", path.stem),
        "time_ms": column("time_ms"),
        "force_kN": column("force_kN"),
        "acceleration_g": column("acceleration_g"),
        "displacement_mm": column("displacement_mm"),
        "force_displacement_force_kN": column(
            "force_displacement_force_kN"
        ),
        "traceability": {
            **metadata,
            "csv_path": str(path.resolve()),
            "csv_sha256": sha256(path),
            "metadata_path": (
                str(metadata_path.resolve())
                if metadata_path.is_file()
                else None
            ),
            "metadata_sha256": sha256(metadata_path),
            "source_document_path": (
                str(source_path.resolve())
                if source_path is not None and source_path.is_file()
                else None
            ),
            "source_document_sha256": sha256(source_path),
        },
    }


def _material_validation(
    path: Path | None,
    cases: list[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if path is None:
        return {
            "status": "fail",
            "reason": (
                "No traceable coupon/dynamic material validation dossier was "
                "provided. Numerical convergence cannot replace material validation."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "material_id",
        "test_standard",
        "lot_id",
        "rate_range_per_s",
        "curve_source",
        "status",
    }
    missing = sorted(required - set(payload))
    if missing:
        return {
            "status": "fail",
            "reason": "Material validation dossier is missing: " + ", ".join(missing),
            "source": str(path.resolve()),
        }
    try:
        rate_range = payload["rate_range_per_s"]
        expected_lot = str(payload["lot_id"])
        configured_rate_min = float(rate_range[0])
        configured_rate_max = float(rate_range[1])
    except (TypeError, ValueError, IndexError):
        return {
            "status": "fail",
            "reason": "Material dossier rate range or lot ID is invalid.",
            "source": str(path.resolve()),
        }
    case_materials = [
        case.get("provenance", {})
        .get("physics_inputs", {})
        .get("material", {})
        for case in (cases or [])
    ]
    validation = validate_material_dossier(
        path,
        expected_lot_id=expected_lot,
        configured_rate_min=configured_rate_min,
        configured_rate_max=configured_rate_max,
        rate_model_required=any(
            float(material.get("strain_rate_c") or 0.0) > 0.0
            and float(material.get("strain_rate_p") or 0.0) > 0.0
            for material in case_materials
        ),
        failure_model_required=any(
            bool(material.get("enable_fracture"))
            for material in case_materials
        ),
    )
    expected_hash = validation.get("validation_report_sha256")
    embedded = [
        material.get("validation", {})
        for material in case_materials
    ]
    covered = bool(embedded) and all(
        item.get("status") == "pass"
        and item.get("material_lot_id") == expected_lot
        and item.get("validation_report_sha256") == expected_hash
        for item in embedded
    )
    validation["all_solver_cases_use_dossier"] = covered
    validation["solver_case_count"] = len(case_materials)
    if validation.get("status") == "pass" and not covered:
        validation["status"] = "fail"
        validation["reason"] = (
            "The dossier is valid, but not every saved solver case was run "
            "with the same validated material lot and dossier hash."
        )
    return validation


def run_baseline(cad_path: Path, output_dir: Path) -> Path:
    result = _run_case(cad_path, "baseline", {})
    target = output_dir / "baseline_result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


def rebuild_baseline_result(output_dir: Path) -> Path:
    """Reprocess a saved baseline without launching the solver again."""
    target = output_dir / "baseline_result.json"
    if not target.is_file():
        raise FileNotFoundError(f"No saved baseline result was found: {target}")
    case = json.loads(target.read_text(encoding="utf-8"))
    updated = _requalify_case(case)
    target.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return target


def _requalify_case(case: Mapping[str, object]) -> dict[str, object]:
    """Apply the current gate schema to saved raw/processed histories."""
    updated = dict(case)
    provenance = updated.get("provenance", {})
    solver_settings = (
        provenance.get("physics_inputs", {}).get("solver_settings", {})
        if isinstance(provenance, Mapping)
        else {}
    )
    end_time_ms = float(solver_settings.get("end_time_ms") or 0.0)
    histories = dict(updated.get("histories", {}))
    measurement = dict(histories.get("measurement", {}))
    mesh_info = (
        provenance.get("physics_inputs", {}).get("mesh", {})
        if isinstance(provenance, Mapping)
        else {}
    )
    measurement["source_point_count"] = int(
        mesh_info.get("point_count") or 0
    )
    work_value = str(updated.get("work_dir") or "").strip()
    work_dir = Path(work_value) if work_value else None
    deck_paths = (
        sorted(work_dir.glob("*.k"))
        if work_dir is not None and work_dir.is_dir()
        else []
    )
    if deck_paths and histories.get("raw"):
        measurement["solver_diagnostics"] = _read_radioss_diagnostics(
            work_dir,
            deck_paths[0].stem,
        )
        engine_executable = (
            provenance.get("executables", {}).get("engine", {}).get("path")
            if isinstance(provenance, Mapping)
            else None
        )
        solver_history = read_openradioss_time_history(
            work_dir,
            deck_paths[0].stem,
            solver_executable=engine_executable,
        )
        frames, frame_warnings = read_animation_frames(
            work_dir,
            deck_paths[0].stem,
            converter=resolve_anim_to_vtk(),
            end_time=end_time_ms,
        )
        if frames:
            histories = build_crash_measurements(
                solver_history=solver_history,
                frames=frames,
                measurement=measurement,
                acceleration_cfc=int(
                    solver_settings.get("acceleration_cfc") or 60
                ),
                force_cfc=int(
                    solver_settings.get("force_cfc") or 600
                ),
            )
            updated["histories"] = histories
            updated["metrics"] = dict(histories.get("metrics", {}))
        if frame_warnings:
            updated["warnings"] = list(updated.get("warnings", [])) + list(
                frame_warnings
            )
        history_warnings = solver_history.get("warnings") or []
        if history_warnings:
            updated["warnings"] = list(updated.get("warnings", [])) + list(
                history_warnings
            )
    quality = evaluate_crash_quality(
        histories,
        external_status=str(updated.get("external_status") or ""),
        end_time_ms=end_time_ms,
    )
    updated["quality"] = quality
    updated["quality_status"] = quality["status"]
    updated["ml_eligible"] = bool(quality["ml_eligible"])
    return updated


def rebuild_existing_report(
    output_dir: Path,
    benchmark_csv: Path | None,
    material_validation_path: Path | None,
) -> Path:
    """Rebuild a matrix report without rerunning expensive solver cases."""
    cases_path = output_dir / "cases.json"
    if not cases_path.is_file():
        raise FileNotFoundError(
            f"No saved numerical cases were found: {cases_path}"
        )
    cases = [
        _requalify_case(case)
        for case in json.loads(cases_path.read_text(encoding="utf-8"))
    ]
    mesh_cases = [
        case for case in cases if str(case.get("case_id", "")).startswith("mesh_")
    ]
    timestep_cases = [
        case
        for case in cases
        if str(case.get("case_id", "")).startswith("timestep_")
    ]
    repeats = (
        [timestep_cases[-1]["metrics"]]
        if timestep_cases
        else []
    )
    repeats.extend(
        case["metrics"]
        for case in cases
        if str(case.get("case_id", "")).startswith("repeat_")
    )

    mesh_convergence = assess_convergence(mesh_cases)
    timestep_convergence = assess_convergence(timestep_cases)
    temporal_evidence = []
    previous_mean = None
    temporal_order_valid = True
    for case in timestep_cases:
        timestep = np.asarray(
            case.get("histories", {}).get("raw", {}).get("timestep_ms", []),
            dtype=float,
        )
        timestep = timestep[np.isfinite(timestep) & (timestep > 0.0)]
        settings = (
            case.get("provenance", {})
            .get("physics_inputs", {})
            .get("solver_settings", {})
        )
        mean_dt = float(np.mean(timestep)) if timestep.size else None
        if (
            previous_mean is not None
            and mean_dt is not None
            and not mean_dt < previous_mean
        ):
            temporal_order_valid = False
        if mean_dt is not None:
            previous_mean = mean_dt
        temporal_evidence.append(
            {
                "case_id": case.get("case_id"),
                "time_step_scale": settings.get("time_step_scale"),
                "mass_scaling_dt_ms": settings.get("mass_scaling_dt_ms"),
                "minimum_time_step_us": (
                    float(np.min(timestep) * 1000.0)
                    if timestep.size
                    else None
                ),
                "mean_time_step_us": (
                    mean_dt * 1000.0 if mean_dt is not None else None
                ),
                "maximum_time_step_us": (
                    float(np.max(timestep) * 1000.0)
                    if timestep.size
                    else None
                ),
                "added_mass_percent": float(
                    case.get("quality", {}).get("added_mass_ratio") or 0.0
                )
                * 100.0,
            }
        )
    timestep_convergence["time_step_control"] = {
        "policy": "mass-neutral OpenRadioss /DT/NODA/STOP",
        "actual_time_step_strictly_decreases": temporal_order_valid,
        "cases": temporal_evidence,
        "status": "pass" if temporal_order_valid else "fail",
    }
    if not temporal_order_valid:
        timestep_convergence["status"] = "fail"
    repeatability = assess_repeatability(repeats)
    material = _material_validation(material_validation_path, cases=cases)
    if benchmark_csv is None:
        correlation = {
            "status": "fail",
            "reason": (
                "No traceable physical/reference benchmark CSV was supplied. "
                "The solver is numerically qualified but not physically correlated."
            ),
        }
    elif not timestep_cases:
        correlation = {
            "status": "fail",
            "reason": "No time-step case is available for benchmark correlation.",
        }
    else:
        benchmark = _read_benchmark(benchmark_csv)
        correlation = correlate_crash_benchmark(
            timestep_cases[-1]["histories"],
            benchmark,
        )

    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return write_validation_report(
        output_dir / "qualification_report.json",
        solver_quality=summarize_solver_quality(cases),
        mesh_convergence=mesh_convergence,
        timestep_convergence=timestep_convergence,
        repeatability=repeatability,
        material_validation=material,
        benchmark_correlation=correlation,
    )


def run_numerical_matrix(
    cad_path: Path,
    output_dir: Path,
    mesh_sizes: list[float],
    time_step_scales: list[float],
    repeat_count: int,
    benchmark_csv: Path | None,
    material_validation_path: Path | None,
) -> Path:
    keys = _setting_keys(cad_path)
    cases = []
    mesh_cases = []
    for index, size in enumerate(mesh_sizes):
        settings = {
            keys["element_size"]: float(size),
            keys["refinement_size"]: 0.5 * float(size),
        }
        case = _run_case(cad_path, f"mesh_{index}_{size:g}mm", settings)
        mesh_cases.append(case)
        cases.append(case)

    if len(time_step_scales) < 2:
        raise ValueError("At least two time-step scale factors are required.")
    if any(not 0.0 < float(value) <= 0.9 for value in time_step_scales):
        raise ValueError("Time-step scale factors must be > 0 and <= 0.9.")

    timestep_cases = []
    fixed_mesh = float(mesh_sizes[-1])
    for index, scale in enumerate(time_step_scales):
        settings = {
            keys["element_size"]: fixed_mesh,
            keys["refinement_size"]: 0.5 * fixed_mesh,
            keys["enable_mass_scaling"]: False,
            keys["time_step_scale"]: float(scale),
        }
        case = _run_case(
            cad_path,
            f"timestep_{index}_sf{float(scale):g}",
            settings,
        )
        timestep_cases.append(case)
        cases.append(case)

    repeats = []
    repeat_settings = {
        keys["element_size"]: fixed_mesh,
        keys["refinement_size"]: 0.5 * fixed_mesh,
        keys["enable_mass_scaling"]: False,
        keys["time_step_scale"]: float(time_step_scales[-1]),
    }
    # Reuse the last time-step case as the first repeat.
    repeats.append(timestep_cases[-1]["metrics"])
    for index in range(1, max(int(repeat_count), 2)):
        case = _run_case(cad_path, f"repeat_{index}", repeat_settings)
        repeats.append(case["metrics"])
        cases.append(case)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cases.json").write_text(
        json.dumps(cases, indent=2),
        encoding="utf-8",
    )
    return rebuild_existing_report(
        output_dir,
        benchmark_csv=benchmark_csv,
        material_validation_path=material_validation_path,
    )


def run_mesh_extension(
    cad_path: Path,
    output_dir: Path,
    mesh_sizes: list[float],
    benchmark_csv: Path | None,
    material_validation_path: Path | None,
    baseline_mesh_size: float | None = None,
) -> Path:
    """Run/replace selected mesh cases while preserving time-step/repeat runs."""
    cases_path = output_dir / "cases.json"
    existing = (
        json.loads(cases_path.read_text(encoding="utf-8"))
        if cases_path.is_file()
        else []
    )
    keys = _setting_keys(cad_path)
    mesh_by_size: dict[float, dict] = {}
    non_mesh = []
    for case in existing:
        match = re.search(
            r"mesh_\d+_([-+0-9.eE]+)mm$",
            str(case.get("case_id", "")),
        )
        if match:
            work_value = str(case.get("work_dir") or "").strip()
            if work_value and Path(work_value).is_dir():
                mesh_by_size[float(match.group(1))] = case
            else:
                print(
                    "[qualification] DROP stale mesh case without raw solver "
                    f"artifacts: {case.get('case_id')}",
                    flush=True,
                )
        else:
            non_mesh.append(case)
    if baseline_mesh_size is not None:
        baseline_path = output_dir / "baseline_result.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(
                "Baseline mesh reuse was requested, but no baseline result "
                f"exists: {baseline_path}"
            )
        baseline = _requalify_case(
            json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        baseline_work = str(baseline.get("work_dir") or "").strip()
        if not baseline_work or not Path(baseline_work).is_dir():
            raise FileNotFoundError(
                "Baseline mesh reuse requires the raw solver run directory."
            )
        baseline = dict(baseline)
        baseline["settings"] = {
            keys["element_size"]: float(baseline_mesh_size),
            keys["refinement_size"]: 0.5 * float(baseline_mesh_size),
        }
        mesh_by_size[float(baseline_mesh_size)] = baseline
        print(
            "[qualification] REUSE baseline as mesh point: "
            f"{baseline_mesh_size:g} mm",
            flush=True,
        )
    for size in mesh_sizes:
        if (
            baseline_mesh_size is not None
            and np.isclose(float(size), float(baseline_mesh_size))
        ):
            continue
        settings = {
            keys["element_size"]: float(size),
            keys["refinement_size"]: 0.5 * float(size),
        }
        mesh_by_size[float(size)] = _run_case(
            cad_path,
            f"mesh_pending_{size:g}mm",
            settings,
        )
    ordered_mesh = []
    for index, size in enumerate(sorted(mesh_by_size, reverse=True)):
        case = dict(mesh_by_size[size])
        case["case_id"] = f"mesh_{index}_{size:g}mm"
        ordered_mesh.append(case)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        json.dumps(ordered_mesh + non_mesh, indent=2),
        encoding="utf-8",
    )
    return rebuild_existing_report(
        output_dir,
        benchmark_csv=benchmark_csv,
        material_validation_path=material_validation_path,
    )


def run_temporal_extension(
    cad_path: Path,
    output_dir: Path,
    reference_mesh_size: float,
    time_step_scales: list[float],
    repeat_count: int,
    benchmark_csv: Path | None,
    material_validation_path: Path | None,
    resume_existing: bool = False,
) -> Path:
    """Replace temporal/repeat cases using a converged, mass-neutral mesh."""
    if len(time_step_scales) < 2:
        raise ValueError("At least two time-step scale factors are required.")
    if any(not 0.0 < float(value) <= 0.9 for value in time_step_scales):
        raise ValueError("Time-step scale factors must be > 0 and <= 0.9.")

    cases_path = output_dir / "cases.json"
    existing = (
        json.loads(cases_path.read_text(encoding="utf-8"))
        if cases_path.is_file()
        else []
    )
    mesh_cases = [
        case
        for case in existing
        if str(case.get("case_id", "")).startswith("mesh_")
    ]
    if not mesh_cases:
        raise FileNotFoundError(
            "Temporal qualification requires saved mesh-convergence cases."
        )
    reference_suffix = f"_{float(reference_mesh_size):g}mm"
    reference_cases = [
        case
        for case in mesh_cases
        if str(case.get("case_id", "")).endswith(reference_suffix)
    ]
    if not reference_cases:
        raise ValueError(
            "The requested reference mesh is not present in cases.json: "
            f"{reference_mesh_size:g} mm"
        )
    reference_case = reference_cases[-1]

    keys = _setting_keys(cad_path)
    common = {
        keys["element_size"]: float(reference_mesh_size),
        keys["refinement_size"]: 0.5 * float(reference_mesh_size),
        keys["enable_mass_scaling"]: False,
    }
    candidates_by_scale: dict[float, list[dict]] = {
        float(scale): [] for scale in time_step_scales
    }
    if resume_existing:
        reference_hash = (
            reference_case.get("provenance", {})
            .get("physics_inputs", {})
            .get("mesh", {})
            .get("sha256")
        )
        runs_root = REPO_ROOT / "external_solvers" / "runs"
        for manifest_path in sorted(
            runs_root.glob("pylcss_openradioss_*/pylcss_crash_manifest.json")
        ):
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            provenance = payload.get("provenance", {})
            physics = provenance.get("physics_inputs", {})
            solver = physics.get("solver_settings", {})
            if physics.get("mesh", {}).get("sha256") != reference_hash:
                continue
            if float(solver.get("mass_scaling_dt_ms") or 0.0) != 0.0:
                continue
            scale_value = solver.get("time_step_scale")
            if scale_value is None:
                continue
            matched = next(
                (
                    value
                    for value in candidates_by_scale
                    if np.isclose(float(scale_value), value)
                ),
                None,
            )
            if matched is None:
                continue
            work_dir = manifest_path.parent
            diagnostics = _read_radioss_diagnostics(
                work_dir,
                "pylcss_openradioss",
            )
            if not all(
                stage.get("normal_termination")
                for stage in diagnostics.values()
            ):
                continue
            recovered = dict(reference_case)
            recovered.update(
                {
                    "settings": {
                        **common,
                        keys["time_step_scale"]: float(matched),
                    },
                    "external_status": "engine_completed",
                    "provenance": provenance,
                    "manifest_file": str(manifest_path.resolve()),
                    "work_dir": str(work_dir.resolve()),
                    "warnings": [],
                }
            )
            candidates_by_scale[matched].append(
                _requalify_case(recovered)
            )
        for candidates in candidates_by_scale.values():
            candidates.sort(
                key=lambda case: str(
                    case.get("provenance", {}).get("created_utc", "")
                )
            )

    temporal_cases = []
    repeats = []

    def save_progress() -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cases_path.write_text(
            json.dumps(
                mesh_cases + temporal_cases + repeats,
                indent=2,
            ),
            encoding="utf-8",
        )

    for index, scale in enumerate(time_step_scales):
        scale = float(scale)
        settings = {
            **common,
            keys["time_step_scale"]: scale,
        }
        recovered = candidates_by_scale[scale]
        if recovered:
            case = dict(recovered.pop(0))
            case["case_id"] = f"timestep_{index}_sf{scale:g}"
            print(
                f"[qualification] RESUME {case['case_id']} from "
                f"{case.get('work_dir')}",
                flush=True,
            )
        else:
            case = _run_case(
                cad_path,
                f"timestep_{index}_sf{scale:g}",
                settings,
            )
        temporal_cases.append(case)
        save_progress()

    repeat_settings = {
        **common,
        keys["time_step_scale"]: float(time_step_scales[-1]),
    }
    for index in range(1, max(int(repeat_count), 2)):
        recovered = candidates_by_scale[float(time_step_scales[-1])]
        if recovered:
            case = dict(recovered.pop(0))
            case["case_id"] = f"repeat_{index}"
            print(
                f"[qualification] RESUME {case['case_id']} from "
                f"{case.get('work_dir')}",
                flush=True,
            )
        else:
            case = _run_case(
                cad_path,
                f"repeat_{index}",
                repeat_settings,
            )
        repeats.append(case)
        save_progress()

    return rebuild_existing_report(
        output_dir,
        benchmark_csv=benchmark_csv,
        material_validation_path=material_validation_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "baseline-report",
            "numerical",
            "mesh",
            "temporal",
            "report",
        ),
        default="baseline",
    )
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-sizes", nargs="+", type=float, default=[10.0, 8.0, 6.0])
    parser.add_argument(
        "--time-step-scales",
        nargs="+",
        type=float,
        default=[0.9, 0.67, 0.5],
    )
    parser.add_argument("--reference-mesh-size", type=float)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--baseline-mesh-size",
        type=float,
        help=(
            "In mesh mode, reuse baseline_result.json as this mesh size "
            "instead of rerunning that point."
        ),
    )
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--material-validation", type=Path)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse normally terminated matching temporal runs by provenance "
            "and continue only missing cases."
        ),
    )
    args = parser.parse_args()

    cad_path = args.cad.resolve()
    output_dir = args.output.resolve()
    if args.mode == "baseline":
        target = run_baseline(cad_path, output_dir)
    elif args.mode == "baseline-report":
        target = rebuild_baseline_result(output_dir)
    elif args.mode == "numerical":
        target = run_numerical_matrix(
            cad_path,
            output_dir,
            mesh_sizes=args.mesh_sizes,
            time_step_scales=args.time_step_scales,
            repeat_count=args.repeat_count,
            benchmark_csv=(
                args.benchmark_csv.resolve() if args.benchmark_csv else None
            ),
            material_validation_path=(
                args.material_validation.resolve()
                if args.material_validation
                else None
            ),
        )
    elif args.mode == "mesh":
        target = run_mesh_extension(
            cad_path,
            output_dir,
            mesh_sizes=args.mesh_sizes,
            benchmark_csv=(
                args.benchmark_csv.resolve() if args.benchmark_csv else None
            ),
            material_validation_path=(
                args.material_validation.resolve()
                if args.material_validation
                else None
            ),
            baseline_mesh_size=args.baseline_mesh_size,
        )
    elif args.mode == "temporal":
        if args.reference_mesh_size is None:
            parser.error("--mode temporal requires --reference-mesh-size")
        target = run_temporal_extension(
            cad_path,
            output_dir,
            reference_mesh_size=args.reference_mesh_size,
            time_step_scales=args.time_step_scales,
            repeat_count=args.repeat_count,
            benchmark_csv=(
                args.benchmark_csv.resolve() if args.benchmark_csv else None
            ),
            material_validation_path=(
                args.material_validation.resolve()
                if args.material_validation
                else None
            ),
            resume_existing=args.resume_existing,
        )
    else:
        target = rebuild_existing_report(
            output_dir,
            benchmark_csv=(
                args.benchmark_csv.resolve() if args.benchmark_csv else None
            ),
            material_validation_path=(
                args.material_validation.resolve()
                if args.material_validation
                else None
            ),
        )
    print(f"[qualification] REPORT {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
