# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss orchestration for generated and user-supplied crash decks."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pylcss.solver_backends.base import ExternalRunConfig, SolverBackendError
from pylcss.solver_backends.execution import make_work_dir, tail
from pylcss.solver_backends.openradioss_deck import (
    _build_engine_deck,
    _build_keyword_deck,
)
from pylcss.solver_backends.openradioss_impact import _keyword_card
from pylcss.solver_backends.openradioss_results import (
    align_animation_frames as _build_animation_frames_with_mesh,
    build_existing_deck_result,
    build_generated_crash_result,
    build_generated_fallback_result,
    compute_time_history as _compute_time_history,
    read_engine_energy_history as _read_engine_energy_history,
    wrap_deck_result as _wrap_deck_result,
)
from pylcss.solver_backends.openradioss_runtime import (
    read_radioss_diagnostics,
    resolve_openradioss_executables,
    run_engine,
    run_starter,
    stage_file,
)
from pylcss.solver_backends.radioss_time_history import (
    read_openradioss_time_history,
)
from pylcss.solver_backends.radioss_reader import (
    read_animation_frames,
    resolve_anim_to_vtk,
)
from pylcss.solver_backends.validation import (
    nonnegative_float,
    positive_float,
    record_list,
    validate_isotropic_material,
)


logger = logging.getLogger(__name__)

__all__ = [
    "_compute_time_history",
    "_build_animation_frames_with_mesh",
    "_build_engine_deck",
    "_keyword_card",
    "_read_engine_energy_history",
    "run_openradioss_crash",
    "run_openradioss_existing_deck",
]


def _model_job_name(deck_path: Path) -> str:
    stem = deck_path.stem
    for suffix in ("_0000", "_0001"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _mesh_element_count(mesh: Any) -> int | None:
    """Return the element count used to size Engine's SMP width.

    ``mesh.t`` follows the scikit-fem convention of one column per element.
    Returns ``None`` when the count cannot be read, which leaves the thread
    heuristic on its full-machine default.
    """
    connectivity = getattr(mesh, "t", None)
    if connectivity is None:
        return None
    try:
        shape = np.asarray(connectivity).shape
    except (TypeError, ValueError):
        return None
    if len(shape) != 2 or shape[1] <= 0:
        return None
    return int(shape[1])


def _find_engine_deck(work_dir: Path, job_name: str) -> Path:
    preferred = work_dir / f"{job_name}_0001.rad"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        work_dir.glob("*_0001.rad"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SolverBackendError(
            "Starter completed but no `_0001.rad` Engine deck was produced."
        )
    return candidates[0]


def _write_text(path: Path, content: str, *, label: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise SolverBackendError(f"Could not write {label} {path}: {exc}") from exc


def _write_engine_deck(
    path: Path,
    *,
    job_name: str,
    end_time: float,
    output_dt: float,
    history_dt: float | None,
    mass_scaling_dt: float,
    mass_scaling_scale: float,
    time_step_scale: float,
) -> None:
    _write_text(
        path,
        _build_engine_deck(
            job_name,
            end_time,
            output_dt,
            history_dt=history_dt,
            mass_scaling_dt=mass_scaling_dt,
            mass_scaling_scale=mass_scaling_scale,
            time_step_scale=time_step_scale,
        ),
        label="OpenRadioss Engine deck",
    )


def _deck_only_existing_result(
    *,
    work_dir: Path,
    deck_path: Path,
    engine_deck_path: Path | None,
    starter: str | None,
    engine: str | None,
    visualization_mode: str,
    displacement_scale: float,
) -> dict[str, Any]:
    return {
        "type": "external_solver",
        "backend": "OpenRadioss",
        "external_status": "deck_staged",
        "mesh": None,
        "visualization_mode": visualization_mode,
        "disp_scale": displacement_scale,
        "input_file": str(deck_path),
        "engine_file": str(engine_deck_path) if engine_deck_path else None,
        "work_dir": str(work_dir),
        "solver_executable": starter,
        "secondary_solver_executable": engine,
        "solver_log": "",
        "warnings": [
            "deck_only=True - the deck was staged but neither Starter nor "
            "Engine was run. Uncheck deck_only on the node to launch."
        ],
        "message": (
            f"Deck staged in {work_dir}. Toggle off `deck_only` to run "
            "Starter + Engine on this deck."
        ),
    }


def _scale_frame_stress(
    frames: list[dict[str, Any]],
    scale_to_mpa: float,
) -> None:
    if scale_to_mpa == 1.0:
        return
    for frame in frames:
        for field_name in ("stress_vm", "stress_vm_cell"):
            values = frame.get(field_name)
            if values is not None:
                frame[field_name] = np.asarray(values, dtype=float) * scale_to_mpa


def run_openradioss_existing_deck(
    deck_path: str | Path,
    config: ExternalRunConfig,
    engine_deck_path: str | Path | None = None,
    end_time: float | None = None,
    visualization_mode: str = "Von Mises Stress",
    disp_scale: float = 1.0,
    stress_scale_to_mpa: float = 1.0,
) -> dict[str, Any]:
    """Run Starter and Engine on a user-supplied model deck."""
    timeout_s = config.validated_timeout() if config.run_solver else config.timeout_s
    disp_scale = positive_float(disp_scale, label="Displacement scale")
    stress_scale_to_mpa = positive_float(
        stress_scale_to_mpa,
        label="Stress conversion scale",
    )
    if end_time is not None:
        end_time = positive_float(end_time, label="Simulation end time")

    source_deck = Path(deck_path).resolve()
    if not source_deck.is_file():
        raise SolverBackendError(f"Deck file not found: {source_deck}")
    if source_deck.suffix.lower() == ".rad" and source_deck.stem.endswith("_0001"):
        raise SolverBackendError(
            "deck_path points at an Engine control file (*_0001.rad). Select "
            "the matching model/Starter deck (*_0000.rad) and pass the Engine "
            "file as engine_deck_path."
        )

    source_engine: Path | None = None
    if engine_deck_path is not None:
        source_engine = Path(engine_deck_path).resolve()
        if not source_engine.is_file():
            raise SolverBackendError(
                f"OpenRadioss Engine deck file not found: {source_engine}"
            )

    work_dir = make_work_dir("pylcss_radioss_deck_", config.work_dir)
    staged_deck = stage_file(source_deck, work_dir)
    starter, engine = resolve_openradioss_executables(
        config.executable,
        config.secondary_executable,
    )
    staged_engine = (
        stage_file(source_engine, work_dir)
        if source_engine is not None and not config.run_solver
        else None
    )
    if not config.run_solver:
        return _deck_only_existing_result(
            work_dir=work_dir,
            deck_path=staged_deck,
            engine_deck_path=staged_engine,
            starter=starter,
            engine=engine,
            visualization_mode=visualization_mode,
            displacement_scale=disp_scale,
        )

    if starter is None:
        raise SolverBackendError(
            "OpenRadioss Starter executable not found. Run "
            "scripts/install_solvers.py --only radioss or set "
            "PYLCSS_OPENRADIOSS_STARTER."
        )
    job_name = _model_job_name(staged_deck)
    starter_log = run_starter(
        starter,
        staged_deck,
        work_dir=work_dir,
        timeout_s=timeout_s,
        cancel_callback=config.cancel_callback,
        job_name=job_name,
        user_supplied=True,
    )
    status = "starter_completed"
    staged_engine = (
        stage_file(source_engine, work_dir)
        if source_engine is not None
        else _find_engine_deck(work_dir, job_name)
    )
    if engine is None:
        raise SolverBackendError(
            "OpenRadioss Engine executable not found. Run "
            "scripts/install_solvers.py --only radioss or set "
            "PYLCSS_OPENRADIOSS_ENGINE."
        )
    engine_log = run_engine(
        engine,
        staged_engine,
        work_dir=work_dir,
        timeout_s=timeout_s,
        cancel_callback=config.cancel_callback,
        job_name=_model_job_name(staged_engine),
        user_supplied=True,
    )
    solver_log = tail(engine_log + "\n" + starter_log)
    status = "engine_completed"

    frames, animation_warnings = read_animation_frames(
        work_dir,
        _model_job_name(staged_engine),
        converter=resolve_anim_to_vtk(),
        timeout_s=timeout_s,
        end_time=end_time,
    )
    warnings = list(animation_warnings)
    _scale_frame_stress(frames, stress_scale_to_mpa)
    if not frames:
        return _wrap_deck_result(
            status,
            work_dir,
            staged_deck,
            staged_engine,
            starter,
            engine,
            solver_log,
            warnings,
            visualization_mode,
            disp_scale,
        )
    return build_existing_deck_result(
        status=status,
        work_dir=work_dir,
        deck_path=staged_deck,
        engine_deck_path=staged_engine,
        starter_executable=starter,
        engine_executable=engine,
        solver_log=solver_log,
        warnings=warnings,
        visualization_mode=visualization_mode,
        displacement_scale=disp_scale,
        frames=frames,
        end_time=end_time,
        source_name=source_deck.name,
    )


def _validate_generated_inputs(
    material: dict[str, Any],
    constraints: Any,
    impact: Any,
    gravity: Any,
    *,
    end_time: Any,
    output_dt: Any,
    disp_scale: Any,
    mass_scaling_dt: Any,
    mass_scaling_scale: Any,
    impactor_mass: Any,
    hourglass_coefficient: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any] | None,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    if not isinstance(impact, Mapping):
        raise SolverBackendError(
            "OpenRadioss backend requires an impact-condition dictionary."
        )
    if gravity is not None and not isinstance(gravity, Mapping):
        raise SolverBackendError("Gravity must be a dictionary.")
    validate_isotropic_material(material, validate_strain_rate=True)
    validated_constraints = record_list(constraints, label="Crash constraints")
    validated_impact = dict(impact)
    validated_gravity = dict(gravity) if gravity is not None else None
    validated_end_time = positive_float(end_time, label="Simulation end time")
    validated_output_dt = positive_float(
        output_dt,
        label="Animation output interval",
    )
    validated_disp_scale = positive_float(disp_scale, label="Displacement scale")
    validated_mass_dt = nonnegative_float(
        mass_scaling_dt,
        label="Mass-scaling target time step",
    )
    validated_mass_scale = positive_float(
        mass_scaling_scale,
        label="Mass-scaling safety factor",
    )
    if validated_mass_scale > 1.0:
        raise SolverBackendError("Mass-scaling safety factor must not exceed 1.0.")
    return (
        validated_constraints,
        validated_impact,
        validated_gravity,
        validated_end_time,
        validated_output_dt,
        validated_disp_scale,
        validated_mass_dt,
        validated_mass_scale,
        nonnegative_float(impactor_mass, label="Impactor mass"),
        nonnegative_float(
            hourglass_coefficient,
            label="Hourglass coefficient",
        ),
    )


def _import_generated_frames(
    mesh: Any,
    *,
    work_dir: Path,
    job_name: str,
    timeout_s: float,
    end_time: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    raw_frames, animation_warnings = read_animation_frames(
        work_dir,
        job_name,
        converter=resolve_anim_to_vtk(),
        timeout_s=timeout_s,
        end_time=end_time,
    )
    warnings.extend(animation_warnings)
    frames = _build_animation_frames_with_mesh(mesh, raw_frames)
    if frames:
        maximum_displacement = max(
            (
                float(np.max(np.abs(np.asarray(frame["displacement"], dtype=float))))
                for frame in frames
                if np.asarray(frame["displacement"]).size
            ),
            default=0.0,
        )
        maximum_stress = max(
            (
                float(np.max(np.asarray(frame["stress_vm"], dtype=float)))
                for frame in frames
                if np.asarray(frame["stress_vm"]).size
            ),
            default=0.0,
        )
        logger.info(
            "OpenRadioss import: viewer fields ready; "
            "max |u|=%.4e mm, max |VM|=%.4e MPa",
            maximum_displacement,
            maximum_stress,
        )
    return frames


def run_openradioss_crash(
    mesh: Any,
    material: dict[str, Any],
    constraints: list[dict[str, Any]],
    impact: dict[str, Any],
    config: ExternalRunConfig,
    end_time: float,
    output_dt: float,
    history_dt: float | None = None,
    visualization_mode: str = "Von Mises Stress",
    disp_scale: float = 1.0,
    gravity: dict[str, Any] | None = None,
    mass_scaling_dt: float = 0.0,
    mass_scaling_scale: float = 0.9,
    time_step_scale: float = 0.9,
    impactor_mass: float = 0.0,
    hourglass_ihq: int = 4,
    hourglass_coefficient: float = 0.10,
    acceleration_cfc: int = 60,
    force_cfc: int = 600,
) -> dict[str, Any]:
    """Generate a crash deck, optionally solve it, and import animations."""
    (
        constraints,
        impact,
        gravity,
        end_time,
        output_dt,
        disp_scale,
        mass_scaling_dt,
        mass_scaling_scale,
        impactor_mass,
        hourglass_coefficient,
    ) = _validate_generated_inputs(
        material,
        constraints,
        impact,
        gravity,
        end_time=end_time,
        output_dt=output_dt,
        disp_scale=disp_scale,
        mass_scaling_dt=mass_scaling_dt,
        mass_scaling_scale=mass_scaling_scale,
        impactor_mass=impactor_mass,
        hourglass_coefficient=hourglass_coefficient,
    )
    timeout_s = config.validated_timeout() if config.run_solver else config.timeout_s
    history_dt = positive_float(
        output_dt if history_dt is None else history_dt,
        label="Time-history output interval",
    )
    time_step_scale = positive_float(
        time_step_scale,
        label="Time-step safety factor",
    )
    if time_step_scale > 0.9:
        raise SolverBackendError("Time-step safety factor must not exceed 0.9.")
    acceleration_cfc = int(acceleration_cfc)
    force_cfc = int(force_cfc)
    warnings: list[str] = []
    job_name = config.validated_job_name(default="pylcss_openradioss")
    work_dir = make_work_dir("pylcss_openradioss_", config.work_dir)
    deck_path = work_dir / f"{job_name}.k"
    engine_deck_path = work_dir / f"{job_name}_0001.rad"
    deck_metadata: dict[str, Any] = {}
    _write_text(
        deck_path,
        _build_keyword_deck(
            mesh=mesh,
            material=material,
            constraints=constraints,
            impact=impact,
            end_time=end_time,
            output_dt=output_dt,
            history_dt=history_dt,
            gravity=gravity,
            warnings=warnings,
            impactor_mass=impactor_mass,
            out_meta=deck_metadata,
            hourglass_ihq=hourglass_ihq,
            hourglass_coefficient=hourglass_coefficient,
        ),
        label="OpenRadioss input deck",
    )
    _write_engine_deck(
        engine_deck_path,
        job_name=job_name,
        end_time=end_time,
        output_dt=output_dt,
        history_dt=history_dt,
        mass_scaling_dt=mass_scaling_dt,
        mass_scaling_scale=mass_scaling_scale,
        time_step_scale=time_step_scale,
    )
    starter, engine = resolve_openradioss_executables(
        config.executable,
        config.secondary_executable,
    )
    status = "deck_written"
    solver_log = ""
    frames: list[dict[str, Any]] = []
    solver_history: dict[str, Any] = {}

    if config.run_solver:
        if starter is None:
            raise SolverBackendError(
                "OpenRadioss Starter executable not found. Set the node path, "
                "add starter_* to PATH, define PYLCSS_OPENRADIOSS_STARTER, or "
                "run scripts/install_solvers.py."
            )
        starter_log = run_starter(
            starter,
            deck_path,
            work_dir=work_dir,
            timeout_s=timeout_s,
            cancel_callback=config.cancel_callback,
            job_name=job_name,
            user_supplied=False,
        )
        status = "starter_completed"
        if engine is None:
            raise SolverBackendError(
                "Starter completed but no Engine executable was found; set "
                "the node Engine path or reinstall OpenRadioss."
            )

        # Starter rewrites this file, so restore the requested Engine controls.
        _write_engine_deck(
            engine_deck_path,
            job_name=job_name,
            end_time=end_time,
            output_dt=output_dt,
            history_dt=history_dt,
            mass_scaling_dt=mass_scaling_dt,
            mass_scaling_scale=mass_scaling_scale,
            time_step_scale=time_step_scale,
        )
        engine_log = run_engine(
            engine,
            engine_deck_path,
            work_dir=work_dir,
            timeout_s=timeout_s,
            cancel_callback=config.cancel_callback,
            job_name=job_name,
            user_supplied=False,
            element_count=_mesh_element_count(mesh),
        )
        solver_log = tail(engine_log + "\n" + starter_log)
        status = "engine_completed"
        frames = _import_generated_frames(
            mesh,
            work_dir=work_dir,
            job_name=job_name,
            timeout_s=timeout_s,
            end_time=end_time,
            warnings=warnings,
        )
        try:
            solver_history = read_openradioss_time_history(
                work_dir,
                job_name,
                solver_executable=engine,
                timeout_s=min(timeout_s, 120.0),
            )
            warnings.extend(str(item) for item in solver_history.get("warnings", []))
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"OpenRadioss T01 history could not be imported: {exc}")

    measurement = dict(deck_metadata.get("measurement") or {})
    measurement["solver_diagnostics"] = read_radioss_diagnostics(
        work_dir,
        job_name,
    )
    solver_settings = {
        "end_time_ms": end_time,
        "animation_output_dt_ms": output_dt,
        "history_output_dt_ms": history_dt,
        "mass_scaling_dt_ms": mass_scaling_dt,
        "mass_scaling_scale": mass_scaling_scale,
        "time_step_scale": time_step_scale,
        "hourglass_ihq": int(hourglass_ihq),
        "hourglass_coefficient": hourglass_coefficient,
        "acceleration_cfc": acceleration_cfc,
        "force_cfc": force_cfc,
    }

    if frames:
        return build_generated_crash_result(
            mesh=mesh,
            material=material,
            frames=frames,
            status=status,
            visualization_mode=visualization_mode,
            displacement_scale=disp_scale,
            wall=deck_metadata.get("wall"),
            end_time=end_time,
            deck_path=deck_path,
            engine_deck_path=engine_deck_path,
            work_dir=work_dir,
            job_name=job_name,
            starter_executable=starter,
            engine_executable=engine,
            solver_log=solver_log,
            warnings=warnings,
            solver_history=solver_history,
            measurement=measurement,
            impact=impact,
            constraints=constraints,
            solver_settings=solver_settings,
            acceleration_cfc=acceleration_cfc,
            force_cfc=force_cfc,
        )
    return build_generated_fallback_result(
        mesh=mesh,
        status=status,
        visualization_mode=visualization_mode,
        displacement_scale=disp_scale,
        wall=deck_metadata.get("wall"),
        end_time=end_time,
        deck_path=deck_path,
        engine_deck_path=engine_deck_path,
        work_dir=work_dir,
        starter_executable=starter,
        engine_executable=engine,
        solver_log=solver_log,
        warnings=warnings,
        material=material,
        impact=impact,
        constraints=constraints,
        solver_settings=solver_settings,
        solver_history=solver_history,
        measurement=measurement,
        job_name=job_name,
        acceleration_cfc=acceleration_cfc,
        force_cfc=force_cfc,
    )
