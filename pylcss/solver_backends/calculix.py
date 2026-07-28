# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX static-structural backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pylcss.solver_backends.base import ExternalRunConfig, SolverBackendError
from pylcss.solver_backends.calculix_deck import (
    _build_input_deck,
)
from pylcss.solver_backends.calculix_results import (
    _ingest_frd_into_result,
)
from pylcss.solver_backends.execution import (
    make_work_dir,
    resolve_executable,
    run_process,
    tail,
)
from pylcss.solver_backends.validation import validate_isotropic_material
from pylcss.solver_backends.validation import record_list


def _collect_calculix_failure_context(
    work_dir: Path, job_name: str, returncode: int, executable: str
) -> str:
    """Build a diagnostic block when ccx exits non-zero with little to no stdout.

    Reads the small log files CalculiX writes (.sta, .cvg, .dat, .log) and
    surfaces them.  Also detects the Windows 0xC0000135 / -1073741515 case
    where the .exe failed to load because of a missing DLL.
    """
    parts: list[str] = []
    parts.append(f"Exit code: {returncode} (0x{(returncode & 0xFFFFFFFF):08X})")
    parts.append(f"Executable: {executable}")

    if returncode in (-1073741515, 3221225781):
        parts.append(
            "Windows reports STATUS_DLL_NOT_FOUND (0xC0000135).  This ccx "
            "binary needs a runtime DLL that is not next to it on PATH.  The "
            "'ccx_dynamic.exe' build requires Intel MKL's mkl_rt.2.dll; use "
            "'ccx_static.exe' instead (already preferred by install_solvers.py "
            "as of CalculiX 2.23).  Re-run:  python scripts/install_solvers.py "
            "--only ccx  to refresh the path."
        )

    for ext in (".sta", ".cvg", ".log", ".dat"):
        path = work_dir / f"{job_name}{ext}"
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        content = content.strip()
        if not content:
            continue
        parts.append(f"--- {path.name} (last 2000 chars) ---\n{tail(content, 2000)}")
    return "\n".join(parts)


def resolve_calculix_executable(override: str | None = None) -> str | None:
    """Resolve the preferred CalculiX executable without launching it."""
    return resolve_executable(
        override,
        env_vars=("PYLCSS_CALCULIX_CCX", "CALCULIX_CCX"),
        # Prefer the self-contained static build.  Dynamic builds are a final
        # fallback because they may require Intel MKL runtime DLLs.
        candidates=(
            "ccx_static",
            "ccx_static.exe",
            "ccx",
            "ccx.exe",
            "ccx_dynamic",
            "ccx_dynamic.exe",
        ),
    )


def run_calculix_static(
    mesh: Any,
    material: dict,
    constraints: list[dict],
    loads: list[dict],
    config: ExternalRunConfig,
    visualization_mode: str = "Von Mises Stress",
    analysis_type: str = "Linear",
    deformation_scale: Any = "Auto",
) -> dict:
    """Write and optionally run a CalculiX static analysis deck.

    ``analysis_type`` is one of ``'Linear'``, ``'Nonlinear (Geometric)'``,
    or ``'Nonlinear (Plastic)'``.  Plasticity is enabled only by the explicit
    nonlinear-plastic selection.
    """
    warnings: list[str] = []
    allowed_analysis = {"Linear", "Nonlinear (Geometric)", "Nonlinear (Plastic)"}
    if analysis_type not in allowed_analysis:
        raise SolverBackendError(
            f"Unsupported CalculiX analysis type: {analysis_type!r}."
        )
    validate_isotropic_material(
        material,
        require_yield=analysis_type == "Nonlinear (Plastic)",
    )
    constraints = record_list(constraints, label="Constraints")
    loads = record_list(loads, label="Loads")
    effective_analysis_type = analysis_type

    job_name = config.validated_job_name(default="pylcss_calculix")
    timeout_s = config.validated_timeout() if config.run_solver else config.timeout_s
    work_dir = make_work_dir("pylcss_calculix_", config.work_dir)
    inp_path = work_dir / f"{job_name}.inp"
    try:
        inp_path.write_text(
            _build_input_deck(
                mesh,
                material,
                constraints,
                loads,
                warnings,
                analysis_type=effective_analysis_type,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise SolverBackendError(
            f"Could not write CalculiX input deck {inp_path}: {exc}"
        ) from exc

    executable = resolve_calculix_executable(config.executable)

    status = "deck_written"
    solver_log = ""
    result_payload: dict = {}
    if config.run_solver:
        if executable is None:
            raise SolverBackendError(
                "CalculiX executable not found. Set the node path, add ccx to PATH, "
                "define PYLCSS_CALCULIX_CCX, or run scripts/install_solvers.py."
            )
        else:
            exe_dir = str(Path(executable).resolve().parent)
            proc = run_process(
                [executable, job_name],
                cwd=work_dir,
                timeout_s=timeout_s,
                extra_path_dirs=(exe_dir,),
                cancel_callback=config.cancel_callback,
            )
            solver_log = tail(proc.stdout or "")
            if proc.returncode != 0:
                # CCX writes its own diagnostics to .sta / .dat even when stdout
                # is empty (which it always is when the .exe fails to load due
                # to a missing DLL).  Capture those for the user.
                aux = _collect_calculix_failure_context(
                    work_dir, job_name, proc.returncode, executable
                )
                raise SolverBackendError(
                    "CalculiX failed. Last solver output:\n"
                    + (solver_log or "(stdout was empty)\n")
                    + "\n"
                    + aux
                )
            status = "completed"
            frd_path = work_dir / f"{job_name}.frd"
            if not frd_path.is_file():
                raise SolverBackendError(
                    f"CalculiX run completed but {frd_path.name} was not produced. "
                    "Check the .dat / .sta logs in the work directory."
                )
            else:
                try:
                    result_payload = _ingest_frd_into_result(
                        mesh,
                        frd_path,
                        visualization_mode,
                        warnings,
                        deformation_scale=deformation_scale,
                        analysis_type=effective_analysis_type,
                        constraints=constraints,
                    )
                    rho = float(material.get("rho", material.get("density", 0.0)))
                    if rho > 0.0 and "volume" in result_payload:
                        result_payload["mass"] = float(result_payload["volume"] * rho)
                except Exception as exc:
                    raise SolverBackendError(
                        f"CalculiX FRD ingest failed: {exc}"
                    ) from exc

    output = {
        "type": result_payload.get("type", "external_solver"),
        "backend": "CalculiX",
        "external_status": status,
        "mesh": mesh,
        "visualization_mode": visualization_mode,
        "input_file": str(inp_path),
        "work_dir": str(work_dir),
        "solver_executable": executable,
        "solver_log": solver_log,
        "analysis_type": effective_analysis_type,
        "warnings": warnings,
        "message": (
            "CalculiX deck generated. Enable external execution and configure ccx "
            "to run the solve from PyLCSS."
            if status == "deck_written"
            else (
                "CalculiX run complete; results imported."
                if result_payload
                else "CalculiX run finished but no FRD results could be loaded."
            )
        ),
    }
    output.update(result_payload)
    return output
