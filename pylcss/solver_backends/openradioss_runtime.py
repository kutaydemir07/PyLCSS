# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss-specific executable discovery and process supervision."""

from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import re
import shutil
import time

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.execution import resolve_executable, run_process, tail


logger = logging.getLogger(__name__)

_STARTER_NAMES = (
    "starter_win64.exe",
    "starter_win64",
    "starter_linux64_gf",
    "starter_linux64_gf_sp",
    "starter_linux64_gf_dp",
)
_ENGINE_NAMES = (
    "engine_win64.exe",
    "engine_win64",
    "engine_linux64_gf",
    "engine_linux64_gf_sp",
    "engine_linux64_gf_dp",
)


def read_radioss_diagnostics(
    work_dir: Path,
    job_name: str,
) -> dict[str, object]:
    """Read Starter/Engine termination and summary counts from OUT files."""

    def stage(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {
                "path": str(path),
                "available": False,
                "normal_termination": False,
                "error_count": None,
                "warning_count": None,
            }
        text = path.read_text(encoding="utf-8", errors="replace")
        errors = re.findall(r"(?mi)^\s*(\d+)\s+ERROR\(S\)\s*$", text)
        warnings = re.findall(r"(?mi)^\s*(\d+)\s+WARNING\(S\)\s*$", text)
        normal_termination = "NORMAL TERMINATION" in text.upper()
        return {
            "path": str(path.resolve()),
            "available": True,
            "normal_termination": normal_termination,
            "error_count": (
                int(errors[-1]) if errors else 0 if normal_termination else None
            ),
            "warning_count": (
                int(warnings[-1]) if warnings else 0 if normal_termination else None
            ),
        }

    return {
        "starter": stage(work_dir / f"{job_name}_0000.out"),
        "engine": stage(work_dir / f"{job_name}_0001.out"),
    }


def resolve_openradioss_executables(
    starter_override: str | None,
    engine_override: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the Starter and Engine binaries using the shared policy."""
    starter = resolve_executable(
        starter_override,
        env_vars=("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
        candidates=_STARTER_NAMES,
    )
    engine = resolve_executable(
        engine_override,
        env_vars=("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
        candidates=_ENGINE_NAMES,
    )
    return starter, engine


def stage_file(source: Path, work_dir: Path) -> Path:
    """Copy a deck into the run directory without overwriting itself."""
    destination = work_dir / source.name
    try:
        already_staged = destination.resolve() == source.resolve()
    except OSError:
        already_staged = False
    if already_staged:
        return destination
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise SolverBackendError(
            f"Could not stage OpenRadioss deck {source} in {work_dir}: {exc}"
        ) from exc
    return destination


def _runtime_environment(binary_path: str) -> tuple[list[str], dict[str, str]]:
    """Return paths required by an official OpenRadioss installation."""
    binary = Path(binary_path).resolve()
    extra_paths = [str(binary.parent)]
    root = next(
        (
            parent
            for parent in (binary.parent, *binary.parent.parents)
            if (parent / "extlib").is_dir()
        ),
        None,
    )
    if root is None:
        return extra_paths, {}

    platform = "win64" if binary.suffix.lower() == ".exe" else "linux64"
    for candidate in (
        root / "extlib" / "intelOneAPI_runtime" / platform,
        root / "extlib" / "hm_reader" / platform,
        root / "extlib" / "h3d" / "lib" / platform,
    ):
        if candidate.is_dir():
            extra_paths.append(str(candidate))

    environment: dict[str, str] = {}
    config_dir = root / "hm_cfg_files"
    if config_dir.is_dir():
        environment["RAD_CFG_PATH"] = str(config_dir)
    h3d_dir = root / "extlib" / "h3d" / "lib" / platform
    if h3d_dir.is_dir():
        environment["RAD_H3D_PATH"] = str(h3d_dir)
    return extra_paths, environment


def _failure_context(
    work_dir: Path,
    job_name: str,
    returncode: int,
    executable: str,
    stage: str,
) -> str:
    parts = [
        f"Stage: {stage}",
        f"Exit code: {returncode} (0x{(returncode & 0xFFFFFFFF):08X})",
        f"Executable: {executable}",
    ]
    if returncode in (-1073741515, 3221225781):
        parts.append(
            "Windows STATUS_DLL_NOT_FOUND (0xC0000135). A runtime DLL is "
            "missing; verify the OpenRadioss installation or rerun "
            "scripts/install_solvers.py --only radioss --force."
        )

    for name in (
        f"{job_name}_0000.out",
        f"{job_name}_0001.out",
        f"{job_name}_0000.txt",
        f"{job_name}_0001.txt",
        f"{job_name}.out",
        f"{job_name}.log",
    ):
        path = work_dir / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if content:
            parts.append(
                f"--- {path.name} (last 2500 chars) ---\n{tail(content, 2500)}"
            )
    return "\n".join(parts)


def _run_stage(
    *,
    executable: str,
    arguments: list[str],
    input_path: Path,
    work_dir: Path,
    timeout_s: float,
    cancel_callback: Callable[[], bool] | None,
    job_name: str,
    stage: str,
    failure_subject: str,
) -> str:
    extra_paths, extra_environment = _runtime_environment(executable)
    started_at = time.monotonic()
    logger.info("OpenRadioss %s: launching %s", stage, input_path.name)
    process = run_process(
        [executable, "-i", str(input_path), *arguments],
        cwd=work_dir,
        timeout_s=timeout_s,
        extra_path_dirs=extra_paths,
        extra_env=extra_environment,
        stdout_file=work_dir / f"_pylcss_{stage.lower()}.log",
        cancel_callback=cancel_callback,
    )
    logger.info(
        "OpenRadioss %s: completed in %.1fs (exit=%d)",
        stage,
        time.monotonic() - started_at,
        process.returncode,
    )
    output = tail(process.stdout or "")
    if process.returncode == 0:
        return output

    context = _failure_context(
        work_dir,
        job_name,
        process.returncode,
        executable,
        stage,
    )
    raise SolverBackendError(
        f"OpenRadioss {stage} failed{failure_subject}. Last solver output:\n"
        + (output or "(stdout was empty)\n")
        + "\n"
        + context
    )


def run_starter(
    executable: str,
    deck_path: Path,
    *,
    work_dir: Path,
    timeout_s: float,
    cancel_callback: Callable[[], bool] | None,
    job_name: str,
    user_supplied: bool,
) -> str:
    """Run Starter in the single SPMD domain required by the bundled Engine."""
    return _run_stage(
        executable=executable,
        arguments=["-nspmd", "1"],
        input_path=deck_path,
        work_dir=work_dir,
        timeout_s=timeout_s,
        cancel_callback=cancel_callback,
        job_name=job_name,
        stage="Starter",
        failure_subject=" on the user-supplied deck" if user_supplied else "",
    )


def run_engine(
    executable: str,
    engine_deck_path: Path,
    *,
    work_dir: Path,
    timeout_s: float,
    cancel_callback: Callable[[], bool] | None,
    job_name: str,
    user_supplied: bool,
) -> str:
    """Run Engine with one shared-memory worker per available logical CPU."""
    thread_count = max(1, os.cpu_count() or 1)
    logger.info(
        "OpenRadioss Engine: using %d SMP thread(s)",
        thread_count,
    )
    return _run_stage(
        executable=executable,
        arguments=["-nthread", str(thread_count)],
        input_path=engine_deck_path,
        work_dir=work_dir,
        timeout_s=timeout_s,
        cancel_callback=cancel_callback,
        job_name=job_name,
        stage="Engine",
        failure_subject=" on the user-supplied deck" if user_supplied else "",
    )
