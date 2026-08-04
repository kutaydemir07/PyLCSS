# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Executable discovery, work-directory management, and process supervision."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pylcss.process_utils import headless_subprocess_kwargs
from pylcss.solver_backends.base import SolverBackendError


logger = logging.getLogger(__name__)

_EXTERNAL_SOLVERS_ROOT = Path(__file__).resolve().parents[2] / "external_solvers"
_RUNS_ROOT = _EXTERNAL_SOLVERS_ROOT / "runs"
_DEFAULT_KEEP_RUNS = 6
_SOLVER_PATHS_CACHE: dict[str, str] | None = None


def _solver_paths_config_path() -> Path:
    """Return the executable-path config written by ``install_solvers.py``."""
    return _EXTERNAL_SOLVERS_ROOT / "solver_paths.json"


def _load_solver_paths_config() -> dict[str, str]:
    """Load configured executable paths once per Python process."""
    global _SOLVER_PATHS_CACHE
    if _SOLVER_PATHS_CACHE is not None:
        return _SOLVER_PATHS_CACHE

    try:
        raw: Any = json.loads(_solver_paths_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}

    _SOLVER_PATHS_CACHE = (
        {str(key): str(value) for key, value in raw.items() if value}
        if isinstance(raw, dict)
        else {}
    )
    return _SOLVER_PATHS_CACHE


def _keep_runs_count() -> int:
    """Return retained scratch runs per backend prefix."""
    try:
        return max(
            1,
            int(os.environ.get("PYLCSS_KEEP_SOLVER_RUNS", _DEFAULT_KEEP_RUNS)),
        )
    except (TypeError, ValueError):
        return _DEFAULT_KEEP_RUNS


def _prune_solver_runs(
    root: Path,
    prefix: str,
    keep: int,
    *,
    protected: Path | None = None,
) -> None:
    """Remove old managed scratch directories without touching explicit runs."""
    if keep <= 0:
        return
    try:
        resolved_root = root.resolve()
        protected_path = protected.resolve() if protected is not None else None
        runs = sorted(
            (
                path
                for path in root.glob(f"{prefix}*")
                if path.is_dir()
                and path.resolve().parent == resolved_root
                and path.resolve() != protected_path
            ),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return

    retained_other_runs = max(keep - (1 if protected_path is not None else 0), 0)
    for old_run in runs[: -retained_other_runs or None]:
        shutil.rmtree(old_run, ignore_errors=True)


def make_work_dir(prefix: str, requested_dir: str | None) -> Path:
    """Create a managed scratch directory or reuse an explicit directory."""
    if requested_dir:
        path = Path(
            os.path.expandvars(os.path.expanduser(str(requested_dir)))
        ).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SolverBackendError(
                f"Could not create solver work directory {path}: {exc}"
            ) from exc
        if not path.is_dir():
            raise SolverBackendError(
                f"Solver work directory is not a directory: {path}"
            )
        return path

    if (
        not prefix
        or any(char in prefix for char in ("/", "\\", "*", "?", "[", "]"))
        or prefix in {".", ".."}
    ):
        raise SolverBackendError("Solver scratch-directory prefix is invalid.")

    try:
        _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(_RUNS_ROOT))).resolve()
    except OSError as exc:
        raise SolverBackendError(
            f"Could not create solver work directory under {_RUNS_ROOT}: {exc}"
        ) from exc

    _prune_solver_runs(
        _RUNS_ROOT,
        prefix,
        _keep_runs_count(),
        protected=work_dir,
    )
    return work_dir


def resolve_executable(
    explicit: str | None,
    env_vars: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    """Resolve an executable from an override, config, environment, or PATH."""
    probes: list[str] = []
    if explicit:
        probes.append(str(explicit))

    config = _load_solver_paths_config()
    probes.extend(config[name] for name in env_vars if config.get(name))
    probes.extend(os.environ[name] for name in env_vars if os.environ.get(name))
    probes.extend(candidates)

    for probe in probes:
        expanded = os.path.expandvars(os.path.expanduser(probe))
        if Path(expanded).is_file():
            return str(Path(expanded).resolve())
        found = shutil.which(expanded)
        if found:
            return str(Path(found).resolve())
    return None


def _cancel_requested(callback: Callable[[], bool] | None) -> bool:
    if callback is None:
        return False
    try:
        return bool(callback())
    except Exception as exc:
        raise SolverBackendError(f"Solver cancellation callback failed: {exc}") from exc


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop a solver and its child processes without leaving an orphan."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
                **headless_subprocess_kwargs(),
            )
        else:
            # These members are available on POSIX even though Windows
            # typeshed intentionally omits them.
            getpgid = getattr(os, "getpgid")
            killpg = getattr(os, "killpg")
            killpg(getpgid(process.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _popen_platform_options() -> dict[str, Any]:
    if os.name == "nt":
        return headless_subprocess_kwargs(new_process_group=True)
    return {"start_new_session": True}


def _wait_for_process(
    process: subprocess.Popen[str],
    *,
    timeout_s: float,
    cancel_callback: Callable[[], bool] | None,
) -> int:
    deadline = time.monotonic() + timeout_s
    while process.poll() is None:
        try:
            cancelled = _cancel_requested(cancel_callback)
        except SolverBackendError:
            _terminate_process_tree(process)
            raise
        if cancelled:
            _terminate_process_tree(process)
            raise SolverBackendError("Solver run cancelled by the user.")
        if time.monotonic() >= deadline:
            _terminate_process_tree(process)
            raise SolverBackendError(
                f"Solver exceeded the configured timeout of {timeout_s:g} seconds."
            )
        time.sleep(0.1)
    return int(process.returncode or 0)


def _log_solver_progress(
    log_path: Path,
    stop_event: threading.Event,
) -> None:
    """Emit throttled progress records while a verbose solver writes its log."""
    emit_interval_s = 30.0
    offset = 0
    last_emit = 0.0
    latest_cycle = ""
    latest_elapsed = ""

    while not stop_event.is_set():
        try:
            with log_path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
        except OSError:
            stop_event.wait(0.5)
            continue

        for line in chunk.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("NC="):
                latest_cycle = stripped
            elif "ELAPSED TIME" in stripped:
                latest_elapsed = stripped
            elif "TERMINATION" in stripped:
                logger.info("Solver: %s", stripped)

        now = time.monotonic()
        if now - last_emit >= emit_interval_s and (latest_cycle or latest_elapsed):
            if latest_cycle:
                logger.info("Solver: %s", latest_cycle)
            if latest_elapsed:
                logger.info("Solver: %s", latest_elapsed)
            last_emit = now
        stop_event.wait(2.0)


def _child_environment(
    extra_path_dirs: Sequence[str],
    extra_env: Mapping[str, object] | None,
) -> dict[str, str] | None:
    if not extra_path_dirs and not extra_env:
        return None
    environment = os.environ.copy()
    prepend = os.pathsep.join(str(path) for path in extra_path_dirs if path)
    if prepend:
        environment["PATH"] = prepend + os.pathsep + environment.get("PATH", "")
    if extra_env:
        environment.update({str(key): str(value) for key, value in extra_env.items()})
    return environment


def run_process(
    args: Sequence[str],
    cwd: Path,
    timeout_s: float,
    extra_path_dirs: Sequence[str] = (),
    extra_env: Mapping[str, object] | None = None,
    stdout_file: Path | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external solver with cancellation, timeout, and tree cleanup."""
    command = [str(arg) for arg in args]
    if not command or not command[0].strip():
        raise SolverBackendError("Solver command is empty.")

    cwd = Path(cwd).resolve()
    if not cwd.is_dir():
        raise SolverBackendError(f"Solver work directory does not exist: {cwd}")
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError("Solver timeout must be a number.") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise SolverBackendError("Solver timeout must be finite and greater than zero.")
    if _cancel_requested(cancel_callback):
        raise SolverBackendError("Solver run cancelled by the user.")

    environment = _child_environment(extra_path_dirs, extra_env)

    if stdout_file is not None:
        log_path = Path(stdout_file).resolve()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            output_stream = log_path.open("w", encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SolverBackendError(
                f"Could not open solver log file {log_path}: {exc}"
            ) from exc

        stop_event = threading.Event()
        watcher = threading.Thread(
            target=_log_solver_progress,
            args=(log_path, stop_event),
            daemon=True,
            name="pylcss-solver-log",
        )
        watcher.start()
        try:
            with output_stream:
                try:
                    child: subprocess.Popen[str] = subprocess.Popen(
                        command,
                        cwd=str(cwd),
                        text=True,
                        stdout=output_stream,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        **_popen_platform_options(),
                    )
                except OSError as exc:
                    raise SolverBackendError(
                        f"Could not start solver executable {command[0]!r}: {exc}"
                    ) from exc
                returncode = _wait_for_process(
                    child,
                    timeout_s=timeout,
                    cancel_callback=cancel_callback,
                )
        finally:
            stop_event.set()
            watcher.join(timeout=2.0)

        try:
            stdout = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stdout = ""
        return subprocess.CompletedProcess(command, returncode, stdout=stdout)

    try:
        child = subprocess.Popen(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            **_popen_platform_options(),
        )
    except OSError as exc:
        raise SolverBackendError(
            f"Could not start solver executable {command[0]!r}: {exc}"
        ) from exc

    deadline = time.monotonic() + timeout
    while True:
        try:
            cancelled = _cancel_requested(cancel_callback)
        except SolverBackendError:
            _terminate_process_tree(child)
            child.communicate()
            raise
        if cancelled:
            _terminate_process_tree(child)
            child.communicate()
            raise SolverBackendError("Solver run cancelled by the user.")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            _terminate_process_tree(child)
            child.communicate()
            raise SolverBackendError(
                f"Solver exceeded the configured timeout of {timeout:g} seconds."
            )
        try:
            stdout, _ = child.communicate(timeout=min(0.2, remaining))
            return subprocess.CompletedProcess(
                command,
                int(child.returncode or 0),
                stdout=stdout,
            )
        except subprocess.TimeoutExpired:
            continue


def tail(text: str, limit: int = 4000) -> str:
    """Return at most the final ``limit`` characters of a solver log."""
    if limit < 0:
        raise ValueError("tail limit must be non-negative")
    return text if len(text) <= limit else text[-limit:]
