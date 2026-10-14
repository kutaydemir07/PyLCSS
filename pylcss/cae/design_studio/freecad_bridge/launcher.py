# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Launch FreeCAD with an export hook and track one GUI process per document."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from pylcss.cae.design_studio.freecad_bridge.mod_installer import install_pylcss_mod
from pylcss.cae.design_studio.freecad_bridge.paths import (
    freecad_console_command,
    find_freecad_executable,
)
from pylcss.process_utils import headless_subprocess_kwargs

logger = logging.getLogger(__name__)


# Single per-process registry of running FreeCAD instances, keyed by the
# absolute .FCStd path. Prevents the "user clicks twice and we open two
# FreeCAD windows fighting over the same file" failure mode.
_RUNNING: Dict[str, QProcess] = {}


class FreeCadLauncher(QObject):
    """Spawn or reuse a tracked FreeCAD process for a specific ``.FCStd`` file.

    Signals
    -------
    process_started : str
        FCStd path of the file the launched FreeCAD opened.
    process_exited : (str, int)
        FCStd path + the exit code reported by FreeCAD.
    error_occurred : (str, str)
        FCStd path + a human-readable error message.
    """

    process_started = Signal(str)
    process_exited = Signal(str, int)
    error_occurred = Signal(str, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._exe: Optional[str] = None
        # When this launcher object is torn down (parent widget closed,
        # test process ending, ...) disconnect everything we own from
        # _RUNNING so stale QProcess.finished / errorOccurred signals
        # don't try to call lambdas whose Python scope is already gone.
        self.destroyed.connect(self._cleanup_running)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        if self._exe is None:
            self._exe = find_freecad_executable()
        return bool(self._exe)

    def open(self, fcstd_path: Path | str) -> bool:
        """Launch FreeCAD on ``fcstd_path``.  Returns True on a clean spawn.

        Missing documents are seeded through the console runner. An existing
        tracked process is reused without bringing its window to the foreground.
        The export observer is loaded by the installed FreeCAD Mod.
        """
        # Make sure the Mod-based observer is installed before we spawn
        # FreeCAD -- safe to call every time, the installer only rewrites
        # when contents change.
        install_pylcss_mod()
        if not self.is_available():
            msg = (
                "FreeCAD executable not found. Run "
                "`python scripts/install_solvers.py --only freecad`, "
                "or set PYLCSS_FREECAD_EXE."
            )
            logger.error(msg)
            self.error_occurred.emit(str(fcstd_path), msg)
            return False

        fcstd_path = Path(fcstd_path).resolve()
        key = str(fcstd_path)

        existing = _RUNNING.get(key)
        if existing is not None and existing.state() != QProcess.NotRunning:
            logger.info(
                "FreeCAD already running on %s; re-emitting started signal.", key
            )
            self.process_started.emit(key)
            return True

        # Seed an empty .FCStd if the user has never opened this node before.
        # FreeCAD refuses to open a path that doesn't exist (the GUI prints
        # "File '...' does not exist!"), so we materialise a valid empty
        # document via FreeCADCmd before launching the GUI.
        if not fcstd_path.exists():
            ok = _seed_empty_fcstd(fcstd_path)
            if not ok:
                msg = (
                    f"Could not seed an empty FreeCAD document at {fcstd_path}.\n"
                    "FreeCAD will open without a file -- use File > Save As and "
                    "save to that exact path so PyLCSS can pick the geometry up."
                )
                logger.warning(msg)
                self.error_occurred.emit(key, msg)

        proc = QProcess(self)
        proc.setProgram(self._exe)

        # Only the .FCStd as a positional arg now -- the observer is
        # auto-loaded by the Mod/PyLCSS/InitGui.py we installed above.
        args: list[str] = [str(fcstd_path)] if fcstd_path.exists() else []
        proc.setArguments(args)

        proc.finished.connect(lambda code, _status, k=key: self._on_finished(k, code))
        proc.errorOccurred.connect(
            lambda err, k=key: self.error_occurred.emit(k, _qprocess_error_text(err))
        )

        cmdline = self._exe + " " + " ".join(shlex.quote(a) for a in args)
        logger.info("Launching FreeCAD: %s", cmdline)
        proc.start()
        if not proc.waitForStarted(5_000):
            self.error_occurred.emit(key, "FreeCAD failed to start within 5s.")
            return False

        _RUNNING[key] = proc
        self.process_started.emit(key)
        return True

    def is_open(self, fcstd_path: Path | str) -> bool:
        proc = _RUNNING.get(str(Path(fcstd_path).resolve()))
        return proc is not None and proc.state() != QProcess.NotRunning

    def close(self, fcstd_path: Path | str, force: bool = False) -> None:
        """Ask the FreeCAD instance owning this file to terminate.

        Without ``force`` we send a polite terminate (Qt-translated SIGTERM
        on POSIX, WM_CLOSE on Windows); with ``force`` we kill.
        """
        key = str(Path(fcstd_path).resolve())
        proc = _RUNNING.get(key)
        if proc is None or proc.state() == QProcess.NotRunning:
            return
        if force:
            proc.kill()
        else:
            proc.terminate()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _on_finished(self, key: str, code: int) -> None:
        _RUNNING.pop(key, None)
        try:
            self.process_exited.emit(key, int(code))
        except RuntimeError:
            # Signal owner already gone -- normal during app teardown.
            pass

    def _cleanup_running(self, *_args) -> None:
        """Drop QProcess entries this launcher started. Connected to
        ``self.destroyed`` so global ``_RUNNING`` never holds dangling
        references to processes whose parent launcher is gone."""
        for key, proc in list(_RUNNING.items()):
            if proc.parent() is self:
                try:
                    proc.finished.disconnect()
                except (TypeError, RuntimeError):
                    pass
                try:
                    proc.errorOccurred.disconnect()
                except (TypeError, RuntimeError):
                    pass
                _RUNNING.pop(key, None)


def _seed_empty_fcstd(target: Path) -> bool:
    """Create a valid empty document through FreeCAD's console runner."""
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = freecad_console_command()
    if not cmd:
        return False

    # Write a one-shot macro that creates + saves an empty doc, then
    # exits.  We pass it to FreeCADCmd as a positional argument because
    # the ``-c`` flag is not a stable Python entry on all FreeCAD builds.
    script = (
        "import FreeCAD\n"
        "import sys\n"
        f"doc = FreeCAD.newDocument({target.stem!r})\n"
        f"doc.saveAs({str(target)!r})\n"
        "FreeCAD.closeDocument(doc.Name)\n"
        "sys.exit(0)\n"
    )
    script_handle = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".FCMacro", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            script_handle = fh.name
        proc = subprocess.run(
            [*cmd, script_handle],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            **headless_subprocess_kwargs(),
        )
        if proc.returncode != 0 or not target.exists():
            logger.warning(
                "FreeCADCmd seed failed (rc=%s): %s",
                proc.returncode,
                (proc.stderr or proc.stdout or "").strip()[:300],
            )
            return False
        return True
    except Exception as exc:
        logger.warning("FreeCADCmd seed exception: %s", exc)
        return False
    finally:
        if script_handle:
            try:
                os.unlink(script_handle)
            except OSError:
                pass


def _qprocess_error_text(err: QProcess.ProcessError) -> str:
    """Translate a QProcess error enum into something a user can act on."""
    mapping = {
        QProcess.FailedToStart: "FreeCAD failed to start (executable missing or not runnable).",
        QProcess.Crashed: "FreeCAD crashed.",
        QProcess.Timedout: "FreeCAD start-up timed out.",
        QProcess.ReadError: "FreeCAD stdio read error.",
        QProcess.WriteError: "FreeCAD stdio write error.",
        QProcess.UnknownError: "FreeCAD reported an unknown error.",
    }
    if err in mapping:
        return mapping[err]
    # PySide6's ProcessError enum is not directly int()-castable on all
    # versions -- use .value when available, fall back to the repr so the
    # message is at least readable rather than throwing a TypeError that
    # would explode inside the signal-lambda.
    code = getattr(err, "value", err)
    return f"QProcess error #{code}"
