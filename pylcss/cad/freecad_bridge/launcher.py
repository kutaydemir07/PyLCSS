# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Launch FreeCAD as a subprocess from inside PyLCSS.

In-process embedding is intentionally out of scope: FreeCAD owns its own
QApplication (still PySide2 in 1.x), PyLCSS owns its own (PySide6), and
the two cannot coexist in one Python process.  ``QProcess`` keeps us
honest -- the user gets the full real FreeCAD UI, side-by-side with
PyLCSS.

What this launcher adds on top of plain ``QProcess.startDetached``:

  - **Auto-locates the FreeCAD executable** via
    :func:`pylcss.cad.freecad_bridge.paths.find_freecad_executable`.
  - **Injects a start-up macro** that registers a post-save hook in
    FreeCAD so every save inside the GUI also writes a ``<file>.brep``
    and a ``<file>.fcmeta.json`` sidecar PyLCSS reads back.
  - **Manages one-process-per-node** so opening the same node twice
    re-focuses the already-running FreeCAD instance instead of spawning
    a second one (which would race on the .FCStd file).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from pylcss.cad.freecad_bridge.mod_installer import install_pylcss_mod
from pylcss.cad.freecad_bridge.paths import (
    find_freecad_cmd,
    find_freecad_executable,
    freecad_data_dir,
)

logger = logging.getLogger(__name__)


# Single per-process registry of running FreeCAD instances, keyed by the
# absolute .FCStd path. Prevents the "user clicks twice and we open two
# FreeCAD windows fighting over the same file" failure mode.
_RUNNING: Dict[str, QProcess] = {}


class FreeCadLauncher(QObject):
    """Spawn / re-focus a FreeCAD subprocess for a specific ``.FCStd`` file.

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

    def fcstd_path_for(self, node_id: str) -> Path:
        """Return the canonical .FCStd path for a PyLCSS node.

        Stable across sessions -- the file lives under ``data_freecad/``
        and is keyed by a sanitised node id so the AI assistant and the
        viewer can both find it without consulting node state.
        """
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in node_id)
        return freecad_data_dir() / f"{safe_id}.FCStd"

    def open(self, fcstd_path: Path | str, startup_macro: Optional[Path | str] = None) -> bool:
        """Launch FreeCAD on ``fcstd_path``.  Returns True on a clean spawn.

        If the path doesn't exist FreeCAD creates a new empty document
        with that name on its first save.  If FreeCAD is already running
        on this file (from a previous click) we re-emit ``process_started``
        without spawning a second instance.

        The auto-export observer is installed into FreeCAD's user
        Mod/PyLCSS/InitGui.py directory the first time we run, NOT passed
        as a CLI argument.  FreeCAD 1.x doesn't auto-execute .FCMacro
        positional args reliably, so the Mod path is the canonical hook.
        ``startup_macro`` is therefore ignored unless explicitly given.
        """
        # Make sure the Mod-based observer is installed before we spawn
        # FreeCAD -- safe to call every time, the installer only rewrites
        # when contents change.
        install_pylcss_mod()
        if not self.is_available():
            msg = ("FreeCAD executable not found. Run "
                   "`python scripts/install_solvers.py --only freecad`, "
                   "or set PYLCSS_FREECAD_EXE.")
            logger.error(msg)
            self.error_occurred.emit(str(fcstd_path), msg)
            return False

        fcstd_path = Path(fcstd_path).resolve()
        key = str(fcstd_path)

        existing = _RUNNING.get(key)
        if existing is not None and existing.state() != QProcess.NotRunning:
            logger.info("FreeCAD already running on %s; re-emitting started signal.", key)
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
        args: list[str] = [str(fcstd_path)]
        if startup_macro:
            # Power-user override: still pass an explicit macro path if a
            # caller forced one. FreeCAD's behaviour here is unreliable
            # (see open() docstring) but we honour it for advanced uses.
            args.insert(0, str(Path(startup_macro).resolve()))
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
    """Create a valid empty ``.FCStd`` at ``target`` so the GUI can open it.

    FreeCAD's GUI refuses to open a path that doesn't exist (it logs
    ``File '...' does not exist!``).  We can't fabricate a .FCStd by
    hand (it's a ZIP archive with a specific internal schema), so we
    shell out to FreeCADCmd's headless mode for ~1-2 s to make one.

    Returns True on success.  False means FreeCADCmd wasn't available or
    the headless run failed; the GUI launcher then opens FreeCAD with
    no file and the user has to Save As to the right path manually.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = find_freecad_cmd()
    if not cmd:
        return False

    # Write a one-shot macro that creates + saves an empty doc, then
    # exits.  We pass it to FreeCADCmd as a positional argument because
    # the ``-c`` flag is not a stable Python entry on all FreeCAD builds.
    script = (
        "import FreeCAD\n"
        "import sys\n"
        f"doc = FreeCAD.newDocument({_py_str(target.stem)})\n"
        f"doc.saveAs({_py_str(str(target))})\n"
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
            [cmd, script_handle],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        if proc.returncode != 0 or not target.exists():
            logger.warning(
                "FreeCADCmd seed failed (rc=%s): %s",
                proc.returncode, (proc.stderr or proc.stdout or "").strip()[:300],
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


def _py_str(s: str) -> str:
    """Embed a Python string literal safely. ``repr`` already escapes
    backslashes, so a Windows path round-trips unchanged."""
    return repr(s)


def _default_startup_macro() -> Optional[Path]:
    """Return the absolute path to the bundled auto-export macro, or None
    if it has been removed (degraded mode: FreeCAD opens without the BREP
    + sidecar hook -- saves still work, PyLCSS just won't get notified)."""
    here = Path(__file__).resolve().parent
    candidate = here / "macros" / "pylcss_autoexport.FCMacro"
    return candidate if candidate.is_file() else None


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
