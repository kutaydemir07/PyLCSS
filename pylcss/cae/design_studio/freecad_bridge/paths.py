# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Locate FreeCAD through explicit paths, solver config, environment, and PATH."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

from pylcss.cae.solver_backends.execution import resolve_executable

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_USER_DATA_ROOT = _REPO_ROOT / "data_freecad"


# Default install locations probed in addition to env / config.  Match the
# Windows installer's "FreeCAD 1.x" suffix family because users who install
# from the .exe wizard instead of the portable 7z end up there.
_WIN_INSTALL_CANDIDATES = [
    r"C:\Program Files\FreeCAD 1.1\bin\FreeCAD.exe",
    r"C:\Program Files\FreeCAD 1.0\bin\FreeCAD.exe",
    r"C:\Program Files\FreeCAD\bin\FreeCAD.exe",
]
_LINUX_INSTALL_CANDIDATES = [
    "/usr/bin/freecad",
    "/usr/local/bin/freecad",
    "/opt/freecad/bin/FreeCAD",
]
_MAC_INSTALL_CANDIDATES = [
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
]


def _platform_candidates() -> list[str]:
    sysname = platform.system()
    if sysname == "Windows":
        return _WIN_INSTALL_CANDIDATES + ["FreeCAD.exe"]
    if sysname == "Linux":
        return _LINUX_INSTALL_CANDIDATES + ["freecad", "FreeCAD"]
    if sysname == "Darwin":
        return _MAC_INSTALL_CANDIDATES + ["freecad"]
    return ["freecad"]


def find_freecad_executable(explicit: Optional[str] = None) -> Optional[str]:
    """Return the path to FreeCAD's main GUI executable, or ``None`` if not
    installed.

    Lookup order:
      1. ``explicit`` argument (e.g. a value pulled off a node property)
      2. ``solver_paths.json`` written by ``scripts/install_solvers.py``
      3. ``PYLCSS_FREECAD_EXE`` env var
      4. Platform-default install paths + ``PATH``
    """
    return resolve_executable(
        explicit=explicit,
        env_vars=("PYLCSS_FREECAD_EXE",),
        candidates=_platform_candidates(),
    )


def find_freecad_cmd(explicit: Optional[str] = None) -> Optional[str]:
    """Find the console executable, including beside a configured GUI executable."""
    gui = find_freecad_executable()
    companion_names = (
        ["FreeCADCmd.exe"]
        if platform.system() == "Windows"
        else ["FreeCADCmd", "freecadcmd"]
    )
    companions = (
        [str(Path(gui).parent / name) for name in companion_names] if gui else []
    )
    return resolve_executable(
        explicit=explicit,
        env_vars=("PYLCSS_FREECAD_CMD",),
        candidates=[
            *companions,
            r"C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe",
            r"C:\Program Files\FreeCAD 1.0\bin\FreeCADCmd.exe",
            *companion_names,
        ],
    )


def freecad_console_command() -> Optional[list[str]]:
    """Return a console invocation, adding the GUI's console flag when needed."""
    cmd = find_freecad_cmd()
    if cmd:
        return [cmd]
    gui = find_freecad_executable()
    return [gui, "--console"] if gui else None


def freecad_data_dir(create: bool = True) -> Path:
    """Directory where node-owned ``.FCStd`` files live.

    Path is stable across sessions so node IDs map deterministically to files
    (``data_freecad/<safe_node_id>.FCStd``).  Created lazily so importing this
    module on a clean checkout is side-effect-free.
    """
    if create:
        _USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _USER_DATA_ROOT
