# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Locate the FreeCAD executable + companion tools.

PyLCSS treats FreeCAD the same way it treats CalculiX and OpenRadioss:
``scripts/install_solvers.py --only freecad`` downloads the GitHub
release archive, unpacks it under ``external_solvers/freecad/unpacked``,
and writes the resolved paths into ``external_solvers/solver_paths.json``.

These helpers re-use the solver_backends path-resolution machinery so the
same lookup order (explicit > solver_paths.json > env var > PATH) applies.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Optional

from pylcss.solver_backends.common import resolve_executable

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_INSTALL_ROOT = _REPO_ROOT / "external_solvers" / "freecad" / "unpacked"
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
    """Path to the ``FreeCADCmd`` console runner used for headless macro
    execution (e.g. exporting BREP from a saved .FCStd).  Falls back to
    the main GUI executable with ``--console`` when no separate cmd binary
    is present (the AppImage / macOS app).
    """
    cmd = resolve_executable(
        explicit=explicit,
        env_vars=("PYLCSS_FREECAD_CMD",),
        candidates=[
            r"C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe",
            r"C:\Program Files\FreeCAD 1.0\bin\FreeCADCmd.exe",
            "freecadcmd", "FreeCADCmd",
        ],
    )
    if cmd:
        return cmd
    # No dedicated console runner -- caller can pass --console to the GUI exe.
    return find_freecad_executable(explicit=None)


def find_freecad_python(explicit: Optional[str] = None) -> Optional[str]:
    """Path to FreeCAD's bundled Python 3.11 interpreter.

    Critical for headless ``.FCStd`` parsing: FreeCAD's Python modules
    (``FreeCAD``, ``Part``, ``Sketcher``) can only be imported from that
    bundled interpreter without conda gymnastics.
    """
    return resolve_executable(
        explicit=explicit,
        env_vars=("PYLCSS_FREECAD_PYTHON",),
        candidates=[
            r"C:\Program Files\FreeCAD 1.1\bin\python.exe",
            r"C:\Program Files\FreeCAD 1.0\bin\python.exe",
        ],
    )


def is_freecad_installed() -> bool:
    """Cheap probe: returns True iff the GUI exe resolves to something on disk."""
    return find_freecad_executable() is not None


def freecad_data_dir(create: bool = True) -> Path:
    """Directory where node-owned ``.FCStd`` files live.

    Path is stable across sessions so node IDs map deterministically to files
    (``data_freecad/<safe_node_id>.FCStd``).  Created lazily so importing this
    module on a clean checkout is side-effect-free.
    """
    if create:
        _USER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _USER_DATA_ROOT
