# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Cross-platform options for child processes that must stay headless."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def headless_subprocess_kwargs(
    *,
    new_process_group: bool = False,
) -> dict[str, Any]:
    """Return subprocess options that prevent console-window flashes.

    PyLCSS is a windowed application, but Windows console executables such as
    CalculiX, OpenRadioss, ``anim_to_vtk``, and FreeCADCmd otherwise create a
    short-lived black or white console window.  ``CREATE_NO_WINDOW`` is the
    primary control; the hidden ``STARTUPINFO`` is included for executables
    that inspect the show-window setting themselves.
    """
    if os.name != "nt":
        return {}

    creationflags = subprocess.CREATE_NO_WINDOW
    if new_process_group:
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": creationflags,
        "startupinfo": startupinfo,
    }
