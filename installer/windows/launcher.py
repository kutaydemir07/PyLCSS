# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Small Windows launcher for installed and development PyLCSS applications.

The launcher is frozen as ``PyLCSS.exe`` at the application root. In a source
checkout it uses the repository's ``.venv`` and live Python sources. In an
installed release it uses the isolated runtime provisioned beside the
application.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys


CREATE_NEW_PROCESS_GROUP = 0x00000200


def _show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, "PyLCSS could not start", 0x10)


def _installation_root() -> Path:
    executable = Path(sys.executable).resolve()
    return executable.parent


def _runtime_layout(root: Path) -> tuple[Path, Path, bool]:
    """Return the app directory, Python executable, and development-mode flag."""
    development_python = root / ".venv" / "Scripts" / "pythonw.exe"
    if development_python.is_file() and (root / "pylcss" / "main.py").is_file():
        return root, development_python, True
    return root / "app", root / "runtime" / "python" / "pythonw.exe", False


def main() -> int:
    root = _installation_root()
    app_dir, pythonw, development_mode = _runtime_layout(root)

    if not pythonw.is_file():
        _show_error(
            "No PyLCSS Python runtime was found.\n\n"
            "For development, create the repository .venv and install "
            "requirements.\nFor an installed copy, repair or reinstall PyLCSS."
        )
        return 2
    if not (app_dir / "pylcss" / "main.py").is_file():
        _show_error(
            "The PyLCSS application files are missing.\n\n"
            "Repair or reinstall PyLCSS, then try again."
        )
        return 3

    environment = os.environ.copy()
    current_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(app_dir) if not current_path else f"{app_dir}{os.pathsep}{current_path}"
    )
    environment["PYTHONNOUSERSITE"] = "1"
    if development_mode:
        environment["PYLCSS_PROJECT_ROOT"] = str(root)
    else:
        environment["PYLCSS_INSTALL_ROOT"] = str(root)

    try:
        subprocess.Popen(
            [str(pythonw), "-m", "pylcss.main", *sys.argv[1:]],
            cwd=str(app_dir),
            env=environment,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except OSError as exc:
        _show_error(f"Windows could not launch PyLCSS.\n\n{exc}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
