# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Update FreeCAD spreadsheet aliases, recompute, and save through its console.

Names shared by multiple sheets use ``SheetName.alias``. Unique aliases remain
unqualified. Values use FreeCAD's internal units, as exported by the observer.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from pylcss.cae.design_studio.freecad_bridge.paths import freecad_console_command
from pylcss.process_utils import headless_subprocess_kwargs

logger = logging.getLogger(__name__)


# Resolve every alias before changing cells; failed requests must not be saved.
_SCRIPT = """\
import sys

try:
    import FreeCAD
except Exception:
    sys.stderr.write("PYLCSS_PARAM_WRITER: FreeCAD module missing\\n")
    sys.exit(2)

DOC_PATH = {doc_path!r}
PARAMS = {params!r}

doc = None
try:
    doc = FreeCAD.openDocument(DOC_PATH)
    sheets = [obj for obj in doc.Objects if obj.TypeId == "Spreadsheet::Sheet"]
    updates = []
    for name, new_value in PARAMS.items():
        sheet_name, separator, alias = name.rpartition(".")
        if not separator:
            alias = name
        matches = []
        for sheet in sheets:
            if separator and sheet.Name != sheet_name:
                continue
            cell = sheet.getCellFromAlias(alias)
            if cell:
                matches.append((sheet, cell))
        if len(matches) != 1:
            raise ValueError("Alias {{!r}} matched {{}} cells; use SheetName.alias for duplicates".format(name, len(matches)))
        sheet, cell = matches[0]
        previous = getattr(sheet, cell)
        expression = str(FreeCAD.Units.Quantity(new_value, previous.Unit)) if hasattr(previous, "Unit") else str(new_value)
        updates.append((sheet, cell, expression))
    for sheet, cell, expression in updates:
        sheet.set(cell, expression)
    doc.recompute()
    doc.save()
except Exception as exc:
    sys.stderr.write("PYLCSS_PARAM_WRITER: update failed: {{}}\\n".format(exc))
    sys.exit(3)
finally:
    if doc is not None:
        FreeCAD.closeDocument(doc.Name)

print("PYLCSS_PARAM_WRITER: updated", list(PARAMS))
sys.exit(0)
"""


def write_parameters_to_fcstd(
    fcstd_path: Path | str,
    params: Mapping[str, float],
    timeout_s: float = 60.0,
) -> bool:
    """Push ``params`` into the spreadsheet of ``fcstd_path`` + save.

    Returns False for invalid inputs, missing/ambiguous aliases, or execution
    failure. Callers must not use stale geometry after a failed update.
    """
    if not params:
        return True
    try:
        values = {str(k): float(v) for k, v in params.items()}
        timeout_s = float(timeout_s)
        if not all(
            name.strip() and math.isfinite(value) for name, value in values.items()
        ):
            raise ValueError(
                "Parameter names must be nonempty and values must be finite."
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(
                "Parameter update timeout must be a positive finite number."
            )
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid FreeCAD parameter update: %s", exc)
        return False
    fcstd_path = Path(fcstd_path).resolve()
    if not fcstd_path.is_file():
        logger.warning("write_parameters_to_fcstd: %s does not exist", fcstd_path)
        return False
    cmd = freecad_console_command()
    if not cmd:
        logger.warning(
            "FreeCADCmd not found -- cannot push parameters to %s. "
            "Run scripts/install_solvers.py --only freecad.",
            fcstd_path.name,
        )
        return False

    # FreeCADCmd loads Init.py (not the GUI-only InitGui.py). Ensure the Mod is
    # installed here as well as in the GUI launcher so headless optimization
    # always refreshes the sibling BREP and sidecar.
    from pylcss.cae.design_studio.freecad_bridge.mod_installer import install_pylcss_mod

    if not install_pylcss_mod():
        logger.warning("PyLCSS FreeCAD auto-export Mod could not be installed.")
        return False

    script = _SCRIPT.format(
        doc_path=str(fcstd_path),
        params=values,
    )

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".FCMacro",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(script)
            script_path = fh.name
        proc = subprocess.run(
            [*cmd, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            **headless_subprocess_kwargs(),
        )
        if proc.returncode != 0:
            logger.warning(
                "FreeCADCmd param push rc=%s for %s\n  stderr: %s\n  stdout: %s",
                proc.returncode,
                fcstd_path.name,
                (proc.stderr or "").strip()[:300],
                (proc.stdout or "").strip()[:300],
            )
            return False
        logger.debug(
            "FreeCADCmd param push ok for %s: %s",
            fcstd_path.name,
            (proc.stdout or "").strip()[:200],
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "FreeCADCmd param push timed out after %.1fs for %s",
            timeout_s,
            fcstd_path.name,
        )
        return False
    except Exception as exc:
        logger.warning("FreeCADCmd param push raised: %s", exc)
        return False
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass
