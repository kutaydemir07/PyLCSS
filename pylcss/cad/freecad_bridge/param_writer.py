# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Headless parameter push into a FreeCAD ``.FCStd`` document.

Why this exists
---------------
The :class:`FreeCadPartNode` exposes Spreadsheet aliases discovered in the
document as ``param_<i>_name`` / ``param_<i>_value`` properties so the rest
of PyLCSS (optimizer, sensitivity sweeper, AI assistant) can edit them
exactly the same way they edit CadQuery code-part parameters.

When the optimizer mutates ``param_<i>_value`` the next graph execute
needs to materialise the **new** geometry, not the old cached one.  We do
that by spawning FreeCADCmd headless, opening the .FCStd, writing the
new values into the matching Spreadsheet cells (via their aliases),
recomputing, and saving.  The PyLCSS Mod observer (installed by
:mod:`mod_installer`) fires its ``slotFinishSaveDocument`` hook and
writes a fresh ``.brep`` + ``.fcmeta.json`` next to the .FCStd; the
node's ``run()`` then re-reads them and emits the new shape downstream.

This is a one-shot subprocess per parameter change.  For sweep-style
optimization that's fine: each iteration is bounded by a single
FreeCADCmd invocation (~1-2 s on a small model).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Mapping

from pylcss.cad.freecad_bridge.paths import find_freecad_cmd

logger = logging.getLogger(__name__)


# Headless script template.  Runs inside FreeCADCmd, opens the doc, walks
# every Spreadsheet, sets each alias-mapped cell to the new value, then
# recomputes + saves.  The PyLCSS Mod observer fires on save and writes
# the BREP + sidecar JSON.
_SCRIPT = '''\
import sys
import traceback

try:
    import FreeCAD
except Exception:
    sys.stderr.write("PYLCSS_PARAM_WRITER: FreeCAD module missing\\n")
    sys.exit(2)

DOC_PATH = {doc_path!r}
PARAMS = {params!r}

try:
    doc = FreeCAD.openDocument(DOC_PATH)
except Exception as exc:
    sys.stderr.write("PYLCSS_PARAM_WRITER: openDocument failed: {{}}\\n".format(exc))
    sys.exit(3)

updated = []
unmatched = list(PARAMS.keys())

for obj in list(doc.Objects):
    if obj.TypeId != "Spreadsheet::Sheet":
        continue
    for alias, new_value in PARAMS.items():
        # FreeCAD exposes spreadsheet aliases as ordinary properties on the
        # Sheet object: ``sheet.<alias>`` is the live value.  Setting via
        # ``sheet.set("<cell>", "<expr>")`` requires knowing the cell
        # address; we look that up by scanning until ``getAlias`` matches.
        cell = None
        for column in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            for row in range(1, 200):
                addr = "{{}}{{}}".format(column, row)
                try:
                    cur_alias = obj.getAlias(addr)
                except Exception:
                    cur_alias = None
                if cur_alias == alias:
                    cell = addr
                    break
            if cell is not None:
                break
        if cell is None:
            continue
        try:
            obj.set(cell, str(new_value))
            updated.append("{{}}={{}}".format(alias, new_value))
            if alias in unmatched:
                unmatched.remove(alias)
        except Exception as exc:
            sys.stderr.write(
                "PYLCSS_PARAM_WRITER: failed to set {{}}={{}} -> {{}}\\n".format(alias, new_value, exc)
            )

try:
    doc.recompute()
    doc.save()
except Exception as exc:
    sys.stderr.write("PYLCSS_PARAM_WRITER: recompute/save failed: {{}}\\n".format(exc))
    try:
        FreeCAD.closeDocument(doc.Name)
    except Exception:
        pass
    sys.exit(4)

try:
    FreeCAD.closeDocument(doc.Name)
except Exception:
    pass

print("PYLCSS_PARAM_WRITER: updated", updated, "unmatched", unmatched)
sys.exit(0)
'''


def write_parameters_to_fcstd(
    fcstd_path: Path | str,
    params: Mapping[str, float],
    timeout_s: float = 60.0,
) -> bool:
    """Push ``params`` into the spreadsheet of ``fcstd_path`` + save.

    Returns True on a clean run (rc=0).  Logs at WARNING when FreeCADCmd
    is missing or the headless run fails; callers should treat False as
    "geometry not updated -- fall back to the last saved BREP".
    """
    if not params:
        return True
    fcstd_path = Path(fcstd_path).resolve()
    if not fcstd_path.is_file():
        logger.warning("write_parameters_to_fcstd: %s does not exist", fcstd_path)
        return False
    cmd = find_freecad_cmd()
    if not cmd:
        logger.warning(
            "FreeCADCmd not found -- cannot push parameters to %s. "
            "Run scripts/install_solvers.py --only freecad.",
            fcstd_path.name,
        )
        return False

    script = _SCRIPT.format(
        doc_path=str(fcstd_path),
        params={str(k): float(v) for k, v in params.items()},
    )

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".FCMacro", delete=False, encoding="utf-8",
        ) as fh:
            fh.write(script)
            script_path = fh.name
        proc = subprocess.run(
            [cmd, script_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            logger.warning(
                "FreeCADCmd param push rc=%s for %s\n  stderr: %s\n  stdout: %s",
                proc.returncode, fcstd_path.name,
                (proc.stderr or "").strip()[:300],
                (proc.stdout or "").strip()[:300],
            )
            return False
        logger.debug(
            "FreeCADCmd param push ok for %s: %s",
            fcstd_path.name, (proc.stdout or "").strip()[:200],
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "FreeCADCmd param push timed out after %.1fs for %s",
            timeout_s, fcstd_path.name,
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


def collect_param_values_from_node(node, max_slots: int = 8) -> Dict[str, float]:
    """Walk a :class:`FreeCadPartNode`'s ``param_<i>_*`` properties and
    return ``{alias_name: float_value}``.

    Skips empty slots, treats non-numeric ``param_<i>_value`` as missing
    -- the optimizer hands us strings sometimes and we don't want to
    push garbage into FreeCAD.
    """
    out: Dict[str, float] = {}
    for i in range(1, max_slots + 1):
        try:
            name = (node.get_property(f"param_{i}_name") or "").strip()
        except Exception:
            name = ""
        if not name:
            continue
        try:
            raw = node.get_property(f"param_{i}_value")
        except Exception:
            continue
        if raw is None or raw == "":
            continue
        try:
            out[name] = float(raw)
        except (TypeError, ValueError):
            continue
    return out
