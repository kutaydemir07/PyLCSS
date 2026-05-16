# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""FreeCAD-backed parametric part node.

This is the GUI/sketch counterpart to :class:`CadQueryCodeNode`: instead of
authoring geometry in a Python snippet inside PyLCSS, the user opens a real
FreeCAD window (subprocess), sketches, adds PartDesign features, defines
named selections / FEM loads, saves, and the saved geometry round-trips
back into the PyLCSS node graph through a sibling ``.brep`` + sidecar
``.fcmeta.json``.

POC scope
---------
- `run()` returns the consolidated ``cadquery.Shape`` read from the BREP, so
  downstream PyLCSS nodes (assemblies, FEA mesh / constraint / load, export)
  see a normal CadQuery shape.  Returns ``None`` (with a clear log) when the
  user hasn't saved in FreeCAD yet.
- `open_in_freecad()` launches the subprocess on the node-owned .FCStd.  UI
  code (cad_widget context menu / double-click) wires the user gesture to
  this method; the node itself is GUI-toolkit-free for headless tests.

Parameters surface
------------------
The FreeCAD startup macro reads any Spreadsheet aliases in the document and
writes them into the sidecar.  When this node sees those, it auto-creates
matching `param_<i>_name` / `param_<i>_value` properties so the existing
optimizer + sensitivity layers can drive them exactly like they drive
CadQueryCodeNode params.  Out-of-scope for the POC: pushing PyLCSS-side
parameter changes back into FreeCAD (one-way for now).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


class FreeCadPartNode(CadQueryNode):
    """CAD body authored interactively in FreeCAD."""

    __identifier__ = "com.cad.freecad_part"
    NODE_NAME = "FreeCAD Part"

    # How many synthetic parameter ports we expose before requiring the user
    # to flatten further -- matches CadQueryCodeNode for consistency.
    MAX_PARAMS = 8

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Synthetic parameter ports: optional inputs the optimizer can drive.
        for idx in range(1, self.MAX_PARAMS + 1):
            self.add_input(f"param_{idx}", color=(180, 180, 0))
        self.add_output("shape", color=(100, 255, 100))

        # ``fcstd_filename`` is set lazily on first open: deriving it from the
        # node id keeps it stable across sessions, but we don't have the id at
        # __init__ in some NodeGraphQt versions, so resolve on demand.
        self.create_property("fcstd_filename", "", widget_type="text")
        self.create_property("auto_open_on_double_click", True, widget_type="checkbox")

        # Cached read-back state -- not pickled into the graph save file.
        self._last_imported = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fcstd_path(self) -> Path:
        """Resolve the node's owned ``.FCStd`` path.

        Stored as a bare filename (no directory) inside ``fcstd_filename`` so
        the project file stays portable across machines; the real directory
        is always :func:`freecad_data_dir`.

        Filename pattern: ``<sanitised-display-name>_<short-id>.FCStd`` so
        the file is recognisable on disk (``FreeCAD_Part_a9e8da0.FCStd``)
        while still uniquely keyed to this graph node -- two nodes named
        "FreeCAD Part" don't share the same file.
        """
        from pylcss.cad.freecad_bridge.paths import freecad_data_dir

        name = self.get_property("fcstd_filename") or ""
        if not name:
            display = str(self.name() or "FreeCAD_Part")
            safe_display = "".join(c if c.isalnum() or c in "-_" else "_" for c in display)
            safe_display = safe_display.strip("_") or "FreeCAD_Part"
            # Short tail of the NodeGraphQt id to make the filename unique
            # even when several nodes share a display name.  We strip the
            # "0x" hex prefix so the filename doesn't look like a memory
            # address dump.
            raw_id = str(getattr(self, "id", "") or "")
            short_id = raw_id.replace("0x", "").lstrip("0")[-8:] or "x"
            name = f"{safe_display}_{short_id}.FCStd"
            self.set_property("fcstd_filename", name)
        return freecad_data_dir() / name

    def open_in_freecad(self, parent_qobject: Optional[Any] = None) -> bool:
        """Spawn the FreeCAD GUI on this node's .FCStd.

        Returns True on a clean spawn.  ``parent_qobject`` is forwarded as
        the parent for the :class:`FreeCadLauncher` so signals get cleaned
        up when the host widget is destroyed.
        """
        from pylcss.cad.freecad_bridge.launcher import FreeCadLauncher

        launcher = FreeCadLauncher(parent=parent_qobject)
        if not launcher.is_available():
            self.set_error(
                "FreeCAD executable not found. Run "
                "`python scripts/install_solvers.py --only freecad`."
            )
            return False
        target = self.fcstd_path()
        ok = launcher.open(target)
        if ok:
            self.clear_error()
        return ok

    # ------------------------------------------------------------------
    # CadQueryNode contract
    # ------------------------------------------------------------------
    def run(self) -> Any:
        """Read the BREP FreeCAD wrote on its last save and return a Shape.

        If our local parameter properties have drifted from the values
        baked into the .FCStd (e.g. the optimizer just bumped
        ``param_1_value``), push them headlessly into FreeCAD's
        Spreadsheet first, recompute + save -- the Mod observer then
        emits a fresh BREP + sidecar we re-read here.

        Returns ``None`` (without raising) when the user hasn't saved a
        geometry yet, matching the rest of PyLCSS's lazy-graph semantics.
        """
        from pylcss.cad.freecad_bridge.brep_reader import read_brep_from_fcstd
        from pylcss.cad.freecad_bridge.param_writer import (
            collect_param_values_from_node, write_parameters_to_fcstd,
        )

        target = self.fcstd_path()

        # Push optimizer-driven parameter changes back into FreeCAD before
        # we read.  We only push when:
        #   - the .FCStd already exists (no point pushing into a blank file)
        #   - the user has populated at least one named slot
        #   - the current slot values differ from what we last applied
        if target.is_file():
            current = collect_param_values_from_node(self, max_slots=self.MAX_PARAMS)
            last = getattr(self, "_last_applied_params", None)
            if current and current != last:
                ok = write_parameters_to_fcstd(target, current)
                if ok:
                    self._last_applied_params = dict(current)
                else:
                    logger.info(
                        "FreeCadPartNode(%s): param push failed; using last saved BREP.",
                        self.name(),
                    )

        imported = read_brep_from_fcstd(target)
        if imported is None or imported.shape is None:
            logger.debug(
                "FreeCadPartNode(%s): no BREP yet at %s -- open in FreeCAD + save.",
                self.name(), target.name,
            )
            return None

        self._last_imported = imported
        # Only seed properties from the .FCStd when nothing has been
        # pushed yet -- otherwise the optimizer's pending value would be
        # silently overwritten by the disk version on every execute.
        if not getattr(self, "_last_applied_params", None):
            self._sync_parameter_properties(imported.parameters)
        return imported.shape

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sync_parameter_properties(self, fc_params: dict) -> None:
        """Mirror FreeCAD spreadsheet aliases onto this node's parameter
        property slots so PyLCSS's existing parametric machinery can edit
        them through the normal property panel.

        Only fills empty slots; never silently overwrites a value the user
        already set.  Out-of-scope for the POC: pushing updated values back
        into the FreeCAD spreadsheet.
        """
        for slot, (name, value) in enumerate(fc_params.items(), start=1):
            if slot > self.MAX_PARAMS:
                break
            name_prop = f"param_{slot}_name"
            val_prop = f"param_{slot}_value"
            try:
                if not self.has_property(name_prop):
                    self.create_property(name_prop, name, widget_type="text")
                elif not self.get_property(name_prop):
                    self.set_property(name_prop, name)
                if not self.has_property(val_prop):
                    self.create_property(val_prop, float(value), widget_type="float")
            except Exception:
                logger.debug("Param slot %d sync failed", slot, exc_info=True)
