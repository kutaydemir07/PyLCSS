# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""FreeCAD-backed parametric part node.

This is the GUI/sketch counterpart to :class:`CadQueryCodeNode`: instead of
authoring geometry in a Python snippet inside PyLCSS, the user opens a real
FreeCAD window (subprocess), sketches, adds PartDesign features, defines
named selections / FEM loads, saves, and the saved geometry round-trips
back into the PyLCSS node graph through a sibling ``.brep`` + sidecar
``.fcmeta.json``.

Integration scope
-----------------
- `run()` returns the consolidated ``cadquery.Shape`` read from the BREP, so
  downstream PyLCSS nodes (assemblies, FEA mesh / constraint / load, export)
  see a normal CadQuery shape.  A document that has not been authored yet is
  reported as pending, so it and connected selection nodes can remain on the
  canvas without turning the expected double-click workflow into an error.
- `open_in_freecad()` launches the subprocess on the node-owned .FCStd.  UI
  code (cad_widget context menu / double-click) wires the user gesture to
  this method; the node itself is GUI-toolkit-free for headless tests.

Parameters surface
------------------
The FreeCAD startup macro reads any Spreadsheet aliases in the document and
writes them into the sidecar.  When this node sees those, it auto-creates
matching `param_<i>_name` / `param_<i>_value` properties.  PyLCSS-side changes
are pushed through FreeCADCmd, while changes saved from the FreeCAD GUI are
pulled back without overwriting them with stale graph values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pylcss.design_studio.core.base_node import CadQueryNode, resolve_numeric_input

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
            # Materialise the slots up front.  FreeCAD aliases discovered on
            # save replace these blank names, but visible generic sockets let
            # users wire Parameters before or while authoring the document.
            self.create_property(f"param_{idx}_name", "", widget_type="text")
            self.create_property(f"param_{idx}_value", 0.0, widget_type="float")
        self.add_output("shape", color=(100, 255, 100))

        # ``fcstd_filename`` is set lazily on first open: deriving it from the
        # node id keeps it stable across sessions, but we don't have the id at
        # __init__ in some NodeGraphQt versions, so resolve on demand.
        self.create_property("fcstd_filename", "", widget_type="text")
        self.create_property("auto_open_on_double_click", True, widget_type="checkbox")

        # Cached read-back state -- not pickled into the graph save file.
        self._last_imported = None
        self.set_pending(
            "Double-click this node to create or edit the part in FreeCAD, "
            "then save the document."
        )

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
        from pylcss.design_studio.freecad_bridge.paths import freecad_data_dir

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
        else:
            # Saved graph data is untrusted input.  Keep the file node-owned:
            # no absolute paths or ``..`` segments may escape data_freecad.
            stem = Path(str(name)).stem
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
            safe_stem = safe_stem.strip("_") or "FreeCAD_Part"
            safe_name = f"{safe_stem}.FCStd"
            if str(name) != safe_name:
                name = safe_name
                self.set_property("fcstd_filename", name)
        return freecad_data_dir() / name

    def open_in_freecad(self, parent_qobject: Optional[Any] = None) -> bool:
        """Spawn the FreeCAD GUI on this node's .FCStd.

        Returns True on a clean spawn.  ``parent_qobject`` is forwarded as
        the parent for the :class:`FreeCadLauncher` so signals get cleaned
        up when the host widget is destroyed.
        """
        from pylcss.design_studio.freecad_bridge.launcher import FreeCadLauncher

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

        Returns ``None`` with an actionable pending state when the document is
        still empty.  Broken exports and parameter updates remain real errors.
        """
        from pylcss.design_studio.freecad_bridge.brep_reader import read_brep_from_fcstd
        from pylcss.design_studio.freecad_bridge.param_writer import (
            write_parameters_to_fcstd,
        )

        target = self.fcstd_path()

        imported = read_brep_from_fcstd(target)
        if imported is None or (
            imported.shape is None
            and imported.brep_path is None
            and not imported.per_shape_metadata
        ):
            self.set_pending(
                f"Double-click {self.name()} to create the part in FreeCAD, "
                "then save the document."
            )
            return None
        if imported.shape is None:
            self.set_error(
                f"The FreeCAD export for {target.name} contains geometry metadata "
                "but no readable BREP. Open the node in FreeCAD and save again."
            )
            return None

        disk = collect_param_values_from_mapping(imported.parameters)
        try:
            current, has_connected_parameter = self._collect_parameter_values()
        except ValueError as exc:
            self.set_error(exc)
            return None
        if current and hasattr(imported, "sidecar") and not imported.sidecar:
            self.set_error(
                "FreeCAD parameter metadata is missing; refusing to apply node values "
                "to unverifiable spreadsheet aliases. Open FreeCAD and save again."
            )
            return None
        last = getattr(self, "_last_applied_params", None)
        forced = bool(getattr(self, "_parameter_override_pending", False))

        # First read establishes FreeCAD as the source of truth unless a
        # connected Parameter or an explicit runtime/API override owns the
        # value.  This makes the visible sockets real inputs, including on a
        # freshly reloaded graph, without needlessly rewriting FreeCAD when a
        # connected value already matches the document.
        if last is None:
            if disk and not forced and not has_connected_parameter:
                self._sync_parameter_properties(disk)
                try:
                    current, has_connected_parameter = (
                        self._collect_parameter_values()
                    )
                except ValueError as exc:
                    self.set_error(exc)
                    return None
            last = dict(
                disk if forced or has_connected_parameter else current
            )

        local_changed = current != last
        disk_changed = disk != last
        if local_changed and disk_changed and current != disk and not forced:
            self.set_error(
                "FreeCAD parameters changed both in PyLCSS and on disk. "
                "Run once after reloading the FreeCAD save, then reapply the PyLCSS edit."
            )
            return None

        if current and (forced or local_changed):
            if not write_parameters_to_fcstd(target, current):
                self.set_error(
                    "FreeCAD parameter update failed; stale geometry was not used. "
                    "Check FreeCADCmd and the spreadsheet aliases."
                )
                return None
            imported = read_brep_from_fcstd(target)
            if imported is None or imported.shape is None:
                self.set_error("FreeCAD saved the parameters but did not export a readable BREP.")
                return None
            written = collect_param_values_from_mapping(imported.parameters)
            mismatched = [
                key for key, value in current.items()
                if key not in written or abs(written[key] - value) > 1e-9 * max(1.0, abs(value))
            ]
            if mismatched:
                self.set_error(
                    "FreeCAD did not confirm updated spreadsheet aliases: "
                    + ", ".join(mismatched)
                )
                return None
            disk = written
        elif disk_changed and current == last:
            # The FreeCAD GUI was edited since our previous graph execution.
            self._sync_parameter_properties(disk)

        self._last_imported = imported
        self._last_applied_params = dict(disk or current)
        self._parameter_override_pending = False
        self.clear_pending()
        self.clear_error()
        return imported.shape

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_parameter_values(self) -> tuple[dict[str, float], bool]:
        """Resolve named slots from connections, falling back to properties."""
        import math

        values: dict[str, float] = {}
        has_connected_parameter = False
        for slot in range(1, self.MAX_PARAMS + 1):
            name = str(
                self.get_property(f"param_{slot}_name") or ""
            ).strip()
            if not name:
                continue
            if name in values:
                raise ValueError(f"Duplicate FreeCAD parameter alias: {name!r}.")

            port = self.get_input(f"param_{slot}")
            connected = bool(port and port.connected_ports())
            fallback = self.get_property(f"param_{slot}_value")
            raw_value = resolve_numeric_input(port, fallback)
            if raw_value is None:
                raise ValueError(
                    f"Connected FreeCAD parameter {name!r} did not produce a number."
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"FreeCAD parameter {name!r} must be numeric."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"FreeCAD parameter {name!r} must be finite."
                )
            values[name] = value
            has_connected_parameter = has_connected_parameter or connected
        return values, has_connected_parameter

    def _sync_parameter_properties(self, fc_params: dict) -> None:
        """Mirror FreeCAD spreadsheet aliases onto this node's parameter
        property slots so PyLCSS's existing parametric machinery can edit
        them through the normal property panel.

        Existing slots are updated as well: this method is only called after
        conflict resolution has established that the FreeCAD save should win.
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
                else:
                    self.set_property(val_prop, float(value))
            except Exception:
                logger.debug("Param slot %d sync failed", slot, exc_info=True)


def collect_param_values_from_mapping(values: dict) -> dict[str, float]:
    """Normalize finite sidecar parameters for comparison and verification."""
    import math

    result: dict[str, float] = {}
    for name, raw in dict(values or {}).items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if str(name).strip() and math.isfinite(value):
            result[str(name).strip()] = value
    return result
