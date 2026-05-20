# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Watch ``.FCStd`` (and sibling BREP/JSON exports) for changes.

When the user saves inside the FreeCAD subprocess, our startup macro
writes three files next to the .FCStd: ``<base>.brep`` for the geometry
the PyLCSS viewer can read with OCCT, and ``<base>.fcmeta.json`` for the
sidecar that maps face/edge indices back to FreeCAD's semantic names +
any FEM analysis constraints/loads the user authored.

``FCStdWatcher`` raises a single ``saved`` signal once **all three**
files have stabilised, so consumers don't try to read a half-written
BREP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal

logger = logging.getLogger(__name__)


# How long to wait after the last mtime bump before declaring "save finished".
# FreeCAD writes the three files (FCStd, brep, sidecar) in quick succession;
# 250 ms is long enough to cover the gap, short enough to feel immediate.
_DEBOUNCE_MS = 250


class FCStdWatcher(QObject):
    """Debounced filesystem watcher for one node-owned ``.FCStd``.

    Signals
    -------
    saved : str
        Emitted with the .FCStd path once the FCStd + .brep + .fcmeta.json
        triple has been quiet for ``_DEBOUNCE_MS``.
    """

    saved = Signal(str)

    def __init__(self, fcstd_path: Path | str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._fcstd_path = Path(fcstd_path).resolve()
        self._brep_path = self._fcstd_path.with_suffix(".brep")
        self._sidecar_path = self._fcstd_path.with_suffix(".fcmeta.json")

        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watcher.directoryChanged.connect(self._on_dir_changed)

        # Always watch the directory because FreeCAD may write atomically by
        # writing to a tmp file and renaming on top of the target; in that
        # case fileChanged misses the create event but directoryChanged
        # catches it.
        self._watcher.addPath(str(self._fcstd_path.parent))
        for p in self._existing_paths():
            self._watcher.addPath(str(p))

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_saved_if_ready)

    # ------------------------------------------------------------------
    @property
    def fcstd_path(self) -> Path:
        return self._fcstd_path

    @property
    def brep_path(self) -> Path:
        return self._brep_path

    @property
    def sidecar_path(self) -> Path:
        return self._sidecar_path

    def stop(self) -> None:
        if self._watcher.files():
            self._watcher.removePaths(self._watcher.files())
        if self._watcher.directories():
            self._watcher.removePaths(self._watcher.directories())
        self._debounce.stop()

    # ------------------------------------------------------------------
    def _existing_paths(self) -> list[Path]:
        return [p for p in (self._fcstd_path, self._brep_path, self._sidecar_path) if p.exists()]

    def _on_file_changed(self, _path: str) -> None:
        # On some editors (and on Windows atomic renames) fileChanged fires
        # with the file briefly absent.  Re-add it so we keep watching.
        for p in self._existing_paths():
            if str(p) not in self._watcher.files():
                self._watcher.addPath(str(p))
        self._debounce.start()

    def _on_dir_changed(self, _path: str) -> None:
        for p in self._existing_paths():
            if str(p) not in self._watcher.files():
                self._watcher.addPath(str(p))
        self._debounce.start()

    def _emit_saved_if_ready(self) -> None:
        # The minimum we need to proceed is the .FCStd itself; .brep and
        # the sidecar are optional in the POC -- consumers degrade
        # gracefully if they're missing.
        if not self._fcstd_path.exists():
            return
        logger.debug(
            "FCStdWatcher: save detected for %s (brep=%s, sidecar=%s)",
            self._fcstd_path.name,
            self._brep_path.exists(),
            self._sidecar_path.exists(),
        )
        self.saved.emit(str(self._fcstd_path))
