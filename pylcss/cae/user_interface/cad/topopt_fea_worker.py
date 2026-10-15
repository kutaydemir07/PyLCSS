# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Background CalculiX analysis requested from the TopOpt inspector."""

import logging

from PySide6 import QtCore

from .cad_background import run_cad_job


class TopOptFeaWorker(QtCore.QThread):
    analysis_finished = QtCore.Signal(object)
    analysis_error = QtCore.Signal(str)
    analysis_cancelled = QtCore.Signal()
    analysis_progress = QtCore.Signal(str)

    def __init__(
        self,
        result,
        material,
        supports,
        loads,
        *,
        quality="Standard",
        mesh_options=None,
        parent=None,
    ):
        super().__init__(parent)
        self.result = dict(result)
        self.material = dict(material or {})
        self.supports = list(supports)
        self.loads = list(loads)
        self.quality = quality
        self.mesh_options = dict(mesh_options or {})

    def run(self):
        try:
            # Netgen/OCC can hold the GIL or crash natively on malformed
            # surfaces. Keep meshing and the solver out of the GUI process.
            analysis = run_cad_job(
                'fea',
                self.result,
                self.material,
                self.supports,
                self.loads,
                {'quality': self.quality, 'mesh_options': self.mesh_options},
                progress_callback=lambda payload, _density, _step, _total:
                    self.analysis_progress.emit(str(payload.get('stage', 'CAD FEA'))),
                cancel_callback=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested() or analysis is None:
                self.analysis_cancelled.emit()
            else:
                self.analysis_finished.emit(analysis)
        except Exception as exc:
            if self.isInterruptionRequested():
                self.analysis_cancelled.emit()
            else:
                logging.getLogger(__name__).exception("TopOpt FEA analysis failed")
                self.analysis_error.emit(str(exc))
