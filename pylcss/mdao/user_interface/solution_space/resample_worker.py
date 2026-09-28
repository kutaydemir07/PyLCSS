# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

import logging

from PySide6 import QtCore

from ...solution_space.resampling import resample_solution_space


logger = logging.getLogger(__name__)


class ResampleThread(QtCore.QThread):
    result_ready = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def __init__(
        self,
        problem,
        dv_par_box,
        dsl,
        dsu,
        reqU,
        reqL,
        parameters,
        sample_size,
        active_plots=None,
        dv_par_box_mutex=None,
        center_slice=False,
    ) -> None:
        super().__init__()
        self.problem = problem
        # Make a thread-safe copy of dv_par_box
        if dv_par_box_mutex:
            dv_par_box_mutex.lock()
            try:
                self.dv_par_box = dv_par_box.copy() if dv_par_box is not None else None
            finally:
                dv_par_box_mutex.unlock()
        else:
            self.dv_par_box = dv_par_box.copy() if dv_par_box is not None else None
        self.dsl = dsl
        self.dsu = dsu
        self.reqU = reqU
        self.reqL = reqL
        self.parameters = parameters
        self.sample_size = sample_size
        self.active_plots = active_plots
        self.center_slice = center_slice

    def run(self) -> None:
        try:
            if self.dv_par_box is None:
                raise ValueError("A solution-space box is required for resampling")
            samples = resample_solution_space(
                self.problem,
                self.dv_par_box,
                self.dsl,
                self.dsu,
                self.reqU,
                self.reqL,
                self.parameters,
                self.sample_size,
                active_plots=self.active_plots,
                center_slice=self.center_slice,
            )
            self.result_ready.emit(samples)
        except Exception as exc:
            logger.exception("Solution-space resampling failed")
            self.error_signal.emit(str(exc))


__all__ = ["ResampleThread"]
