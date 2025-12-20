# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from PySide6 import QtCore
from .computation_engine import resample_solution_space

class ResampleThread(QtCore.QThread):
    finished = QtCore.Signal(object) # samples
    error = QtCore.Signal(str)

    def __init__(self, problem, dv_par_box, dsl, dsu, reqU, reqL, parameters, sample_size, active_plots=None, dv_par_box_mutex=None):
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

    def run(self):
        try:
            # Pass self.problem directly, not evaluate_matrix
            samples = resample_solution_space(
                self.problem,
                self.dv_par_box, self.dsl, self.dsu, self.reqU, self.reqL, self.parameters, self.sample_size,
                active_plots=self.active_plots
            )
            self.finished.emit(samples)
        except Exception as e:
            self.error.emit(str(e))






