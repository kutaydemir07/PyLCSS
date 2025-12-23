# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

import time
from PySide6 import QtCore
from .computation_engine import compute_product_family_solutions

class SolverWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(str)  # Only msg to prevent GUI freezing from large data
    finished_signal = QtCore.Signal(object, float, object)  # box, time, samples
    error_signal = QtCore.Signal(str)

    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self._stop_requested = False
        # Add mutex to protect shared data access
        self.result_mutex = QtCore.QMutex()

    def stop(self):
        self._stop_requested = True
        # Fallback for solvers that don't support stop_callback yet
        if hasattr(self.solver, '_stop'):
            self.solver._stop = True

    def run(self):
        try:
            start_time = time.time()
            # Pass a callback that emits the signal and checks for stop
            result = self.solver.solve(
                callback=self.emit_progress,
                stop_callback=lambda: self._stop_requested
            )
                
            if not self._stop_requested:
                elapsed_time = time.time() - start_time
                # Protect the write of heavy data with mutex
                self.result_mutex.lock()
                try:
                    self.solver.latest_results = result[3] # samples
                finally:
                    self.result_mutex.unlock()
                
                # Emit signal with None for samples, UI should read from solver.latest_results
                self.finished_signal.emit(result[0], elapsed_time, None)  # box, time, samples=None
        except Exception as e:
            if not self._stop_requested:
                self.error_signal.emit(str(e))

    def emit_progress(self, dvbox, samples, msg):
        if self._stop_requested:
            if hasattr(self.solver, '_stop'):
                self.solver._stop = True
        self.progress_signal.emit(msg)


class ProductFamilyWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(str, int, int, str)  # variant_name, current_variant, total_variants, progress_msg
    finished_signal = QtCore.Signal(object)  # results dict
    error_signal = QtCore.Signal(str)

    def __init__(self, problem, weight, dsl, dsu, l, u, reqU, reqL, parameters, slider_value, solver_type):
        super().__init__()
        self.problem = problem
        self.weight = weight
        self.dsl = dsl
        self.dsu = dsu
        self.l = l
        self.u = u
        self.reqU = reqU
        self.reqL = reqL
        self.parameters = parameters
        self.slider_value = slider_value
        self.solver_type = solver_type
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            # Get total number of variants for progress tracking
            self.total_variants = len(self.problem.requirement_sets) + 1  # +1 for platform
            
            def progress_callback(variant_name, current_variant, total_variants, progress_msg=""):
                if self._stop_requested:
                    return
                self.progress_signal.emit(variant_name, current_variant, total_variants, progress_msg)
            
            # Run product family computation
            results = compute_product_family_solutions(
                self.problem, self.weight, self.dsl, self.dsu, self.l, self.u, 
                self.reqU, self.reqL, self.parameters, self.slider_value, self.solver_type,
                progress_callback=progress_callback,
                stop_callback=lambda: self._stop_requested
            )
            
            if not self._stop_requested:
                # Emit final progress for platform calculation
                self.progress_signal.emit("Platform", self.total_variants, self.total_variants, "Complete")
                self.finished_signal.emit(results)
                
        except Exception as e:
            if not self._stop_requested:
                self.error_signal.emit(str(e))

    def emit_progress(self, variant_name, progress_msg=""):
        """Emit progress signal with variant name and progress message."""
        if self._stop_requested:
            return
        # Emit variant progress with message (e.g., "Phase I - Iter 5")
        self.progress_signal.emit(variant_name, self.current_variant, self.total_variants, progress_msg)






