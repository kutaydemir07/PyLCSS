# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

import logging
import time

from PySide6 import QtCore

from ...solution_space.models import MMSSParameters, MultiModalResult
from ...solution_space.multimodal import MultiModalSolutionSpaceSolver
from ...solution_space.product_family import compute_product_family_solutions
from ...solution_space.resampling import resample_multimodal_solution_spaces

logger = logging.getLogger(__name__)


class MultiModalResampleWorker(QtCore.QThread):
    """Resample visible multi-modal boxes without blocking plot controls."""

    result_signal = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def __init__(
        self,
        problem,
        bounds,
        dsl,
        dsu,
        req_u,
        req_l,
        sample_size,
        *,
        active_plots=None,
        center_slice=False,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = [box_bounds.copy() for box_bounds in bounds]
        self.dsl = dsl.copy()
        self.dsu = dsu.copy()
        self.req_u = req_u.copy()
        self.req_l = req_l.copy()
        self.sample_size = int(sample_size)
        self.active_plots = list(active_plots or [])
        self.center_slice = bool(center_slice)

    def run(self) -> None:
        try:
            samples = resample_multimodal_solution_spaces(
                self.problem,
                self.bounds,
                self.dsl,
                self.dsu,
                self.req_u,
                self.req_l,
                None,
                self.sample_size,
                active_plots=self.active_plots,
                center_slice=self.center_slice,
            )
        except Exception as exc:
            logger.exception("Multi-modal resampling failed")
            self.error_signal.emit(str(exc))
        else:
            self.result_signal.emit(samples)


class MultiModalSolverWorker(QtCore.QThread):
    """Run Multi-Modal discovery and box refinement off the UI thread."""

    progress_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def __init__(
        self,
        problem,
        dsl,
        dsu,
        reqL,
        reqU,
        parameters=None,
        params=None,
        weight=None,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.dsl = dsl
        self.dsu = dsu
        self.reqL = reqL
        self.reqU = reqU
        self.parameters = parameters
        self.params = params if params is not None else MMSSParameters()
        self.weight = weight
        self._stop_requested = False
        self.result: MultiModalResult | None = None
        self.result_mutex = QtCore.QMutex()
        self.solver = None

    def stop(self) -> None:
        self._stop_requested = True
        if self.solver is not None:
            self.solver.stop()

    def run(self) -> None:
        try:
            self.solver = MultiModalSolutionSpaceSolver(
                problem=self.problem,
                dsl=self.dsl,
                dsu=self.dsu,
                reqL=self.reqL,
                reqU=self.reqU,
                parameters=self.parameters,
                params=self.params,
                weight=self.weight,
            )

            def progress_callback(_box, _samples, message):
                if self._stop_requested:
                    self.solver.stop()
                self.progress_signal.emit(message)

            result = self.solver.solve(
                callback=progress_callback,
                stop_callback=lambda: self._stop_requested,
            )
            if self._stop_requested:
                return

            self.result_mutex.lock()
            try:
                self.result = result
            finally:
                self.result_mutex.unlock()
            self.finished_signal.emit(result)
        except Exception as exc:
            logger.exception("Multi-modal solution-space solve failed")
            if not self._stop_requested:
                self.error_signal.emit(str(exc))


class SolverWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(
        str
    )  # Only msg to prevent GUI freezing from large data
    finished_signal = QtCore.Signal(object, float, object)  # box, time, samples
    error_signal = QtCore.Signal(str)

    def __init__(self, solver) -> None:
        super().__init__()
        self.solver = solver
        self._stop_requested = False
        # Add mutex to protect shared data access
        self.result_mutex = QtCore.QMutex()

    def stop(self) -> None:
        self._stop_requested = True
        if hasattr(self.solver, "stop"):
            self.solver.stop()

    def run(self) -> None:
        try:
            start_time = time.time()
            # Pass a callback that emits the signal and checks for stop
            result = self.solver.solve(
                callback=self.emit_progress, stop_callback=lambda: self._stop_requested
            )
            if not self._stop_requested:
                if not result[1]:
                    raise RuntimeError(
                        "No solution space reached the requested reliability target"
                    )
                elapsed_time = time.time() - start_time
                # Protect the write of heavy data with mutex
                self.result_mutex.lock()
                try:
                    self.solver.latest_results = result[3]
                finally:
                    self.result_mutex.unlock()

                # Emit signal with None for samples, UI should read from solver.latest_results
                self.finished_signal.emit(
                    result[0], elapsed_time, None
                )  # box, time, samples=None
        except Exception as exc:
            logger.exception("Solution-space solve failed")
            if not self._stop_requested:
                self.error_signal.emit(str(exc))

    def emit_progress(self, _dvbox, _samples, msg) -> None:
        if self._stop_requested:
            if hasattr(self.solver, "stop"):
                self.solver.stop()
        self.progress_signal.emit(msg)


class ProductFamilyWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(
        str, int, int, str
    )  # variant_name, current_variant, total_variants, progress_msg
    finished_signal = QtCore.Signal(object)  # results dict
    error_signal = QtCore.Signal(str)

    def __init__(
        self,
        problem,
        weight,
        dsl,
        dsu,
        search_lower,
        search_upper,
        reqU,
        reqL,
        parameters,
        solver_type,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.weight = weight
        self.dsl = dsl
        self.dsu = dsu
        self.search_lower = search_lower
        self.search_upper = search_upper
        self.reqU = reqU
        self.reqL = reqL
        self.parameters = parameters
        self.solver_type = solver_type
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            # Get total number of variants for progress tracking
            self.total_variants = (
                len(self.problem.requirement_sets) + 1
            )  # +1 for platform

            def progress_callback(
                variant_name, current_variant, total_variants, progress_msg=""
            ):
                if self._stop_requested:
                    return
                self.progress_signal.emit(
                    variant_name, current_variant, total_variants, progress_msg
                )

            # Run product family computation
            results = compute_product_family_solutions(
                self.problem,
                self.weight,
                self.dsl,
                self.dsu,
                self.search_lower,
                self.search_upper,
                self.reqU,
                self.reqL,
                self.parameters,
                self.solver_type,
                progress_callback=progress_callback,
                stop_callback=lambda: self._stop_requested,
            )

            if not self._stop_requested:
                # Emit final progress for platform calculation
                self.progress_signal.emit(
                    "Platform", self.total_variants, self.total_variants, "Complete"
                )
                self.finished_signal.emit(results)

        except Exception as exc:
            logger.exception("Product-family solution-space solve failed")
            if not self._stop_requested:
                self.error_signal.emit(str(exc))

    def emit_progress(self, variant_name, progress_msg="") -> None:
        """Emit progress signal with variant name and progress message."""
        if self._stop_requested:
            return
        # Emit variant progress with message (e.g., "Phase I - Iter 5")
        self.progress_signal.emit(
            variant_name, self.current_variant, self.total_variants, progress_msg
        )


__all__ = [
    "MultiModalResampleWorker",
    "MultiModalSolverWorker",
    "ProductFamilyWorker",
    "SolverWorker",
]
