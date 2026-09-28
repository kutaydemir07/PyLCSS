# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Qt worker that runs an optimization without blocking the application UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
import time
from typing import Any

from PySide6 import QtCore

from .configuration import parse_optimization_setup
from .evaluator import ModelEvaluator
from .parsing import parse_boolean
from .solvers.base import BaseSolver
from .solvers.factory import get_solver

logger = logging.getLogger(__name__)


class OptimizationWorker(QtCore.QThread):
    """Execute a validated optimization problem on a Qt worker thread."""

    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        model_func: Callable[..., Mapping[str, Any]],
        setup_data: Mapping[str, Any],
        solver_settings: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model_func = model_func
        self.setup = dict(setup_data)
        self.settings = dict(solver_settings)
        self.solver: BaseSolver | None = None

    def run(self) -> None:
        try:
            setup = parse_optimization_setup(self.setup, self.settings)
            method = str(self.settings.get("method") or "SLSQP")
            self.settings["method"] = method

            evaluator = ModelEvaluator(
                self.model_func,
                setup.variables,
                setup.objectives,
                setup.constraints,
                parameters=setup.parameters,
                scaling=parse_boolean(
                    self.settings.get("scaling", True),
                    "Variable scaling",
                ),
                scaling_mode=str(self.settings.get("scaling_mode", "auto")),
                penalty_weight=float(self.settings.get("penalty_weight", 1e6)),
                objective_scale=float(self.settings.get("objective_scale", 1.0)),
                constraint_margin=float(self.settings.get("constraint_margin", 0.0)),
                feasibility_tolerance=float(
                    self.settings.get("feasibility_tol", self.settings.get("tol", 1e-6))
                ),
            )
            solver_initial = (
                evaluator.to_normalized(setup.initial_design)
                if evaluator.scaling
                else setup.initial_design
            )
            _, initial_outputs, _ = evaluator.evaluate(solver_initial)
            if not evaluator.is_valid_result(initial_outputs):
                detail = (
                    evaluator.evaluation_error(initial_outputs)
                    or "unknown evaluation error"
                )
                raise ValueError(
                    f"The system model is invalid at the initial design: {detail}"
                )

            self.solver = get_solver(method, self.settings)
            if self.isInterruptionRequested():
                self.solver.stop()

            last_emit_time = 0.0

            def on_step(
                solver_x: Any,
                cost: float,
                raw_results: Mapping[str, float],
                violation: float,
            ) -> None:
                nonlocal last_emit_time
                now = time.monotonic()
                if now - last_emit_time < 0.05:
                    return
                self.progress.emit(
                    {
                        "iteration": evaluator.evaluation_count,
                        "x": evaluator.to_physical(solver_x),
                        "cost": float(cost),
                        "raw": dict(raw_results),
                        "violation": float(violation),
                    }
                )
                last_emit_time = now

            result = self.solver.solve(
                evaluator,
                setup.initial_design,
                on_step,
            )
            result.feasibility_tolerance = evaluator.feasibility_tolerance
            self.progress.emit(
                {
                    "iteration": evaluator.evaluation_count,
                    "x": result.x,
                    "cost": result.cost,
                    "raw": {**result.objectives, **result.constraints},
                    "violation": result.max_violation,
                }
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Optimization worker failed.")
            self.error.emit(str(exc))

    def stop(self) -> None:
        """Request cancellation without terminating the QThread forcibly."""
        self.requestInterruption()
        if self.solver is not None:
            self.solver.stop()


__all__ = ["OptimizationWorker"]
