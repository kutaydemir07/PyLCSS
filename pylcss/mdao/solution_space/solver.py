# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

"""High-level solver for the single-box solution-space workflow."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ..numerics import (
    run_constrained_slsqp,
    solve_with_nevergrad,
)
from .box_optimization import optimize_box
from .contracts import (
    EvaluatableProblem,
    FloatArray,
    ProgressCallback,
    SampleBatch,
    StopCallback,
)
from .coordinates import VariableLayout
from .feasibility import FeasibilityProblem
from .samples import empty_samples, merge_samples
from .sampling import evaluate_designs, sample_box_batch

logger = logging.getLogger(__name__)

SolverResult = tuple[FloatArray, int, float, SampleBatch]


class SolutionSpaceSolver:
    """Compute a single axis-aligned solution-space box."""

    def __init__(
        self,
        problem: EvaluatableProblem,
        weight: FloatArray,
        dsl: FloatArray,
        dsu: FloatArray,
        search_lower: FloatArray,
        search_upper: FloatArray,
        reqU: FloatArray,
        reqL: FloatArray,
        parameters: Optional[FloatArray],
        solver_type: str = "goal_attainment",
        include_objectives: bool = False,
    ) -> None:
        self.problem = problem
        self.original_dsl = np.asarray(dsl, dtype=float)
        self.original_dsu = np.asarray(dsu, dtype=float)
        self.original_search_lower = np.asarray(search_lower, dtype=float)
        self.original_search_upper = np.asarray(search_upper, dtype=float)
        self.reqU = np.asarray(reqU, dtype=float)
        self.reqL = np.asarray(reqL, dtype=float)
        self._validate_problem_bounds()
        self.layout = VariableLayout.from_bounds(
            self.original_dsl, self.original_dsu, parameters
        )
        self.parameters = self.layout.parameters
        self.ind_parameters = self.layout.parameter_indices
        self.solver_type = solver_type
        self.include_objectives = include_objectives

        self.original_dim = len(self.original_dsl)
        self.active_mask = self.layout.active_mask
        self.dsl = self.original_dsl[self.active_mask]
        self.dsu = self.original_dsu[self.active_mask]
        self.search_lower = self.original_search_lower[self.active_mask]
        self.search_upper = self.original_search_upper[self.active_mask]
        self.dim = len(self.dsl)

        weight = np.asarray(weight, dtype=float)
        if weight.ndim != 1 or weight.size not in {self.original_dim, self.dim}:
            raise ValueError(
                "weight must contain one value per original or active design variable"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("weight values must be finite and non-negative")
        self.weight = (
            weight[self.active_mask] if weight.size == self.original_dim else weight
        )

        self.dv_norm_l = self.dsl.copy()
        self.dv_norm = self.dsu - self.dsl

        self.search_lower_norm = (self.search_lower - self.dv_norm_l) / self.dv_norm
        self.search_upper_norm = (self.search_upper - self.dv_norm_l) / self.dv_norm

        self.initial_sample_size = max(5000, 200 * self.dim)
        self.optimization_sample_size = 200
        self.final_sample_size = 2000
        self.max_iter_phase_1 = 100
        self.max_iter_phase_2 = 100
        self.growth_rate_init = 0.05
        self.tol_phase_1 = 1e-4
        self.target_good_fraction = 0.99
        self.good_fraction_confidence = 0.95

        self._stop = False
        self.objective_indices = []
        self.objective_weights = []
        self._process_objectives()
        self._selected_feasible_point_solver()

    def _validate_problem_bounds(self) -> None:
        arrays = (
            self.original_dsl,
            self.original_dsu,
            self.original_search_lower,
            self.original_search_upper,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("design and search bounds must be one-dimensional")
        if self.original_dsl.size == 0:
            raise ValueError("at least one design variable is required")
        if any(array.shape != self.original_dsl.shape for array in arrays):
            raise ValueError("design and search bound vectors must have equal length")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("design and search bounds must contain finite values")
        if np.any(self.original_dsl > self.original_dsu):
            raise ValueError("design bounds contain an inverted interval")
        if np.any(self.original_search_lower > self.original_search_upper):
            raise ValueError("search bounds contain an inverted interval")
        if np.any(self.original_search_lower < self.original_dsl) or np.any(
            self.original_search_upper > self.original_dsu
        ):
            raise ValueError("search bounds must lie inside the design space")
        if (
            self.reqL.ndim != 1
            or self.reqU.shape != self.reqL.shape
            or self.reqL.size == 0
        ):
            raise ValueError(
                "requirement bounds must be equally sized, non-empty vectors"
            )
        if np.any(np.isnan(self.reqL)) or np.any(np.isnan(self.reqU)):
            raise ValueError("requirement bounds must not contain NaN")
        if np.any(self.reqL > self.reqU):
            raise ValueError("requirement bounds contain an inverted interval")

    def _process_objectives(self):
        for i, qoi in enumerate(getattr(self.problem, "quantities_of_interest", [])):
            if qoi.get("minimize", False) or qoi.get("maximize", False):
                weight = float(qoi.get("weight", 1.0))
                if qoi.get("maximize", False):
                    weight = -weight
                self.objective_indices.append(i)
                self.objective_weights.append(weight)

    def stop(self) -> None:
        """Request cooperative cancellation of the current solve."""
        self._stop = True

    def solve(
        self,
        callback: Optional[ProgressCallback] = None,
        stop_callback: Optional[StopCallback] = None,
    ) -> SolverResult:
        start_time = time.time()
        self._stop = False

        def should_stop() -> bool:
            return self._stop or bool(stop_callback and stop_callback())

        x0, initial_samples = self._find_feasible_point(stop_callback=should_stop)
        if x0 is None:
            return self._empty_result(start_time)

        x0_physical = self.layout.expand_points(self._denormalize(x0.reshape(-1, 1)))[
            :, 0
        ]
        phase_result = optimize_box(
            problem=self.problem,
            x0=x0_physical,
            init_bounds=None,
            dsl=self.original_dsl,
            dsu=self.original_dsu,
            reqL=self.reqL,
            reqU=self.reqU,
            parameters=self.parameters,
            ind_parameters=self.ind_parameters,
            sample_size=self.optimization_sample_size,
            growth_rate=self.growth_rate_init,
            target_good_fraction=self.target_good_fraction,
            confidence=self.good_fraction_confidence,
            phase1_max_iterations=self.max_iter_phase_1,
            phase2_max_iterations=self.max_iter_phase_2,
            phase1_convergence_tol=self.tol_phase_1,
            weight=self.weight,
            callback=callback,
            stop_callback=should_stop,
            label="Solution space",
        )

        dvbox = (
            phase_result.bounds[self.active_mask] - self.dv_norm_l[:, None]
        ) / self.dv_norm[:, None]
        return self._finalize_result(
            dvbox,
            initial_samples,
            start_time,
            extra_point=x0,
            success=phase_result.target_reached and not phase_result.cancelled,
        )

    def _find_feasible_point(self, stop_callback=None):
        if stop_callback and stop_callback():
            return None, empty_samples(self.original_dim, len(self.reqL))
        initial_box = np.column_stack((self.search_lower_norm, self.search_upper_norm))
        points, samples = sample_box_batch(
            self.problem,
            initial_box,
            self.parameters,
            self.reqL,
            self.reqU,
            self.dv_norm,
            self.dv_norm_l,
            self.ind_parameters,
            self.initial_sample_size,
        )
        good = np.flatnonzero(samples["is_good"])
        logger.info("Initial LHS found %d feasible points", good.size)
        if good.size:
            anchor = points[:, good[-1]]
            if self.dim == 0 or not (
                self.include_objectives and self.objective_indices
            ):
                return anchor, samples
            start = anchor
        else:
            if self.dim == 0:
                return None, samples
            start = points[:, -1]
        if stop_callback and stop_callback():
            return None, samples
        optimized = self._selected_feasible_point_solver()(
            start, stop_callback=stop_callback
        )
        return optimized, samples

    def _selected_feasible_point_solver(self):
        solver_type = str(self.solver_type).lower()
        if solver_type in {"goal_attainment", "slsqp"}:
            return self._solve_goal_attainment
        if solver_type == "nevergrad":
            return self._solve_with_nevergrad
        raise ValueError(
            "Unknown solver_type: "
            f"{self.solver_type}. Must be 'goal_attainment' or 'nevergrad'."
        )

    def _finalize_result(
        self,
        dvbox,
        initial_samples,
        start_time,
        extra_point=None,
        *,
        success: bool,
    ) -> SolverResult:
        _, final_samples = sample_box_batch(
            self.problem,
            dvbox,
            self.parameters,
            self.reqL,
            self.reqU,
            self.dv_norm,
            self.dv_norm_l,
            self.ind_parameters,
            self.final_sample_size,
        )
        batches = [initial_samples, final_samples]
        if extra_point is not None:
            anchor_samples, _ = evaluate_designs(
                self.problem,
                self._denormalize(extra_point.reshape(-1, 1)),
                self.parameters,
                self.ind_parameters,
                self.reqL,
                self.reqU,
            )
            batches.append(anchor_samples)
        full_box = self.layout.expand_bounds(self._denormalize_box(dvbox))
        return full_box, int(success), time.time() - start_time, merge_samples(batches)

    def _denormalize(self, dv_norm_samples):
        return dv_norm_samples * self.dv_norm.reshape(-1, 1) + self.dv_norm_l.reshape(
            -1, 1
        )

    def _denormalize_box(self, box_norm):
        return box_norm * self.dv_norm.reshape(-1, 1) + self.dv_norm_l.reshape(-1, 1)

    def _make_feasibility_problem(self):
        return FeasibilityProblem(
            self.problem,
            self.parameters,
            self.ind_parameters,
            self.reqL,
            self.reqU,
            self.dv_norm,
            self.dv_norm_l,
            self.include_objectives,
            self.objective_indices,
            self.objective_weights,
        )

    def _solver_bounds(self):
        return list(zip(self.search_lower_norm, self.search_upper_norm))

    def _solve_goal_attainment(self, x_start, stop_callback=None):
        if stop_callback and stop_callback():
            return None
        logger.info("Running SLSQP feasible-point search")
        feas_prob = self._make_feasibility_problem()
        bounds = self._solver_bounds()
        x_start = np.clip(
            np.asarray(x_start, dtype=float),
            self.search_lower_norm,
            self.search_upper_norm,
        )
        x = run_constrained_slsqp(
            feas_prob.compute_design_objective,
            feas_prob.compute_constraints_normalized,
            x_start,
            bounds,
            maxiter=500,
        )
        return x if self._is_feasible(feas_prob, x) else None

    def _solve_with_nevergrad(self, x_start, stop_callback=None):
        if stop_callback and stop_callback():
            return None
        logger.info("Running Nevergrad feasible-point search")
        feas_prob = self._make_feasibility_problem()
        bounds = self._solver_bounds()
        constraints = [
            {"type": "ineq", "fun": feas_prob.compute_constraints_finite_only}
        ]
        result = solve_with_nevergrad(
            feas_prob.compute_design_objective,
            np.asarray(x_start, dtype=float),
            bounds,
            maxiter=5000,
            constraints=constraints,
        )
        return (
            result.x
            if result.success and self._is_feasible(feas_prob, result.x)
            else None
        )

    @staticmethod
    def _is_feasible(feas_prob, x):
        if x is None:
            return False
        try:
            return bool(np.all(feas_prob.compute_constraints_raw(x) >= -1e-6))
        except Exception:
            return False

    def _empty_result(self, start_time):
        return (
            np.zeros((self.original_dim, 2)),
            0,
            time.time() - start_time,
            empty_samples(self.original_dim, len(self.reqL)),
        )


__all__ = ["SolutionSpaceSolver", "SolverResult"]
