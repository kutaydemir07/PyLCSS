# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

"""High-level solver engine for the single-box solution-space workflow."""

from __future__ import annotations

import logging
import time

import numpy as np

from .compute_solution_space import compute_solution_space as compute_phase_solution_space
from .monte_carlo import monte_carlo

from ..optimization.solvers.feasibility import FeasibilityProblem
from ..optimization.solvers.legacy import (
    run_goal_attainment_slsqp,
    solve_with_nevergrad,
)

logger = logging.getLogger(__name__)


class SolutionSpaceSolver:
    """Compute a single axis-aligned solution-space box."""

    def __init__(
        self,
        problem,
        weight,
        dsl,
        dsu,
        l,
        u,
        reqU,
        reqL,
        parameters,
        solver_type="goal_attainment",
        include_objectives=False,
    ):
        self.problem = problem
        self.original_dsl = np.asarray(dsl, dtype=float)
        self.original_dsu = np.asarray(dsu, dtype=float)
        self.original_l = np.asarray(l, dtype=float)
        self.original_u = np.asarray(u, dtype=float)
        self.reqU = np.asarray(reqU, dtype=float)
        self.reqL = np.asarray(reqL, dtype=float)
        self.parameters = self._process_parameters(parameters)
        self.ind_parameters = np.where(~np.isnan(self.parameters[0, :]))[0]
        self.solver_type = solver_type
        self.include_objectives = include_objectives

        self.original_dim = len(self.original_dsl)
        self.active_mask = np.isnan(self.parameters[0, :])
        self.fixed_mask = np.isclose(self.original_dsl, self.original_dsu)

        self.dsl = self.original_dsl[self.active_mask]
        self.dsu = self.original_dsu[self.active_mask]
        self.l = self.original_l[self.active_mask]
        self.u = self.original_u[self.active_mask]
        self.dim = len(self.dsl)

        weight = np.asarray(weight, dtype=float)
        self.weight = weight[self.active_mask] if len(weight) == self.original_dim else weight

        self.dv_norm_l = self.dsl.copy()
        self.dv_norm = self.dsu - self.dsl
        self.dv_norm[self.dv_norm == 0] = 1.0

        self.l_norm = (self.l - self.dv_norm_l) / self.dv_norm
        self.u_norm = (self.u - self.dv_norm_l) / self.dv_norm
        self.dsl_norm = (self.dsl - self.dv_norm_l) / self.dv_norm
        self.dsu_norm = (self.dsu - self.dv_norm_l) / self.dv_norm

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

    def _process_parameters(self, parameters):
        original_dim = len(self.original_dsl)
        if parameters is None or np.asarray(parameters).size == 0:
            processed = np.full((2, original_dim), np.nan)
        else:
            processed = np.asarray(parameters, dtype=float).copy()
            if processed.shape != (2, original_dim):
                raise ValueError(
                    f"Parameters array must have shape (2, {original_dim}), "
                    f"got {processed.shape}"
                )

        fixed_mask = np.isclose(self.original_dsl, self.original_dsu)
        if np.any(fixed_mask):
            processed[:, fixed_mask] = np.vstack(
                (
                    self.original_dsl[fixed_mask],
                    self.original_dsu[fixed_mask],
                )
            )
        return processed

    def _process_objectives(self):
        for i, qoi in enumerate(getattr(self.problem, "quantities_of_interest", [])):
            if qoi.get("minimize", False) or qoi.get("maximize", False):
                weight = float(qoi.get("weight", 1.0))
                if qoi.get("maximize", False):
                    weight = -weight
                self.objective_indices.append(i)
                self.objective_weights.append(weight)

    def solve(self, callback=None, stop_callback=None):
        start_time = time.time()
        self._stop = False

        x0, initial_samples = self._find_feasible_point(stop_callback=stop_callback)
        if x0 is None:
            return self._empty_result(start_time)

        x0_physical = self._denormalize(x0.reshape(-1, 1)).ravel()
        phase_result = compute_phase_solution_space(
            problem=self.problem,
            x0=x0_physical,
            init_bounds=None,
            dsl=self.dsl,
            dsu=self.dsu,
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
            stop_callback=stop_callback,
            label="Solution space",
        )

        dvbox = (phase_result.bounds - self.dv_norm_l[:, None]) / self.dv_norm[:, None]
        return self._finalize_result(
            dvbox,
            initial_samples,
            start_time,
            extra_point=x0,
        )

    def _find_feasible_point(self, stop_callback=None):
        if self.dim <= 0:
            logger.warning("No active design variables available for solution-space solve")
            return None, self._empty_samples()

        init_box = np.column_stack((self.l_norm, self.u_norm))
        try:
            points_a, m, points_b, dv_sample, violation_idx, y_sample = monte_carlo(
                self.problem,
                init_box,
                self.parameters,
                self.reqL,
                self.reqU,
                self.dv_norm,
                self.dv_norm_l,
                self.ind_parameters,
                self.initial_sample_size,
                self.dim,
            )
        except Exception as exc:
            logger.error("Initial Monte Carlo sampling failed: %s", exc, exc_info=True)
            return None, self._empty_samples()

        initial_samples = self._sample_dict(
            dv_sample,
            points_a,
            points_b,
            violation_idx,
            y_sample,
        )

        logger.info("Initial Monte Carlo found %d feasible points", m)
        if m > 0:
            ind_a = np.where(points_a)[0]
            x0 = dv_sample[:, ind_a[-1]]
            if not (self.include_objectives and self.objective_indices):
                return x0, initial_samples
            x_start = x0
        else:
            x_start = dv_sample[:, -1] if dv_sample.shape[1] else (self.l_norm + self.u_norm) / 2.0

        if stop_callback and stop_callback():
            self._stop = True
            return None, initial_samples

        solver = self._selected_feasible_point_solver()
        x_optimized = solver(x_start, stop_callback=stop_callback)
        return x_optimized, initial_samples

    def _selected_feasible_point_solver(self):
        solver_type = str(self.solver_type).lower()
        if solver_type in {"goal_attainment", "slsqp", "pymoo"}:
            return self._solve_goal_attainment
        if solver_type in {"nevergrad", "ng"}:
            return self._solve_with_nevergrad
        raise ValueError(
            "Unknown solver_type: "
            f"{self.solver_type}. Must be 'goal_attainment' or 'nevergrad'."
        )

    def _finalize_result(self, dvbox, initial_samples, start_time, extra_point=None):
        points_a, _m, points_b, dv_sample, violation_idx, y_sample = monte_carlo(
            self.problem,
            dvbox,
            self.parameters,
            self.reqL,
            self.reqU,
            self.dv_norm,
            self.dv_norm_l,
            self.ind_parameters,
            self.final_sample_size,
            self.dim,
        )

        final_samples = self._sample_dict(
            dv_sample,
            points_a,
            points_b,
            violation_idx,
            y_sample,
        )
        if extra_point is not None:
            self._append_extra_point(final_samples, extra_point)

        samples = self._merge_samples(initial_samples, final_samples)
        active_box = self._denormalize_box(dvbox)
        full_box = self._full_box(active_box)

        return full_box, 1, time.time() - start_time, samples

    def _sample_dict(self, dv_sample, points_a, points_b, violation_idx, y_sample):
        points = self._full_points(self._denormalize(dv_sample))
        return {
            "points": points,
            "is_good": np.asarray(points_a, dtype=bool),
            "is_bad": np.asarray(points_b, dtype=bool),
            "violation_idx": np.asarray(violation_idx, dtype=int),
            "qoi_values": np.asarray(y_sample, dtype=float),
        }

    def _empty_samples(self):
        return {
            "points": np.zeros((self.original_dim, 0)),
            "is_good": np.array([], dtype=bool),
            "is_bad": np.array([], dtype=bool),
            "violation_idx": np.array([], dtype=int),
            "qoi_values": np.zeros((len(self.reqL), 0)),
        }

    def _merge_samples(self, first, second):
        if first is None:
            first = self._empty_samples()
        return {
            "points": np.hstack((first["points"], second["points"])),
            "is_good": np.concatenate((first["is_good"], second["is_good"])),
            "is_bad": np.concatenate((first["is_bad"], second["is_bad"])),
            "violation_idx": np.concatenate((first["violation_idx"], second["violation_idx"])),
            "qoi_values": np.hstack((first["qoi_values"], second["qoi_values"])),
        }

    def _append_extra_point(self, samples, x_norm):
        x_active = self._denormalize(x_norm.reshape(-1, 1))
        x_full = self._full_points(x_active)
        y_val = np.asarray(self.problem.evaluate_matrix(x_full), dtype=float)

        upper_slack = self.reqU.reshape(-1, 1) - y_val
        lower_slack = y_val - self.reqL.reshape(-1, 1)
        upper_slack[np.isinf(self.reqU), :] = np.inf
        lower_slack[np.isinf(self.reqL), :] = np.inf
        slack = np.vstack((upper_slack, lower_slack))
        min_slack = np.min(slack, axis=0)
        is_good = min_slack >= -1e-12
        violation_idx = np.argmin(slack, axis=0)

        samples["points"] = np.hstack((samples["points"], x_full))
        samples["is_good"] = np.append(samples["is_good"], is_good)
        samples["is_bad"] = np.append(samples["is_bad"], ~is_good)
        samples["violation_idx"] = np.append(samples["violation_idx"], violation_idx)
        samples["qoi_values"] = np.hstack((samples["qoi_values"], y_val))

    def _full_points(self, active_points):
        active_points = np.asarray(active_points, dtype=float)
        if active_points.ndim == 1:
            active_points = active_points.reshape(-1, 1)
        full = np.zeros((self.original_dim, active_points.shape[1]))
        full[self.active_mask] = active_points
        inactive = ~self.active_mask
        if np.any(inactive):
            values = self.parameters[0].copy()
            values[np.isnan(values)] = self.original_dsl[np.isnan(values)]
            full[inactive] = values[inactive].reshape(-1, 1)
        return full

    def _full_box(self, active_box):
        full = np.zeros((self.original_dim, 2))
        full[self.active_mask] = active_box
        inactive = ~self.active_mask
        if np.any(inactive):
            full[inactive, 0] = self.parameters[0, inactive]
            full[inactive, 1] = self.parameters[1, inactive]
        return full

    def _denormalize(self, dv_norm_samples):
        return dv_norm_samples * self.dv_norm.reshape(-1, 1) + self.dv_norm_l.reshape(-1, 1)

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
        return list(zip(self.l_norm, self.u_norm))

    def _solve_goal_attainment(self, x_start, stop_callback=None):
        if stop_callback and stop_callback():
            return None
        logger.info("Running SLSQP feasible-point search")
        feas_prob = self._make_feasibility_problem()
        bounds = self._solver_bounds()
        x_start = np.clip(np.asarray(x_start, dtype=float), self.l_norm, self.u_norm)
        x = run_goal_attainment_slsqp(
            feas_prob.compute_objective,
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
        constraints = [{"type": "ineq", "fun": feas_prob.compute_constraints_finite_only}]
        result = solve_with_nevergrad(
            feas_prob.compute_objective,
            np.asarray(x_start, dtype=float),
            bounds,
            maxiter=5000,
            constraints=constraints,
        )
        return result.x if result.success and self._is_feasible(feas_prob, result.x) else None

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
            self._empty_samples(),
        )
