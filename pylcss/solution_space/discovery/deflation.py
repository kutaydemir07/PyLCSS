# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

import logging
from typing import Optional

import numpy as np

from .feasibility import FeasibilityProblem
from ..contracts import FloatArray, ProgressCallback, StopCallback

logger = logging.getLogger(__name__)


class DeflationOptimizationMixin:
    def _find_feasible_regions_via_optimization(
        self,
        n_starts: int = 10,
        callback: Optional[ProgressCallback] = None,
        stop_callback: Optional[StopCallback] = None,
    ) -> Optional[FloatArray]:
        """
        Find feasible representatives using serial deflation.

        Each start uses the most up-to-date deflation centers (updated
        immediately after every successful find). Starts are processed one
        at a time.
        """
        if n_starts <= 0:
            raise ValueError("n_starts must be positive")
        solver_type = str(self.params.solver_type).lower()
        gamma = self.params.deflation_gamma
        solver_maxiter = max(1, int(self.params.discovery_solver_maxiter))

        sigma = self.params.deflation_sigma * np.sqrt(self.dim)

        logger.info(
            f"  Deflated searches from {n_starts} Latin-hypercube starts, "
            f"solver={solver_type}, gamma={gamma}, sigma={sigma:.3f} "
            f"(base {self.params.deflation_sigma}*sqrt({self.dim}))"
        )

        found_points_norm: list[FloatArray] = []
        found_points_phys: list[FloatArray] = []

        feas_prob = FeasibilityProblem(
            self.problem,
            self.parameters,
            self.ind_parameters,
            self.reqL,
            self.reqU,
            self.active_dv_norm,
            self.active_dsl,
            include_objectives=False,
        )

        def base_violation(x_norm: np.ndarray):
            """V_0(x): aggregate normalized constraint violation."""
            return feas_prob.compute_objective(x_norm)

        def deflated_objective(x_norm: np.ndarray):
            """Return the paper's deflated objective ``V_r(x)``.

            ``V_r = V_0 + gamma * sum_q exp(-||x_tilde-x_q*||^2 /
            (2*sigma_eff^2))`` in normalized coordinates, with
            ``sigma_eff = sigma_0*sqrt(n)``.
            """
            v0 = base_violation(x_norm)

            if len(found_points_norm) == 0:
                return v0

            is_2d = x_norm.ndim == 2
            x_eval = x_norm if is_2d else x_norm.reshape(1, -1)

            penalty = np.zeros(x_eval.shape[0])
            for xk in found_points_norm:
                dist_sq = np.sum((x_eval - xk) ** 2, axis=1)
                penalty += np.exp(-dist_sq / (2.0 * sigma**2))

            v_r = np.atleast_1d(v0) + gamma * penalty
            return v_r if is_2d else v_r[0]

        bounds = [(0.0, 1.0) for _ in range(self.dim)]
        start_points = self._generate_space_filling_starts(n_starts)

        _minimize = None
        _solve_ng = None

        if solver_type == "nevergrad":
            from ...optimization.solvers.backends import (
                solve_with_nevergrad as _solve_ng,
            )
        else:
            from scipy.optimize import minimize as _minimize

        failed_starts = 0
        last_exception: Exception | None = None
        completed_starts = 0
        for i, x_start in enumerate(start_points):
            if self._stop or (stop_callback and stop_callback()):
                break
            completed_starts = i + 1

            try:
                if solver_type == "nevergrad" and _solve_ng is not None:
                    ng_result = _solve_ng(
                        deflated_objective,
                        x_start,
                        bounds,
                        maxiter=solver_maxiter,
                        capture_feasible=False,
                    )
                    x_rec = np.asarray(ng_result.x).flatten()
                else:
                    result = _minimize(
                        deflated_objective,
                        x_start,
                        method="SLSQP",
                        bounds=bounds,
                        options={"maxiter": min(100, solver_maxiter), "ftol": 1e-8},
                    )
                    x_rec = np.asarray(result.x).flatten()

                x_rec = np.clip(x_rec, 0.0, 1.0)
                violation = float(np.atleast_1d(base_violation(x_rec))[0])

                if violation < 1e-6:
                    found_points_norm.append(x_rec.copy())
                    found_points_phys.append(
                        x_rec * self.active_dv_norm + self.active_dsl
                    )
                    logger.info(
                        f"    Start {i + 1}: new feasible seed (total: {len(found_points_norm)})"
                    )
            except Exception as exc:
                failed_starts += 1
                last_exception = exc
                logger.debug("Start %d failed: %s", i + 1, exc, exc_info=True)
                continue

            if callback:
                callback(
                    None,
                    None,
                    f"Phase 1 start {i + 1}/{len(start_points)}: "
                    f"{len(found_points_norm)} seeds found",
                )

        if len(found_points_phys) > 0:
            logger.info(
                f"  Deflation search discovered {len(found_points_phys)} distinct "
                f"seed candidates in {completed_starts} optimization runs"
            )
            return np.array(found_points_phys).T

        if failed_starts == completed_starts and last_exception is not None:
            raise RuntimeError(
                "Every deflation start failed during model evaluation"
            ) from last_exception
        logger.info("  Deflation search found no feasible points")
        return None
