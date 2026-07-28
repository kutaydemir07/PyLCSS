# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Multi-objective optimization and Pareto-front computation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..evaluator import ModelEvaluator
from ..models import OptimizationResult
from .base import BaseSolver, StepCallback
from .multi_start import MultiStartSolver
from .nsga2_operators import Nsga2OperatorsMixin
from .weighted_sum import WeightedSumSolver


class ParetoSolver(Nsga2OperatorsMixin, BaseSolver):
    """
    Multi-objective Pareto optimization using NSGA-II or weighted-sum scalarization.

    Capabilities:
        - NSGA-II for true multi-objective optimization
        - Weighted-sum scalarization over explicit weight sets
        - Pareto front extraction and crowding distance
        - A normalized utopia-distance compromise for display
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        super().__init__(settings)
        self._active_solver: BaseSolver | None = None

    def stop(self) -> None:
        super().stop()
        if self._active_solver is not None:
            self._active_solver.stop()

    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        If evaluator has multiple objectives, uses NSGA-II.
        Otherwise falls back to weighted-sum approach.
        """
        self._prepare_evaluator(evaluator)
        method = self.settings.get("pareto_method", "nsga2")
        n_objectives = len(evaluator.objs)

        if n_objectives <= 1 or method == "weighted_sum":
            return self._solve_weighted_sum(evaluator, x0, callback)
        else:
            return self._solve_nsga2(evaluator, x0, callback)

    def _solve_nsga2(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        """
        NSGA-II: Non-dominated Sorting Genetic Algorithm II.

        Reference: Deb et al. (2002)
        """
        # SBX crossover, polynomial mutation and the LHS seeding all operate on a
        # normalized unit hypercube, so NSGA-II needs finite box bounds and must
        # run with scaling enabled regardless of the global scaling preference.
        lowers = np.array([v.min_val for v in evaluator.vars], dtype=float)
        uppers = np.array([v.max_val for v in evaluator.vars], dtype=float)
        if not (np.all(np.isfinite(lowers)) and np.all(np.isfinite(uppers))):
            return OptimizationResult(
                x=np.asarray(x0, dtype=float),
                cost=float("inf"),
                objectives={},
                constraints={},
                max_violation=float("inf"),
                message="NSGA-II requires finite lower and upper bounds on every variable.",
                success=False,
                feasibility_tolerance=evaluator.feasibility_tolerance,
                converged=False,
            )

        original_scaling = evaluator.scaling
        evaluator.scaling = True
        try:
            return self._run_nsga2(evaluator, x0, callback)
        finally:
            evaluator.scaling = original_scaling

    def _run_nsga2(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None,
    ) -> OptimizationResult:
        """Run NSGA-II while the evaluator is in normalized coordinates."""
        pop_size = int(self.settings.get("nsga_popsize", 100))
        n_gen = int(self.settings.get("nsga_generations", 200))
        crossover_prob = float(self.settings.get("nsga_crossover_prob", 0.9))
        eta_c = float(self.settings.get("nsga_eta_c", 20.0))
        eta_m = float(self.settings.get("nsga_eta_m", 20.0))
        n_vars = len(evaluator.vars)
        n_obj = len(evaluator.objs)
        self._feasibility_tolerance = evaluator.feasibility_tolerance

        mutation_setting = self.settings.get("nsga_mutation_prob")
        mutation_prob = (
            1.0 / max(1, n_vars)
            if mutation_setting is None
            else float(mutation_setting)
        )
        if pop_size < 4 or n_gen < 1:
            raise ValueError("NSGA-II needs population size >= 4 and generations >= 1.")
        if not 0.0 <= crossover_prob <= 1.0:
            raise ValueError("NSGA-II crossover probability must be in [0, 1].")
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("NSGA-II mutation probability must be in [0, 1].")
        if eta_c <= 0.0 or eta_m <= 0.0:
            raise ValueError("NSGA-II distribution indices must be positive.")

        lowers = np.asarray(
            [variable.min_val for variable in evaluator.vars],
            dtype=float,
        )
        uppers = np.asarray(
            [variable.max_val for variable in evaluator.vars],
            dtype=float,
        )
        self._rng = np.random.default_rng(self.settings.get("seed", 42))
        fixed_variables = np.abs(uppers - lowers) <= 1e-15

        # Initialize population
        population = self._initialize_population(evaluator, x0, pop_size, n_vars)
        population[:, fixed_variables] = 0.0

        # Evaluate initial population
        pop_objectives = np.zeros((pop_size, n_obj))
        pop_violations = np.zeros(pop_size)
        evaluated_initial = 0
        for i in range(pop_size):
            if self.stop_requested:
                break
            _, results, _ = evaluator.evaluate(population[i])
            pop_objectives[i] = self._extract_objectives(results, evaluator)
            pop_violations[i] = evaluator.solve_violation(results)
            evaluated_initial += 1
        if evaluated_initial == 0:
            return OptimizationResult(
                x=np.asarray(x0, dtype=float),
                cost=float("inf"),
                objectives={},
                constraints={},
                max_violation=float("inf"),
                message="NSGA-II stopped before evaluating its initial population.",
                success=False,
                pareto_front=[],
                feasibility_tolerance=evaluator.feasibility_tolerance,
                converged=False,
            )
        if evaluated_initial < pop_size:
            population = population[:evaluated_initial]
            pop_objectives = pop_objectives[:evaluated_initial]
            pop_violations = pop_violations[:evaluated_initial]

        for gen in range(n_gen):
            if self.stop_requested:
                break

            parent_fronts = self._non_dominated_sort(pop_objectives, pop_violations)
            ranks: np.ndarray = np.full(pop_size, np.iinfo(np.int32).max, dtype=int)
            crowding: np.ndarray = np.zeros(pop_size, dtype=float)
            for rank, front in enumerate(parent_fronts):
                ranks[front] = rank
                crowding[np.asarray(front, dtype=int)] = self._crowding_distance(
                    pop_objectives[front]
                )

            # Create offspring
            offspring = self._create_offspring(
                population,
                ranks,
                crowding,
                pop_size,
                crossover_prob,
                mutation_prob,
                eta_c,
                eta_m,
            )
            offspring[:, fixed_variables] = 0.0

            # Evaluate offspring
            off_objectives = np.zeros((pop_size, n_obj))
            off_violations = np.zeros(pop_size)
            evaluated_offspring = 0
            for i in range(pop_size):
                if self.stop_requested:
                    break
                _, results, _ = evaluator.evaluate(offspring[i])
                off_objectives[i] = self._extract_objectives(results, evaluator)
                off_violations[i] = evaluator.solve_violation(results)
                evaluated_offspring += 1
            if evaluated_offspring == 0:
                break
            offspring = offspring[:evaluated_offspring]
            off_objectives = off_objectives[:evaluated_offspring]
            off_violations = off_violations[:evaluated_offspring]

            # Combined population
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([pop_objectives, off_objectives])
            combined_viol = np.concatenate([pop_violations, off_violations])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_obj, combined_viol)

            # Select next generation with crowding distance
            population, pop_objectives, pop_violations = self._select_next_gen(
                combined_pop,
                combined_obj,
                combined_viol,
                fronts,
                min(pop_size, len(combined_pop)),
            )

            # Callback. The worker's callback expects positional args
            # (x_normalized, cost, raw_results, violation), so mirror that.
            if callback and gen % 5 == 0:
                feasible = np.where(
                    pop_violations <= evaluator.feasibility_tolerance
                )[0]
                best_idx = (
                    int(feasible[np.argmin(pop_objectives[feasible, 0])])
                    if len(feasible)
                    else int(np.argmin(pop_violations))
                )
                bx = population[best_idx]
                _, braw, bviol = evaluator.evaluate(bx)
                if evaluator.is_valid_result(braw):
                    callback(
                        bx,
                        self._weighted_objective(braw, evaluator),
                        braw,
                        bviol,
                    )

        # Extract the front from the retained population. Choose a transparent
        # compromise: minimum normalized distance to the front's utopia point.
        final_fronts = self._non_dominated_sort(pop_objectives, pop_violations)
        front_indices = [
            i
            for i in (final_fronts[0] if final_fronts else [])
            if pop_violations[i] <= evaluator.feasibility_tolerance
        ]
        if front_indices:
            front_obj = pop_objectives[front_indices]
            mins = np.min(front_obj, axis=0)
            spans = np.ptp(front_obj, axis=0)
            spans[spans < 1e-15] = 1.0
            normalized_distance = np.linalg.norm((front_obj - mins) / spans, axis=1)
            compromise_local = int(np.argmin(normalized_distance))
            compromise_distance = float(normalized_distance[compromise_local])
            best_index = int(front_indices[compromise_local])
        else:
            compromise_distance = float("inf")
            feasible = np.where(
                pop_violations <= evaluator.feasibility_tolerance
            )[0]
            if len(feasible):
                best_index = int(feasible[np.argmin(pop_objectives[feasible, 0])])
            else:
                best_index = int(np.argmin(pop_violations))
        best_x = population[best_index]

        pareto_front = []
        for index in sorted(front_indices, key=lambda i: pop_objectives[i, 0]):
            _, front_raw, front_viol = evaluator.evaluate(population[index])
            if not evaluator.is_valid_result(front_raw):
                continue
            pareto_front.append(
                {
                    "x": evaluator.to_physical(population[index]).tolist(),
                    "objectives": {
                        obj.name: float(front_raw[obj.name]) for obj in evaluator.objs
                    },
                    "constraints": {
                        con.name: float(front_raw[con.name]) for con in evaluator.cons
                    },
                    "max_violation": float(front_viol),
                }
            )

        _, results, viol = evaluator.evaluate(best_x)
        x_phys = evaluator.to_physical(best_x)

        valid = evaluator.is_valid_result(results)
        objectives = (
            {obj.name: results[obj.name] for obj in evaluator.objs} if valid else {}
        )
        constraints = (
            {con.name: results[con.name] for con in evaluator.cons} if valid else {}
        )
        state = "stopped" if self.stop_requested else "completed"
        message = (
            f"NSGA-II {state} ({len(pareto_front)} non-dominated feasible "
            "solutions; displaying normalized utopia-distance compromise)"
        )
        if not valid:
            message += ": " + (
                evaluator.evaluation_error(results)
                or "selected model evaluation failed"
            )

        return OptimizationResult(
            x=x_phys,
            cost=compromise_distance if valid else float("inf"),
            objectives=objectives,
            constraints=constraints,
            max_violation=viol,
            message=message,
            success=(
                not self.stop_requested
                and valid
                and viol <= evaluator.feasibility_tolerance
                and bool(pareto_front)
            ),
            pareto_front=pareto_front,
            feasibility_tolerance=evaluator.feasibility_tolerance,
            converged=not self.stop_requested,
        )

    def _initialize_population(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        pop_size: int,
        n_vars: int,
    ) -> np.ndarray:
        """Initialize population with LHS + x0."""
        from scipy.stats.qmc import LatinHypercube

        population = np.zeros((pop_size, n_vars))
        # First individual is x0
        population[0] = evaluator.to_normalized(x0)

        # LHS for rest
        sampler = LatinHypercube(d=n_vars, seed=self.settings.get("seed", 42))
        samples = sampler.random(n=pop_size - 1)
        population[1:] = samples
        fixed = np.asarray(
            [abs(v.max_val - v.min_val) <= 1e-15 for v in evaluator.vars],
            dtype=bool,
        )
        population[:, fixed] = 0.0

        return np.clip(population, 0, 1)

    @staticmethod
    def _weighted_objective(
        raw: Mapping[str, Any],
        evaluator: ModelEvaluator,
    ) -> float:
        """Signed, weighted sum of the raw objective values (display/cost scalar)."""
        total = 0.0
        for obj in evaluator.objs:
            val = raw.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            total += sign * obj.weight * val
        return total

    def _extract_objectives(
        self,
        results: Mapping[str, Any],
        evaluator: ModelEvaluator,
    ) -> np.ndarray:
        """Extract objective values from evaluation results."""
        if not evaluator.is_valid_result(results):
            return np.full(len(evaluator.objs), np.inf, dtype=float)
        obj_values = []
        for obj in evaluator.objs:
            val = results[obj.name]
            if isinstance(val, (list, np.ndarray)):
                val = float(np.mean(val))
            if not obj.minimize:
                val = -val  # Convert max to min
            # Positive weighting is a scalarization concept, not Pareto
            # dominance. Scaling an objective by a positive constant cannot
            # add engineering preference, while a zero weight would silently
            # erase the objective from non-dominated sorting. Keep every
            # selected objective in its physical direction here; the displayed
            # compromise is normalized separately.
            obj_values.append(val)
        return np.array(obj_values)

    def _solve_weighted_sum(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        """Delegate weighted scalarization to its dedicated strategy."""
        solver = WeightedSumSolver(self.settings)
        self._active_solver = solver
        if self.stop_requested:
            solver.stop()
        try:
            return solver.solve(evaluator, x0, callback)
        finally:
            self._active_solver = None


__all__ = ["MultiStartSolver", "ParetoSolver"]
