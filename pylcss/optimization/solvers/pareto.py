# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Multi-objective optimization with Pareto front computation.
Implements NSGA-II, multi-start, and Pareto-based analysis.
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pylcss.optimization.solvers.base import BaseSolver
from pylcss.optimization.core import OptimizationResult

logger = logging.getLogger(__name__)


class ParetoSolver(BaseSolver):
    """
    Multi-objective Pareto optimization using NSGA-II or weighted-sum scalarization.
    
    Capabilities:
        - NSGA-II for true multi-objective optimization
        - Weighted-sum scalarization with adaptive weights
        - Pareto front extraction and crowding distance
        - Hypervolume indicator computation
        - Utopia/nadir point computation
    """

    def solve(self, evaluator, x0, callback=None):
        """
        Run multi-objective optimization.
        
        If evaluator has multiple objectives, uses NSGA-II.
        Otherwise falls back to weighted-sum approach.
        """
        method = self.settings.get("pareto_method", "nsga2")
        n_objectives = len(evaluator.objs)

        if n_objectives <= 1 or method == "weighted_sum":
            return self._solve_weighted_sum(evaluator, x0, callback)
        else:
            return self._solve_nsga2(evaluator, x0, callback)

    def _solve_nsga2(self, evaluator, x0, callback=None):
        """
        NSGA-II: Non-dominated Sorting Genetic Algorithm II.
        
        Reference: Deb et al. (2002)
        """
        pop_size = int(self.settings.get("nsga_popsize", 100))
        n_gen = int(self.settings.get("nsga_generations", 200))
        crossover_prob = float(self.settings.get("nsga_crossover_prob", 0.9))
        eta_c = float(self.settings.get("nsga_eta_c", 20.0))  # SBX distribution index
        eta_m = float(self.settings.get("nsga_eta_m", 20.0))  # polynomial mutation index

        n_vars = len(evaluator.vars)
        n_obj = len(evaluator.objs)

        # Mutation probability defaults to the standard 1/n_vars heuristic.
        mutation_prob = self.settings.get("nsga_mutation_prob", None)
        mutation_prob = float(mutation_prob) if mutation_prob else 1.0 / max(1, n_vars)

        # SBX crossover, polynomial mutation and the LHS seeding all operate on a
        # normalized unit hypercube, so NSGA-II needs finite box bounds and must
        # run with scaling enabled regardless of the global scaling preference.
        lowers = np.array([v.min_val for v in evaluator.vars], dtype=float)
        uppers = np.array([v.max_val for v in evaluator.vars], dtype=float)
        if not (np.all(np.isfinite(lowers)) and np.all(np.isfinite(uppers))):
            return OptimizationResult(
                x=np.asarray(x0, dtype=float), cost=float("inf"),
                objectives={}, constraints={}, max_violation=float("inf"),
                message="NSGA-II requires finite lower and upper bounds on every variable.",
                success=False,
            )

        original_scaling = evaluator.scaling
        evaluator.scaling = True

        # Initialize population
        population = self._initialize_population(evaluator, x0, pop_size, n_vars)

        # Evaluate initial population
        pop_objectives = np.zeros((pop_size, n_obj))
        pop_violations = np.zeros(pop_size)
        for i in range(pop_size):
            cost, results, viol = evaluator.evaluate(population[i])
            pop_objectives[i] = self._extract_objectives(results, evaluator)
            pop_violations[i] = evaluator.solve_violation(results)

        best_front = []
        start_time = time.time()

        for gen in range(n_gen):
            if self.stop_requested:
                break

            # Create offspring
            offspring = self._create_offspring(
                population, pop_size, n_vars, evaluator,
                crossover_prob, mutation_prob, eta_c, eta_m
            )

            # Evaluate offspring
            off_objectives = np.zeros((pop_size, n_obj))
            off_violations = np.zeros(pop_size)
            for i in range(pop_size):
                cost, results, viol = evaluator.evaluate(offspring[i])
                off_objectives[i] = self._extract_objectives(results, evaluator)
                off_violations[i] = evaluator.solve_violation(results)

            # Combined population
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([pop_objectives, off_objectives])
            combined_viol = np.concatenate([pop_violations, off_violations])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_obj, combined_viol)

            # Select next generation with crowding distance
            population, pop_objectives, pop_violations = self._select_next_gen(
                combined_pop, combined_obj, combined_viol, fronts, pop_size
            )

            # Extract current Pareto front
            front_0 = fronts[0] if fronts else []
            best_front = [(combined_pop[i], combined_obj[i]) for i in front_0
                         if combined_viol[i] <= 1e-6]

            # Callback. The worker's callback expects positional args
            # (x_normalized, cost, raw_results, violation), so mirror that.
            if callback and gen % 5 == 0:
                best_idx = front_0[0] if front_0 else 0
                bx = combined_pop[best_idx]
                _, braw, bviol = evaluator.evaluate(bx)
                callback(bx, self._weighted_objective(braw, evaluator), braw, bviol)

        # Build result from best feasible solution (or least-infeasible fallback)
        if best_front:
            best_x, _ = best_front[0]
        else:
            feasible = pop_violations < 1e-6
            if np.any(feasible):
                feas_indices = np.where(feasible)[0]
                best_local = np.argmin(pop_objectives[feasible, 0])
                best_x = population[feas_indices[best_local]]
            else:
                best_x = population[np.argmin(pop_violations)]

        cost, results, viol = evaluator.evaluate(best_x)
        x_phys = evaluator.to_physical(best_x)
        evaluator.scaling = original_scaling

        objectives = {obj.name: results.get(obj.name, 0.0) for obj in evaluator.objs}
        constraints = {con.name: results.get(con.name, 0.0) for con in evaluator.cons}

        return OptimizationResult(
            x=x_phys,
            cost=self._weighted_objective(results, evaluator),
            objectives=objectives,
            constraints=constraints,
            max_violation=viol,
            message=f"NSGA-II completed ({len(best_front)} Pareto solutions)",
            success=viol <= 1e-6,
        )

    def _initialize_population(self, evaluator, x0, pop_size, n_vars):
        """Initialize population with LHS + x0."""
        from scipy.stats.qmc import LatinHypercube

        population = np.zeros((pop_size, n_vars))
        # First individual is x0
        population[0] = evaluator.to_normalized(x0)

        # LHS for rest
        sampler = LatinHypercube(d=n_vars, seed=42)
        samples = sampler.random(n=pop_size - 1)
        population[1:] = samples

        return np.clip(population, 0, 1)

    @staticmethod
    def _weighted_objective(raw, evaluator):
        """Signed, weighted sum of the raw objective values (display/cost scalar)."""
        total = 0.0
        for obj in evaluator.objs:
            val = raw.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            total += sign * obj.weight * val
        return total

    def _extract_objectives(self, results, evaluator):
        """Extract objective values from evaluation results."""
        obj_values = []
        for obj in evaluator.objs:
            val = results.get(obj.name, 0.0)
            if isinstance(val, (list, np.ndarray)):
                val = float(np.mean(val))
            if not obj.minimize:
                val = -val  # Convert max to min
            obj_values.append(val * obj.weight)
        return np.array(obj_values)

    def _create_offspring(
        self, population, pop_size, n_vars, evaluator,
        crossover_prob, mutation_prob, eta_c, eta_m
    ):
        """Create offspring via SBX crossover + polynomial mutation."""
        offspring = np.zeros_like(population)

        for i in range(0, pop_size, 2):
            # Tournament selection
            p1 = self._tournament_select(population, pop_size)
            p2 = self._tournament_select(population, pop_size)

            c1, c2 = p1.copy(), p2.copy()

            # SBX crossover
            if np.random.random() < crossover_prob:
                c1, c2 = self._sbx_crossover(p1, p2, eta_c)

            # Polynomial mutation
            c1 = self._polynomial_mutation(c1, mutation_prob, eta_m)
            c2 = self._polynomial_mutation(c2, mutation_prob, eta_m)

            offspring[i] = np.clip(c1, 0, 1)
            if i + 1 < pop_size:
                offspring[i + 1] = np.clip(c2, 0, 1)

        return offspring

    def _tournament_select(self, population, pop_size, k=2):
        """Binary tournament selection."""
        indices = np.random.choice(pop_size, k, replace=False)
        return population[indices[0]].copy()

    def _sbx_crossover(self, p1, p2, eta):
        """Simulated Binary Crossover (SBX)."""
        c1, c2 = p1.copy(), p2.copy()
        for j in range(len(p1)):
            if np.random.random() < 0.5:
                if abs(p1[j] - p2[j]) > 1e-14:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    c1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    c2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
        return c1, c2

    def _polynomial_mutation(self, individual, prob, eta):
        """Polynomial mutation."""
        result = individual.copy()
        for j in range(len(result)):
            if np.random.random() < prob:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                result[j] += delta
        return result

    def _non_dominated_sort(self, objectives, violations):
        """Fast non-dominated sorting (O(MN^2) for M objectives, N individuals)."""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(objectives[i], violations[i], objectives[j], violations[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], violations[j], objectives[i], violations[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        # First front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Subsequent fronts
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _dominates(self, obj_a, viol_a, obj_b, viol_b):
        """Check if solution a dominates solution b (constraint-handling)."""
        # Feasibility first
        if viol_a <= 1e-6 and viol_b > 1e-6:
            return True
        if viol_a > 1e-6 and viol_b <= 1e-6:
            return False
        if viol_a > 1e-6 and viol_b > 1e-6:
            return viol_a < viol_b

        # Both feasible: Pareto dominance
        return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))

    def _select_next_gen(self, pop, obj, viol, fronts, target_size):
        """Select next generation using fronts + crowding distance."""
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(front)
            else:
                # Need partial front — use crowding distance
                remaining = target_size - len(selected)
                crowd_dist = self._crowding_distance(obj[front])
                sorted_by_crowd = [front[i] for i in np.argsort(-crowd_dist)]
                selected.extend(sorted_by_crowd[:remaining])
                break

        selected = np.array(selected, dtype=int)
        return pop[selected], obj[selected], viol[selected]

    def _crowding_distance(self, objectives):
        """Compute crowding distance for a set of solutions."""
        n = len(objectives)
        if n <= 2:
            return np.full(n, float("inf"))

        distances = np.zeros(n)
        n_obj = objectives.shape[1]

        for m in range(n_obj):
            sorted_idx = np.argsort(objectives[:, m])
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")

            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
            if obj_range < 1e-15:
                continue

            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    objectives[sorted_idx[i + 1], m] - objectives[sorted_idx[i - 1], m]
                ) / obj_range

        return distances

    def _solve_weighted_sum(self, evaluator, x0, callback=None):
        """Multi-start weighted-sum scalarization for Pareto approximation."""
        n_points = self.settings.get("pareto_points", 11)
        n_obj = len(evaluator.objs)

        if n_obj < 2:
            n_points = 1
            weight_sets = [np.array([1.0])]
        else:
            # Generate weight combinations
            weight_sets = self._generate_weight_sets(n_obj, n_points)

        from pylcss.optimization.solvers.scipy_solver import ScipySolver

        all_results = []
        for i, weights in enumerate(weight_sets):
            if self.stop_requested:
                break

            settings = dict(self.settings)
            # ScipySolver feeds settings["method"] straight to scipy.optimize;
            # it must be a real method name, not "NSGA-II".
            settings["method"] = settings.get("ms_local_solver", "SLSQP")
            settings["objective_weights"] = weights.tolist()
            solver = ScipySolver(settings)

            try:
                result = solver.solve(evaluator, x0, callback=callback)
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Weighted-sum run {i} failed: {e}")

        if all_results:
            best = min(all_results, key=lambda r: r.cost)
            best.message += f" ({len(all_results)} Pareto points explored)"
            return best

        return OptimizationResult(
            x=x0, cost=float("inf"), objectives={}, constraints={},
            max_violation=float("inf"), message="All runs failed", success=False,
        )

    def _generate_weight_sets(self, n_obj, n_points):
        """Generate evenly distributed weight vectors for n objectives."""
        if n_obj == 2:
            weights = []
            for i in range(n_points):
                w1 = i / (n_points - 1) if n_points > 1 else 0.5
                weights.append(np.array([w1, 1 - w1]))
            return weights
        else:
            # Simplex-lattice design
            from itertools import product
            weights = []
            levels = np.linspace(0, 1, n_points)
            for combo in product(levels, repeat=n_obj - 1):
                if sum(combo) <= 1.0:
                    w = list(combo) + [1.0 - sum(combo)]
                    weights.append(np.array(w))
            return weights[:n_points]


class MultiStartSolver(BaseSolver):
    """
    Multi-start optimization for global search.
    Runs multiple local optimizations from different starting points.
    """

    def solve(self, evaluator, x0, callback=None):
        n_starts = int(self.settings.get("ms_n_starts", 10))
        n_vars = len(evaluator.vars)
        local_method = self.settings.get("ms_local_solver", "SLSQP")

        from pylcss.optimization.solvers.factory import get_solver
        from scipy.stats.qmc import LatinHypercube

        # Local runs need a real scipy method name, not "Multi-Start", or
        # get_solver() would recurse straight back into this solver.
        sub_settings = dict(self.settings)
        sub_settings["method"] = local_method

        # Build starting points directly in PHYSICAL space so this works
        # regardless of the scaling preference. x0 is always the first start.
        lowers = np.array([v.min_val for v in evaluator.vars], dtype=float)
        uppers = np.array([v.max_val for v in evaluator.vars], dtype=float)
        finite = np.isfinite(lowers) & np.isfinite(uppers)

        x0 = np.asarray(x0, dtype=float)
        starts = [x0]
        if n_starts > 1:
            unit = LatinHypercube(d=n_vars, seed=42).random(n=n_starts - 1)
            for row in unit:
                pt = x0.copy()
                pt[finite] = lowers[finite] + row[finite] * (uppers[finite] - lowers[finite])
                # Unbounded variables jitter around x0 so the starts still differ.
                pt[~finite] = x0[~finite] + (row[~finite] - 0.5)
                starts.append(pt)

        best_result = None
        for i, x_start in enumerate(starts):
            if self.stop_requested:
                break

            try:
                sub_solver = get_solver(local_method, sub_settings)
                result = sub_solver.solve(evaluator, x_start, callback=callback)
                if best_result is None or (
                    result.max_violation <= 1e-6
                    and (best_result.max_violation > 1e-6 or result.cost < best_result.cost)
                ):
                    best_result = result
            except Exception as e:
                logger.warning(f"Multi-start run {i} failed: {e}")

        if best_result is None:
            return OptimizationResult(
                x=x0, cost=float("inf"), objectives={}, constraints={},
                max_violation=float("inf"), message="All starts failed", success=False,
            )

        best_result.message += f" (best of {n_starts} starts)"
        return best_result
