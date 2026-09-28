# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Selection and variation operators used by the NSGA-II solver."""

from __future__ import annotations

import numpy as np

from ..evaluator import FEASIBILITY_TOLERANCE


class Nsga2OperatorsMixin:
    """Reusable, side-effect-free population operations apart from RNG state."""

    _rng: np.random.Generator
    _feasibility_tolerance: float = FEASIBILITY_TOLERANCE

    def _create_offspring(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pop_size: int,
        crossover_prob: float,
        mutation_prob: float,
        eta_c: float,
        eta_m: float,
    ) -> np.ndarray:
        """Create offspring via SBX crossover and polynomial mutation."""
        offspring = np.zeros_like(population)
        for index in range(0, pop_size, 2):
            parent_1 = self._tournament_select(population, ranks, crowding)
            parent_2 = self._tournament_select(population, ranks, crowding)
            child_1, child_2 = parent_1.copy(), parent_2.copy()
            if self._rng.random() < crossover_prob:
                child_1, child_2 = self._sbx_crossover(
                    parent_1,
                    parent_2,
                    eta_c,
                )
            child_1 = self._polynomial_mutation(
                child_1,
                mutation_prob,
                eta_m,
            )
            child_2 = self._polynomial_mutation(
                child_2,
                mutation_prob,
                eta_m,
            )
            offspring[index] = np.clip(child_1, 0.0, 1.0)
            if index + 1 < pop_size:
                offspring[index + 1] = np.clip(child_2, 0.0, 1.0)
        return offspring

    def _tournament_select(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
    ) -> np.ndarray:
        """Prefer a lower Pareto rank, then a larger crowding distance."""
        if len(population) == 1:
            return population[0].copy()
        first, second = self._rng.choice(
            len(population),
            2,
            replace=False,
        )
        if ranks[first] < ranks[second]:
            winner = first
        elif ranks[second] < ranks[first]:
            winner = second
        elif crowding[first] > crowding[second]:
            winner = first
        elif crowding[second] > crowding[first]:
            winner = second
        else:
            winner = first if self._rng.random() < 0.5 else second
        return population[winner].copy()

    def _sbx_crossover(
        self,
        parent_1: np.ndarray,
        parent_2: np.ndarray,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply simulated binary crossover (SBX)."""
        child_1, child_2 = parent_1.copy(), parent_2.copy()
        for index in range(len(parent_1)):
            if (
                self._rng.random() >= 0.5
                or abs(parent_1[index] - parent_2[index]) <= 1e-14
            ):
                continue
            sample = self._rng.random()
            beta = (
                (2.0 * sample) ** (1.0 / (eta + 1.0))
                if sample <= 0.5
                else (1.0 / (2.0 * (1.0 - sample))) ** (1.0 / (eta + 1.0))
            )
            child_1[index] = 0.5 * (
                (1.0 + beta) * parent_1[index] + (1.0 - beta) * parent_2[index]
            )
            child_2[index] = 0.5 * (
                (1.0 - beta) * parent_1[index] + (1.0 + beta) * parent_2[index]
            )
        return child_1, child_2

    def _polynomial_mutation(
        self,
        individual: np.ndarray,
        probability: float,
        eta: float,
    ) -> np.ndarray:
        """Apply polynomial mutation in normalized coordinates."""
        result = individual.copy()
        for index in range(len(result)):
            if self._rng.random() >= probability:
                continue
            sample = self._rng.random()
            delta = (
                (2.0 * sample) ** (1.0 / (eta + 1.0)) - 1.0
                if sample < 0.5
                else 1.0 - (2.0 * (1.0 - sample)) ** (1.0 / (eta + 1.0))
            )
            result[index] += delta
        return result

    def _non_dominated_sort(
        self,
        objectives: np.ndarray,
        violations: np.ndarray,
    ) -> list[list[int]]:
        """Sort candidates by constrained Pareto dominance."""
        candidate_count = len(objectives)
        domination_count: np.ndarray = np.zeros(candidate_count, dtype=int)
        dominated: list[list[int]] = [[] for _ in range(candidate_count)]
        fronts: list[list[int]] = [[]]

        for first in range(candidate_count):
            for second in range(first + 1, candidate_count):
                if self._dominates(
                    objectives[first],
                    float(violations[first]),
                    objectives[second],
                    float(violations[second]),
                ):
                    dominated[first].append(second)
                    domination_count[second] += 1
                elif self._dominates(
                    objectives[second],
                    float(violations[second]),
                    objectives[first],
                    float(violations[first]),
                ):
                    dominated[second].append(first)
                    domination_count[first] += 1

        fronts[0].extend(
            index for index in range(candidate_count) if domination_count[index] == 0
        )
        front_index = 0
        while fronts[front_index]:
            next_front: list[int] = []
            for candidate in fronts[front_index]:
                for dominated_candidate in dominated[candidate]:
                    domination_count[dominated_candidate] -= 1
                    if domination_count[dominated_candidate] == 0:
                        next_front.append(dominated_candidate)
            fronts.append(next_front)
            front_index += 1
        return [front for front in fronts if front]

    def _dominates(
        self,
        objectives_a: np.ndarray,
        violation_a: float,
        objectives_b: np.ndarray,
        violation_b: float,
    ) -> bool:
        """Return whether candidate A dominates candidate B."""
        feasible_a = violation_a <= self._feasibility_tolerance
        feasible_b = violation_b <= self._feasibility_tolerance
        if feasible_a and not feasible_b:
            return True
        if feasible_b and not feasible_a:
            return False
        if not feasible_a and not feasible_b:
            return violation_a < violation_b
        return bool(
            np.all(objectives_a <= objectives_b) and np.any(objectives_a < objectives_b)
        )

    def _select_next_gen(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        violations: np.ndarray,
        fronts: list[list[int]],
        target_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retain complete fronts, then the least crowded partial front."""
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(front)
                continue
            remaining = target_size - len(selected)
            crowding = self._crowding_distance(objectives[front])
            by_crowding = [front[index] for index in np.argsort(-crowding)]
            selected.extend(by_crowding[:remaining])
            break
        indices = np.asarray(selected, dtype=int)
        return (
            population[indices],
            objectives[indices],
            violations[indices],
        )

    @staticmethod
    def _crowding_distance(objectives: np.ndarray) -> np.ndarray:
        """Compute normalized NSGA-II crowding distances."""
        candidate_count = len(objectives)
        if candidate_count <= 2:
            return np.full(candidate_count, float("inf"))

        distances: np.ndarray = np.zeros(candidate_count, dtype=float)
        for objective_index in range(objectives.shape[1]):
            order = np.argsort(objectives[:, objective_index])
            distances[order[0]] = float("inf")
            distances[order[-1]] = float("inf")
            objective_range = (
                objectives[order[-1], objective_index]
                - objectives[order[0], objective_index]
            )
            if objective_range < 1e-15:
                continue
            for rank in range(1, candidate_count - 1):
                distances[order[rank]] += (
                    objectives[order[rank + 1], objective_index]
                    - objectives[order[rank - 1], objective_index]
                ) / objective_range
        return distances


__all__ = ["Nsga2OperatorsMixin"]
