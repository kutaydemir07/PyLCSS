# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Model execution, coordinate scaling, caching, and scalarization."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .models import Constraint, FloatArray, Objective, Variable

EvaluationOutputs: TypeAlias = dict[str, Any]
EvaluationResult: TypeAlias = tuple[float, EvaluationOutputs, float]
ModelFunction: TypeAlias = Callable[..., Mapping[str, Any]]

EVALUATION_FAILURE: Final = 1e15
EVALUATION_ERROR_KEY: Final = "__pylcss_evaluation_error__"
FEASIBILITY_TOLERANCE: Final = 1e-6
LOG_SCALING_RATIO: Final = 1e3
SCALING_MODES: Final = ("auto", "linear", "log")


def uses_log_scaling(
    lower: float,
    upper: float,
    mode: str = "auto",
) -> bool:
    """Return whether a bounded variable should use logarithmic coordinates."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SCALING_MODES:
        raise ValueError(
            "Scaling mode must be one of: " + ", ".join(SCALING_MODES) + "."
        )
    lower_value = float(lower)
    upper_value = float(upper)
    if (
        not np.isfinite(lower_value)
        or not np.isfinite(upper_value)
        or lower_value <= 0.0
        or upper_value <= lower_value
    ):
        return False
    return normalized_mode == "log" or (
        normalized_mode == "auto"
        and np.log(upper_value) - np.log(lower_value)
        >= np.log(LOG_SCALING_RATIO)
    )


def default_initial_value(
    lower: float,
    upper: float,
    *,
    scaling: bool = True,
    scaling_mode: str = "auto",
) -> float:
    """Choose the center of a variable's active solver coordinate interval."""
    minimum = float(lower)
    maximum = float(upper)
    if np.isfinite(minimum) and np.isfinite(maximum):
        if scaling and uses_log_scaling(minimum, maximum, scaling_mode):
            return float(np.exp(0.5 * (np.log(minimum) + np.log(maximum))))
        return 0.5 * (minimum + maximum)
    if np.isfinite(minimum):
        return minimum + 1.0
    if np.isfinite(maximum):
        return maximum - 1.0
    return 0.0


class ModelEvaluator:
    """Evaluate a named-output model for optimization solvers.

    Solver coordinates may be normalized to the unit box while the wrapped
    model always receives physical values. Successful model responses are held
    in a bounded least-recently-used cache; objective and penalty scores are
    recomputed on every access so changing scalarization weights cannot return
    stale costs.
    """

    def __init__(
        self,
        system_model: ModelFunction,
        variables: Sequence[Variable],
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        parameters: Mapping[str, Any] | None = None,
        scaling: bool = True,
        scaling_mode: str = "auto",
        penalty_weight: float = 1e6,
        objective_scale: float = 1.0,
        constraint_margin: float = 0.0,
        feasibility_tolerance: float = FEASIBILITY_TOLERANCE,
        cache_size: int = 5000,
    ) -> None:
        if not callable(system_model):
            raise TypeError("System model must be callable.")

        self.model = system_model
        self.variables = tuple(variables)
        self.objectives = tuple(objectives)
        self.constraints = tuple(constraints)
        # Compatibility attributes used by existing solver/UI integrations.
        self.vars = self.variables
        self.objs = self.objectives
        self.cons = self.constraints

        self.parameters = dict(parameters or {})
        self.scaling = bool(scaling)
        self.scaling_mode = str(scaling_mode).strip().lower()
        if self.scaling_mode not in SCALING_MODES:
            raise ValueError(
                "Scaling mode must be one of: " + ", ".join(SCALING_MODES) + "."
            )
        self.penalty_weight = _positive_finite(
            penalty_weight,
            "Penalty weight",
        )
        self.objective_scale = _positive_finite(
            objective_scale,
            "Objective scale",
        )
        self.constraint_margin = float(constraint_margin)
        if (
            not np.isfinite(self.constraint_margin)
            or not 0.0 <= self.constraint_margin < 1.0
        ):
            raise ValueError("Constraint margin must be finite and in [0, 1).")
        if int(cache_size) < 0:
            raise ValueError("Evaluation cache size cannot be negative.")
        self._cache_size = int(cache_size)
        self.set_feasibility_tolerance(feasibility_tolerance)

        self._validate_problem()
        self._objective_scales: FloatArray | None = None
        self._cache: OrderedDict[tuple[float, ...], EvaluationOutputs] = OrderedDict()
        self.last_error: str | None = None
        self.failure_count = 0
        self.evaluation_count = 0

        self._lower = np.asarray(
            [variable.min_val for variable in self.variables],
            dtype=float,
        )
        self._upper = np.asarray(
            [variable.max_val for variable in self.variables],
            dtype=float,
        )
        physical_ranges = self._upper - self._lower
        self._fixed = np.abs(physical_ranges) <= 1e-15
        if self.scaling and self.scaling_mode == "log":
            invalid_log_bounds = (
                ~self._fixed
                & (
                    ~np.isfinite(self._lower)
                    | ~np.isfinite(self._upper)
                    | (self._lower <= 0.0)
                )
            )
            if np.any(invalid_log_bounds):
                names = ", ".join(
                    self.variables[index].name
                    for index in np.flatnonzero(invalid_log_bounds)
                )
                raise ValueError(
                    "Logarithmic scaling requires finite positive bounds for: "
                    + names
                    + "."
                )
        self._log_scaled = np.asarray(
            [
                not self._fixed[index]
                and uses_log_scaling(
                    variable.min_val,
                    variable.max_val,
                    self.scaling_mode,
                )
                for index, variable in enumerate(self.variables)
            ],
            dtype=bool,
        )
        self._coordinate_lower = self._lower.copy()
        self._coordinate_upper = self._upper.copy()
        self._coordinate_lower[self._log_scaled] = np.log(
            self._lower[self._log_scaled]
        )
        self._coordinate_upper[self._log_scaled] = np.log(
            self._upper[self._log_scaled]
        )
        raw_ranges = self._coordinate_upper - self._coordinate_lower
        self._ranges = raw_ranges.copy()
        self._ranges[self._fixed] = 1.0
        if self.scaling and (
            not np.all(np.isfinite(self._lower)) or not np.all(np.isfinite(self._upper))
        ):
            raise ValueError("Variable scaling requires finite lower and upper bounds.")

        self._constraint_scales = np.asarray(
            [self._constraint_scale(item) for item in self.constraints],
            dtype=float,
        )
        self._solve_lower = np.asarray(
            [item.min_val for item in self.constraints],
            dtype=float,
        )
        self._solve_upper = np.asarray(
            [item.max_val for item in self.constraints],
            dtype=float,
        )
        self._tighten_constraint_bounds()

    def _validate_problem(self) -> None:
        _validate_unique_names(self.variables, "Variable")
        _validate_unique_names(self.objectives, "Objective")
        _validate_unique_names(self.constraints, "Constraint")
        collisions = sorted(
            set(self.parameters).intersection(
                variable.name for variable in self.variables
            )
        )
        if collisions:
            raise ValueError(
                "Parameters cannot reuse design-variable names: "
                + ", ".join(collisions)
                + "."
            )

    def _tighten_constraint_bounds(self) -> None:
        if not self.constraints or self.constraint_margin == 0.0:
            return

        backoff = self.constraint_margin * self._constraint_scales
        finite_lower = np.isfinite(self._solve_lower)
        finite_upper = np.isfinite(self._solve_upper)
        self._solve_lower[finite_lower] += backoff[finite_lower]
        self._solve_upper[finite_upper] -= backoff[finite_upper]
        inverted = finite_lower & finite_upper & (self._solve_lower > self._solve_upper)
        if np.any(inverted):
            names = ", ".join(
                self.constraints[index].name for index in np.flatnonzero(inverted)
            )
            raise ValueError(
                f"Constraint margin is too large for the admissible band of: {names}."
            )

    def to_normalized(self, x_physical: ArrayLike) -> FloatArray:
        """Convert physical variables to solver coordinates."""
        values = np.asarray(x_physical, dtype=float)
        if not self.scaling:
            return values.copy()
        coordinates = values.copy()
        if np.any(self._log_scaled):
            selected = coordinates[..., self._log_scaled]
            if np.any(selected <= 0.0):
                raise ValueError(
                    "Log-scaled design variables must be greater than zero."
                )
            coordinates[..., self._log_scaled] = np.log(selected)
        normalized = (coordinates - self._coordinate_lower) / self._ranges
        normalized[..., self._fixed] = 0.0
        return normalized

    def to_physical(self, x_solver: ArrayLike) -> FloatArray:
        """Convert current solver coordinates to physical variables."""
        values = np.asarray(x_solver, dtype=float)
        if not self.scaling:
            return values.copy()
        physical = values * self._ranges + self._coordinate_lower
        if np.any(self._log_scaled):
            physical[..., self._log_scaled] = np.exp(
                physical[..., self._log_scaled]
            )
        physical[..., self._fixed] = self._lower[self._fixed]
        return physical

    @property
    def log_scaled_variables(self) -> tuple[str, ...]:
        """Names of variables currently represented in logarithmic coordinates."""
        return tuple(
            self.variables[index].name
            for index in np.flatnonzero(self._log_scaled)
        )

    def set_feasibility_tolerance(self, value: float) -> None:
        """Set the relative feasibility tolerance shared by solver strategies."""
        self.feasibility_tolerance = _positive_finite(
            value,
            "Feasibility tolerance",
        )

    def is_feasible(self, violation: float) -> bool:
        return bool(float(violation) <= self.feasibility_tolerance)

    @staticmethod
    def _constraint_scale(constraint: Constraint) -> float:
        """Return a characteristic magnitude for relative violations."""
        lower, upper = constraint.min_val, constraint.max_val
        if np.isfinite(lower) and np.isfinite(upper):
            width = abs(upper - lower)
            if width > 1e-12:
                return width
        magnitudes = [
            abs(bound)
            for bound in (lower, upper)
            if np.isfinite(bound) and abs(bound) > 1e-12
        ]
        return min(magnitudes) if magnitudes else 1.0

    def constraint_solve_bounds(self, index: int) -> tuple[float, float]:
        """Return safety-tightened bounds used to construct constraints."""
        return (
            float(self._solve_lower[index]),
            float(self._solve_upper[index]),
        )

    def constraint_solver_scale(self, index: int) -> float:
        """Return the characteristic scale used by numerical constraints."""
        return float(self._constraint_scales[index])

    def _ensure_objective_scales(
        self,
        raw_results: Mapping[str, Any],
    ) -> None:
        if self._objective_scales is not None:
            return
        scales = []
        for objective in self.objectives:
            if objective.scale is not None:
                scale = objective.scale
            else:
                scale = abs(float(raw_results[objective.name]))
                if scale < 1e-12:
                    scale = 1.0
            scales.append(scale * self.objective_scale)
        self._objective_scales = np.asarray(scales, dtype=float)

    def normalized_objective(
        self,
        raw_results: Mapping[str, Any],
    ) -> float:
        """Return the signed weighted objective in scaled units."""
        self._ensure_objective_scales(raw_results)
        assert self._objective_scales is not None
        return float(
            sum(
                (1.0 if objective.minimize else -1.0)
                * objective.weight
                * float(raw_results[objective.name])
                / self._objective_scales[index]
                for index, objective in enumerate(self.objectives)
            )
        )

    def displayed_objective(
        self,
        raw_results: Mapping[str, Any],
    ) -> float:
        """Return the signed weighted objective in original output units."""
        return float(
            sum(
                (1.0 if objective.minimize else -1.0)
                * objective.weight
                * float(raw_results[objective.name])
                for objective in self.objectives
            )
        )

    @property
    def objective_scales(self) -> dict[str, float]:
        if self._objective_scales is None:
            return {}
        return {
            objective.name: float(self._objective_scales[index])
            for index, objective in enumerate(self.objectives)
        }

    def solve_violation(self, raw_results: Mapping[str, Any]) -> float:
        """Return maximum relative violation of safety-tightened bounds."""
        if not self.is_valid_result(raw_results):
            return EVALUATION_FAILURE

        worst = 0.0
        for index, constraint in enumerate(self.constraints):
            value = float(raw_results[constraint.name])
            lower = self._solve_lower[index]
            upper = self._solve_upper[index]
            violation = 0.0
            if value < lower:
                violation = lower - value
            elif value > upper:
                violation = value - upper
            if violation > 0.0:
                worst = max(
                    worst,
                    violation / self._constraint_scales[index],
                )
        return worst

    @staticmethod
    def is_valid_result(raw_results: Mapping[str, Any]) -> bool:
        """Return whether a model response passed output validation."""
        return (
            isinstance(raw_results, Mapping) and EVALUATION_ERROR_KEY not in raw_results
        )

    @staticmethod
    def evaluation_error(raw_results: Mapping[str, Any]) -> str | None:
        if isinstance(raw_results, Mapping):
            error = raw_results.get(EVALUATION_ERROR_KEY)
            return str(error) if error else None
        return "System model did not return a mapping of named outputs."

    def _invalid_result(self, message: str) -> EvaluationResult:
        self.last_error = str(message)
        self.failure_count += 1
        return (
            EVALUATION_FAILURE,
            {EVALUATION_ERROR_KEY: self.last_error},
            EVALUATION_FAILURE,
        )

    def evaluate(self, x_solver: ArrayLike) -> EvaluationResult:
        """Evaluate one solver-coordinate design.

        Returns ``(penalized_cost, model_outputs, max_relative_violation)``.
        Model failures are converted into a finite failure result so numerical
        backends can continue exploring other candidates.
        """
        values = np.asarray(x_solver, dtype=float)
        if values.shape != (len(self.variables),) or not np.all(np.isfinite(values)):
            return self._invalid_result(
                "Design variables must be a finite one-dimensional vector "
                f"with {len(self.variables)} values."
            )

        physical = self.to_physical(values)
        cache_key = tuple(float(value) for value in physical)
        raw_results = self._cache_get(cache_key)
        if raw_results is None:
            inputs = {
                variable.name: value
                for variable, value in zip(self.variables, physical)
            }
            inputs.update(self.parameters)
            try:
                self.evaluation_count += 1
                model_result = self.model(**inputs)
            except Exception as exc:
                return self._invalid_result(
                    f"System model raised {type(exc).__name__}: {exc}"
                )
            validated = self._validate_model_result(model_result)
            if isinstance(validated, str):
                return self._invalid_result(validated)
            raw_results = validated
            self._cache_put(cache_key, raw_results)

        self.last_error = None
        return self._score(raw_results)

    def _validate_model_result(
        self,
        raw_results: Any,
    ) -> EvaluationOutputs | str:
        if not isinstance(raw_results, Mapping):
            return (
                "System model must return a dictionary-like mapping of named outputs."
            )

        result = dict(raw_results)
        required_names = tuple(
            dict.fromkeys(
                [item.name for item in self.objectives]
                + [item.name for item in self.constraints]
            )
        )
        missing = [name for name in required_names if name not in result]
        if missing:
            return "System model omitted required output(s): " + ", ".join(missing)

        for name in required_names:
            value = result[name]
            if isinstance(value, np.ndarray) and value.ndim != 0:
                return f"System model output {name!r} must be a scalar number."
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                return f"System model output {name!r} must be a scalar number: {exc}"
            if not np.isfinite(scalar):
                return f"System model output {name!r} must be finite, got {scalar!r}."
            result[name] = scalar
        return result

    def _score(self, raw_results: EvaluationOutputs) -> EvaluationResult:
        objective = self.normalized_objective(raw_results)
        max_violation = 0.0
        penalty = 0.0
        for index, constraint in enumerate(self.constraints):
            value = float(raw_results[constraint.name])
            raw_violation = max(
                constraint.min_val - value,
                value - constraint.max_val,
                0.0,
            )
            if raw_violation > 0.0:
                relative = raw_violation / self._constraint_scales[index]
                max_violation = max(max_violation, relative)
                penalty += self.penalty_weight * relative

        cost = float(
            np.clip(objective + penalty, -EVALUATION_FAILURE, EVALUATION_FAILURE)
        )
        if not np.isfinite(cost):
            cost = EVALUATION_FAILURE
        return cost, dict(raw_results), max_violation

    def clear_cache(self) -> None:
        """Discard cached model outputs without resetting objective scales."""
        self._cache.clear()

    def _cache_get(
        self,
        key: tuple[float, ...],
    ) -> EvaluationOutputs | None:
        result = self._cache.get(key)
        if result is None:
            return None
        self._cache.move_to_end(key)
        return dict(result)

    def _cache_put(
        self,
        key: tuple[float, ...],
        result: EvaluationOutputs,
    ) -> None:
        if self._cache_size == 0:
            return
        self._cache[key] = dict(result)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _validate_unique_names(items: Sequence[Any], kind: str) -> None:
    names = [item.name for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{kind} names must be unique: {', '.join(duplicates)}.")


def _positive_finite(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return result


__all__ = [
    "EVALUATION_ERROR_KEY",
    "EVALUATION_FAILURE",
    "EvaluationOutputs",
    "EvaluationResult",
    "FEASIBILITY_TOLERANCE",
    "LOG_SCALING_RATIO",
    "ModelEvaluator",
    "ModelFunction",
    "SCALING_MODES",
    "default_initial_value",
    "uses_log_scaling",
]
