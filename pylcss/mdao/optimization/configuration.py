# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Parse and validate UI/JSON optimization configuration.

Keeping this boundary outside the Qt worker makes malformed persisted projects
fail with clear messages before a numerical backend is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .evaluator import SCALING_MODES
from .models import Constraint, FloatArray, Objective, Variable
from .methods import GLOBAL_METHODS, SCIPY_METHODS, SUPPORTED_METHODS
from .parsing import parse_boolean


@dataclass(frozen=True)
class OptimizationSetup:
    """Validated problem data used to construct an evaluator."""

    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    initial_design: FloatArray
    parameters: dict[str, Any]


def parse_optimization_setup(
    setup_data: Mapping[str, Any],
    solver_settings: Mapping[str, Any],
) -> OptimizationSetup:
    """Convert serialized/UI dictionaries into validated domain models."""
    setup = _as_mapping(setup_data, "Optimization setup")
    settings = _as_mapping(solver_settings, "Solver settings")
    method = str(settings.get("method") or "SLSQP")

    variables = tuple(
        _parse_variable(item)
        for item in _mapping_items(setup.get("variables"), "variables")
    )
    objectives = tuple(
        _parse_objective(item)
        for item in _mapping_items(setup.get("objectives"), "objectives")
    )
    constraints = tuple(
        _parse_constraint(item)
        for item in _mapping_items(setup.get("constraints", ()), "constraints")
    )

    if method == "NSGA-II":
        # Weights and scalar reference scales have no role in Pareto dominance.
        objectives = tuple(
            replace(objective, weight=1.0, scale=None) for objective in objectives
        )

    if not variables:
        raise ValueError("Optimization needs at least one design variable.")
    if not objectives:
        raise ValueError("Optimization needs at least one objective.")
    _require_unique_names(variables, "Design-variable")
    _require_unique_names(objectives, "Objective output")
    _require_unique_names(constraints, "Constraint output")
    if not any(objective.weight > 0.0 for objective in objectives):
        raise ValueError("At least one objective must have a positive weight.")

    initial_design = np.asarray(setup.get("x0"), dtype=float)
    if initial_design.shape != (len(variables),) or not np.all(
        np.isfinite(initial_design)
    ):
        raise ValueError(f"Initial design must contain {len(variables)} finite values.")
    for value, variable in zip(initial_design, variables):
        if value < variable.min_val or value > variable.max_val:
            raise ValueError(
                f"Initial value for {variable.name!r} lies outside its bounds."
            )

    parameters = dict(
        _as_mapping(setup.get("parameters", {}), "Optimization parameters")
    )
    collisions = sorted(set(parameters).intersection(item.name for item in variables))
    if collisions:
        raise ValueError(
            "Parameters cannot reuse design-variable names: "
            + ", ".join(collisions)
            + "."
        )

    parsed = OptimizationSetup(
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        initial_design=initial_design.copy(),
        parameters=parameters,
    )
    validate_solver_settings(settings, parsed)
    return parsed


def validate_solver_settings(
    settings: Mapping[str, Any],
    setup: OptimizationSetup,
) -> None:
    """Validate common and method-specific numerical settings."""
    method = str(settings.get("method") or "SLSQP")
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown optimization method {method!r}. Supported methods: "
            + ", ".join(SUPPORTED_METHODS)
            + "."
        )

    scaling = parse_boolean(settings.get("scaling", True), "Variable scaling")
    scaling_mode = str(settings.get("scaling_mode", "auto")).strip().lower()
    if scaling_mode not in SCALING_MODES:
        raise ValueError(
            "Scaling mode must be one of: " + ", ".join(SCALING_MODES) + "."
        )
    finite_box = all(
        np.isfinite(variable.min_val) and np.isfinite(variable.max_val)
        for variable in setup.variables
    )
    if scaling and not finite_box:
        raise ValueError(
            "Variable scaling requires finite lower and upper bounds on every "
            "design variable."
        )
    if scaling and scaling_mode == "log":
        nonpositive = [
            variable.name
            for variable in setup.variables
            if variable.min_val <= 0.0 and variable.min_val != variable.max_val
        ]
        if nonpositive:
            raise ValueError(
                "Logarithmic scaling requires positive lower bounds for: "
                + ", ".join(nonpositive)
                + "."
            )
    if method in (*GLOBAL_METHODS, "NSGA-II") and not finite_box:
        raise ValueError(f"{method} requires finite variable bounds.")

    objective_scale = _finite_float(
        settings.get("objective_scale", 1.0),
        "Objective scale",
    )
    if objective_scale <= 0.0:
        raise ValueError("Objective scale must be greater than zero.")

    constraint_margin = _finite_float(
        settings.get("constraint_margin", 0.0),
        "Constraint safety margin",
    )
    if not 0.0 <= constraint_margin < 1.0:
        raise ValueError("Constraint safety margin must be in [0, 1).")

    penalty_weight = _finite_float(
        settings.get("penalty_weight", 1e6),
        "Penalty weight",
    )
    if penalty_weight <= 0.0:
        raise ValueError("Penalty weight must be greater than zero.")

    max_iterations = int(settings.get("maxiter", 1000))
    if max_iterations < 1:
        raise ValueError("Iteration/evaluation budget must be at least 1.")
    tolerance = _finite_float(
        settings.get("tol", 1e-6),
        "Optimization tolerance",
    )
    if tolerance <= 0.0:
        raise ValueError("Optimization tolerance must be positive.")
    feasibility_tolerance = _finite_float(
        settings.get("feasibility_tol", tolerance),
        "Feasibility tolerance",
    )
    if feasibility_tolerance <= 0.0:
        raise ValueError("Feasibility tolerance must be positive.")

    if method == "Differential Evolution":
        _validate_differential_evolution(settings)
    elif method == "Nevergrad":
        if int(settings.get("num_workers", 1)) < 1:
            raise ValueError("Nevergrad worker count must be at least 1.")
    elif method == "NSGA-II":
        _validate_nsga2(settings, len(setup.objectives))
    elif method == "Multi-Start":
        if int(settings.get("ms_n_starts", 10)) < 1:
            raise ValueError("Multi-start optimization needs at least one start.")
        local_method = str(settings.get("ms_local_solver", "SLSQP"))
        if local_method not in SCIPY_METHODS:
            raise ValueError(
                "Multi-start local solver must be one of: "
                + ", ".join(SCIPY_METHODS)
                + "."
            )


def _parse_variable(item: Mapping[str, Any]) -> Variable:
    name = _required_name(item, "variable")
    return Variable(
        name=name,
        min_val=item.get("min_val", item.get("min", 0.0)),
        max_val=item.get("max_val", item.get("max", 1.0)),
        value=item.get("value", 0.0),
    )


def _parse_objective(item: Mapping[str, Any]) -> Objective:
    raw_scale = item.get("scale")
    scale = (
        None
        if raw_scale is None
        or (isinstance(raw_scale, str) and raw_scale.strip().lower() in {"", "auto"})
        else raw_scale
    )
    return Objective(
        name=_required_name(item, "objective"),
        weight=item.get("weight", 1.0),
        minimize=parse_boolean(item.get("minimize", True), "Objective direction"),
        scale=scale,
    )


def _parse_constraint(item: Mapping[str, Any]) -> Constraint:
    minimum = item.get(
        "min_val",
        item.get("min", item.get("req_min", float("-inf"))),
    )
    maximum = item.get(
        "max_val",
        item.get("max", item.get("req_max", float("inf"))),
    )
    return Constraint(
        name=_required_name(item, "constraint"),
        min_val=float("-inf") if minimum is None else minimum,
        max_val=float("inf") if maximum is None else maximum,
    )


def _validate_differential_evolution(settings: Mapping[str, Any]) -> None:
    if int(settings.get("popsize", 15)) < 1:
        raise ValueError("Population multiplier must be at least 1.")

    mutation = settings.get("mutation", (0.5, 1.0))
    mutation_values: tuple[float, ...]
    if isinstance(mutation, (int, float, np.number)):
        mutation_values = (float(mutation),)
    else:
        try:
            mutation_values = tuple(float(value) for value in mutation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Differential Evolution mutation must be one number or a "
                "two-number range."
            ) from exc
    if len(mutation_values) not in (1, 2):
        raise ValueError(
            "Differential Evolution mutation must be one number or a two-number range."
        )
    if not all(np.isfinite(value) and 0.0 <= value < 2.0 for value in mutation_values):
        raise ValueError("Mutation values must satisfy 0 <= value < 2.")
    if len(mutation_values) == 2 and mutation_values[0] > mutation_values[1]:
        raise ValueError("Mutation range minimum cannot exceed its maximum.")

    recombination = _finite_float(
        settings.get("recombination", 0.7),
        "Recombination",
    )
    if not 0.0 <= recombination <= 1.0:
        raise ValueError("Recombination must be between 0 and 1.")
    worker_count = int(settings.get("num_workers", settings.get("workers", 1)))
    if worker_count != 1:
        raise ValueError(
            "Differential Evolution currently requires one worker because "
            "in-process engineering model callbacks are not safely picklable."
        )


def _validate_nsga2(settings: Mapping[str, Any], objective_count: int) -> None:
    if objective_count < 2:
        raise ValueError("NSGA-II is a Pareto optimizer and needs two objectives.")
    if int(settings.get("nsga_popsize", 100)) < 4:
        raise ValueError("NSGA-II population size must be at least 4.")
    if int(settings.get("nsga_generations", 200)) < 1:
        raise ValueError("NSGA-II generations must be at least 1.")

    crossover = _finite_float(
        settings.get("nsga_crossover_prob", 0.9),
        "NSGA-II crossover probability",
    )
    if not 0.0 <= crossover <= 1.0:
        raise ValueError("NSGA-II crossover probability must be in [0, 1].")
    mutation = settings.get("nsga_mutation_prob")
    if mutation is not None:
        mutation_probability = _finite_float(
            mutation,
            "NSGA-II mutation probability",
        )
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("NSGA-II mutation probability must be in [0, 1].")
    for key, label in (
        ("nsga_eta_c", "NSGA-II crossover distribution index"),
        ("nsga_eta_m", "NSGA-II mutation distribution index"),
    ):
        if _finite_float(settings.get(key, 20.0), label) <= 0.0:
            raise ValueError(f"{label} must be positive.")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a dictionary-like mapping.")
    return value


def _mapping_items(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Optimization {label} must be a list.")
    return tuple(_as_mapping(item, f"Optimization {label} entry") for item in value)


def _required_name(item: Mapping[str, Any], kind: str) -> str:
    try:
        name = item["name"]
    except KeyError as exc:
        raise ValueError(f"Every optimization {kind} needs a name.") from exc
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Every optimization {kind} needs a non-empty name.")
    return name


def _require_unique_names(items: Sequence[Any], label: str) -> None:
    names = [item.name for item in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} names must be unique: {', '.join(duplicates)}.")


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


__all__ = [
    "OptimizationSetup",
    "parse_optimization_setup",
    "validate_solver_settings",
]
