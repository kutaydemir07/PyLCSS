# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Design-problem definition evaluated by solution-space algorithms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
import math
from os import PathLike
import keyword
from typing import Any

import numpy as np
import scipy.io

from .units import (
    UNIT_REGISTRY,
    UnitError,
    convert_value,
    is_specified_unit,
    units_compatible,
)

logger = logging.getLogger(__name__)


class ModelEvaluationError(RuntimeError):
    """Raised when every sampled model evaluation fails."""


class DesignProblem:
    """Design variables, requirements, samples, and an executable model."""

    def __init__(self, name: str, sample_size: int = 300) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Problem name must be a non-empty string.")
        if not isinstance(sample_size, int) or isinstance(sample_size, bool):
            raise TypeError("sample_size must be an integer.")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")

        self.name = name.strip()
        self.sample_size = sample_size
        self.ureg = UNIT_REGISTRY
        self.design_variables: list[dict[str, Any]] = []
        self.parameters: list[dict[str, Any]] = []
        self.quantities_of_interest: list[dict[str, Any]] = []
        self.samples: dict[str, np.ndarray] = {}
        self.results: dict[str, np.ndarray] = {}
        self.diagram: list[tuple[str, str]] = []
        self.system_model: Callable[..., Mapping[str, Any]] | None = None
        self.system_code: str | None = None
        self.requirement_sets: dict[str, dict[str, dict[str, float]]] = {}

    def add_design_variable(
        self,
        name: str,
        unit: str,
        min_val: int | float | str,
        max_val: int | float | str,
    ) -> None:
        self._ensure_unique_input(name)
        lower, upper = _bounds(min_val, max_val, f"design variable {name!r}")
        _validate_unit(unit)
        self.design_variables.append(
            {
                "name": name,
                "unit": unit,
                "min": lower,
                "max": upper,
                "type": "continuous",
                "granularity": 1.0,
            }
        )

    def add_parameter(
        self,
        name: str,
        unit: str,
        value: int | float | str,
    ) -> None:
        self._ensure_unique_input(name)
        _validate_unit(unit)
        self.parameters.append(
            {"name": name, "unit": unit, "value": _finite_float(value, name)}
        )

    def add_quantity_of_interest(
        self,
        name: str,
        unit: str,
        min_val: int | float | str,
        max_val: int | float | str,
        minimize: bool = False,
        maximize: bool = False,
        weight: float = 1.0,
        display_name: str | None = None,
        show_in_legend: bool = True,
    ) -> None:
        _validate_name(name, "quantity of interest")
        if name in {item["name"] for item in self.quantities_of_interest}:
            raise ValueError(f"Duplicate quantity of interest: {name!r}.")
        if minimize and maximize:
            raise ValueError(f"{name!r} cannot be minimized and maximized.")
        # Infinite requirement limits express one-sided or unconstrained
        # quantities. They are valid here, unlike design-variable bounds used
        # by finite-domain samplers.
        lower, upper = _bounds(
            min_val,
            max_val,
            f"quantity {name!r}",
            allow_infinite=True,
        )
        numeric_weight = _finite_float(weight, f"weight for {name}")
        if numeric_weight < 0:
            raise ValueError("Objective weight cannot be negative.")
        _validate_unit(unit)
        self.quantities_of_interest.append(
            {
                "name": name,
                "display_name": display_name or name,
                "unit": unit,
                "min": lower,
                "max": upper,
                "minimize": bool(minimize),
                "maximize": bool(maximize),
                "weight": numeric_weight,
                "show_in_legend": bool(show_in_legend),
            }
        )

    def add_requirement_set(
        self,
        name: str,
        overrides: Mapping[str, Mapping[str, int | float]],
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Requirement-set name must be non-empty.")
        known_outputs = {item["name"] for item in self.quantities_of_interest}
        unknown = set(overrides) - known_outputs
        if unknown:
            raise ValueError(
                f"Requirement set {name!r} references unknown outputs: "
                f"{', '.join(sorted(unknown))}."
            )
        self.requirement_sets[name.strip()] = {
            output: {
                key: _number(
                    value,
                    f"{output}.{key}",
                    allow_infinite=True,
                )
                for key, value in values.items()
            }
            for output, values in overrides.items()
        }

    def set_system_model(
        self,
        model: Callable[..., Mapping[str, Any]],
    ) -> None:
        if not callable(model):
            raise TypeError("System model must be callable.")
        self.system_model = model

    def set_system_code(self, code: str) -> None:
        if not isinstance(code, str):
            raise TypeError("System code must be a string.")
        self.system_code = code

    def generate_samples(
        self,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Generate independent uniform samples for all design variables."""

        generator = rng or np.random.default_rng()
        self.samples = {
            variable["name"]: generator.uniform(
                variable["min"],
                variable["max"],
                self.sample_size,
            )
            for variable in self.design_variables
        }
        for parameter in self.parameters:
            self.samples[parameter["name"]] = np.full(
                self.sample_size,
                parameter["value"],
                dtype=float,
            )

    def evaluate(self) -> None:
        """Evaluate generated samples, retaining NaN for isolated bad rows."""

        model = self._require_model()
        expected_inputs = {
            item["name"] for item in self.design_variables + self.parameters
        }
        missing = expected_inputs - self.samples.keys()
        if missing:
            raise ValueError(
                f"Generate or provide samples for: {', '.join(sorted(missing))}."
            )

        rows: list[dict[str, Any]] = []
        failures: list[tuple[int, Exception]] = []
        for index in range(self.sample_size):
            row_input = {name: values[index] for name, values in self.samples.items()}
            try:
                result = model(**row_input)
                if not isinstance(result, Mapping):
                    raise TypeError("Model result is not a mapping.")
                rows.append(dict(result))
            except Exception as exc:
                failures.append((index, exc))
                rows.append({})

        if len(failures) == self.sample_size:
            first_index, first_error = failures[0]
            raise ModelEvaluationError(
                f"All {self.sample_size} model evaluations failed; "
                f"row {first_index}: {first_error}"
            ) from first_error
        if failures:
            logger.warning(
                "%d of %d model evaluations failed; failed rows contain NaN.",
                len(failures),
                self.sample_size,
            )

        output_names = {item["name"] for item in self.quantities_of_interest}
        output_names.update(key for row in rows for key in row)
        self.results = {
            name: np.asarray([row.get(name, np.nan) for row in rows])
            for name in sorted(output_names)
        }

    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """Evaluate a ``(design_variables, samples)`` input matrix."""

        model = self._require_model()
        matrix = np.asarray(x_matrix)
        if matrix.ndim != 2:
            raise ValueError(
                f"x_matrix must be two-dimensional, got shape {matrix.shape}."
            )
        expected_rows = len(self.design_variables)
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"x_matrix has {matrix.shape[0]} rows; expected {expected_rows}."
            )
        sample_count = matrix.shape[1]
        if sample_count == 0:
            raise ValueError("x_matrix must contain at least one sample.")
        output_names = [
            output["name"] for output in self.quantities_of_interest
        ]
        inputs = {
            variable["name"]: matrix[index, :]
            for index, variable in enumerate(self.design_variables)
        }
        for parameter in self.parameters:
            inputs[parameter["name"]] = np.full(
                sample_count,
                parameter["value"],
            )

        try:
            vector_result = model(**inputs)
            return _result_matrix(vector_result, output_names, sample_count)
        except Exception as vector_error:
            logger.debug(
                "Vectorized evaluation failed; using scalar fallback.",
                exc_info=True,
            )
            result = np.full((len(output_names), sample_count), np.nan)
            failures = 0
            for column in range(sample_count):
                row_input = {
                    name: values[column] for name, values in inputs.items()
                }
                try:
                    row = model(**row_input)
                    if not isinstance(row, Mapping):
                        raise TypeError("Model result is not a mapping.")
                    for output_index, output_name in enumerate(output_names):
                        result[output_index, column] = row.get(output_name, np.nan)
                except Exception:
                    failures += 1
            if failures == sample_count:
                raise ModelEvaluationError(
                    f"All {sample_count} model evaluations failed after "
                    f"vectorized evaluation failed: {vector_error}"
                ) from vector_error
            if failures:
                logger.warning(
                    "%d of %d scalar fallback evaluations failed after: %s",
                    failures,
                    sample_count,
                    vector_error,
                )
            return result

    def validate_unit_compatibility(
        self,
        output_unit: str,
        input_unit: str,
    ) -> bool:
        try:
            return units_compatible(output_unit, input_unit)
        except UnitError:
            return False

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        return float(convert_value(value, from_unit, to_unit))

    def get_common_unit(self, unit1: str, unit2: str) -> str:
        if not units_compatible(unit1, unit2):
            raise UnitError(f"Incompatible units: {unit1!r} and {unit2!r}.")
        return unit1

    def export_to_mat(self, filename: str | PathLike[str]) -> None:
        data = {
            "design_variables": self.design_variables,
            "parameters": self.parameters,
            "quantities_of_interest": self.quantities_of_interest,
            "samples": self.samples,
            "results": self.results,
        }
        scipy.io.savemat(filename, data)
        logger.info("Exported XRay data to %s", filename)

    def _ensure_unique_input(self, name: str) -> None:
        _validate_name(name, "model input")
        existing = {
            item["name"] for item in self.design_variables + self.parameters
        }
        if name in existing:
            raise ValueError(f"Duplicate model input: {name!r}.")

    def _require_model(self) -> Callable[..., Mapping[str, Any]]:
        if self.system_model is None:
            raise ValueError("System model is not set.")
        return self.system_model


def _result_matrix(
    result: Mapping[str, Any],
    output_names: list[str],
    sample_count: int,
) -> np.ndarray:
    if not isinstance(result, Mapping):
        raise TypeError("Model result is not a mapping.")
    matrix = np.full((len(output_names), sample_count), np.nan)
    for index, name in enumerate(output_names):
        if name not in result:
            continue
        values = np.asarray(result[name])
        if values.ndim == 0:
            matrix[index, :] = values.item()
        elif values.size == sample_count:
            matrix[index, :] = values.reshape(sample_count)
        else:
            raise ValueError(
                f"Output {name!r} has {values.size} values; "
                f"expected {sample_count}."
            )
    return matrix


def _validate_name(name: str, kind: str) -> None:
    if (
        not isinstance(name, str)
        or not name.isidentifier()
        or keyword.iskeyword(name)
    ):
        raise ValueError(f"{kind.capitalize()} name {name!r} is not valid Python.")


def _number(
    value: int | float | str,
    context: str,
    *,
    allow_infinite: bool = False,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, got {value!r}.") from exc
    if math.isnan(number) or (not allow_infinite and not math.isfinite(number)):
        raise ValueError(f"{context} must be finite, got {value!r}.")
    return number


def _finite_float(value: int | float | str, context: str) -> float:
    return _number(value, context)


def _bounds(
    lower: int | float | str,
    upper: int | float | str,
    context: str,
    *,
    allow_infinite: bool = False,
) -> tuple[float, float]:
    minimum = _number(
        lower,
        f"minimum for {context}",
        allow_infinite=allow_infinite,
    )
    maximum = _number(
        upper,
        f"maximum for {context}",
        allow_infinite=allow_infinite,
    )
    if minimum > maximum:
        raise ValueError(
            f"Minimum {minimum} exceeds maximum {maximum} for {context}."
        )
    return minimum, maximum


def _validate_unit(unit: str) -> None:
    if not isinstance(unit, str):
        raise TypeError("Unit must be a string.")
    if is_specified_unit(unit):
        units_compatible(unit, unit)


# Compatibility for projects and integrations created before the domain name
# was clarified. New code should use ``DesignProblem``.
XRayProblem = DesignProblem


__all__ = ["DesignProblem", "ModelEvaluationError", "XRayProblem"]
