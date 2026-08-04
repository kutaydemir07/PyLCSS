# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Runtime validation shared by all sensitivity-analysis methods."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray, SalibProblem, SensitivityMethod

_METHOD_BY_NAME: dict[str, SensitivityMethod] = {
    "sobol": "Sobol",
    "morris": "Morris",
    "fast": "FAST",
    "delta": "Delta",
}


def normalize_method(method: str) -> SensitivityMethod:
    """Return the canonical display name for a supported method."""
    if not isinstance(method, str) or not method.strip():
        raise TypeError("Sensitivity method must be a non-empty string.")
    try:
        return _METHOD_BY_NAME[method.strip().casefold()]
    except KeyError as exc:
        supported = ", ".join(_METHOD_BY_NAME.values())
        raise ValueError(
            f"Unsupported sensitivity method {method!r}; choose one of: {supported}."
        ) from exc


def normalize_problem(problem_definition: Mapping[str, Any]) -> SalibProblem:
    """Validate an independent-uniform problem and return SALib's mapping."""
    if not isinstance(problem_definition, Mapping):
        raise TypeError("Problem definition must be a mapping.")

    try:
        raw_names = problem_definition["names"]
        raw_bounds = problem_definition["bounds"]
    except KeyError as exc:
        raise ValueError(
            "Problem definition requires both 'names' and 'bounds'."
        ) from exc

    if isinstance(raw_names, (str, bytes)):
        raise TypeError("Problem 'names' must be a sequence of strings.")
    try:
        names = list(raw_names)
    except TypeError as exc:
        raise TypeError("Problem 'names' must be a sequence of strings.") from exc

    if not names:
        raise ValueError("Sensitivity analysis needs at least one design variable.")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Sensitivity variable name at position {index} must be non-empty."
            )
    if len(set(names)) != len(names):
        raise ValueError("Sensitivity variable names must be unique.")

    if isinstance(raw_bounds, (str, bytes)):
        raise TypeError("Problem 'bounds' must be a sequence of [min, max] pairs.")
    try:
        bounds = list(raw_bounds)
    except TypeError as exc:
        raise TypeError(
            "Problem 'bounds' must be a sequence of [min, max] pairs."
        ) from exc
    if len(bounds) != len(names):
        raise ValueError("Every sensitivity variable needs one [min, max] bound.")

    normalized_bounds: list[list[float]] = []
    for name, bound in zip(names, bounds):
        if isinstance(bound, (str, bytes)):
            raise ValueError(f"Variable {name!r} needs a [min, max] bound.")
        try:
            pair = list(bound)
        except TypeError as exc:
            raise ValueError(f"Variable {name!r} needs a [min, max] bound.") from exc
        if len(pair) != 2:
            raise ValueError(f"Variable {name!r} needs a [min, max] bound.")
        try:
            lower, upper = float(pair[0]), float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Variable {name!r} bounds must be numeric.") from exc
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"Variable {name!r} must have finite bounds with min < max."
            )
        normalized_bounds.append([lower, upper])

    declared_count = problem_definition.get("num_vars")
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, Integral)
        or int(declared_count) != len(names)
    ):
        raise ValueError("Problem 'num_vars' must equal the number of variable names.")

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": normalized_bounds,
    }


def positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate an integer option without silently truncating floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def validate_seed(seed: int | None) -> int | None:
    """Validate a seed supported by NumPy and SALib."""
    if seed is None:
        return None
    normalized = positive_int(seed, "Random seed", minimum=0)
    if normalized > np.iinfo(np.uint32).max:
        raise ValueError("Random seed must fit in an unsigned 32-bit integer.")
    return normalized


def salib_analysis_seed(seed: int | None) -> int | None:
    """Work around SALib treating the valid seed zero as an unseeded analysis."""
    normalized = validate_seed(seed)
    if normalized == 0:
        return int(np.iinfo(np.uint32).max)
    return normalized


def confidence_options(
    confidence_level: float,
    resamples: int,
) -> tuple[float, int]:
    """Validate bootstrap confidence settings used by SALib analyzers."""
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise TypeError("Confidence level must be a number.") from exc
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Confidence level must be strictly between 0 and 1.")
    return confidence, positive_int(resamples, "Bootstrap resamples", minimum=2)


def validate_samples(
    samples: ArrayLike,
    problem: SalibProblem,
    *,
    minimum_rows: int = 2,
) -> FloatArray:
    """Return a finite two-dimensional sample matrix with the expected width."""
    try:
        values = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sensitivity samples must be numeric.") from exc
    if values.ndim != 2:
        raise ValueError("Sensitivity samples must be a two-dimensional matrix.")
    if values.shape[1] != problem["num_vars"]:
        raise ValueError(
            "Sensitivity sample matrix has "
            f"{values.shape[1]} columns; {problem['num_vars']} were expected."
        )
    if values.shape[0] < minimum_rows:
        raise ValueError(
            f"Sensitivity sample matrix needs at least {minimum_rows} rows."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Sensitivity samples must contain only finite values.")

    bounds = np.asarray(problem["bounds"], dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    tolerance = (
        np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
        * np.finfo(float).eps
        * 64.0
    )
    outside = (values < lower - tolerance) | (values > upper + tolerance)
    if np.any(outside):
        row, column = np.argwhere(outside)[0]
        raise ValueError(
            f"Sample row {int(row)} for variable "
            f"{problem['names'][int(column)]!r} lies outside its declared bounds."
        )
    return values


def validate_response(
    response: ArrayLike,
    *,
    expected_rows: int,
) -> FloatArray:
    """Return one finite, non-constant scalar response per sample row."""
    try:
        values = np.asarray(response, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sensitivity outputs must be numeric scalar values.") from exc

    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            "Sensitivity outputs must be a one-dimensional array with one "
            "scalar value per sample."
        )
    if values.size != expected_rows:
        raise ValueError(
            f"Sensitivity response has {values.size} values; "
            f"{expected_rows} sample rows were expected."
        )
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Sensitivity outputs must be finite scalar values.")

    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.ptp(values)) <= np.finfo(float).eps * scale * 32.0:
        raise ValueError(
            "The selected output is constant over the sampled design space; "
            "sensitivity indices are undefined."
        )
    return values


def validate_fraction(value: float, name: str) -> float:
    """Validate a finite fraction in the closed interval [0, 1]."""
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number.") from exc
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return fraction
