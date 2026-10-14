# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validated SALib sampling and estimation for Sobol sensitivity analysis."""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import (
    confidence_options,
    normalize_problem,
    salib_analysis_seed,
    sobol_sample_size,
    validate_response,
    validate_samples,
    validate_seed,
)
from .types import FloatArray, SalibProblem, SobolResult

_SALIB_IMPORT_ERROR: ImportError | None = None
try:
    from SALib.analyze import sobol as sobol_analyzer
    from SALib.sample import sobol as sobol_sampler
except ImportError as exc:
    _SALIB_IMPORT_ERROR = exc
    SALIB_AVAILABLE = False
else:
    SALIB_AVAILABLE = True


class SalibBackend:
    """Sobol operations inherited by :class:`SensitivityAnalyzer`."""

    def __init__(self) -> None:
        if not SALIB_AVAILABLE:
            raise ImportError(
                "SALib is required for sensitivity analysis. Install the "
                "project dependencies or run: pip install SALib"
            ) from _SALIB_IMPORT_ERROR

    @staticmethod
    def available_methods() -> list[Literal["Sobol"]]:
        """Return Sobol when its SALib components are available."""
        return ["Sobol"] if SALIB_AVAILABLE else []

    @staticmethod
    def normalize_problem(problem_definition: Mapping[str, Any]) -> SalibProblem:
        """Validate and normalize an independent-uniform problem definition."""
        return normalize_problem(problem_definition)

    def generate_sobol_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_samples: int = 1024,
        *,
        calc_second_order: bool = True,
        seed: int | None = None,
        allow_non_power_of_two: bool = False,
    ) -> FloatArray:
        """Generate a scrambled Saltelli extension of a Sobol sequence.

        Non-power-of-two sizes require explicit opt-in; SALib's sampling-balance
        warning remains visible when that option is used.
        """
        problem = normalize_problem(problem_definition)
        base_size = sobol_sample_size(
            n_samples,
            allow_non_power_of_two=allow_non_power_of_two,
        )
        samples = sobol_sampler.sample(
            problem,
            base_size,
            calc_second_order=bool(calc_second_order),
            scramble=True,
            seed=validate_seed(seed),
        )
        return np.asarray(samples, dtype=float)

    def analyze_sobol(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem_definition: Mapping[str, Any],
        *,
        calc_second_order: bool = True,
        confidence_level: float = 0.95,
        resamples: int = 100,
        seed: int | None = None,
    ) -> SobolResult:
        """Estimate first-, total-, and optional second-order Sobol indices."""
        problem = normalize_problem(problem_definition)
        x_values = validate_samples(samples, problem)
        row_factor = (
            2 * problem["num_vars"] + 2
            if calc_second_order
            else problem["num_vars"] + 2
        )
        if x_values.shape[0] % row_factor:
            raise ValueError(
                "Sobol sample rows are incompatible with "
                f"calc_second_order={bool(calc_second_order)}; expected a "
                f"multiple of {row_factor}."
            )
        if x_values.shape[0] // row_factor < 2:
            raise ValueError("Sobol analysis needs a base sample size of at least 2.")
        y_values = validate_response(response, expected_rows=x_values.shape[0])
        # Keep SALib's variance calculation finite across different output units.
        y_values = y_values / np.max(np.abs(y_values))
        confidence, bootstrap_count = confidence_options(
            confidence_level,
            resamples,
        )

        indices = sobol_analyzer.analyze(
            problem,
            y_values,
            calc_second_order=bool(calc_second_order),
            num_resamples=bootstrap_count,
            conf_level=confidence,
            print_to_console=False,
            seed=salib_analysis_seed(seed),
        )
        second_order = indices.get("S2")
        second_order_confidence = indices.get("S2_conf")
        return {
            "variable_names": problem["names"],
            "first_order": np.asarray(indices["S1"], dtype=float),
            "total_order": np.asarray(indices["ST"], dtype=float),
            "second_order": (
                None if second_order is None else np.asarray(second_order, dtype=float)
            ),
            "confidence_first": np.asarray(indices["S1_conf"], dtype=float),
            "confidence_total": np.asarray(indices["ST_conf"], dtype=float),
            "confidence_second": (
                None
                if second_order_confidence is None
                else np.asarray(second_order_confidence, dtype=float)
            ),
            "calc_second_order": bool(calc_second_order),
            "method": "Sobol",
        }


__all__ = ["SalibBackend"]
