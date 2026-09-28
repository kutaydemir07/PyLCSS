# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""SALib-specific samplers and estimators used by the public analyzer."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from ._validation import (
    confidence_options,
    normalize_problem,
    positive_int,
    salib_analysis_seed,
    validate_fraction,
    validate_response,
    validate_samples,
    validate_seed,
)
from .types import (
    DeltaResult,
    FastResult,
    FloatArray,
    MorrisResult,
    SalibProblem,
    SensitivityMethod,
    SobolResult,
)

_SALIB_IMPORT_ERROR: ImportError | None = None
try:
    from SALib.analyze import morris as morris_analyzer
    from SALib.analyze import sobol as sobol_analyzer
    from SALib.sample import morris as morris_sampler
    from SALib.sample import sobol as sobol_sampler
except ImportError as exc:
    _SALIB_IMPORT_ERROR = exc
    SALIB_AVAILABLE = False
else:
    SALIB_AVAILABLE = True

try:
    from SALib.analyze import fast as fast_analyzer
    from SALib.sample import fast_sampler
except ImportError:
    FAST_AVAILABLE = False
else:
    FAST_AVAILABLE = SALIB_AVAILABLE

try:
    from SALib.analyze import delta as delta_analyzer
    from SALib.sample import latin as latin_sampler
except ImportError:
    DELTA_AVAILABLE = False
else:
    DELTA_AVAILABLE = SALIB_AVAILABLE


class SalibBackend:
    """Validated SALib operations inherited by :class:`SensitivityAnalyzer`."""

    METHODS: ClassVar[tuple[SensitivityMethod, ...]] = (
        "Sobol",
        "Morris",
        "FAST",
        "Delta",
    )

    def __init__(self) -> None:
        if not SALIB_AVAILABLE:
            raise ImportError(
                "SALib is required for sensitivity analysis. Install the "
                "project dependencies or run: pip install SALib"
            ) from _SALIB_IMPORT_ERROR

    @staticmethod
    def available_methods() -> list[SensitivityMethod]:
        """Return methods supported by the installed SALib version."""
        if not SALIB_AVAILABLE:
            return []
        methods: list[SensitivityMethod] = ["Sobol", "Morris"]
        if FAST_AVAILABLE:
            methods.append("FAST")
        if DELTA_AVAILABLE:
            methods.append("Delta")
        return methods

    @staticmethod
    def normalize_problem(problem_definition: Mapping[str, Any]) -> SalibProblem:
        """Validate and normalize a public problem definition."""
        return normalize_problem(problem_definition)

    def generate_morris_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_trajectories: int = 20,
        *,
        num_levels: int = 4,
        seed: int | None = None,
    ) -> tuple[FloatArray, SalibProblem]:
        """Generate Morris elementary-effects trajectories."""
        problem = normalize_problem(problem_definition)
        trajectories = positive_int(
            n_trajectories,
            "Morris trajectory count",
            minimum=4,
        )
        levels = positive_int(num_levels, "Morris grid levels", minimum=4)
        if levels % 2:
            raise ValueError("Morris grid levels must be an even integer.")
        samples = morris_sampler.sample(
            problem,
            trajectories,
            num_levels=levels,
            seed=validate_seed(seed),
        )
        return np.asarray(samples, dtype=float), problem

    def analyze_morris(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem_definition: Mapping[str, Any],
        *,
        threshold: float | None = None,
        num_levels: int = 4,
        confidence_level: float = 0.95,
        resamples: int = 100,
        seed: int | None = None,
    ) -> tuple[list[str], MorrisResult]:
        """Analyze Morris effects and return selected variables plus all metrics.

        With no ``threshold``, the first return value contains every variable in
        descending ``mu_star`` order. With a threshold, it contains variables
        whose ``mu_star`` exceeds that fraction of the largest finite effect.
        """
        problem = normalize_problem(problem_definition)
        levels = positive_int(num_levels, "Morris grid levels", minimum=4)
        if levels % 2:
            raise ValueError("Morris grid levels must be an even integer.")
        x_values = validate_samples(samples, problem)
        trajectory_width = problem["num_vars"] + 1
        if x_values.shape[0] % trajectory_width:
            raise ValueError(
                "Morris sample rows must be a multiple of D + 1 "
                f"({trajectory_width} for this problem)."
            )
        if x_values.shape[0] // trajectory_width < 4:
            raise ValueError("Morris analysis needs at least four trajectories.")
        y_values = validate_response(response, expected_rows=x_values.shape[0])
        confidence, bootstrap_count = confidence_options(
            confidence_level,
            resamples,
        )

        indices = morris_analyzer.analyze(
            problem,
            x_values,
            y_values,
            num_resamples=bootstrap_count,
            conf_level=confidence,
            print_to_console=False,
            num_levels=levels,
            seed=salib_analysis_seed(seed),
        )
        mu = np.asarray(indices["mu"], dtype=float)
        mu_star = np.asarray(indices["mu_star"], dtype=float)
        sigma = np.asarray(indices["sigma"], dtype=float)
        mu_star_conf = np.asarray(indices["mu_star_conf"], dtype=float)

        finite_effects = np.where(np.isfinite(mu_star), mu_star, -np.inf)
        ranked_indices = np.argsort(-finite_effects, kind="stable")
        ranked_variables = [problem["names"][int(i)] for i in ranked_indices]

        selection_threshold: float | None = None
        selected_variables = ranked_variables
        if threshold is not None:
            selection_threshold = validate_fraction(
                threshold,
                "Morris screening threshold",
            )
            largest_effect = (
                float(np.max(finite_effects)) if np.any(np.isfinite(mu_star)) else 0.0
            )
            cutoff = selection_threshold * max(0.0, largest_effect)
            selected_variables = [
                problem["names"][int(i)]
                for i in ranked_indices
                if np.isfinite(mu_star[int(i)]) and mu_star[int(i)] > cutoff
            ]

        results: MorrisResult = {
            "variable_names": problem["names"],
            "mu": mu.tolist(),
            "mu_star": mu_star.tolist(),
            "sigma": sigma.tolist(),
            "mu_star_conf": mu_star_conf.tolist(),
            # Compatibility key: this is a ranking when no threshold is supplied.
            "important_variables": selected_variables,
            "ranked_variables": ranked_variables,
            "selection_threshold": selection_threshold,
            "method": "Morris",
        }
        return selected_variables, results

    def generate_sobol_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_samples: int = 1024,
        *,
        calc_second_order: bool = True,
        seed: int | None = None,
    ) -> FloatArray:
        """Generate a scrambled Saltelli extension of a Sobol sequence."""
        problem = normalize_problem(problem_definition)
        base_size = positive_int(n_samples, "Sobol base sample size", minimum=2)
        if base_size & (base_size - 1):
            raise ValueError(
                "Sobol base sample size must be a power of two (2, 4, 8, ...)."
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
        s2_matrix = _symmetric_second_order_matrix(
            second_order,
            problem["num_vars"],
        )

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
            "s2_matrix": s2_matrix,
            "calc_second_order": bool(calc_second_order),
            "method": "Sobol",
        }

    def generate_fast_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_samples: int = 1024,
        *,
        harmonics: int = 4,
        seed: int | None = None,
    ) -> FloatArray:
        """Generate extended FAST samples."""
        if not FAST_AVAILABLE:
            raise ImportError("The installed SALib version has no FAST sampler.")
        problem = normalize_problem(problem_definition)
        harmonic_count = positive_int(harmonics, "FAST harmonics")
        per_variable_size = positive_int(
            n_samples,
            "FAST sample size",
            minimum=4 * harmonic_count**2 + 1,
        )
        samples = fast_sampler.sample(
            problem,
            per_variable_size,
            M=harmonic_count,
            seed=validate_seed(seed),
        )
        return np.asarray(samples, dtype=float)

    def analyze_fast(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem_definition: Mapping[str, Any],
        *,
        harmonics: int = 4,
        confidence_level: float = 0.95,
        resamples: int = 100,
        seed: int | None = None,
    ) -> FastResult:
        """Estimate extended FAST first- and total-order effects."""
        if not FAST_AVAILABLE:
            raise ImportError("The installed SALib version has no FAST analyzer.")
        problem = normalize_problem(problem_definition)
        harmonic_count = positive_int(harmonics, "FAST harmonics")
        x_values = validate_samples(samples, problem)
        if x_values.shape[0] % problem["num_vars"]:
            raise ValueError("FAST sample rows must be a multiple of D.")
        per_variable_size = x_values.shape[0] // problem["num_vars"]
        minimum_size = 4 * harmonic_count**2 + 1
        if per_variable_size < minimum_size:
            raise ValueError(
                f"FAST needs at least {minimum_size} samples per variable "
                f"for harmonics={harmonic_count}."
            )
        y_values = validate_response(response, expected_rows=x_values.shape[0])
        confidence, bootstrap_count = confidence_options(
            confidence_level,
            resamples,
        )
        indices = fast_analyzer.analyze(
            problem,
            y_values,
            M=harmonic_count,
            num_resamples=bootstrap_count,
            conf_level=confidence,
            print_to_console=False,
            seed=salib_analysis_seed(seed),
        )
        return {
            "variable_names": problem["names"],
            "first_order": np.asarray(indices["S1"], dtype=float),
            "total_order": np.asarray(indices["ST"], dtype=float),
            "confidence_first": np.asarray(indices["S1_conf"], dtype=float),
            "confidence_total": np.asarray(indices["ST_conf"], dtype=float),
            "method": "FAST",
        }

    def generate_delta_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_samples: int = 1024,
        *,
        seed: int | None = None,
    ) -> FloatArray:
        """Generate Latin-hypercube samples for Delta analysis."""
        if not DELTA_AVAILABLE:
            raise ImportError("The installed SALib version has no Delta analyzer.")
        problem = normalize_problem(problem_definition)
        sample_count = positive_int(n_samples, "Delta sample size", minimum=2)
        samples = latin_sampler.sample(
            problem,
            sample_count,
            seed=validate_seed(seed),
        )
        return np.asarray(samples, dtype=float)

    def analyze_delta(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem_definition: Mapping[str, Any],
        *,
        confidence_level: float = 0.95,
        resamples: int = 100,
        response_resamples: int | None = None,
        seed: int | None = None,
    ) -> DeltaResult:
        """Estimate Delta moment-independent and first-order effects."""
        if not DELTA_AVAILABLE:
            raise ImportError("The installed SALib version has no Delta analyzer.")
        problem = normalize_problem(problem_definition)
        x_values = validate_samples(samples, problem)
        y_values = validate_response(response, expected_rows=x_values.shape[0])
        confidence, bootstrap_count = confidence_options(
            confidence_level,
            resamples,
        )
        y_resamples = (
            None
            if response_resamples is None
            else positive_int(response_resamples, "Delta response resamples")
        )
        if y_resamples is not None and y_resamples > y_values.size:
            raise ValueError("Delta response resamples cannot exceed the sample count.")
        indices = delta_analyzer.analyze(
            problem,
            x_values,
            y_values,
            num_resamples=bootstrap_count,
            conf_level=confidence,
            print_to_console=False,
            seed=salib_analysis_seed(seed),
            y_resamples=y_resamples,
        )
        return {
            "variable_names": problem["names"],
            "delta": np.asarray(indices["delta"], dtype=float),
            "delta_conf": np.asarray(indices["delta_conf"], dtype=float),
            "S1": np.asarray(indices["S1"], dtype=float),
            "S1_conf": np.asarray(indices["S1_conf"], dtype=float),
            "method": "Delta",
        }


def _symmetric_second_order_matrix(
    raw_values: object,
    variable_count: int,
) -> FloatArray:
    """Convert SALib's upper-triangular S2 output to a symmetric plot matrix."""
    matrix = np.zeros((variable_count, variable_count), dtype=float)
    if raw_values is None:
        return matrix

    raw = np.asarray(raw_values, dtype=float)
    if raw.shape != matrix.shape:
        raise RuntimeError(
            f"SALib returned S2 shape {raw.shape}; expected {matrix.shape}."
        )
    for first in range(variable_count):
        for second in range(first + 1, variable_count):
            value = raw[first, second]
            if not np.isfinite(value):
                value = raw[second, first]
            if np.isfinite(value):
                matrix[first, second] = float(value)
                matrix[second, first] = float(value)
    return matrix


__all__ = [
    "DELTA_AVAILABLE",
    "FAST_AVAILABLE",
    "SALIB_AVAILABLE",
    "SalibBackend",
]
