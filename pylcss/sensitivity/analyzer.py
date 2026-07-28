# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public orchestration API for global sensitivity analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from logging import getLogger
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._salib import (
    DELTA_AVAILABLE,
    FAST_AVAILABLE,
    SALIB_AVAILABLE,
    SalibBackend,
)
from ._validation import (
    normalize_method,
    normalize_problem,
    positive_int,
    validate_seed,
)
from .types import (
    AnalysisError,
    BatchResult,
    ConvergenceResult,
    FloatArray,
    MorrisResult,
    PlotData,
    SalibProblem,
    SobolResult,
)

logger = getLogger(__name__)


class SensitivityAnalyzer(SalibBackend):
    """Global sensitivity-analysis facade.

    The inherited method-specific API generates and analyzes Sobol, Morris,
    FAST, and Delta designs. This layer coordinates multi-output analysis,
    convergence studies, result ranking, and compatibility with older callers.
    """

    def batch_analyze(
        self,
        samples: ArrayLike,
        responses: Mapping[str, ArrayLike],
        problem_definition: Mapping[str, Any],
        method: str = "Sobol",
        *,
        calc_second_order: bool = True,
        seed: int | None = None,
        morris_num_levels: int = 4,
        strict: bool = False,
    ) -> BatchResult:
        """Analyze several scalar outputs against one shared sample design.

        By default, one bad output is isolated and represented by a structured
        error while other outputs continue. Set ``strict=True`` to fail fast.
        """
        canonical_method = normalize_method(method)
        if canonical_method not in self.available_methods():
            raise ImportError(
                f"{canonical_method} is unavailable in the installed SALib version."
            )
        if not isinstance(responses, Mapping):
            raise TypeError("Batch responses must map output names to arrays.")

        results: BatchResult = {}
        for output_name, response in responses.items():
            if not isinstance(output_name, str) or not output_name:
                raise ValueError("Batch output names must be non-empty strings.")
            try:
                if canonical_method == "Sobol":
                    result = self.analyze_sobol(
                        samples,
                        response,
                        problem_definition,
                        calc_second_order=calc_second_order,
                        seed=seed,
                    )
                elif canonical_method == "Morris":
                    _, result = self.analyze_morris(
                        samples,
                        response,
                        problem_definition,
                        num_levels=morris_num_levels,
                        seed=seed,
                    )
                elif canonical_method == "FAST":
                    result = self.analyze_fast(
                        samples,
                        response,
                        problem_definition,
                        seed=seed,
                    )
                else:
                    result = self.analyze_delta(
                        samples,
                        response,
                        problem_definition,
                        seed=seed,
                    )
            except Exception as exc:
                if strict:
                    raise
                logger.exception(
                    "%s sensitivity analysis failed for output %r",
                    canonical_method,
                    output_name,
                )
                error: AnalysisError = {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "method": canonical_method,
                }
                results[output_name] = error
            else:
                results[output_name] = result
        return results

    def analyze_convergence(
        self,
        problem_definition: Mapping[str, Any],
        evaluate: Callable[[FloatArray], Mapping[str, Any]],
        output_name: str,
        sample_sizes: Sequence[int] | None = None,
        *,
        calc_second_order: bool = False,
        seed: int | None = None,
    ) -> ConvergenceResult:
        """Run comparable Sobol analyses over increasing base sample sizes."""
        problem = normalize_problem(problem_definition)
        if not callable(evaluate):
            raise TypeError("Convergence evaluator must be callable.")
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("Convergence output name must be a non-empty string.")

        requested_sizes = (
            [64, 128, 256, 512, 1024] if sample_sizes is None else list(sample_sizes)
        )
        if not requested_sizes:
            raise ValueError("Convergence analysis needs at least one sample size.")
        sizes = [
            positive_int(value, "Sobol base sample size", minimum=2)
            for value in requested_sizes
        ]
        if len(set(sizes)) != len(sizes):
            raise ValueError("Convergence sample sizes must be unique.")
        if sizes != sorted(sizes):
            raise ValueError("Convergence sample sizes must be increasing.")

        run_seed = validate_seed(seed)
        if run_seed is None:
            run_seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]
            )

        first_order_traces = np.empty((len(sizes), problem["num_vars"]))
        total_order_traces = np.empty_like(first_order_traces)
        for size_index, base_size in enumerate(sizes):
            samples = self.generate_sobol_samples(
                problem,
                base_size,
                calc_second_order=calc_second_order,
                seed=run_seed,
            )
            response = np.empty(samples.shape[0], dtype=float)
            for sample_index, sample in enumerate(samples):
                try:
                    model_output = evaluate(sample)
                    if not isinstance(model_output, Mapping):
                        raise TypeError("model result is not a mapping")
                    if output_name not in model_output:
                        raise KeyError(f"model output {output_name!r} is missing")
                    value = float(model_output[output_name])
                    if not np.isfinite(value):
                        raise ValueError(f"model output {output_name!r} is not finite")
                except Exception as exc:
                    raise RuntimeError(
                        "Sobol convergence analysis cannot discard or replace "
                        f"structured sample {sample_index + 1}/{len(samples)} "
                        f"at base size {base_size}: {exc}"
                    ) from exc
                response[sample_index] = value

            result = self.analyze_sobol(
                samples,
                response,
                problem,
                calc_second_order=calc_second_order,
                seed=run_seed,
            )
            first_order_traces[size_index] = result["first_order"]
            total_order_traces[size_index] = result["total_order"]

        return {
            "sample_sizes": sizes,
            "variable_names": problem["names"],
            "S1_traces": first_order_traces,
            "ST_traces": total_order_traces,
            "calc_second_order": bool(calc_second_order),
            "seed": run_seed,
            "method": "Sobol",
        }

    @staticmethod
    def rank_variables(
        results: Mapping[str, Any],
        metric: str = "total_order",
    ) -> list[dict[str, int | float | str]]:
        """Rank variables by a named one-dimensional sensitivity metric."""
        if "variable_names" not in results:
            raise ValueError("Sensitivity results have no 'variable_names'.")
        names = list(results["variable_names"])
        if metric not in results:
            raise ValueError(f"Sensitivity results have no metric {metric!r}.")
        values = np.asarray(results[metric], dtype=float)
        if values.ndim != 1 or values.size != len(names):
            raise ValueError(
                f"Sensitivity metric {metric!r} must contain one value per variable."
            )

        sortable = np.where(np.isfinite(values), values, -np.inf)
        order = np.argsort(-sortable, kind="stable")
        return [
            {
                "name": str(names[int(index)]),
                "index": float(values[int(index)]),
                "rank": rank,
            }
            for rank, index in enumerate(order, start=1)
        ]

    @staticmethod
    def build_plot_data(
        results: Mapping[str, Any],
        output_name: str = "Output",
    ) -> PlotData:
        """Build the common data mapping consumed by chart adapters."""
        if "variable_names" not in results:
            raise ValueError("Sensitivity results have no 'variable_names'.")
        return {
            "variables": list(results["variable_names"]),
            "first_order": results.get("first_order", []),
            "total_order": results.get("total_order", []),
            "confidence": results.get("confidence_total"),
            "output_name": output_name,
            "method": results.get("method", "Sobol"),
        }

    # Compatibility aliases for the pre-2.2 public API.
    def run_screening(
        self,
        problem_definition: Mapping[str, Any],
        n_trajectories: int = 20,
        num_levels: int = 4,
        seed: int | None = None,
    ) -> tuple[FloatArray, SalibProblem]:
        return self.generate_morris_samples(
            problem_definition,
            n_trajectories,
            num_levels=num_levels,
            seed=seed,
        )

    def analyze_screening(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem: Mapping[str, Any],
        threshold_pct: float | None = None,
        num_levels: int = 4,
        seed: int | None = None,
    ) -> tuple[list[str], MorrisResult]:
        return self.analyze_morris(
            samples,
            response,
            problem,
            threshold=threshold_pct,
            num_levels=num_levels,
            seed=seed,
        )

    def generate_samples(
        self,
        problem_definition: Mapping[str, Any],
        n_samples: int = 1024,
        calc_second_order: bool = True,
        seed: int | None = None,
    ) -> FloatArray:
        return self.generate_sobol_samples(
            problem_definition,
            n_samples,
            calc_second_order=calc_second_order,
            seed=seed,
        )

    def analyze_sensitivity(
        self,
        samples: ArrayLike,
        response: ArrayLike,
        problem_definition: Mapping[str, Any],
        calc_second_order: bool = True,
        seed: int | None = None,
    ) -> SobolResult:
        return self.analyze_sobol(
            samples,
            response,
            problem_definition,
            calc_second_order=calc_second_order,
            seed=seed,
        )

    def convergence_analysis(
        self,
        problem_definition: Mapping[str, Any],
        evaluate_fn: Callable[[FloatArray], Mapping[str, Any]],
        output_name: str,
        sample_sizes: Sequence[int] | None = None,
    ) -> ConvergenceResult:
        return self.analyze_convergence(
            problem_definition,
            evaluate_fn,
            output_name,
            sample_sizes,
        )

    @staticmethod
    def plot_sensitivity_indices(
        results: Mapping[str, Any],
        output_name: str = "Output",
    ) -> PlotData:
        return SensitivityAnalyzer.build_plot_data(results, output_name)


__all__ = [
    "DELTA_AVAILABLE",
    "FAST_AVAILABLE",
    "SALIB_AVAILABLE",
    "SensitivityAnalyzer",
]
