# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Background workers for sensitivity-analysis UI operations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from PySide6 import QtCore

from pylcss.optimization.evaluator import ModelEvaluator
from pylcss.optimization.models import Variable
from pylcss.sensitivity import SensitivityAnalyzer

__all__ = ["OutputRefreshWorker", "SensitivityWorker"]

logger = logging.getLogger(__name__)


class SensitivityProblem(Protocol):
    """Subset of the optimization problem consumed by the UI workers."""

    design_variables: Sequence[Mapping[str, Any]]
    parameters: Sequence[Mapping[str, Any]]
    quantities_of_interest: Sequence[Mapping[str, Any]]
    system_model: Callable[..., Mapping[str, Any]]


class OutputRefreshWorker(QtCore.QThread):
    """Probe an optimization problem for output names without blocking Qt."""

    done_sig = QtCore.Signal(object)
    error_sig = QtCore.Signal(str)

    def __init__(
        self,
        problem: SensitivityProblem,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.problem = problem

    def run(self) -> None:
        try:
            sample_inputs = {
                str(variable["name"]): (float(variable["min"]) + float(variable["max"]))
                / 2.0
                for variable in self.problem.design_variables
            }
            sample_inputs.update(
                {
                    str(parameter["name"]): parameter["value"]
                    for parameter in self.problem.parameters
                }
            )
            sample_output = self.problem.system_model(**sample_inputs)
            if not isinstance(sample_output, Mapping):
                raise TypeError("The system model must return a mapping of outputs.")
            self.done_sig.emit(list(sample_output))
        except Exception as exc:
            logger.exception("Could not probe the system model outputs.")
            self.error_sig.emit(str(exc))


class SensitivityWorker(QtCore.QThread):
    """Run Sobol, Morris, FAST, or Delta analysis away from the GUI thread."""

    progress_sig = QtCore.Signal(int, str)
    done_sig = QtCore.Signal(object, object)

    def __init__(
        self,
        problem: SensitivityProblem,
        output_name: str,
        n_samples: int,
        method: str = "Sobol",
        *,
        n_trajectories: int = 20,
        morris_levels: int = 4,
        calc_second_order: bool = False,
        random_seed: int = 42,
        batch: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.problem = problem
        self.output_name = str(output_name)
        self.n_samples = int(n_samples)
        self.method = str(method)
        self.n_trajectories = int(n_trajectories)
        self.morris_levels = int(morris_levels)
        self.calc_second_order = bool(calc_second_order)
        self.random_seed = int(random_seed)
        self.batch = bool(batch)

    def run(self) -> None:
        try:
            results = self._analyze()
        except Exception as exc:
            logger.exception("%s sensitivity analysis failed.", self.method)
            self.done_sig.emit(None, str(exc))
        else:
            self.done_sig.emit(results, None)

    def _analyze(self) -> dict[str, Any]:
        analyzer = SensitivityAnalyzer()
        evaluator = self._build_evaluator()
        problem_definition = self._problem_definition()
        output_names = self._declared_output_names()

        self.progress_sig.emit(0, f"Setting up {self.method} analysis...")
        samples, morris_problem = self._generate_samples(
            analyzer,
            problem_definition,
        )
        self.progress_sig.emit(30, "Evaluating system model...")
        output_values = self._evaluate_samples(
            evaluator,
            samples,
            output_names,
        )

        if self.isInterruptionRequested():
            raise RuntimeError("Sensitivity analysis cancelled.")
        self.progress_sig.emit(80, "Analyzing sensitivity...")

        if self.batch:
            result = analyzer.batch_analyze(
                samples,
                output_values,
                problem_definition,
                self.method,
                calc_second_order=self.calc_second_order,
                seed=self.random_seed,
                morris_num_levels=self.morris_levels,
            )
            self.progress_sig.emit(100, "Batch analysis complete.")
            return {"batch_results": result}

        if self.output_name not in output_values:
            raise KeyError(
                f"Selected output {self.output_name!r} is not a declared "
                "quantity of interest."
            )
        values = output_values[self.output_name]
        if self.method == "Sobol":
            result = analyzer.analyze_sobol(
                samples,
                values,
                problem_definition,
                calc_second_order=self.calc_second_order,
                seed=self.random_seed,
            )
        elif self.method == "Morris":
            if morris_problem is None:
                raise RuntimeError("Morris sampling metadata was not produced.")
            _, result = analyzer.analyze_morris(
                samples,
                values,
                morris_problem,
                num_levels=self.morris_levels,
                seed=self.random_seed,
            )
        elif self.method == "FAST":
            result = analyzer.analyze_fast(
                samples,
                values,
                problem_definition,
                seed=self.random_seed,
            )
        elif self.method == "Delta":
            result = analyzer.analyze_delta(
                samples,
                values,
                problem_definition,
                seed=self.random_seed,
            )
        else:
            raise ValueError(f"Unsupported sensitivity method {self.method!r}.")

        self.progress_sig.emit(100, "Analysis complete.")
        return result

    def _build_evaluator(self) -> ModelEvaluator:
        variables = [
            Variable(
                name=str(variable["name"]),
                min_val=float(variable["min"]),
                max_val=float(variable["max"]),
            )
            for variable in self.problem.design_variables
        ]
        parameters = {
            str(parameter["name"]): parameter["value"]
            for parameter in self.problem.parameters
        }
        return ModelEvaluator(
            self.problem.system_model,
            variables,
            [],
            [],
            parameters=parameters,
            scaling=False,
        )

    def _problem_definition(self) -> dict[str, Any]:
        if not self.problem.design_variables:
            raise ValueError("Sensitivity analysis needs at least one design variable.")
        names: list[str] = []
        bounds: list[list[float]] = []
        for variable in self.problem.design_variables:
            name = str(variable.get("name") or "").strip()
            if not name:
                raise ValueError("Every design variable needs a name.")
            lower = float(variable["min"])
            upper = float(variable["max"])
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(
                    f"Design variable {name!r} needs finite bounds with min < max."
                )
            names.append(name)
            bounds.append([lower, upper])
        if len(names) != len(set(names)):
            raise ValueError("Design variable names must be unique.")
        return {"names": names, "bounds": bounds}

    def _declared_output_names(self) -> list[str]:
        names = [
            str(item.get("name") or "").strip()
            for item in self.problem.quantities_of_interest
        ]
        names = [name for name in names if name]
        if not names:
            raise ValueError(
                "The problem has no declared quantity-of-interest outputs."
            )
        if len(names) != len(set(names)):
            raise ValueError("Quantity-of-interest names must be unique.")
        return names

    def _generate_samples(
        self,
        analyzer: SensitivityAnalyzer,
        problem_definition: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        self.progress_sig.emit(10, "Generating samples...")
        if self.method == "Sobol":
            samples = analyzer.generate_sobol_samples(
                problem_definition,
                self.n_samples,
                calc_second_order=self.calc_second_order,
                seed=self.random_seed,
            )
            return np.asarray(samples, dtype=float), None
        if self.method == "Morris":
            samples, metadata = analyzer.generate_morris_samples(
                problem_definition,
                self.n_trajectories,
                num_levels=self.morris_levels,
                seed=self.random_seed,
            )
            return np.asarray(samples, dtype=float), metadata
        if self.method == "FAST":
            samples = analyzer.generate_fast_samples(
                problem_definition,
                self.n_samples,
                seed=self.random_seed,
            )
            return np.asarray(samples, dtype=float), None
        if self.method == "Delta":
            samples = analyzer.generate_delta_samples(
                problem_definition,
                self.n_samples,
                seed=self.random_seed,
            )
            return np.asarray(samples, dtype=float), None
        raise ValueError(f"Unsupported sensitivity method {self.method!r}.")

    def _evaluate_samples(
        self,
        evaluator: ModelEvaluator,
        samples: np.ndarray,
        output_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        total = len(samples)
        if total == 0:
            raise RuntimeError("The sensitivity sampler returned no points.")
        outputs = {name: np.empty(total, dtype=float) for name in output_names}
        for index, sample in enumerate(samples):
            if self.isInterruptionRequested():
                raise RuntimeError("Sensitivity analysis cancelled.")
            _, result, _ = evaluator.evaluate(np.asarray(sample, dtype=float))
            if not evaluator.is_valid_result(result) or not result:
                detail = evaluator.evaluation_error(result)
                raise RuntimeError(
                    f"System model failed at sensitivity sample "
                    f"{index + 1}/{total}. "
                    + (
                        detail
                        or "No value was imputed because that would invalidate "
                        f"the {self.method} sampling design."
                    )
                )
            for name in output_names:
                if name not in result:
                    raise RuntimeError(
                        f"System model omitted output {name!r} at sensitivity "
                        f"sample {index + 1}/{total}."
                    )
                try:
                    value = float(result[name])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Output {name!r} is not a scalar number at sensitivity "
                        f"sample {index + 1}/{total}."
                    ) from exc
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"Output {name!r} is not finite at sensitivity sample "
                        f"{index + 1}/{total}."
                    )
                outputs[name][index] = value

            if index % max(1, total // 20) == 0 or index == total - 1:
                progress = 30 + int(50 * (index + 1) / total)
                self.progress_sig.emit(
                    progress,
                    f"Evaluating samples... ({index + 1}/{total})",
                )
        return outputs
