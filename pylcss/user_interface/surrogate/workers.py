# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Qt workers used by the surrogate-training interface."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore

from pylcss.surrogate_modeling.training_engine import SurrogateTrainer
from pylcss.surrogate_modeling.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSelector,
)

__all__ = [
    "AdaptiveTrainingWorker",
    "DataGenerationWorker",
    "ModelTrainingWorker",
    "TrainingWorker",
]

logger = logging.getLogger(__name__)

SpyFunction = Callable[
    ...,
    tuple[Mapping[str, Any], Mapping[str, Any]],
]


class DataGenerationWorker(QtCore.QThread):
    """Generate surrogate train/test arrays outside the GUI thread."""

    progress_sig = QtCore.Signal(int, str)
    done_sig = QtCore.Signal(object, object)

    def __init__(
        self,
        spy_code: str,
        spy_inputs: Sequence[Any],
        spy_outputs: Sequence[Any],
        input_bounds: Sequence[tuple[float, float]],
        samples: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.spy_code = spy_code
        self.spy_inputs = list(spy_inputs)
        self.spy_outputs = list(spy_outputs)
        self.input_bounds = list(input_bounds)
        self.samples = int(samples)
        self.trainer = SurrogateTrainer()

    def run(self) -> None:
        try:
            arrays = self.trainer.generate_data(
                self.spy_code,
                self.spy_inputs,
                self.spy_outputs,
                self.input_bounds,
                self.samples,
                callback=lambda progress, message: self.progress_sig.emit(
                    progress,
                    message,
                ),
                stop_flag=self.isInterruptionRequested,
            )
            train_X, train_y, test_X, test_y, _, _ = arrays
        except Exception as exc:
            logger.exception("Surrogate data generation failed.")
            self.done_sig.emit(None, str(exc))
        else:
            self.done_sig.emit((train_X, train_y, test_X, test_y), None)


class ModelTrainingWorker(QtCore.QThread):
    """Train a surrogate from arrays already loaded by the user."""

    progress_sig = QtCore.Signal(int, str)
    loss_sig = QtCore.Signal(dict)
    done_sig = QtCore.Signal(object, object, object)

    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        config: Mapping[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.config = dict(config)
        self.trainer = SurrogateTrainer()
        self.stop_flag = False

    def run(self) -> None:
        try:
            model, metrics = self.trainer.train_model(
                self.train_X,
                self.train_y,
                self.config,
                self.test_X,
                self.test_y,
                callback=lambda progress, message: self.progress_sig.emit(
                    progress,
                    message,
                ),
                stop_flag=lambda: self.stop_flag,
                loss_callback=lambda data: self.loss_sig.emit(data),
            )
        except Exception as exc:
            logger.exception("Surrogate model training failed.")
            self.done_sig.emit(None, None, str(exc))
        else:
            self.done_sig.emit(model, metrics, None)


class TrainingWorker(QtCore.QThread):
    """Generate a small debug data set and train one surrogate model."""

    progress_sig = QtCore.Signal(int, str)
    loss_sig = QtCore.Signal(dict)
    done_sig = QtCore.Signal(object, object, object)

    def __init__(
        self,
        spy_code: str,
        spy_inputs: Sequence[Any],
        spy_outputs: Sequence[Any],
        input_bounds: Sequence[tuple[float, float]],
        samples: int,
        config: Mapping[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.spy_code = spy_code
        self.spy_inputs = list(spy_inputs)
        self.spy_outputs = list(spy_outputs)
        self.input_bounds = list(input_bounds)
        self.samples = int(samples)
        self.config = dict(config)
        self.trainer = SurrogateTrainer()
        self.stop_flag = False

    def run(self) -> None:
        try:
            self.progress_sig.emit(0, "Generating debug data...")
            train_X, train_y, test_X, test_y, _, _ = self.trainer.generate_data(
                self.spy_code,
                self.spy_inputs,
                self.spy_outputs,
                self.input_bounds,
                self.samples,
                callback=None,
                stop_flag=lambda: self.stop_flag,
            )
            if self.stop_flag:
                raise RuntimeError("Surrogate training was cancelled.")

            self.progress_sig.emit(20, "Training debug model...")
            model, metrics = self.trainer.train_model(
                train_X,
                train_y,
                self.config,
                test_X,
                test_y,
                callback=lambda progress, message: self.progress_sig.emit(
                    20 + int(progress * 0.8),
                    message,
                ),
                stop_flag=lambda: self.stop_flag,
                loss_callback=lambda data: self.loss_sig.emit(data),
            )
        except Exception as exc:
            logger.exception("Debug surrogate training failed.")
            self.done_sig.emit(None, None, str(exc))
        else:
            self.done_sig.emit(model, metrics, None)


class AdaptiveTrainingWorker(QtCore.QThread):
    """Perform deterministic uncertainty-driven adaptive sampling."""

    progress_sig = QtCore.Signal(int, str)
    done_sig = QtCore.Signal(object, object, object, object, object)

    def __init__(
        self,
        trainer: SurrogateTrainer,
        spy_code: str,
        spy_inputs: Sequence[Any],
        spy_outputs: Sequence[Any],
        bounds: Sequence[tuple[float, float]],
        initial_X: np.ndarray,
        initial_y: np.ndarray,
        test_X: np.ndarray | None,
        test_y: np.ndarray | None,
        config: Mapping[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.trainer = trainer
        self.spy_code = spy_code
        self.spy_inputs = list(spy_inputs)
        self.spy_outputs = list(spy_outputs)
        self.bounds = np.asarray(bounds, dtype=float)
        self.X = np.asarray(initial_X, dtype=float).copy()
        self.y = np.asarray(initial_y, dtype=float).copy()
        self.test_X = (
            np.asarray(test_X, dtype=float).copy() if test_X is not None else None
        )
        self.test_y = (
            np.asarray(test_y, dtype=float).copy() if test_y is not None else None
        )
        if len(self.spy_outputs) == 1 and self.y.ndim > 1:
            self.y = self.y.reshape(-1)
        if (
            self.test_y is not None
            and len(self.spy_outputs) == 1
            and self.test_y.ndim > 1
        ):
            self.test_y = self.test_y.reshape(-1)
        self.config = dict(config)
        self.stop_flag = False
        self.random_state = int(self.config.get("random_state", 42))

    def run(self) -> None:
        try:
            model, metrics = self._train_adaptively()
        except Exception as exc:
            logger.exception("Adaptive surrogate training failed.")
            self.done_sig.emit(None, None, None, None, str(exc))
        else:
            self.done_sig.emit(model, metrics, self.X, self.y, None)

    def _train_adaptively(self) -> tuple[Any, Mapping[str, Any]]:
        self._validate_inputs()
        active_config = ActiveLearningConfig.from_mapping(
            self.config.get("active_learning")
        )
        initial_sample_count = len(self.X)
        selector = ActiveLearningSelector(self.bounds, active_config)
        acquisition_sources: list[str] = []
        fallback_rounds = 0
        failed_evaluations: list[str] = []
        completed_rounds = 0

        for round_index in range(active_config.n_rounds):
            self._raise_if_cancelled()
            round_number = round_index + 1
            round_start = int(90 * round_index / active_config.n_rounds)
            round_end = int(90 * round_number / active_config.n_rounds)
            round_span = max(1, round_end - round_start)
            self.progress_sig.emit(
                round_start,
                f"Adaptive round {round_number}/{active_config.n_rounds}: "
                f"scoring {active_config.n_candidates} candidates...",
            )

            primary_model = None
            if active_config.strategy == "uncertainty":
                primary_model, _ = self.trainer.train_model(
                    self.X,
                    self.y,
                    self.config,
                    self.test_X,
                    self.test_y,
                    stop_flag=lambda: self.stop_flag,
                )
            selection = selector.select(
                self.X,
                self.y,
                primary_model=primary_model,
            )
            acquisition_sources.append(selection.acquisition_source)
            fallback_rounds += int(selection.fallback_used)

            self.progress_sig.emit(
                round_start + int(0.4 * round_span),
                f"Adaptive round {round_number}/{active_config.n_rounds}: "
                f"evaluating {len(selection.points)} selected simulations via "
                f"{selection.acquisition_source}...",
            )
            new_X, new_y, failures = self.trainer.evaluate_points(
                self.spy_code,
                self.spy_inputs,
                self.spy_outputs,
                selection.points,
                callback=lambda progress, message, start=round_start, span=round_span: (
                    self.progress_sig.emit(
                        start + int(span * (0.4 + 0.6 * progress / 100.0)),
                        message,
                    )
                ),
                stop_flag=lambda: self.stop_flag,
            )
            failed_evaluations.extend(failures)
            if len(new_X) == 0:
                if self.stop_flag:
                    break
                raise RuntimeError(
                    "Every selected simulation in the active-learning batch failed. "
                    "No fabricated response values were added."
                )
            if self.y.ndim == 2 and new_y.ndim == 1:
                new_y = new_y.reshape(-1, 1)
            elif self.y.ndim == 1 and new_y.ndim == 2:
                if new_y.shape[1] != 1:
                    raise RuntimeError(
                        "A multi-output response cannot be appended to single-output data."
                    )
                new_y = new_y.ravel()
            self.X = np.vstack([self.X, new_X])
            self.y = np.concatenate([self.y, new_y])
            completed_rounds += 1

        self._raise_if_cancelled()
        self.progress_sig.emit(95, "Final training on adaptive data...")
        model, metrics = self.trainer.train_model(
            self.X,
            self.y,
            self.config,
            self.test_X,
            self.test_y,
            stop_flag=lambda: self.stop_flag,
        )
        metrics = dict(metrics)
        metrics["active_learning"] = {
            **active_config.to_dict(),
            "completed_rounds": completed_rounds,
            "new_samples": int(len(self.X) - initial_sample_count),
            "failed_evaluations": len(failed_evaluations),
            "failure_messages": failed_evaluations[:20],
            "fallback_rounds": fallback_rounds,
            "acquisition_sources": acquisition_sources,
            "stopped_early": bool(self.stop_flag),
        }
        return model, metrics

    def _validate_inputs(self) -> None:
        if self.bounds.shape != (self.X.shape[1], 2):
            raise ValueError(
                "Adaptive training requires one finite bound pair per model input."
            )
        if not np.isfinite(self.bounds).all():
            raise ValueError("Adaptive-training bounds contain NaN or infinity.")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Adaptive-training bounds must satisfy lower < upper.")
        if len(self.X) != len(self.y):
            raise ValueError("Adaptive-training feature/target lengths differ.")
        if not self.spy_outputs:
            raise ValueError("The generated spy model exposes no outputs.")

    def _candidate_points(
        self,
        round_index: int,
        *,
        count: int,
    ) -> np.ndarray:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        try:
            from scipy.stats import qmc
        except ImportError:
            rng = np.random.default_rng(self.random_state + round_index)
            return lower + rng.random((count, len(lower))) * (upper - lower)

        sampler = qmc.LatinHypercube(
            d=len(lower),
            seed=self.random_state + round_index,
        )
        return np.asarray(
            qmc.scale(sampler.random(count), lower, upper),
            dtype=float,
        )

    def _evaluate_points(
        self,
        spy_model: SpyFunction,
        samples: np.ndarray,
    ) -> np.ndarray:
        results: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            self._raise_if_cancelled()
            try:
                _, output_mapping = spy_model(*sample)
                values = np.asarray(
                    [
                        output_mapping[f"output_{output_index}"]
                        for output_index in range(len(self.spy_outputs))
                    ],
                    dtype=float,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Generated system model failed at adaptive sample "
                    f"{index + 1}/{len(samples)}: {exc}"
                ) from exc
            if values.shape != (len(self.spy_outputs),):
                raise RuntimeError(
                    "Every adaptive spy-model output must be one scalar value."
                )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    "The generated system model returned NaN or infinity "
                    f"at adaptive sample {index + 1}/{len(samples)}."
                )
            results.append(values)

        array = np.vstack(results)
        return array.reshape(-1) if len(self.spy_outputs) == 1 else array

    def _raise_if_cancelled(self) -> None:
        if self.stop_flag or self.isInterruptionRequested():
            raise RuntimeError("Adaptive surrogate training was cancelled.")


def _compile_spy_model(source: str) -> SpyFunction:
    """Compile trusted graph-generated source once for adaptive evaluation."""
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(Path.cwd() / "_pylcss_adaptive_spy.py"),
        "__name__": "_pylcss_adaptive_spy",
        "__package__": None,
    }
    try:
        exec(compile(source, "<adaptive spy model>", "exec"), namespace)
    except Exception as exc:
        raise RuntimeError(f"Could not compile the generated spy model: {exc}") from exc
    function = namespace.get("spy_model")
    if not callable(function):
        raise RuntimeError(
            "Generated adaptive source must define a callable 'spy_model'."
        )
    return function


def _prediction_uncertainty(
    model: Any,
    candidates: np.ndarray,
) -> np.ndarray | None:
    """Return one uncertainty score per point when the model supports it."""
    try:
        prediction = model.predict(candidates, return_std=True)
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        logger.debug(
            "Surrogate does not expose predictive uncertainty.",
            exc_info=True,
        )
        return None
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        return None
    uncertainty = np.asarray(prediction[1], dtype=float)
    if uncertainty.ndim > 1:
        uncertainty = uncertainty.mean(axis=1)
    uncertainty = uncertainty.reshape(-1)
    if uncertainty.shape != (len(candidates),):
        return None
    if not np.isfinite(uncertainty).all():
        return None
    return uncertainty
