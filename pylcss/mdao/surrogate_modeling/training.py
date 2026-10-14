# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Coordinate the surrogate workflows exposed by the GUI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .auto_surrogate import AutoSurrogateStrategy
from .contracts import LossCallback, Metrics, ProgressCallback, SpyPort, StopFlag
from .data_generation import evaluate_spy_points, generate_spy_data
from .strategies import SurrogateModelStrategy
from .torch_utils import TORCH_AVAILABLE


class SurrogateTrainer:
    def __init__(self) -> None:
        self.strategies: dict[str, SurrogateModelStrategy] = {
            "Auto Mode": AutoSurrogateStrategy(),
        }
        if TORCH_AVAILABLE:
            from .custom_architecture import CustomArchitectureStrategy

            self.strategies["Your Own Architecture (Deep Learning)"] = (
                CustomArchitectureStrategy()
            )

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(self.strategies)

    def generate_data(
        self,
        spy_code: str,
        spy_inputs: Sequence[SpyPort | str],
        spy_outputs: Sequence[SpyPort | str],
        input_bounds: Sequence[tuple[float, float]],
        num_samples: int = 1_000,
        test_samples: int = 200,
        random_state: int = 42,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        *,
        max_consecutive_failures: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        return generate_spy_data(
            spy_code,
            spy_inputs,
            spy_outputs,
            input_bounds,
            num_samples=num_samples,
            test_samples=test_samples,
            random_state=random_state,
            callback=callback,
            stop_flag=stop_flag,
            max_consecutive_failures=max_consecutive_failures,
        )

    def evaluate_points(
        self,
        spy_code: str,
        spy_inputs: Sequence[SpyPort | str],
        spy_outputs: Sequence[SpyPort | str],
        points: np.ndarray,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return evaluate_spy_points(
            spy_code,
            spy_inputs,
            spy_outputs,
            points,
            callback=callback,
            stop_flag=stop_flag,
        )

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
    ) -> tuple[Any, Metrics]:
        model_type = str(config.get("model_type", "Auto Mode"))
        try:
            strategy = self.strategies[model_type]
        except KeyError:
            raise ValueError(
                f"Unknown model type {model_type!r}. Available models: "
                + ", ".join(self.available_models)
            ) from None
        return strategy.train(
            X, y, config, X_test, y_test, callback, stop_flag, loss_callback
        )
