# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Training contract shared by Auto Mode and custom architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import LossCallback, Metrics, ProgressCallback, StopFlag

TrainingResult = tuple[Any, Metrics]


class SurrogateModelStrategy(ABC):
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
    ) -> TrainingResult:
        """Return a fitted predictor and evaluation metrics."""
