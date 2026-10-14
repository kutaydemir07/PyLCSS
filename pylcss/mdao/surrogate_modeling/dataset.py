# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Consistent training and holdout preparation for the supported workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .metrics import validate_training_data


@dataclass(frozen=True)
class TrainingData:
    train_X: np.ndarray
    train_y: np.ndarray
    eval_X: np.ndarray
    eval_y: np.ndarray
    evaluation_source: str
    external_holdout: bool = False


def prepare_training_data(
    X: np.ndarray,
    y: np.ndarray,
    config: Mapping[str, Any],
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> TrainingData:
    """Validate paired arrays and honor the configured split and seed."""
    features, targets = validate_training_data(X, y)
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must both be provided or both omitted.")
    if X_test is not None:
        empty_X, empty_y = np.asarray(X_test).size == 0, np.asarray(y_test).size == 0
        if empty_X != empty_y:
            raise ValueError(
                "X_test and y_test must both be empty or both contain rows."
            )
        if not empty_X:
            eval_X, eval_y = validate_training_data(X_test, y_test, minimum_samples=1)
            if eval_X.shape[1] != features.shape[1]:
                raise ValueError("Training and test feature counts differ.")
            if eval_y.shape[1] != targets.shape[1]:
                raise ValueError("Training and test output counts differ.")
            return TrainingData(features, targets, eval_X, eval_y, "holdout", True)

    fraction = float(config.get("validation_split", 0.2))
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("validation_split must be finite and in [0, 1).")
    if fraction == 0:
        return TrainingData(features, targets, features, targets, "training")
    train_X, eval_X, train_y, eval_y = train_test_split(
        features,
        targets,
        test_size=fraction,
        random_state=int(config.get("random_state", 42)),
        shuffle=True,
    )
    return TrainingData(train_X, train_y, eval_X, eval_y, "holdout")
