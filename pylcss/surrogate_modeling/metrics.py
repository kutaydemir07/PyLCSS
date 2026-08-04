# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Regression metrics and array validation shared by all model backends."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error, r2_score

from .contracts import FloatArray, Metrics


def as_feature_matrix(values: ArrayLike, *, name: str = "X") -> FloatArray:
    """Return a finite ``(samples, features)`` float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; received shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def as_target_matrix(values: ArrayLike, *, name: str = "y") -> FloatArray:
    """Return a finite ``(samples, outputs)`` float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be one- or two-dimensional; received shape {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one output.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_training_data(
    X: ArrayLike,
    y: ArrayLike,
    *,
    minimum_samples: int = 2,
) -> tuple[FloatArray, FloatArray]:
    """Validate and normalize a supervised regression data set."""
    features = as_feature_matrix(X)
    targets = as_target_matrix(y)
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and y must have the same number of samples; "
            f"received {features.shape[0]} and {targets.shape[0]}."
        )
    if features.shape[0] < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} samples are required; "
            f"received {features.shape[0]}."
        )
    return features, targets


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    max_samples: int = 100,
) -> Metrics:
    """Calculate stable aggregate metrics without hiding shape mismatches."""
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")

    actual = as_target_matrix(y_true, name="y_true")
    predicted = as_target_matrix(y_pred, name="y_pred")
    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes; "
            f"received {actual.shape} and {predicted.shape}."
        )

    mse = float(mean_squared_error(actual, predicted))
    if actual.shape[0] < 2:
        r2 = 1.0 if mse <= np.finfo(np.float64).eps else 0.0
    else:
        r2 = float(r2_score(actual, predicted, multioutput="uniform_average"))

    return {
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "y_test": actual[:max_samples].tolist(),
        "y_pred": predicted[:max_samples].tolist(),
    }


def empty_metrics(*, debug_mode: bool = False) -> Metrics:
    """Return the UI-compatible representation of an unevaluated model."""
    return {
        "RMSE": None,
        "R2": None,
        "y_test": [],
        "y_pred": [],
        "debug_mode": debug_mode,
    }
