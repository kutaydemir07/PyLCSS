# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Cross-validation, model comparison, and feature-importance utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts import ProgressCallback
from .metrics import as_target_matrix, validate_training_data

logger = logging.getLogger(__name__)

try:
    from sklearn.inspection import (
        permutation_importance as sklearn_permutation_importance,
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class CVResult:
    """Per-fold metrics and failure state for one model configuration."""

    model_type: str
    n_folds: int
    r2_scores: list[float] = field(default_factory=list)
    rmse_scores: list[float] = field(default_factory=list)
    mae_scores: list[float] = field(default_factory=list)
    error: str | None = None

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(finite.mean()) if finite.size else float("nan")

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.rmse_scores)

    @property
    def r2_mean(self) -> float:
        return self._mean(self.r2_scores)

    @property
    def r2_std(self) -> float:
        finite = np.asarray(self.r2_scores, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(finite.std()) if finite.size else float("nan")

    @property
    def rmse_mean(self) -> float:
        return self._mean(self.rmse_scores)

    @property
    def mae_mean(self) -> float:
        return self._mean(self.mae_scores)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "succeeded": self.succeeded,
                "r2_mean": self.r2_mean,
                "r2_std": self.r2_std,
                "rmse_mean": self.rmse_mean,
                "mae_mean": self.mae_mean,
            }
        )
        return result


def _require_sklearn() -> None:
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for surrogate validation.")


def _fit_target(y: np.ndarray) -> np.ndarray:
    """Use sklearn's conventional 1-D target for a single output."""
    return y.ravel() if y.shape[1] == 1 else y


def _prediction_matrix(
    prediction: Any,
    *,
    samples: int,
    outputs: int,
) -> np.ndarray:
    array = as_target_matrix(prediction, name="prediction")
    expected = (samples, outputs)
    if array.shape != expected:
        raise ValueError(
            f"Estimator returned shape {array.shape}; expected {expected}."
        )
    return array


def _score_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float]:
    if actual.shape[0] < 2:
        r2 = float("nan")
    else:
        r2 = float(r2_score(actual, predicted, multioutput="uniform_average"))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    return r2, rmse, mae


class CrossValidator:
    """K-fold and leave-one-out validation for sklearn-compatible estimators."""

    def __init__(self, *, random_state: int = 42) -> None:
        _require_sklearn()
        self.random_state = int(random_state)

    def kfold_cv(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        model_type: str = "Unknown",
        callback: ProgressCallback | None = None,
    ) -> CVResult:
        """Fit a fresh estimator in each shuffled fold."""
        features, targets = validate_training_data(X, y)
        if not 2 <= n_folds <= features.shape[0]:
            raise ValueError(
                f"n_folds must be between 2 and {features.shape[0]}; "
                f"received {n_folds}."
            )
        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        result = CVResult(model_type=model_type, n_folds=n_folds)

        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(features)
        ):
            if callback:
                callback(
                    int(100 * fold_index / n_folds),
                    f"CV fold {fold_index + 1}/{n_folds}...",
                )
            estimator = model_factory()
            estimator.fit(
                features[train_indices],
                _fit_target(targets[train_indices]),
            )
            prediction = _prediction_matrix(
                estimator.predict(features[validation_indices]),
                samples=len(validation_indices),
                outputs=targets.shape[1],
            )
            r2, rmse, mae = _score_predictions(
                targets[validation_indices],
                prediction,
            )
            result.r2_scores.append(r2)
            result.rmse_scores.append(rmse)
            result.mae_scores.append(mae)

        if callback:
            callback(
                100,
                f"CV complete: R²={result.r2_mean:.4f} ± {result.r2_std:.4f}",
            )
        return result

    def loo_cv(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "Unknown",
        callback: ProgressCallback | None = None,
    ) -> CVResult:
        """Compute aggregate leave-one-out predictions for a small data set."""
        features, targets = validate_training_data(X, y, minimum_samples=3)
        count = features.shape[0]
        predictions = np.empty_like(targets)
        result = CVResult(model_type=model_type, n_folds=count)

        for index in range(count):
            if callback and index % max(1, count // 20) == 0:
                callback(int(100 * index / count), f"LOO sample {index + 1}/{count}...")
            training_mask = np.ones(count, dtype=bool)
            training_mask[index] = False
            estimator = model_factory()
            estimator.fit(features[training_mask], _fit_target(targets[training_mask]))
            prediction = _prediction_matrix(
                estimator.predict(features[index : index + 1]),
                samples=1,
                outputs=targets.shape[1],
            )
            predictions[index] = prediction[0]

        r2, rmse, mae = _score_predictions(targets, predictions)
        result.r2_scores.append(r2)
        result.rmse_scores.append(rmse)
        result.mae_scores.append(mae)
        if callback:
            callback(100, f"LOO complete: R²={r2:.4f}")
        return result


class FeatureImportanceAnalyzer:
    """Compute and rank model-agnostic or tree-native feature importances."""

    @staticmethod
    def _names(
        feature_count: int,
        feature_names: Sequence[str] | None,
    ) -> list[str]:
        if feature_names is None:
            return [f"X{index}" for index in range(feature_count)]
        names = [str(name) for name in feature_names]
        if len(names) != feature_count:
            raise ValueError(
                f"Expected {feature_count} feature names, received {len(names)}."
            )
        return names

    @staticmethod
    def permutation_importance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
        n_repeats: int = 10,
    ) -> dict[str, Any]:
        """Compute held-out permutation importance using aggregate R²."""
        _require_sklearn()
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")
        features, targets = validate_training_data(X, y, minimum_samples=2)
        names = FeatureImportanceAnalyzer._names(
            features.shape[1],
            feature_names,
        )
        result = sklearn_permutation_importance(
            model,
            features,
            _fit_target(targets),
            n_repeats=n_repeats,
            random_state=42,
            scoring="r2",
        )
        means = np.asarray(result.importances_mean, dtype=np.float64)
        standard_deviations = np.asarray(result.importances_std, dtype=np.float64)
        ranking = np.argsort(-means)
        return {
            "feature_names": names,
            "importances_mean": means.tolist(),
            "importances_std": standard_deviations.tolist(),
            "ranking": [names[index] for index in ranking],
            "ranking_values": [float(means[index]) for index in ranking],
        }

    @staticmethod
    def tree_feature_importance(
        model: Any,
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Extract and rank a fitted tree ensemble's native importances."""
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return {"error": "Model does not expose feature_importances_."}
        values = np.asarray(importances, dtype=np.float64)
        if values.ndim != 1:
            return {"error": "Model feature importances are not one-dimensional."}
        names = FeatureImportanceAnalyzer._names(len(values), feature_names)
        ranking = np.argsort(-values)
        return {
            "feature_names": names,
            "importances": values.tolist(),
            "ranking": [names[index] for index in ranking],
            "ranking_values": [float(values[index]) for index in ranking],
        }


class ModelComparator:
    """Compare model factories on identical folds and rank successful runs."""

    def __init__(self, *, random_state: int = 42) -> None:
        _require_sklearn()
        self.random_state = int(random_state)

    def compare_models(
        self,
        model_factories: Mapping[str, Callable[[], Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        callback: ProgressCallback | None = None,
    ) -> list[CVResult]:
        if not model_factories:
            raise ValueError("At least one model factory is required.")
        validator = CrossValidator(random_state=self.random_state)
        results: list[CVResult] = []
        total = len(model_factories)
        for index, (name, factory) in enumerate(model_factories.items()):
            if callback:
                callback(int(100 * index / total), f"Evaluating {name}...")
            try:
                result = validator.kfold_cv(
                    factory,
                    X,
                    y,
                    n_folds=n_folds,
                    model_type=name,
                )
            except Exception as exc:
                logger.exception("Model comparison failed for %s.", name)
                result = CVResult(
                    model_type=name,
                    n_folds=n_folds,
                    error=str(exc),
                )
            results.append(result)

        results.sort(
            key=lambda result: (
                not result.succeeded,
                -result.r2_mean if result.succeeded else float("inf"),
            )
        )
        if callback:
            successful = next((result for result in results if result.succeeded), None)
            message = (
                f"Best: {successful.model_type} (R²={successful.r2_mean:.4f})"
                if successful
                else "No model completed successfully."
            )
            callback(100, message)
        return results


__all__ = [
    "SKLEARN_AVAILABLE",
    "CVResult",
    "CrossValidator",
    "FeatureImportanceAnalyzer",
    "ModelComparator",
]
