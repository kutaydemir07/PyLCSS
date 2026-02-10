# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Cross-validation, hyperparameter optimization, and model comparison
utilities for surrogate modeling.

Provides:
- K-Fold / Leave-One-Out cross-validation with per-output metrics
- Grid / Random / Bayesian hyperparameter search
- Feature importance via permutation
- Model comparison across multiple algorithms
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CVResult:
    """Cross-validation results for a single model configuration."""
    model_type: str
    n_folds: int
    r2_scores: List[float] = field(default_factory=list)
    rmse_scores: List[float] = field(default_factory=list)
    mae_scores: List[float] = field(default_factory=list)

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.r2_scores)) if self.r2_scores else 0.0

    @property
    def r2_std(self) -> float:
        return float(np.std(self.r2_scores)) if self.r2_scores else 0.0

    @property
    def rmse_mean(self) -> float:
        return float(np.mean(self.rmse_scores)) if self.rmse_scores else 0.0

    @property
    def mae_mean(self) -> float:
        return float(np.mean(self.mae_scores)) if self.mae_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'n_folds': self.n_folds,
            'r2_mean': self.r2_mean,
            'r2_std': self.r2_std,
            'rmse_mean': self.rmse_mean,
            'mae_mean': self.mae_mean,
            'r2_scores': self.r2_scores,
            'rmse_scores': self.rmse_scores,
            'mae_scores': self.mae_scores
        }


@dataclass
class HPOResult:
    """Hyperparameter optimization result."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0


# ============================================================================
# Cross-Validation Engine
# ============================================================================

class CrossValidator:
    """
    K-Fold and Leave-One-Out cross-validation for surrogate models.
    
    Works with any scikit-learn-compatible estimator (including PyTorchWrapper).
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for cross-validation.")

    def kfold_cv(self, model_factory: Callable[[], Any],
                 X: np.ndarray, y: np.ndarray,
                 n_folds: int = 5,
                 model_type: str = "Unknown",
                 callback: Optional[Callable[[int, str], None]] = None) -> CVResult:
        """
        Perform K-Fold cross-validation.

        Args:
            model_factory: Callable that returns a fresh (unfitted) model instance.
            X: Input features (n_samples, n_features).
            y: Target values (n_samples,) or (n_samples, n_outputs).
            n_folds: Number of CV folds.
            model_type: Label for the model.
            callback: Optional progress callback(percent, message).

        Returns:
            CVResult with per-fold metrics.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        result = CVResult(model_type=model_type, n_folds=n_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            if callback:
                pct = int(100 * fold_idx / n_folds)
                callback(pct, f"CV Fold {fold_idx + 1}/{n_folds}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Handle multi-output
            if y_val.ndim == 1:
                y_val_flat, y_pred_flat = y_val, y_pred
            else:
                y_val_flat = y_val.ravel()
                y_pred_flat = y_pred.ravel()

            r2 = r2_score(y_val_flat, y_pred_flat)
            rmse = float(np.sqrt(mean_squared_error(y_val_flat, y_pred_flat)))
            mae = float(mean_absolute_error(y_val_flat, y_pred_flat))

            result.r2_scores.append(float(r2))
            result.rmse_scores.append(rmse)
            result.mae_scores.append(mae)

        if callback:
            callback(100, f"CV complete: R² = {result.r2_mean:.4f} ± {result.r2_std:.4f}")
        return result

    def loo_cv(self, model_factory: Callable[[], Any],
               X: np.ndarray, y: np.ndarray,
               model_type: str = "Unknown",
               callback: Optional[Callable[[int, str], None]] = None) -> CVResult:
        """Leave-One-Out cross-validation (expensive, use for small datasets)."""
        n = len(X)
        result = CVResult(model_type=model_type, n_folds=n)
        predictions = np.zeros_like(y)

        for i in range(n):
            if callback and i % max(1, n // 20) == 0:
                callback(int(100 * i / n), f"LOO {i}/{n}...")

            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = model_factory()
            model.fit(X[mask], y[mask])
            predictions[i] = model.predict(X[i:i + 1])

        if y.ndim == 1:
            r2 = r2_score(y, predictions)
            rmse = float(np.sqrt(mean_squared_error(y, predictions)))
            mae = float(mean_absolute_error(y, predictions))
        else:
            r2 = r2_score(y.ravel(), predictions.ravel())
            rmse = float(np.sqrt(mean_squared_error(y.ravel(), predictions.ravel())))
            mae = float(mean_absolute_error(y.ravel(), predictions.ravel()))

        result.r2_scores = [float(r2)]
        result.rmse_scores = [rmse]
        result.mae_scores = [mae]

        if callback:
            callback(100, f"LOO complete: R² = {r2:.4f}")
        return result


# ============================================================================
# Hyperparameter Optimization
# ============================================================================

class HyperparameterOptimizer:
    """
    Grid search and random search for surrogate model hyperparameters.
    """

    # Default search spaces per model type
    SEARCH_SPACES = {
        'MLP Regressor': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (256, 128, 64)],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-4, 1e-3, 1e-2],
            'max_iter': [500, 1000, 2000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Gaussian Process': {
            'alpha': [1e-10, 1e-8, 1e-6, 1e-4],
            'n_restarts_optimizer': [0, 5, 10, 20]
        }
    }

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for HPO.")

    def grid_search(self, model_factory_fn: Callable[[Dict], Any],
                    X: np.ndarray, y: np.ndarray,
                    param_grid: Dict[str, List],
                    n_folds: int = 5,
                    callback: Optional[Callable[[int, str], None]] = None) -> HPOResult:
        """
        Exhaustive grid search with K-fold CV.

        Args:
            model_factory_fn: Callable(params_dict) -> unfitted model.
            X, y: Training data.
            param_grid: Dict of param_name -> list of values.
            n_folds: CV folds.
            callback: Progress callback.

        Returns:
            HPOResult with best params and all trial results.
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))
        n_total = len(all_combos)

        best_score = -np.inf
        best_params = {}
        all_results = []

        cv = CrossValidator()

        for idx, combo in enumerate(all_combos):
            if callback:
                callback(int(100 * idx / n_total), f"Grid Search {idx + 1}/{n_total}")

            params = dict(zip(param_names, combo))

            try:
                def factory():
                    return model_factory_fn(params)

                cv_result = cv.kfold_cv(factory, X, y, n_folds=n_folds, model_type="HPO")
                score = cv_result.r2_mean

                all_results.append({
                    'params': params,
                    'r2_mean': score,
                    'r2_std': cv_result.r2_std,
                    'rmse_mean': cv_result.rmse_mean
                })

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning("Grid search trial failed: %s", e)
                all_results.append({'params': params, 'error': str(e)})

        return HPOResult(
            best_params=best_params,
            best_score=float(best_score),
            all_results=all_results,
            n_trials=n_total
        )

    def random_search(self, model_factory_fn: Callable[[Dict], Any],
                      X: np.ndarray, y: np.ndarray,
                      param_grid: Dict[str, List],
                      n_trials: int = 20,
                      n_folds: int = 5,
                      callback: Optional[Callable[[int, str], None]] = None) -> HPOResult:
        """
        Random search: sample param_grid randomly for n_trials.
        """
        rng = np.random.RandomState(42)
        param_names = list(param_grid.keys())

        best_score = -np.inf
        best_params = {}
        all_results = []

        cv = CrossValidator()

        for trial in range(n_trials):
            if callback:
                callback(int(100 * trial / n_trials), f"Random Search {trial + 1}/{n_trials}")

            params = {name: rng.choice(values) for name, values in param_grid.items()}

            try:
                def factory(p=params):
                    return model_factory_fn(p)

                cv_result = cv.kfold_cv(factory, X, y, n_folds=n_folds, model_type="HPO")
                score = cv_result.r2_mean

                all_results.append({
                    'params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()},
                    'r2_mean': score,
                    'r2_std': cv_result.r2_std,
                    'rmse_mean': cv_result.rmse_mean
                })

                if score > best_score:
                    best_score = score
                    best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()}

            except Exception as e:
                logger.warning("Random search trial failed: %s", e)

        return HPOResult(
            best_params=best_params,
            best_score=float(best_score),
            all_results=all_results,
            n_trials=n_trials
        )


# ============================================================================
# Feature Importance
# ============================================================================

class FeatureImportanceAnalyzer:
    """Compute and rank feature importances for trained surrogate models."""

    @staticmethod
    def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                n_repeats: int = 10) -> Dict[str, Any]:
        """
        Compute permutation-based feature importance.

        Works with any model that has a .predict() method.

        Returns:
            Dict with 'feature_names', 'importances_mean', 'importances_std', 'ranking'.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required.")

        result = permutation_importance(model, X, y, n_repeats=n_repeats,
                                         random_state=42, scoring='r2')

        importances = result.importances_mean
        std = result.importances_std

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]

        ranking = np.argsort(-importances)

        return {
            'feature_names': feature_names,
            'importances_mean': importances.tolist(),
            'importances_std': std.tolist(),
            'ranking': [feature_names[i] for i in ranking],
            'ranking_values': [float(importances[i]) for i in ranking]
        }

    @staticmethod
    def tree_feature_importance(model, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract built-in feature importance from tree-based models (RF, GBR).
        """
        if not hasattr(model, 'feature_importances_'):
            return {'error': 'Model does not have feature_importances_ attribute.'}

        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(importances))]

        ranking = np.argsort(-importances)
        return {
            'feature_names': feature_names,
            'importances': importances.tolist(),
            'ranking': [feature_names[i] for i in ranking],
            'ranking_values': [float(importances[i]) for i in ranking]
        }


# ============================================================================
# Model Comparison
# ============================================================================

class ModelComparator:
    """
    Compare multiple surrogate model types on the same dataset using CV.
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required.")

    def compare_models(self, model_factories: Dict[str, Callable[[], Any]],
                       X: np.ndarray, y: np.ndarray,
                       n_folds: int = 5,
                       callback: Optional[Callable[[int, str], None]] = None) -> List[CVResult]:
        """
        Run K-fold CV for each model and return ranked results.

        Args:
            model_factories: Dict of model_name -> factory callable.
            X, y: Training data.
            n_folds: Number of CV folds.

        Returns:
            List of CVResult sorted by R² (descending).
        """
        cv = CrossValidator()
        results = []
        names = list(model_factories.keys())

        for idx, (name, factory) in enumerate(model_factories.items()):
            if callback:
                callback(int(100 * idx / len(names)), f"Evaluating {name}...")

            try:
                cv_result = cv.kfold_cv(factory, X, y, n_folds=n_folds, model_type=name)
                results.append(cv_result)
            except Exception as e:
                logger.error("Model comparison failed for %s: %s", name, e)
                failed = CVResult(model_type=name, n_folds=n_folds)
                failed.r2_scores = [0.0]
                failed.rmse_scores = [float('inf')]
                failed.mae_scores = [float('inf')]
                results.append(failed)

        # Sort by R² descending
        results.sort(key=lambda r: r.r2_mean, reverse=True)

        if callback:
            best = results[0] if results else None
            callback(100, f"Best: {best.model_type} (R²={best.r2_mean:.4f})" if best else "No results")

        return results
