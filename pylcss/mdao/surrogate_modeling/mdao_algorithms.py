# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Engineering regression estimators used by Auto Mode and validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import optimize
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from .metrics import (
    as_target_matrix,
    evaluate_predictions,
    format_prediction,
    prediction_features,
    validate_training_data,
)

__all__ = [
    "MdaoEnsembleOfSurrogates",
    "MdaoKPLSRegressor",
    "MdaoKrigingRegressor",
    "MdaoPolynomialRegressor",
    "MdaoRBFRegressor",
    "MdaoSVRRegressor",
    "compute_mdao_metrics",
]


def compute_mdao_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Score matching target matrices; normalize maximum error per output."""
    actual = as_target_matrix(y_true, name="y_true")
    predicted = as_target_matrix(y_pred, name="y_pred")
    basic = evaluate_predictions(actual, predicted, max_samples=0)
    errors = np.abs(actual - predicted)
    scales = actual.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return {
        "R2": basic["R2"],
        "RMSE": basic["RMSE"],
        "MAE": float(errors.mean()),
        "RMAE": float(np.max(errors / scales)),
    }


class MdaoKrigingRegressor(RegressorMixin, BaseEstimator):
    """Gaussian process with per-input length scales and standardized targets."""

    def __init__(
        self,
        kernel_type: str = "matern52",
        n_restarts: int = 5,
        nugget: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.kernel_type = kernel_type
        self.n_restarts = n_restarts
        self.nugget = nugget
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoKrigingRegressor:
        features, targets = validate_training_data(X, y, minimum_samples=1)
        X_s = self.scaler_x.fit_transform(features)
        y_s = self.scaler_y.fit_transform(targets)
        scales = np.ones(features.shape[1])
        if self.kernel_type == "matern32":
            covariance = Matern(scales, (1e-3, 1e3), nu=1.5)
        elif self.kernel_type == "gaussian":
            covariance = RBF(scales, (1e-3, 1e3))
        elif self.kernel_type == "matern52":
            covariance = Matern(scales, (1e-3, 1e3), nu=2.5)
        else:
            raise ValueError(f"Unknown Kriging kernel {self.kernel_type!r}.")
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * covariance + WhiteKernel(
            noise_level=self.nugget, noise_level_bounds=(1e-9, 1e-1)
        )
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
            normalize_y=False,
        )
        self.gp_.fit(X_s, y_s[:, 0] if y_s.shape[1] == 1 else y_s)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.gp_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        features, single = prediction_features(X)
        scaled = self.scaler_x.transform(features)
        if return_std:
            mean_s, std_s = self.gp_.predict(scaled, return_std=True)
        else:
            mean_s = self.gp_.predict(scaled)
        mean = self.scaler_y.inverse_transform(as_target_matrix(mean_s))
        result = format_prediction(mean, single=single)
        if not return_std:
            return result
        std = as_target_matrix(std_s) * self.scaler_y.scale_
        return result, format_prediction(std, single=single)


class MdaoKPLSRegressor(RegressorMixin, BaseEstimator):
    """Project inputs with partial least squares, then fit Kriging."""

    def __init__(
        self, n_comp: int = 3, n_restarts: int = 4, random_state: int = 42
    ) -> None:
        self.n_comp = n_comp
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pls_ = None
        self.kriging_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoKPLSRegressor:
        features, targets = validate_training_data(X, y)
        if self.n_comp < 1:
            raise ValueError("n_comp must be positive.")
        scaled_X = self.scaler_x.fit_transform(features)
        scaled_y = self.scaler_y.fit_transform(targets)
        count = min(self.n_comp, features.shape[1], len(features) - 1)
        self.pls_ = PLSRegression(n_components=count)
        self.pls_.fit(scaled_X, scaled_y)
        latent = self.pls_.transform(scaled_X)
        self.kriging_ = MdaoKrigingRegressor(
            n_restarts=self.n_restarts,
            random_state=self.random_state,
        ).fit(latent, targets)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pls_ is None or self.kriging_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        features, single = prediction_features(X)
        latent = self.pls_.transform(self.scaler_x.transform(features))
        prediction = self.kriging_.predict(latent, return_std=return_std)
        if return_std:
            return tuple(
                format_prediction(values, single=single) for values in prediction
            )
        return format_prediction(prediction, single=single)


class MdaoRBFRegressor(RegressorMixin, BaseEstimator):
    """Scaled radial-basis interpolation with smoothing and polynomial drift."""

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        smoothing: float = 1e-4,
        degree: int = 1,
        epsilon: float = 1.0,
    ) -> None:
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self.epsilon = epsilon
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.rbf_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoRBFRegressor:
        features, targets = validate_training_data(X, y, minimum_samples=1)
        kwargs = dict(kernel=self.kernel, smoothing=self.smoothing, degree=self.degree)
        if self.kernel in {
            "multiquadric",
            "inverse_multiquadric",
            "gaussian",
            "inverse_quadratic",
        }:
            kwargs["epsilon"] = self.epsilon
        scaled = self.scaler_x.fit_transform(features)
        self._neighbors_ = cKDTree(scaled)
        self.rbf_ = RBFInterpolator(
            scaled,
            self.scaler_y.fit_transform(targets),
            **kwargs,
        )
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.rbf_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        features, single = prediction_features(X)
        scaled = self.scaler_x.transform(features)
        mean = self.scaler_y.inverse_transform(self.rbf_(scaled))
        result = format_prediction(mean, single=single)
        if not return_std:
            return result
        # A distance heuristic, not a calibrated posterior standard deviation.
        distances, _ = self._neighbors_.query(scaled)
        std = distances[:, None] * 0.1 * self.scaler_y.scale_
        return result, format_prediction(std, single=single)


class MdaoPolynomialRegressor(RegressorMixin, BaseEstimator):
    """Sparse monomial response surface fitted with ElasticNet."""

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_terms: int | None = None,
    ) -> None:
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_terms = max_terms
        self.pipe_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoPolynomialRegressor:
        features, targets = validate_training_data(X, y, minimum_samples=1)
        if not isinstance(self.degree, (int, np.integer)) or self.degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        terms = math.comb(features.shape[1] + self.degree, self.degree)
        if self.max_terms is not None:
            if self.max_terms < 1:
                raise ValueError("max_terms must be positive when specified.")
            if terms > self.max_terms:
                raise ValueError(
                    f"Polynomial expansion needs {terms} terms, exceeding max_terms={self.max_terms}."
                )
        self.pipe_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=True)),
                (
                    "sparse_reg",
                    ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000),
                ),
            ]
        )
        self.pipe_.fit(features, targets[:, 0] if targets.shape[1] == 1 else targets)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pipe_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        features, single = prediction_features(X)
        result = format_prediction(self.pipe_.predict(features), single=single)
        return (result, np.zeros_like(result)) if return_std else result


class MdaoSVRRegressor(RegressorMixin, BaseEstimator):
    """Input-standardized SVR, with one estimator per output when needed."""

    def __init__(
        self, C: float = 10.0, epsilon: float = 0.05, kernel: str = "rbf"
    ) -> None:
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.pipe_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoSVRRegressor:
        features, targets = validate_training_data(X, y, minimum_samples=1)
        estimator = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        multiple = targets.shape[1] > 1
        if multiple:
            estimator = MultiOutputRegressor(estimator)
        self.pipe_ = Pipeline([("scaler", StandardScaler()), ("svr", estimator)])
        self.pipe_.fit(features, targets if multiple else targets[:, 0])
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pipe_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        features, single = prediction_features(X)
        result = format_prediction(self.pipe_.predict(features), single=single)
        return (result, np.zeros_like(result)) if return_std else result


class MdaoEnsembleOfSurrogates(RegressorMixin, BaseEstimator):
    """Convex combination of fitted models, weighted by validation residuals."""

    def __init__(
        self, models: list[tuple[str, Any]], error_matrix: np.ndarray | None = None
    ) -> None:
        if not models:
            raise ValueError("An ensemble needs at least one fitted model.")
        self.models = models
        self.error_matrix = error_matrix
        self.names = [name for name, _ in models]
        if error_matrix is None:
            self.weights_ = np.full(len(models), 1 / len(models))
        else:
            errors = as_target_matrix(error_matrix, name="error_matrix")
            if errors.shape[1] != len(models):
                raise ValueError("error_matrix needs one column per ensemble model.")
            self.weights_ = self._solve_optimal_weights(errors)

    @staticmethod
    def _solve_optimal_weights(error_matrix: np.ndarray) -> np.ndarray:
        """Minimize the residual second moment with nonnegative unit-sum weights."""
        count = error_matrix.shape[1]
        if count == 1:
            return np.ones(1)
        covariance = error_matrix.T @ error_matrix / len(error_matrix)
        covariance += np.eye(count) * 1e-8
        result = optimize.minimize(
            lambda weights: 0.5 * float(weights @ covariance @ weights),
            np.full(count, 1 / count),
            jac=lambda weights: covariance @ weights,
            bounds=[(0, 1)] * count,
            constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
            method="SLSQP",
        )
        if result.success and np.isfinite(result.x).all():
            weights = np.clip(result.x, 0, 1)
            if weights.sum() > 0:
                return weights / weights.sum()
        inverse_rmse = 1 / np.maximum(np.sqrt(np.diag(covariance)), 1e-8)
        return inverse_rmse / inverse_rmse.sum()

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        features, single = prediction_features(X)
        predictions = [
            as_target_matrix(model.predict(features)) for _, model in self.models
        ]
        expected = predictions[0].shape
        if expected[0] != len(features) or any(
            p.shape != expected for p in predictions
        ):
            raise ValueError(
                "Ensemble models must return matching sample/output shapes."
            )
        stacked = np.stack(predictions)
        mean = np.sum(stacked * self.weights_[:, None, None], axis=0)
        result = format_prediction(mean, single=single)
        if not return_std:
            return result
        return result, format_prediction(stacked.std(axis=0), single=single)
