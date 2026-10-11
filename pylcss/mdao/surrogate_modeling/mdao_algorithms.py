# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""State-of-the-art surrogate modeling algorithms for Multidisciplinary Design
Analysis and Optimization (MDAO).

Grounded in scientific publications from:
- Structural and Multidisciplinary Optimization (SMO, Springer)
- AIAA Journal and AIAA Aviation/SciTech Forums
- Surrogate Modeling Toolbox (SMT, Bouhlel et al. 2019)
- ASME Journal of Mechanical Design
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import linalg, optimize
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

__all__ = [
    "MdaoEnsembleOfSurrogates",
    "MdaoKPLSRegressor",
    "MdaoKrigingRegressor",
    "MdaoRBFRegressor",
    "MdaoSVRRegressor",
    "MdaoSparsePCERegressor",
    "compute_mdao_metrics",
]


def compute_mdao_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scientific MDAO accuracy and error metrics.

    Includes Relative Maximum Absolute Error (RMAE) as established in
    Viana et al. (2009) and Goel et al. (2007) for engineering constraint safety.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    residuals = yt - yp
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    y_std = float(np.std(yt))
    var_y = y_std**2
    if var_y > 1e-12:
        r2 = float(1.0 - (mse / var_y))
        rmae = float(np.max(np.abs(residuals)) / (y_std + 1e-12))
    else:
        r2 = 1.0 if mse < 1e-12 else 0.0
        rmae = float(np.max(np.abs(residuals)))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RMAE": rmae,  # Relative Maximum Absolute Error: critical in MDAO
    }


class MdaoKrigingRegressor(BaseEstimator, RegressorMixin):
    """Ordinary and Universal Kriging with Automatic Relevance Determination (ARD).

    Scientific Reference:
    - Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn, H. P. (1989).
      Design and analysis of computer experiments. Statistical Science, 409-423.
    - Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
      Efficient global optimization of expensive black-box functions.
      Journal of Global Optimization, 13(4), 455-492.
    """

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
        self.gp_: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoKrigingRegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        X_s = self.scaler_x.fit_transform(X_arr)
        y_s = self.scaler_y.fit_transform(y_arr)

        d = X_s.shape[1]
        length_scales = np.ones(d)

        # Anisotropic ARD kernel construction
        if self.kernel_type == "matern32":
            cov = Matern(length_scale=length_scales, length_scale_bounds=(1e-3, 1e3), nu=1.5)
        elif self.kernel_type in ("gaussian", "rbf", "sqexp"):
            cov = RBF(length_scale=length_scales, length_scale_bounds=(1e-3, 1e3))
        else:  # default matern52 (C2 continuous - recommended for CFD/FEA)
            cov = Matern(length_scale=length_scales, length_scale_bounds=(1e-3, 1e3), nu=2.5)

        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * cov + WhiteKernel(
            noise_level=self.nugget, noise_level_bounds=(1e-9, 1e-1)
        )

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
            normalize_y=False,
        )
        self.gp_.fit(X_s, y_s.ravel() if y_s.shape[1] == 1 else y_s)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.gp_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        X_s = self.scaler_x.transform(X_arr)

        if not return_std:
            y_s_pred = self.gp_.predict(X_s)
            if y_s_pred.ndim == 1:
                y_s_pred = y_s_pred.reshape(-1, 1)
            y_pred = self.scaler_y.inverse_transform(y_s_pred)
            return y_pred.ravel() if single else y_pred.squeeze()

        y_s_pred, std_s = self.gp_.predict(X_s, return_std=True)
        if y_s_pred.ndim == 1:
            y_s_pred = y_s_pred.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_s_pred)
        scale_val = getattr(self.scaler_y, "scale_", np.array([1.0]))[0]
        y_std = std_s * scale_val
        if single:
            return y_pred.ravel(), y_std.ravel()
        return y_pred.squeeze(), y_std


class MdaoKPLSRegressor(BaseEstimator, RegressorMixin):
    """Kriging with Partial Least Squares for High-Dimensional MDAO Spaces.

    Scientific Reference:
    - Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. (2016).
      Improving kriging surrogates of high-dimensional design spaces by using
      partial least squares decomposition.
      Structural and Multidisciplinary Optimization, 53(5), 935-952.
    """

    def __init__(
        self,
        n_comp: int = 3,
        n_restarts: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_comp = n_comp
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pls_: PLSRegression | None = None
        self.kriging_: MdaoKrigingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoKPLSRegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        X_s = self.scaler_x.fit_transform(X_arr)
        y_s = self.scaler_y.fit_transform(y_arr)

        d = X_s.shape[1]
        comp_count = min(self.n_comp, d, max(1, len(X_arr) - 1))

        self.pls_ = PLSRegression(n_components=comp_count)
        self.pls_.fit(X_s, y_s)

        # Project inputs onto high-covariance PLS latent components
        T_train = self.pls_.transform(X_s)

        # Train Kriging surrogate in the dimension-reduced PLS subspace
        self.kriging_ = MdaoKrigingRegressor(
            kernel_type="matern52",
            n_restarts=self.n_restarts,
            random_state=self.random_state,
        )
        self.kriging_.scaler_x = StandardScaler()
        self.kriging_.scaler_y = self.scaler_y
        self.kriging_.fit(T_train, y_arr)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pls_ is None or self.kriging_ is None:
            raise RuntimeError("Model must be fitted before predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        X_s = self.scaler_x.transform(X_arr)
        T_test = self.pls_.transform(X_s)
        return self.kriging_.predict(T_test, return_std=return_std)


class MdaoRBFRegressor(BaseEstimator, RegressorMixin):
    """Radial Basis Function interpolation with polynomial drift and smoothing.

    Scientific Reference:
    - Gutmann, H. M. (2001). A radial basis function method for global optimization.
      Journal of Global Optimization, 19(3), 201-227.
    - Regis, R. G., & Shoemaker, C. A. (2007).
      A stochastic radial basis function method for constrained global optimization.
      INFORMS Journal on Computing, 19(4), 497-509.
    """

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
        self.rbf_: RBFInterpolator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoRBFRegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        X_s = self.scaler_x.fit_transform(X_arr)
        y_s = self.scaler_y.fit_transform(y_arr)

        rbf_kwargs: dict[str, Any] = {
            "kernel": self.kernel,
            "smoothing": self.smoothing,
            "degree": self.degree,
        }
        if self.kernel in (
            "multiquadric",
            "inverse_multiquadric",
            "gaussian",
            "inverse_quadratic",
        ):
            rbf_kwargs["epsilon"] = self.epsilon

        self.rbf_ = RBFInterpolator(
            X_s,
            y_s,
            **rbf_kwargs,
        )
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.rbf_ is None:
            raise RuntimeError("Model must be fitted before predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        X_s = self.scaler_x.transform(X_arr)
        y_s_pred = self.rbf_(X_s)
        y_pred = self.scaler_y.inverse_transform(y_s_pred)

        pred = y_pred.ravel() if single else y_pred.squeeze()
        if not return_std:
            return pred

        # Heuristic distance-to-data epistemic uncertainty estimate
        X_train_s = self.rbf_.x
        dist = np.min(np.linalg.norm(X_s[:, None, :] - X_train_s[None, :, :], axis=2), axis=1)
        scale_val = getattr(self.scaler_y, "scale_", np.array([1.0]))[0]
        std = dist * 0.1 * scale_val
        return pred, (std.ravel() if single else std)


class MdaoSparsePCERegressor(BaseEstimator, RegressorMixin):
    """Sparse Polynomial Chaos Expansion / Orthogonal Response Surface.

    Scientific Reference:
    - Xiu, D., & Karniadakis, G. E. (2002).
      The Wiener-Askey polynomial chaos for stochastic systems.
      SIAM Journal on Scientific Computing, 24(2), 619-644.
    - Blatman, G., & Sudret, B. (2011).
      Adaptive sparse polynomial chaos expansion based on least angle regression.
      Journal of Computational Physics, 230(6), 2345-2367.
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
    ) -> None:
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pipe_: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoSparsePCERegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        target = y_arr.ravel() if y_arr.ndim > 1 and y_arr.shape[1] == 1 else y_arr

        d = X_arr.shape[1]
        # Restrict polynomial expansion degree if input dimension is high to prevent combinatorial blowup
        effective_degree = 2 if d > 8 else self.degree

        self.pipe_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=effective_degree, include_bias=True)),
                ("sparse_reg", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)),
            ]
        )
        self.pipe_.fit(X_arr, target)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pipe_ is None:
            raise RuntimeError("Model must be fitted before predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        pred = self.pipe_.predict(X_arr)
        res = pred.ravel() if single else pred
        if not return_std:
            return res
        std = np.zeros_like(res)
        return res, std


class MdaoSVRRegressor(BaseEstimator, RegressorMixin):
    """Anisotropic Support Vector Regression with tuned epsilon margin.

    Scientific Reference:
    - Clarke, S. M., Griebsch, J. H., & Simpson, T. W. (2005).
      Analysis of support vector regression for approximation of complex
      engineering analyses.
      Journal of Mechanical Design, 127(6), 1077-1087.
    """

    def __init__(
        self,
        C: float = 10.0,
        epsilon: float = 0.05,
        kernel: str = "rbf",
    ) -> None:
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.pipe_: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MdaoSVRRegressor:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        target = y_arr.ravel() if y_arr.ndim > 1 and y_arr.shape[1] == 1 else y_arr

        self.pipe_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)),
            ]
        )
        self.pipe_.fit(X_arr, target)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if self.pipe_ is None:
            raise RuntimeError("Model must be fitted before predict.")

        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        pred = self.pipe_.predict(X_arr)
        res = pred.ravel() if single else pred
        if not return_std:
            return res
        return res, np.zeros_like(res)


class MdaoEnsembleOfSurrogates(BaseEstimator, RegressorMixin):
    """Goel-Haftka Optimal Convex Ensemble of Surrogates (EoS).

    Solves the quadratic optimization problem for optimal convex combination
    weights minimizing cross-validation error covariance (GMSE).

    Scientific Reference:
    - Goel, T., Haftka, R. T., Shyy, W., & Queipo, N. V. (2007).
      Ensemble of surrogates.
      Structural and Multidisciplinary Optimization, 33(3), 199-216.
    - Viana, F. A., Haftka, R. T., & Steffen, V. (2009).
      Multiple surrogates: how to estimate the accuracy of an ensemble of surrogates.
      AIAA Journal, 47(8), 1947-1957.
    """

    def __init__(
        self,
        models: list[tuple[str, Any]],
        cv_error_matrix: np.ndarray | None = None,
    ) -> None:
        self.models = models  # list of (name, fitted_model)
        self.names = [m[0] for m in models]
        self.weights_: np.ndarray | None = None
        if cv_error_matrix is not None:
            self.weights_ = self._solve_optimal_weights(cv_error_matrix)
        else:
            # Equal weighting baseline
            self.weights_ = np.ones(len(models)) / max(1, len(models))

    @staticmethod
    def _solve_optimal_weights(error_matrix: np.ndarray) -> np.ndarray:
        """Solve min_w w^T C w subject to sum(w) = 1, w >= 0.

        C is the error covariance matrix across validation samples.
        """
        k = error_matrix.shape[1]
        if k == 1:
            return np.array([1.0])

        # Error covariance matrix C_ij = (1/N) * sum(e_i * e_j)
        C = (error_matrix.T @ error_matrix) / max(1, len(error_matrix))
        # Numerical jitter to guarantee positive-definiteness
        C += np.eye(k) * 1e-8

        def loss(w):
            return 0.5 * float(w.T @ C @ w)

        def loss_grad(w):
            return C @ w

        w0 = np.ones(k) / k
        bounds = [(0.0, 1.0) for _ in range(k)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            res = optimize.minimize(
                loss,
                w0,
                jac=loss_grad,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
            if res.success and np.all(res.x >= -1e-6):
                w = np.clip(res.x, 0.0, 1.0)
                return w / np.sum(w)
        except Exception:
            pass

        # Fallback to inverse RMSE weighting
        rmses = np.sqrt(np.diag(C))
        inv_rmses = 1.0 / np.maximum(rmses, 1e-8)
        return inv_rmses / np.sum(inv_rmses)

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)

        preds = []
        for _, model in self.models:
            p = model.predict(X_arr)
            preds.append(np.asarray(p, dtype=np.float64))

        stacked = np.stack(preds, axis=0)  # (k, N)
        weights = self.weights_[:, None] if self.weights_ is not None else np.ones((len(self.models), 1)) / len(self.models)
        ensemble_pred = np.sum(stacked * weights, axis=0)

        res = ensemble_pred.ravel() if single else ensemble_pred
        if not return_std:
            return res

        # Inter-model epistemic disagreement
        inter_std = np.std(stacked, axis=0)
        return res, (inter_std.ravel() if single else inter_std)
