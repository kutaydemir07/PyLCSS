# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Training strategies for tabular surrogate regressors."""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .contracts import LossCallback, Metrics, ProgressCallback, StopFlag
from .metrics import empty_metrics, evaluate_predictions, validate_training_data
from .models import TORCH_AVAILABLE, UncertaintyRegressor

logger = logging.getLogger(__name__)

TrainingResult = tuple[Any, Metrics]


class SurrogateModelStrategy(ABC):
    """Interface implemented by every surrogate training backend."""

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
        """Fit a model and return it with UI-compatible metrics."""


class _SklearnStrategy(SurrogateModelStrategy):
    """Shared validation, holdout, and evaluation behavior."""

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        features, targets = validate_training_data(X, y)
        debug_mode = bool(config.get("debug_mode", False))
        if debug_mode:
            return features, targets, features, targets, "training (debug)"

        if (X_test is None) != (y_test is None):
            raise ValueError(
                "X_test and y_test must either both be provided or both be omitted."
            )
        if X_test is not None and y_test is not None:
            raw_test_X = np.asarray(X_test)
            raw_test_y = np.asarray(y_test)
            if raw_test_X.size == 0 or raw_test_y.size == 0:
                logger.warning(
                    "No holdout samples were provided; reporting training-set metrics."
                )
                return features, targets, features, targets, "training"
            test_features, test_targets = validate_training_data(
                raw_test_X,
                raw_test_y,
                minimum_samples=1,
            )
            if test_features.shape[1] != features.shape[1]:
                raise ValueError(
                    "Training and test feature counts differ: "
                    f"{features.shape[1]} != {test_features.shape[1]}."
                )
            if test_targets.shape[1] != targets.shape[1]:
                raise ValueError(
                    "Training and test output counts differ: "
                    f"{targets.shape[1]} != {test_targets.shape[1]}."
                )
            return features, targets, test_features, test_targets, "holdout"

        validation_fraction = float(config.get("validation_split", 0.2))
        if not 0.0 < validation_fraction < 1.0:
            logger.warning(
                "validation_split=%s leaves no holdout; "
                "reporting training-set metrics.",
                validation_fraction,
            )
            return features, targets, features, targets, "training"
        random_state = int(config.get("random_state", 42))
        train_X, test_X, train_y, test_y = train_test_split(
            features,
            targets,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
        )
        return train_X, train_y, test_X, test_y, "holdout"

    @staticmethod
    def _evaluate(
        model: Any,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        config: Mapping[str, Any],
        source: str,
    ) -> Metrics:
        if X_eval.shape[0] == 0:
            return empty_metrics(debug_mode=bool(config.get("debug_mode", False)))
        prediction = model.predict(X_eval)
        metrics = evaluate_predictions(y_eval, prediction)
        metrics["debug_mode"] = bool(config.get("debug_mode", False))
        metrics["evaluation_source"] = source
        return metrics


def _parse_hidden_layers(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parsed = value

    widths: tuple[int, ...]
    if isinstance(parsed, int):
        widths = (parsed,)
    elif isinstance(parsed, list | tuple):
        try:
            widths = tuple(int(width) for width in parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden_layers must contain integer widths.") from exc
    else:
        widths = default
    if not widths or any(width < 1 for width in widths):
        raise ValueError("hidden_layers must contain positive integer widths.")
    return widths


class MLPStrategy(_SklearnStrategy):
    """Scaled scikit-learn multilayer-perceptron regression."""

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
        train_X, train_y, eval_X, eval_y, source = self._split(
            X, y, config, X_test, y_test
        )
        if stop_flag and stop_flag():
            raise RuntimeError("Training was cancelled before model fitting started.")

        solver = str(config.get("solver", "adam")).lower()
        if solver not in {"adam", "lbfgs", "sgd"}:
            raise ValueError(f"Unsupported MLP solver: {solver!r}.")
        use_early_stopping = (
            bool(config.get("early_stopping", False))
            and solver != "lbfgs"
            and train_X.shape[0] >= 20
            and not bool(config.get("debug_mode", False))
        )
        regressor = MLPRegressor(
            hidden_layer_sizes=_parse_hidden_layers(
                config.get("hidden_layers", (100, 50)),
                default=(100, 50),
            ),
            activation=str(config.get("activation", "relu")),
            solver=solver,
            alpha=float(config.get("alpha", 1e-4)),
            max_iter=int(config.get("max_iter", 2_000)),
            early_stopping=use_early_stopping,
            validation_fraction=float(config.get("early_stopping_fraction", 0.1)),
            n_iter_no_change=int(config.get("n_iter_no_change", 20)),
            random_state=int(config.get("random_state", 42)),
        )
        model = TransformedTargetRegressor(
            regressor=Pipeline(
                [("scale_features", StandardScaler()), ("regressor", regressor)]
            ),
            transformer=StandardScaler(),
        )
        if callback:
            callback(85, "Training MLP regressor...")
        model.fit(train_X, train_y)

        if loss_callback:
            try:
                fitted_mlp = model.regressor_.named_steps["regressor"]
                curve = list(fitted_mlp.loss_curve_)
                stride = max(1, len(curve) // 100)
                for epoch, loss in enumerate(curve):
                    if epoch % stride == 0 or epoch == len(curve) - 1:
                        loss_callback(
                            {"epoch": epoch, "train": float(loss), "val": float(loss)}
                        )
            except (AttributeError, TypeError):
                logger.debug(
                    "The fitted MLP did not expose a loss curve.", exc_info=True
                )

        return model, self._evaluate(model, eval_X, eval_y, config, source)


class GaussianProcessStrategy(_SklearnStrategy):
    """Matern Gaussian process with scaled features and optional target scaling."""

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
        train_X, train_y, eval_X, eval_y, source = self._split(
            X, y, config, X_test, y_test
        )
        if stop_flag and stop_flag():
            raise RuntimeError("Training was cancelled before model fitting started.")

        n_features = train_X.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        ) + WhiteKernel(
            noise_level=float(config.get("noise_level", 1e-5)),
            noise_level_bounds=(1e-10, 1e1),
        )
        optimize_kernel = bool(config.get("optimize_kernel", True))
        regressor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(config.get("alpha", 1e-6)),
            optimizer="fmin_l_bfgs_b" if optimize_kernel else None,
            n_restarts_optimizer=(
                int(config.get("n_restarts_optimizer", 10)) if optimize_kernel else 0
            ),
            normalize_y=False,
            random_state=int(config.get("random_state", 42)),
        )
        target_transformer: Any
        if bool(config.get("normalize_y", True)):
            target_transformer = StandardScaler()
        else:
            target_transformer = FunctionTransformer(validate=True)
        transformed = TransformedTargetRegressor(
            regressor=Pipeline(
                [("scale_features", StandardScaler()), ("regressor", regressor)]
            ),
            transformer=target_transformer,
        )
        if callback:
            callback(85, "Training Gaussian process...")
        transformed.fit(train_X, train_y)
        model = UncertaintyRegressor(
            transformed,
            UncertaintyRegressor._GAUSSIAN_PROCESS,
        )
        return model, self._evaluate(model, eval_X, eval_y, config, source)


class RandomForestStrategy(_SklearnStrategy):
    """Random-forest surrogate with ensemble-spread uncertainty."""

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
        train_X, train_y, eval_X, eval_y, source = self._split(
            X, y, config, X_test, y_test
        )
        if callback:
            callback(85, "Training random forest...")
        regressor = RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=config.get("max_depth"),
            min_samples_split=int(config.get("min_samples_split", 2)),
            min_samples_leaf=int(config.get("min_samples_leaf", 1)),
            bootstrap=bool(config.get("bootstrap", True)),
            random_state=int(config.get("random_state", 42)),
            n_jobs=int(config.get("n_jobs", -1)),
        )
        fit_y = train_y.ravel() if train_y.shape[1] == 1 else train_y
        regressor.fit(train_X, fit_y)
        model = UncertaintyRegressor(
            regressor,
            UncertaintyRegressor._RANDOM_FOREST,
        )
        return model, self._evaluate(model, eval_X, eval_y, config, source)


class GradientBoostingStrategy(_SklearnStrategy):
    """Independent gradient-boosting regressor per output channel."""

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
        train_X, train_y, eval_X, eval_y, source = self._split(
            X, y, config, X_test, y_test
        )
        if callback:
            callback(85, "Training gradient boosting...")
        base = GradientBoostingRegressor(
            n_estimators=int(config.get("n_estimators", 100)),
            learning_rate=float(config.get("learning_rate", 0.1)),
            max_depth=int(config.get("max_depth", 3)),
            subsample=float(config.get("subsample", 1.0)),
            loss=str(config.get("loss", "squared_error")),
            random_state=int(config.get("random_state", 42)),
        )
        model = MultiOutputRegressor(base, n_jobs=int(config.get("n_jobs", 1)))
        model.fit(train_X, train_y)
        return model, self._evaluate(model, eval_X, eval_y, config, source)


class PyTorchStrategy(SurrogateModelStrategy):
    """Adapter that delegates the custom PyTorch loop to the facade."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

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
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is unavailable; install it or choose an sklearn model."
            )
        return self.trainer._train_torch_model(
            X,
            y,
            config,
            callback,
            stop_flag,
            loss_callback,
            X_test,
            y_test,
        )


__all__ = [
    "GaussianProcessStrategy",
    "GradientBoostingStrategy",
    "MLPStrategy",
    "PyTorchStrategy",
    "RandomForestStrategy",
    "SurrogateModelStrategy",
]
