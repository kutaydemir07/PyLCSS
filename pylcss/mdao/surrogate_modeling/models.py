# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Serializable estimator wrappers used by surrogate-model backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

Prediction: TypeAlias = NDArray[np.float64]


class UncertaintyRegressor:
    """Add a uniform ``predict(..., return_std=True)`` API to sklearn models."""

    _GAUSSIAN_PROCESS = "Gaussian Process (Kriging)"
    _RANDOM_FOREST = "Random Forest"

    def __init__(self, model: Any, model_type: str) -> None:
        self.model = model
        self.model_type = model_type

    @overload
    def predict(self, X: ArrayLike, return_std: Literal[False] = False) -> Any: ...

    @overload
    def predict(
        self,
        X: ArrayLike,
        return_std: Literal[True],
    ) -> tuple[Any, NDArray[np.float64]]: ...

    def predict(self, X: ArrayLike, return_std: bool = False) -> Any:
        """Predict means and, where supported, calibrated standard deviations."""
        if not return_std:
            return self.model.predict(X)

        if self.model_type == self._GAUSSIAN_PROCESS:
            return self._predict_gaussian_process(X)
        if self.model_type == self._RANDOM_FOREST:
            return self._predict_random_forest(X)

        mean = np.asarray(self.model.predict(X), dtype=np.float64)
        return mean, np.zeros_like(mean)

    def _predict_gaussian_process(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reach the GP through fitted input/target transforms."""
        if not hasattr(self.model, "regressor_") or not hasattr(
            self.model, "transformer_"
        ):
            mean, std = self.model.predict(X, return_std=True)
            return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)

        pipeline = self.model.regressor_
        regressor = pipeline.named_steps["regressor"]
        features = pipeline[:-1].transform(X)
        mean_scaled, std_scaled = regressor.predict(features, return_std=True)

        mean_2d = np.asarray(mean_scaled, dtype=np.float64)
        std_2d = np.asarray(std_scaled, dtype=np.float64)
        mean_was_1d = mean_2d.ndim == 1
        if mean_was_1d:
            mean_2d = mean_2d.reshape(-1, 1)
        if std_2d.ndim == 1:
            std_2d = std_2d.reshape(-1, 1)

        mean = np.asarray(
            self.model.transformer_.inverse_transform(mean_2d),
            dtype=np.float64,
        )
        target_scale = getattr(self.model.transformer_, "scale_", None)
        std = std_2d if target_scale is None else std_2d * target_scale
        if mean_was_1d:
            return mean.ravel(), std.ravel()
        return mean, std

    def _predict_random_forest(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        estimators = getattr(self.model, "estimators_", None)
        if not estimators:
            raise RuntimeError(
                "Random forest must be fitted before requesting uncertainty."
            )
        predictions = np.asarray(
            [tree.predict(X) for tree in estimators],
            dtype=np.float64,
        )
        return predictions.mean(axis=0), predictions.std(axis=0)

    def __getattr__(self, name: str) -> Any:
        """Expose fitted-estimator attributes such as ``feature_importances_``."""
        if name in {"model", "model_type"}:
            raise AttributeError(name)
        return getattr(self.model, name)


if TORCH_AVAILABLE:
    _ActivationFactory: TypeAlias = type[nn.Module]
    _ACTIVATIONS: dict[str, _ActivationFactory] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "logistic": nn.Sigmoid,
        "identity": nn.Identity,
        "gelu": nn.GELU,
    }

    class RegressionMLP(nn.Module):
        """Small configurable multilayer perceptron for tabular regression."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            output_dim: int,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if input_dim < 1 or output_dim < 1:
                raise ValueError("input_dim and output_dim must be positive.")
            if not hidden_dims or any(width < 1 for width in hidden_dims):
                raise ValueError("hidden_dims must contain positive layer widths.")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in the half-open interval [0, 1).")
            try:
                activation_factory = _ACTIVATIONS[activation.lower()]
            except KeyError as exc:
                supported = ", ".join(sorted(_ACTIVATIONS))
                raise ValueError(
                    f"Unknown activation {activation!r}; expected one of: {supported}."
                ) from exc

            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(previous, width),
                        activation_factory(),
                        nn.Dropout(dropout),
                    ]
                )
                previous = width
            layers.append(nn.Linear(previous, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TorchRegressor:
        """Scikit-learn-style facade around a PyTorch regression network."""

        def __init__(
            self,
            model: nn.Module,
            scaler_x: Any,
            scaler_y: Any,
            n_mc_samples: int = 50,
        ) -> None:
            if n_mc_samples < 1:
                raise ValueError("n_mc_samples must be at least 1.")
            self.model = model
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y
            self.n_mc_samples = int(n_mc_samples)
            self.device = next(model.parameters()).device

        @classmethod
        def load(
            cls,
            filepath: str | Path,
            *,
            trusted: bool = False,
        ) -> TorchRegressor:
            """Load trusted joblib data and refresh its device metadata.

            Joblib uses pickle-compatible deserialization and can execute code.
            Callers must explicitly confirm that the file and its source are
            trusted before passing ``trusted=True``.
            """
            if not trusted:
                raise ValueError(
                    "Refusing to load an untrusted joblib model. Verify the file "
                    "and pass trusted=True explicitly."
                )
            loaded = joblib.load(filepath)
            if not isinstance(loaded, cls):
                raise TypeError(
                    f"{filepath!s} contains {type(loaded).__name__}, "
                    f"not {cls.__name__}."
                )
            loaded.device = next(loaded.model.parameters()).device
            return loaded

        @overload
        def predict(
            self,
            X: ArrayLike,
            return_std: Literal[False] = False,
        ) -> Prediction: ...

        @overload
        def predict(
            self,
            X: ArrayLike,
            return_std: Literal[True],
        ) -> tuple[Prediction, Prediction]: ...

        def predict(
            self,
            X: ArrayLike,
            return_std: bool = False,
        ) -> Prediction | tuple[Prediction, Prediction]:
            features = np.asarray(X, dtype=np.float64)
            single_sample = features.ndim == 1
            if single_sample:
                features = features.reshape(1, -1)
            if features.ndim != 2:
                raise ValueError(
                    f"X must be one- or two-dimensional; got {features.shape}."
                )
            if not np.isfinite(features).all():
                raise ValueError("X contains NaN or infinite values.")

            scaled = self.scaler_x.transform(features)
            tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)

            if not return_std:
                self.model.eval()
                with torch.no_grad():
                    output_scaled = self.model(tensor).detach().cpu().numpy()
                prediction = np.asarray(
                    self.scaler_y.inverse_transform(output_scaled),
                    dtype=np.float64,
                )
                return prediction.squeeze() if single_sample else prediction

            was_training = self.model.training
            self.model.train()
            try:
                batch_size = tensor.shape[0]
                repeated = tensor.repeat(self.n_mc_samples, 1)
                with torch.no_grad():
                    output = self.model(repeated).detach().cpu().numpy()
                samples = output.reshape(self.n_mc_samples, batch_size, -1)
            finally:
                self.model.train(was_training)

            mean_scaled = samples.mean(axis=0)
            std_scaled = samples.std(axis=0)
            mean = np.asarray(
                self.scaler_y.inverse_transform(mean_scaled),
                dtype=np.float64,
            )
            target_scale = getattr(self.scaler_y, "scale_", None)
            if target_scale is not None:
                std = np.asarray(std_scaled * target_scale, dtype=np.float64)
            else:
                flattened = samples.reshape(self.n_mc_samples * batch_size, -1)
                unscaled = self.scaler_y.inverse_transform(flattened)
                std = np.asarray(
                    unscaled.reshape(self.n_mc_samples, batch_size, -1).std(axis=0),
                    dtype=np.float64,
                )
            if single_sample:
                return mean.squeeze(), std.squeeze()
            return mean, std

        def to_cpu(self) -> TorchRegressor:
            """Move the network to CPU before portable serialization."""
            self.model.cpu()
            self.device = torch.device("cpu")
            return self

    # Compatibility names for models serialized by PyLCSS 2.2 and earlier.
    ConfigurableNet = RegressionMLP
    PyTorchWrapper = TorchRegressor

else:

    class _TorchRequired:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is required for this model.")

    class RegressionMLP(_TorchRequired):  # type: ignore[no-redef]
        """Unavailable placeholder that fails with an actionable message."""

    class TorchRegressor(_TorchRequired):  # type: ignore[no-redef]
        """Unavailable placeholder that fails with an actionable message."""

    ConfigurableNet = RegressionMLP  # type: ignore[misc]
    PyTorchWrapper = TorchRegressor  # type: ignore[misc]


# Compatibility name for objects imported from the former training engine.
UncertaintyWrapper = UncertaintyRegressor

__all__ = [
    "TORCH_AVAILABLE",
    "ConfigurableNet",
    "PyTorchWrapper",
    "RegressionMLP",
    "TorchRegressor",
    "UncertaintyRegressor",
    "UncertaintyWrapper",
]
