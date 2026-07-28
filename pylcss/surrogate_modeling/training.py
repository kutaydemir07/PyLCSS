# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public facade for surrogate data generation and model training."""

from __future__ import annotations

import copy
import importlib.util
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .contracts import LossCallback, Metrics, ProgressCallback, SpyPort, StopFlag
from .data_generation import (
    DILL_AVAILABLE,
    JOBLIB_AVAILABLE,
    QMC_AVAILABLE,
    generate_spy_data,
)
from .metrics import evaluate_predictions, validate_training_data
from .models import (
    TORCH_AVAILABLE,
    RegressionMLP,
    TorchRegressor,
)

logger = logging.getLogger(__name__)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def evaluate_model_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    max_samples: int = 100,
) -> Metrics:
    """Compatibility wrapper for the former training-engine helper."""
    return evaluate_predictions(y_true, y_pred, max_samples=max_samples)


class SurrogateTrainer:
    """Coordinate data generation and model-specific training strategies."""

    def __init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for surrogate modeling.")

        from .strategies import (
            GaussianProcessStrategy,
            GradientBoostingStrategy,
            MLPStrategy,
            PyTorchStrategy,
            RandomForestStrategy,
            SurrogateModelStrategy,
        )

        self.strategies: dict[str, SurrogateModelStrategy] = {
            "MLP Regressor": MLPStrategy(),
            "Gaussian Process": GaussianProcessStrategy(),
            "Gaussian Process (Kriging)": GaussianProcessStrategy(),
            "Random Forest": RandomForestStrategy(),
            "Gradient Boosting": GradientBoostingStrategy(),
        }
        if TORCH_AVAILABLE:
            self.strategies["Deep Neural Network (PyTorch)"] = PyTorchStrategy(self)
            try:
                from .geometry import TRIMESH_AVAILABLE

                if TRIMESH_AVAILABLE:
                    from .geometry_training import (
                        GeomDeepONetStrategy,
                        GINOStrategy,
                    )

                    self.strategies["Geom-DeepONet"] = GeomDeepONetStrategy()
                    self.strategies["GINO"] = GINOStrategy()
            except ImportError:
                logger.debug(
                    "Geometry-aware surrogate strategies are unavailable.",
                    exc_info=True,
                )

    @property
    def available_models(self) -> tuple[str, ...]:
        """Names accepted by :meth:`train_model`, in UI display order."""
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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
        list[str],
    ]:
        """Generate train/test arrays by evaluating a compiled spy model."""
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
        """Train the configured model or fail with a list of supported names."""
        model_type = str(config.get("model_type", "MLP Regressor"))
        strategy = self.strategies.get(model_type)
        if strategy is None:
            supported = ", ".join(self.available_models)
            raise ValueError(
                f"Unknown model type {model_type!r}. Available models: {supported}."
            )
        if callback:
            callback(80, f"Training {model_type}...")
        return strategy.train(
            X,
            y,
            config,
            X_test,
            y_test,
            callback,
            stop_flag,
            loss_callback,
        )

    def _train_torch_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        callback: ProgressCallback | None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> tuple[TorchRegressor, Metrics]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for neural-network training.")

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        from .geometry_training import select_torch_device
        from .strategies import _parse_hidden_layers

        features, targets = validate_training_data(X, y)
        debug_mode = bool(config.get("debug_mode", False))
        random_state = int(config.get("random_state", 42))
        torch.manual_seed(random_state)

        if debug_mode:
            train_X, train_y = features, targets
            eval_X, eval_y = features, targets
            evaluation_source = "training (debug)"
        elif (X_test is None) != (y_test is None):
            raise ValueError(
                "X_test and y_test must both be provided or both be omitted."
            )
        elif X_test is not None and y_test is not None:
            if np.asarray(X_test).size == 0 or np.asarray(y_test).size == 0:
                train_X, train_y = features, targets
                eval_X, eval_y = features, targets
                evaluation_source = "training"
            else:
                eval_X, eval_y = validate_training_data(
                    X_test,
                    y_test,
                    minimum_samples=1,
                )
                if eval_X.shape[1] != features.shape[1]:
                    raise ValueError("Training and test feature counts differ.")
                if eval_y.shape[1] != targets.shape[1]:
                    raise ValueError("Training and test output counts differ.")
                train_X, train_y = features, targets
                evaluation_source = "holdout"
        else:
            validation_fraction = float(config.get("validation_split", 0.2))
            if 0.0 < validation_fraction < 1.0:
                train_X, eval_X, train_y, eval_y = train_test_split(
                    features,
                    targets,
                    test_size=validation_fraction,
                    random_state=random_state,
                    shuffle=True,
                )
                evaluation_source = "holdout"
            else:
                train_X, train_y = features, targets
                eval_X, eval_y = features, targets
                evaluation_source = "training"

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        train_X_scaled = scaler_x.fit_transform(train_X)
        train_y_scaled = scaler_y.fit_transform(train_y)
        eval_X_scaled = scaler_x.transform(eval_X)
        eval_y_scaled = scaler_y.transform(eval_y)

        train_X_tensor = torch.as_tensor(train_X_scaled, dtype=torch.float32)
        train_y_tensor = torch.as_tensor(train_y_scaled, dtype=torch.float32)
        device = select_torch_device()
        eval_X_tensor = torch.as_tensor(
            eval_X_scaled,
            dtype=torch.float32,
            device=device,
        )
        eval_y_tensor = torch.as_tensor(
            eval_y_scaled,
            dtype=torch.float32,
            device=device,
        )

        hidden_dims = list(
            _parse_hidden_layers(
                config.get("hidden_layers", (64, 64)),
                default=(64, 64),
            )
        )
        dropout = float(config.get("dropout", 0.1))
        network = RegressionMLP(
            features.shape[1],
            hidden_dims,
            targets.shape[1],
            dropout=dropout,
            activation=str(config.get("activation", "relu")),
        ).to(device)
        criterion = nn.MSELoss()
        learning_rate = float(config.get("learning_rate", 0.01))
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        optimizer_name = str(config.get("optimizer", "Adam"))
        optimizer_factories: dict[str, Any] = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adagrad": optim.Adagrad,
        }
        try:
            optimizer_factory = optimizer_factories[optimizer_name]
        except KeyError as exc:
            supported = ", ".join(optimizer_factories)
            raise ValueError(
                f"Unknown optimizer {optimizer_name!r}; expected {supported}."
            ) from exc
        optimizer = optimizer_factory(network.parameters(), lr=learning_rate)

        epochs = int(config.get("epochs", 5_000))
        batch_size = int(config.get("batch_size", 32))
        patience = int(config.get("patience", 50))
        if epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive.")

        generator = torch.Generator().manual_seed(random_state)
        loader = DataLoader(
            TensorDataset(train_X_tensor, train_y_tensor),
            batch_size=min(batch_size, len(train_X_tensor)),
            shuffle=True,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        best_validation_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        last_update = 0.0

        for epoch in range(epochs):
            if stop_flag and stop_flag():
                logger.info("PyTorch training cancelled after epoch %d.", epoch)
                break
            network.train()
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device, non_blocking=device.type == "cuda")
                batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                output = network(batch_X)
                loss = criterion(output, batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Training produced a non-finite loss at epoch {epoch}."
                    )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
            training_loss = total_loss / len(loader)

            network.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(network(eval_X_tensor), eval_y_tensor).item()
                )
            if validation_loss < best_validation_loss - 1e-12:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            now = time.monotonic()
            if loss_callback and (epoch % 10 == 0 or now - last_update > 0.5):
                loss_callback(
                    {
                        "epoch": epoch,
                        "train": training_loss,
                        "val": validation_loss,
                    }
                )
                last_update = now
            if callback and (epoch % 50 == 0 or epoch == epochs - 1):
                progress = 80 + int(15 * (epoch + 1) / epochs)
                callback(
                    progress,
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train={training_loss:.4g}, val={validation_loss:.4g}",
                )
            if epochs_without_improvement >= patience:
                logger.info("PyTorch training stopped early at epoch %d.", epoch + 1)
                break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            prediction_scaled = network(eval_X_tensor).detach().cpu().numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)
        metrics = evaluate_predictions(eval_y, prediction)
        metrics["debug_mode"] = debug_mode
        metrics["evaluation_source"] = evaluation_source

        wrapped = TorchRegressor(
            network,
            scaler_x,
            scaler_y,
            n_mc_samples=int(config.get("n_mc_samples", 50)),
        )
        _, prediction_std = wrapped.predict(eval_X, return_std=True)
        metrics["y_std"] = np.asarray(prediction_std)[:100].tolist()
        return wrapped, metrics

    # Compatibility for callers that reached into the old implementation.
    def _train_pytorch_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        callback: ProgressCallback | None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> tuple[TorchRegressor, Metrics]:
        return self._train_torch_model(
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
    "DILL_AVAILABLE",
    "JOBLIB_AVAILABLE",
    "QMC_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "TORCH_AVAILABLE",
    "SurrogateTrainer",
    "evaluate_model_predictions",
]
