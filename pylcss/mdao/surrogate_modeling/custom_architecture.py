# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Deep learning custom architecture engine for surrogate modeling.
Supports visual layer stacks, residual connections, Fourier embeddings,
and arbitrary user-defined PyTorch architectures."""

from __future__ import annotations

import copy
import inspect
import logging
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .contracts import LossCallback, Metrics, ProgressCallback, StopFlag
from .dataset import prepare_training_data
from .metrics import evaluate_predictions, format_prediction, prediction_features
from .strategies import SurrogateModelStrategy
from .torch_utils import TORCH_AVAILABLE, prediction_mode, select_torch_device

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

__all__ = [
    "CustomArchitectureModel",
    "CustomArchitectureStrategy",
    "CustomNeuralNet",
    "FourierFeatureEmbedding",
    "ResidualBlock",
    "compile_custom_code",
]


if TORCH_AVAILABLE:

    class FourierFeatureEmbedding(nn.Module):
        """Random Fourier Feature projection for high-frequency physics responses."""

        def __init__(
            self, in_features: int, num_features: int = 32, scale: float = 1.0
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.num_features = num_features
            # Fixed random projection matrix B ~ N(0, scale^2)
            B = torch.randn(in_features, num_features) * scale
            self.register_buffer("B", B)
            self.out_features = num_features * 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            projected = 2.0 * math.pi * torch.matmul(x, self.B)
            return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    class ResidualBlock(nn.Module):
        """Pre-activation or post-activation Residual Block for deep regression networks."""

        def __init__(
            self,
            dim: int,
            hidden_dim: int | None = None,
            activation: str = "gelu",
            dropout: float = 0.0,
            use_norm: bool = True,
        ) -> None:
            super().__init__()
            hidden = hidden_dim or dim
            self.linear1 = nn.Linear(dim, hidden)
            self.linear2 = nn.Linear(hidden, dim)
            self.norm1 = nn.LayerNorm(dim) if use_norm else nn.Identity()
            self.norm2 = nn.LayerNorm(hidden) if use_norm else nn.Identity()
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.act = _get_activation(activation)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.norm1(x)
            out = self.act(self.linear1(out))
            out = self.dropout(out)
            out = self.norm2(out)
            out = self.linear2(out)
            out = self.dropout(out)
            return residual + out

    def _get_activation(name: str) -> nn.Module:
        name_lower = str(name).lower()
        if name_lower in ("gelu",):
            return nn.GELU()
        elif name_lower in ("silu", "swish"):
            return nn.SiLU()
        elif name_lower in ("mish",):
            return nn.Mish()
        elif name_lower in ("tanh",):
            return nn.Tanh()
        elif name_lower in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU(0.01)
        elif name_lower in ("elu",):
            return nn.ELU()
        elif name_lower in ("sigmoid", "logistic"):
            return nn.Sigmoid()
        elif name_lower in ("identity", "none", "linear"):
            return nn.Identity()
        elif name_lower == "relu":
            return nn.ReLU()
        raise ValueError(f"Unknown activation {name!r}.")

    class CustomNeuralNet(nn.Module):
        """Dynamically configured neural network built from layer specifications."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            spec: dict[str, Any],
        ) -> None:
            super().__init__()
            if in_features < 1 or out_features < 1:
                raise ValueError("Input and output widths must be positive.")
            self.in_features = in_features
            self.out_features = out_features

            layers: list[nn.Module] = []
            curr_dim = in_features

            # Optional Fourier Embedding at input
            fourier_spec = spec.get("fourier_features", {})
            if fourier_spec.get("enabled", False):
                num_fourier = int(fourier_spec.get("num_features", 32))
                scale = float(fourier_spec.get("scale", 1.0))
                fourier = FourierFeatureEmbedding(curr_dim, num_fourier, scale)
                layers.append(fourier)
                curr_dim = fourier.out_features

            layer_specs = spec.get("layers", [])
            for l_spec in layer_specs:
                l_type = str(l_spec.get("type", "")).lower()

                if l_type == "linear":
                    units = int(l_spec.get("units", 64))
                    layers.append(nn.Linear(curr_dim, units))
                    curr_dim = units
                    # Visual-builder linear rows carry their own activation
                    # and dropout. Older explicit layer stacks omit these
                    # keys and retain their separate activation/dropout rows.
                    act_name = str(l_spec.get("activation", "none"))
                    if act_name.lower() not in {"none", "identity", "linear"}:
                        layers.append(_get_activation(act_name))
                    dropout = float(l_spec.get("dropout", 0.0))
                    if dropout:
                        layers.append(nn.Dropout(dropout))

                elif l_type == "activation":
                    act_name = str(l_spec.get("activation", "relu"))
                    layers.append(_get_activation(act_name))

                elif l_type == "resblock":
                    units = int(l_spec.get("units", curr_dim))
                    if curr_dim != units:
                        layers.append(nn.Linear(curr_dim, units))
                        curr_dim = units
                    dropout = float(l_spec.get("dropout", 0.0))
                    act_name = str(l_spec.get("activation", "gelu"))
                    layers.append(
                        ResidualBlock(
                            curr_dim, curr_dim, activation=act_name, dropout=dropout
                        )
                    )

                elif l_type == "layernorm":
                    layers.append(nn.LayerNorm(curr_dim))

                elif l_type == "batchnorm":
                    layers.append(nn.BatchNorm1d(curr_dim))

                elif l_type == "dropout":
                    rate = float(l_spec.get("rate", l_spec.get("dropout", 0.1)))
                    layers.append(nn.Dropout(rate))
                else:
                    raise ValueError(f"Unknown layer type {l_type!r}.")

            # Final projection to target output dimension if not already matching
            trainable = any(
                p.requires_grad for layer in layers for p in layer.parameters()
            )
            if curr_dim != out_features or not trainable:
                layers.append(nn.Linear(curr_dim, out_features))

            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def compile_custom_code(
        code_string: str, in_features: int, out_features: int
    ) -> nn.Module:
        """Execute trusted custom code and validate its regression interface."""
        local_scope: dict[str, Any] = {
            "__name__": "_pylcss_custom_architecture",
            "torch": torch,
            "nn": nn,
            "math": math,
            "FourierFeatureEmbedding": FourierFeatureEmbedding,
            "ResidualBlock": ResidualBlock,
        }
        exec(code_string, local_scope)

        # Find user-defined nn.Module subclass
        target_cls = None
        for val in local_scope.values():
            if (
                isinstance(val, type)
                and val.__module__ == local_scope["__name__"]
                and issubclass(val, nn.Module)
                and val
                not in (
                    nn.Module,
                    FourierFeatureEmbedding,
                    ResidualBlock,
                )
            ):
                target_cls = val
                break

        if target_cls is None:
            raise ValueError(
                "Custom code must define a subclass of torch.nn.Module (for example: class CustomSurrogate(nn.Module): ...)"
            )

        # Instantiate with (in_features, out_features)
        signature = inspect.signature(target_cls)
        try:
            signature.bind(in_features, out_features)
        except TypeError:
            signature.bind()
            model = target_cls()
        else:
            model = target_cls(in_features, out_features)

        # Sanity check forward pass
        test_in = torch.randn(2, in_features)
        with prediction_mode(model):
            out = model(test_in)
        if not isinstance(out, torch.Tensor) or out.shape != (2, out_features):
            raise ValueError(
                f"Model must return shape (2, {out_features}); got {getattr(out, 'shape', None)}."
            )
        if not torch.isfinite(out).all():
            raise ValueError("Model returned non-finite values during validation.")
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("Custom architecture must contain trainable parameters.")

        return model


class CustomArchitectureModel:
    """Scikit-learn compatible facade for custom PyTorch surrogate models."""

    def __init__(
        self,
        model: Any,
        scaler_x: Any,
        scaler_y: Any,
        n_mc_samples: int = 50,
        spec: dict[str, Any] | None = None,
        custom_code: str | None = None,
    ) -> None:
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.n_mc_samples = int(n_mc_samples)
        if self.n_mc_samples < 1:
            raise ValueError("n_mc_samples must be positive.")
        self.spec = spec or {}
        self.custom_code = custom_code
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            self.device = next(model.parameters()).device
        else:
            self.device = None

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        features, single_sample = prediction_features(X)

        scaled = self.scaler_x.transform(features)
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)

        if not return_std:
            with prediction_mode(self.model):
                out_scaled = self.model(tensor).detach().cpu().numpy()
            pred = np.asarray(
                self.scaler_y.inverse_transform(out_scaled), dtype=np.float64
            )
            return format_prediction(pred, single=single_sample)

        # Monte Carlo Dropout for uncertainty quantification
        with prediction_mode(self.model, dropout=True):
            batch_size = tensor.shape[0]
            repeated = tensor.repeat(self.n_mc_samples, 1)
            raw_out = self.model(repeated).detach().cpu().numpy()
            samples = raw_out.reshape(self.n_mc_samples, batch_size, -1)

        mean_scaled = samples.mean(axis=0)
        std_scaled = samples.std(axis=0)
        mean = np.asarray(
            self.scaler_y.inverse_transform(mean_scaled), dtype=np.float64
        )

        scale_val = getattr(self.scaler_y, "scale_", None)
        if scale_val is not None:
            std = np.asarray(std_scaled * scale_val, dtype=np.float64)
        else:
            flattened = samples.reshape(self.n_mc_samples * batch_size, -1)
            unscaled = self.scaler_y.inverse_transform(flattened)
            std = np.asarray(
                unscaled.reshape(self.n_mc_samples, batch_size, -1).std(axis=0),
                dtype=np.float64,
            )

        return format_prediction(mean, single=single_sample), format_prediction(
            std, single=single_sample
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.custom_code:
            state["_network_state"] = {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            }
            state["model"] = None
            state["device"] = torch.device("cpu")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_network_state", None)
        self.__dict__.update(state)
        self.custom_code = state.get("custom_code")
        if weights is not None:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required to load this custom model.")
            self.model = compile_custom_code(
                self.custom_code,
                self.scaler_x.n_features_in_,
                self.scaler_y.n_features_in_,
            )
            self.model.load_state_dict(weights)
            self.model.eval()
        if TORCH_AVAILABLE:
            self.device = next(self.model.parameters()).device

    def to_cpu(self) -> CustomArchitectureModel:
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.cpu()
            self.device = torch.device("cpu")
        return self


class CustomArchitectureStrategy(SurrogateModelStrategy):
    """Strategy for training custom user-defined neural network architectures."""

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
    ) -> tuple[Any, Metrics]:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for custom neural network architectures."
            )

        data = prepare_training_data(X, y, config, X_test, y_test)
        train_X, train_y, eval_X, eval_y = (
            data.train_X,
            data.train_y,
            data.eval_X,
            data.eval_y,
        )
        n_features, n_targets = train_X.shape[1], train_y.shape[1]
        random_state = int(config.get("random_state", 42))
        torch.manual_seed(random_state)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        train_X_s = scaler_x.fit_transform(train_X)
        train_y_s = scaler_y.fit_transform(train_y)
        eval_X_s = scaler_x.transform(eval_X)
        eval_y_s = scaler_y.transform(eval_y)

        device = select_torch_device()
        train_X_tensor = torch.as_tensor(train_X_s, dtype=torch.float32)
        train_y_tensor = torch.as_tensor(train_y_s, dtype=torch.float32)
        eval_X_tensor = torch.as_tensor(eval_X_s, dtype=torch.float32, device=device)
        eval_y_tensor = torch.as_tensor(eval_y_s, dtype=torch.float32, device=device)

        # Build network from visual spec or custom code
        mode = str(config.get("custom_arch_mode", "visual")).lower()
        if mode == "code":
            code_str = str(config.get("custom_code", ""))
            network = compile_custom_code(code_str, n_features, n_targets).to(device)
        elif mode == "visual":
            spec = dict(config.get("custom_spec", {}))
            network = CustomNeuralNet(n_features, n_targets, spec).to(device)
        else:
            raise ValueError(f"Unknown architecture mode {mode!r}.")

        criterion = nn.MSELoss()
        lr = float(config.get("learning_rate", 0.001))
        weight_decay = float(config.get("weight_decay", 1e-4))
        epochs = int(config.get("epochs", 1000))
        batch_size = int(config.get("batch_size", 32))
        patience = int(config.get("patience", 40))
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if min(epochs, batch_size, patience, int(config.get("n_mc_samples", 50))) < 1:
            raise ValueError(
                "epochs, batch_size, patience, and n_mc_samples must be positive."
            )

        optimizer_name = str(config.get("optimizer", "AdamW"))
        if optimizer_name == "SGD":
            optimizer = optim.SGD(
                network.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(
                network.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(
                network.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                network.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name!r}.")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(10, epochs), eta_min=1e-6
        )

        dataset = TensorDataset(train_X_tensor, train_y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(train_X_tensor)),
            shuffle=True,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(random_state),
        )

        best_loss = float("inf")
        best_state = None
        no_improvement = 0
        last_update = 0.0

        for epoch in range(epochs):
            if stop_flag and stop_flag():
                logger.info(
                    "Custom architecture training stopped by user at epoch %d", epoch
                )
                break

            network.train()
            total_loss = 0.0
            for bx, by in loader:
                bx = bx.to(device, non_blocking=device.type == "cuda")
                by = by.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                # BatchNorm cannot estimate variance from a singleton tabular
                # batch; retain its running statistics without dropping a row.
                for module in network.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.train(len(bx) > 1)
                out = network(bx)
                loss = criterion(out, by)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())

            scheduler.step()
            train_loss_epoch = total_loss / len(loader)

            network.eval()
            with torch.no_grad():
                val_loss_epoch = float(
                    criterion(network(eval_X_tensor), eval_y_tensor).item()
                )
            if not np.isfinite(val_loss_epoch):
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")

            if val_loss_epoch < best_loss - 1e-12:
                best_loss = val_loss_epoch
                best_state = copy.deepcopy(network.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1

            now = time.monotonic()
            if loss_callback and (epoch % 5 == 0 or now - last_update > 0.4):
                loss_callback(
                    {
                        "epoch": epoch,
                        "train": train_loss_epoch,
                        "val": val_loss_epoch,
                    }
                )
                last_update = now

            if callback and (epoch % 25 == 0 or epoch == epochs - 1):
                progress = 10 + int(85 * (epoch + 1) / epochs)
                callback(
                    progress,
                    f"Epoch {epoch + 1}/{epochs}: train={train_loss_epoch:.4g}, val={val_loss_epoch:.4g}",
                )

            if no_improvement >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        if best_state is not None:
            network.load_state_dict(best_state)

        network.eval()
        with torch.no_grad():
            preds_scaled = network(eval_X_tensor).detach().cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)

        eval_y_orig = eval_y.ravel() if n_targets == 1 else eval_y
        preds_orig = preds.ravel() if n_targets == 1 else preds
        final_metrics = evaluate_predictions(eval_y_orig, preds_orig)
        final_metrics["evaluation_source"] = data.evaluation_source

        model_wrapper = CustomArchitectureModel(
            network,
            scaler_x,
            scaler_y,
            n_mc_samples=int(config.get("n_mc_samples", 50)),
            spec=dict(config.get("custom_spec", {})),
            custom_code=str(config.get("custom_code", "")) if mode == "code" else None,
        )
        _, prediction_std = model_wrapper.predict(eval_X, return_std=True)
        final_metrics["y_std"] = np.asarray(prediction_std)[:100].tolist()

        return model_wrapper, final_metrics
