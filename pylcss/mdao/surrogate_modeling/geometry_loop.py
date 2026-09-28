# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Optimization loop shared by variable-mesh neural operators."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .contracts import LossCallback, ProgressCallback, StopFlag

logger = logging.getLogger(__name__)


def fit_geometry_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training: list[Any],
    validation: list[Any],
    *,
    loss_function: Callable[[Any], torch.Tensor],
    epochs: int,
    random_state: int,
    callback: ProgressCallback | None,
    stop_flag: StopFlag | None,
    loss_callback: LossCallback | None,
    patience: int,
) -> list[float]:
    """Fit a variable-mesh model with early stopping and finite-loss checks."""
    if callback:
        callback(65, f"Training {type(model).__name__} for {epochs} epochs...")
    rng = np.random.default_rng(random_state)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[float] = []

    for epoch in range(epochs):
        if stop_flag and stop_flag():
            logger.info("Geometry training cancelled after epoch %d.", epoch)
            break
        model.train()
        total = 0.0
        for index in rng.permutation(len(training)):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(training[int(index)])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Training produced a non-finite loss at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
        training_loss = total / len(training)
        history.append(training_loss)

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for sample in validation:
                validation_total += float(loss_function(sample).detach().item())
        validation_loss = validation_total / len(validation)

        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if loss_callback and (epoch % 5 == 0 or epoch == epochs - 1):
            loss_callback(
                {
                    "epoch": epoch,
                    "train": training_loss,
                    "val": validation_loss,
                }
            )
        if callback and (epoch % 20 == 0 or epoch == epochs - 1):
            callback(
                65 + int(30 * (epoch + 1) / epochs),
                f"Epoch {epoch + 1}/{epochs}: "
                f"train={training_loss:.4g}, val={validation_loss:.4g}",
            )
        if epochs_without_improvement >= patience:
            logger.info("Geometry training stopped early at epoch %d.", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


__all__ = ["fit_geometry_model"]
