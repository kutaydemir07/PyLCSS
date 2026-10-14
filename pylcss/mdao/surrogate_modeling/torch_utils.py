# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Optional PyTorch support shared by the custom surrogate and its GUI."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    TORCH_AVAILABLE = False


def select_torch_device() -> Any:
    """Choose a supported accelerator, falling back to CPU."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for custom neural networks.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def prediction_mode(model: Any, *, dropout: bool = False):
    """Enable optional dropout without updating normalization statistics."""
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if dropout:
            for module, _ in states:
                if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                    module.train()
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training
