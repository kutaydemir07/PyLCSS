# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility imports for the pre-2.3 monolithic training module.

New code should import :class:`SurrogateTrainer` from
``pylcss.surrogate_modeling`` or ``pylcss.surrogate_modeling.training``.
"""

from .data_generation import DILL_AVAILABLE, JOBLIB_AVAILABLE, QMC_AVAILABLE
from .models import (
    TORCH_AVAILABLE,
    ConfigurableNet,
    PyTorchWrapper,
    UncertaintyWrapper,
)
from .strategies import (
    GaussianProcessStrategy,
    GradientBoostingStrategy,
    MLPStrategy,
    PyTorchStrategy,
    RandomForestStrategy,
    SurrogateModelStrategy,
)
from .training import (
    SKLEARN_AVAILABLE,
    SurrogateTrainer,
    evaluate_model_predictions,
)

if TORCH_AVAILABLE:
    from .geometry_training import GeomDeepONetStrategy, GINOStrategy
else:
    GINOStrategy = None  # type: ignore[assignment,misc]
    GeomDeepONetStrategy = None  # type: ignore[assignment,misc]

__all__ = [
    "DILL_AVAILABLE",
    "JOBLIB_AVAILABLE",
    "QMC_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "TORCH_AVAILABLE",
    "ConfigurableNet",
    "GINOStrategy",
    "GaussianProcessStrategy",
    "GeomDeepONetStrategy",
    "GradientBoostingStrategy",
    "MLPStrategy",
    "PyTorchStrategy",
    "PyTorchWrapper",
    "RandomForestStrategy",
    "SurrogateModelStrategy",
    "SurrogateTrainer",
    "UncertaintyWrapper",
    "evaluate_model_predictions",
]
