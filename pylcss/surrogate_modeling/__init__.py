# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Surrogate data generation, regression, validation, and geometry operators."""

from .models import (
    ConfigurableNet,
    PyTorchWrapper,
    RegressionMLP,
    TorchRegressor,
    UncertaintyRegressor,
)
from .training import SurrogateTrainer
from .active_learning import (
    ACTIVE_LEARNING_STRATEGIES,
    ActiveLearningConfig,
    ActiveLearningError,
    ActiveLearningSelector,
)
from .validation import (
    CrossValidator,
    CVResult,
    FeatureImportanceAnalyzer,
    ModelComparator,
)

__all__ = [
    "CVResult",
    "ConfigurableNet",
    "CrossValidator",
    "FeatureImportanceAnalyzer",
    "ModelComparator",
    "PyTorchWrapper",
    "RegressionMLP",
    "SurrogateTrainer",
    "ACTIVE_LEARNING_STRATEGIES",
    "ActiveLearningConfig",
    "ActiveLearningError",
    "ActiveLearningSelector",
    "TorchRegressor",
    "UncertaintyRegressor",
]
