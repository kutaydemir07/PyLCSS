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
from .data_importer import DataImporter
from .auto_surrogate import AutoSurrogateModel, AutoSurrogateStrategy
from .custom_architecture import (
    CustomArchitectureModel,
    CustomArchitectureStrategy,
    CustomNeuralNet,
    ResidualBlock,
    FourierFeatureEmbedding,
)
from .mdao_algorithms import (
    MdaoEnsembleOfSurrogates,
    MdaoKPLSRegressor,
    MdaoKrigingRegressor,
    MdaoRBFRegressor,
    MdaoSVRRegressor,
    MdaoSparsePCERegressor,
    compute_mdao_metrics,
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
    "DataImporter",
    "AutoSurrogateModel",
    "AutoSurrogateStrategy",
    "CustomArchitectureModel",
    "CustomArchitectureStrategy",
    "CustomNeuralNet",
    "ResidualBlock",
    "FourierFeatureEmbedding",
    "MdaoEnsembleOfSurrogates",
    "MdaoKPLSRegressor",
    "MdaoKrigingRegressor",
    "MdaoRBFRegressor",
    "MdaoSVRRegressor",
    "MdaoSparsePCERegressor",
    "compute_mdao_metrics",
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
