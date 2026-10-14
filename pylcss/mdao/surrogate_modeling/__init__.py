# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Surrogate data, Auto Mode, custom architectures, and validation."""

from .active_learning import (
    ACTIVE_LEARNING_STRATEGIES,
    ActiveLearningConfig,
    ActiveLearningError,
    ActiveLearningSelector,
)
from .auto_surrogate import AutoSurrogateModel, AutoSurrogateStrategy
from .data_importer import DataImporter
from .mdao_algorithms import (
    MdaoEnsembleOfSurrogates,
    MdaoKPLSRegressor,
    MdaoKrigingRegressor,
    MdaoPolynomialRegressor,
    MdaoRBFRegressor,
    MdaoSVRRegressor,
    compute_mdao_metrics,
)
from .torch_utils import TORCH_AVAILABLE
from .training import SurrogateTrainer
from .validation import (
    CVResult,
    CrossValidator,
    FeatureImportanceAnalyzer,
    ModelComparator,
)

__all__ = [
    "ACTIVE_LEARNING_STRATEGIES",
    "ActiveLearningConfig",
    "ActiveLearningError",
    "ActiveLearningSelector",
    "AutoSurrogateModel",
    "AutoSurrogateStrategy",
    "CVResult",
    "CrossValidator",
    "DataImporter",
    "FeatureImportanceAnalyzer",
    "MdaoEnsembleOfSurrogates",
    "MdaoKPLSRegressor",
    "MdaoKrigingRegressor",
    "MdaoPolynomialRegressor",
    "MdaoRBFRegressor",
    "MdaoSVRRegressor",
    "ModelComparator",
    "SurrogateTrainer",
    "TORCH_AVAILABLE",
    "compute_mdao_metrics",
]

if TORCH_AVAILABLE:
    from .custom_architecture import (
        CustomArchitectureModel,
        CustomArchitectureStrategy,
        CustomNeuralNet,
        FourierFeatureEmbedding,
        ResidualBlock,
    )

    __all__ += [
        "CustomArchitectureModel",
        "CustomArchitectureStrategy",
        "CustomNeuralNet",
        "FourierFeatureEmbedding",
        "ResidualBlock",
    ]
