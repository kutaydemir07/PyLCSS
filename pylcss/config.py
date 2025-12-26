# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Centralized configuration management for PyLCSS.

This module contains all configurable constants and parameters used throughout
the application. Modify these values to tune performance and behavior without
changing core logic.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Directory for temporary model files
TEMP_MODELS_DIR = os.path.join(tempfile.gettempdir(), "pylcss_models")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
DEFAULT_LOG_LEVEL = logging.INFO

# Log format string
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Enable file logging (set path to enable, None to disable)
LOG_FILE_PATH: Optional[str] = "pylcss.log"


# ============================================================================
# OPTIMIZATION SETTINGS
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for optimization solvers."""
    
    # Default solver settings
    DEFAULT_TOLERANCE: float = 1e-4  # Relaxed for noisy black-box models (was 1e-6)
    DEFAULT_MAX_ITERATIONS: int = 5000  # Increased for global solvers (was 100)
    DEFAULT_PENALTY_WEIGHT: float = 1e6  # Scaled for typical engineering costs (was 1000.0)
    
    # Constraint violation threshold
    CONSTRAINT_VIOLATION_TOLERANCE: float = 1e-12
    
    # Cache settings
    EVALUATION_CACHE_SIZE: int = 10000
    CACHE_EVICTION_RATIO: float = 0.2  # Remove 20% when full
    
    # Overflow protection
    MAX_PENALTY_VALUE: float = 1e100
    MAX_VIOLATION_BEFORE_CLAMP: float = 1e50
    
    # Convergence detection
    CONVERGENCE_PATIENCE: int = 50  # Iterations without improvement
    CONVERGENCE_RELATIVE_TOLERANCE: float = 1e-4


# ============================================================================
# SOLUTION SPACE SETTINGS
# ============================================================================

@dataclass
class SolutionSpaceConfig:
    """Configuration for solution space exploration."""
    
    # Sampling settings
    INITIAL_SAMPLE_SIZE: int = 1000
    OPTIMIZATION_SAMPLE_SIZE: int = 200
    FINAL_SAMPLE_SIZE: int = 2000
    MONTE_CARLO_BATCH_SIZE: int = 10000
    
    # Phase I expansion
    MAX_ITER_PHASE_1: int = 100
    GROWTH_RATE_INIT: float = 0.2
    TOLERANCE_PHASE_1: float = 1e-3
    PATIENCE_MAX_FAILURES: int = 2
    
    # Phase II refinement
    MAX_ITER_PHASE_2: int = 500
    
    # Expansion control
    MAX_EXPANSION_ATTEMPTS: int = 10
    TRIM_ALPHA: float = 0.8  # Box trimming factor
    
    # Purity thresholds
    MIN_PURITY_ACCEPT: float = 0.5
    HIGH_PURITY_THRESHOLD: float = 0.8


# ============================================================================
# SURROGATE MODELING SETTINGS
# ============================================================================

@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training."""
    
    # Data generation
    DEFAULT_TRAIN_SAMPLES: int = 1000
    DEFAULT_TEST_SAMPLES: int = 200
    MAX_CONSECUTIVE_FAILURES: int = 10
    
    # Neural network settings
    DEFAULT_HIDDEN_LAYERS: tuple = (100, 50)
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_EPOCHS: int = 1000
    EARLY_STOPPING_PATIENCE: int = 50
    
    # GPU settings
    GPU_MEMORY_THRESHOLD: int = 100_000  # Switch to GPU if dataset > 100k samples
    
    # Parallel processing
    USE_PARALLEL_DATA_GENERATION: bool = True
    PARALLEL_BACKEND: str = 'loky'  # 'loky' or 'threading'


# ============================================================================
# UI SETTINGS
# ============================================================================

@dataclass
class UIConfig:
    """Configuration for user interface."""
    
    # Window settings
    DEFAULT_WINDOW_WIDTH: int = 1600
    DEFAULT_WINDOW_HEIGHT: int = 900
    MIN_WINDOW_WIDTH: int = 1024
    MIN_WINDOW_HEIGHT: int = 768
    
    # Plot settings
    PLOT_UPDATE_INTERVAL: float = 0.1  # seconds
    MAX_PLOT_POINTS: int = 10000
    
    # Progress reporting
    PROGRESS_UPDATE_FREQUENCY: int = 20  # Update every N% or 100 samples


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    
    # Complexity thresholds
    MAX_CYCLOMATIC_COMPLEXITY: int = 10
    MAX_NESTED_DEPTH: int = 5
    
    # Security
    WARN_ON_CUSTOM_CODE: bool = True
    
    # Unit checking
    STRICT_UNIT_CHECKING: bool = False  # Set True to block incompatible units


# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    
    # Temporary file management
    TEMP_DIR_NAME: str = "pylcss_models"
    CLEANUP_TEMP_FILES_ON_EXIT: bool = False  # Set True for auto-cleanup
    
    # Numerical precision
    FLOAT_TOLERANCE: float = 1e-9
    ZERO_THRESHOLD: float = 1e-12
    
    # Performance
    ENABLE_NUMBA_JIT: bool = True
    ENABLE_GPU_ACCELERATION: bool = True


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

# Global configuration instances (modify these to change behavior)
optimization_config = OptimizationConfig()
solution_space_config = SolutionSpaceConfig()
surrogate_config = SurrogateConfig()
ui_config = UIConfig()
validation_config = ValidationConfig()
system_config = SystemConfig()


# ============================================================================
# LOGGING SETUP FUNCTION
# ============================================================================

def setup_logging(level: int = DEFAULT_LOG_LEVEL, 
                  log_file: Optional[str] = LOG_FILE_PATH) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"PyLCSS logging initialized at level: {logging.getLevelName(level)}")


# ============================================================================
# SOLVER DESCRIPTIONS (for UI tooltips)
# ============================================================================

SOLVER_DESCRIPTIONS = {
    'SLSQP': {
        'name': 'Sequential Least Squares Programming',
        'description': 'Fast gradient-based optimizer for smooth, differentiable problems',
        'best_for': 'Continuous variables with smooth objective/constraints',
        'supports_constraints': True,
        'speed': 'Fast',
        'robustness': 'Medium',
        'when_to_use': 'Use when your model is smooth and gradients exist. Best for quick iterations.'
    },
    'COBYLA': {
        'name': 'Constrained Optimization BY Linear Approximation',
        'description': 'Derivative-free optimizer using linear approximations',
        'best_for': 'Non-smooth or noisy objectives with constraints',
        'supports_constraints': True,
        'speed': 'Medium',
        'robustness': 'High',
        'when_to_use': 'Use when derivatives are unavailable or unreliable. Handles noise well.'
    },
    'trust-constr': {
        'name': 'Trust Region Constrained',
        'description': 'Modern interior-point method with robust constraint handling',
        'best_for': 'Complex constrained problems requiring high accuracy',
        'supports_constraints': True,
        'speed': 'Slow',
        'robustness': 'Very High',
        'when_to_use': 'Use when SLSQP fails or when you need strict constraint satisfaction.'
    },
    'Nevergrad': {
        'name': 'Nevergrad (NGOpt)',
        'description': 'Gradient-free meta-optimizer combining multiple strategies',
        'best_for': 'Black-box, noisy, or discrete problems',
        'supports_constraints': True,
        'speed': 'Slow',
        'robustness': 'Very High',
        'when_to_use': 'Use for difficult problems where other methods fail. Best for global search.'
    },
    'Differential Evolution': {
        'name': 'Differential Evolution',
        'description': 'Population-based evolutionary global optimizer',
        'best_for': 'Multi-modal landscapes requiring global search',
        'supports_constraints': True,
        'speed': 'Very Slow',
        'robustness': 'Very High',
        'when_to_use': 'Use when solution space has many local minima. Guaranteed exploration.'
    }
}
