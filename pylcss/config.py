# Copyright (c) 2026 Kutay Demir.
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

# Get the directory where config.py resides (the pylcss folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define a local folder for generated models
TEMP_MODELS_DIR = os.path.join(BASE_DIR, "generated_models")

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
# SIMULATION SETTINGS
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration for FEA and simulation operations."""
    
    # Verbosity settings
    SUPPRESS_EXTERNAL_LIBRARY_OUTPUT: bool = True
    SHOW_TOPOLOGY_OPTIMIZATION_PROGRESS: bool = True
    
    # Default mesh settings
    DEFAULT_MESH_SIZE: float = 2.0
    MIN_MESH_SIZE: float = 0.1
    MAX_MESH_SIZE: float = 10.0
    
    # Topology optimization settings
    DEFAULT_VOLUME_FRACTION: float = 0.4
    DEFAULT_FILTER_RADIUS: float = 1.5
    DEFAULT_ITERATIONS: int = 15
    DENSITY_CUTOFF: float = 0.3

simulation_config = SimulationConfig()


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
simulation_config = SimulationConfig()
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
    handlers = []
    
    # 1. Stream Handler with encoding safety
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)
    
    # 2. File Handler with UTF-8 support
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not initialize file logging: {e}")
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('skfem').setLevel(logging.WARNING)
    logging.getLogger('skfem.assembly').setLevel(logging.WARNING)
    logging.getLogger('skfem.assembly.basis').setLevel(logging.WARNING)
    logging.getLogger('skfem.assembly.form').setLevel(logging.WARNING)
    logging.getLogger('skfem.utils').setLevel(logging.WARNING)
    logging.getLogger('netgen').setLevel(logging.WARNING)
    logging.getLogger('ngsolve').setLevel(logging.WARNING)
    
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
    },
    'NSGA-II': {
        'name': 'Non-dominated Sorting Genetic Algorithm II',
        'description': 'Multi-objective evolutionary optimizer producing Pareto-optimal fronts',
        'best_for': 'Multi-objective problems with 2-5 conflicting objectives',
        'supports_constraints': True,
        'speed': 'Slow',
        'robustness': 'Very High',
        'when_to_use': 'Use when you have multiple competing objectives and need the full trade-off front.'
    },
    'Multi-Start': {
        'name': 'Multi-Start Global Search',
        'description': 'Runs a local optimizer from multiple random starting points (LHS)',
        'best_for': 'Multi-modal problems where global optimum is desired',
        'supports_constraints': True,
        'speed': 'Medium',
        'robustness': 'High',
        'when_to_use': 'Use when you suspect multiple local minima but want to use a fast local solver.'
    }
}
