# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Backend engine for surrogate model training and data generation.
Supports multiple architecture types and manages the spy-model generation process.
Includes robust error handling for optional dependencies (PyTorch).
"""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
import numpy as np
import time
import logging
import traceback
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# --- Scikit-learn Imports (Core Requirement) ---
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
    
    
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Scipy Imports for Optimization ---
from scipy.optimize import minimize

# --- Optional Imports for Enhanced Sampling ---
try:
    from scipy.stats import qmc
    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# --- Enhanced Pickling Support ---
try:
    import dill
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False

# --- PyTorch Imports (Optional - with Crash Protection) ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning("PyTorch could not be loaded; advanced neural networks will be disabled.")
    logger.debug("PyTorch import error details", exc_info=True)
    TORCH_AVAILABLE = False

# Import models from local module
from .models import ConfigurableNet, PyTorchWrapper

def evaluate_model_predictions(y_true, y_pred, max_samples=100):
    """
    Standardized evaluation function for model predictions.
    Enforces strict 2D array contract for inputs.
    
    Args:
        y_true: True target values (samples x targets)
        y_pred: Predicted values (samples x targets)
        max_samples: Maximum number of samples to include in metrics
    
    Returns:
        dict: Standardized metrics dictionary
    """
    # Ensure consistent shapes - Enforce 2D Contract
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Force 2D shape (N, 1) if 1D array is passed
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    
    # Safe R2 calculation
    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = 1.0 if mse < 1e-9 else 0.0
    
    return {
        'RMSE': np.sqrt(mse),
        'R2': r2,
        'y_test': y_true[:max_samples].tolist(),
        'y_pred': y_pred[:max_samples].tolist()
    }


class SurrogateModelStrategy(ABC):
    """
    Abstract base class for surrogate model training strategies.
    """
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any], 
              X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
              callback: Optional[Callable[[int, str], None]] = None, 
              stop_flag: Optional[Callable[[], bool]] = None, 
              loss_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[Any, Dict[str, Any]]:
        pass

class MLPStrategy(SurrogateModelStrategy):
    def train(self, X, y, config, X_test=None, y_test=None, callback=None, stop_flag=None, loss_callback=None):
        layers = config.get('hidden_layers', (100, 50))
        if isinstance(layers, str):
            try:
                layers = eval(layers)
            except:
                layers = (100, 50)
        
        is_debug = config.get('debug_mode', False)
        use_early_stopping = True
        if is_debug or len(X) < 20:
            use_early_stopping = False
        
        regressor = MLPRegressor(
            hidden_layer_sizes=layers,
            activation=config.get('activation', 'relu'),
            solver='adam',
            max_iter=config.get('max_iter', 2000), 
            early_stopping=use_early_stopping,
            validation_fraction=0.1 if use_early_stopping else 0.0,
            n_iter_no_change=20,
            random_state=config.get('random_state', 42)
        )
        
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        base_pipeline = Pipeline([
            ('scaler', input_scaler),
            ('regressor', regressor)
        ])
        
        model = TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=target_scaler
        )
        
        # Data splitting logic
        if config.get('debug_mode', False):
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X, y
        elif X_test is not None and y_test is not None:
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X_test, y_test
        else:
            X_train, X_test_eval, y_train, y_test_eval = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model.fit(X_train, y_train)
        
        # Extract loss curve if possible
        if loss_callback:
            try:
                inner_pipeline = model.regressor_
                inner_mlp = inner_pipeline.named_steps['regressor']
                if hasattr(inner_mlp, 'loss_curve_'):
                    total_points = len(inner_mlp.loss_curve_)
                    for i, loss_val in enumerate(inner_mlp.loss_curve_):
                        if total_points < 100 or i % (total_points // 100) == 0:
                            loss_callback({'epoch': i, 'train': float(loss_val), 'val': float(loss_val)})
            except Exception as e:
                logger.warning(f"Could not extract MLP loss curve: {e}")

        metrics = self._evaluate(model, X_test_eval, y_test_eval, config)
        return model, metrics

    def _evaluate(self, model, X_test, y_test, config):
        if len(X_test) == 0:
            return {'RMSE': None, 'R2': None, 'y_test': [], 'y_pred': [], 'debug_mode': config.get('debug_mode', False)}
        
        y_pred = model.predict(X_test)
        metrics = evaluate_model_predictions(y_test, y_pred, max_samples=100)
        metrics['debug_mode'] = config.get('debug_mode', False)
        return metrics

class GaussianProcessStrategy(SurrogateModelStrategy):
    def train(self, X, y, config, X_test=None, y_test=None, callback=None, stop_flag=None, loss_callback=None):
        
        # 1. Define a robust optimizer wrapper
        # This prevents the "ABNORMAL_TERMINATION" by explicitly controlling the solver
        def robust_optimizer(obj_func, initial_theta, bounds):
            res = minimize(
                obj_func, 
                initial_theta, 
                method="SLSQP", 
                jac=True,  # <--- Add this line
                bounds=bounds, 
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            return res.x, res.fun

        # 2. Use Matern Kernel (nu=2.5) instead of RBF
        # Matern(nu=2.5) allows for non-smooth physical behavior (common in engineering)
        # Generalized "wide" bounds to handle any engineering scale
        kernel = C(1.0, (1e-5, 1e10)) * \
                 Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e2))
        
        regressor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer=robust_optimizer, # <--- Inject custom optimizer
            normalize_y=True,
            alpha=0.0,
            random_state=config.get('random_state', 42)
        )
        
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        base_pipeline = Pipeline([
            ('scaler', input_scaler),
            ('regressor', regressor)
        ])
        
        model = TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=target_scaler
        )
        
        # ... (keep the rest of your data splitting logic exactly the same) ...
        if config.get('debug_mode', False):
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X, y
        elif X_test is not None and y_test is not None:
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X_test, y_test
        else:
            X_train, X_test_eval, y_train, y_test_eval = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model.fit(X_train, y_train)
        
        # Change label to 'Gaussian Process (Matern)' if you prefer, or keep it generic
        metrics = self._evaluate(model, X_test_eval, y_test_eval, config)
        return UncertaintyWrapper(model, 'Gaussian Process (Kriging)'), metrics

    # ... keep _evaluate method exactly the same ...
    def _evaluate(self, model, X_test, y_test, config):
        if len(X_test) == 0:
            return {'RMSE': None, 'R2': None, 'y_test': [], 'y_pred': [], 'debug_mode': config.get('debug_mode', False)}
        y_pred = model.predict(X_test)
        metrics = evaluate_model_predictions(y_test, y_pred, max_samples=100)
        metrics['debug_mode'] = config.get('debug_mode', False)
        return metrics

class RandomForestStrategy(SurrogateModelStrategy):
    def train(self, X, y, config, X_test=None, y_test=None, callback=None, stop_flag=None, loss_callback=None):
        if callback: callback(85, "Training Random Forest...")
        regressor = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )
        
        model = regressor # No scaling needed for RF
        
        if config.get('debug_mode', False):
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X, y
        elif X_test is not None and y_test is not None:
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X_test, y_test
        else:
            X_train, X_test_eval, y_train, y_test_eval = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model.fit(X_train, y_train)
        
        metrics = self._evaluate(model, X_test_eval, y_test_eval, config)
        return UncertaintyWrapper(model, 'Random Forest'), metrics

    def _evaluate(self, model, X_test, y_test, config):
        if len(X_test) == 0:
            return {'RMSE': None, 'R2': None, 'y_test': [], 'y_pred': [], 'debug_mode': config.get('debug_mode', False)}
        y_pred = model.predict(X_test)
        metrics = evaluate_model_predictions(y_test, y_pred, max_samples=100)
        metrics['debug_mode'] = config.get('debug_mode', False)
        return metrics

class GradientBoostingStrategy(SurrogateModelStrategy):
    def train(self, X, y, config, X_test=None, y_test=None, callback=None, stop_flag=None, loss_callback=None):
        if callback: callback(85, "Training Gradient Boosting...")
        from sklearn.multioutput import MultiOutputRegressor
        gbr = GradientBoostingRegressor(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3),
            random_state=config.get('random_state', 42)
        )
        model = MultiOutputRegressor(gbr)
        
        if config.get('debug_mode', False):
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X, y
        elif X_test is not None and y_test is not None:
            X_train, y_train = X, y
            X_test_eval, y_test_eval = X_test, y_test
        else:
            X_train, X_test_eval, y_train, y_test_eval = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model.fit(X_train, y_train)
        
        metrics = self._evaluate(model, X_test_eval, y_test_eval, config)
        return model, metrics

    def _evaluate(self, model, X_test, y_test, config):
        if len(X_test) == 0:
            return {'RMSE': None, 'R2': None, 'y_test': [], 'y_pred': [], 'debug_mode': config.get('debug_mode', False)}
        y_pred = model.predict(X_test)
        metrics = evaluate_model_predictions(y_test, y_pred, max_samples=100)
        metrics['debug_mode'] = config.get('debug_mode', False)
        return metrics

class PyTorchStrategy(SurrogateModelStrategy):
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance

    def train(self, X, y, config, X_test=None, y_test=None, callback=None, stop_flag=None, loss_callback=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install it or select 'MLP Regressor'.")
        return self.trainer._train_pytorch_model(X, y, config, callback, stop_flag, loss_callback, X_test, y_test)


class UncertaintyWrapper:
    """
    Wrapper for sklearn models to add uncertainty quantification support.
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
    
    def predict(self, X, return_std=False):
        if not return_std:
            return self.model.predict(X)
        
        if self.model_type == 'Gaussian Process (Kriging)':
            # Gaussian Process has built-in uncertainty
            y_pred, y_std = self.model.predict(X, return_std=True)
            return y_pred, y_std
        elif self.model_type == 'Random Forest':
            # Tree variance: variance across individual tree predictions
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            y_pred = np.mean(tree_predictions, axis=0)
            y_std = np.std(tree_predictions, axis=0)
            return y_pred, y_std
        else:
            # Fallback: no uncertainty available
            y_pred = self.model.predict(X)
            y_std = np.zeros_like(y_pred)
            return y_pred, y_std


class SurrogateTrainer:
    """
    Manages data generation and model training for surrogate models.
    """

    def __init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for surrogate modeling.")
        
        # Initialize strategies
        self.strategies = {
            'MLP Regressor': MLPStrategy(),
            'Gaussian Process': GaussianProcessStrategy(),
            'Gaussian Process (Kriging)': GaussianProcessStrategy(), # Keep old key for compatibility
            'Random Forest': RandomForestStrategy(),
            'Gradient Boosting': GradientBoostingStrategy(),
            'Deep Neural Network (PyTorch)': PyTorchStrategy(self)
        }

    def generate_data(self, spy_code: str, spy_inputs: List[str], spy_outputs: List[str], input_bounds: List[Tuple[float, float]], num_samples: int = 1000, test_samples: int = 200, random_state: int = 42, callback: Optional[Callable[[int, str], None]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Generate training and test data using a pre-compiled spy model code.
        Uses Latin Hypercube Sampling for better space-filling coverage.
        """
        if callback: callback(0, "Initializing data generator...")
        
        # Compile Spy Model
        # Use a temporary file to ensure picklability for parallel processing
        import tempfile
        import os
        import importlib.util
        import uuid
        
        temp_dir = os.path.join(tempfile.gettempdir(), "pylcss_spy_models")
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = f"spy_{uuid.uuid4().hex}.py"
        filepath = os.path.join(temp_dir, filename)
        
        # Write the spy code to temporary file
        with open(filepath, 'w') as f:
            f.write(spy_code)
            
        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location("spy_module", filepath)
            spy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(spy_module)
            
            if hasattr(spy_module, "spy_model"):
                spy_func = spy_module.spy_model
            else:
                raise AttributeError("spy_model function not found in generated code")
                
        except Exception as e:
            # Log the generated code for debugging before raising error
            logger.error("Generated spy model code compilation failed.")
            logger.debug(f"Code:\n{spy_code}")
            raise RuntimeError(f"Failed to compile spy model: {e}")
        finally:
            # Always clean up the temporary file
            try:
                os.unlink(filepath)
            except:
                pass

        # Generate Data
        total_samples = num_samples + test_samples
        if callback: callback(10, f"Generating {total_samples} samples...")
        
        # Use Latin Hypercube Sampling for better coverage
        if QMC_AVAILABLE:
            sampler = qmc.LatinHypercube(d=len(input_bounds), seed=random_state)
            samples = sampler.random(n=total_samples)
            # Scale to bounds
            bounds_list = input_bounds
            l_bounds = np.array([b[0] for b in bounds_list])
            u_bounds = np.array([b[1] for b in bounds_list])
            samples = qmc.scale(samples, l_bounds, u_bounds)
        else:
            # Fallback to random sampling
            np.random.seed(random_state)
            samples = np.random.uniform(
                [b[0] for b in input_bounds],
                [b[1] for b in input_bounds],
                (total_samples, len(input_bounds))
            )
        
        # Pre-allocate arrays for better performance
        X_data = np.empty((total_samples, len(spy_inputs)), dtype=np.float64)
        y_data = np.empty((total_samples, len(spy_outputs)), dtype=np.float64)
        valid_samples = 0
        
        # Use dill for parallelization if available, otherwise fallback to joblib with n_jobs=1
        if DILL_AVAILABLE and JOBLIB_AVAILABLE:
            # Use dill to serialize the spy_func for parallel processing
            # Since spy_func is now from a module on disk, it should pickle correctly
            def evaluate_sample(sample_idx):
                sample_inputs = samples[sample_idx]
                try:
                    inputs_dict, outputs_dict = spy_func(*sample_inputs)
                    X_sample = [inputs_dict[f'input_{j}'] for j in range(len(spy_inputs))]
                    y_sample = [outputs_dict[f'output_{j}'] for j in range(len(spy_outputs))]
                    return X_sample, y_sample, None
                except Exception as e:
                    return None, None, f"{str(e)}\n{traceback.format_exc()}"
            
            # Use parallel processing with loky backend to bypass GIL
            results = Parallel(n_jobs=-1, backend='loky')(delayed(evaluate_sample)(i) for i in range(total_samples))
            
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            for i, (X_sample, y_sample, error) in enumerate(results):
                if error:
                    consecutive_failures += 1
                    logger.warning(f"Sample {i} failed: {error}")
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Data generation aborted: {max_consecutive_failures} consecutive samples failed. Check your model code for errors.")
                    continue
                else:
                    X_data[valid_samples] = X_sample
                    y_data[valid_samples] = y_sample
                    valid_samples += 1
                    consecutive_failures = 0
                    
                    # Progress update every 5% or every 100 samples
                    if callback and (i % max(1, total_samples // 20) == 0 or i % 100 == 0):
                        progress = 10 + int(70 * (i + 1) / total_samples)
                        callback(progress, f"Generated {valid_samples}/{total_samples} samples...")
                        
        elif JOBLIB_AVAILABLE:
            # Fallback: Use joblib with sequential execution (n_jobs=1)
            def evaluate_sample(sample_idx):
                sample_inputs = samples[sample_idx]
                try:
                    inputs_dict, outputs_dict = spy_func(*sample_inputs)
                    X_sample = [inputs_dict[f'input_{j}'] for j in range(len(spy_inputs))]
                    y_sample = [outputs_dict[f'output_{j}'] for j in range(len(spy_outputs))]
                    return X_sample, y_sample, None
                except Exception as e:
                    return None, None, f"{str(e)}\n{traceback.format_exc()}"
            
            results = Parallel(n_jobs=1)(delayed(evaluate_sample)(i) for i in range(total_samples))
            
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            for i, (X_sample, y_sample, error) in enumerate(results):
                if error:
                    consecutive_failures += 1
                    logger.warning(f"Sample {i} failed: {error}")
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Data generation aborted: {max_consecutive_failures} consecutive samples failed. Check your model code for errors.")
                    continue
                else:
                    X_data[valid_samples] = X_sample
                    y_data[valid_samples] = y_sample
                    valid_samples += 1
                    consecutive_failures = 0
                    
                    # Progress update every 5% or every 100 samples
                    if callback and (i % max(1, total_samples // 20) == 0 or i % 100 == 0):
                        progress = 10 + int(70 * (i + 1) / total_samples)
                        callback(progress, f"Generated {valid_samples}/{total_samples} samples...")
        else:
            # Sequential fallback without joblib
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            for i in range(total_samples):
                sample_inputs = samples[i]
                try:
                    inputs_dict, outputs_dict = spy_func(*sample_inputs)
                    X_sample = [inputs_dict[f'input_{j}'] for j in range(len(spy_inputs))]
                    y_sample = [outputs_dict[f'output_{j}'] for j in range(len(spy_outputs))]
                    X_data[valid_samples] = X_sample
                    y_data[valid_samples] = y_sample
                    valid_samples += 1
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(f"Sample {i} failed: {e}")
                    logger.debug(traceback.format_exc())
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(f"Data generation aborted: {max_consecutive_failures} consecutive samples failed.")
                    continue
                
                # Progress update every 5% or every 100 samples
                if callback and (i % max(1, total_samples // 20) == 0 or i % 100 == 0):
                    progress = 10 + int(70 * (i + 1) / total_samples)
                    callback(progress, f"Generated {valid_samples}/{total_samples} samples...")

        if valid_samples == 0:
            raise RuntimeError("Data generation failed. No valid samples were produced.")

        # Trim arrays to actual valid samples
        X_data = X_data[:valid_samples]
        y_data = y_data[:valid_samples]
        
        # Explicitly shuffle data before splitting to ensure no bias from LHS ordering
        from sklearn.utils import shuffle
        X_data, y_data = shuffle(X_data, y_data, random_state=random_state)
        
        # Split into train and test sets with shuffling
        n_total = len(X_data)
        
        # Adjust test_samples if not enough data
        available_test = max(0, n_total - num_samples)
        actual_test_samples = min(test_samples, available_test)
        
        if actual_test_samples <= 0:
            train_X = X_data
            train_y = y_data
            test_X = np.empty((0, X_data.shape[1]), dtype=X_data.dtype)
            test_y = np.empty((0, y_data.shape[1] if y_data.ndim > 1 else 1), dtype=y_data.dtype)
        else:
            # Use train_test_split for proper shuffling
            train_X, test_X, train_y, test_y = train_test_split(
                X_data, y_data, 
                test_size=actual_test_samples, 
                random_state=random_state,
                shuffle=True
            )
        
        if callback: callback(80, "Data generation completed.")
        
        return train_X, train_y, test_X, test_y, [i['name'] for i in spy_inputs], [o['name'] for o in spy_outputs]

    def train_model(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any], X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, callback: Optional[Callable[[int, str], None]] = None, stop_flag: Optional[Callable[[], bool]] = None, loss_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the model based on configuration using the Strategy Pattern.
        """
        if callback: callback(80, f"Training {config.get('model_type', 'Unknown')}...")
        
        model_type = config.get('model_type', 'MLP Regressor')
        strategy = self.strategies.get(model_type)
        
        if not strategy:
            # Fallback to MLP if unknown
            logger.warning(f"Unknown model type '{model_type}', falling back to MLP Regressor.")
            strategy = self.strategies['MLP Regressor']
            
        return strategy.train(X, y, config, X_test, y_test, callback, stop_flag, loss_callback)
        
    def _train_pytorch_model(self, X: np.ndarray, y: np.ndarray, config: Dict[str, Any], callback: Optional[Callable[[int, str], None]], stop_flag: Optional[Callable[[], bool]] = None, loss_callback: Optional[Callable[[Dict[str, Any]], None]] = None, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, Any]]:
        # Move import here if strictly necessary, or top of file is better
        import copy
        
        if config.get('debug_mode', False):
            X_train = X
            y_train = y
            X_test = X
            y_test = y
        elif X_test is not None and y_test is not None:
            X_train = X
            y_train = y
            X_test = X_test
            y_test = y_test
        else:
            validation_split = config.get('validation_split', 0.2)
            if validation_split <= 0.0:
                # No validation split requested, use all data for training
                X_train = X
                y_train = y
                X_test = X[:1]  # Keep one sample to avoid scaler issues
                y_test = y[:1]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        y_train_s = scaler_y.fit_transform(y_train)
        X_test_s = scaler_x.transform(X_test)
        y_test_s = scaler_y.transform(y_test)
        
        t_X_train = torch.FloatTensor(X_train_s)
        t_y_train = torch.FloatTensor(y_train_s)
        t_X_test = torch.FloatTensor(X_test_s)
        t_y_test = torch.FloatTensor(y_test_s)
        
        # GPU/Accelerator device detection and setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA GPU for PyTorch training")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon MPS for PyTorch training")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for PyTorch training")
        
        # Move tensors to device
        # Check dataset size to decide whether to move to GPU upfront
        # Threshold: 500 MB (approx 500 * 1024 * 1024 bytes)
        dataset_size_bytes = t_X_train.element_size() * t_X_train.nelement() + t_y_train.element_size() * t_y_train.nelement()
        move_to_gpu_upfront = dataset_size_bytes < (500 * 1024 * 1024) and device.type != 'cpu'
        
        if move_to_gpu_upfront:
            t_X_train = t_X_train.to(device)
            t_y_train = t_y_train.to(device)
            logger.info("Dataset %.2f MB fits in VRAM; moving to accelerator upfront", dataset_size_bytes / (1024 * 1024))
        else:
            logger.info("Dataset %.2f MB; keeping on CPU and streaming batches", dataset_size_bytes / (1024 * 1024))
            
        t_X_test = t_X_test.to(device)
        t_y_test = t_y_test.to(device)
        
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        hidden_layers = config.get('hidden_layers', '64, 64')
        if isinstance(hidden_layers, str):
            try:
                hidden_dims = [int(x.strip()) for x in hidden_layers.split(',')]
            except:
                hidden_dims = [64, 64]
        else:
            hidden_dims = [64, 64]
        
        dropout_rate = config.get('dropout', 0.1)
        net = ConfigurableNet(input_dim, hidden_dims, output_dim, dropout_rate)
        net = net.to(device)  # Move model to GPU/accelerator
        criterion = nn.MSELoss()
        
        optimizer_name = config.get('optimizer', 'Adam')
        learning_rate = config.get('learning_rate', 0.01)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        epochs = config.get('epochs', 5000)
        batch_size = config.get('batch_size', 32)
        
        dataset = TensorDataset(t_X_train, t_y_train)
        # Enable pin_memory only if data is on CPU and we are using CUDA
        use_pin_memory = (not move_to_gpu_upfront) and (device.type == 'cuda')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin_memory)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_state = None
        
        net.train()
        last_update = 0.0
        for epoch in range(epochs):
            batch_loss_accum = 0
            for batch_x, batch_y in loader:
                # Move batch to device only if not already there
                if not move_to_gpu_upfront:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                batch_loss_accum += loss.item()
            
            avg_train_loss = batch_loss_accum / len(loader)
            
            net.eval()
            with torch.no_grad():
                val_pred = net(t_X_test)
                val_loss = criterion(val_pred, t_y_test).item()
            net.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            
            current_time = time.time()
            # Throttle loss callback emission to reduce UI overhead - emit every 10 epochs or every 0.5 seconds
            if loss_callback and ((epoch % 10 == 0) or (current_time - last_update > 0.5)):
                loss_callback({'epoch': epoch, 'train': avg_train_loss, 'val': val_loss})
                last_update = current_time
            
            if callback and epoch % 50 == 0:
                prog = 80 + int(15 * (epoch / epochs))
                callback(prog, f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if stop_flag and stop_flag():
                break

        if best_state:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            if len(X_test) == 0:
                y_pred = np.array([]).reshape(0, output_dim)
                metrics = {
                    'RMSE': None,
                    'R2': None,
                    'y_test': [],
                    'y_pred': []
                }
            else:
                pred_s = net(t_X_test).numpy()
                y_pred = scaler_y.inverse_transform(pred_s)
                
                # Use standardized evaluation function
                metrics = evaluate_model_predictions(y_test, y_pred, max_samples=100)
        
        wrapped_model = PyTorchWrapper(net, scaler_x, scaler_y, n_mc_samples=config.get('n_mc_samples', 50))
        
        if len(X_test) > 0:
            _, y_std = wrapped_model.predict(X_test, return_std=True)
            metrics['y_std'] = y_std[:100].tolist() if len(y_std) > 0 else []
        else:
            metrics['y_std'] = []
        
        return wrapped_model, metrics