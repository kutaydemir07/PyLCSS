# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
PyTorch model definitions for surrogate modeling.
Moved to separate file to ensure pickling compatibility.
"""

from typing import List, Any, Union, Tuple
import numpy as np
import joblib

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class ConfigurableNet(nn.Module):
        """
        A simple configurable Multi-Layer Perceptron (MLP) for regression.
        Defined globally to allow pickling.
        """
        def __init__(self, in_d: int, hidden_dims: List[int], out_d: int, dropout_rate: float = 0.1) -> None:
            super().__init__()
            layers = []
            prev_dim = in_d
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)  # Add dropout for uncertainty quantification
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, out_d))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class PyTorchWrapper:
        """
        Wraps a PyTorch model and scalers to mimic a Scikit-Learn estimator.
        Defined globally to allow pickling.
        """
        def __init__(self, model: nn.Module, scaler_x: Any, scaler_y: Any, n_mc_samples: int = 50) -> None:
            self.model = model
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y
            self.n_mc_samples = n_mc_samples
            # Detect device from the model parameters
            self.device = next(model.parameters()).device
        
        @classmethod
        def load(cls, filepath: str) -> 'PyTorchWrapper':
            """
            Load a PyTorchWrapper from a joblib file and ensure device consistency.
            """
            loaded_wrapper = joblib.load(filepath)
            # Ensure the model is on the correct device (will be detected in __init__)
            # The device detection in __init__ will handle moving to the right device
            return cls(loaded_wrapper.model, loaded_wrapper.scaler_x, loaded_wrapper.scaler_y, loaded_wrapper.n_mc_samples)

        def predict(self, X_in: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            # Track if input was 1D for proper output shape handling
            input_is_1d = X_in.ndim == 1
            
            if input_is_1d:
                X_in = X_in.reshape(1, -1)

            X_s = self.scaler_x.transform(X_in)
            # CRITICAL FIX: Send tensor to the correct device
            t_in = torch.FloatTensor(X_s).to(self.device)
            
            if return_std:
                # Use Monte Carlo Dropout for uncertainty quantification
                self.model.train()  # Enable dropout during inference
                n_samples = self.n_mc_samples  # Configurable number of MC samples
                
                # Vectorized approach: repeat input tensor for batch processing
                batch_input = t_in.repeat(n_samples, 1)  # Shape: (n_samples * batch_size, input_dim)
                
                with torch.no_grad():
                    # Single forward pass with batched inputs
                    batch_output = self.model(batch_input)  # Shape: (n_samples * batch_size, output_dim)
                    # Move back to CPU for numpy conversion
                    batch_output_np = batch_output.cpu().numpy()
                
                # Reshape to separate MC samples and batch dimension
                batch_size = t_in.shape[0]
                predictions = batch_output_np.reshape(n_samples, batch_size, -1)  # (n_samples, batch_size, output_dim)
                
                # Calculate statistics in scaled space for efficiency
                pred_mean_scaled = np.mean(predictions, axis=0)  # (batch_size, output_dim)
                pred_std_scaled = np.std(predictions, axis=0)    # (batch_size, output_dim)
                
                # Inverse transform mean normally
                pred_mean = self.scaler_y.inverse_transform(pred_mean_scaled)
                
                # Scale std by the scaler's scale factor (std is scale-invariant for translation but not scaling)
                if hasattr(self.scaler_y, 'scale_'):
                    # sklearn StandardScaler uses .scale_ (which is the std dev of each feature)
                    pred_std = pred_std_scaled * self.scaler_y.scale_
                else:
                    # Fallback for complex scalers: use the original method
                    pred_reshaped = predictions.reshape(n_samples * batch_size, -1)
                    pred_inverse = self.scaler_y.inverse_transform(pred_reshaped)
                    pred_inverse = pred_inverse.reshape(n_samples, batch_size, -1)
                    pred_std = np.std(pred_inverse, axis=0)
                
                # Squeeze output if input was 1D
                if input_is_1d:
                    pred_mean = pred_mean.squeeze()
                    pred_std = pred_std.squeeze()
                
                return pred_mean, pred_std
            else:
                # Standard prediction without uncertainty
                self.model.eval()
                with torch.no_grad():
                    # Move back to CPU for numpy conversion
                    out_s = self.model(t_in).cpu().numpy()
                pred = self.scaler_y.inverse_transform(out_s)
                
                # Squeeze output if input was 1D to ensure scalar return for single samples
                if input_is_1d:
                    pred = pred.squeeze()
                
                return pred
else:
    # Dummy classes if PyTorch not available
    class ConfigurableNet:
        pass

    class PyTorchWrapper:
        pass