# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np
import scipy.io
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

# Unit management with pint
try:
    import pint
    ureg = pint.UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    ureg = None
    PINT_AVAILABLE = False

class XRayProblem:
    """
    Represents a design problem for the XRay tool.
    Stores design variables, parameters, and quantities of interest.
    Includes robust evaluation logic with fallbacks for non-vectorized models.
    """
    def __init__(self, name: str, sample_size: int = 3000) -> None:
        self.name = name
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")
        self.sample_size = sample_size
        
        # Unit registry for dimensional analysis
        self.ureg = ureg
        
        # Definitions
        self.design_variables = [] # List of dicts: {name, unit, min, max}
        self.parameters = []       # List of dicts: {name, unit, value}
        self.quantities_of_interest = [] # List of dicts: {name, unit, min, max}
        
        # Data
        self.samples = {} # Dictionary to store sampled data
        self.results = {} # Dictionary to store calculated results
        
        # Visualization
        self.diagram = [] # List of tuples (x_name, y_name) for default plots
        
        # System Function (The merged model)
        self.system_model = None
        self.system_code = None
        
        self.requirement_sets = {} # Stores {'VariantName': {qoi_name: {'req_min': val, ...}}}

    def add_design_variable(self, name: str, unit: str, min_val: Union[int, float], max_val: Union[int, float]) -> None:
        """
        Add a design variable to the problem.
        """
        dv_dict = {
            'name': name, 'unit': unit, 'min': min_val, 'max': max_val, 'type': 'continuous', 'granularity': 1.0
        }
        self.design_variables.append(dv_dict)

    def add_parameter(self, name: str, unit: str, value: Union[int, float]) -> None:
        self.parameters.append({
            'name': name, 'unit': unit, 'value': value
        })

    def add_quantity_of_interest(self, name: str, unit: str, min_val: Union[int, float], max_val: Union[int, float], 
                               minimize: bool = False, maximize: bool = False, weight: float = 1.0) -> None:
        self.quantities_of_interest.append({
            'name': name, 'unit': unit, 'min': min_val, 'max': max_val, 'minimize': minimize, 'maximize': maximize, 'weight': weight
        })

    def add_requirement_set(self, name: str, overrides: Dict[str, Dict[str, Union[int, float]]]) -> None:
        self.requirement_sets[name] = overrides

    def set_system_model(self, model: Any) -> None:
        self.system_model = model

    def set_system_code(self, code: str) -> None:
        self.system_code = code

    def generate_samples(self) -> None:
        """Generates random samples for design variables."""
        n = self.sample_size
        self.samples = {}
        
        for dv in self.design_variables:
            # Uniform sampling for continuous variables
            self.samples[dv['name']] = np.random.uniform(dv['min'], dv['max'], n)
            
        # Add parameters (constant for all samples)
        for p in self.parameters:
            self.samples[p['name']] = np.full(n, p['value'])

    def evaluate(self) -> None:
        """Evaluates the system model for all samples."""
        if not self.system_model:
            raise ValueError("System model not set")
            
        n = self.sample_size
        results_list = []
        inputs = self.samples
        
        # Simple loop
        for i in range(n):
            row_input = {k: v[i] for k, v in inputs.items()}
            try:
                res = self.system_model(**row_input)
                results_list.append(res)
            except Exception as e:
                # Log error but continue
                # print(f"Error at sample {i}: {e}") # Optional: Uncomment for debugging
                results_list.append({})
                
        # Aggregate results
        if not results_list:
            return

        # Instead, collect all unique keys found in any successful run.
        all_keys = set()
        for r in results_list:
            all_keys.update(r.keys())

        if not all_keys:
            # If no keys found (e.g. all runs failed), try to use declared QOIs to at least produce NaNs
            all_keys = {q['name'] for q in self.quantities_of_interest}
            
        for k in all_keys:
            self.results[k] = np.array([r.get(k, np.nan) for r in results_list])

    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluates the system for a matrix of inputs.
        Robustly handles both vectorized and non-vectorized (loop-based) models.
        
        Args:
            x_matrix: Numpy array (dim, N) where dim is number of DVs.
        Returns:
            y_matrix: Numpy array (num_qoi, N).
        """
        dim, N = x_matrix.shape
        
        # Map matrix rows to DV names
        dv_names = [dv['name'] for dv in self.design_variables]
        qoi_names = [q['name'] for q in self.quantities_of_interest]
        
        # Prepare inputs
        inputs = {}
        for i, name in enumerate(dv_names):
            if i < dim:
                inputs[name] = x_matrix[i, :]
            
        for p in self.parameters:
            inputs[p['name']] = np.full(N, p['value'])
            
        # --- Attempt 1: Fast Vectorized Execution ---
        try:
            results_dict = self.system_model(**inputs)
            
            # Extract QOIs
            y_matrix = np.zeros((len(qoi_names), N))
            for j, name in enumerate(qoi_names):
                val = results_dict.get(name, np.nan)
                y_matrix[j, :] = val
            return y_matrix

        except Exception as e:
            # --- Attempt 2: Smart Fallback (Loop Execution) ---
            # This handles models with 'if' statements that can't handle numpy arrays
            # print(f"Vectorized evaluation failed ({e}). Falling back to serial execution.")
            
            y_matrix = np.zeros((len(qoi_names), N))
            
            # Pre-calculate indices to avoid lookups in loop
            input_keys = list(inputs.keys())
            input_vals = [inputs[k] for k in input_keys]
            
            for i in range(N):
                # Construct single input dict
                # Check if value is array-like before indexing to handle scalars safely
                row_input = {k: (v[i] if hasattr(v, '__len__') and len(v) == N else v) 
                             for k, v in zip(input_keys, input_vals)}
                
                try:
                    res = self.system_model(**row_input)
                    for j, name in enumerate(qoi_names):
                        y_matrix[j, i] = res.get(name, np.nan)
                except:
                    y_matrix[:, i] = np.nan
                    
            return y_matrix

    def validate_unit_compatibility(self, output_unit: str, input_unit: str) -> bool:
        """Check if two units are dimensionally compatible."""
        if not PINT_AVAILABLE or not self.ureg:
            return True 
            
        try:
            out_qty = self.ureg.Quantity(1, output_unit)
            in_qty = self.ureg.Quantity(1, input_unit)
            return out_qty.dimensionality == in_qty.dimensionality
        except Exception:
            return True

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert a value between compatible units."""
        if not PINT_AVAILABLE or not self.ureg:
            return value
            
        try:
            qty = self.ureg.Quantity(value, from_unit)
            converted = qty.to(to_unit)
            return converted.magnitude
        except Exception as e:
            logger.warning("Unit conversion failed; returning original value", exc_info=True)
            return value

    def get_common_unit(self, unit1: str, unit2: str) -> str:
        """Find a common unit for two compatible units."""
        if not PINT_AVAILABLE or not self.ureg:
            return unit1
            
        try:
            qty1 = self.ureg.Quantity(1, unit1)
            qty2 = self.ureg.Quantity(1, unit2)
            if qty1.dimensionality == qty2.dimensionality:
                return unit1
            else:
                return unit1
        except Exception:
            return unit1

    def export_to_mat(self, filename: str) -> None:
        """Exports the problem definition and results to a .mat file."""
        data = {
            'design_variables': self.design_variables,
            'parameters': self.parameters,
            'quantities_of_interest': self.quantities_of_interest,
            'samples': self.samples,
            'results': self.results
        }
        try:
            scipy.io.savemat(filename, data)
            logger.info("Exported XRay data to %s", filename)
        except Exception as e:
            logger.exception("Failed to export .mat file: %s", filename)