# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Common optimization setup utilities for PyLCSS.

This module provides shared logic for setting up optimization problems,
including variable scaling, constraint handling, and blackbox functions
for both GUI and headless optimization.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class OptimizationSetup:
    """
    Common setup class for optimization problems.

    Handles variable scaling, constraint setup, and provides methods
    to generate SciPy constraints and PyNomad blackbox functions.
    """

    def __init__(self, problem: Any, objectives_list: List[Dict[str, Any]], 
                 constraints_list: List[Dict[str, Any]], scaling: bool = True, manual_scales: Optional[Dict[str, float]] = None) -> None:
        self.problem = problem
        self.objectives_list = objectives_list
        self.constraints_list = constraints_list
        self.scaling = scaling
        self.manual_scales = manual_scales or {}

        # Extract design variable information
        self.dv_names = [dv['name'] for dv in problem.design_variables]
        self.bounds = [(float(dv['min']), float(dv['max'])) for dv in problem.design_variables]

        # Setup scaling
        self.dv_scaler = None
        self.con_scaler = None
        if scaling:
            self._setup_scaling()

        # Cache for system model evaluations to improve constraint handling efficiency
        self._evaluation_cache = {}
        self._max_cache_size = 10000  # Limit cache size to prevent memory leaks

    def _evaluate_system_model(self, x_norm: np.ndarray, system_model: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate system model with caching to avoid redundant computations."""
        # FIX: Implement cache eviction to prevent unbounded memory growth
        if len(self._evaluation_cache) >= self._max_cache_size:
            # Simple FIFO eviction: remove oldest 20% of cache
            keys_to_remove = list(self._evaluation_cache.keys())[:self._max_cache_size // 5]
            for key in keys_to_remove:
                del self._evaluation_cache[key]
        
        # Create cache key from normalized values AND current parameter values
        # Include parameters to ensure cache invalidation when parameters change
        # Use high precision rounding (15 decimals) to avoid interfering with gradient calculations
        # while still handling floating-point precision issues. This preserves differences
        # much smaller than typical finite difference steps (~1e-8) used by gradient optimizers.
        param_values = tuple(p['value'] for p in self.problem.parameters)
        cache_key = (tuple(np.round(x_norm, 15)), param_values)
        
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        
        # Convert to physical space if scaling
        if self.scaling:
            x_phys = self._to_physical(x_norm)
            inputs = {n: x_phys[i] for i, n in enumerate(self.dv_names)}
        else:
            inputs = {n: x_norm[i] for i, n in enumerate(self.dv_names)}

        for p in self.problem.parameters:
            inputs[p['name']] = p['value']

        result = system_model(**inputs)
        self._evaluation_cache[cache_key] = result
        
        # Prevent memory leak by limiting cache size
        if len(self._evaluation_cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO by clearing half)
            keys_to_remove = list(self._evaluation_cache.keys())[:self._max_cache_size // 2]
            for key in keys_to_remove:
                del self._evaluation_cache[key]
        return result

    def evaluate_system_model_physical(self, x_phys: np.ndarray, system_model: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate system model in physical space with caching to avoid redundant computations."""
        # Create cache key from physical values AND current parameter values
        # Include parameters to ensure cache invalidation when parameters change
        # Use high precision rounding (15 decimals) to avoid interfering with gradient calculations
        param_values = tuple(p['value'] for p in self.problem.parameters)
        cache_key = (tuple(np.round(x_phys, 15)), param_values)
        
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        
        # Create inputs dict with physical values
        inputs = {n: x_phys[i] for i, n in enumerate(self.dv_names)}
        for p in self.problem.parameters:
            inputs[p['name']] = p['value']

        result = system_model(**inputs)
        self._evaluation_cache[cache_key] = result
        
        # Prevent memory leak by limiting cache size
        if len(self._evaluation_cache) > self._max_cache_size:
            # Remove oldest entries (simple FIFO by clearing half)
            keys_to_remove = list(self._evaluation_cache.keys())[:self._max_cache_size // 2]
            for key in keys_to_remove:
                del self._evaluation_cache[key]
        return result

    def _setup_scaling(self) -> None:
        """Setup variable and constraint scaling."""
        # Design variable scaling
        dv_mins = []
        dv_maxs = []
        for dv in self.problem.design_variables:
            mn = float(dv['min']) if dv['min'] != float('-inf') else -1e12
            mx = float(dv['max']) if dv['max'] != float('inf') else 1e12
            dv_mins.append(mn)
            dv_maxs.append(mx)

        dv_mins = np.array(dv_mins)
        dv_maxs = np.array(dv_maxs)
        dv_ranges = dv_maxs - dv_mins
        dv_ranges[dv_ranges == 0] = 1.0  # Avoid division by zero

        # Override with manual scales if provided
        for i, dv in enumerate(self.problem.design_variables):
            if dv['name'] in self.manual_scales:
                dv_ranges[i] = self.manual_scales[dv['name']]

        self.dv_scaler = {
            'mins': dv_mins,
            'ranges': dv_ranges
        }

        # Constraint scaling
        con_mins = []
        con_maxs = []
        for constr in self.constraints_list:
            c_min, c_max = self._get_constr_bounds(constr)
            # For scaling purposes, use reasonable defaults for infinite bounds
            # instead of the massive ranges that would result from -inf/inf
            if c_min == float('-inf'):
                c_min = -1e12  # Reasonable default for scaling
            if c_max == float('inf'):
                c_max = 1e12   # Reasonable default for scaling
            con_mins.append(c_min)
            con_maxs.append(c_max)

        con_mins = np.array(con_mins)
        con_maxs = np.array(con_maxs)
        con_ranges = con_maxs - con_mins
        con_ranges[con_ranges == 0] = 1.0

        self.con_scaler = {
            'mins': con_mins,
            'ranges': con_ranges
        }

    def _get_constr_bounds(self, constr: Dict[str, Any]) -> tuple[float, float]:
        """Helper to safely get min/max from constraint dict."""
        c_min = constr.get('min', constr.get('req_min', float('-inf')))
        c_max = constr.get('max', constr.get('req_max', float('inf')))
        
        # Convert to float, but keep inf values
        c_min = float(c_min) if c_min != float('-inf') else float('-inf')
        c_max = float(c_max) if c_max != float('inf') else float('inf')
        
        return c_min, c_max

    def _to_physical(self, x_norm: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] values back to physical units."""
        lowers = np.array([b[0] for b in self.bounds])
        uppers = np.array([b[1] for b in self.bounds])
        
        # Handle infinite bounds
        finite_mask = np.isfinite(lowers) & np.isfinite(uppers)
        x_phys = np.where(finite_mask, lowers + x_norm * (uppers - lowers), x_norm)
        return x_phys

    def _to_normalized(self, x_phys: np.ndarray) -> np.ndarray:
        """Convert physical values to [0,1] normalized range."""
        x_norm = []
        for i, val in enumerate(x_phys):
            lower, upper = self.bounds[i]
            if np.isinf(lower) or np.isinf(upper):
                x_norm.append(val)
            else:
                rng = upper - lower
                if rng == 0:
                    x_norm.append(0.0)
                else:
                    x_norm.append((val - lower) / rng)
        return np.array(x_norm)

    def get_scipy_constraints(self, system_model: Callable[..., Dict[str, Any]], 
                             is_running_callback: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
        """
        Generate SciPy constraint dictionaries.

        Args:
            system_model: Callable system model function
            is_running_callback: Optional callback to check if optimization should continue

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        cons = []
        for constr in self.constraints_list:
            name = constr['name']
            c_min, c_max = self._get_constr_bounds(constr)

            # Get range from the scaler we calculated earlier
            if self.con_scaler is not None:
                constr_idx = self.constraints_list.index(constr)
                c_range = self.con_scaler['ranges'][constr_idx]
            else:
                c_range = 1.0  # No scaling

            if c_min != float('-inf'):
                def con_lo(x_norm, c_name=name, limit=c_min, scale=c_range):
                    if is_running_callback and not is_running_callback():
                        raise UserStopException("Optimization stopped by user.")

                    try:
                        result = self._evaluate_system_model(x_norm, system_model)
                        val = result.get(c_name, 0)
                        if np.isnan(val) or np.isinf(val):
                            logger.warning("NaN/Inf detected in constraint '%s'; solver may get stuck", c_name)
                            return np.float64(-1e15)
                        # FIX: Scale the violation
                        return np.float64((val - limit) / scale)
                    except:
                        return -1e15
                cons.append({'type': 'ineq', 'fun': con_lo})

            if c_max != float('inf'):
                def con_hi(x_norm, c_name=name, limit=c_max, scale=c_range):
                    if is_running_callback and not is_running_callback():
                        raise UserStopException("Optimization stopped by user.")

                    try:
                        result = self._evaluate_system_model(x_norm, system_model)
                        val = result.get(c_name, 0)
                        if np.isnan(val) or np.isinf(val):
                            logger.warning("NaN/Inf detected in constraint '%s'; solver may get stuck", c_name)
                            return -1e15
                        # FIX: Scale the violation
                        return np.float64((limit - val) / scale)
                    except:
                        return -1e15
                cons.append({'type': 'ineq', 'fun': con_hi})

        return cons

    def get_physical_constraints(self, system_model: Callable[..., Dict[str, Any]], 
                                is_running_callback: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
        """
        Generate constraint dictionaries for solvers that work with physical values.
        Unlike get_scipy_constraints, this expects physical inputs, not normalized [0,1] inputs.

        Args:
            system_model: Callable system model function
            is_running_callback: Optional callback to check if optimization should continue

        Returns:
            List of constraint dictionaries for physical-space solvers
        """
        cons = []
        for constr in self.constraints_list:
            name = constr['name']
            c_min, c_max = self._get_constr_bounds(constr)

            # For physical constraints, we don't scale the violations
            # since we're working in the original physical space
            if c_min != float('-inf'):
                def con_lo(x_phys, c_name=name, limit=c_min):
                    if is_running_callback and not is_running_callback():
                        raise UserStopException("Optimization stopped by user.")

                    try:
                        # Create inputs dict with physical values
                        inputs = {n: x_phys[i] for i, n in enumerate(self.dv_names)}
                        for p in self.problem.parameters:
                            inputs[p['name']] = p['value']

                        result = system_model(**inputs)
                        val = result.get(c_name, 0)
                        if np.isnan(val) or np.isinf(val):
                            logger.warning("NaN/Inf detected in constraint '%s'; solver may get stuck", c_name)
                            return np.float64(-1e15)
                        # Return violation (positive = violation for ineq constraints)
                        return np.float64(val - limit)
                    except:
                        return -1e15
                cons.append({'type': 'ineq', 'fun': con_lo})

            if c_max != float('inf'):
                def con_hi(x_phys, c_name=name, limit=c_max):
                    if is_running_callback and not is_running_callback():
                        raise UserStopException("Optimization stopped by user.")

                    try:
                        # Create inputs dict with physical values
                        inputs = {n: x_phys[i] for i, n in enumerate(self.dv_names)}
                        for p in self.problem.parameters:
                            inputs[p['name']] = p['value']

                        result = system_model(**inputs)
                        val = result.get(c_name, 0)
                        if np.isnan(val) or np.isinf(val):
                            logger.warning("NaN/Inf detected in constraint '%s'; solver may get stuck", c_name)
                            return -1e15
                        # Return violation (positive = violation for ineq constraints)
                        return np.float64(limit - val)
                    except:
                        return -1e15
                cons.append({'type': 'ineq', 'fun': con_hi})

        return cons

class UserStopException(Exception):
    """Exception raised when optimization is stopped by user."""
    pass







