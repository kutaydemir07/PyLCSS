import numpy as np
from typing import List, Callable, Dict, Any, Tuple, Optional
from .core import Variable, Objective, Constraint

class ModelEvaluator:
    """
    Handles model execution, caching, scaling, and cost calculation.
    Decouples the math from the threading/GUI.
    """
    def __init__(self, system_model: Callable, variables: List[Variable], 
                 objectives: List[Objective], constraints: List[Constraint], 
                 parameters: Optional[Dict[str, Any]] = None,
                 scaling: bool = True,
                 penalty_weight: float = 1e6,
                 objective_scale: float = 1.0):
        self.model = system_model
        self.vars = variables
        self.objs = objectives
        self.cons = constraints
        self.parameters = parameters or {}
        # Scaling improves numerical stability for mixed-scale problems
        self.scaling = scaling
        self.penalty_weight = penalty_weight
        self.objective_scale = objective_scale
        
        # Caching
        self._cache = {}
        self._cache_size = 5000
        
        # Pre-calculate bounds for scaling
        self._lower = np.array([v.min_val for v in variables])
        self._upper = np.array([v.max_val for v in variables])
        self._ranges = self._upper - self._lower
        self._ranges[self._ranges == 0] = 1.0

    def to_normalized(self, x_phys: np.ndarray) -> np.ndarray:
        """Convert physical variables to normalized [0,1] space."""
        if not self.scaling:
            return x_phys
        return (x_phys - self._lower) / self._ranges

    def to_physical(self, x_norm: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] variables to physical space."""
        if not self.scaling:
            return x_norm
        return x_norm * self._ranges + self._lower

    def evaluate(self, x_input: np.ndarray) -> Tuple[float, Dict, float]:
        """Returns (total_cost, results_dict, max_violation)"""
        
        # --- FIX: Convert from normalized solver space to physical model space ---
        if self.scaling:
            x_phys = self.to_physical(x_input)
        else:
            x_phys = x_input
        # -----------------------------------------------------------------------
        
        # 2. Check Cache
        # Use x_input (solver space) for caching to avoid float precision issues
        cache_key = tuple(np.round(x_input, 12))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 3. Run System Model
        # Use x_phys (physical values) for the actual model execution
        inputs = {v.name: val for v, val in zip(self.vars, x_phys)}
        # Add parameters
        inputs.update(self.parameters)
        
        try:
            raw_res = self.model(**inputs)
        except Exception:
            # Return high cost for failures
            # Note: We return MAX_COST (1e15) instead of inf to play nice with solvers
            return 1e15, {}, 1e15

        # 4. Calculate Cost & Violations
        total_cost = 0.0
        max_violation = 0.0
        penalty_sum = 0.0
        
        # Objectives
        for obj in self.objs:
            val = raw_res.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            total_cost += sign * obj.weight * (val / self.objective_scale)

        # Constraints
        for con in self.cons:
            val = raw_res.get(con.name, 0.0)
            violation = 0.0
            if val < con.min_val: violation = con.min_val - val
            elif val > con.max_val: violation = val - con.max_val
            
            if violation > 0:
                max_violation = max(max_violation, violation)
                # Apply penalty weight to constraint violations
                penalty_sum += self.penalty_weight * violation

        # 5. Final Cost (Penalized)
        final_cost = total_cost + penalty_sum
        
        # Cap values to avoid overflow in solvers
        # Reduced from 1e50 to 1e15 to prevent RuntimeWarning in scipy internals
        MAX_COST = 1e15
        if np.isfinite(final_cost):
            if final_cost > MAX_COST: final_cost = MAX_COST
            elif final_cost < -MAX_COST: final_cost = -MAX_COST
        else:
            final_cost = MAX_COST if final_cost > 0 else -MAX_COST
        
        result = (final_cost, raw_res, max_violation)
        self._cache[cache_key] = result
        return result
