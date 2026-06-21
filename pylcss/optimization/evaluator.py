# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

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
                 objective_scale: float = 1.0,
                 constraint_margin: float = 0.0):
        self.model = system_model
        self.vars = variables
        self.objs = objectives
        self.cons = constraints
        self.parameters = parameters or {}
        # Scaling improves numerical stability for mixed-scale problems
        self.scaling = scaling
        self.penalty_weight = penalty_weight
        self.objective_scale = objective_scale
        # Relative safety back-off: solvers are constrained to a slightly tighter
        # band so the returned design satisfies the *original* constraints with
        # margin (never even slightly over). 0 disables it.
        self.constraint_margin = float(constraint_margin)

        # Caching
        self._cache = {}
        self._cache_size = 5000
        
        # Pre-calculate bounds for scaling
        self._lower = np.array([v.min_val for v in variables])
        self._upper = np.array([v.max_val for v in variables])
        self._ranges = self._upper - self._lower
        self._ranges[self._ranges == 0] = 1.0

        # Per-constraint scale used to report a *relative* (scale-invariant)
        # constraint violation. Prefer the admissible band width, then the
        # magnitude of the active bound, finally 1.0. This makes `max_violation`
        # and the penalty independent of the unit system, so feasibility
        # tolerances behave the same whether a quantity is ~1e-3 or ~1e8.
        self._con_scales = np.array(
            [self._constraint_scale(c) for c in self.cons], dtype=float
        ) if self.cons else np.array([], dtype=float)

        # Tightened ("solve") bounds: what the solvers are actually constrained to.
        # evaluate() still reports feasibility against the ORIGINAL bounds, so the
        # returned point ends up strictly inside the true feasible region.
        self._solve_lower = np.array([c.min_val for c in self.cons], dtype=float) if self.cons else np.array([])
        self._solve_upper = np.array([c.max_val for c in self.cons], dtype=float) if self.cons else np.array([])
        if self.cons and self.constraint_margin > 0:
            back = self.constraint_margin * self._con_scales
            lo, hi = self._solve_lower.copy(), self._solve_upper.copy()
            fl, fh = np.isfinite(lo), np.isfinite(hi)
            lo[fl] += back[fl]
            hi[fh] -= back[fh]
            # Never invert a finite band; collapse it to its midpoint instead.
            inverted = fl & fh & (lo > hi)
            mid = 0.5 * (self._solve_lower + self._solve_upper)
            lo[inverted] = mid[inverted]
            hi[inverted] = mid[inverted]
            self._solve_lower, self._solve_upper = lo, hi

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

    @staticmethod
    def _constraint_scale(con: Constraint) -> float:
        """Characteristic scale of a constraint, for relative violation reporting."""
        lo, hi = con.min_val, con.max_val
        if np.isfinite(lo) and np.isfinite(hi):
            width = abs(hi - lo)
            if width > 1e-12:
                return width
        mags = [abs(b) for b in (lo, hi) if np.isfinite(b) and abs(b) > 1e-12]
        return min(mags) if mags else 1.0

    def constraint_solve_bounds(self, i: int) -> Tuple[float, float]:
        """Tightened (lower, upper) bounds for building a solver's constraints."""
        return float(self._solve_lower[i]), float(self._solve_upper[i])

    def solve_violation(self, raw_res: Dict) -> float:
        """Max relative violation against the tightened (solve) bounds."""
        if not self.cons:
            return 0.0
        worst = 0.0
        for i, con in enumerate(self.cons):
            val = raw_res.get(con.name, 0.0)
            lo, hi = self._solve_lower[i], self._solve_upper[i]
            v = 0.0
            if val < lo:
                v = lo - val
            elif val > hi:
                v = val - hi
            if v > 0:
                worst = max(worst, v / self._con_scales[i])
        return worst

    def evaluate(self, x_input: np.ndarray) -> Tuple[float, Dict, float]:
        """Returns (total_cost, results_dict, max_violation)"""
        
        if self.scaling:
            x_phys = self.to_physical(x_input)
        else:
            x_phys = x_input
        
        # 2. Check Cache
        # aliasing x and x+eps during gradient calculation.
        cache_key = tuple(np.round(x_input, 15))
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

        # Constraints (violation reported relative to each constraint's scale)
        for i, con in enumerate(self.cons):
            val = raw_res.get(con.name, 0.0)
            raw_violation = 0.0
            if val < con.min_val: raw_violation = con.min_val - val
            elif val > con.max_val: raw_violation = val - con.max_val

            if raw_violation > 0:
                rel_violation = raw_violation / self._con_scales[i]
                max_violation = max(max_violation, rel_violation)
                # Apply penalty weight to the normalized constraint violation
                penalty_sum += self.penalty_weight * rel_violation

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
