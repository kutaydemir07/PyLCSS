# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Legacy solvers and optimizers for PyLCSS.
These are kept for compatibility with the solution_space module.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from typing import List, Dict, Optional, Callable, Tuple, Union, Any
import traceback
import logging
import warnings
import time

logger = logging.getLogger(__name__)

# Default penalty value for constraint violations or failures
PENALTY_VALUE = 1e9

class SolverResult:
    """Standardized result from optimization solvers."""
    def __init__(self, x: np.ndarray, fun: float, message: str, success: bool = True):
        self.x = x
        self.fun = fun
        self.message = message
        self.success = success

def solve_with_nevergrad(objective_func: Callable, x0: np.ndarray,
                         bounds: List[Tuple[float, float]], maxiter: int = 5000,
                         constraints: Optional[List[Union[Dict, NonlinearConstraint]]] = None,
                         callback: Optional[Callable] = None, **kwargs) -> SolverResult:
    """
    Solve optimization problem using Nevergrad with native constraint support.
    Uses the ask/tell interface with constraint violation vectors.
    Includes fallback logic for MetaModel crashes (scikit-learn/numpy incompatibility).
    """
    try:
        import nevergrad as ng
    except ImportError:
        return SolverResult(
            x=x0,
            fun=float('inf'),
            message="Nevergrad not available. Install with: pip install nevergrad",
            success=False
        )

    try:
        # Sanitize bounds for Nevergrad (cannot handle inf or min==max)
        MAX_BOUND = 1e20
        MIN_BOUND = -1e20
        
        # Identify fixed variables
        fixed_indices = []
        active_indices = []
        active_bounds = []
        
        for i, (lower, upper) in enumerate(bounds):
            if np.isclose(lower, upper, atol=1e-9):
                fixed_indices.append(i)
            else:
                active_indices.append(i)
                # Handle inf bounds
                l = max(lower, MIN_BOUND) if np.isfinite(lower) else MIN_BOUND
                u = min(upper, MAX_BOUND) if np.isfinite(upper) else MAX_BOUND
                active_bounds.append((l, u))
        
        # Wrapper to handle fixed variables
        def wrapped_objective(x_active):
            x_full = np.zeros(len(bounds))
            # Fill active variables
            x_full[active_indices] = x_active
            # Fill fixed variables
            for idx in fixed_indices:
                x_full[idx] = bounds[idx][0]
            return objective_func(x_full)
            
        # Clip initial guess to bounds (only active variables)
        x0_arr = np.array(x0)
        x0_active = x0_arr[active_indices]
        
        if not active_bounds:
            # All variables are fixed
            return SolverResult(
                x=x0_arr,
                fun=objective_func(x0_arr),
                message="All variables fixed",
                success=True
            )
            
        lower_bounds = np.array([b[0] for b in active_bounds])
        upper_bounds = np.array([b[1] for b in active_bounds])
        x0_clipped = np.clip(x0_active, lower_bounds, upper_bounds)
        
        # --- OPTIMIZATION SETUP & RETRY LOOP ---
        # We loop to allow falling back to 'TwoPointsDE' if a MetaModel optimizer crashes
        
        import os
        requested_optimizer = kwargs.get('optimizer_name', "NGOpt")
        num_workers = kwargs.get('num_workers', max(1, os.cpu_count() or 1))
        
        # List of error messages known to be caused by scikit-learn/nevergrad incompatibility
        KNOWN_METAMODEL_ERRORS = [
            "only 0-dimensional arrays can be converted to Python scalars",
            "only size-1 arrays can be converted to Python scalars"
        ]

        # Try up to 2 times: First with requested optimizer, then fallback to TwoPointsDE
        current_optimizer_name = requested_optimizer
        
        for attempt in range(2):
            try:
                # 1. Initialize Parametrization
                parametrization = ng.p.Array(init=x0_clipped)
                parametrization.set_bounds(lower_bounds, upper_bounds)
                
                # 2. Initialize Optimizer
                try:
                    opt_cls = ng.optimizers.registry[current_optimizer_name]
                except KeyError:
                    logger.warning(f"Optimizer {current_optimizer_name} not found, using NGOpt")
                    current_optimizer_name = "NGOpt"
                    opt_cls = ng.optimizers.registry["NGOpt"]
                
                optimizer = opt_cls(parametrization=parametrization, budget=maxiter, num_workers=num_workers)
                
                if not np.array_equal(x0_active, x0_clipped):
                    logger.warning("Initial guess was out of bounds and has been clipped for Nevergrad.")

                # Helper to calculate violations
                def calculate_violations(x_active):
                    """Return list of constraint violations for a candidate (positive = violation)."""
                    x_full = np.zeros(len(bounds))
                    x_full[active_indices] = x_active
                    for idx in fixed_indices:
                        x_full[idx] = bounds[idx][0]
                        
                    violations = []
                    if not constraints:
                        return violations
                        
                    for constraint in constraints:
                        if isinstance(constraint, NonlinearConstraint):
                            try:
                                vals = constraint.fun(x_full)
                                lb = constraint.lb
                                ub = constraint.ub
                            except Exception:
                                vals = np.nan 

                            if np.isscalar(vals):
                                vals_iter = [vals]
                                lb_iter = [lb] if np.isscalar(lb) else lb
                                ub_iter = [ub] if np.isscalar(ub) else ub
                            else:
                                vals_iter = vals
                                lb_iter = lb if hasattr(lb, '__iter__') else [lb] * len(vals)
                                ub_iter = ub if hasattr(ub, '__iter__') else [ub] * len(vals)

                            for i, val in enumerate(vals_iter):
                                if np.isnan(val) or np.isinf(val):
                                    violations.append(PENALTY_VALUE)
                                    continue
                                l = float(lb_iter[i]) if (np.isscalar(lb_iter) or i < len(lb_iter)) else -np.inf
                                u = float(ub_iter[i]) if (np.isscalar(ub_iter) or i < len(ub_iter)) else np.inf

                                if val < l:
                                    violations.append(l - val)
                                elif val > u:
                                    violations.append(val - u)
                                else:
                                    violations.append(0.0)
                            continue

                        if isinstance(constraint, dict):
                            if not callable(constraint.get('fun')):
                                continue
                            try:
                                vals = constraint['fun'](x_full)
                            except Exception:
                                vals = -PENALTY_VALUE 
                            
                            vals_iter = [vals] if np.isscalar(vals) else vals
                            for val in vals_iter:
                                val = float(val)
                                if constraint['type'] == 'ineq':
                                    if val < 0: violations.append(-val)
                                    else: violations.append(0.0)
                                elif constraint['type'] == 'eq':
                                    violations.append(abs(val))
                    return violations

                # 3. Run Optimization Loop
                for _ in range(maxiter):
                    if optimizer.num_ask >= maxiter:
                        break
                        
                    # Ask for a candidate
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        candidate = optimizer.ask()
                    
                    x_candidate_active = np.array(candidate.value)
                    
                    objective_value = float('inf')
                    current_violations = []

                    try:
                        objective_value = wrapped_objective(x_candidate_active)
                        
                        if constraints:
                            current_violations = calculate_violations(x_candidate_active)
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            if current_violations:
                                optimizer.tell(candidate, objective_value, current_violations)
                            else:
                                optimizer.tell(candidate, objective_value)
                        
                        if callback:
                            x_full_cb = np.zeros(len(bounds))
                            x_full_cb[active_indices] = x_candidate_active
                            for idx in fixed_indices:
                                x_full_cb[idx] = bounds[idx][0]
                            callback(x_full_cb)

                    except StopIteration:
                        logger.info("Optimization cancelled by user.")
                        break
                            
                    except Exception as e:
                        # Re-raise if this is the MetaModel crash to be caught by outer loop
                        if any(msg in str(e) for msg in KNOWN_METAMODEL_ERRORS):
                            raise e

                        logger.exception("Evaluation error for candidate %s", x_candidate_active)
                        if not current_violations and constraints:
                            current_violations = [PENALTY_VALUE] * len(constraints)
                        elif not current_violations:
                            current_violations = [PENALTY_VALUE]

                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning)
                                optimizer.tell(candidate, 1e10, current_violations)
                        except Exception:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning)
                                optimizer.tell(candidate, 1e10)

                # 4. Get Recommendation (Success)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    recommendation = optimizer.recommend()
                
                x_final_active = np.array(recommendation.value)
                x_final = np.zeros(len(bounds))
                x_final[active_indices] = x_final_active
                for idx in fixed_indices:
                    x_final[idx] = bounds[idx][0]

                return SolverResult(
                    x=x_final,
                    fun=recommendation.loss,
                    message=f"{current_optimizer_name} completed after {optimizer.num_ask} evaluations",
                    success=True
                )

            except TypeError as e:
                # Catch the specific MetaModel/Scikit-learn incompatibility error
                if any(msg in str(e) for msg in KNOWN_METAMODEL_ERRORS):
                    if current_optimizer_name == "TwoPointsDE":
                        # If TwoPointsDE fails with this, we are truly stuck
                        raise e
                    
                    logger.error(f"Optimizer '{current_optimizer_name}' crashed due to library incompatibility (MetaModel/scikit-learn issue).")
                    logger.warning("Falling back to 'TwoPointsDE' and restarting optimization.")
                    
                    current_optimizer_name = "TwoPointsDE"
                    continue # Restart loop with new optimizer
                
                # Re-raise other TypeErrors
                raise e

    except (ImportError, ValueError, RuntimeError) as e:
        return SolverResult(
            x=x0,
            fun=float('inf'),
            message=f"Nevergrad failed: {str(e)}",
            success=False
        )
    except Exception as e:
        # Re-raise unexpected errors (syntax, etc)
        raise e


def solve_with_differential_evolution(objective_func: Callable, bounds: List[Tuple[float, float]],
                                       constraints: Optional[List[Union[Dict, NonlinearConstraint]]] = None,
                                       maxiter: int = 5000, x0: Optional[np.ndarray] = None,
                                       callback: Optional[Callable] = None, **kwargs) -> SolverResult:
    """
    Solve optimization problem using Scipy Differential Evolution with native constraint support.
    """
    try:
        # --- REMOVED WRAPPER HACK ---
        # We no longer wrap objective_func to call callback.
        # This prevents plotting random candidate evaluations.
        
        # --- NEW CODE: Native Callback Adapter ---
        def native_callback(xk, convergence=None):
            """
            Called by Scipy DE at the end of each generation.
            xk is the best solution of the current population.
            """
            if callback:
                callback(xk)
                # Keep the GIL yield for GUI responsiveness
                time.sleep(0.001)
        # ----------------------------------------

        # Prepare constraints for differential_evolution
        de_constraints = []
        if constraints:
            for constraint in constraints:
                if isinstance(constraint, NonlinearConstraint):
                    de_constraints.append(constraint)
                elif isinstance(constraint, dict):
                    if constraint['type'] == 'ineq':
                        # inequality: fun >= 0 -> [0, inf]
                        nlc = NonlinearConstraint(constraint['fun'], 0.0, np.inf)
                        de_constraints.append(nlc)
                    elif constraint['type'] == 'eq':
                        # equality: fun == 0 -> [0, 0]
                        nlc = NonlinearConstraint(constraint['fun'], 0.0, 0.0)
                        de_constraints.append(nlc)
        
        # Extract kwargs with defaults
        popsize = kwargs.get('popsize', 15)
        mutation = kwargs.get('mutation', (0.5, 1))
        recombination = kwargs.get('recombination', 0.7)
        strategy = kwargs.get('strategy', 'best1bin')
        tol = kwargs.get('tol', 0.01)
        seed = kwargs.get('seed', None)
        
        res = differential_evolution(
            objective_func, # Pass the original function directly
            bounds,
            constraints=tuple(de_constraints),
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            tol=tol,
            seed=seed,
            callback=native_callback, # Use our new adapter
            disp=False,
            polish=True
        )

        return SolverResult(
            x=res.x,
            fun=res.fun,
            message=res.message,
            success=res.success
        )
    except Exception as e:
        logger.error(f"Differential Evolution failed: {e}")
        logger.debug(traceback.format_exc())
        return SolverResult(
            x=None,
            fun=float('inf'),
            message=f"Differential Evolution failed: {str(e)}",
            success=False
        )


def run_goal_attainment_slsqp(objective_func: Callable, constraints_func: Callable, 
                              x0: np.ndarray, bounds: List[Tuple[float, float]], 
                              maxiter: int = 5000) -> np.ndarray:
    """
    Runs SLSQP optimization for goal attainment / feasible point search.
    Used specifically by the SolutionSpaceSolver for finding initial feasible points.
    """
    # Wrap constraints for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': constraints_func}
    ]
    
    try:
        res = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'disp': False}
        )
        return res.x
    except Exception as e:
        logger.error(f"SLSQP Goal Attainment failed: {e}")
        return x0 # Return initial guess on failure