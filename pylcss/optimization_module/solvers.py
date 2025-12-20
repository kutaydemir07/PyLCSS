# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Consolidated solvers and optimizers for PyLCSS.

This module contains all optimization algorithms used across the application,
including those for finding feasible points in solution space analysis and
single-objective optimization in the optimization tab.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from typing import List, Dict, Optional, Callable, Tuple
import traceback
import logging
import warnings

# Suppress CMA-ES numerical warnings
warnings.filterwarnings('ignore', message='.*overflow encountered in exp.*')
warnings.filterwarnings('ignore', message='.*elements of z2.*are larger than.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='cma')

# Suppress scipy optimization numerical warnings
warnings.filterwarnings('ignore', message='.*overflow encountered in scalar multiply.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')

# Suppress COBYLA RHOEND warnings from scipy
warnings.filterwarnings('ignore', message='.*COBYLA: Invalid RHOEND.*')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy._lib.pyprima')

logger = logging.getLogger(__name__)

# Optional: Import UserStopException if it's used elsewhere, otherwise this line is fine
# from .common import UserStopException 

# Default penalty value for constraint violations or failures
PENALTY_VALUE = 1e9

class SolverResult:
    """Standardized result from optimization solvers."""
    def __init__(self, x: np.ndarray, fun: float, message: str, success: bool = True):
        self.x = x
        self.fun = fun
        self.message = message
        self.success = success


def solve_with_slsqp(objective_func: Callable, x0: np.ndarray, bounds: List[Tuple[float, float]],
                     constraints: Optional[List[Dict]] = None, maxiter: int = 100, **kwargs) -> SolverResult:
    """
    Solve optimization problem using SLSQP.

    Args:
        objective_func: Function to minimize
        x0: Initial guess
        bounds: Variable bounds as list of (min, max) tuples
        constraints: Scipy-style constraints
        maxiter: Maximum iterations
        **kwargs: Additional arguments for the solver (e.g., tol, ftol)

    Returns:
        SolverResult with solution
    """
    try:
        options = {'maxiter': maxiter, 'disp': False}
        if 'ftol' in kwargs:
            options['ftol'] = kwargs['ftol']
        
        res = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=kwargs.get('tol', None),
            options=options
        )

        return SolverResult(
            x=np.array(res.x),
            fun=res.fun,
            message=res.message if hasattr(res, 'message') else str(res.success),
            success=res.success
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError, OverflowError) as e:
        logger.error(f"SLSQP failed with numerical error: {e}")
        logger.debug(traceback.format_exc())
        return SolverResult(
            x=x0,
            fun=float('inf'),
            message=f"SLSQP failed: {str(e)}",
            success=False
        )
    except Exception as e:
        # Re-raise unexpected errors (syntax, etc)
        logger.critical(f"SLSQP failed with unexpected error: {e}")
        logger.debug(traceback.format_exc())
        raise e


def solve_with_nevergrad(objective_func: Callable, x0: np.ndarray,
                         bounds: List[Tuple[float, float]], maxiter: int = 1000,
                         constraints: Optional[List[Dict]] = None,
                         callback: Optional[Callable] = None, **kwargs) -> SolverResult:
    """
    Solve optimization problem using Nevergrad with native constraint support.
    Uses the ask/tell interface with constraint violation vectors.

    Args:
        objective_func: Function to minimize
        x0: Initial guess
        bounds: Variable bounds as list of (min, max) tuples
        maxiter: Maximum function evaluations
        constraints: Scipy-style constraints list
        callback: Optional callback function called on each evaluation
        **kwargs: Additional arguments (budget_multiplier, num_workers)

    Returns:
        SolverResult with solution
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
        
        # Create Nevergrad parametrization
        parametrization = ng.p.Array(init=x0_clipped)
        parametrization.set_bounds(lower_bounds, upper_bounds)
        
        # Use as the default robust optimizer
        optimizer_name = kwargs.get('optimizer_name', "NGOpt")
        num_workers = kwargs.get('num_workers', 1)
        
        try:
            opt_cls = ng.optimizers.registry[optimizer_name]
        except KeyError:
            opt_cls = ng.optimizers.registry["NGOpt"]
            
        optimizer = opt_cls(parametrization=parametrization, budget=maxiter, num_workers=num_workers)
        
        if not np.array_equal(x0_active, x0_clipped):
            logger.warning("Initial guess was out of bounds and has been clipped for Nevergrad.")

        def calculate_violations(x_active):
            """Return list of constraint violations for a candidate (positive = violation)."""
            # Reconstruct full vector
            x_full = np.zeros(len(bounds))
            x_full[active_indices] = x_active
            for idx in fixed_indices:
                x_full[idx] = bounds[idx][0]
                
            violations = []
            if not constraints:
                return violations
                
            for constraint in constraints:
                if not callable(constraint.get('fun')):
                    continue
                
                # Evaluate constraint
                try:
                    vals = constraint['fun'](x_full)
                except Exception:
                    # If constraint calculation fails, treat as massive violation
                    vals = -PENALTY_VALUE 
                
                # Normalize scalar to iterable
                if np.isscalar(vals):
                    vals_iter = [vals]
                else:
                    vals_iter = vals
                
                # Process values
                for val in vals_iter:
                    val = float(val)
                    if constraint['type'] == 'ineq':
                        # ineq: satisfied if val >= 0
                        # violation if val < 0
                        # Nevergrad expects violation > 0
                        if val < 0:
                            violations.append(-val)
                        else:
                            violations.append(0.0)
                            
                    elif constraint['type'] == 'eq':
                        # eq: satisfied if val == 0
                        # violation = |val|
                        violations.append(abs(val))
            
            return violations

        # Run optimization using ask/tell interface
        for _ in range(maxiter):
            if optimizer.num_ask >= maxiter:
                break
                
            # Ask for a candidate
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                candidate = optimizer.ask()
            
            x_candidate_active = np.array(candidate.value)
            
            # Initialize variables to avoid UnboundLocalError in finally/except blocks
            objective_value = float('inf')
            current_violations = []

            try:
                objective_value = wrapped_objective(x_candidate_active)
                
                # Calculate constraints if they exist
                if constraints:
                    current_violations = calculate_violations(x_candidate_active)
                
                # Tell the optimizer about the evaluation (suppress warnings from internal optimizers)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    # Pass 3 arguments: candidate, loss, and violations
                    if current_violations:
                        optimizer.tell(candidate, objective_value, current_violations)
                    else:
                        optimizer.tell(candidate, objective_value)
                
                if callback:
                    # Reconstruct full vector for callback
                    x_full_cb = np.zeros(len(bounds))
                    x_full_cb[active_indices] = x_candidate_active
                    for idx in fixed_indices:
                        x_full_cb[idx] = bounds[idx][0]
                    callback(x_full_cb)

            except StopIteration:
                logger.info("Optimization cancelled by user.")
                break
                    
            except Exception as e:
                logger.exception("Evaluation error for candidate %s", x_candidate_active)
                # FIX: We must ensure current_violations is a list, even if calculation failed
                if not current_violations and constraints:
                    # Assume 1 violation per constraint if we failed before calculating them
                    current_violations = [PENALTY_VALUE] * len(constraints)
                elif not current_violations:
                    current_violations = [PENALTY_VALUE]

                # Tell with high penalty
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        optimizer.tell(candidate, 1e10, current_violations)
                except Exception:
                    # Fallback if 3-arg tell fails (e.g. very old ng version)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        optimizer.tell(candidate, 1e10)

        # Get the recommendation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            recommendation = optimizer.recommend()
        
        # Reconstruct full solution
        x_final_active = np.array(recommendation.value)
        x_final = np.zeros(len(bounds))
        x_final[active_indices] = x_final_active
        for idx in fixed_indices:
            x_final[idx] = bounds[idx][0]

        return SolverResult(
            x=x_final,
            fun=recommendation.loss,
            message=f"Nevergrad completed after {optimizer.num_ask} evaluations",
            success=True
        )
        
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
                                       constraints: Optional[List[Dict]] = None,
                                       maxiter: int = 1000, x0: Optional[np.ndarray] = None,
                                       callback: Optional[Callable] = None, **kwargs) -> SolverResult:
    """
    Solve optimization problem using Scipy Differential Evolution with native constraint support.

    Args:
        objective_func: Function to minimize
        bounds: Variable bounds as list of (min, max) tuples
        constraints: Scipy-style constraints list
        maxiter: Maximum iterations
        x0: Initial guess (not used by DE, population is random)
        callback: Optional callback function called on each evaluation
        **kwargs: Additional arguments (popsize, mutation, recombination, tol, seed)

    Returns:
        SolverResult with solution
    """
    try:
        eval_counter = 0
        def wrapped_objective(x):
            nonlocal eval_counter
            eval_counter += 1
            # Only call callback every 50 evaluations to reduce GUI overhead
            if callback and eval_counter % 50 == 0:
                # Ensure we yield to the event loop if running in a thread
                # But callback is just a function. The caller (SolverWorker) handles signals.
                callback(x)
            return objective_func(x)

        # Prepare constraints for differential_evolution
        de_constraints = []
        if constraints:
            for constraint in constraints:
                if constraint['type'] == 'ineq':
                    # inequality: fun >= 0 -> [0, inf]
                    nlc = NonlinearConstraint(constraint['fun'], 0, np.inf)
                    de_constraints.append(nlc)
                elif constraint['type'] == 'eq':
                    # equality: fun == 0 -> [0, 0]
                    nlc = NonlinearConstraint(constraint['fun'], 0, 0)
                    de_constraints.append(nlc)
        
        # Extract kwargs with defaults
        popsize = kwargs.get('popsize', 15)
        mutation = kwargs.get('mutation', (0.5, 1))
        recombination = kwargs.get('recombination', 0.7)
        tol = kwargs.get('tol', 0.01)
        seed = kwargs.get('seed', None)
        
        res = differential_evolution(
            wrapped_objective,
            bounds,
            constraints=tuple(de_constraints),
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            seed=seed,
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
    
    Args:
        objective_func: Function to minimize
        constraints_func: Function returning array of constraint values (must be >= 0)
        x0: Initial guess
        bounds: List of (min, max) tuples
        maxiter: Maximum iterations
        
    Returns:
        Optimized x vector (numpy array)
    """
    # Wrap constraints for SLSQP
    # SLSQP expects constraints as a list of dictionaries
    # Since constraints_func returns a vector of values >= 0, we can use a single 'ineq' constraint
    
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


