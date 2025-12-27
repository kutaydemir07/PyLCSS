from .base import BaseSolver
from ..core import OptimizationResult
from .legacy import solve_with_nevergrad, solve_with_differential_evolution
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)

class GlobalSolver(BaseSolver):
    def solve(self, evaluator, x0, callback):
        
        method = self.settings.get('method', 'Nevergrad')
        maxiter = int(self.settings.get('maxiter', 1000))
        
        # Scaling Setup
        original_scaling = evaluator.scaling
        desired_scaling = bool(self.settings.get('scaling', False))
        
        lowers = np.array([v.min_val for v in evaluator.vars], dtype=float)
        uppers = np.array([v.max_val for v in evaluator.vars], dtype=float)
        all_finite_bounds = np.all(np.isfinite(lowers)) and np.all(np.isfinite(uppers))

        if desired_scaling and all_finite_bounds:
            evaluator.scaling = True
            bounds = [(0.0, 1.0) for _ in evaluator.vars]
            x0_use = evaluator.to_normalized(np.array(x0))
        else:
            if desired_scaling and not all_finite_bounds:
                logger.warning("Scaling requested but bounds infinite; disabling.")
            evaluator.scaling = False
            bounds = [(v.min_val, v.max_val) for v in evaluator.vars]
            x0_use = np.array(x0)
        
        # Prepare Constraints
        real_constraints = []
        if evaluator.cons:
            for cons in evaluator.cons:
                def make_fun(name):
                    return lambda x: evaluator.evaluate(x)[1].get(name, 0.0)
                fun = make_fun(cons.name)
                
                if cons.min_val is not None and np.isfinite(cons.min_val):
                    real_constraints.append({'type': 'ineq', 'fun': lambda x, f=fun, m=cons.min_val: f(x) - m})
                if cons.max_val is not None and np.isfinite(cons.max_val):
                    real_constraints.append({'type': 'ineq', 'fun': lambda x, f=fun, m=cons.max_val: m - f(x)})

        # Callbacks & Wrappers
        best_plot_val = float('inf')

        def objective_wrapper(x):
            cost, raw, _ = evaluator.evaluate(x)
            # Return unpenalized objective for global solvers
            unpenalized = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                unpenalized += sign * obj.weight * (val / evaluator.objective_scale)
            return unpenalized

        def callback_wrapper(x):
            nonlocal best_plot_val
            cost, raw, viol = evaluator.evaluate(x)
            
            # Calculate Real Objective for user
            real_obj_val = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                real_obj_val += sign * obj.weight * val
            
            # Filter noise
            if viol <= 1e-6 and real_obj_val < best_plot_val:
                best_plot_val = real_obj_val
                callback(x, real_obj_val, raw, viol)
            elif best_plot_val == float('inf'):
                best_plot_val = real_obj_val
                callback(x, real_obj_val, raw, viol)

        # Execution
        if method == 'Nevergrad':
            ng_kwargs = {
                'optimizer_name': self.settings.get('optimizer_name', 'NGOpt'),
                'num_workers': int(self.settings.get('num_workers', 1))
            }
            with warnings.catch_warnings():
                res = solve_with_nevergrad(
                    objective_wrapper, 
                    x0_use, 
                    bounds, 
                    maxiter=maxiter, 
                    constraints=real_constraints,
                    callback=callback_wrapper,
                    **ng_kwargs
                )
            
        elif method == 'Differential Evolution':
            de_kwargs = {}
            # ADD 'workers' and 'updating' to this list
            valid_keys = ['strategy', 'popsize', 'tol', 'mutation', 'recombination', 
                          'seed', 'disp', 'polish', 'atol', 'workers', 'updating']
            
            for k in valid_keys:
                if k in self.settings:
                    de_kwargs[k] = self.settings[k]

            # MAP the dialog setting 'num_workers' to SciPy's 'workers'
            if 'num_workers' in self.settings:
                 de_kwargs['workers'] = int(self.settings['num_workers'])
                 
                 # CRITICAL: As per SciPy docs, parallelization requires deferred updating
                 if de_kwargs['workers'] != 1:
                     de_kwargs['updating'] = 'deferred'

            res = solve_with_differential_evolution(
                objective_wrapper,
                bounds,
                constraints=real_constraints,
                maxiter=maxiter,
                x0=x0_use,
                callback=callback_wrapper,
                **de_kwargs
            )
        else:
            raise ValueError(f"Unknown global method: {method}")
            
        # Finalize
        final_cost, final_raw, final_viol = evaluator.evaluate(res.x)
        evaluator.scaling = original_scaling
        
        if desired_scaling and all_finite_bounds:
            x_phys = evaluator.to_physical(res.x)
        else:
            x_phys = res.x
            
        real_obj_final = 0.0
        for obj in evaluator.objs:
            val = final_raw.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            real_obj_final += sign * obj.weight * val

        objectives = {obj.name: final_raw.get(obj.name, 0.0) for obj in evaluator.objs}
        constraints = {con.name: final_raw.get(con.name, 0.0) for con in evaluator.cons}

        return OptimizationResult(
            x=x_phys,
            cost=real_obj_final,
            objectives=objectives,
            constraints=constraints,
            max_violation=final_viol,
            message=res.message,
            success=res.success
        )