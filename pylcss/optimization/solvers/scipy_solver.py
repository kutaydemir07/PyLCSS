from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import warnings
import time
from .base import BaseSolver
from ..core import OptimizationResult

class ScipySolver(BaseSolver):
    def solve(self, evaluator, x0, callback):
        
        method = self.settings.get('method', 'SLSQP')
        maxiter = int(self.settings.get('maxiter', 1000))
        tol = float(self.settings.get('tol', 1e-6))
        atol = float(self.settings.get('atol', 1e-8))
        
        # Constrained methods list
        constrained_methods = ['SLSQP', 'COBYLA', 'trust-constr']
        supports_constraints = method in constrained_methods
        
        # Safe cap constant
        MAX_COST = 1e15

        # Track best solution
        best_cost = float('inf')
        best_x = None
        best_raw = None
        best_viol = None

        def obj_wrapper(x):
            if self.stop_requested: raise StopIteration
            
            # --- Evaluate ---
            # If scaling is on, 'x' here is normalized [0,1].
            # evaluator.evaluate handles the conversion to physical.
            cost, raw, viol = evaluator.evaluate(x)
            
            # Calculate unpenalized objective
            unpenalized_obj = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                unpenalized_obj += sign * obj.weight * (val / evaluator.objective_scale)
            
            # --- Track Best ---
            nonlocal best_cost, best_x, best_raw, best_viol
            if unpenalized_obj < best_cost:
                best_cost = unpenalized_obj
                best_x = np.array(x)
                best_raw = raw
                best_viol = viol

            # --- Callback ---
            # Just call the callback directly. Let the worker handle throttling.
            # Calculate real obj for UI
            real_obj_val = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                real_obj_val += sign * obj.weight * val

            callback(x, real_obj_val, raw, viol)
            
            # --- Return to Solver ---
            if supports_constraints:
                # Return unpenalized objective for constrained solvers
                return unpenalized_obj
            else:
                # For unconstrained methods, return full cost
                return cost

        if evaluator.scaling:
            x0_use = evaluator.to_normalized(np.array(x0))
            bounds = []
            for i, v in enumerate(evaluator.vars):
                if abs(v.max_val - v.min_val) < 1e-12:
                    bounds.append((0.0, 0.0))
                    x0_use[i] = 0.0
                else:
                    bounds.append((0.0, 1.0))
                    x0_use[i] = np.clip(x0_use[i], 0.0, 1.0)
        else:
            x0_use = np.array(x0)
            bounds = [(v.min_val, v.max_val) for v in evaluator.vars]

        cons = []
        if evaluator.cons and supports_constraints:
            
            if method == 'COBYLA':
                # COBYLA expects separate constraint functions, not vectorized
                for con in evaluator.cons:
                    def make_con_fun(con_name):
                        return lambda x: evaluator.evaluate(x)[1].get(con_name, 0.0)
                    con_fun = make_con_fun(con.name)
                    
                    if con.min_val != float('-inf'):
                        cons.append({'type': 'ineq', 'fun': lambda x, f=con_fun, m=con.min_val: f(x) - m})
                    if con.max_val != float('inf'):
                        cons.append({'type': 'ineq', 'fun': lambda x, f=con_fun, m=con.max_val: m - f(x)})
            else:
                def vectorized_cons(x):
                    # This calls evaluate() once, hitting the cache
                    _, raw, _ = evaluator.evaluate(x)
                    
                    residuals = []
                    for con in evaluator.cons:
                        val = raw.get(con.name, 0.0)
                        
                        # Inequality: val >= min  =>  val - min >= 0
                        if con.min_val != float('-inf'):
                            residuals.append(val - con.min_val)
                        
                        # Inequality: val <= max  =>  max - val >= 0
                        if con.max_val != float('inf'):
                            residuals.append(con.max_val - val)
                            
                    return np.array(residuals)

                # Register as a single vectorized constraint
                cons.append({'type': 'ineq', 'fun': vectorized_cons})

        # COBYLA Bound Fix
        if method == 'COBYLA':
            for i, (mn, mx) in enumerate(bounds):
                if mn is not None and mn > -1e19:
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, m=mn: x[i] - m})
                if mx is not None and mx < 1e19:
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, m=mx: m - x[i]})

        # --- Solver Options ---
        options = {'maxiter': maxiter}
            
        if method == 'SLSQP':
            options['ftol'] = tol
            # Add eps for gradient step size if specified
            if 'eps' in self.settings:
                options['eps'] = self.settings['eps']
            
        elif method == 'COBYLA':
            options['rhobeg'] = 0.5  # Allow faster movement across unit hypercube
            options['disp'] = False
        elif method == 'trust-constr':
            # Force a larger finite difference step to avoid delta_grad == 0
            # and tell it not to approximate the Hessian if it's unstable
            options['finite_diff_rel_step'] = 1e-4  # Larger step size

        kwargs = {
            'method': method,
            'bounds': bounds,
            'tol': tol, 
            'options': options
        }
        
        if supports_constraints or method == 'COBYLA':
            kwargs['constraints'] = cons

        try:
            res = minimize(obj_wrapper, x0_use, **kwargs)
            
            # Select Best Found vs Last Returned
            # Sometimes the last step of the solver is slightly worse than the best internal step
            final_x = res.x
            final_cost, final_raw, final_viol = evaluator.evaluate(final_x)
            
            # Calculate unpenalized objective for final point
            final_unpenalized = 0.0
            for obj in evaluator.objs:
                val = final_raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                final_unpenalized += sign * obj.weight * (val / evaluator.objective_scale)
            
            # If our tracked best is significantly better and valid, use it
            better_cost = best_cost < (final_unpenalized - 1e-9)
            valid_track = best_viol is None or best_viol < 1e-6
            
            if best_x is not None and better_cost and valid_track:
                x_phys = evaluator.to_physical(best_x)
                raw = best_raw
                viol = best_viol
                success = res.success # Trust solver's status even if we pick a better point
                message = getattr(res, 'message', 'Done') + " (Best Tracked)"
            else:
                x_phys = evaluator.to_physical(final_x)
                raw = final_raw
                viol = final_viol
                success = res.success
                message = getattr(res, 'message', str(res.success))

            # Reconstruct Real Objectives
            real_objectives = {obj.name: raw.get(obj.name, 0.0) for obj in evaluator.objs}
            constraints_val = {con.name: raw.get(con.name, 0.0) for con in evaluator.cons}
            
            # Calculate final real cost for the result object
            final_real_cost = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                final_real_cost += sign * obj.weight * val

            return OptimizationResult(
                x=x_phys,
                cost=final_real_cost,
                objectives=real_objectives,
                constraints=constraints_val,
                max_violation=viol,
                message=message,
                success=success
            )
            
        except StopIteration:
            return self._fallback_result(evaluator, best_x, best_raw, best_viol, "Stopped by user", x0_use)
        except Exception as e:
            return self._fallback_result(evaluator, best_x, best_raw, best_viol, f"Solver Failed: {str(e)}", x0_use)

    def _fallback_result(self, evaluator, best_x, best_raw, best_viol, msg, x0_use):
        # ... (Same as before, just ensure fallback logic handles None correctly)
        if best_x is not None:
            x_final = best_x
            raw = best_raw
            viol = best_viol
        else:
            x_final = x0_use
            _, raw, viol = evaluator.evaluate(x0_use)
            
        x_phys = evaluator.to_physical(x_final)
        
        # Calculate real cost
        real_cost = 0.0
        for obj in evaluator.objs:
            val = raw.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            real_cost += sign * obj.weight * val
            
        return OptimizationResult(
            x=x_phys,
            cost=real_cost,
            objectives={obj.name: raw.get(obj.name, 0.0) for obj in evaluator.objs},
            constraints={con.name: raw.get(con.name, 0.0) for con in evaluator.cons},
            max_violation=viol,
            message=msg,
            success=False
        )