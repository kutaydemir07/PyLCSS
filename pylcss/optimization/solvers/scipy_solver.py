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

        # --- Throttling State for GUI Updates ---
        last_callback_time = 0.0
        CALLBACK_INTERVAL = 0.1  # 100ms minimum between GUI updates

        # --- Explicit Warning Filtering ---
        # Only filter specific expected warnings, not all RuntimeWarnings
        # This prevents masking legitimate bugs in user-defined nodes
        with warnings.catch_warnings():
            # Filter expected numerical issues from optimization
            warnings.filterwarnings("ignore", message=".*divide by zero.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*overflow.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*invalid value.*", category=RuntimeWarning)
            # Allow other RuntimeWarnings to show (potential user code issues)

        # Track best solution
        best_cost = float('inf')
        best_x = None
        best_raw = None
        best_viol = None

        def obj_wrapper(x):
            if self.stop_requested: raise StopIteration
            
            # --- 1. Linear Penalty for Bound Violations ---
            # Replace "brick wall" with distance-based penalty that guides solver back
            penalty = 0.0
            is_out_of_bounds = False
            
            # Check bounds and calculate distance
            if evaluator.scaling:
                # Scaled mode: bounds are [0, 1]
                if np.any(x < 0):
                    penalty += np.sum(np.abs(x[x < 0]))
                    is_out_of_bounds = True
                if np.any(x > 1):
                    penalty += np.sum(np.abs(x[x > 1] - 1))
                    is_out_of_bounds = True
            else:
                # Physical bounds
                for i, val in enumerate(x):
                    v_var = evaluator.vars[i]
                    if val < v_var.min_val:
                        penalty += abs(v_var.min_val - val)
                        is_out_of_bounds = True
                    elif val > v_var.max_val:
                        penalty += abs(val - v_var.max_val)
                        is_out_of_bounds = True
            
            if is_out_of_bounds:
                # Return Max Cost + Penalty Slope to guide solver back
                return MAX_COST + (penalty * 1e5)

            # --- 2. Evaluate ---
            cost, raw, viol = evaluator.evaluate(x)
            
            # Add penalty to cost (only if not already returned above)
            penalized_cost = cost + penalty
            
            # --- 3. Track Best ---
            nonlocal best_cost, best_x, best_raw, best_viol
            if cost < best_cost:
                best_cost = cost
                best_x = np.array(x)
                best_raw = raw
                best_viol = viol

            # --- 4. Throttled Callback to Prevent GUI Freeze ---
            nonlocal last_callback_time
            current_time = time.time()
            if current_time - last_callback_time >= CALLBACK_INTERVAL:
                # Calculate real obj for UI
                real_obj_val = 0.0
                for obj in evaluator.objs:
                    val = raw.get(obj.name, 0.0)
                    sign = 1.0 if obj.minimize else -1.0
                    real_obj_val += sign * obj.weight * val

                callback(x, real_obj_val, raw, viol)
                last_callback_time = current_time 
            
            # Return PENALIZED cost to solver
            if supports_constraints:
                # Re-calculate Scaled Objective (without penalty)
                scaled_obj_cost = 0.0
                for obj in evaluator.objs:
                    val = raw.get(obj.name, 0.0)
                    sign = 1.0 if obj.minimize else -1.0
                    scaled_obj_cost += sign * obj.weight * (val / evaluator.objective_scale)
                # Add penalty to scaled objective for constrained solvers
                return scaled_obj_cost + penalty
            else:
                # For unconstrained solvers, return penalized cost directly
                return penalized_cost

        # Handle Scaling for x0 and bounds
        if evaluator.scaling:
            x0_use = evaluator.to_normalized(np.array(x0))
            bounds = [(0, 1)] * len(x0)
        else:
            x0_use = np.array(x0)
            bounds = [(v.min_val, v.max_val) for v in evaluator.vars]

        # Constraints Construction
        cons = []
        if evaluator.cons and supports_constraints:
            for con in evaluator.cons:
                if con.min_val != float('-inf'):
                    def lb_fun(x, name=con.name, limit=con.min_val):
                        _, raw, _ = evaluator.evaluate(x)
                        val = raw.get(name, 0.0)
                        return val - limit
                    cons.append({'type': 'ineq', 'fun': lb_fun})
                
                if con.max_val != float('inf'):
                    def ub_fun(x, name=con.name, limit=con.max_val):
                        _, raw, _ = evaluator.evaluate(x)
                        val = raw.get(name, 0.0)
                        return limit - val
                    cons.append({'type': 'ineq', 'fun': ub_fun})

        # COBYLA Bound Fix
        if method == 'COBYLA':
            for i, (mn, mx) in enumerate(bounds):
                if mn is not None and mn > -1e19:
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, m=mn: x[i] - m})
                if mx is not None and mx < 1e19:
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, m=mx: m - x[i]})

        # --- 5. Strict Option Handling ---
        options = {}
        max_evals = maxiter * 10
        
        if method == 'COBYLA':
            options['maxiter'] = maxiter
            options['tol'] = tol
            # FIX: rhobeg must be smaller than the domain (1.0). 
            # 0.2 is a safe start for [0,1] scaled variables.
            options['rhobeg'] = 0.2 
            
        elif method == 'SLSQP':
            options['maxiter'] = maxiter
            options['ftol'] = tol
            # Good standard step for normalized variables
            options['eps'] = 1e-4 
            
        elif method in ['L-BFGS-B', 'TNC']:
            options['maxfun'] = max_evals
            options['ftol'] = tol
            options['gtol'] = atol
            if method == 'L-BFGS-B': 
                options['maxiter'] = maxiter
                options['eps'] = 1e-4
                
        elif method == 'trust-constr':
            options['maxiter'] = maxiter
            options['xtol'] = tol
            options['gtol'] = atol
            # FIX: Explicit finite diff step prevents "delta_grad == 0.0" errors
            options['finite_diff_rel_step'] = 1e-4
            
        else:
            options['maxiter'] = maxiter
            options['gtol'] = tol

        kwargs = {
            'method': method,
            'bounds': bounds,
            'options': options
        }
        
        if supports_constraints or method == 'COBYLA':
            kwargs['constraints'] = cons

        try:
            # Execute optimization with specific warning filtering
            res = minimize(obj_wrapper, x0_use, **kwargs)
            
            # Result Extraction
            if best_x is not None and best_cost < (evaluator.evaluate(res.x)[0] - 1e-9):
                x_final = best_x
                raw = best_raw
                viol = best_viol
                message = getattr(res, 'message', 'Done') + " (Best Found)"
                cost = best_cost
            else:
                x_final = res.x
                cost, raw, viol = evaluator.evaluate(x_final)
                message = getattr(res, 'message', str(res.success))

            x_phys = evaluator.to_physical(x_final)
            
            # Real Objective for Final Result
            real_obj_val = 0.0
            for obj in evaluator.objs:
                val = raw.get(obj.name, 0.0)
                sign = 1.0 if obj.minimize else -1.0
                real_obj_val += sign * obj.weight * val

            objectives = {obj.name: raw.get(obj.name, 0.0) for obj in evaluator.objs}
            constraints = {con.name: raw.get(con.name, 0.0) for con in evaluator.cons}
            
            return OptimizationResult(
                x=x_phys, 
                cost=real_obj_val, # Real objective
                objectives=objectives,
                constraints=constraints,
                max_violation=viol,
                message=message,
                success=res.success
            )
            
        except StopIteration:
            return self._fallback_result(evaluator, best_x, best_raw, best_viol, best_cost, x0_use, "Stopped by user")
            
        except Exception as e:
            return self._fallback_result(evaluator, best_x, best_raw, best_viol, best_cost, x0_use, f"Solver Failed: {str(e)}")

    def _fallback_result(self, evaluator, best_x, best_raw, best_viol, best_cost, x0_use, msg):
        if best_x is not None:
            x_final = best_x
            raw = best_raw
            viol = best_viol
        else:
            x_final = x0_use
            _, raw, viol = evaluator.evaluate(x0_use)
            
        x_phys = evaluator.to_physical(x_final)
        
        real_obj_val = 0.0
        for obj in evaluator.objs:
            val = raw.get(obj.name, 0.0)
            sign = 1.0 if obj.minimize else -1.0
            real_obj_val += sign * obj.weight * val

        objectives = {obj.name: raw.get(obj.name, 0.0) for obj in evaluator.objs}
        constraints = {con.name: raw.get(con.name, 0.0) for con in evaluator.cons}
        
        return OptimizationResult(
            x=x_phys,
            cost=real_obj_val,
            objectives=objectives,
            constraints=constraints,
            max_violation=viol,
            message=msg,
            success=False
        )