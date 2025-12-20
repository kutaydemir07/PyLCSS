# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np
import logging
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy.optimize import minimize
from ..problem_definition.problem_setup import XRayProblem
import time
import colorsys
import tempfile
import importlib.util
import warnings

from ..user_interface.text_utils import format_html
from .common import OptimizationSetup, UserStopException
from .solvers import solve_with_nevergrad, solve_with_differential_evolution
from .solver_wizard import SolverSelectionWizard, ConvergenceDiagnostics
from ..config import SOLVER_DESCRIPTIONS
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_plot_color(index: int, total_lines: int) -> str:
    if total_lines <= 1: return 'b'  # Blue instead of white
    colors = []
    num_colors = max(20, total_lines)
    for i in range(num_colors):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors[index % len(colors)]

class OptimizationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, float, list, dict, dict, float)
    finished = QtCore.Signal(object, str)
    error = QtCore.Signal(str)

    def __init__(self, problem: XRayProblem, objectives_list: List[Dict[str, Any]], 
                 constraints_list: List[Dict[str, Any]], x0: List[float], method: str = 'SLSQP', 
                 maxiter: int = 100, constraint_scaling: bool = True, **kwargs) -> None:
        super().__init__()
        # Ensure method is a valid string
        valid_methods = ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr', 'COBYLA', 'Nelder-Mead', 'Powell', 'Nevergrad', 'Differential Evolution']
        if not isinstance(method, str) or method not in valid_methods:
            method = 'SLSQP'
        self.problem = problem
        self.objectives_list = objectives_list
        self.constraints_list = constraints_list
        self.x0 = x0
        self.method = method
        self.maxiter = maxiter
        self.constraint_scaling = constraint_scaling
        self.solver_options = kwargs
        
        # Store system_model reference to avoid potential corruption
        self.system_model = problem.system_model
        
        # Ensure system_model is callable
        if not callable(self.system_model):
            raise TypeError(f"system_model is not callable: {type(self.system_model)}")
        
        self.history_cost = []
        self.history_x = []
        self.iteration = 0
        
        # State tracking for "Best So Far"
        self.best_cost = float('inf')
        self.best_x = list(x0)
        self.best_cons_vals = {}
        self.best_obj_vals = {}
        self.best_max_violation = 0.0
        
        # Store last evaluation results to avoid double evaluation in callback
        self.last_eval_x = None
        self.last_eval_cost = None
        self.last_eval_obj_vals = {}
        self.last_eval_cons_vals = {}
        self.last_eval_max_violation = 0.0
        
        self._is_running = True
        self.last_update_time = 0.0
        
        # Create common optimization setup
        self.opt_setup = OptimizationSetup(problem, objectives_list, constraints_list, constraint_scaling)

    def _progress_callback(self, x: np.ndarray, cost: float, obj_vals: List[float], 
                          cons_vals: Dict[str, float], max_violation: float) -> None:
        """Callback for progress updates from blackbox function."""
        # Update GUI Logic
        self.iteration += 1
        
        # Always update best tracking based on the cost the optimizer is minimizing
        update_best = False
        if self.best_cost == float('inf'):
            update_best = True
        elif cost < self.best_cost:
            update_best = True
        
        if update_best:
            self.best_cost = cost
            self.best_x = x[:]
            self.best_cons_vals = cons_vals
            self.best_obj_vals = {obj['name']: val for obj, val in zip(self.objectives_list, obj_vals)}
            self.best_max_violation = max_violation
        
        # Limit GUI updates to 5Hz (every 0.2s) to prevent freezing
        current_time = time.time()
        if (current_time - self.last_update_time) > 0.2:
            self.progress.emit(
                self.iteration, 
                self.best_cost,  # Show best cost so far
                self.best_x,  # Show best x so far
                self.best_cons_vals,  # Show best constraint values
                self.best_obj_vals,  # Show best objective values
                self.best_max_violation  # Show best max violation
            )
            self.last_update_time = current_time

    def _evaluate_objective_constraints(self, x):
        """Evaluate objective and constraints for a given design vector x."""
        if not self._is_running:
            raise StopIteration("Optimization cancelled")

        # Convert to physical space
        x_phys = []
        for i, val in enumerate(x):
            lower, upper = self.phys_bounds[i]
            
            if np.isinf(lower) or np.isinf(upper):
                phys_val = val
            else:
                # Nevergrad works in bounded space, but we ensure bounds are respected
                phys_val = np.clip(val, lower, upper)
            
            x_phys.append(phys_val)

        # Evaluate objectives and constraints
        dv_names = self.opt_setup.dv_names
        inputs = {name: x_phys[i] for i, name in enumerate(dv_names)}
        for p in self.problem.parameters:
            inputs[p['name']] = p['value']

        try:
            res = self.system_model(**inputs)

            # Calculate objective value
            if len(self.objectives_list) == 1:
                obj = self.objectives_list[0]
                val = float(res.get(obj['name'], 0.0))
                weight = obj.get('weight', 1.0)
                if obj.get('minimize', False):
                    obj_val = weight * val
                elif obj.get('maximize', False):
                    obj_val = -weight * val
                else:
                    obj_val = val
            else:
                # Single objective optimization
                obj_val = 0.0
                for obj in self.objectives_list:
                    val = float(res.get(obj['name'], 0.0))
                    weight = obj.get('weight', 1.0)
                    if obj.get('minimize', False):
                        obj_val += weight * val
                    elif obj.get('maximize', False):
                        obj_val += -weight * val

            # Calculate constraint violations
            max_violation = 0.0
            penalty = 0.0
            cons_vals = {}
            obj_vals = {}
            
            for constr in self.constraints_list:
                val = float(res.get(constr['name'], 0.0))
                cons_vals[constr['name']] = val
                c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                if c_min > -1e8:
                    violation = max(0, c_min - val)
                    max_violation = max(max_violation, violation)
                    penalty += violation ** 2
                if c_max < 1e8:
                    violation = max(0, val - c_max)
                    max_violation = max(max_violation, violation)
                    penalty += violation ** 2

            # Store objective values
            for obj in self.objectives_list:
                val = float(res.get(obj['name'], 0.0))
                obj_vals[obj['name']] = val

            penalty_weight = self.solver_options.get('penalty_weight', 1000.0)
            total_cost = obj_val + penalty_weight * penalty

            # Progress callback
            self._progress_callback(np.array(x_phys), total_cost, [obj_val], cons_vals, max_violation)

            return total_cost

        except Exception as e:
            # Return high penalty for failed evaluations
            return 1e10

    def _evaluate_raw_objective(self, x):
        """
        Evaluate objective WITHOUT adding penalty. 
        Used for solvers that handle constraints natively (Nevergrad, Differential Evolution).
        """
        if not self._is_running:
            raise StopIteration("Optimization cancelled")

        # Evaluate objectives and constraints
        dv_names = self.opt_setup.dv_names
        inputs = {name: x[i] for i, name in enumerate(dv_names)}
        for p in self.problem.parameters:
            inputs[p['name']] = p['value']

        try:
            res = self.system_model(**inputs)

            # Calculate objective value
            obj = self.objectives_list[0]
            val = float(res.get(obj['name'], 0.0))
            weight = obj.get('weight', 1.0)
            if obj.get('minimize', False):
                obj_val = weight * val
            elif obj.get('maximize', False):
                obj_val = -weight * val
            else:
                obj_val = val

            # Calculate constraint violations and penalties for GUI consistency
            max_violation = 0.0
            penalty = 0.0
            cons_vals = {}
            
            for constr in self.constraints_list:
                val = float(res.get(constr['name'], 0.0))
                cons_vals[constr['name']] = val
                c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                
                if c_min > -1e8:
                    violation = max(0, c_min - val)
                    max_violation = max(max_violation, violation)
                    penalty += violation ** 2
                if c_max < 1e8:
                    violation = max(0, val - c_max)
                    max_violation = max(max_violation, violation)
                    penalty += violation ** 2

            # Add penalty to total cost for GUI consistency (even though Nevergrad handles constraints natively)
            penalty_weight = self.solver_options.get('penalty_weight', 1000.0)
            total_cost = obj_val + penalty_weight * penalty 
            
            # Progress callback
            self._progress_callback(x, total_cost, [obj_val], cons_vals, max_violation)

            return total_cost
        except Exception as e:
            # Still call progress callback to show failed evaluation
            self._progress_callback(x, 1e10, [1e10], {}, 1e10)
            return 1e10

    def run(self):
        try:
            x0 = self.x0
            
            # Use bounds and names from common setup
            bounds = self.opt_setup.bounds
            dv_names = self.opt_setup.dv_names
            
            # Define physical bounds for both optimization methods
            self.phys_bounds = bounds
            
            # Ensure system_model is callable
            if not callable(self.system_model):
                raise TypeError(f"system_model is not callable: {type(self.system_model)}")
            
            # --- Execution ---
            try:
                if self.method == 'Nevergrad':
                    # FIX C: Handle initial guess clipping - evaluate original x0 first for GUI consistency
                    x0_original = np.array(x0)
                    
                    # Check if x0 needs clipping for Nevergrad
                    lower_bounds = np.array([b[0] for b in bounds])
                    upper_bounds = np.array([b[1] for b in bounds])
                    x0_clipped = np.clip(x0_original, lower_bounds, upper_bounds)
                    
                    # If x0 was clipped, evaluate original first for plot consistency
                    if not np.array_equal(x0_original, x0_clipped):
                        try:
                            # Evaluate original x0 to show in plots
                            self._evaluate_raw_objective(x0_original)
                        except:
                            pass  # If evaluation fails, just proceed
                    
                    # Use centralized Nevergrad solver
                    solver_result = solve_with_nevergrad(
                        objective_func=self._evaluate_raw_objective,  # Use raw objective since constraints are handled separately
                        x0=x0_clipped,  # Use clipped version for Nevergrad
                        bounds=bounds,
                        constraints=self.opt_setup.get_physical_constraints(self.system_model, lambda: self._is_running),
                        maxiter=self.maxiter,
                        **self.solver_options
                    )

                    # FIX A: Robust Result Handling - Accept result as long as x exists
                    if solver_result.x is not None:
                        class NevergradResult:
                            def __init__(self, x, fun, msg):
                                self.x = np.array(x)
                                self.fun = fun
                                self.message = msg

                        res = NevergradResult(solver_result.x, solver_result.fun, solver_result.message)
                    else:
                        self.error.emit(solver_result.message)
                        return

                elif self.method == 'Differential Evolution':
                    # Use centralized Differential Evolution solver
                    solver_result = solve_with_differential_evolution(
                        objective_func=self._evaluate_objective_constraints,  # Use penalized objective for consistency
                        bounds=bounds,
                        constraints=self.opt_setup.get_physical_constraints(self.system_model, lambda: self._is_running),
                        maxiter=self.maxiter,
                        **self.solver_options
                    )

                    # FIX A: Robust Result Handling - Accept result as long as x exists
                    if solver_result.x is not None:
                        res = solver_result
                    else:
                        self.error.emit(solver_result.message)
                        return

                else:
                    # Scipy Logic - Always normalize variables to [0,1] range
                        
                        # 2. Normalize Initial Guess (x0)
                        # optimizer will work in [0, 1] space
                        x0_array = np.array(self.x0, dtype=np.float64)
                        x0_norm = np.array(self.opt_setup._to_normalized(x0_array), dtype=np.float64)
                        
                        # 3. Create Normalized Bounds (0, 1)
                        norm_bounds = [(0.0, 1.0) for _ in self.phys_bounds]

                        def objective(x_norm):
                            if not self._is_running: raise UserStopException("Optimization stopped by user.")
                            
                            # SCALE BACK TO PHYSICAL
                            x_phys = self.opt_setup._to_physical(x_norm)
                            
                            # x_phys is NumPy array from scipy
                            inputs = {name: x_phys[i] for i, name in enumerate(dv_names)}
                            for p in self.problem.parameters: inputs[p['name']] = p['value']
                            
                            try:
                                if not callable(self.system_model):
                                    raise TypeError(f"system_model is not callable: {type(self.system_model)}")
                                
                                # Standard Deterministic Calculation
                                res = self.system_model(**inputs)
                                total_cost = 0.0
                                obj_vals = {}
                                for obj in self.objectives_list:
                                    val = res.get(obj['name'], 0.0)
                                    obj_vals[obj['name']] = val
                                    weight = obj.get('weight', 1.0)
                                    if obj.get('minimize', False): total_cost += weight * val
                                    if obj.get('maximize', False): total_cost -= weight * val
                                
                                # Store last evaluation results
                                self.last_eval_x = x_phys
                                self.last_eval_cost = total_cost
                                self.last_eval_obj_vals = obj_vals
                                
                                # Calculate constraint values for storage
                                cons_vals = {}
                                max_violation = 0.0
                                for constr in self.constraints_list:
                                    val = res.get(constr['name'], 0.0)
                                    cons_vals[constr['name']] = val
                                    c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                                    if c_min > -1e8: max_violation = max(max_violation, max(0, c_min - val))
                                    if c_max < 1e8: max_violation = max(max_violation, max(0, val - c_max))
                                
                                self.last_eval_cons_vals = cons_vals
                                self.last_eval_max_violation = max_violation
                                
                                # Add constraint penalties for unconstrained methods
                                if self.method in ['Nelder-Mead', 'Powell']:
                                    penalty = 0.0
                                    for constr in self.constraints_list:
                                        val = res.get(constr['name'], 0.0)
                                        if not np.isfinite(val):
                                            penalty += 1e15
                                            continue

                                        c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                                        if c_min > -1e8:
                                            violation = max(0, c_min - val)
                                            # Clamp violation to avoid overflow when squaring (limit to 1e50)
                                            if violation > 1e50: violation = 1e50
                                            penalty += 1000 * violation ** 2
                                        if c_max < 1e8:
                                            violation = max(0, val - c_max)
                                            if violation > 1e50: violation = 1e50
                                            penalty += 1000 * violation ** 2
                                    penalty_weight = self.solver_options.get('penalty_weight', 1000.0)
                                    total_cost += penalty_weight * penalty
                                
                                if np.isnan(total_cost) or np.isinf(total_cost) or total_cost > 1e100: 
                                    return np.float64(1e100)
                                
                                # FIX: Track best solution directly in objective function (works even without callbacks)
                                update_best = False
                                if self.best_cost == float('inf'): update_best = True
                                elif total_cost < self.best_cost: update_best = True
                                
                                if update_best:
                                    self.best_cost = total_cost
                                    self.best_x = x_phys.tolist() if hasattr(x_phys, 'tolist') else list(x_phys)
                                    self.best_cons_vals = cons_vals.copy()
                                    self.best_obj_vals = obj_vals.copy()
                                    self.best_max_violation = max_violation
                                
                                self.iteration += 1
                                return np.float64(total_cost)
                            except UserStopException: raise
                            except Exception: return np.float64(1e15)
                        
                        cons = self.opt_setup.get_scipy_constraints(self.system_model, lambda: self._is_running)
                                
                        def callback(xk_norm, *args):
                            if not self._is_running: raise UserStopException("Optimization stopped by user.")
                            
                            # Use stored results from last objective evaluation to avoid double evaluation
                            if self.last_eval_x is not None:
                                xk_phys = self.last_eval_x
                                total_cost = self.last_eval_cost
                                obj_vals = self.last_eval_obj_vals
                                cons_vals = self.last_eval_cons_vals
                                max_violation = self.last_eval_max_violation
                            else:
                                # Fallback: if no stored results, evaluate (shouldn't happen in normal flow)
                                xk_phys = self.opt_setup._to_physical(xk_norm)
                                inputs = {name: xk_phys[i] for i, name in enumerate(dv_names)}
                                for p in self.problem.parameters: inputs[p['name']] = p['value']
                                try:
                                    res = self.system_model(**inputs)
                                    total_cost = 0.0
                                    obj_vals = {}
                                    for obj in self.objectives_list:
                                        val = float(res.get(obj['name'], 0.0))
                                        if not np.isfinite(val): val = 1e15 # Safety clamp
                                        obj_vals[obj['name']] = val
                                        weight = obj.get('weight', 1.0)
                                        if obj.get('minimize', False): total_cost += weight * val
                                        if obj.get('maximize', False): total_cost -= weight * val
                                    cons_vals = {}
                                    max_violation = 0.0
                                    for constr in self.constraints_list:
                                        val = float(res.get(constr['name'], 0.0))
                                        cons_vals[constr['name']] = val
                                        
                                        if not np.isfinite(val):
                                            max_violation = 1e15
                                            continue

                                        c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                                        if c_min > -1e8: max_violation = max(max_violation, max(0, c_min - val))
                                        if c_max < 1e8: max_violation = max(max_violation, max(0, val - c_max))
                                except:
                                    return  # Skip update on error
                            
                            # NOTE: Best tracking is now done in the objective function, so callback focuses on GUI updates
                            current_time = time.time()
                            if (current_time - self.last_update_time) > 0.1:
                                # Plot current state (not just best)
                                self.progress.emit(
                                    self.iteration, 
                                    total_cost,  # Use current cost, not best_cost
                                    xk_phys if hasattr(xk_phys, 'tolist') else np.array(xk_phys),     
                                    cons_vals,   # Use current constraint values
                                    obj_vals,    # Use current objective values
                                    max_violation
                                )
                                self.last_update_time = current_time

                        # Execute optimization
                        options = {'maxiter': self.maxiter}
                        
                        # Handle method-specific options to avoid warnings
                        if self.method == 'SLSQP':
                            if 'tol' in self.solver_options:
                                options['ftol'] = self.solver_options['tol']
                        elif self.method == 'L-BFGS-B':
                            if 'tol' in self.solver_options:
                                options['ftol'] = self.solver_options['tol']
                                options['gtol'] = self.solver_options['tol']
                        elif self.method == 'COBYLA':
                            # Fix for COBYLA warning: Ensure rhobeg is set and valid for normalized [0,1] space
                            # rhobeg is the initial trust region radius, rhoend is the final trust region radius
                            options['rhobeg'] = 0.5
                            # Set rhoend properly to avoid warnings (must be <= rhobeg)
                            if 'tol' in self.solver_options:
                                options['rhoend'] = min(self.solver_options['tol'], 0.5)
                            else:
                                options['rhoend'] = 1e-4  # Default value that's less than rhobeg
                        
                        # Methods that support constraints
                        constrained_methods = ['SLSQP', 'COBYLA', 'trust-constr']
                        
                        # Filter constraints for unsupported methods
                        final_constraints = cons if self.method in constrained_methods else None
                        
                        if cons and self.method not in constrained_methods:
                                logger.warning("Method %s does not support constraints; constraints will be ignored", self.method)

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            warnings.filterwarnings('ignore', message='Unknown solver options: maxiter')
                            
                            # For COBYLA, don't pass tol separately since rhoend is already in options
                            if self.method == 'COBYLA':
                                res = minimize(objective, x0_norm, method=self.method, bounds=norm_bounds,
                                             constraints=final_constraints,
                                             callback=callback, options=options)
                            else:
                                res = minimize(objective, x0_norm, method=self.method, bounds=norm_bounds,
                                             constraints=final_constraints,
                                             tol=self.solver_options.get('tol', None),
                                             callback=callback, options=options)

            except UserStopException:
                self.finished.emit(None, "Stopped by user")
                return
            except Exception as e:
                if "Optimization stopped by user" in str(e):
                    self.finished.emit(None, "Stopped by user")
                    return
                raise e
            
            if not self._is_running:
                self.finished.emit(None, "Stopped by user")
                return

            message = getattr(res, 'message', 'Optimization completed')
            # Convert result back to physical units
            if hasattr(res, 'x'):
                res.x = self.opt_setup._to_physical(res.x)
            self.finished.emit(res, message)

        except UserStopException:
            self.finished.emit(None, "Stopped by user")
        except Exception as e:
            if str(e) == "Optimization stopped by user.":
                self.finished.emit(None, "Stopped by user")
            else:
                self.error.emit(str(e))

    def stop(self):
        """Stop the optimization by setting the running flag to False."""
        self._is_running = False


class AdvancedSettingsDialog(QtWidgets.QDialog):
    """Dialog for advanced optimization settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Optimization Settings")
        self.setModal(True)
        self.settings = {}
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # General Settings
        grp_general = QtWidgets.QGroupBox("General Solver Settings")
        form_general = QtWidgets.QFormLayout(grp_general)
        
        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(1e-12, 1.0)
        self.spin_tol.setValue(1e-6)
        self.spin_tol.setDecimals(12)
        self.spin_tol.setToolTip("Tolerance for termination. Smaller values mean higher precision.")
        form_general.addRow("Tolerance:", self.spin_tol)
        
        self.spin_penalty = QtWidgets.QDoubleSpinBox()
        self.spin_penalty.setRange(1.0, 1e6)
        self.spin_penalty.setValue(1000.0)
        self.spin_penalty.setToolTip("Penalty weight for constraint violations (for non-native constraint solvers).")
        form_general.addRow("Penalty Weight:", self.spin_penalty)
        
        layout.addWidget(grp_general)
        
        # Differential Evolution Settings
        grp_de = QtWidgets.QGroupBox("Differential Evolution")
        form_de = QtWidgets.QFormLayout(grp_de)
        
        self.spin_popsize = QtWidgets.QSpinBox()
        self.spin_popsize.setRange(5, 200)
        self.spin_popsize.setValue(15)
        self.spin_popsize.setToolTip("Population size multiplier (popsize * num_vars).")
        form_de.addRow("Population Size:", self.spin_popsize)
        
        self.spin_mutation_min = QtWidgets.QDoubleSpinBox()
        self.spin_mutation_min.setRange(0.0, 1.9)
        self.spin_mutation_min.setValue(0.5)
        self.spin_mutation_max = QtWidgets.QDoubleSpinBox()
        self.spin_mutation_max.setRange(0.0, 1.9)
        self.spin_mutation_max.setValue(1.0)
        mut_layout = QtWidgets.QHBoxLayout()
        mut_layout.addWidget(self.spin_mutation_min)
        mut_layout.addWidget(QtWidgets.QLabel("-"))
        mut_layout.addWidget(self.spin_mutation_max)
        form_de.addRow("Mutation Range:", mut_layout)
        
        self.spin_recombination = QtWidgets.QDoubleSpinBox()
        self.spin_recombination.setRange(0.0, 1.0)
        self.spin_recombination.setValue(0.7)
        form_de.addRow("Recombination:", self.spin_recombination)
        
        layout.addWidget(grp_de)
        
        # Nevergrad Settings
        grp_ng = QtWidgets.QGroupBox("Nevergrad")
        form_ng = QtWidgets.QFormLayout(grp_ng)
        
        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"])
        self.combo_ng_opt.setToolTip("Specific Nevergrad optimizer variant.")
        form_ng.addRow("Optimizer:", self.combo_ng_opt)
        
        self.spin_ng_workers = QtWidgets.QSpinBox()
        self.spin_ng_workers.setRange(1, 32)
        self.spin_ng_workers.setValue(1)
        form_ng.addRow("Num Workers:", self.spin_ng_workers)
        
        layout.addWidget(grp_ng)
        
        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_settings(self):
        return {
            'tol': self.spin_tol.value(),
            'penalty_weight': self.spin_penalty.value(),
            'popsize': self.spin_popsize.value(),
            'mutation': (self.spin_mutation_min.value(), self.spin_mutation_max.value()),
            'recombination': self.spin_recombination.value(),
            'optimizer_name': self.combo_ng_opt.currentText(),
            'num_workers': self.spin_ng_workers.value()
        }

class OptimizationWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OptimizationWidget, self).__init__(parent)
        self.problem = None
        self.worker = None
        self.system_code = None
        self.models = []
        self.objectives = []
        self.constraints = []
        self.advanced_settings = {
            'tol': 1e-6,
            'penalty_weight': 1000.0,
            'popsize': 15,
            'mutation': (0.5, 1.0),
            'recombination': 0.7,
            'optimizer_name': 'NGOpt',
            'num_workers': 1
        }
        self.init_ui()
        
    def init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # --- Left Panel ---
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(left_panel)
        
        grp_settings = QtWidgets.QGroupBox("Optimization Settings")
        form_layout = QtWidgets.QFormLayout(grp_settings)
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr', 'COBYLA', 'Nelder-Mead', 'Powell', 'Nevergrad', 'Differential Evolution'])
        self.combo_method.setToolTip("Select the optimization algorithm:\n• SLSQP: Sequential Least Squares Programming (recommended for constrained problems)\n• L-BFGS-B: Limited-memory BFGS with bounds\n• TNC: Truncated Newton algorithm\n• trust-constr: Trust-region constrained algorithm\n• COBYLA: Constrained Optimization BY Linear Approximation\n• Nelder-Mead: Simplex method (derivative-free)\n• Powell: Powell's method (derivative-free)\n• Nevergrad: Gradient-free optimization with native constraint support\n• Differential Evolution: Population-based stochastic optimization with native constraint support")
        self.combo_method.currentTextChanged.connect(self.update_method_settings)
        self.btn_algo_info = QtWidgets.QPushButton("?")
        self.btn_algo_info.setFixedWidth(30)
        self.btn_algo_info.clicked.connect(self.show_algorithm_info)
        self.btn_algo_info.setToolTip("Show detailed information about the selected algorithm")
        
        self.btn_wizard = QtWidgets.QPushButton("Wizard")
        self.btn_wizard.setFixedWidth(60)
        self.btn_wizard.clicked.connect(self.show_solver_wizard)
        self.btn_wizard.setToolTip("Launch the Solver Selection Wizard to help choose the best algorithm")
        
        self.btn_advanced = QtWidgets.QPushButton("Advanced")
        self.btn_advanced.setFixedWidth(70)
        self.btn_advanced.clicked.connect(self.show_advanced_settings)
        self.btn_advanced.setToolTip("Configure advanced solver settings")
        
        algo_layout = QtWidgets.QHBoxLayout()
        algo_layout.addWidget(self.combo_method)
        algo_layout.addWidget(self.btn_algo_info)
        algo_layout.addWidget(self.btn_wizard)
        algo_layout.addWidget(self.btn_advanced)
        form_layout.addRow("Algorithm:", algo_layout)
        
        self.chk_constraint_scaling = QtWidgets.QCheckBox("Enable Scaling")
        self.chk_constraint_scaling.setChecked(True)
        self.chk_constraint_scaling.setToolTip("Constraint Scaling\nAutomatically scale variables and constraints to [0,1] range.\n• Improves optimizer convergence and stability\n• Prevents issues with variables of different scales\n• Recommended for most engineering problems\n• Can be disabled for debugging scaling issues")
        form_layout.addRow("", self.chk_constraint_scaling)
        
        # Max Iterations Control
        maxiter_layout = QtWidgets.QHBoxLayout()
        maxiter_layout.addWidget(QtWidgets.QLabel("Max Iterations:"))
        self.spin_maxiter = QtWidgets.QSpinBox()
        self.spin_maxiter.setRange(10, 10000)
        self.spin_maxiter.setValue(500)
        self.spin_maxiter.setToolTip("Max Iterations\nMaximum number of iterations allowed for optimization.\n• Higher values allow more thorough search but take longer\n• Most algorithms converge before reaching this limit\n• Increase for complex, high-dimensional problems\n• Decrease for quick feasibility checks")
        maxiter_layout.addWidget(self.spin_maxiter)
        form_layout.addRow("", maxiter_layout)
        
        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.setToolTip("Select the system model to optimize. Models are created in the Modeling Environment tab.")
        self.system_combo.currentIndexChanged.connect(self.on_system_changed)
        form_layout.addRow("System:", self.system_combo)
        left_layout.addWidget(grp_settings)
        
        # Add objectives table
        grp_objectives = QtWidgets.QGroupBox("Objectives")
        objectives_layout = QtWidgets.QVBoxLayout(grp_objectives)
        self.table_objectives = QtWidgets.QTableWidget()
        self.table_objectives.setColumnCount(3)
        self.table_objectives.setHorizontalHeaderLabels(["Name", "Type", "Weight"])
        self.table_objectives.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_objectives.verticalHeader().setVisible(False)
        self.table_objectives.setMaximumHeight(150)
        self.table_objectives.itemChanged.connect(self.on_objective_weight_changed)
        objectives_layout.addWidget(self.table_objectives)
        left_layout.addWidget(grp_objectives)
        
        grp_exec = QtWidgets.QGroupBox("Execution")
        exec_layout = QtWidgets.QVBoxLayout(grp_exec)
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 5px;")
        self.btn_run.setToolTip("Start the optimization process with the selected algorithm and settings. The optimization will run until convergence or maximum iterations are reached.")
        self.btn_run.clicked.connect(self.start_optimization)
        self.btn_run.setEnabled(False)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop.setToolTip("Stop the currently running optimization process. The best solution found so far will be retained.")
        self.btn_stop.clicked.connect(self.stop_optimization)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        exec_layout.addLayout(btn_layout)
        exec_layout.addWidget(self.progress_bar)
        exec_layout.addWidget(self.lbl_status)
        left_layout.addWidget(grp_exec)

        grp_results = QtWidgets.QGroupBox("Current Results")
        results_layout = QtWidgets.QVBoxLayout(grp_results)
        self.table_results = QtWidgets.QTableWidget()
        self.table_results.setColumnCount(2)
        self.table_results.setHorizontalHeaderLabels(["Variable", "Value"])
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        results_layout.addWidget(self.table_results)
        left_layout.addWidget(grp_results)
        
        self.btn_code = QtWidgets.QPushButton("View Generated Code")
        self.btn_code.clicked.connect(self.view_source_code)
        left_layout.addWidget(self.btn_code)
        
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel)
        
        self.plot_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.plot_tabs)
        
        self.tab_cost = QtWidgets.QWidget()

        self.plot_tabs.addTab(self.tab_cost, "Total Objective Cost")
        l_cost = QtWidgets.QVBoxLayout(self.tab_cost)
        
        # Top bar
        h_cost = QtWidgets.QHBoxLayout()
        h_cost.addStretch()
        btn_save_cost = QtWidgets.QPushButton("Save Plot")
        btn_save_cost.clicked.connect(lambda: self.save_plot(self.plot_cost, "Total Objective Cost"))
        h_cost.addWidget(btn_save_cost)
        l_cost.addLayout(h_cost)
        
        self.plot_cost = pg.PlotWidget()
        self.plot_cost.setBackground('w')  # White background
        self.plot_cost.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cost.setTitle("Total Objective Cost")
        self.plot_cost.setLabel('bottom', "Iteration")
        self.plot_cost.setLabel('left', "Cost")
        # Keep axes black for white background
        l_cost.addWidget(self.plot_cost)
        
        self.tab_dv = QtWidgets.QWidget()
        self.plot_tabs.addTab(self.tab_dv, "Design Variables")
        l_dv = QtWidgets.QVBoxLayout(self.tab_dv)
        h_dv = QtWidgets.QHBoxLayout()
        self.combo_dv = QtWidgets.QComboBox()
        self.combo_dv.addItem("All")
        self.combo_dv.setToolTip("Select which design variable to display in the convergence plot, or 'All' to show all variables.")
        self.combo_dv.currentTextChanged.connect(self.update_dv_plot)
        h_dv.addWidget(QtWidgets.QLabel("Show:"))
        h_dv.addWidget(self.combo_dv)
        h_dv.addStretch()
        
        btn_save_dv = QtWidgets.QPushButton("Save Plot")
        btn_save_dv.clicked.connect(lambda: self.save_plot(self.plot_dv, "Design Variables"))
        h_dv.addWidget(btn_save_dv)
        
        l_dv.addLayout(h_dv)
        self.plot_dv = pg.PlotWidget()
        self.plot_dv.setBackground('w')  # White background
        self.plot_dv.showGrid(x=True, y=True, alpha=0.3)
        self.plot_dv.setTitle("Design Variables")
        # Keep axes black for white background
        l_dv.addWidget(self.plot_dv)

        self.tab_cons = QtWidgets.QWidget()
        self.plot_tabs.addTab(self.tab_cons, "Constraints")
        l_cons = QtWidgets.QVBoxLayout(self.tab_cons)
        h_cons = QtWidgets.QHBoxLayout()
        self.combo_cons = QtWidgets.QComboBox()
        self.combo_cons.addItem("All")
        self.combo_cons.setToolTip("Select which constraint to display in the convergence plot, or 'All' to show all constraints.")
        self.combo_cons.currentTextChanged.connect(self.update_cons_plot)
        h_cons.addWidget(QtWidgets.QLabel("Show:"))
        h_cons.addWidget(self.combo_cons)
        h_cons.addStretch()
        
        btn_save_cons = QtWidgets.QPushButton("Save Plot")
        btn_save_cons.clicked.connect(lambda: self.save_plot(self.plot_cons, "Constraints"))
        h_cons.addWidget(btn_save_cons)
        
        l_cons.addLayout(h_cons)
        self.plot_cons = pg.PlotWidget()
        self.plot_cons.setBackground('w')  # White background
        self.plot_cons.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cons.setTitle("Constraints")
        # Keep axes black for white background
        l_cons.addWidget(self.plot_cons)

        self.tab_violation = QtWidgets.QWidget()
        self.plot_tabs.addTab(self.tab_violation, "Max Violation")
        l_violation = QtWidgets.QVBoxLayout(self.tab_violation)
        
        h_violation = QtWidgets.QHBoxLayout()
        h_violation.addStretch()
        btn_save_violation = QtWidgets.QPushButton("Save Plot")
        btn_save_violation.clicked.connect(lambda: self.save_plot(self.plot_violation, "Maximum Constraint Violation"))
        h_violation.addWidget(btn_save_violation)
        l_violation.addLayout(h_violation)
        
        self.plot_violation = pg.PlotWidget()
        self.plot_violation.setBackground('w')  # White background
        self.plot_violation.showGrid(x=True, y=True, alpha=0.3)
        self.plot_violation.setTitle("Maximum Constraint Violation")
        # Keep axes black for white background
        l_violation.addWidget(self.plot_violation)

        self.tab_objs = QtWidgets.QWidget()
        self.plot_tabs.addTab(self.tab_objs, "Individual Objectives")
        l_objs = QtWidgets.QVBoxLayout(self.tab_objs)
        h_objs = QtWidgets.QHBoxLayout()
        self.combo_objs = QtWidgets.QComboBox()
        self.combo_objs.addItem("All")
        self.combo_objs.setToolTip("Select which objective function to display in the convergence plot, or 'All' to show all objectives.")
        self.combo_objs.currentTextChanged.connect(self.update_objs_plot)
        h_objs.addWidget(QtWidgets.QLabel("Show:"))
        h_objs.addWidget(self.combo_objs)
        h_objs.addStretch()
        
        btn_save_objs = QtWidgets.QPushButton("Save Plot")
        btn_save_objs.clicked.connect(lambda: self.save_plot(self.plot_objs, "Individual Objectives"))
        h_objs.addWidget(btn_save_objs)
        
        l_objs.addLayout(h_objs)
        self.plot_objs = pg.PlotWidget()
        self.plot_objs.setBackground('w')  # White background
        self.plot_objs.showGrid(x=True, y=True, alpha=0.3)
        self.plot_objs.setTitle("Individual Objectives")
        # Keep axes black for white background
        l_objs.addWidget(self.plot_objs)

        self.iter_data = []
        self.cost_data = []
        self.violation_data = []
        self.dv_data = []
        self.cons_data = {}
        self.objs_data = {}
        
        # Plot items for efficient updating (avoid clearing/redrawing on every progress update)
        self.cost_plot_item = None
        self.violation_plot_item = None
        self.dv_plot_items = []
        self.dv_bound_items = []
        self.cons_plot_items = []
        self.cons_bound_items = []
        self.objs_plot_items = []
    
    def save_plot(self, plot_widget, title):
        """Save the current plot to a file."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot", f"{title}.png", "Images (*.png *.jpg *.bmp)"
        )
        if filename:
            exporter = pg.exporters.ImageExporter(plot_widget.plotItem)
            exporter.export(filename)

    def update_method_settings(self):
        method = self.combo_method.currentText()
        
        # Check for warnings
        warning_text = ""
        
        # Warn about multi-objective requirements
        # No special requirements for Nevergrad or Differential Evolution
        
        # Warn about unconstrained methods with constraints
        if method in ['Nelder-Mead', 'Powell'] and hasattr(self, 'constraints') and self.constraints:
            if warning_text:
                warning_text += " | Uses penalty method for constraints"
            else:
                warning_text = f"⚠️ Warning: {method} uses penalty method for constraints"
        
        # Warn about soft constraints
        if hasattr(self, 'chk_soft_constraints') and self.chk_soft_constraints.isChecked():
            if warning_text:
                warning_text += " | Soft constraints enabled"
            else:
                warning_text = "⚠️ Warning: Soft constraints enabled (penalties added to objective)"
        
        if warning_text:
            self.lbl_status.setText(warning_text)
            self.lbl_status.setStyleSheet("color: orange;")
        else:
            self.lbl_status.setText("Ready")
            self.lbl_status.setStyleSheet("")
        
    def show_algorithm_info(self):
        method = self.combo_method.currentText()
        info = {
            'SLSQP': "<b>SLSQP</b><br><br>Gradient-based. Handles constraints. Best for smooth functions.",
            'L-BFGS-B': "<b>L-BFGS-B</b><br><br>Gradient-based. Handles bounds. Good for large problems.",
            'TNC': "<b>TNC</b><br><br>Gradient-based. Handles bounds. Newton conjugate-gradient method.",
            'trust-constr': "<b>Trust-Region Constrained</b><br><br>Gradient-based. Advanced constraint handling. Best for difficult constrained problems.",
            'COBYLA': "<b>COBYLA</b><br><br>Gradient-free. Handles inequality constraints. Good for non-smooth functions.",
            'Nelder-Mead': "<b>Nelder-Mead</b><br><br>Gradient-free heuristic. Uses penalty method for constraints (may be less reliable).",
            'Powell': "<b>Powell</b><br><br>Gradient-free conjugate direction. Uses penalty method for constraints (may be less reliable).",
            'Nevergrad': "<b>Nevergrad</b><br><br>Gradient-free optimization framework with native constraint support using cheap constraints. Robust for blackbox optimization problems.",
            'Differential Evolution': "<b>Differential Evolution</b><br><br>Population-based stochastic optimization with native constraint support. Good for global optimization of constrained, non-smooth functions."
        }
        QtWidgets.QMessageBox.information(self, f"Info: {method}", info.get(method, "No info."))

    def _parse_float(self, val):
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("inf", "+inf"): return float('inf')
            if v == "-inf": return float('-inf')
            try: return float(val)
            except: pass
        raise ValueError(f"Invalid float: {val}")

    def _execute_code_safely(self, code):
        """Execute code by writing to a temporary file and importing as module.
        This provides better debugging with proper tracebacks."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("temp_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the system_function
            if hasattr(module, 'system_function'):
                system_function = module.system_function
                if callable(system_function):
                    return system_function
                else:
                    raise TypeError(f"system_function is not callable: {type(system_function)}")
            else:
                raise AttributeError("system_function not found in generated code")
        finally:
            # Clean up the temporary file
            import os
            try:
                os.unlink(temp_file)
            except:
                pass  # Ignore cleanup errors

    def load_model(self, code, inputs, outputs):
        self.system_code = code
        try:
            # Use safe execution instead of exec()
            system_function = self._execute_code_safely(code)
            self.problem = XRayProblem("Optimization_Model", sample_size=3000)
            self.problem.set_system_model(system_function)
            self.problem.set_system_code(code)

            for inp in inputs:
                self.problem.add_design_variable(inp['name'], inp.get('unit', '-'), self._parse_float(inp['min']), self._parse_float(inp['max']))
            for out in outputs:
                self.problem.add_quantity_of_interest(out['name'], out.get('unit', '-'), self._parse_float(out['req_min']), self._parse_float(out['req_max']), minimize=out.get('minimize', False), maximize=out.get('maximize', False))
            self.set_problem(self.problem)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def load_model_from_system_model(self, system_model):
        """
        Load a model from a SystemModel instance.
        """
        try:
            self.system_code = system_model.source_code
            self.problem = XRayProblem("Optimization_Model", sample_size=3000)
            if not callable(system_model.system_function):
                raise TypeError(f"system_function is not callable: {type(system_model.system_function)}")
            self.problem.set_system_model(system_model.system_function)
            self.problem.set_system_code(system_model.source_code)

            for inp in system_model.inputs:
                self.problem.add_design_variable(inp['name'], inp.get('unit', '-'), self._parse_float(inp['min']), self._parse_float(inp['max']))
            for out in system_model.outputs:
                self.problem.add_quantity_of_interest(out['name'], out.get('unit', '-'), self._parse_float(out['req_min']), self._parse_float(out['req_max']), minimize=out.get('minimize', False), maximize=out.get('maximize', False))
            self.set_problem(self.problem)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def load_models(self, models):
        self.models = models
        self.system_combo.clear()
        for m in models:
            # Handle both SystemModel instances and legacy dicts
            name = m.name if hasattr(m, 'name') else m['name']
            self.system_combo.addItem(name)
        if models:
            self.system_combo.setCurrentIndex(0)
            self.load_selected_system()

    def on_system_changed(self):
        self.load_selected_system()

    def load_selected_system(self):
        idx = self.system_combo.currentIndex()
        if idx >= 0 and idx < len(self.models):
            m = self.models[idx]
            # Handle both SystemModel instances and legacy dicts
            if hasattr(m, 'name'):  # SystemModel instance
                self.load_model_from_system_model(m)
            else:  # Legacy dict format
                self.load_model(m['code'], m['inputs'], m['outputs'])

    def set_problem(self, problem):
        self.problem = problem
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Problem Loaded: {problem.name}")
        self.objectives = []
        self.constraints = []
        for qoi in problem.quantities_of_interest:
            if qoi.get('minimize', False) or qoi.get('maximize', False): self.objectives.append(qoi)
            else: self.constraints.append(qoi)
        
        # Populate objectives table
        self.populate_objectives_table()
        
        # Populate plot combo boxes
        self.populate_plot_combos()
        
        if self.objectives:
            names = [o['name'] for o in self.objectives]
            self.lbl_status.setText(f"Ready. Objectives: {', '.join(names)}")
        else:
            self.lbl_status.setText("Warning: No objective selected.")
            self.btn_run.setEnabled(False)

    def populate_objectives_table(self):
        """Populate the objectives table with current objectives and their weights."""
        self.table_objectives.blockSignals(True)
        self.table_objectives.setRowCount(len(self.objectives))
        
        for row, obj in enumerate(self.objectives):
            # Name column
            name_item = QtWidgets.QTableWidgetItem(obj['name'])
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table_objectives.setItem(row, 0, name_item)
            
            # Type column
            obj_type = "Minimize" if obj.get('minimize', False) else "Maximize"
            type_item = QtWidgets.QTableWidgetItem(obj_type)
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table_objectives.setItem(row, 1, type_item)
            
            # Weight column
            weight = obj.get('weight', 1.0)
            weight_item = QtWidgets.QTableWidgetItem(str(weight))
            self.table_objectives.setItem(row, 2, weight_item)
        
        self.table_objectives.blockSignals(False)

    def on_objective_weight_changed(self, item):
        """Handle weight changes in the objectives table."""
        if item.column() != 2:  # Only handle weight column changes
            return
        
        row = item.row()
        if row >= len(self.objectives):
            return
        
        try:
            new_weight = float(item.text())
            if new_weight <= 0:
                QtWidgets.QMessageBox.warning(self, "Invalid Weight", "Weight must be a positive number.")
                # Reset to previous value
                old_weight = self.objectives[row].get('weight', 1.0)
                item.setText(str(old_weight))
                return
            
            # Update the objective's weight
            self.objectives[row]['weight'] = new_weight
            
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Weight", "Weight must be a valid number.")
            # Reset to previous value
            old_weight = self.objectives[row].get('weight', 1.0)
            item.setText(str(old_weight))

    def populate_plot_combos(self):
        """Populate plot combo boxes with current problem data."""
        # Save current selections
        curr_dv = self.combo_dv.currentText()
        curr_cons = self.combo_cons.currentText()
        curr_objs = self.combo_objs.currentText()

        # Design Variables
        self.combo_dv.blockSignals(True)
        self.combo_dv.clear()
        self.combo_dv.addItem("All")
        if self.problem:
            for dv in self.problem.design_variables: 
                self.combo_dv.addItem(dv['name'])
        if self.combo_dv.findText(curr_dv) >= 0:
            self.combo_dv.setCurrentText(curr_dv)
        self.combo_dv.blockSignals(False)

        # Constraints
        self.combo_cons.blockSignals(True)
        self.combo_cons.clear()
        self.combo_cons.addItem("All")
        for c in self.constraints: 
            self.combo_cons.addItem(c['name'])
        if self.combo_cons.findText(curr_cons) >= 0:
            self.combo_cons.setCurrentText(curr_cons)
        self.combo_cons.blockSignals(False)

        # Objectives
        self.combo_objs.blockSignals(True)
        self.combo_objs.clear()
        self.combo_objs.addItem("All")
        for o in self.objectives: 
            self.combo_objs.addItem(o['name'])
        if self.combo_objs.findText(curr_objs) >= 0:
            self.combo_objs.setCurrentText(curr_objs)
        self.combo_objs.blockSignals(False)

    def show_solver_wizard(self):
        """Launch the Solver Selection Wizard to help choose the best algorithm."""
        wizard = SolverSelectionWizard(self)
        if wizard.exec_():
            selected_solver = wizard.selected_solver
            if selected_solver:
                # Find the solver in the combo box and select it
                idx = self.combo_method.findText(selected_solver)
                if idx >= 0:
                    self.combo_method.setCurrentIndex(idx)
                    self.lbl_status.setText(f"Solver selected: {selected_solver}")
                else:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Solver Not Available",
                        f"The recommended solver '{selected_solver}' is not available in the current version."
                    )

    def show_advanced_settings(self):
        """Show the advanced settings dialog."""
        dialog = AdvancedSettingsDialog(self)
        
        # Load current settings
        dialog.spin_tol.setValue(self.advanced_settings.get('tol', 1e-6))
        dialog.spin_penalty.setValue(self.advanced_settings.get('penalty_weight', 1000.0))
        dialog.spin_popsize.setValue(self.advanced_settings.get('popsize', 15))
        
        mut = self.advanced_settings.get('mutation', (0.5, 1.0))
        dialog.spin_mutation_min.setValue(mut[0])
        dialog.spin_mutation_max.setValue(mut[1])
        
        dialog.spin_recombination.setValue(self.advanced_settings.get('recombination', 0.7))
        
        opt_name = self.advanced_settings.get('optimizer_name', 'NGOpt')
        idx = dialog.combo_ng_opt.findText(opt_name)
        if idx >= 0: dialog.combo_ng_opt.setCurrentIndex(idx)
        
        dialog.spin_ng_workers.setValue(self.advanced_settings.get('num_workers', 1))
        
        if dialog.exec_():
            self.advanced_settings = dialog.get_settings()
            self.lbl_status.setText("Advanced settings updated.")

    def start_optimization(self):
        if not self.problem or not self.objectives: return
        
        # 1. capture current settings
        method = self.combo_method.currentText()
        maxiter = self.spin_maxiter.value()
        
        # No special validation needed for Nevergrad or Differential Evolution
        
        # Check if system model is callable
        if not callable(self.problem.system_model):
            QtWidgets.QMessageBox.critical(self, "Error", f"System model is not callable: {type(self.problem.system_model)}")
            return
        
        self.iter_data = []
        self.cost_data = []
        self.dv_data = []
        self.cons_data = {}
        self.objs_data = {}
        self.violation_data = []
        
        # Clear plot items
        if self.cost_plot_item:
            self.plot_cost.removeItem(self.cost_plot_item)
            self.cost_plot_item = None
        if self.violation_plot_item:
            self.plot_violation.removeItem(self.violation_plot_item)
            self.violation_plot_item = None
        for item in self.dv_plot_items:
            self.plot_dv.removeItem(item)
        for item in self.dv_bound_items:
            self.plot_dv.removeItem(item)
        for item in self.cons_plot_items:
            self.plot_cons.removeItem(item)
        for item in self.cons_bound_items:
            self.plot_cons.removeItem(item)
        for item in self.objs_plot_items:
            self.plot_objs.removeItem(item)
        self.dv_plot_items.clear()
        self.dv_bound_items.clear()
        self.cons_plot_items.clear()
        self.cons_bound_items.clear()
        self.objs_plot_items.clear()
        
        self.plot_cost.clear()
        self.plot_dv.clear()
        self.plot_cons.clear()
        self.plot_violation.clear()
        self.plot_objs.clear()
        
        names = [format_html(o['name']) for o in self.objectives]
        self.plot_cost.setTitle(f"Cost ({' + '.join(names)})" if len(names) > 1 else f"Cost ({names[0]})")
        
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Running...")
        self.progress_bar.setRange(0, 0)
        
        method = self.combo_method.currentText()
        # Ensure method is a valid string
        valid_methods = ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr', 'COBYLA', 'Nelder-Mead', 'Powell', 'Nevergrad', 'Differential Evolution']
        if not isinstance(method, str) or method not in valid_methods:
            method = 'SLSQP'
        maxiter = self.spin_maxiter.value()
        
        # Calculate x0 with check for infinite bounds
        x0_list = []
        for dv in self.problem.design_variables:
            mn = float(dv['min'])
            mx = float(dv['max'])
            if np.isfinite(mn) and np.isfinite(mx):
                x0_list.append((mn + mx) / 2.0)
            elif np.isfinite(mn):
                x0_list.append(mn + 1.0)
            elif np.isfinite(mx):
                x0_list.append(mx - 1.0)
            else:
                x0_list.append(0.0)
        x0 = np.array(x0_list)
        
        self.worker = OptimizationWorker(
            self.problem, self.objectives, self.constraints, x0, 
            method=method, maxiter=maxiter, 
            constraint_scaling=self.chk_constraint_scaling.isChecked(),
            **self.advanced_settings
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        
    def on_error(self, error_msg):
        """Handle optimization errors from the worker thread."""
        QtWidgets.QMessageBox.critical(self, "Optimization Error", f"An error occurred during optimization:\n\n{error_msg}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Error occurred")
        
    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.lbl_status.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def update_dv_plot(self):
        selected = self.combo_dv.currentText()
        if self.dv_data:
            dv_array = np.array(self.dv_data)
            
            plot_count = 0
            for i, dv in enumerate(self.problem.design_variables):
                if selected == "All" or selected == dv['name']: plot_count += 1
            
            # Update existing plot items or create new ones as needed
            current_item_index = 0
            line_index = 0
            
            for i, dv in enumerate(self.problem.design_variables):
                if selected == "All" or selected == dv['name']:
                    color = get_plot_color(line_index, plot_count)
                    name = format_html(dv['name'])
                    
                    if current_item_index < len(self.dv_plot_items):
                        # Check if name matches
                        item = self.dv_plot_items[current_item_index]
                        if item.name() != name:
                            # Name mismatch, recreate item
                            self.plot_dv.removeItem(item)
                            plot_item = self.plot_dv.plot(self.iter_data, dv_array[:, i], pen=pg.mkPen(color, width=2), name=name)
                            self.dv_plot_items[current_item_index] = plot_item
                        else:
                            # Update existing plot item
                            item.setData(self.iter_data, dv_array[:, i])
                            item.setPen(pg.mkPen(color, width=2))
                    else:
                        # Create new plot item
                        plot_item = self.plot_dv.plot(self.iter_data, dv_array[:, i], pen=pg.mkPen(color, width=2), name=name)
                        self.dv_plot_items.append(plot_item)
                    
                    current_item_index += 1
                    line_index += 1
            
            # Remove excess plot items
            while len(self.dv_plot_items) > current_item_index:
                item = self.dv_plot_items.pop()
                self.plot_dv.removeItem(item)
            
            # Update bounds
            for item in self.dv_bound_items:
                self.plot_dv.removeItem(item)
            self.dv_bound_items.clear()
            
            for i, dv in enumerate(self.problem.design_variables):
                if selected == "All" or selected == dv['name']:
                    mn = float(dv.get('min', -1e9))
                    mx = float(dv.get('max', 1e9))
                    if mn > -1e8:
                        bound_item = pg.InfiniteLine(pos=mn, angle=0, pen=pg.mkPen('k', style=QtCore.Qt.DashLine, alpha=0.3))
                        self.plot_dv.addItem(bound_item)
                        self.dv_bound_items.append(bound_item)
                    if mx < 1e8:
                        bound_item = pg.InfiniteLine(pos=mx, angle=0, pen=pg.mkPen('k', style=QtCore.Qt.DashLine, alpha=0.3))
                        self.plot_dv.addItem(bound_item)
                        self.dv_bound_items.append(bound_item)
        
        if self.dv_data: 
            self.plot_dv.addLegend()

    def update_cons_plot(self):
        selected = self.combo_cons.currentText()
        
        # Clear existing plot items
        for item in self.cons_plot_items:
            self.plot_cons.removeItem(item)
        for item in self.cons_bound_items:
            self.plot_cons.removeItem(item)
        self.cons_plot_items.clear()
        self.cons_bound_items.clear()
        
        plot_count = 0
        for name in self.cons_data:
            if selected == "All" or selected == name: plot_count += 1
        line_index = 0
        for name, vals in self.cons_data.items():
            if selected == "All" or selected == name:
                color = get_plot_color(line_index, plot_count)
                plot_item = self.plot_cons.plot(self.iter_data, vals, pen=pg.mkPen(color, width=2), name=format_html(name))
                self.cons_plot_items.append(plot_item)
                for c in self.constraints:
                    if c['name'] == name:
                        # Safe check for keys using the helper or manual get
                        c_min = float(c.get('min', c.get('req_min', -1e9)))
                        c_max = float(c.get('max', c.get('req_max', 1e9)))
                        if c_min > -1e8: 
                            bound_item = pg.InfiniteLine(pos=c_min, angle=0, pen=pg.mkPen('k', style=QtCore.Qt.DashLine, alpha=0.3))
                            self.plot_cons.addItem(bound_item)
                            self.cons_bound_items.append(bound_item)
                        if c_max < 1e8: 
                            bound_item = pg.InfiniteLine(pos=c_max, angle=0, pen=pg.mkPen('k', style=QtCore.Qt.DashLine, alpha=0.3))
                            self.plot_cons.addItem(bound_item)
                            self.cons_bound_items.append(bound_item)
                line_index += 1
        if self.cons_data: self.plot_cons.addLegend()

    def update_objs_plot(self):
        selected = self.combo_objs.currentText()
        plot_count = 0
        for name in self.objs_data:
            if selected == "All" or selected == name: plot_count += 1
            
        line_index = 0
        current_item_index = 0
        
        for name, vals in self.objs_data.items():
            if selected == "All" or selected == name:
                color = get_plot_color(line_index, plot_count)
                display_name = format_html(name)
                
                if current_item_index < len(self.objs_plot_items):
                    item = self.objs_plot_items[current_item_index]
                    if item.name() != display_name:
                        self.plot_objs.removeItem(item)
                        plot_item = self.plot_objs.plot(self.iter_data, vals, pen=pg.mkPen(color, width=2), name=display_name)
                        self.objs_plot_items[current_item_index] = plot_item
                    else:
                        item.setData(self.iter_data, vals)
                        item.setPen(pg.mkPen(color, width=2))
                else:
                    plot_item = self.plot_objs.plot(self.iter_data, vals, pen=pg.mkPen(color, width=2), name=display_name)
                    self.objs_plot_items.append(plot_item)
                
                current_item_index += 1
                line_index += 1
        
        while len(self.objs_plot_items) > current_item_index:
            item = self.objs_plot_items.pop()
            self.plot_objs.removeItem(item)
            
        if self.objs_data: self.plot_objs.addLegend()

    def on_progress(self, iteration, cost, x_vals, cons_vals, obj_vals, max_violation):
        self.iter_data.append(iteration)
        self.cost_data.append(cost)
        self.violation_data.append(max_violation)
        self.dv_data.append(x_vals)
        for k, v in cons_vals.items():
            if k not in self.cons_data: self.cons_data[k] = []
            self.cons_data[k].append(v)
        for k, v in obj_vals.items():
            if k not in self.objs_data: self.objs_data[k] = []
            self.objs_data[k].append(v)
        
        # Update cost plot efficiently (don't clear/redraw on every update)
        if self.cost_plot_item is None:
            self.cost_plot_item = self.plot_cost.plot(self.iter_data, self.cost_data, pen=pg.mkPen('b', width=2), symbol='o', symbolSize=6, symbolBrush='b')
        else:
            self.cost_plot_item.setData(self.iter_data, self.cost_data)
        
        names = [format_html(o['name']) for o in self.objectives]
        self.plot_cost.setTitle(f"Cost ({' + '.join(names)})" if len(names) > 1 else f"Cost ({names[0]})")
        
        # Update violation plot efficiently
        if self.violation_plot_item is None:
            self.violation_plot_item = self.plot_violation.plot(self.iter_data, self.violation_data, pen=pg.mkPen('r', width=2), symbol='o', symbolSize=6, symbolBrush='r')
        else:
            self.violation_plot_item.setData(self.iter_data, self.violation_data)
        
        # Update other plots less frequently to reduce overhead (every 100 iterations)
        if len(self.iter_data) % 100 == 0 or len(self.iter_data) == 1:
            self.update_dv_plot()
            self.update_cons_plot()
            self.update_objs_plot()
        
        self.table_results.setRowCount(len(x_vals) + 1 + len(cons_vals) + len(obj_vals))
        row = 0
        for i, val in enumerate(x_vals):
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(self.problem.design_variables[i]['name']))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.4f}"))
            row += 1
        self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem("Total Cost"))
        self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{cost:.4f}"))
        row += 1
        for name, val in obj_vals.items():
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(f"Obj: {name}"))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.4f}"))
            row += 1
        for name, val in cons_vals.items():
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(f"Con: {name}"))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.4f}"))
            row += 1
    
    def on_finished(self, res, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # FIX: Initialize variables and prioritize worker's best solution over solver's final result
        x_vals = None
        cost = float('inf')
        
        # Use worker's best solution as the baseline
        if self.worker and hasattr(self.worker, 'best_x') and self.worker.best_x is not None:
            x_vals = np.array(self.worker.best_x)
            cost = self.worker.best_cost
        
        if res:
            self.lbl_status.setText(f"Finished: {msg}")
            # Only use res.x if it actually improved over the best tracked solution
            if hasattr(res, 'x') and hasattr(res, 'fun'):
                if res.fun < cost:
                    # Update worker's best data with solver's final result if better
                    self.worker.best_x = res.x.tolist() if hasattr(res.x, 'tolist') else list(res.x)
                    self.worker.best_cost = res.fun
                    
                    # Re-evaluate to get consistent objective/constraint values
                    x_vals = res.x
                    inputs = {dv['name']: x_vals[i] for i, dv in enumerate(self.problem.design_variables)}
                    for p in self.problem.parameters: inputs[p['name']] = p['value']
                    try:
                        if not callable(self.worker.system_model):
                            raise TypeError(f"system_model is not callable: {type(self.worker.system_model)}")
                        res_sys = self.worker.system_model(**inputs)
                        
                        # Update objective values
                        for obj in self.objectives:
                            val = float(res_sys.get(obj['name'], 0))
                            self.worker.best_obj_vals[obj['name']] = val
                        
                        # Update constraint values
                        for constr in self.constraints:
                            val = float(res_sys.get(constr['name'], 0))
                            self.worker.best_cons_vals[constr['name']] = val
                        
                        # Recalculate max violation
                        max_violation = 0.0
                        for constr in self.constraints:
                            val = self.worker.best_cons_vals[constr['name']]
                            c_min, c_max = self.opt_setup._get_constr_bounds(constr)
                            if c_min > -1e8: max_violation = max(max_violation, max(0, c_min - val))
                            if c_max < 1e8: max_violation = max(max_violation, max(0, val - c_max))
                        self.worker.best_max_violation = max_violation
                        
                    except: pass
        else:
            self.lbl_status.setText(f"Finished: {msg}")
        
        # FIX: Ensure plot and table show consistent optimal solution
        # Trigger one final progress update with the best data so plot and table align
        if self.worker and self.worker.best_x is not None:
            self.on_progress(
                getattr(self.worker, 'iteration', 0), 
                self.worker.best_cost, 
                np.array(self.worker.best_x), 
                self.worker.best_cons_vals, 
                self.worker.best_obj_vals, 
                self.worker.best_max_violation
            )
            
        # Force update all plots to ensure final state is shown
        self.update_dv_plot()
        self.update_cons_plot()
        self.update_objs_plot()
        
        self.lbl_status.setText(f"Finished: {msg}. Optimal solution displayed.")
    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def view_source_code(self):
        if not self.system_code:
            QtWidgets.QMessageBox.warning(self, "Warning", "No system loaded.")
            return
        try:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Compiled Function Source")
            dialog.resize(800, 600)
            layout = QtWidgets.QVBoxLayout(dialog)
            text_edit = QtWidgets.QTextEdit()
            text_edit.setPlainText(self.system_code)
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Consolas", 11))
            layout.addWidget(text_edit)
            dialog.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not display source code: {e}")

    def generate_optimization_code(self):
        """Generate standalone optimization code that matches the actual execution logic."""
        if not self.problem or not self.objectives:
            return "# No problem loaded"
        
        # Create optimization setup to ensure consistency
        opt_setup = OptimizationSetup(self.problem, self.objectives, self.constraints, self.chk_constraint_scaling.isChecked())
        
        code = []
        
        # Collect all imports at the top
        imports = []
        imports.append("import numpy as np")
        
        # Add other imports that might be in system_code
        system_code = self.system_code.replace("system_function", "model_system_function")
        
        # Extract imports from system_code and add them to the top
        lines = system_code.split('\n')
        system_lines = []
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            else:
                system_lines.append(line)
        
        # Reconstruct system_code without the moved imports
        system_code = '\n'.join(system_lines)
        
        # Add optimization-specific imports
        imports.extend([
            "from scipy.optimize import minimize",
            "import nevergrad as ng",
            "from scipy.optimize import differential_evolution"
        ])
        
        # Remove duplicates
        imports = list(dict.fromkeys(imports))
        
        # Add imports to code
        code.extend(imports)
        code.append("")
        
        # Add system code
        code.append(system_code)    
        
        return "\n".join(code)

    def save_to_folder(self, folder_path):
        """Save optimization settings to a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'optimization.json')
        
        data = {
            'method': self.combo_method.currentText(),
            'max_iter': self.spin_maxiter.value(),
            'scaling': self.chk_constraint_scaling.isChecked(),
            'system_index': self.system_combo.currentIndex(),
            'objectives': self.objectives,
            'constraints': self.constraints,
            'plot_settings': {
                'dv_selection': self.combo_dv.currentText(),
                'cons_selection': self.combo_cons.currentText(),
                'objs_selection': self.combo_objs.currentText()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_folder(self, folder_path):
        """Load optimization settings from a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'optimization.json')
        if not os.path.exists(json_path):
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            self.combo_method.setCurrentText(data.get('method', 'SLSQP'))
            self.spin_maxiter.setValue(data.get('max_iter', 500))
            self.chk_constraint_scaling.setChecked(data.get('scaling', True))
            
            # Restore system selection if possible
            sys_idx = data.get('system_index', 0)
            if sys_idx < self.system_combo.count():
                self.system_combo.setCurrentIndex(sys_idx)
                
            # Restore objectives and constraints
            # Note: These lists are usually rebuilt from the problem definition
            # but we might have custom weights/settings
            saved_objs = data.get('objectives', [])
            saved_cons = data.get('constraints', [])
            
            # We need to be careful here. The problem definition might have changed.
            # We should try to match by name.
            
            # Update current objectives with saved values
            for saved_obj in saved_objs:
                for curr_obj in self.objectives:
                    if curr_obj['name'] == saved_obj['name']:
                        curr_obj.update(saved_obj)
                        break
            
            # Update current constraints with saved values
            for saved_con in saved_cons:
                for curr_con in self.constraints:
                    if curr_con['name'] == saved_con['name']:
                        curr_con.update(saved_con)
                        break
            
            # Refresh tables
            self.populate_objectives_table()
            # self.update_constraints_table() # No constraint table currently
            
            # Restore plot settings
            plot_settings = data.get('plot_settings', {})
            if 'dv_selection' in plot_settings:
                self.combo_dv.setCurrentText(plot_settings['dv_selection'])
            if 'cons_selection' in plot_settings:
                self.combo_cons.setCurrentText(plot_settings['cons_selection'])
            if 'objs_selection' in plot_settings:
                self.combo_objs.setCurrentText(plot_settings['objs_selection'])
            
        except Exception as e:
            logger.exception("Failed to load optimization settings")







