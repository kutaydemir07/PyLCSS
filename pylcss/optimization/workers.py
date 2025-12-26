from PySide6 import QtCore
from .core import Variable, Objective, Constraint, OptimizationResult
from .evaluator import ModelEvaluator
from .solvers.factory import get_solver
import time

class OptimizationWorker(QtCore.QThread):
    # Consolidate signals to reduce overhead
    progress = QtCore.Signal(dict) # Emit a dictionary of current state
    finished = QtCore.Signal(object) # Emit OptimizationResult
    error = QtCore.Signal(str)

    def __init__(self, model_func, setup_data: dict, solver_settings: dict):
        super().__init__()
        self.model_func = model_func
        self.setup = setup_data
        self.settings = solver_settings
        self.solver = None

    def run(self):
        try:
            # Helper to map dictionary keys to Dataclass fields
            def map_variable(v):
                return Variable(
                    name=v['name'],
                    min_val=float(v.get('min_val', v.get('min', 0.0))),
                    max_val=float(v.get('max_val', v.get('max', 1.0))),
                    value=float(v.get('value', 0.0))
                )

            def map_objective(o):
                return Objective(
                    name=o['name'],
                    weight=float(o.get('weight', 1.0)),
                    minimize=bool(o.get('minimize', True))
                )

            def map_constraint(c):
                # Handle req_min/req_max legacy keys
                min_v = c.get('min_val', c.get('min', c.get('req_min', float('-inf'))))
                max_v = c.get('max_val', c.get('max', c.get('req_max', float('inf'))))
                return Constraint(
                    name=c['name'],
                    min_val=float(min_v) if min_v is not None else float('-inf'),
                    max_val=float(max_v) if max_v is not None else float('inf')
                )

            # 1. Setup Evaluator
            # Extract parameters if available
            parameters = self.setup.get('parameters', {})
            
            evaluator = ModelEvaluator(
                self.model_func,
                [map_variable(v) for v in self.setup['variables']],
                [map_objective(o) for o in self.setup['objectives']],
                [map_constraint(c) for c in self.setup['constraints']],
                parameters=parameters,
                scaling=self.settings.get('scaling', True),
                penalty_weight=self.settings.get('penalty_weight', 1e6),
                objective_scale=self.settings.get('objective_scale', 1.0) # <--- Pass it here
            )

            # 2. Get Solver Strategy
            self.solver = get_solver(self.settings['method'], self.settings)

            # 3. Define Throttled Callback (Limit to 20Hz)
            last_emit_time = 0.0
            eval_count = 0  # <--- NEW: Track actual evaluations
            
            def on_step(x, cost, raw_res, violation):
                nonlocal last_emit_time, eval_count
                eval_count += 1  # <--- NEW: Increment count
                current_time = time.time()
                # Limit updates to ~20Hz (50ms) to prevent GUI freezing
                if current_time - last_emit_time >= 0.05:
                    self.progress.emit({
                        'iteration': eval_count,  # <--- NEW: Send real count
                        'x': evaluator.to_physical(x),
                        'cost': cost,
                        'raw': raw_res,
                        'violation': violation
                    })
                    last_emit_time = current_time

            # 4. Execute
            x0 = self.setup['x0']
                
            result = self.solver.solve(evaluator, x0, on_step)
            
            # Emit final progress to ensure UI is up to date with the final result
            # Note: result.x is already physical, so we don't use evaluator.to_physical
            raw_combined = {**result.objectives, **result.constraints}
            self.progress.emit({
                'iteration': eval_count,  # <--- NEW: Send final count
                'x': result.x,
                'cost': result.cost,
                'raw': raw_combined,
                'violation': result.max_violation
            })
            
            self.finished.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        if self.solver:
            self.solver.stop()
