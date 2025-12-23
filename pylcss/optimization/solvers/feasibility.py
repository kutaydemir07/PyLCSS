# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np

class FeasibilityProblem:
    """
    Helper class to define the feasibility optimization problem for Solution Space exploration.
    Encapsulates the logic for objective function (feasibility + optional objectives) and 
    constraint definitions.
    """
    def __init__(self, problem, parameters, ind_parameters, reqL, reqU, dv_norm, dv_norm_l, include_objectives=False, objective_indices=None, objective_weights=None):
        self.problem = problem
        self.parameters = parameters
        self.ind_parameters = ind_parameters
        self.reqL = reqL
        self.reqU = reqU
        self.dv_norm = dv_norm
        self.dv_norm_l = dv_norm_l
        self.include_objectives = include_objectives
        self.objective_indices = objective_indices if objective_indices is not None else []
        self.objective_weights = objective_weights if objective_weights is not None else []

    def denormalize(self, x_norm):
        """Convert normalized design variables [0, 1] to physical space."""
        return x_norm * self.dv_norm + self.dv_norm_l

    def construct_full_vector(self, x_phys):
        """Construct the full input vector including fixed parameters."""
        total_vars = self.parameters.shape[1]
        x_full = np.zeros(total_vars)
        ind_dvs = np.setdiff1d(np.arange(total_vars), self.ind_parameters)
        x_full[ind_dvs] = x_phys
        if len(self.ind_parameters) > 0:
            x_full[self.ind_parameters] = self.parameters[0, self.ind_parameters]
        return x_full

    def evaluate(self, x_norm):
        """Evaluate the system model for a normalized input vector."""
        x_phys = self.denormalize(x_norm)
        x_full = self.construct_full_vector(x_phys)
        # evaluate_matrix expects (n_vars, n_samples), so reshape to (n_vars, 1)
        y = self.problem.evaluate_matrix(x_full.reshape(-1, 1)).flatten()
        return y

    def compute_objective(self, x_norm):
        """
        Compute the objective function value.
        Objective = Sum of positive constraint violations + Weighted Objectives (if enabled).
        """
        y = self.evaluate(x_norm)
        
        # Compute sum of positive constraint violations: sum(max(0, violations))
        # Normalize constraints by requirement width for balanced optimization
        req_width = self.reqU - self.reqL
        # Avoid division by zero for infinite bounds
        req_width = np.where(np.isinf(req_width), 1.0, req_width)
        
        # Calculate normalized constraint violations (only when outside bounds)
        c_upper = np.maximum(0, (y - self.reqU) / req_width)  # Violation when y > reqU
        c_lower = np.maximum(0, (self.reqL - y) / req_width)  # Violation when y < reqL
            
        # Handle infinite bounds - these don't contribute to violations
        c_upper = np.where(np.isinf(self.reqU), 0.0, c_upper)
        c_lower = np.where(np.isinf(self.reqL), 0.0, c_lower)
            
        # Sum of positive violations (constraint violations)
        violations = np.concatenate([c_upper, c_lower])
        obj_sum = np.sum(np.maximum(0, violations))
        
        # Add objectives if requested and objectives exist
        if self.include_objectives and self.objective_indices:
            for idx, weight in zip(self.objective_indices, self.objective_weights):
                obj_sum += weight * y[idx]
            
        return obj_sum

    def compute_constraints_normalized(self, x_norm):
        """
        Compute normalized constraints for gradient-based solvers (SLSQP).
        Returns array where values >= 0 indicate satisfaction.
        """
        y = self.evaluate(x_norm)
        
        # Normalize constraints by requirement width for balanced optimization
        req_width = self.reqU - self.reqL
        # Avoid division by zero for infinite bounds
        req_width = np.where(np.isinf(req_width), 1.0, req_width)
        
        # Req: y <= reqU  --> (reqU - y) / req_width >= 0
        # Req: y >= reqL  --> (y - reqL) / req_width >= 0
        c_upper = (self.reqU - y) / req_width
        c_lower = (y - self.reqL) / req_width
        
        # Handle infinite bounds - these are always satisfied
        c_upper = np.where(np.isinf(self.reqU), 1e19, c_upper)
        c_lower = np.where(np.isinf(self.reqL), 1e19, c_lower)
        
        return np.concatenate((c_upper, c_lower))

    def compute_constraints_raw(self, x_norm):
        """
        Compute raw constraints (Bound - Value >= 0).
        Useful for solvers that don't need normalization or handle it differently.
        """
        y = self.evaluate(x_norm)
        
        c1 = self.reqU - y 
        c2 = y - self.reqL
        
        # Use a large but reasonable value for infinite bounds to avoid ill-conditioning
        c1 = np.where(np.isinf(self.reqU), 1e6, c1)
        c2 = np.where(np.isinf(self.reqL), 1e6, c2)
        
        return np.concatenate((c1, c2))

    def compute_constraints_finite_only(self, x_norm):
        """
        Compute only finite constraints.
        Useful for Nevergrad or other solvers where infinite constraints might cause issues.
        """
        y = self.evaluate(x_norm)
        
        constraints = []
        if np.any(np.isfinite(self.reqU)):
            constraints.append(self.reqU[np.isfinite(self.reqU)] - y[np.isfinite(self.reqU)])
        if np.any(np.isfinite(self.reqL)):
            constraints.append(y[np.isfinite(self.reqL)] - self.reqL[np.isfinite(self.reqL)])
        
        if constraints:
            return np.concatenate(constraints)
        else:
            return np.array([0.0])  # No constraints, dummy satisfied
