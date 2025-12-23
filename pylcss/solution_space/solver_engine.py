# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

import numpy as np
import time
import logging
from scipy.optimize import minimize
from .monte_carlo_sampling import monte_carlo
from .step_analysis import step_a_vectorized as step_a

from ..optimization.solvers.legacy import solve_with_nevergrad, solve_with_differential_evolution, run_goal_attainment_slsqp
from ..optimization.solvers.feasibility import FeasibilityProblem

logger = logging.getLogger(__name__)

class SolutionSpaceSolver:
    def __init__(self, problem, weight, dsl, dsu, l, u, reqU, reqL, parameters, slider_value=1, solver_type='goal_attainment', include_objectives=False):
        self.problem = problem
        self.weight = weight
        self.dsl = dsl
        self.dsu = dsu
        self.l = l
        self.u = u
        self.reqU = np.asarray(reqU, dtype=float)
        self.reqL = np.asarray(reqL, dtype=float)
        self.parameters = parameters
        self.slider_value = slider_value
        self.solver_type = solver_type
        self.include_objectives = include_objectives
        
        # Configuration
        self.initial_sample_size = 1000
        self.optimization_sample_size = 200
        self.final_sample_size = 2000
        self.max_iter_phase_1 = 100
        self.max_iter_phase_2 = 500
        self.growth_rate_init = 0.2
        self.tol_phase_1 = 1e-3
        
        # State
        self.original_dim = len(dsl)
        self.original_dsl = dsl.copy()
        self.original_dsu = dsu.copy()
        self.dv_norm = dsu - dsl
        
        # Remove fixed design variables (min == max)
        self.fixed_mask = self.dv_norm == 0
        if np.any(self.fixed_mask):
            # Keep only variable dimensions
            self.dsl = dsl[~self.fixed_mask]
            self.dsu = dsu[~self.fixed_mask]
            self.l = l[~self.fixed_mask]
            self.u = u[~self.fixed_mask]
            self.dv_norm = self.dv_norm[~self.fixed_mask]
        
        self.dim = len(self.dsl)
        
        self.dv_norm_l = self.dsl
        # Avoid division by zero for remaining variables
        self.dv_norm[self.dv_norm == 0] = 1.0
        
        self.l_norm = (l - self.dv_norm_l) / self.dv_norm
        self.u_norm = (u - self.dv_norm_l) / self.dv_norm
        self.dsl_norm = (dsl - self.dv_norm_l) / self.dv_norm
        self.dsu_norm = (dsu - self.dv_norm_l) / self.dv_norm
        
        # Stop flag for graceful cancellation
        self._stop = False
        
        self._process_parameters()        

    def _process_parameters(self):
        if self.parameters is None or self.parameters.size == 0:
            self.ind_parameters = np.array([], dtype=int)
            self.parameters = np.full((2, self.original_dim), np.nan)
        else:
            # Validate shape before accessing
            if self.parameters.shape[0] != 2:
                raise ValueError(f"Parameters array must have shape (2, n_vars), got {self.parameters.shape}")
            if self.parameters.shape[1] != self.original_dim:
                raise ValueError(f"Parameters array columns ({self.parameters.shape[1]}) must match design variables ({self.original_dim})")
            is_dv = np.isnan(self.parameters[0, :])
            self.ind_parameters = np.where(~is_dv)[0]
        
        # Process objectives
        self.objective_indices = []
        self.objective_weights = []
        
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            if qoi.get('minimize', False) or qoi.get('maximize', False):
                self.objective_indices.append(i)
                # For minimization: weight is positive, for maximization: weight is negative
                weight = qoi.get('weight', 1.0)
                if qoi.get('maximize', False):
                    weight = -weight  # Negative weight for maximization
                self.objective_weights.append(weight)

    def solve(self, callback=None, stop_callback=None):
        start_time = time.time()
        
        # 1. Initial Sampling & Feasible Point Search
        x0, initial_samples = self._find_feasible_point()
        
        # FIX: Even if x0 is None (extreme failure), return empty result safely
        if x0 is None:
            return self._empty_result(start_time)

        # 2. Phase I: Expansion
        if self.slider_value == 1:
            dvbox, phase1_samples = self._phase_i_expansion(x0, callback, stop_callback)
        else:
            dvbox, phase1_samples = self._phase_i_expansion_fixed_lower(x0, callback, stop_callback)
                    
        # 3. Intermediate Phase 1 (Pareto Selection)
        if self.slider_value != 1:
            dvbox = self._intermediate_phase_1(phase1_samples)
            # Re-sample for Phase 2 prep
            Points_A, m, Points_B, dv_sample, violation_idx, _ = monte_carlo(
                self.problem, dvbox, self.parameters, self.reqL, self.reqU, 
                self.dv_norm, self.dv_norm_l, self.ind_parameters, self.optimization_sample_size, self.dim
            )
        
        # 4. Phase II: Refinement
        dvbox, phase2_samples = self._phase_ii_refinement(dvbox, callback, stop_callback)
        
        # 5. Final Analysis
        # If we found a specific feasible point x0 via solver, ensure it's included in the final results
        # This fixes the "plot vs table" mismatch where the best point wasn't shown
        extra_point = None
        if x0 is not None:
             extra_point = x0
             
        result = self._finalize_result(dvbox, initial_samples, phase2_samples, start_time, extra_point)
        
        return result

    def _find_feasible_point(self):
        # Initial box is the full design space
        init_box = np.column_stack((self.l_norm, self.u_norm))
        
        Points_A, m, Points_B, dv_sample, violation_idx, y_sample = monte_carlo(
            self.problem, init_box, self.parameters, self.reqL, self.reqU, 
            self.dv_norm, self.dv_norm_l, self.ind_parameters, self.initial_sample_size, self.dim
        )
        
        initial_samples = {
            'points': self._denormalize(dv_sample),
            'is_good': Points_A,
            'is_bad': Points_B,
            'violation_idx': violation_idx,
            'qoi_values': y_sample
        }
        
        # Make points full-dimensional
        if np.any(self.fixed_mask):
            full_points = np.full((self.original_dim, initial_samples['points'].shape[1]), np.nan)
            full_points[~self.fixed_mask] = initial_samples['points']
            full_points[self.fixed_mask] = self.original_dsl[self.fixed_mask].reshape(-1, 1)
            initial_samples['points'] = full_points
        
        if m > 0:
            ind_A = np.where(Points_A)[0]
            # Pick the most robust point (last one after sorting by c_max in monte_carlo)
            x0 = dv_sample[:, ind_A[-1]]
            
            # If we have objectives to optimize, don't stop here.
            if not (self.include_objectives and self.objective_indices):
                return x0, initial_samples
            
            # Use this feasible point as start for optimization
            x_start = x0
            # Also save the center of good points for better box initialization later
            good_points = dv_sample[:, ind_A]
            x_center = np.mean(good_points, axis=1)
        else:
            x_start = None
            x_center = None
            
        # Choose solver based on solver_type
        if self.solver_type == 'nevergrad':
            if x_start is None: logger.info("No feasible point found in random sample. Switching to Nevergrad...")
            solver_method = self._solve_with_nevergrad
        elif self.solver_type == 'differential_evolution':
            if x_start is None: logger.info("No feasible point found in random sample. Switching to Differential Evolution...")
            solver_method = self._solve_with_differential_evolution
        elif self.solver_type == 'goal_attainment':
            if x_start is None: logger.info("No feasible point found in random sample. Switching to SLSQP...")
            solver_method = self._solve_goal_attainment
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}. Must be 'nevergrad', 'differential_evolution', or 'goal_attainment'")
        
        # Pick the best candidate from random sampling to guide us if we don't have one
        if x_start is None:
            x_start = dv_sample[:, -1]
            x_center = x_start
        
        # Run chosen solver
        x_optimized = solver_method(x_start)
        
        return x_optimized, initial_samples

    def _trim_box(self, dvbox, dv_sample, Points_A):
        """
        Trims the box to the bounding box of the good points, with some relaxation.
        """
        if not np.any(Points_A):
            return dvbox
            
        good_samples = dv_sample[:, Points_A]
        
        # Find bounding box of good samples
        min_good = np.min(good_samples, axis=1)
        max_good = np.max(good_samples, axis=1)
        
        # Current box
        current_min = dvbox[:, 0]
        current_max = dvbox[:, 1]
        
        # Relaxed trim: Don't shrink all the way to the samples, keep some margin
        # This prevents over-fitting to the specific random samples
        # Adaptive alpha based on sample count: more samples = more confident trimming
        n_good_samples = good_samples.shape[1]
        alpha = min(0.8, max(0.5, n_good_samples / (self.dim * 50)))
        
        new_min = current_min * (1 - alpha) + min_good * alpha
        new_max = current_max * (1 - alpha) + max_good * alpha
        
        # Ensure we don't accidentally expand (though math above shouldn't allow it if samples are inside)
        new_min = np.maximum(new_min, current_min)
        new_max = np.minimum(new_max, current_max)
        
        return np.column_stack((new_min, new_max))

    def _phase_i_expansion(self, x0, callback, stop_callback=None):
        g = self.growth_rate_init
        smoothed_g = g  # Smoothed growth rate to prevent oscillations
        N = self.optimization_sample_size
        dvbox = np.column_stack((x0, x0))
        dvbox_old = dvbox.copy()
        mu_vec = []
        
        # Pareto tracking
        pareto_samples = {
            'dvbox': [dvbox.copy()],
            'phi_1': [-0.0], # -mu (initial mu is 0)
            'phi_2': [np.linalg.norm(self.weight * (dvbox[:, 0] - x0))] # Distance to x0
        }
        
        anisotropic = False
        # Track which dimensions are still expanding (anisotropic growth)
        expanding_dims = np.ones(self.dim, dtype=bool)
        
        for i in range(self.max_iter_phase_1):
            if self._stop or (stop_callback and stop_callback()):
                self._stop = True
                break
            # Expansion Step
            expanded = False
            expansion_attempts = 0
            patience_counter = 0 # Counter for consecutive failures
            max_expansion_attempts = 10  # Safety limit
            
            while not expanded and expansion_attempts < max_expansion_attempts:
                if not anisotropic:
                    expansion = g * (self.dsu_norm - self.dsl_norm)
                    dvbox_new = dvbox_old.copy()
                    dvbox_new[:, 0] -= expansion
                    dvbox_new[:, 1] += expansion
                else:
                    # Anisotropic expansion: only expand in directions that haven't hit constraints
                    box_size = dvbox_old[:, 1] - dvbox_old[:, 0]
                    expansion = g * box_size
                    dvbox_new = dvbox_old.copy()
                    
                    # Only expand in dimensions that are still active
                    expansion_masked = expansion * expanding_dims.astype(float)
                    dvbox_new[:, 0] -= expansion_masked
                    dvbox_new[:, 1] += expansion_masked
                
                dvbox_new[:, 0] = np.maximum(dvbox_new[:, 0], self.dsl_norm)
                dvbox_new[:, 1] = np.minimum(dvbox_new[:, 1], self.dsu_norm)
                
                # Use higher sampling density in Phase II (anisotropic mode)
                sample_size_phase = N * 2 if anisotropic else N
                
                Points_A, m, Points_B, dv_sample, violation_idx, _ = monte_carlo(
                    self.problem, dvbox_new, self.parameters, self.reqL, self.reqU, 
                    self.dv_norm, self.dv_norm_l, self.ind_parameters, sample_size_phase, self.dim
                )
                
                purity = m / sample_size_phase
                
                # Heuristic: If we found enough good points, accept expansion
                if purity >= 0.5 or expansion_attempts >= 9:
                    dvbox = dvbox_new
                    # Adaptive growth update
                    growth_factor = 2.0 if purity > 0.8 else (1.0 + purity)
                    new_g = growth_factor * g
                    smoothed_g = 0.7 * smoothed_g + 0.3 * new_g  # Smooth the growth rate
                    g = smoothed_g
                    expanded = True
                    patience_counter = 0 # Reset patience on success
                    
                    # Update anisotropic expansion: check if any dimension hit design space bounds
                    if anisotropic:
                        # If we hit the design space bounds in any direction, stop expanding there
                        hit_lower = np.abs(dvbox[:, 0] - self.dsl_norm) < 1e-6
                        hit_upper = np.abs(dvbox[:, 1] - self.dsu_norm) < 1e-6
                        expanding_dims = expanding_dims & ~(hit_lower | hit_upper)
                        
                        # If expansion was paused due to constraint violations, try to resume
                        if not np.any(expanding_dims):
                            # Check if we can resume expansion in some directions
                            # by testing a small expansion
                            test_expansion = 0.01 * (self.dsu_norm - self.dsl_norm)
                            test_box = dvbox.copy()
                            test_box[:, 0] -= test_expansion
                            test_box[:, 1] += test_expansion
                            test_box[:, 0] = np.maximum(test_box[:, 0], self.dsl_norm)
                            test_box[:, 1] = np.minimum(test_box[:, 1], self.dsu_norm)
                            
                            # Quick test - if we can find any good points, resume expansion
                            test_PA, test_m, _, _, _, _ = monte_carlo(
                                self.problem, test_box, self.parameters, self.reqL, self.reqU, 
                                self.dv_norm, self.dv_norm_l, self.ind_parameters, min(100, N), self.dim
                            )
                            if test_m > 0:
                                expanding_dims = ~hit_lower & ~hit_upper  # Resume in non-boundary directions
                        
                        # If no dimensions are expanding anymore, stop the phase
                        if not np.any(expanding_dims):
                            break
                elif purity > 0.1:
                    # Partial Success: We expanded too much, but there are still good points.
                    # Instead of rejecting entirely, let's TRIM the box to the good points.
                    # This mimics the "Growth -> Trim" cycle.
                    dvbox_trimmed = self._trim_box(dvbox_new, dv_sample, Points_A)
                    
                    # Verify the trimmed box is actually better
                    # (Quick check with existing samples is hard because they might be outside now)
                    # So we just accept it but reduce growth rate slightly
                    dvbox = dvbox_trimmed
                    expanded = True
                    patience_counter = 0 # Reset patience on success
                    
                    # Reduce growth rate since we had to trim
                    g *= 0.8
                    smoothed_g = g
                else:
                    # Expansion failed - implement growth decay with directional preference
                    patience_counter += 1
                    
                    # Aggressive decay if we fail repeatedly (Patience Logic)
                    decay_factor = 2 * (m / N)
                    if patience_counter > 2:
                        decay_factor *= 0.5 # Halve the growth rate more aggressively
                        
                    new_g = decay_factor * g
                    smoothed_g = 0.7 * smoothed_g + 0.3 * new_g
                    g = smoothed_g
                    
                    if g <= 0:
                        g = 0.006
                    else:
                        # If we're in anisotropic mode and expansion failed, 
                        # analyze constraint violations to identify problematic directions
                        if anisotropic and expansion_attempts < 5 and len(violation_idx) > 0:
                            try:
                                # Analyze violations from the failed expansion attempt
                                bad_points_mask = ~Points_A  # Points that failed constraints
                                if np.any(bad_points_mask):
                                    bad_violations = violation_idx[bad_points_mask]
                                    
                                    # Count violations per constraint type
                                    unique_violations, violation_counts = np.unique(bad_violations, return_counts=True)
                                    
                                    # If 90% of violations belong to one constraint, pause expansion
                                    if len(violation_counts) > 0:
                                        max_violation_count = violation_counts.max()
                                        total_bad_points = bad_points_mask.sum()
                                        
                                        if max_violation_count / total_bad_points >= 0.9:
                                            # Most violations are from one constraint - pause all expansion briefly
                                            # This gives the growth decay a chance to work
                                            expanding_dims_temp = expanding_dims.copy()
                                            expanding_dims[:] = False  # Pause expansion in all directions
                                            
                                            # Resume after a few iterations (let growth decay work)
                                            # We'll restore expansion dims in the next successful expansion
                            
                            except Exception:
                                # If constraint analysis fails, continue with current strategy
                                pass
                
                expansion_attempts += 1
            
            if m == 0:
                # If we expanded into a totally invalid region, stop expansion
                break
                
            # Shrink Step (Step A)
            dvbox, mu = step_a(dvbox, dv_sample, Points_A, Points_B, self.dim, self.weight)
            mu_vec.append(mu)
            
            if self.slider_value != 1:
                pareto_samples['dvbox'].append(dvbox.copy())
                pareto_samples['phi_1'].append(-mu)
                pareto_samples['phi_2'].append(np.linalg.norm(self.weight * (dvbox[:, 0] - self.dsl_norm)))
            
            # Convergence Check
            if len(mu_vec) > 3:
                rel_change = abs((mu_vec[-1] - mu_vec[-2]) / mu_vec[-1]) if mu_vec[-1] != 0 else 0
                if rel_change < self.tol_phase_1:
                    # Check if we have enough good points before assuming convergence
                    min_good_points = max(10, self.optimization_sample_size // 20)  # At least 10 or 5% of sample size
                    if m < min_good_points:
                        # Too few good points - likely poor sampling, trigger resampling by continuing
                        if callback:
                            self._report_progress(callback, dvbox, dv_sample, Points_A, Points_B, violation_idx, f"Phase I - Iter {i} (Resampling - too few good points)")
                        continue
                    
                    if not anisotropic:
                        anisotropic = True
                        g = 0.1 
                        self.tol_phase_1 = 5e-3
                    else:
                        break
            
            dvbox_old = dvbox.copy()
            
            if callback:
                self._report_progress(callback, dvbox, dv_sample, Points_A, Points_B, violation_idx, f"Phase I - Iter {i}")
                
        return dvbox, pareto_samples

    def _intermediate_phase_1(self, pareto_samples):
        phi_1 = np.array(pareto_samples['phi_1'])
        phi_2 = np.array(pareto_samples['phi_2'])
        
        # Normalize
        def normalize(arr):
            mn, mx = np.min(arr), np.max(arr)
            if mx - mn == 0: return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)
            
        phi_1_norm = normalize(phi_1)
        phi_2_norm = normalize(phi_2)
        
        costs = np.column_stack((phi_1_norm, phi_2_norm))
        is_pareto = self._get_pareto_front(costs)
        
        pareto_indices = np.where(is_pareto)[0]
        p_phi_1 = phi_1_norm[pareto_indices]
        p_phi_2 = phi_2_norm[pareto_indices]
        
        # CHIM Selection
        q = np.array([0, 2 * self.slider_value - 1, 0])
        u = np.array([1, 1, 0])
        u_norm = np.linalg.norm(u)
        
        best_idx = -1
        min_dist = np.inf
        
        for i in range(len(pareto_indices)):
            p = np.array([p_phi_1[i], p_phi_2[i], 0])
            d = np.linalg.norm(np.cross(p - q, u)) / u_norm
            
            if d <= min_dist:
                min_dist = d
                best_idx = pareto_indices[i]
                
        return pareto_samples['dvbox'][best_idx]

    def _get_pareto_front(self, costs):
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        
        is_pareto = np.zeros(n_points, dtype=bool)
        is_pareto[is_efficient] = True
        return is_pareto

    def _shift_box(self, dvbox, dv_sample, Points_A):
        """
        Shifts the box towards the center of mass of the good points.
        """
        if not np.any(Points_A):
            return dvbox
            
        good_samples = dv_sample[:, Points_A]
        mean_good = np.mean(good_samples, axis=1)
        
        box_center = np.mean(dvbox, axis=1)
        shift_vector = mean_good - box_center
        
        # Damping factor for shift to avoid jumping around too much
        alpha = 0.5 
        shift_vector *= alpha
        
        new_box = dvbox.copy()
        new_box[:, 0] += shift_vector
        new_box[:, 1] += shift_vector
        
        # Clip to design space
        new_box[:, 0] = np.maximum(new_box[:, 0], self.dsl_norm)
        new_box[:, 1] = np.minimum(new_box[:, 1], self.dsu_norm)
        
        return new_box

    def _phase_ii_refinement(self, dvbox, callback, stop_callback=None):
        N = self.optimization_sample_size
        for i in range(self.max_iter_phase_2):
            if self._stop or (stop_callback and stop_callback()):
                self._stop = True
                break
            Points_A, m, Points_B, dv_sample, violation_idx, _ = monte_carlo(
                self.problem, dvbox, self.parameters, self.reqL, self.reqU, 
                self.dv_norm, self.dv_norm_l, self.ind_parameters, N, self.dim
            )
            
            if m == N:
                break
                
            # Step 1: Trim (Step A)
            dvbox, mu = step_a(dvbox, dv_sample, Points_A, Points_B, self.dim, self.weight)
            
            # Step 2: Shift
            dvbox = self._shift_box(dvbox, dv_sample, Points_A)
            
            if callback and i % 10 == 0:
                 self._report_progress(callback, dvbox, dv_sample, Points_A, Points_B, violation_idx, f"Phase II - Iter {i}")
                 
        return dvbox, None

    def _finalize_result(self, dvbox, initial_samples, phase2_samples, start_time, extra_point=None):
        Points_A, m, Points_B, dv_sample, violation_idx, y_sample = monte_carlo(
            self.problem, dvbox, self.parameters, self.reqL, self.reqU, 
            self.dv_norm, self.dv_norm_l, self.ind_parameters, self.final_sample_size, self.dim
        )
        
        # If we have an extra point (e.g. from solver), evaluate it and add it
        if extra_point is not None:
            # extra_point is normalized (dim,)
            extra_point_2d = extra_point.reshape(-1, 1)
            
            # Evaluate using monte_carlo logic (reusing function for consistency)
            # We create a tiny box around it or just pass it directly if monte_carlo supported it
            # Instead, let's just manually evaluate it to match the data structure
            
            # Denormalize
            x_phys = self._denormalize(extra_point_2d)
            
            # Full vector construction
            if np.any(self.fixed_mask):
                x_full = np.full((self.original_dim, 1), np.nan)
                x_full[~self.fixed_mask] = x_phys
                x_full[self.fixed_mask] = self.original_dsl[self.fixed_mask].reshape(-1, 1)
            else:
                x_full = x_phys
                
            # Evaluate
            y_val = self.problem.evaluate_matrix(x_full)
            
            # Check constraints
            c_upper = y_val - self.reqU.reshape(-1, 1)
            c_lower = self.reqL.reshape(-1, 1) - y_val
            
            # Handle infinite bounds
            c_upper[np.isinf(self.reqU)] = -np.inf
            c_lower[np.isinf(self.reqL)] = -np.inf
            
            c_max = np.maximum(c_upper, c_lower)
            max_violation = np.max(c_max, axis=0)
            is_good_pt = max_violation <= 0
            
            # Determine violation index
            viol_idx = np.zeros(1, dtype=int)
            if not is_good_pt[0]:
                viol_idx[0] = np.argmax(c_max[:, 0])
            
            # Append to Monte Carlo results
            dv_sample = np.hstack((dv_sample, extra_point_2d))
            Points_A = np.append(Points_A, is_good_pt)
            Points_B = np.append(Points_B, ~is_good_pt)
            violation_idx = np.append(violation_idx, viol_idx)
            y_sample = np.hstack((y_sample, y_val))
        
        dv_par_box = self._denormalize_box(dvbox)
        
        # Reconstruct full-dimensional box
        if np.any(self.fixed_mask):
            full_dv_par_box = np.full((self.original_dim, 2), np.nan)
            full_dv_par_box[~self.fixed_mask] = dv_par_box
            full_dv_par_box[self.fixed_mask, 0] = self.original_dsl[self.fixed_mask]
            full_dv_par_box[self.fixed_mask, 1] = self.original_dsu[self.fixed_mask]
        else:
            full_dv_par_box = dv_par_box
        
        dv_sample_phys = self._denormalize(dv_sample)
        
        # For samples, also need to add fixed variables
        if np.any(self.fixed_mask):
            full_points = np.full((self.original_dim, dv_sample_phys.shape[1]), np.nan)
            full_points[~self.fixed_mask] = dv_sample_phys
            full_points[self.fixed_mask] = self.original_dsl[self.fixed_mask].reshape(-1, 1)
            dv_sample_phys = full_points
            
            # Also for initial_samples
            if initial_samples['points'].shape[0] != self.original_dim:
                full_initial_points = np.full((self.original_dim, initial_samples['points'].shape[1]), np.nan)
                full_initial_points[~self.fixed_mask] = initial_samples['points']
                full_initial_points[self.fixed_mask] = self.original_dsl[self.fixed_mask].reshape(-1, 1)
                initial_samples['points'] = full_initial_points
        
        # Merge samples
        points_all = np.hstack((initial_samples['points'], dv_sample_phys))
        good_all = np.concatenate((initial_samples['is_good'], Points_A))
        bad_all = np.concatenate((initial_samples['is_bad'], Points_B))
        violation_all = np.concatenate((initial_samples['violation_idx'], violation_idx))
        
        if 'qoi_values' in initial_samples:
            qoi_all = np.hstack((initial_samples['qoi_values'], y_sample))
        else:
            qoi_all = y_sample
            
        samples = {
            'points': points_all,
            'is_good': good_all,
            'is_bad': bad_all,
            'violation_idx': violation_all,
            'qoi_values': qoi_all
        }
        
        elapsed_time = time.time() - start_time
        
        return full_dv_par_box, 1, elapsed_time, samples

    def _denormalize(self, dv_norm_samples):
        return dv_norm_samples * self.dv_norm.reshape(-1, 1) + self.dv_norm_l.reshape(-1, 1)

    def _denormalize_box(self, box_norm):
        return box_norm * self.dv_norm.reshape(-1, 1) + self.dv_norm_l.reshape(-1, 1)

    def _report_progress(self, callback, dvbox, dv_sample, Points_A, Points_B, violation_idx, msg):
        curr_box = self._denormalize_box(dvbox)
        
        # Reconstruct full-dimensional box
        if np.any(self.fixed_mask):
            full_curr_box = np.full((self.original_dim, 2), np.nan)
            full_curr_box[~self.fixed_mask] = curr_box
            full_curr_box[self.fixed_mask, 0] = self.original_dsl[self.fixed_mask]
            full_curr_box[self.fixed_mask, 1] = self.original_dsu[self.fixed_mask]
        else:
            full_curr_box = curr_box
        
        curr_samples = self._denormalize(dv_sample)
        
        # Reconstruct full-dimensional samples
        if np.any(self.fixed_mask):
            full_curr_samples = np.full((self.original_dim, curr_samples.shape[1]), np.nan)
            full_curr_samples[~self.fixed_mask] = curr_samples
            full_curr_samples[self.fixed_mask] = self.original_dsl[self.fixed_mask].reshape(-1, 1)
        else:
            full_curr_samples = curr_samples
        
        sample_data = {
            'points': full_curr_samples,
            'is_good': Points_A,
            'is_bad': Points_B,
            'violation_idx': violation_idx
        }
        callback(full_curr_box, sample_data, msg)

    def _solve_goal_attainment(self, x_start):
        """
        Standard constrained optimization using SLSQP.
        Minimizes the sum of objectives subject to requirement constraints.
        """
        logger.info("Running Standard SLSQP Optimization...")
        
        feas_prob = FeasibilityProblem(
            self.problem, self.parameters, self.ind_parameters, 
            self.reqL, self.reqU, self.dv_norm, self.dv_norm_l,
            self.include_objectives, self.objective_indices, self.objective_weights
        )
        
        # Bounds: Design variables bounded [0, 1] (normalized)
        bounds = list(zip(self.dsl_norm, self.dsu_norm))
        
        return run_goal_attainment_slsqp(feas_prob.compute_objective, feas_prob.compute_constraints_normalized, x_start, bounds)

    def _solve_with_nevergrad(self, x_start):
        """
        Uses Nevergrad with constraint handling to find a feasible point.
        """
        logger.info("Running Nevergrad with constraint handling")

        feas_prob = FeasibilityProblem(
            self.problem, self.parameters, self.ind_parameters, 
            self.reqL, self.reqU, self.dv_norm, self.dv_norm_l,
            self.include_objectives, self.objective_indices, self.objective_weights
        )

        bounds = list(zip(self.l_norm, self.u_norm))

        # Create constraint for Nevergrad
        constraints = [{'type': 'ineq', 'fun': feas_prob.compute_constraints_finite_only}]

        result = solve_with_nevergrad(feas_prob.compute_objective, x_start, bounds, maxiter=5000, constraints=constraints)
        return result.x if result.success else None

    def _solve_with_differential_evolution(self, x_start):
        """
        Uses Differential Evolution with constraint handling to find a feasible point.
        """
        logger.info("Running Differential Evolution with constraint handling")

        feas_prob = FeasibilityProblem(
            self.problem, self.parameters, self.ind_parameters, 
            self.reqL, self.reqU, self.dv_norm, self.dv_norm_l,
            self.include_objectives, self.objective_indices, self.objective_weights
        )

        # Use single constraint for efficiency and robustness
        constraints = [{'type': 'ineq', 'fun': feas_prob.compute_constraints_raw}]
        bounds = list(zip(self.l_norm, self.u_norm))

        # Pass a dummy callback to keep the UI responsive if needed, 
        # but solve_with_differential_evolution handles its own internal callback logic
        result = solve_with_differential_evolution(feas_prob.compute_objective, bounds, constraints=constraints, maxiter=5000)
        
        # If we found a valid point, return it.
        if result.success:
            return result.x
        else:
            # If DE failed but returned a "best so far" (result.x), check if it's actually feasible
            # Evaluate constraints one last time
            cons = feas_prob.compute_constraints_raw(result.x)
            if np.all(cons >= -1e-6): # Allow small tolerance
                return result.x
            return None

    def _empty_result(self, start_time):
        empty_samples = {
            'points': np.zeros((self.original_dim, 0)), 
            'is_good': np.array([], dtype=bool), 
            'is_bad': np.array([], dtype=bool),
            'violation_idx': np.array([], dtype=int),
            'qoi_values': np.zeros((len(self.reqL), 0))
        }
        return np.zeros((self.original_dim, 2)), 0, time.time() - start_time, empty_samples


    def _phase_i_expansion_fixed_lower(self, x0, callback, stop_callback=None):
        """
        Expands the solution space box only in the upper direction, keeping 
        lower bounds fixed at x0. Used for Multi-Objective SSE.
        """
        g = self.growth_rate_init
        N = self.optimization_sample_size
        
        # Initial box is just the point x0
        dvbox = np.column_stack((x0, x0)) 
        dvbox_old = dvbox.copy()
        
        mu_vec = []
        # Pareto tracking
        pareto_samples = {
            'dvbox': [dvbox.copy()],
            'phi_1': [-0.0],
            'phi_2': [0.0] 
        }
        
        anisotropic = False
        
        for i in range(self.max_iter_phase_1):
            if self._stop or (stop_callback and stop_callback()):
                self._stop = True
                break
                
            # --- Expansion Step (Fixed Lower Bound) ---
            expanded = False
            expansion_attempts = 0
            
            while not expanded and expansion_attempts < 10:
                dvbox_new = dvbox_old.copy()
                
                if not anisotropic:
                    # Isotropic: Expand Upper Bound by fraction of design space
                    expansion = g * (self.dsu_norm - self.dsl_norm)
                    dvbox_new[:, 1] += expansion # Only add to Upper Bound
                else:
                    # Anisotropic: Expand proportional to current box size
                    box_size = dvbox_old[:, 1] - dvbox_old[:, 0]
                    expansion = g * box_size
                    dvbox_new[:, 1] += expansion # Only add to Upper Bound
                
                # Clip to Design Space limits
                dvbox_new[:, 1] = np.minimum(dvbox_new[:, 1], self.dsu_norm)
                # Ensure Lower Bound stays fixed (no change to dvbox_new[:, 0])

                # Monte Carlo Check
                Points_A, m, Points_B, dv_sample, violation_idx, _ = monte_carlo(
                    self.problem, dvbox_new, self.parameters, self.reqL, self.reqU, 
                    self.dv_norm, self.dv_norm_l, self.ind_parameters, N, self.dim
                )
                
                # Growth Control (Same as standard)
                if m / N >= 0.95 or expansion_attempts >= 9:
                    dvbox = dvbox_new
                    g = 2 * (m / N) * g
                    expanded = True
                else:
                    g = 2 * (m / N) * g
                    if g <= 0: g = 0.006
                
                expansion_attempts += 1
            
            if m == 0: break # Stop if invalid region
                
            # --- Shrink Step (Step A) ---
            # Important: step_a might shrink the lower bound *upwards* (good), 
            # but we must ensure it doesn't shrink *below* our fixed anchor if that was the intent.
            # However, standard Step A is usually fine here as it shrinks tightest good box.
            dvbox, mu = step_a(dvbox, dv_sample, Points_A, Points_B, self.dim, self.weight)
            mu_vec.append(mu)
            
            if self.slider_value != 1:
                pareto_samples['dvbox'].append(dvbox.copy())
                pareto_samples['phi_1'].append(-mu)
                pareto_samples['phi_2'].append(np.linalg.norm(self.weight * (dvbox[:, 0] - self.dsl_norm)))

            # Convergence Check
            if len(mu_vec) > 3:
                rel_change = abs((mu_vec[-1] - mu_vec[-2]) / mu_vec[-1]) if mu_vec[-1] != 0 else 0
                if rel_change < self.tol_phase_1:
                    if not anisotropic:
                        anisotropic = True
                        g = 0.1 
                        self.tol_phase_1 = 5e-3
                    else:
                        break
            
            dvbox_old = dvbox.copy()
            
            if callback:
                self._report_progress(callback, dvbox, dv_sample, Points_A, Points_B, violation_idx, f"Phase I (Fixed LB) - Iter {i}")
        
        return dvbox, pareto_samples



