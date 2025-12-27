# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

"""
Computation engine for solution space analysis.

This module provides the core computational functions for solution space
exploration, including box-shaped solution space computation, resampling
for visualization, and product family analysis.
"""

import numpy as np
import logging
from .solver_engine import SolutionSpaceSolver
from .monte_carlo_sampling import monte_carlo

logger = logging.getLogger(__name__)

def compute_solution_space(problem, weight, dsl, dsu, l, u, reqU, reqL, parameters, sample_size=1000, callback=None, solver_type='pymoo'):
    """
    Compute the maximal box-shaped solution space using evolutionary optimization.

    Uses the SolutionSpaceSolver to find the largest hyper-rectangle in the
    design space that satisfies all constraints. The algorithm evolves box
    boundaries to maximize the feasible region volume.

    Args:
        problem: XRayProblem instance defining the system model
        weight: Weight vector for multi-objective optimization
        dsl: Lower bounds of design space
        dsu: Upper bounds of design space
        l: Lower bounds for box optimization
        u: Upper bounds for box optimization
        reqU: Upper requirement bounds (constraints)
        reqL: Lower requirement bounds (constraints)
        parameters: Parameter matrix defining fixed vs variable parameters
        sample_size: Number of Monte Carlo samples for validation
        callback: Optional progress callback function
        solver_type: Solver type for feasible point finding ('pymoo' or 'goal_attainment')

    Returns:
        tuple: (final_box, convergence_data, population_data, samples)
            - final_box: Optimized design variable bounds
            - convergence_data: Evolution convergence history
            - population_data: Final population of solutions
            - samples: Monte Carlo validation samples
    """
    solver = SolutionSpaceSolver(problem, weight, dsl, dsu, l, u, reqU, reqL, parameters, solver_type=solver_type)

    solver.final_sample_size = sample_size

    # Run solver
    return solver.solve(callback=callback)

def resample_solution_space(problem, dv_par_box, dsl, dsu, reqU, reqL, parameters, sample_size=1000, active_plots=None):
    """
    Resample the solution space for visualization and analysis.

    Performs slice sampling through the solution space for each active plot
    combination. For DV-DV plots, samples slices through the full solution space.
    For DV-QoI plots, samples the full feasible region.

    Args:
        problem: XRayProblem instance
        dv_par_box: Current design variable parameter bounds
        dsl: Lower design space bounds
        dsu: Upper design space bounds
        reqU: Upper requirement bounds
        reqL: Lower requirement bounds
        parameters: Parameter matrix
        sample_size: Number of samples per slice
        active_plots: List of (idx1, idx2) tuples for plot axes

    Returns:
        list: List of sample dictionaries, one per active plot

    Sample Dictionary Structure:
        {
            "points": design variable samples,
            "is_good": feasibility flags,
            "is_bad": constraint violation flags,
            "violation_idx": indices of violated constraints,
            "qoi_values": quantity of interest values
        }
    """
    # Determine dimension
    if parameters is None or parameters.size == 0:
        dim = dv_par_box.shape[0]
        ind_parameters = np.array([], dtype=int)
        parameters = np.full((2, dim), np.nan)
    else:
        is_dv = np.isnan(parameters[0, :])
        dim = np.sum(is_dv)
        ind_parameters = np.where(~is_dv)[0]

    # Use physical units (norm=1, norm_l=0)
    dv_norm = np.ones(dim)
    dv_norm_l = np.zeros(dim)

    samples_list = []

    if active_plots:
        num_dv = dim
        num_qoi = len(reqL)
        # Sequential processing for plot sampling
        for (idx1, idx2) in active_plots:
            if idx1 < num_dv and idx2 < num_dv:
                # DV-DV slice sampling
                slice_box = dv_par_box.copy()
                slice_box[idx1, 0] = dsl[idx1]
                slice_box[idx1, 1] = dsu[idx1]
                slice_box[idx2, 0] = dsl[idx2]
                slice_box[idx2, 1] = dsu[idx2]
                PA, m, PB, samp, viol, y = monte_carlo(
                    problem, slice_box, parameters, reqL, reqU, dv_norm, dv_norm_l, ind_parameters, sample_size, dim
                )
            else:
                # DV-QoI full space sampling
                full_box = dv_par_box.copy()
                PA, m, PB, samp, viol, y = monte_carlo(
                    problem, full_box, parameters, reqL, reqU, dv_norm, dv_norm_l, ind_parameters, sample_size, dim
                )
            
            samples = {
                "points": samp,
                "is_good": PA,
                "is_bad": PB,
                "violation_idx": viol,
                "qoi_values": y
            }
            samples_list.append(samples)

    return samples_list

def compute_product_family_solutions(problem, weight, dsl, dsu, l, u, reqU, reqL, parameters, solver_type, progress_callback=None, stop_callback=None):
    """
    Compute solution spaces for product family variants and platform with progress reporting.

    Analyzes multiple requirement sets to find variant-specific solution spaces
    and their common platform (intersection). Useful for product family
    optimization and commonality analysis.

    Args:
        problem: XRayProblem with requirement_sets defined
        weight: Weight vector for multi-objective optimization
        dsl: Lower design space bounds
        dsu: Upper design space bounds
        l: Lower bounds for optimization
        u: Upper bounds for optimization
        reqU: Upper requirement bounds (base requirements)
        reqL: Lower requirement bounds (base requirements)
        parameters: Parameter matrix
        solver_type: Solver type for feasible point finding ('pymoo', 'goal_attainment', or 'nomad')
        progress_callback: Optional callback function for progress reporting
        stop_callback: Optional callback function for cancellation checking

    Returns:
        dict: Mapping of variant names to solution boxes, including 'Platform'
    """
    results = {}

    # 1. Base Requirements (Default)
    base_reqs = problem.quantities_of_interest

    # 2. Process variants in parallel using joblib
    results = {}
    total_variants = len(problem.requirement_sets)
    
    # Prepare tasks for parallel execution
    tasks = []
    variant_names = []
    
    for var_name, overrides in problem.requirement_sets.items():
        variant_names.append(var_name)
        
        # Build specific Req vectors
        reqL_var = []
        reqU_var = []

        for q in base_reqs:
            # Start with default
            r_min = q['min']
            r_max = q['max']

            # Apply override if exists
            if q['name'] in overrides:
                if 'req_min' in overrides[q['name']]: r_min = overrides[q['name']]['req_min']
                if 'req_max' in overrides[q['name']]: r_max = overrides[q['name']]['req_max']

            reqL_var.append(r_min)
            reqU_var.append(r_max)

        # Explicitly cast to float to prevent string subtraction errors
        reqL_var = np.array(reqL_var, dtype=float)
        reqU_var = np.array(reqU_var, dtype=float)
        
        tasks.append((reqL_var, reqU_var))

    # Define worker function for parallel execution
    def solve_variant(task_data):
        rL, rU = task_data
        # Create a new solver instance for each task (thread-safe)
        solver = SolutionSpaceSolver(problem, weight, dsl, dsu, l, u, rU, rL, parameters, solver_type=solver_type)
        final_box, _, _, _ = solver.solve(callback=None)
        return final_box

    try:
        from joblib import Parallel, delayed
        import dill
        
        # Use 'loky' backend for process-based parallelism (bypasses GIL)
        # This requires the problem object and system model to be picklable (dill handles this)
        parallel_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(solve_variant)(task) for task in tasks
        )
        
        # Map results back to variant names
        for i, var_name in enumerate(variant_names):
            results[var_name] = parallel_results[i]
            if progress_callback:
                progress_callback(var_name, i + 1, total_variants + 1, f"Completed {var_name}")
                
    except (ImportError, Exception) as e:
        logger.warning(f"Parallel execution failed: {e}. Falling back to sequential execution.")
        # Fallback to sequential execution
        current_variant = 0
        for i, var_name in enumerate(variant_names):
            if stop_callback and stop_callback():
                break
            current_variant += 1
            if progress_callback:
                progress_callback(var_name, current_variant, total_variants + 1, f"Starting {var_name}")
            
            reqL_var, reqU_var = tasks[i]
            solver = SolutionSpaceSolver(problem, weight, dsl, dsu, l, u, reqU_var, reqL_var, parameters, solver_type=solver_type)
            final_box, _, _, _ = solver.solve(callback=None)
            results[var_name] = final_box

    # 3. Calculate Intersection (Platform)
    if progress_callback:
        progress_callback("Platform", total_variants + 1, total_variants + 1, "Calculating platform")

    # 3. Calculate Intersection (Platform)
    if results:
        # Get all boxes (exclude any None values or non-array results)
        boxes = [box for box in results.values() if isinstance(box, np.ndarray) and box is not None]
        
        if boxes:  # Only calculate platform if we have valid boxes

            # Intersection: max of lower bounds, min of upper bounds
            platform = np.zeros_like(boxes[0])
            platform[:, 0] = np.maximum.reduce([box[:, 0] for box in boxes])  # Max of mins
            platform[:, 1] = np.minimum.reduce([box[:, 1] for box in boxes])  # Min of maxes
            
            # Check for infeasible platform (disjoint variants)
            is_feasible = np.all(platform[:, 0] <= platform[:, 1])
            
            # Always set platform (intersection per variable)
            results['Platform'] = platform
            results['Platform_Infeasible'] = not is_feasible
            
            # Calculate communality per variable (handles infeasible variables)
            communality = calculate_variable_communality(boxes, platform)
            results['Communality'] = communality
        else:
            # No valid boxes to intersect
            results['Platform'] = None
            results['Platform_Infeasible'] = True
            results['Communality'] = None

    return results


def calculate_variable_communality(variant_boxes, platform_box):
    """
    Calculate communality for each design variable across product family variants.
    
    Communality measures the degree to which a design variable is shared/common
    across different product variants. A communality of 1.0 indicates complete
    commonality (same value/range across all variants), while lower values
    indicate differentiation.
    
    Args:
        variant_boxes: List of solution space boxes for each variant
        platform_box: The common platform box (intersection of all variants)
    
    Returns:
        np.ndarray: Communality values for each design variable (0.0 to 1.0)
    """
    if not variant_boxes or platform_box is None:
        return None
    
    n_variables = platform_box.shape[0]
    communality = np.zeros(n_variables)
    
    # Vectorize communality calculation across all variables
    if variant_boxes:
        # Stack all variant boxes into a 3D array: (n_variants, n_variables, 2)
        variant_array = np.stack(variant_boxes, axis=0)  # (n_variants, n_variables, 2)
        
        # Extract mins and maxes for all variables at once
        variant_mins = variant_array[:, :, 0]  # (n_variants, n_variables)
        variant_maxes = variant_array[:, :, 1]  # (n_variants, n_variables)
        
        # Calculate total range (union) for all variables: (n_variables,)
        total_mins = np.min(variant_mins, axis=0)  # Min across variants for each variable
        total_maxes = np.max(variant_maxes, axis=0)  # Max across variants for each variable
        total_ranges = total_maxes - total_mins
        
        # Platform ranges: (n_variables,)
        platform_ranges = platform_box[:, 1] - platform_box[:, 0]
        
        # Handle zero platform ranges (fixed values)
        zero_platform_mask = platform_ranges <= 0
        if np.any(zero_platform_mask):
            platform_values = platform_box[zero_platform_mask, 0]
            # Check if all variants agree on the fixed value
            variant_mins_fixed = variant_mins[:, zero_platform_mask]  # (n_variants, n_fixed_vars)
            variant_maxes_fixed = variant_maxes[:, zero_platform_mask]  # (n_variants, n_fixed_vars)
            
            # All variants agree if min == max == platform_value for each variant
            all_agree = np.allclose(variant_mins_fixed, platform_values, atol=1e-10) & \
                       np.allclose(variant_maxes_fixed, platform_values, atol=1e-10)
            communality[zero_platform_mask] = np.where(all_agree, 1.0, 0.0)
        
        # Handle non-zero platform ranges
        nonzero_platform_mask = ~zero_platform_mask
        if np.any(nonzero_platform_mask):
            # Handle zero total ranges (all variants have same fixed value)
            zero_total_mask = total_ranges[nonzero_platform_mask] <= 0
            communality[nonzero_platform_mask] = np.where(
                zero_total_mask, 
                1.0,  # All variants have same fixed value
                platform_ranges[nonzero_platform_mask] / total_ranges[nonzero_platform_mask]  # Common / total range
            )
    
    return communality







