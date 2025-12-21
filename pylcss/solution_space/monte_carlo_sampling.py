# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

import numpy as np
from scipy.stats import qmc

def monte_carlo(problem, dvbox, parameters, reqL, reqU, dv_norm, dv_norm_l, ind_parameters, N, dim):
    """
    Performs Monte Carlo Sampling in a given DV box.
    
    Args:
        problem: The system model/problem object.
        dvbox: Intervals of the DVs (normalized). Shape (dim, 2).
        parameters: Parameters matrix (2, total_vars). 
                    Columns with NaN in row 0 are DVs.
                    Columns with values are parameters [min; max].
        reqL: Lower bound requirements (num_qoi,).
        reqU: Upper bound requirements (num_qoi,).
        dv_norm: Norming factor for the DVs (dim,).
        dv_norm_l: Lower bound for norming (dim,).
        ind_parameters: Indices of parameters in the full input vector.
        N: Number of sample points.
        dim: Number of DVs.
        
    Returns:
        Points_A: Boolean array indicating good points (N,).
        m: Number of good points.
        Points_B: Boolean array indicating bad points (N,).
        dv_sample: The sampled DVs (normalized) (dim, N).
    """
    
    # Input validation
    if N <= 0:
        raise ValueError(f"Number of samples N must be positive, got {N}")
    if dim <= 0:
        raise ValueError(f"Dimension dim must be positive, got {dim}")
    if dvbox.shape != (dim, 2):
        raise ValueError(f"dvbox shape mismatch: expected ({dim}, 2), got {dvbox.shape}")
    
    # 1. Sample DVs (normalized) using Quasi-Monte Carlo
    # dvbox is (dim, 2) -> [min, max]
    lower = dvbox[:, 0].reshape(-1, 1)
    upper = dvbox[:, 1].reshape(-1, 1)
    
    # Batch processing configuration
    BATCH_SIZE = 10000
    num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Pre-allocate result arrays
    # We need to determine num_qoi from the first batch or assume it matches reqU
    num_qoi = reqU.shape[0]
    
    # Results containers
    c_min_all = np.zeros(N)
    violation_idx_all = np.zeros(N, dtype=int)
    dv_sample_all = np.zeros((dim, N))
    y_sample_all = np.zeros((num_qoi, N))
    
    # Total variables (DVs + Parameters)
    total_vars = parameters.shape[1]
    
    # Construct indices for DVs
    all_indices = np.arange(total_vars)
    ind_dvs = np.setdiff1d(all_indices, ind_parameters)
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, N)
        current_batch_size = end_idx - start_idx
        
        # Generate batch samples using LHS
        # Note: Generating independent LHS batches is a valid strategy (Replicated LHS)
        # It avoids allocating the full N x dim matrix at once
        sampler = qmc.LatinHypercube(d=dim, seed=None) # Random seed for each batch
        lhs_samples = sampler.random(n=current_batch_size).T  # Shape: (dim, batch_size)
        
        # Scale to the DV box bounds
        dv_sample_batch = (upper - lower) * lhs_samples + lower
        
        # Store DV samples
        dv_sample_all[:, start_idx:end_idx] = dv_sample_batch
        
        # Create full input matrix for this batch
        x_sample_batch = np.zeros((total_vars, current_batch_size))
        
        # Denormalize DVs for evaluation
        x_dvs_phys = dv_sample_batch * dv_norm.reshape(-1, 1) + dv_norm_l.reshape(-1, 1)
        x_sample_batch[ind_dvs, :] = x_dvs_phys
        
        # Fill in Parameters
        if len(ind_parameters) > 0:
            p_min = parameters[0, ind_parameters].reshape(-1, 1)
            p_max = parameters[1, ind_parameters].reshape(-1, 1)
            
            # Sample parameters
            param_sampler = qmc.LatinHypercube(d=len(ind_parameters), seed=None)
            param_lhs = param_sampler.random(n=current_batch_size).T
            p_sample = (p_max - p_min) * param_lhs + p_min
            
            x_sample_batch[ind_parameters, :] = p_sample
            
        # Evaluate System for this batch
        y_sample_batch = problem.evaluate_matrix(x_sample_batch)
        
        # Store outputs
        y_sample_all[:, start_idx:end_idx] = y_sample_batch
        
        # Calculate constraints for this batch
        # Positive = satisfied
        ct = reqU.reshape(-1, 1) - y_sample_batch
        cd = y_sample_batch - reqL.reshape(-1, 1)
        
        c = np.vstack((ct, cd))
        c_min_batch = np.min(c, axis=0)
        violation_idx_batch = np.argmin(c, axis=0)
        
        c_min_all[start_idx:end_idx] = c_min_batch
        violation_idx_all[start_idx:end_idx] = violation_idx_batch

    # Removed expensive sorting for performance (O(N log N))
    # ind_sort = np.argsort(c_min_all)
    
    # Return unsorted arrays directly
    c_min_sorted = c_min_all
    dv_sample_sorted = dv_sample_all
    violation_idx_sorted = violation_idx_all
    y_sample_sorted = y_sample_all
    
    Points_A_sorted = c_min_sorted >= -1e-12  # Feasible points (all constraints satisfied, with tolerance for floating-point errors)
    m = np.sum(Points_A_sorted)
    Points_B_sorted = ~Points_A_sorted  # Infeasible points
    
    return Points_A_sorted, m, Points_B_sorted, dv_sample_sorted, violation_idx_sorted, y_sample_sorted







