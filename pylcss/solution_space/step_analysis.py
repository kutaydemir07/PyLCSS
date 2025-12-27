# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

import numpy as np
import warnings

def _check_memory_usage(dim, chunk_size, num_B, operation_name="operation"):
    """
    Check if the requested array sizes could cause memory issues.
    
    Args:
        dim: Number of dimensions
        chunk_size: Size of current chunk
        num_B: Number of bad points
        operation_name: Name of the operation for warning message
    """
    # Estimate memory usage for key arrays (in elements, roughly)
    # A_exp: (dim, chunk_size, num_B)
    # B_exp: (dim, chunk_size, num_B) - but B is broadcasted
    # mask arrays: (dim, chunk_size, num_B)
    # cost arrays: (dim, chunk_size, num_B)
    
    array_elements = dim * chunk_size * num_B
    
    # Rough estimate: 8 bytes per float64 element, plus overhead
    estimated_mb = (array_elements * 8) / (1024 * 1024)
    
    # Warn if estimated memory usage is very high (>500MB)
    if estimated_mb > 500:
        warnings.warn(
            f"High memory usage in {operation_name}: ~{estimated_mb:.1f}MB estimated for "
            f"arrays of shape ({dim}, {chunk_size}, {num_B}). Consider reducing chunk_size "
            f"or processing fewer points.",
            UserWarning,
            stacklevel=2
        )
    elif estimated_mb > 100:
        # Less severe warning for moderately high usage
        warnings.warn(
            f"Moderate memory usage in {operation_name}: ~{estimated_mb:.1f}MB estimated. "
            f"Monitor system memory.",
            UserWarning,
            stacklevel=2
        )

def step_a_vectorized(dvbox, dv_sample, Points_A, Points_B, dim, weight):
    """
    Performs Step A for computing the maximal Solution Space using vectorized operations.
    
    Args:
        dvbox: Current intervals of DVs (dim, 2).
        dv_sample: Sampled DVs (dim, N).
        Points_A: Boolean mask for good points (N,).
        Points_B: Boolean mask for bad points (N,).
        dim: Number of DVs.
        weight: Weights of the DVs (dim,).
        
    Returns:
        dvbox: Updated intervals.
        mu: Size of the solution space.
    """
    
    ind_A = np.where(Points_A)[0]
    ind_B = np.where(Points_B)[0]
    
    num_A = len(ind_A)
    num_B = len(ind_B)
    
    if num_A == 0:
        return dvbox, 0.0
    
    if num_B == 0:
        # No bad points, return full box
        diffs = dvbox[:, 1] - dvbox[:, 0]
        log_mu = np.sum(np.log(np.maximum(weight * diffs, 1e-20)))
        mu = np.exp(log_mu)  # Convert back to actual volume for return
        return dvbox, mu

    # Extract samples
    samples_A = dv_sample[:, ind_A] # (dim, num_A)
    samples_B = dv_sample[:, ind_B] # (dim, num_B)
    
    # Pre-calculate costs for all possible cuts
    # We need to know for each dimension and each bad point, 
    # what is the cost (number of good points lost) if we cut there.
    
    # Cost matrices: (dim, num_B)
    # cost_keep_left: cost if we keep the range [-inf, B] (i.e., A < B)
    # cost_keep_right: cost if we keep the range [B, inf] (i.e., A > B)
    
    cost_keep_left = np.zeros((dim, num_B))
    cost_keep_right = np.zeros((dim, num_B))
    
    for d in range(dim):
        # Sort good points in this dimension for fast counting
        sorted_A_d = np.sort(samples_A[d, :])
        
        # Find positions of bad points in the sorted good points array
        # searchsorted returns indices where elements should be inserted to maintain order
        # indices[j] = number of good points < samples_B[d, j]
        indices = np.searchsorted(sorted_A_d, samples_B[d, :])
        
        # If we keep left (A < B), we lose points where A >= B
        # Number of points < B is 'indices'.
        # Number of points >= B is num_A - indices.
        # Wait, strict inequality check in original code:
        # if val_A < val_B: N_i = sum(good_vals > val_B) -> Lost points are those > B (strictly)
        # searchsorted(side='left') (default): a[i-1] < v <= a[i]
        # indices is count of A < B.
        # So A >= B count is num_A - indices.
        # Original code: sum(good_vals > val_B).
        # If we use searchsorted with side='right': a[i-1] <= v < a[i]
        # indices_right is count of A <= B.
        # num_A - indices_right is count of A > B.
        
        indices_right = np.searchsorted(sorted_A_d, samples_B[d, :], side='right')
        cost_keep_left[d, :] = num_A - indices_right
        
        # If we keep right (A > B), we lose points where A < B (strictly? original code says <)
        # Original code: else (val_A >= val_B): N_i = sum(good_vals < val_B)
        # indices (left) is count of A < B.
        cost_keep_right[d, :] = indices

    # Apply weights
    # (dim, num_B)
    weighted_cost_keep_left = cost_keep_left * weight[:, np.newaxis]
    weighted_cost_keep_right = cost_keep_right * weight[:, np.newaxis]
    
    # Batch processing to avoid O(N^2) memory explosion
    chunk_size = 500  # Process 500 good points at a time for better vectorization
    best_mu = -1
    best_log_mu = -np.inf  # Initialize to negative infinity for log comparison
    best_box = dvbox.copy()
    
    for start in range(0, num_A, chunk_size):
        end = min(start + chunk_size, num_A)
        chunk_size_actual = end - start
        
        # Check memory usage before creating large arrays
        _check_memory_usage(dim, chunk_size_actual, num_B, "step_a_vectorized chunk processing")
        
        # Extract chunk of A
        samples_A_chunk = samples_A[:, start:end]  # (dim, chunk_size)
        
        # Initialize candidate boxes for this chunk
        dvbox_A_chunk = np.repeat(dvbox[:, :, np.newaxis], chunk_size_actual, axis=2)  # (dim, 2, chunk_size)
        
        # Expand A_chunk to (dim, chunk_size, num_B)
        A_exp = samples_A_chunk[:, :, np.newaxis]
        # Expand B to (dim, chunk_size, num_B) - B is already (dim, num_B)
        B_exp = samples_B[:, np.newaxis, :]
        
        # Determine direction: True if A < B (keep left), False if A >= B (keep right)
        mask_keep_left = A_exp < B_exp
        
        # Select costs based on direction
        costs = np.where(mask_keep_left, 
                         weighted_cost_keep_left[:, np.newaxis, :], 
                         weighted_cost_keep_right[:, np.newaxis, :])
        
        # Find best dimension to cut for each pair (A_chunk, B)
        best_dims = np.argmin(costs, axis=0)  # (chunk_size, num_B)
        
        # Create a mask for best dimensions (memory efficient)
        dim_indices = np.arange(dim)[:, np.newaxis, np.newaxis]
        is_best_dim = np.zeros((dim, chunk_size_actual, num_B), dtype=bool)
        np.equal(dim_indices, best_dims[np.newaxis, :, :], out=is_best_dim)
        
        # Identify cuts using in-place operations
        mask_update_upper = np.zeros((dim, chunk_size_actual, num_B), dtype=bool)
        np.logical_and(is_best_dim, mask_keep_left, out=mask_update_upper)
        
        mask_update_lower = np.zeros((dim, chunk_size_actual, num_B), dtype=bool)
        mask_keep_right = np.logical_not(mask_keep_left)
        np.logical_and(is_best_dim, mask_keep_right, out=mask_update_lower)
        
        # Values to apply - use relative epsilon based on variable ranges
        range_vec = dvbox[:, 1] - dvbox[:, 0]  # Range for each dimension
        eps_vec = range_vec * 1e-7  # Relative epsilon (0.00001% of range)
        cut_vals_upper = B_exp - eps_vec[:, np.newaxis, np.newaxis]
        cut_vals_lower = B_exp + eps_vec[:, np.newaxis, np.newaxis]
        
        # Upper Bounds
        vals_upper = np.where(mask_update_upper, cut_vals_upper, np.inf)
        min_cuts_upper = np.min(vals_upper, axis=2)  # (dim, chunk_size)
        
        # Lower Bounds
        vals_lower = np.where(mask_update_lower, cut_vals_lower, -np.inf)
        max_cuts_lower = np.max(vals_lower, axis=2)  # (dim, chunk_size)
        
        # Update the boxes
        current_lb = dvbox_A_chunk[:, 0, :]
        current_ub = dvbox_A_chunk[:, 1, :]
        
        new_lb = np.maximum(current_lb, max_cuts_lower)
        new_ub = np.minimum(current_ub, min_cuts_upper)
        
        # Calculate log-volumes to avoid underflow
        diffs = new_ub - new_lb
        diffs = np.maximum(diffs, 1e-20)  # Avoid log(0)
        
        # Filter out zero-weighted dimensions to avoid -inf in log calculation
        active_dims = weight > 0
        if np.any(active_dims):
            log_vols = np.sum(np.log(diffs[active_dims, :]), axis=0)  # (chunk_size,)
        else:
            # Fallback if all weights are zero (shouldn't happen in practice)
            log_vols = np.zeros(chunk_size_actual)
        
        # Check if this chunk has a better box
        chunk_best_idx = np.argmax(log_vols)
        chunk_best_log_mu = log_vols[chunk_best_idx]
        
        if chunk_best_log_mu > best_log_mu:
            best_log_mu = chunk_best_log_mu
            best_mu = np.exp(chunk_best_log_mu)  # Convert back for return value
            best_box[:, 0] = new_lb[:, chunk_best_idx]
            best_box[:, 1] = new_ub[:, chunk_best_idx]
    
    return best_box, best_mu

