# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle 
# Computing solution spaces for robust design 
# https://doi.org/10.1002/nme.4450

import numpy as np
from PySide6 import QtCore

class InterpolationThread(QtCore.QThread):
    finished = QtCore.Signal(object)  # interpolated result
    quick_result = QtCore.Signal(object)  # quick low-res result
    error = QtCore.Signal(str)

    def __init__(self, points, color_indices, x_grid, y_grid, generation_id=None):
        super().__init__()
        self.points = points.copy()
        # Ensure color_indices are integers for bincount
        self.color_indices = color_indices.copy().astype(int)
        self.x_grid = x_grid.copy()
        self.y_grid = y_grid.copy()
        self.generation_id = generation_id or id(self)  # Unique ID for this interpolation request
        self.cancelled = False

    def cancel(self):
        """Cancel this interpolation operation."""
        self.cancelled = True

    def run(self):
        try:
            if self.cancelled:
                return
                
            # Quick pass: low resolution (100x100) for immediate feedback
            quick_interpolated = self._categorical_interpolation(
                self.points, self.color_indices, self.x_grid, self.y_grid, grid_size=100
            )
            
            if not self.cancelled:
                self.quick_result.emit(quick_interpolated)
                
            # High-res pass: full resolution for final quality
            interpolated = self._high_res_categorical_interpolation(
                self.points, self.color_indices, self.x_grid, self.y_grid
            )
            
            if not self.cancelled:
                self.finished.emit(interpolated)
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))

    def _categorical_interpolation(self, points, color_indices, x_grid, y_grid, grid_size=100):
        """
        Low-resolution categorical interpolation for quick feedback.
        Uses k-Nearest Neighbors with Majority Voting on NORMALIZED coordinates.
        """
        from scipy.spatial import cKDTree
        
        # Check for cancellation periodically
        if self.cancelled:
            return None
            
        # Create low-res grid
        x_min, x_max = x_grid.min(), x_grid.max()
        y_min, y_max = y_grid.min(), y_grid.max()
        
        # Create uniform grid at specified resolution
        x_grid_low = np.linspace(x_min, x_max, grid_size)
        y_grid_low = np.linspace(y_min, y_max, grid_size)
        grid_x_low, grid_y_low = np.meshgrid(x_grid_low, y_grid_low)
        
        # --- FIX: Normalize Coordinates ---
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        
        # Normalize samples (points)
        points_norm = np.zeros_like(points)
        points_norm[:, 0] = (points[:, 0] - x_min) / x_range
        points_norm[:, 1] = (points[:, 1] - y_min) / y_range
        
        # Normalize grid targets
        grid_x_flat = grid_x_low.ravel()
        grid_y_flat = grid_y_low.ravel()
        
        grid_x_norm = (grid_x_flat - x_min) / x_range
        grid_y_norm = (grid_y_flat - y_min) / y_range
        
        grid_points_norm = np.column_stack((grid_x_norm, grid_y_norm))
        
        if self.cancelled:
            return None
            
        # Build Tree from NORMALIZED samples
        tree = cKDTree(points_norm)
        
        if self.cancelled:
            return None
            
        # Query k-nearest neighbors (smaller k for speed)
        k = 3 
        dist, idx = tree.query(grid_points_norm, k=k)
        
        if self.cancelled:
            return None
            
        # Majority Vote - use fast bincount-based mode
        neighbor_colors = color_indices[idx]
        # Determine num_colors dynamically
        max_color = int(np.max(color_indices)) if len(color_indices) > 0 else 0
        num_colors = max_color + 1
        
        most_common = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=num_colors).argmax(), 
            axis=1, 
            arr=neighbor_colors
        ).reshape(-1, 1) 
        
        if self.cancelled:
            return None
            
        # Reshape
        interpolated = most_common.reshape(grid_x_low.shape)
        
        return interpolated

    def _high_res_categorical_interpolation(self, points, color_indices, x_grid, y_grid):
        """
        Uses k-Nearest Neighbors with Majority Voting on NORMALIZED coordinates.
        Normalization prevents axis scaling artifacts (e.g., "red rows").
        """
        from scipy.spatial import cKDTree
        
        # Check for cancellation periodically
        if self.cancelled:
            return None
            
        # --- FIX: Normalize Coordinates ---
        # 1. Determine ranges
        x_min, x_max = x_grid.min(), x_grid.max()
        y_min, y_max = y_grid.min(), y_grid.max()
        
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        
        # 2. Normalize samples (points)
        # Copy to avoid modifying original data
        points_norm = np.zeros_like(points)
        points_norm[:, 0] = (points[:, 0] - x_min) / x_range
        points_norm[:, 1] = (points[:, 1] - y_min) / y_range
        
        if self.cancelled:
            return None
            
        # 3. Normalize grid targets
        grid_x_flat = x_grid.ravel()
        grid_y_flat = y_grid.ravel()
        
        grid_x_norm = (grid_x_flat - x_min) / x_range
        grid_y_norm = (grid_y_flat - y_min) / y_range
        
        # Stack normalized grid points
        grid_points_norm = np.column_stack((grid_x_norm, grid_y_norm))
        # ----------------------------------
        
        if self.cancelled:
            return None
            
        # 4. Build Tree from NORMALIZED samples
        tree = cKDTree(points_norm)
        
        if self.cancelled:
            return None
            
        # 5. Query k-nearest neighbors
        k = 5 
        dist, idx = tree.query(grid_points_norm, k=k)
        
        if self.cancelled:
            return None
            
        # 6. Majority Vote - use fast bincount-based mode
        neighbor_colors = color_indices[idx]
        # Determine num_colors dynamically
        max_color = int(np.max(color_indices)) if len(color_indices) > 0 else 0
        num_colors = max_color + 1
        
        most_common = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=num_colors).argmax(), 
            axis=1, 
            arr=neighbor_colors
        ).reshape(-1, 1) 
        
        if self.cancelled:
            return None
            
        # 7. Reshape
        interpolated = most_common.reshape(x_grid.shape)
        
        return interpolated






