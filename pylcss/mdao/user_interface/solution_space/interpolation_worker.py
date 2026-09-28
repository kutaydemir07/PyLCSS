# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Background categorical interpolation for solution-space plots."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class InterpolationThread(QtCore.QThread):
    """Produce a quick preview followed by the requested full-resolution grid."""

    finished = QtCore.Signal(object)
    quick_result = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        points,
        color_indices,
        x_grid,
        y_grid,
        generation_id=None,
    ) -> None:
        super().__init__()
        self.points = np.asarray(points, dtype=float).copy()
        self.color_indices = np.asarray(color_indices, dtype=int).copy()
        self.x_grid = np.asarray(x_grid, dtype=float).copy()
        self.y_grid = np.asarray(y_grid, dtype=float).copy()
        self.generation_id = id(self) if generation_id is None else generation_id
        self.cancelled = False
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (n_samples, 2)")
        if self.points.shape[0] == 0:
            raise ValueError("at least one interpolation point is required")
        if self.color_indices.shape != (self.points.shape[0],):
            raise ValueError("color_indices must contain one label per point")
        if np.any(self.color_indices < 0):
            raise ValueError("color_indices must not contain negative labels")
        if self.x_grid.shape != self.y_grid.shape or self.x_grid.size == 0:
            raise ValueError("x_grid and y_grid must be non-empty and equally shaped")
        if not all(
            np.all(np.isfinite(array))
            for array in (self.points, self.x_grid, self.y_grid)
        ):
            raise ValueError("interpolation inputs must contain only finite values")

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self.cancelled = True

    def run(self) -> None:
        try:
            if self.cancelled:
                return
            quick = self._categorical_interpolation(grid_size=100, neighbors=3)
            if not self.cancelled:
                self.quick_result.emit(quick)
            interpolated = self._high_res_categorical_interpolation()
            if not self.cancelled:
                self.finished.emit(interpolated)
        except Exception as exc:
            logger.exception("Solution-space interpolation failed")
            if not self.cancelled:
                self.error.emit(str(exc))

    def _categorical_interpolation(
        self,
        points=None,
        color_indices=None,
        x_grid=None,
        y_grid=None,
        grid_size: int = 100,
        neighbors: int = 3,
    ):
        """Interpolate labels on a uniform preview grid."""
        del points, color_indices, x_grid, y_grid
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        x_axis = np.linspace(self.x_grid.min(), self.x_grid.max(), grid_size)
        y_axis = np.linspace(self.y_grid.min(), self.y_grid.max(), grid_size)
        target_x, target_y = np.meshgrid(x_axis, y_axis)
        return self._interpolate(target_x, target_y, neighbors)

    def _high_res_categorical_interpolation(
        self,
        points=None,
        color_indices=None,
        x_grid=None,
        y_grid=None,
    ):
        """Interpolate labels on the caller-provided grid."""
        del points, color_indices, x_grid, y_grid
        return self._interpolate(self.x_grid, self.y_grid, neighbors=5)

    def _interpolate(self, target_x, target_y, neighbors: int):
        if self.cancelled:
            return None

        x_min, x_max = float(self.x_grid.min()), float(self.x_grid.max())
        y_min, y_max = float(self.y_grid.min()), float(self.y_grid.max())
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0
        normalized_points = np.column_stack(
            (
                (self.points[:, 0] - x_min) / x_range,
                (self.points[:, 1] - y_min) / y_range,
            )
        )
        normalized_targets = np.column_stack(
            (
                (target_x.ravel() - x_min) / x_range,
                (target_y.ravel() - y_min) / y_range,
            )
        )
        neighbor_count = min(neighbors, self.points.shape[0])
        _distance, indices = cKDTree(normalized_points).query(
            normalized_targets,
            k=neighbor_count,
        )
        indices = np.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
        neighbor_labels = self.color_indices[indices]
        label_count = int(self.color_indices.max()) + 1
        labels = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=label_count).argmax(),
            axis=1,
            arr=neighbor_labels,
        )
        return labels.reshape(target_x.shape)


__all__ = ["InterpolationThread"]
