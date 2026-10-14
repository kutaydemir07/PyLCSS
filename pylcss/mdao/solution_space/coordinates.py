# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Mapping between active design coordinates and full model inputs."""

from dataclasses import dataclass

import numpy as np

from .contracts import FloatArray, IntArray


@dataclass(frozen=True)
class VariableLayout:
    parameters: FloatArray

    @classmethod
    def from_parameters(cls, parameters: FloatArray) -> "VariableLayout":
        values = np.asarray(parameters, dtype=float).copy()
        if values.ndim != 2 or values.shape[0] != 2:
            raise ValueError("parameters must have shape (2, n_total_variables)")
        if np.any(np.isnan(values[0]) != np.isnan(values[1])):
            raise ValueError(
                "each variable needs two parameter bounds or two NaN values"
            )
        declared = ~np.isnan(values[0])
        if not np.all(np.isfinite(values[:, declared])):
            raise ValueError("parameter bounds must be finite")
        if np.any(values[0, declared] > values[1, declared]):
            raise ValueError("parameter bounds contain an inverted interval")
        return cls(values)

    @classmethod
    def from_bounds(
        cls, lower: FloatArray, upper: FloatArray, parameters: FloatArray | None
    ) -> "VariableLayout":
        count = lower.size
        values = (
            np.full((2, count), np.nan)
            if parameters is None or np.asarray(parameters).size == 0
            else np.asarray(parameters, dtype=float).copy()
        )
        if values.shape != (2, count):
            raise ValueError(f"parameters must have shape (2, {count})")
        layout = cls.from_parameters(values)
        # A narrow interval remains an active variable; only equal endpoints fix it.
        fixed = lower == upper
        layout.parameters[:, fixed] = np.vstack((lower[fixed], upper[fixed]))
        return layout

    @property
    def active_mask(self) -> np.ndarray:
        return np.isnan(self.parameters[0])

    @property
    def active_indices(self) -> IntArray:
        return np.flatnonzero(self.active_mask)

    @property
    def parameter_indices(self) -> IntArray:
        return np.flatnonzero(~self.active_mask)

    def active_points(self, points: FloatArray) -> FloatArray:
        if points.shape[0] == self.parameters.shape[1]:
            return points[self.active_mask]
        if points.shape[0] == self.active_indices.size:
            return points
        raise ValueError("point rows must match either active or all model variables")

    def expand_points(
        self, active_points: FloatArray, parameter_values: FloatArray | None = None
    ) -> FloatArray:
        points = np.asarray(active_points, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        full = np.empty((self.parameters.shape[1], points.shape[1]))
        full[self.active_mask] = points
        indices = self.parameter_indices
        if indices.size:
            full[indices] = (
                self.parameters[0, indices, None]
                if parameter_values is None
                else parameter_values
            )
        return full

    def expand_bounds(self, active_bounds: FloatArray) -> FloatArray:
        full = self.parameters.T.copy()
        full[self.active_mask] = active_bounds
        return full
