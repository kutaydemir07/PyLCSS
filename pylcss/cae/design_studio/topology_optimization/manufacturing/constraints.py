# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Manufacturing-constraint density projections applied during optimization."""

from __future__ import annotations

import numpy as np

_SYMMETRY_AXES = {"x": 0, "y": 1, "z": 2}


def _apply_symmetry(x_3d: np.ndarray, planes: str) -> np.ndarray:
    """Average the density across each requested mirror plane.

    `planes` is a string subset of {'x','y','z'} (e.g. 'y', 'xy', 'xyz').
    Returns a new array; input is not modified.
    """
    if not planes or planes.lower() == "none":
        return x_3d
    out = x_3d.copy()
    requested = {c for c in planes.lower() if c in _SYMMETRY_AXES}
    for plane in requested:
        axis = _SYMMETRY_AXES[plane]
        flipper = [slice(None), slice(None), slice(None)]
        flipper[axis] = slice(None, None, -1)
        out = 0.5 * (out + out[tuple(flipper)])
    return out


def _apply_extrusion(x_3d: np.ndarray, axis: str) -> np.ndarray:
    """Force uniform density along one axis (extruded prism)."""
    if not axis or axis.lower() == "none":
        return x_3d
    ax = _SYMMETRY_AXES.get(axis.lower())
    if ax is None:
        return x_3d
    avg = np.mean(x_3d, axis=ax, keepdims=True)
    return np.broadcast_to(avg, x_3d.shape).copy()


_BUILD_AXIS_MAP = {
    "+x": (0, 1),
    "-x": (0, -1),
    "+y": (1, 1),
    "-y": (1, -1),
    "+z": (2, 1),
    "-z": (2, -1),
}


_PATTERN_AXIS_PLANE = {
    "x": (1, 2),  # rotate in the (Y, Z) plane around X
    "y": (0, 2),  # rotate in the (X, Z) plane around Y
    "z": (0, 1),  # rotate in the (X, Y) plane around Z
}
