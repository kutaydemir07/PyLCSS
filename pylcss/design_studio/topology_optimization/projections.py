# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Manufacturing-constraint projections applied to the density grid each iteration."""
from __future__ import annotations

import numpy as np

_SYMMETRY_AXES = {'x': 0, 'y': 1, 'z': 2}


def _apply_symmetry(x_3d: np.ndarray, planes: str) -> np.ndarray:
    """Average the density across each requested mirror plane.

    `planes` is a string subset of {'x','y','z'} (e.g. 'y', 'xy', 'xyz').
    Returns a new array; input is not modified.
    """
    if not planes or planes.lower() == 'none':
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
    if not axis or axis.lower() == 'none':
        return x_3d
    ax = _SYMMETRY_AXES.get(axis.lower())
    if ax is None:
        return x_3d
    avg = np.mean(x_3d, axis=ax, keepdims=True)
    return np.broadcast_to(avg, x_3d.shape).copy()


_BUILD_AXIS_MAP = {
    '+x': (0,  1), '-x': (0, -1),
    '+y': (1,  1), '-y': (1, -1),
    '+z': (2,  1), '-z': (2, -1),
}


def _apply_am_overhang(x_3d: np.ndarray, build_axis: str) -> np.ndarray:
    """Project the density field to be self-supporting under a 45° overhang rule.

    Walks layer-by-layer in the build direction.  Each voxel's density is
    capped at the maximum of its 3×3 supporting neighbourhood in the layer
    immediately below.  This is the discretised Langelaar AM filter applied
    as a projection (rather than a differentiable filter inside the network),
    which is good enough for engineering convergence.

    `build_axis` ∈ {'+x','-x','+y','-y','+z','-z'} or 'none'.
    """
    if not build_axis or build_axis.lower() == 'none':
        return x_3d
    key = build_axis.lower()
    if key not in _BUILD_AXIS_MAP:
        return x_3d

    try:
        from scipy.ndimage import maximum_filter
    except ImportError:
        return x_3d

    axis, step = _BUILD_AXIS_MAP[key]
    n_layers = x_3d.shape[axis]
    if n_layers < 2:
        return x_3d

    x = x_3d.copy()
    start = 1 if step > 0 else n_layers - 2
    stop  = n_layers if step > 0 else -1

    # 3×3 max kernel applied in the two in-plane axes only
    kernel = [1, 1, 1]
    kernel[axis] = 1
    for k in range(start, stop, step):
        k_below = k - step
        idx       = [slice(None), slice(None), slice(None)]
        idx_below = [slice(None), slice(None), slice(None)]
        idx[axis]       = k
        idx_below[axis] = k_below

        below = x[tuple(idx_below)]
        # The supporting layer is 2-D; max-filter over 3×3 in-plane.
        max_below = maximum_filter(below, size=3, mode='constant', cval=0.0)
        x[tuple(idx)] = np.minimum(x[tuple(idx)], max_below)
    return x


def _apply_max_member_size(
    x_3d: np.ndarray,
    radius_voxels: float,
    threshold: float = 0.6,
) -> np.ndarray:
    """Cap local average density to enforce a maximum member size.

    For every voxel, computes the mean density in a cubic window of half-size
    `radius_voxels`.  Where this local mean exceeds `threshold`, the voxel
    density is scaled by `threshold / local_mean`, pushing thick members to
    spawn an internal void (an "infill" pattern in the AM literature).
    """
    if radius_voxels is None or radius_voxels <= 0.0:
        return x_3d
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return x_3d

    size = max(3, int(round(2.0 * float(radius_voxels) + 1.0)))
    local_avg = uniform_filter(x_3d, size=size, mode='constant', cval=0.0)
    safe_avg  = np.maximum(local_avg, float(threshold))
    return x_3d * (float(threshold) / safe_avg)


_PATTERN_AXIS_PLANE = {
    'x': (1, 2),  # rotate in the (Y, Z) plane around X
    'y': (0, 2),  # rotate in the (X, Z) plane around Y
    'z': (0, 1),  # rotate in the (X, Y) plane around Z
}


def _apply_pattern_repeat(
    x_3d: np.ndarray,
    n_fold: int,
    axis: str = 'y',
) -> np.ndarray:
    """Average the density with its N-fold rotations around the given axis.

    Produces an N-fold rotationally symmetric density field through the
    domain centre.  N=2 (180°) and N=4 (90°) are common; N=3/5/6/… also
    work but use bilinear interpolation, which may slightly blur features.
    """
    if n_fold is None or int(n_fold) <= 1:
        return x_3d
    axis_key = (axis or '').lower()
    if axis_key not in _PATTERN_AXIS_PLANE:
        return x_3d
    try:
        from scipy.ndimage import rotate
    except ImportError:
        return x_3d

    plane = _PATTERN_AXIS_PLANE[axis_key]
    n     = int(n_fold)
    angle = 360.0 / float(n)

    accum = x_3d.astype(float)
    for k in range(1, n):
        rotated = rotate(
            x_3d, angle * k, axes=plane,
            reshape=False, order=1, mode='constant', cval=0.0,
        )
        accum = accum + rotated
    return accum / float(n)


