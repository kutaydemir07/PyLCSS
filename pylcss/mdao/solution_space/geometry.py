# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Volume of a union of axis-aligned boxes, without double-counting overlap."""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .contracts import FloatArray


def relative_union_volume(
    bounds: Sequence[FloatArray], design_widths: FloatArray
) -> float:
    if not bounds:
        return 0.0
    active = np.asarray(design_widths) > 0.0
    if not active.any():
        return 1.0
    boxes = np.asarray(bounds, dtype=float)[:, active, :]
    boxes = boxes[np.all(boxes[:, :, 1] > boxes[:, :, 0], axis=1)]
    if not boxes.size:
        return 0.0
    boxes = boxes / np.asarray(design_widths)[active][None, :, None]

    @lru_cache(maxsize=None)
    def sweep(indices: tuple[int, ...], axes: tuple[int, ...]) -> float:
        if not axes:
            return 1.0
        selected = boxes[list(indices)]
        if len(indices) == 1:
            widths = selected[0, list(axes), 1] - selected[0, list(axes), 0]
            return float(np.prod(widths))
        axis = min(axes, key=lambda dim: np.unique(selected[:, dim, :]).size)
        remaining = tuple(dim for dim in axes if dim != axis)
        boundaries = np.unique(selected[:, axis, :])
        volume = 0.0
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            covered = tuple(
                i
                for i in indices
                if boxes[i, axis, 0] <= lower and boxes[i, axis, 1] >= upper
            )
            if covered:
                volume += (upper - lower) * sweep(covered, remaining)
        return float(volume)

    return sweep(tuple(range(len(boxes))), tuple(range(boxes.shape[1])))
