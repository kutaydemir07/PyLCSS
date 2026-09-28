# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

import numpy as np


def expand_box(
    bounds: np.ndarray,
    dsl: np.ndarray,
    dsu: np.ndarray,
    growth_rate: float,
) -> np.ndarray:
    """Grow every dimension's interval by ``g * (xu_ds - xl_ds)``, clipped to
    the design-space bounds (Zimmermann & Hoessle 2013, Section 4).

    Args:
        bounds:      (n_dims, 2) current candidate box.
        dsl, dsu:    (n_dims,) lower / upper design-space bounds.
        growth_rate: scalar ``g``, fixed for the run.

    Returns:
        (n_dims, 2) extended box.
    """
    bounds = np.asarray(bounds, dtype=float)
    dsl = np.asarray(dsl, dtype=float)
    dsu = np.asarray(dsu, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dimensions, 2)")
    if dsl.shape != (bounds.shape[0],) or dsu.shape != dsl.shape:
        raise ValueError("design-space bounds must match the box dimensions")
    if not all(np.all(np.isfinite(array)) for array in (bounds, dsl, dsu)):
        raise ValueError("box and design-space bounds must be finite")
    if np.any(bounds[:, 0] > bounds[:, 1]) or np.any(dsl > dsu):
        raise ValueError("box and design-space intervals must be ordered")
    if growth_rate < 0.0:
        raise ValueError("growth_rate must not be negative")

    width = dsu - dsl
    expansion = growth_rate * width
    new_low = np.maximum(dsl, bounds[:, 0] - expansion)
    new_up = np.minimum(dsu, bounds[:, 1] + expansion)
    return np.column_stack((new_low, new_up))


# Compatibility with the paper-oriented name used before PyLCSS 2.2.
modification_step_b = expand_box
