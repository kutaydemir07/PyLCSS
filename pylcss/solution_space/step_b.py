# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

import numpy as np


def modification_step_b(
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
    width = dsu - dsl
    expansion = growth_rate * width
    new_low = np.maximum(dsl, bounds[:, 0] - expansion)
    new_up = np.minimum(dsu, bounds[:, 1] + expansion)
    return np.column_stack((new_low, new_up))
