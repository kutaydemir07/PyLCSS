# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

import logging
import numpy as np
from scipy.stats import qmc

logger = logging.getLogger(__name__)

class LHSInitializationMixin:
    def _generate_space_filling_starts(self, n_starts: int) -> np.ndarray:
        """
        Generate normalized restart points for basin discovery.

        Latin Hypercube Sampling (LHS) is used exclusively as the default
        and only supported start-sampling method. On failure
        the code falls back to uniform random starts.
        """

        try:
            try:
                sampler = qmc.LatinHypercube(
                    d=self.dim,
                    seed=42,
                    optimization="random-cd",
                )
            except TypeError:
                sampler = qmc.LatinHypercube(d=self.dim, seed=42)
            return sampler.random(n=n_starts)
        except Exception as exc:
            logger.warning(f"LHS start design failed ({exc}); falling back to uniform random starts.")
            rng = np.random.default_rng(42)
            return rng.uniform(0.0, 1.0, (n_starts, self.dim))

