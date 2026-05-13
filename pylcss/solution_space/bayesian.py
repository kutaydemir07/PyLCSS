# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from scipy.stats import beta as _beta


def good_fraction_lower_bound(m: int, N: int, confidence: float = 0.95) -> float:
    """Lower bound ``a_l`` on the true good fraction at the given confidence.

    Solves ``P(a > a_l | m, N) = confidence`` using the Beta posterior with
    uniform prior. Returns 0.0 when ``N == 0``.
    """
    if N <= 0:
        return 0.0
    return float(_beta.ppf(1.0 - confidence, m + 1, N - m + 1))
