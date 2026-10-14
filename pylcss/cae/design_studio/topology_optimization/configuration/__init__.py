# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""User-facing defaults and study configuration."""

from .presets import (
    INDUSTRIAL_DESIGN_GOALS,
    industrial_topopt_defaults,
    normalize_design_goal,
)

__all__ = [
    "INDUSTRIAL_DESIGN_GOALS",
    "industrial_topopt_defaults",
    "normalize_design_goal",
]
