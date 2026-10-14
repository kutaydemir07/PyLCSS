# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Axis-aligned robust solution-space computation.

The package keeps numerical algorithms independent of Qt. UI workers live in
``pylcss.mdao.user_interface.solution_space``.
"""

from .api import compute_solution_space
from .bayesian import good_fraction_lower_bound
from .box_optimization import (
    BoxOptimizationResult,
    optimize_box,
)
from .modal_optimization import optimize_modal_boxes
from .models import (
    BoxSolutionSpace,
    DecoupledMultiModalForm,
    MMSSParameters,
    MultiModalResult,
)
from .multimodal import MultiModalSolutionSpaceSolver
from .resampling import (
    resample_multimodal_solution_spaces,
    resample_solution_space,
)
from .sampling import (
    classify_good_bad,
    draw_samples,
    sample_and_classify,
    sample_box_batch,
)
from .solver import SolutionSpaceSolver
from .step_a import trim_box
from .step_b import expand_box
from .validation import minimum_all_success_sample_size

__all__ = [
    "BoxOptimizationResult",
    "BoxSolutionSpace",
    "DecoupledMultiModalForm",
    "MMSSParameters",
    "MultiModalResult",
    "MultiModalSolutionSpaceSolver",
    "SolutionSpaceSolver",
    "classify_good_bad",
    "compute_solution_space",
    "draw_samples",
    "expand_box",
    "good_fraction_lower_bound",
    "minimum_all_success_sample_size",
    "optimize_box",
    "optimize_modal_boxes",
    "resample_multimodal_solution_spaces",
    "resample_solution_space",
    "sample_and_classify",
    "sample_box_batch",
    "trim_box",
]
