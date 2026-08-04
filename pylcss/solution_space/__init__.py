# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Axis-aligned robust solution-space computation.

The package keeps numerical algorithms independent of Qt. UI workers live in
``pylcss.user_interface.solution_space``.
"""

from .api import compute_solution_space
from .bayesian import good_fraction_lower_bound
from .box_optimization import (
    BoxOptimizationResult,
    optimize_box,
)
from .models import (
    BoxSolutionSpace,
    DecoupledMultiModalForm,
    MMSSParameters,
    MultiModalResult,
    SharedIntervalFamily,
)
from .multimodal import MultiModalSolutionSpaceSolver
from .modal_optimization import optimize_modal_boxes
from .resampling import (
    resample_multimodal_solution_spaces,
    resample_solution_space,
)
from .sampling import (
    classify_good_bad,
    draw_samples,
    monte_carlo,
    sample_and_classify,
    sample_box,
)
from .solver import SolutionSpaceSolver
from .step_a import modification_step_a, step_a_vectorized, trim_box
from .step_b import expand_box, modification_step_b
from .validation import minimum_all_success_sample_size

# Source-compatible aliases for the pre-2.2 API.
SolutionSpaceResult = BoxOptimizationResult
compute_phase_solution_space = optimize_box
run_multimodal = optimize_modal_boxes

__all__ = [
    "BoxOptimizationResult",
    "BoxSolutionSpace",
    "DecoupledMultiModalForm",
    "MMSSParameters",
    "MultiModalResult",
    "MultiModalSolutionSpaceSolver",
    "SharedIntervalFamily",
    "SolutionSpaceResult",
    "SolutionSpaceSolver",
    "classify_good_bad",
    "compute_phase_solution_space",
    "compute_solution_space",
    "draw_samples",
    "expand_box",
    "good_fraction_lower_bound",
    "minimum_all_success_sample_size",
    "modification_step_a",
    "modification_step_b",
    "monte_carlo",
    "optimize_box",
    "optimize_modal_boxes",
    "resample_multimodal_solution_spaces",
    "resample_solution_space",
    "run_multimodal",
    "sample_and_classify",
    "sample_box",
    "step_a_vectorized",
    "trim_box",
]
