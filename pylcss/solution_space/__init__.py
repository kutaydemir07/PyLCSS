# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from .bayesian import good_fraction_lower_bound
from .computation_engine import compute_solution_space, resample_solution_space
from .compute_solution_space import (
    SolutionSpaceResult,
    compute_solution_space as compute_phase_solution_space,
    run_multimodal,
)
from .multimodal_models import (
    BoxSolutionSpace,
    DecoupledMultiModalForm,
    MMSSParameters,
    MultiModalResult,
    SharedIntervalFamily,
)
from .multimodal_solver import MultiModalSolutionSpaceSolver
from .monte_carlo import classify_good_bad, draw_samples, monte_carlo, sample_and_classify
from .solver_engine import SolutionSpaceSolver
from .step_a import modification_step_a, step_a_vectorized
from .step_b import modification_step_b

__all__ = [
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
    "modification_step_a",
    "modification_step_b",
    "monte_carlo",
    "resample_solution_space",
    "run_multimodal",
    "sample_and_classify",
    "step_a_vectorized",
]
