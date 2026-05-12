# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""External solver backend adapters for PyLCSS simulation nodes.

The CAD node graph owns pre-processing and visualization.  Backend adapters
translate the already-built PyLCSS mesh/material/load dictionaries into solver
input artifacts and, where practical, launch the external executable.
"""

from pylcss.solver_backends.common import ExternalRunConfig, SolverBackendError
from pylcss.solver_backends.calculix import (
    run_calculix_static,
    run_calculix_topopt_iteration,
)
from pylcss.solver_backends.openradioss import (
    run_openradioss_crash,
    run_openradioss_existing_deck,
)

__all__ = [
    "ExternalRunConfig",
    "SolverBackendError",
    "run_calculix_static",
    "run_calculix_topopt_iteration",
    "run_openradioss_crash",
    "run_openradioss_existing_deck",
]
