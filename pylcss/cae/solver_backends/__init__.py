# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""External solver backend adapters for PyLCSS simulation nodes.

The CAD node graph owns pre-processing and visualization.  Backend adapters
translate the already-built PyLCSS mesh/material/load dictionaries into solver
input artifacts and, where practical, launch the external executable.
"""

from pylcss.cae.solver_backends.base import ExternalRunConfig, SolverBackendError
from pylcss.cae.solver_backends.calculix import (
    run_calculix_assembly,
    run_calculix_static,
)
from pylcss.cae.solver_backends.opencourant import (
    run_opencourant_crash,
    run_opencourant_existing_deck,
)

__all__ = [
    "ExternalRunConfig",
    "SolverBackendError",
    "run_calculix_assembly",
    "run_calculix_static",
    "run_opencourant_crash",
    "run_opencourant_existing_deck",
]
