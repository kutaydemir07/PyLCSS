# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Backward-compatible imports for solver backend helpers.

New code should import from :mod:`base`, :mod:`execution`, :mod:`mesh`, or
:mod:`selection`. This façade preserves existing PyLCSS project extensions
that imported the former all-purpose ``common`` module.
"""

from pylcss.input_values import as_bool, flatten_inputs
from pylcss.solver_backends.base import (
    ExternalRunConfig,
    SolverBackendError,
    validate_job_name,
)
from pylcss.solver_backends.execution import (
    make_work_dir,
    resolve_executable,
    run_process,
    tail,
)
from pylcss.solver_backends.mesh import (
    id_lines,
    is_shell_mesh,
    load_vector,
    mesh_to_shell,
    mesh_to_tet4,
    mesh_to_tet10,
    tet10_connectivity,
)
from pylcss.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
    normalize_geometries,
    tet_face_sets_for_geometries,
)

__all__ = [
    "ExternalRunConfig",
    "SolverBackendError",
    "as_bool",
    "dict_geometries",
    "flatten_inputs",
    "id_lines",
    "is_shell_mesh",
    "load_vector",
    "make_work_dir",
    "mesh_to_shell",
    "mesh_to_tet4",
    "mesh_to_tet10",
    "nodes_matching_condition",
    "nodes_matching_geometries",
    "normalize_geometries",
    "resolve_executable",
    "run_process",
    "tail",
    "tet10_connectivity",
    "tet_face_sets_for_geometries",
    "validate_job_name",
]
