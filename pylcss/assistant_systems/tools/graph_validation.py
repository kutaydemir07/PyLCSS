# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility facade for graph validation entry points."""

from pylcss.assistant_systems.tools.cad_graph_validation import (
    run_cad_verified,
    verify_cad_graph,
)
from pylcss.assistant_systems.tools.system_graph_validation import (
    run_system_verified,
    verify_system_graph,
)

__all__ = [
    "run_cad_verified",
    "run_system_verified",
    "verify_cad_graph",
    "verify_system_graph",
]
