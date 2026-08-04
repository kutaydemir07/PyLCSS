# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Identity of the density-based study nodes.

Topology optimization and lattice optimization are separate graph nodes that
share one execution pipeline, so preflight checks, rendering, persistence and
the assistant catalogue all have to recognize either of them. Holding the
identifiers in one module keeps a new study from leaving stale string
comparisons behind in the UI layer.
"""

from __future__ import annotations

from typing import Any

TOPOLOGY_SOLVER_IDENTIFIER = "com.cad.sim.topopt_voxel"
LATTICE_SOLVER_IDENTIFIER = "com.cad.sim.lattice_voxel"
DENSITY_STUDY_IDENTIFIERS: frozenset[str] = frozenset(
    {TOPOLOGY_SOLVER_IDENTIFIER, LATTICE_SOLVER_IDENTIFIER}
)

TOPOLOGY_SOLVER_CLASS_NAME = "TopologyOptVoxelNode"
LATTICE_SOLVER_CLASS_NAME = "LatticeOptVoxelNode"
DENSITY_STUDY_CLASS_NAMES: frozenset[str] = frozenset(
    {TOPOLOGY_SOLVER_CLASS_NAME, LATTICE_SOLVER_CLASS_NAME}
)


def is_density_study_node(node: Any) -> bool:
    """Return whether *node* is a topology or a lattice optimization study."""
    return getattr(node, "__identifier__", "") in DENSITY_STUDY_IDENTIFIERS


def is_lattice_study_node(node: Any) -> bool:
    """Return whether *node* is the lattice optimization study."""
    return getattr(node, "__identifier__", "") == LATTICE_SOLVER_IDENTIFIER


def is_density_study_class(class_name: Any) -> bool:
    """Return whether a NodeGraphQt class name is one of the study nodes."""
    return str(class_name or "") in DENSITY_STUDY_CLASS_NAMES


def is_lattice_study_record(record: Any) -> bool:
    """Return whether a serialized node record is a lattice study."""
    return bool(
        isinstance(record, dict)
        and str(record.get("type_") or "").endswith(
            f".{LATTICE_SOLVER_CLASS_NAME}"
        )
    )


def is_density_study_record(record: Any) -> bool:
    """Return whether a serialized node record is one of the study nodes."""
    if not isinstance(record, dict):
        return False
    type_name = str(record.get("type_") or "")
    return any(
        type_name.endswith(f".{name}") for name in DENSITY_STUDY_CLASS_NAMES
    )
