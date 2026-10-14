# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Identity of the density-based study nodes.

Topology optimization and geometric lattice infill share result consumers.
Cell generation from a completed TopOpt result is an inspector action.
"""

from __future__ import annotations

from typing import Any

TOPOLOGY_SOLVER_IDENTIFIER = "com.cad.sim.topopt_voxel"
LATTICE_INFILL_IDENTIFIER = "com.cad.sim.lattice_infill"
DENSITY_STUDY_IDENTIFIERS: frozenset[str] = frozenset(
    {
        TOPOLOGY_SOLVER_IDENTIFIER,
        LATTICE_INFILL_IDENTIFIER,
    }
)

TOPOLOGY_SOLVER_CLASS_NAME = "TopologyOptVoxelNode"
LATTICE_INFILL_CLASS_NAME = "LatticeInfillNode"
DENSITY_STUDY_CLASS_NAMES: frozenset[str] = frozenset(
    {
        TOPOLOGY_SOLVER_CLASS_NAME,
        LATTICE_INFILL_CLASS_NAME,
    }
)


def is_density_study_node(node: Any) -> bool:
    """Return whether *node* is a topology study or geometric lattice infill."""
    return getattr(node, "__identifier__", "") in DENSITY_STUDY_IDENTIFIERS


def is_lattice_infill_node(node: Any) -> bool:
    """Return whether *node* fills a body with a lattice instead of solving.

    A lattice infill shares the density study's execution pipeline and its
    result payload, so every consumer that recognizes a study has to recognize
    it too. What it does not share is a solve: it has no material, no support
    and no load, so the checks that exist to catch a mis-specified *analysis*
    have nothing to check and must not be applied to it.
    """
    return getattr(node, "__identifier__", "") == LATTICE_INFILL_IDENTIFIER


def is_density_study_class(class_name: Any) -> bool:
    """Return whether a NodeGraphQt class name is one of the study nodes."""
    return str(class_name or "") in DENSITY_STUDY_CLASS_NAMES


def is_lattice_study_record(record: Any) -> bool:
    """Return whether a serialized node record builds a lattice."""
    if not isinstance(record, dict):
        return False
    type_name = str(record.get("type_") or "")
    return any(
        type_name.endswith(f".{name}")
        for name in (LATTICE_INFILL_CLASS_NAME,)
    )


def is_density_study_record(record: Any) -> bool:
    """Return whether a serialized node record is one of the study nodes."""
    if not isinstance(record, dict):
        return False
    type_name = str(record.get("type_") or "")
    return any(
        type_name.endswith(f".{name}") for name in DENSITY_STUDY_CLASS_NAMES
    )
