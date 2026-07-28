# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt adapters for topology studies."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "TopologyHeatLoadNode",
    "TopologyJointNode",
    "TopologyLoadNode",
    "TopologyNonDesignRegionNode",
    "TopologyOperatingCaseNode",
    "TopologyOptVoxelNode",
    "TopologySupportNode",
    "TopologyThermalSinkNode",
]

_STUDY_NODES = (
    "pylcss.design_studio.topology_optimization.integration.study_definition_nodes"
)
_LAZY_EXPORTS = {
    "TopologyHeatLoadNode": (_STUDY_NODES, "TopologyHeatLoadNode"),
    "TopologyJointNode": (_STUDY_NODES, "TopologyJointNode"),
    "TopologyLoadNode": (_STUDY_NODES, "TopologyLoadNode"),
    "TopologyNonDesignRegionNode": (
        _STUDY_NODES,
        "TopologyNonDesignRegionNode",
    ),
    "TopologyOperatingCaseNode": (
        _STUDY_NODES,
        "TopologyOperatingCaseNode",
    ),
    "TopologyOptVoxelNode": (
        "pylcss.design_studio.topology_optimization.integration.topology_node",
        "TopologyOptVoxelNode",
    ),
    "TopologySupportNode": (_STUDY_NODES, "TopologySupportNode"),
    "TopologyThermalSinkNode": (_STUDY_NODES, "TopologyThermalSinkNode"),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
