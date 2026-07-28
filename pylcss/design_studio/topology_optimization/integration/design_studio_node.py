# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility facade for the former topology-node module name.

New code should import from
:mod:`pylcss.design_studio.topology_optimization.integration.topology_node`.
"""

from __future__ import annotations

from pylcss.design_studio.topology_optimization.integration import (
    topology_node as _topology_node,
)

TopologyOptVoxelNode = _topology_node.TopologyOptVoxelNode

__all__ = ["TopologyOptVoxelNode"]


def __getattr__(name: str) -> object:
    return getattr(_topology_node, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_topology_node)))
