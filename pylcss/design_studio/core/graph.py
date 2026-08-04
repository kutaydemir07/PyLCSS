# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Graph discovery, preview filtering, and topological planning."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
import logging

from pylcss.design_studio.core.contracts import (
    GraphLike,
    InputPortLike,
    NodeLike,
)

logger = logging.getLogger(__name__)

# Solver or meshing nodes omitted from automatic lightweight graph updates.
SIMULATION_NODE_IDENTIFIERS = {
    "com.cad.sim.material",
    "com.cad.sim.mesh",
    "com.cad.sim.constraint",
    "com.cad.sim.load",
    "com.cad.sim.pressure_load",
    "com.cad.sim.solver",
    "com.cad.sim.topopt_voxel",
    "com.cad.sim.lattice_voxel",
    "com.cad.sim.impact",
    "com.cad.sim.crash_solver",
    # Standalone decks can launch Starter and Engine with no graph inputs.
}

# Meshing is cheap enough to keep in an interactive preview, where the mesh is
# usually already cached and only the edited property has to propagate. It is
# the dominant cost when a project is opened cold, so the load-time preview
# excludes it: geometry still renders, and the mesh waits for an explicit Run.
MESHING_NODE_IDENTIFIERS = {
    "com.cad.sim.mesh",
}

# These nodes remain useful and bounded during an interactive preview.
PREVIEW_SAFE_IDENTIFIERS = {
    "com.cad.select_face",
    "com.cad.select_face_interactive",
    *MESHING_NODE_IDENTIFIERS,
}


@dataclass(frozen=True)
class ExecutionPlan:
    """Topological order and dependency indexes for one execution."""

    order: tuple[NodeLike, ...]
    dependencies: dict[NodeLike, set[NodeLike]]
    reverse_dependencies: dict[NodeLike, set[NodeLike]]


def input_ports(node: NodeLike) -> tuple[InputPortLike, ...]:
    """Normalize NodeGraphQt's list-or-dictionary input-port API."""
    if not hasattr(node, "input_ports"):
        return ()
    ports = node.input_ports()
    if isinstance(ports, Mapping):
        ports = ports.values()
    return tuple(ports)


def connected_upstream_nodes(node: NodeLike) -> Iterator[NodeLike]:
    """Yield valid nodes connected to any input port."""
    for input_port in input_ports(node):
        connected_ports = getattr(input_port, "connected_ports", None)
        if not callable(connected_ports):
            continue
        for output_port in connected_ports():
            try:
                yield output_port.node()
            except Exception:
                logger.debug(
                    "Ignoring a disconnected or stale output port",
                    exc_info=True,
                )


def _is_simulation_node(node: NodeLike) -> bool:
    return (
        str(getattr(node, "__identifier__", ""))
        in SIMULATION_NODE_IDENTIFIERS
    )


def _is_preview_safe(
    node: NodeLike,
    preview_safe: Collection[str] = PREVIEW_SAFE_IDENTIFIERS,
) -> bool:
    return str(getattr(node, "__identifier__", "")) in preview_safe


def filter_for_preview(
    nodes: Iterable[NodeLike],
    *,
    skip_meshing: bool = False,
) -> list[NodeLike]:
    """Remove heavy simulation nodes and their downstream consumers.

    Mesh, remesh, and selection nodes are explicitly preview-safe so users can
    inspect and select generated geometry without launching solver workflows.
    ``skip_meshing`` tightens that for the one preview the user did not ask
    for — the automatic refresh after a project is opened — where meshing an
    entire model from scratch is indistinguishable from a solve.
    """
    node_list = list(nodes)
    preview_safe = (
        PREVIEW_SAFE_IDENTIFIERS - MESHING_NODE_IDENTIFIERS
        if skip_meshing
        else PREVIEW_SAFE_IDENTIFIERS
    )
    blocked = {
        node
        for node in node_list
        if _is_simulation_node(node) and not _is_preview_safe(node, preview_safe)
    }
    changed = True
    while changed:
        changed = False
        for node in node_list:
            if node in blocked or _is_preview_safe(node, preview_safe):
                continue
            if any(
                upstream in blocked
                for upstream in connected_upstream_nodes(node)
            ):
                blocked.add(node)
                changed = True
    return [node for node in node_list if node not in blocked]


def node_name(node: NodeLike) -> str:
    """Return a useful, exception-safe node name for diagnostics."""
    name = getattr(node, "name", None)
    try:
        value = name() if callable(name) else name
    except Exception:
        value = None
    return str(
        value
        or getattr(node, "NODE_NAME", None)
        or node.__class__.__name__
    )


def source_nodes(
    graph_or_nodes: GraphLike | Iterable[NodeLike],
) -> list[NodeLike]:
    """Return nodes from either a NodeGraphQt graph or an iterable."""
    all_nodes = getattr(graph_or_nodes, "all_nodes", None)
    if callable(all_nodes):
        return list(all_nodes())
    return list(graph_or_nodes)


def build_execution_plan(nodes: Iterable[NodeLike]) -> ExecutionPlan:
    """Build a deterministic Kahn topological order."""
    node_list = list(nodes)
    dependencies = {node: set() for node in node_list}
    reverse_dependencies = {node: set() for node in node_list}

    for node in node_list:
        for upstream in connected_upstream_nodes(node):
            if upstream not in dependencies:
                continue
            dependencies[node].add(upstream)
            reverse_dependencies[upstream].add(node)

    remaining = {
        node: set(upstream)
        for node, upstream in dependencies.items()
    }
    ready = deque(node for node in node_list if not remaining[node])
    order: list[NodeLike] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for downstream in reverse_dependencies[node]:
            remaining[downstream].discard(node)
            if not remaining[downstream]:
                ready.append(downstream)

    if len(order) != len(node_list):
        cyclic = [node for node in node_list if remaining[node]]
        names = [node_name(node) for node in cyclic[:10]]
        suffix = (
            ""
            if len(cyclic) <= 10
            else f" and {len(cyclic) - 10} more"
        )
        raise RuntimeError(
            "Graph contains a dependency cycle involving: "
            + ", ".join(names)
            + suffix
            + ". Remove the feedback connection before running."
        )

    return ExecutionPlan(
        order=tuple(order),
        dependencies=dependencies,
        reverse_dependencies=reverse_dependencies,
    )
