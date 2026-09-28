# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Structural types at the NodeGraphQt boundary.

NodeGraphQt does not publish complete static types. These protocols document
the small interface used by the headless graph engine without coupling it to
Qt implementation classes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

NodeResult: TypeAlias = Any
CancelCallback: TypeAlias = Callable[[], bool]


class OutputPortLike(Protocol):
    """Output-port operations used when resolving graph connections."""

    def name(self) -> str:
        """Return the port's display name."""

    def node(self) -> NodeLike:
        """Return the node that owns this port."""


class InputPortLike(Protocol):
    """Input-port operations used by the graph engine."""

    def connected_ports(self) -> Sequence[OutputPortLike]:
        """Return connected upstream output ports."""


class NodeLike(Protocol):
    """Minimal executable node interface used outside the GUI."""

    def input_ports(
        self,
    ) -> Iterable[InputPortLike] | Mapping[str, InputPortLike]:
        """Return this node's inputs."""

    def run(self, **kwargs: object) -> NodeResult:
        """Evaluate the node."""


class GraphLike(Protocol):
    """Minimal graph interface accepted by the execution engine."""

    def all_nodes(self) -> Iterable[NodeLike]:
        """Return every node in the graph."""
