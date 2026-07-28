# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Base class and connection resolvers shared by Design Studio nodes."""

from __future__ import annotations

import logging
from numbers import Real
from typing import TypeGuard, TypeVar

from NodeGraphQt import BaseNode

from pylcss.design_studio.core.contracts import (
    InputPortLike,
    NodeLike,
    NodeResult,
    OutputPortLike,
)

logger = logging.getLogger(__name__)

FallbackT = TypeVar("FallbackT")


def _has_callable(value: object, name: str) -> bool:
    """Return whether a dynamically typed CAD object exposes a method."""
    return callable(getattr(value, name, None))


def is_numeric(value: object) -> TypeGuard[Real]:
    """Return whether *value* is a real scalar but not a boolean."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _connected_result(node: NodeLike) -> NodeResult:
    """Return a connected node result without repeating a completed run.

    A legitimate result can be ``None`` for a side-effect-only node. Only
    nodes that have never been visited by the engine use lazy execution.
    """
    if hasattr(node, "_last_result"):
        return node._last_result  # type: ignore[attr-defined]
    return node.run()


def is_shape(value: object) -> bool:
    """Return whether *value* looks like CadQuery geometry.

    CadQuery intentionally exposes several runtime types without a shared
    public protocol. Requiring a meaningful combination of methods avoids the
    previous false positive where any object with ``add`` (such as ``set``)
    was treated as an assembly.
    """
    if value is None:
        return False

    if _has_callable(value, "toCompound"):
        return True
    if _has_callable(value, "tessellate"):
        return True
    if not _has_callable(value, "val"):
        return False
    return any(
        _has_callable(value, method_name)
        for method_name in ("edges", "extrude", "faces")
    )


def _value_for_connected_output(
    connected_port: OutputPortLike,
    result: NodeResult,
) -> NodeResult:
    """Select a named output when a node returns a result dictionary."""
    if not isinstance(result, dict):
        return result
    try:
        output_name = connected_port.name()
    except Exception:
        logger.debug("Could not read a connected output name", exc_info=True)
        output_name = ""

    # Selection nodes expose a port named "workplane", but downstream
    # engineering nodes need the full payload to retain entity metadata.
    if output_name == "workplane" and "faces" in result:
        return result
    if output_name and output_name in result:
        return result[output_name]
    return result


def _first_connected_port(port: InputPortLike | None) -> OutputPortLike | None:
    if port is None:
        return None
    connected = port.connected_ports()
    return connected[0] if connected else None


def resolve_numeric_input(
    port: InputPortLike | None,
    fallback: FallbackT,
) -> Real | FallbackT | None:
    """Resolve a numeric connection, or return *fallback* when disconnected.

    A connected non-numeric value returns ``None``. Falling back in that case
    would make the visible graph disagree with the value used for computation.
    """
    try:
        connected_port = _first_connected_port(port)
        if connected_port is None:
            return fallback
        result = _connected_result(connected_port.node())
        return result if is_numeric(result) else None
    except Exception:
        logger.debug("Could not resolve numeric node input", exc_info=True)
        return None


def resolve_shape_input(port: InputPortLike | None) -> object | None:
    """Resolve a CadQuery shape or selection payload from one connection."""
    try:
        connected_port = _first_connected_port(port)
        if connected_port is None:
            return None
        result = _value_for_connected_output(
            connected_port,
            _connected_result(connected_port.node()),
        )
        if is_shape(result):
            return result
        if isinstance(result, dict):
            shape = result.get("shape")
            return shape if is_shape(shape) else None
    except Exception:
        logger.debug("Could not resolve shape node input", exc_info=True)
    return None


def resolve_any_input(port: InputPortLike | None) -> NodeResult:
    """Resolve the first connection without constraining its result type."""
    try:
        connected_port = _first_connected_port(port)
        if connected_port is None:
            return None
        return _value_for_connected_output(
            connected_port,
            _connected_result(connected_port.node()),
        )
    except Exception:
        logger.debug("Could not resolve node input", exc_info=True)
        return None


def resolve_all_inputs(port: InputPortLike | None) -> list[NodeResult]:
    """Resolve all successful, non-``None`` values connected to an input."""
    if port is None:
        return []

    results: list[NodeResult] = []
    try:
        connected_ports = port.connected_ports()
    except Exception:
        logger.debug("Could not inspect connected node inputs", exc_info=True)
        return results

    for connected_port in connected_ports:
        try:
            result = _value_for_connected_output(
                connected_port,
                _connected_result(connected_port.node()),
            )
        except Exception:
            logger.debug("Could not resolve one connected input", exc_info=True)
            continue
        if result is not None:
            results.append(result)
    return results


class CadQueryNode(BaseNode):
    """Base node for CAD operations and engineering workflows."""

    __identifier__ = "com.cad"
    NODE_NAME = "Base CAD"

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize shared error state after NodeGraphQt construction."""
        super().__init__(*args, **kwargs)
        self._error_message: str | None = None
        self._error_state = False

    def run(self) -> NodeResult:
        """Evaluate the node; concrete nodes override this method."""
        return None

    def set_error(self, message: object) -> None:
        """Set an error state visible to both the engine and GUI."""
        self._error_message = str(message)
        self._error_state = True
        try:
            self.set_property("error_state", True)
            self.set_property("error_message", self._error_message)
        except Exception:
            logger.debug("Could not update node error properties", exc_info=True)

    def clear_error(self) -> None:
        """Clear the current engine and GUI error state."""
        self._error_message = None
        self._error_state = False
        try:
            self.set_property("error_state", False)
            self.set_property("error_message", "")
        except Exception:
            logger.debug("Could not clear node error properties", exc_info=True)

    def get_error(self) -> str | None:
        """Return the current error message."""
        return self._error_message

    def has_error(self) -> bool:
        """Return whether this node currently reports an error."""
        return self._error_state

    def get_input_value(
        self,
        port_name: str,
        prop_name: str | None = None,
    ) -> NodeResult:
        """Return a connected value, or a named property when disconnected."""
        port = self.get_input(port_name)
        if port is not None and port.connected_ports():
            return resolve_any_input(port)
        return self.get_property(prop_name) if prop_name else None

    def get_input_list(self, port_name: str) -> list[NodeResult]:
        """Return all values connected to a multi-input port."""
        return resolve_all_inputs(self.get_input(port_name))

    def get_input_shape(self, port_name: str) -> object | None:
        """Return geometry connected to *port_name*, if valid."""
        return resolve_shape_input(self.get_input(port_name))
