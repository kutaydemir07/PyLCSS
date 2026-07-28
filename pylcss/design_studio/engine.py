# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Dependency-aware graph execution with caching and preview filtering."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
import hashlib
import inspect
import logging
import pickle
from pylcss.design_studio.core.contracts import (
    CancelCallback,
    GraphLike,
    NodeLike,
    NodeResult,
)
from pylcss.design_studio.core.graph import (
    PREVIEW_SAFE_IDENTIFIERS,
    SIMULATION_NODE_IDENTIFIERS,
    build_execution_plan as _build_execution_plan,
    filter_for_preview as _filter_for_preview,
    input_ports as _input_ports,
    node_name as _node_name,
    source_nodes as _source_nodes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GraphExecutionCancelled",
    "PREVIEW_SAFE_IDENTIFIERS",
    "SIMULATION_NODE_IDENTIFIERS",
    "execute_graph",
]


class GraphExecutionCancelled(RuntimeError):
    """Raised when a graph worker stops between safe node boundaries."""

    def __init__(
        self,
        results: Mapping[NodeLike, NodeResult] | None = None,
    ) -> None:
        """Capture results completed before cancellation."""
        super().__init__("Graph execution cancelled by the user.")
        self.results = dict(results or {})


_NON_EXECUTION_PROPERTIES = {
    # Viewer settings update cached results without changing the solve.
    "visualization",
    "deformation_scale",
    "disp_scale",
    "density_cutoff",
    "advanced_settings_visible",
    "cad_export_filename",
    "cad_reconstruction_method",
    # Compatibility-only properties retained for old project files.
    "solver_backend",
    "moment_x",
    "moment_y",
    "moment_z",
    "damping_alpha",
    "damping_beta",
    "enable_corotation",
    "enable_contact",
    "contact_stiffness",
    "contact_thickness",
    "contact_update_interval",
    "mass_scaling_threshold",
    "bc_preset",
    "quality_preset",
    # Internal UI/error bookkeeping.
    "error_state",
    "error_message",
}


def _hash_value(value: object) -> str:
    """Create a compact, process-local digest for change detection."""
    try:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        data = repr(value).encode("utf-8", errors="replace")
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _execution_properties(node: NodeLike) -> dict[str, object]:
    """Return custom properties that can affect ``node.run()``."""
    properties = getattr(node, "properties", None)
    if not callable(properties):
        return {}
    try:
        raw_properties = properties()
    except Exception:
        logger.debug("Could not read node properties", exc_info=True)
        return {}
    if not isinstance(raw_properties, Mapping):
        return {}
    custom = raw_properties.get("custom", {})
    if not isinstance(custom, Mapping):
        return {}
    return {
        str(key): value
        for key, value in custom.items()
        if key not in _NON_EXECUTION_PROPERTIES
    }


def _set_execution_state(
    node: NodeLike,
    *,
    result: NodeResult,
    input_hash: str | None,
    dirty: bool,
    force_execute: bool = False,
) -> None:
    """Update dynamic cache fields in one documented place."""
    node._last_result = result  # type: ignore[attr-defined]
    node._last_input_hash = input_hash  # type: ignore[attr-defined]
    node._dirty = dirty  # type: ignore[attr-defined]
    node._force_execute = force_execute  # type: ignore[attr-defined]


def _set_node_error(node: NodeLike, message: str) -> None:
    setter = getattr(node, "set_error", None)
    if not callable(setter):
        return
    try:
        setter(message)
    except Exception:
        logger.exception("Could not set error state on %s", _node_name(node))


def _clear_node_error(node: NodeLike) -> None:
    clear = getattr(node, "clear_error", None)
    if not callable(clear):
        return
    try:
        clear()
    except Exception:
        logger.debug(
            "Could not clear error state on %s",
            _node_name(node),
            exc_info=True,
        )


def _reported_node_error(node: NodeLike) -> str | None:
    has_error = getattr(node, "has_error", None)
    if not callable(has_error):
        return None
    try:
        if not has_error():
            return None
    except Exception:
        logger.debug(
            "Could not inspect error state on %s",
            _node_name(node),
            exc_info=True,
        )
        return None

    get_error = getattr(node, "get_error", None)
    if callable(get_error):
        try:
            return str(get_error() or "Node reported an execution error.")
        except Exception:
            pass
    return "Node reported an execution error."


def _invalidate_downstream_cache(
    node: NodeLike,
    reverse_dependencies: Mapping[NodeLike, set[NodeLike]],
) -> None:
    """Clear cached results affected by an upstream execution failure."""
    pending = deque(reverse_dependencies.get(node, ()))
    visited: set[NodeLike] = set()
    while pending:
        downstream = pending.popleft()
        if downstream in visited:
            continue
        visited.add(downstream)
        _set_execution_state(
            downstream,
            result=None,
            input_hash=None,
            dirty=True,
        )
        _clear_node_error(downstream)
        pending.extend(reverse_dependencies.get(downstream, ()))


def _input_signature(node: NodeLike) -> list[tuple[int, str, int]]:
    """Describe upstream result revisions for cache invalidation."""
    signature: list[tuple[int, str, int]] = []
    for input_port in _input_ports(node):
        connected_ports = getattr(input_port, "connected_ports", None)
        if not callable(connected_ports):
            continue
        for output_port in connected_ports():
            upstream = output_port.node()
            signature.append(
                (
                    id(upstream),
                    str(output_port.name()),
                    int(getattr(upstream, "_result_revision", 0)),
                )
            )
    signature.sort(key=lambda value: (value[0], value[1]))
    return signature


def _accepted_run_kwargs(
    node: NodeLike,
    execution_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Select keyword arguments supported by a node's ``run`` method."""
    try:
        parameters = inspect.signature(node.run).parameters
    except (TypeError, ValueError):
        return {}
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(execution_kwargs)
    return {
        key: value
        for key, value in execution_kwargs.items()
        if key in parameters
    }


def execute_graph(
    graph_or_nodes: GraphLike | Iterable[NodeLike],
    skip_simulation: bool = False,
    **kwargs: object,
) -> dict[NodeLike, NodeResult]:
    """Execute a graph in dependency order.

    Results are cached against upstream revisions and execution-relevant node
    properties. Independent failures are collected; dependent nodes are
    skipped and invalidated. A single aggregated ``RuntimeError`` is raised
    after every runnable branch has completed.
    """
    nodes = _source_nodes(graph_or_nodes)
    if skip_simulation:
        nodes = _filter_for_preview(nodes)
    plan = _build_execution_plan(nodes)

    raw_cancel_callback = kwargs.pop("cancel_callback", None)
    cancel_callback: CancelCallback | None = (
        raw_cancel_callback if callable(raw_cancel_callback) else None
    )
    kwargs.setdefault("preview", bool(skip_simulation))

    results: dict[NodeLike, NodeResult] = {}
    failed_nodes: set[NodeLike] = set()
    errors: list[tuple[str, str]] = []

    for node in plan.order:
        if cancel_callback is not None and cancel_callback():
            raise GraphExecutionCancelled(results)

        failed_upstream = [
            upstream
            for upstream in plan.dependencies[node]
            if upstream in failed_nodes
        ]
        if failed_upstream:
            message = "Skipped because upstream failed: " + ", ".join(
                _node_name(upstream) for upstream in failed_upstream
            )
            _set_execution_state(
                node,
                result=None,
                input_hash=None,
                dirty=True,
            )
            _set_node_error(node, message)
            failed_nodes.add(node)
            continue

        current_input_hash = _hash_value(
            (
                _input_signature(node),
                _execution_properties(node),
                bool(skip_simulation),
            )
        )
        cached_result = getattr(node, "_last_result", None)
        can_skip = (
            current_input_hash
            == getattr(node, "_last_input_hash", None)
            and hasattr(node, "_last_result")
            and cached_result is not None
            and not getattr(node, "_dirty", False)
            and not getattr(node, "_force_execute", False)
        )
        if can_skip:
            results[node] = cached_result
            continue

        try:
            _clear_node_error(node)
            run_kwargs = _accepted_run_kwargs(node, kwargs)
            result = node.run(**run_kwargs) if run_kwargs else node.run()
            reported_error = _reported_node_error(node)
            if reported_error is not None:
                raise RuntimeError(reported_error)

            _set_execution_state(
                node,
                result=result,
                input_hash=current_input_hash,
                dirty=False,
            )
            node._result_revision = (  # type: ignore[attr-defined]
                int(getattr(node, "_result_revision", 0)) + 1
            )
            results[node] = result
        except Exception as exc:
            _set_execution_state(
                node,
                result=None,
                input_hash=None,
                dirty=True,
            )
            _invalidate_downstream_cache(
                node,
                plan.reverse_dependencies,
            )
            _set_node_error(node, str(exc))
            errors.append((_node_name(node), str(exc)))
            failed_nodes.add(node)

    if cancel_callback is not None and cancel_callback():
        raise GraphExecutionCancelled(results)

    if errors:
        details = "; ".join(
            f"{name}: {message}" for name, message in errors[:8]
        )
        if len(errors) > 8:
            details += f"; and {len(errors) - 8} more"
        raise RuntimeError(f"Graph execution failed. {details}")

    return results
