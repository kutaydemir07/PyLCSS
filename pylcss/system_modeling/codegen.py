# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Low-level helpers shared by system and surrogate source generation."""

from __future__ import annotations

import ast
import keyword
import re
import textwrap
from typing import Any

import networkx as nx

from .types import InputSpec
from .units import UnitError, conversion_parameters, is_specified_unit

INPUT_NODE = "com.pfd.input"
OUTPUT_NODE = "com.pfd.output"
INTERMEDIATE_NODE = "com.pfd.intermediate"
BLOCK_NODE = "com.pfd.custom_block"
RESERVED_NAMES = {
    "BASE_DIR",
    "cad",
    "inputs",
    "joblib",
    "np",
    "os",
    "outputs",
    "results",
    "sys",
}


class GraphCompilationError(ValueError):
    """Raised when a graph cannot be represented as a valid Python model."""


def block_function_source(
    block: Any,
    function_name: str,
    surrogate_name: str | None,
) -> list[str]:
    input_names = [
        require_identifier(port.name(), f"input port on {block.name()!r}")
        for port in block.input_ports()
    ]
    output_names = [
        require_identifier(port.name(), f"output port on {block.name()!r}")
        for port in block.output_ports()
    ]
    duplicates = set(input_names) & set(output_names)
    if duplicates:
        raise GraphCompilationError(
            f"Block {block.name()!r} reuses names as both inputs and outputs: "
            f"{', '.join(sorted(duplicates))}."
        )

    lines = [f"def {function_name}({', '.join(input_names)}):"]
    if surrogate_name is not None:
        if not input_names or not output_names:
            raise GraphCompilationError(
                f"Surrogate block {block.name()!r} needs inputs and outputs."
            )
        lines.extend(_surrogate_body(surrogate_name, input_names, output_names))
    else:
        user_code = block_code(block)
        _require_assigned_outputs(block, user_code, output_names)
        lines.extend(textwrap.indent(user_code, "    ").splitlines())
        if output_names:
            lines.append(f"    return {', '.join(output_names)}")
        else:
            lines.append("    return None")
    lines.append("")
    return lines


def wrapper_source(
    function_name: str,
    core_name: str,
    inputs: list[InputSpec],
) -> list[str]:
    names = [item["name"] for item in inputs]
    arguments = ", ".join(names)
    indexed_arguments = ", ".join(f"_row_{name}" for name in names)
    lines = [
        "",
        f"def {function_name}({arguments}):",
        "    try:",
        f"        return {core_name}({arguments})",
        "    except Exception as _vector_error:",
        f"        _input_values = [{arguments}]",
        "        _lengths = [",
        "            len(value) for value in _input_values if np.ndim(value) > 0",
        "        ]",
        "        if not _lengths:",
        "            raise",
        "        if len(set(_lengths)) != 1:",
        "            raise ValueError(",
        "                f'Vectorized inputs have inconsistent lengths: {_lengths}'",
        "            ) from _vector_error",
        "        _results = {}",
        "        for _index in range(_lengths[0]):",
    ]
    for name in names:
        lines.append(
            f"            _row_{name} = "
            f"{name}[_index] if np.ndim({name}) > 0 else {name}"
        )
    lines.extend(
        [
            f"            _point = {core_name}({indexed_arguments})",
            "            for _key, _value in _point.items():",
            "                _results.setdefault(_key, []).append(_value)",
            "        return {",
            "            key: np.asarray(values) for key, values in _results.items()",
            "        }",
        ]
    )
    return lines


def topological_nodes(nodes: list[Any]) -> list[Any]:
    graph = nx.DiGraph()
    node_by_id = {node.id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise GraphCompilationError("Graph contains duplicate node identifiers.")
    graph.add_nodes_from(node_by_id)
    for node in nodes:
        for port in node.output_ports():
            for connected in port.connected_ports():
                graph.add_edge(node.id, connected.node().id)
    try:
        return [node_by_id[node_id] for node_id in nx.topological_sort(graph)]
    except nx.NetworkXUnfeasible as exc:
        raise GraphCompilationError("Graph contains a circular dependency.") from exc


def connected_value(
    values: dict[tuple[str, str], str],
    source_port: Any,
    target_node: Any,
) -> str:
    source_node = source_port.node()
    key = (source_node.id, source_port.name())
    try:
        expression = values[key]
    except KeyError as exc:
        raise GraphCompilationError(
            f"No value is available from {source_node.name()!r}.{source_port.name()}."
        ) from exc

    source_unit = property_value(source_node, "unit", "-")
    target_unit = property_value(target_node, "unit", "-")
    if (
        is_specified_unit(source_unit)
        and is_specified_unit(target_unit)
        and source_unit != target_unit
    ):
        try:
            scale, offset = conversion_parameters(source_unit, target_unit)
        except UnitError as exc:
            raise GraphCompilationError(str(exc)) from exc
        if offset:
            expression = f"(({expression}) * {scale!r} + {offset!r})"
        elif scale != 1:
            expression = f"(({expression}) * {scale!r})"
    return expression


def single_source(port: Any, node: Any) -> Any:
    connected = list(port.connected_ports())
    if not connected:
        raise GraphCompilationError(
            f"Node {node.name()!r} has an unconnected input {port.name()!r}."
        )
    if len(connected) > 1:
        raise GraphCompilationError(
            f"Node {node.name()!r} input {port.name()!r} has multiple sources."
        )
    return connected[0]


def only_port(ports: list[Any], node: Any, direction: str) -> Any:
    ports = list(ports)
    if len(ports) != 1:
        raise GraphCompilationError(
            f"Node {node.name()!r} must have exactly one {direction} port."
        )
    return ports[0]


def property_value(node: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(node, "has_property") and not node.has_property(name):
            return default
        value = node.get_property(name)
    except (AttributeError, KeyError, RuntimeError):
        return default
    return default if value is None else value


def block_code(block: Any) -> str:
    widget = block.get_widget("code_content")
    code = (
        widget.get_value()
        if widget is not None
        else property_value(block, "code_content", "")
    )
    if not code:
        code = property_value(block, "code", "")
    return str(code or "").strip("\n")


def require_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GraphCompilationError(f"Missing Python name for {context}.")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise GraphCompilationError(
            f"{value!r} is not a valid Python identifier for {context}."
        )
    if value in RESERVED_NAMES:
        raise GraphCompilationError(f"{value!r} is reserved and cannot name {context}.")
    return value


def plot_color(node: Any) -> str | None:
    color = property_value(node, "plot_color")
    if not color:
        return None
    try:
        red, green, blue = (int(channel) for channel in tuple(color)[:3])
    except (TypeError, ValueError):
        return None
    if (red, green, blue) == (0, 0, 255):
        return None
    if any(channel < 0 or channel > 255 for channel in (red, green, blue)):
        return None
    return f"#{red:02x}{green:02x}{blue:02x}"


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


def safe_node_id(node_id: Any) -> str:
    return re.sub(r"\W+", "_", str(node_id))


def is_node(node: Any, prefix: str) -> bool:
    return str(getattr(node, "type_", "")).startswith(prefix)


def _surrogate_body(
    model_name: str,
    input_names: list[str],
    output_names: list[str],
) -> list[str]:
    joined = ", ".join(input_names)
    lines = [
        f"    _values = [{joined}]",
        "    if any(np.ndim(value) > 0 for value in _values):",
        "        _matrix = np.column_stack(_values)",
        f"        _prediction = np.asarray({model_name}.predict(_matrix))",
        "    else:",
        "        _matrix = np.asarray(_values).reshape(1, -1)",
        f"        _prediction = np.asarray({model_name}.predict(_matrix))",
        "    if _prediction.ndim == 1:",
        "        _prediction = _prediction.reshape(-1, 1)",
    ]
    if len(output_names) == 1:
        lines.append(
            "    return _prediction[:, 0] if len(_prediction) > 1 "
            "else _prediction[0, 0]"
        )
    else:
        items = [
            f"_prediction[:, {index}] if len(_prediction) > 1 else "
            f"_prediction[0, {index}]"
            for index in range(len(output_names))
        ]
        lines.append(f"    return {', '.join(items)}")
    return lines


def _require_assigned_outputs(
    block: Any,
    source: str,
    output_names: list[str],
) -> None:
    if not output_names:
        return
    try:
        tree = ast.parse(source or "pass")
    except SyntaxError as exc:
        raise GraphCompilationError(
            f"Invalid code in {block.name()!r}, line {exc.lineno}: {exc.msg}."
        ) from exc
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    missing = sorted(set(output_names) - assigned)
    if missing:
        raise GraphCompilationError(
            f"Block {block.name()!r} never assigns output(s): {', '.join(missing)}."
        )


__all__ = [
    "BLOCK_NODE",
    "INPUT_NODE",
    "INTERMEDIATE_NODE",
    "OUTPUT_NODE",
    "RESERVED_NAMES",
    "GraphCompilationError",
    "as_bool",
    "block_code",
    "block_function_source",
    "connected_value",
    "is_node",
    "only_port",
    "plot_color",
    "property_value",
    "require_identifier",
    "safe_node_id",
    "single_source",
    "topological_nodes",
    "wrapper_source",
]
