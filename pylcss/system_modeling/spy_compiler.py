# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Generate the focused probe used to train a block surrogate."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .codegen import (
    BLOCK_NODE,
    INTERMEDIATE_NODE,
    GraphCompilationError,
    as_bool,
    block_function_source,
    connected_value,
    is_node,
    only_port,
    property_value,
    require_identifier,
    safe_node_id,
    single_source,
)


def build_spy_model(
    nodes: list[Any],
    input_nodes: list[Any],
    _output_nodes: list[Any],
    target_node_id: str,
    function_name: str = "spy_model",
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Return source that captures one target block's inputs and outputs."""

    require_identifier(function_name, "spy function")
    node_by_id = {node.id: node for node in nodes}
    target = node_by_id.get(target_node_id)
    if target is None:
        raise GraphCompilationError(f"Target node {target_node_id!r} was not found.")
    if not is_node(target, BLOCK_NODE):
        raise GraphCompilationError("Only custom function blocks can be probed.")

    input_ports = list(target.input_ports())
    output_ports = list(target.output_ports())
    if not input_ports or not output_ports:
        raise GraphCompilationError("The target block needs inputs and outputs.")
    spy_inputs = [{"name": port.name()} for port in input_ports]
    spy_outputs = [{"name": port.name()} for port in output_ports]

    ordered_blocks = _dependency_order(nodes, target_node_id)
    lines = [
        "import os",
        "import numpy as np",
        "from pylcss.design_studio import runtime as cad",
        "",
        "BASE_DIR = os.path.dirname(os.path.abspath(__file__))",
        "",
    ]
    surrogate_names = _emit_surrogates(lines, ordered_blocks)
    function_names: dict[str, str] = {}
    for block in ordered_blocks:
        if not is_node(block, BLOCK_NODE):
            continue
        block_name = f"_spy_block_{safe_node_id(block.id)}"
        function_names[block.id] = block_name
        lines.extend(
            block_function_source(
                block,
                block_name,
                surrogate_names.get(block.id),
            )
        )

    lines.extend(
        _spy_function_source(
            function_name,
            ordered_blocks,
            input_nodes,
            target,
            function_names,
        )
    )
    source = "\n".join(lines).rstrip() + "\n"
    try:
        compile(source, f"<generated {function_name}>", "exec")
    except SyntaxError as exc:
        raise GraphCompilationError(
            f"Generated spy source is invalid at line {exc.lineno}: {exc.msg}."
        ) from exc
    return source, spy_inputs, spy_outputs


def _dependency_order(nodes: list[Any], target_node_id: str) -> list[Any]:
    graph = nx.DiGraph()
    node_by_id = {node.id: node for node in nodes}
    graph.add_nodes_from(node_by_id)
    for node in nodes:
        for port in node.output_ports():
            for connected in port.connected_ports():
                graph.add_edge(node.id, connected.node().id)
    if target_node_id not in graph:
        raise GraphCompilationError(f"Target node {target_node_id!r} was not found.")

    dependencies = nx.ancestors(graph, target_node_id)
    dependencies.add(target_node_id)
    relevant = graph.subgraph(dependencies)
    try:
        ordered_ids = list(nx.topological_sort(relevant))
    except nx.NetworkXUnfeasible as exc:
        raise GraphCompilationError("Graph contains a circular dependency.") from exc
    return [
        node_by_id[node_id]
        for node_id in ordered_ids
        if is_node(node_by_id[node_id], BLOCK_NODE)
        or is_node(node_by_id[node_id], INTERMEDIATE_NODE)
    ]


def _emit_surrogates(lines: list[str], blocks: list[Any]) -> dict[str, str]:
    selected = [
        block
        for block in blocks
        if is_node(block, BLOCK_NODE)
        and as_bool(property_value(block, "use_surrogate", False))
        and property_value(block, "surrogate_model_path")
    ]
    if not selected:
        return {}

    lines.extend(
        [
            "import joblib",
            "",
            "def _load_spy_surrogate(path):",
            "    candidates = [path]",
            "    if not os.path.isabs(path):",
            "        candidates.extend([",
            "            os.path.join(os.getcwd(), path),",
            "            os.path.join(BASE_DIR, path),",
            "        ])",
            "    for candidate in candidates:",
            "        if os.path.isfile(candidate):",
            "            return joblib.load(candidate)",
            "    raise FileNotFoundError(f'Surrogate model not found: {path}')",
            "",
        ]
    )
    names: dict[str, str] = {}
    for block in selected:
        name = f"_spy_surrogate_{safe_node_id(block.id)}"
        path = str(property_value(block, "surrogate_model_path"))
        lines.append(f"{name} = _load_spy_surrogate({path!r})")
        names[block.id] = name
    lines.append("")
    return names


def _spy_function_source(
    function_name: str,
    blocks: list[Any],
    input_nodes: list[Any],
    target: Any,
    function_names: dict[str, str],
) -> list[str]:
    lines = [
        f"def {function_name}(*args):",
        f"    if len(args) != {len(input_nodes)}:",
        "        raise ValueError(",
        f"            'Expected {len(input_nodes)} system inputs, got ' + str(len(args))",
        "        )",
    ]
    values: dict[tuple[str, str], str] = {}
    for index, node in enumerate(input_nodes):
        outputs = list(node.output_ports())
        if not outputs:
            raise GraphCompilationError(
                f"Input node {node.name()!r} has no output port."
            )
        variable = f"_system_input_{index}"
        lines.append(f"    {variable} = args[{index}]")
        values[(node.id, outputs[0].name())] = variable

    for block in blocks:
        if is_node(block, INTERMEDIATE_NODE):
            source = single_source(
                only_port(block.input_ports(), block, "input"),
                block,
            )
            expression = connected_value(values, source, block)
            output = only_port(block.output_ports(), block, "output")
            values[(block.id, output.name())] = expression
            continue

        arguments = [
            connected_value(values, single_source(port, block), block)
            for port in block.input_ports()
        ]
        result_names: list[str] = []
        for port in block.output_ports():
            require_identifier(port.name(), f"output port on {block.name()!r}")
            result = f"_spy_value_{safe_node_id(block.id)}_{port.name()}"
            result_names.append(result)
            values[(block.id, port.name())] = result
        call = f"{function_names[block.id]}({', '.join(arguments)})"
        if result_names:
            lines.append(f"    {', '.join(result_names)} = {call}")
        else:
            lines.append(f"    {call}")

        if block.id == target.id:
            captured_inputs = ", ".join(
                f"'input_{index}': {value}"
                for index, value in enumerate(arguments)
            )
            captured_outputs = ", ".join(
                f"'output_{index}': {value}"
                for index, value in enumerate(result_names)
            )
            lines.append(
                f"    return {{{captured_inputs}}}, {{{captured_outputs}}}"
            )
            return lines

    raise GraphCompilationError("The target block is not executable from system inputs.")


__all__ = ["build_spy_model"]
