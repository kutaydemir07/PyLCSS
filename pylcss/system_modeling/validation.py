# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Pure validation for system-modeling graphs."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import keyword
from typing import Any

import networkx as nx

from .types import SystemRecord
from .units import UnitError, is_specified_unit, units_compatible

_INPUT_NODE = "com.pfd.input"
_OUTPUT_NODE = "com.pfd.output"
_INTERMEDIATE_NODE = "com.pfd.intermediate"
_BLOCK_NODE = "com.pfd.custom_block"


@dataclass(frozen=True)
class ValidationReport:
    """Errors and non-blocking warnings produced by graph validation."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.is_valid


def validate_systems(systems: Sequence[SystemRecord]) -> ValidationReport:
    """Validate graph structure, code, names, units, and cross-system flow."""

    errors: list[str] = []
    warnings: list[str] = []
    if not systems:
        return ValidationReport(errors=("The product contains no systems.",))

    system_names = [str(system.get("name", "")).strip() for system in systems]
    duplicate_systems = {
        name for name, count in Counter(system_names).items() if name and count > 1
    }
    for name in sorted(duplicate_systems):
        errors.append(f"Duplicate system name: {name!r}.")

    definitions: list[dict[str, Any]] = []
    output_locations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    input_locations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for index, system in enumerate(systems):
        system_name = system_names[index] or f"System {index + 1}"
        if not system_names[index]:
            errors.append(f"System {index + 1} has an empty name.")
        graph = system.get("graph")
        if graph is None or not hasattr(graph, "all_nodes"):
            errors.append(f"System {system_name!r} has no valid graph.")
            definitions.append({"inputs": {}, "outputs": {}, "flow": nx.DiGraph()})
            continue

        nodes = list(graph.all_nodes())
        local_graph = _connection_graph(nodes)
        if not nx.is_directed_acyclic_graph(local_graph):
            errors.append(f"System {system_name!r} contains a circular dependency.")

        named_nodes: list[tuple[str, str, str]] = []
        inputs: dict[str, str] = {}
        outputs: dict[str, str] = {}
        for node in nodes:
            kind = _node_kind(node)
            if kind in {"input", "output", "intermediate"}:
                variable = _property(node, "var_name")
                if not isinstance(variable, str) or not variable:
                    errors.append(
                        f"System {system_name!r}: {kind} node {node.name()!r} "
                        "has no variable name."
                    )
                else:
                    named_nodes.append((variable, kind, node.name()))
                    if not variable.isidentifier() or keyword.iskeyword(variable):
                        errors.append(
                            f"System {system_name!r}: {variable!r} is not a "
                            "valid Python variable name."
                        )
                    unit = str(_property(node, "unit", "-"))
                    _validate_unit(unit, system_name, node.name(), errors)
                    if kind == "input":
                        inputs[variable] = unit
                        input_locations[variable].append((index, system_name, unit))
                    elif kind == "output":
                        outputs[variable] = unit
                        output_locations[variable].append((index, system_name, unit))

            if kind == "block":
                _validate_block(node, system_name, errors)

            if kind != "input":
                _validate_input_connections(node, system_name, errors)

        for variable, count in Counter(
            name for name, _, _ in named_nodes
        ).items():
            if count <= 1:
                continue
            locations = [
                f"{node_name} ({kind})"
                for name, kind, node_name in named_nodes
                if name == variable
            ]
            errors.append(
                f"System {system_name!r}: duplicate variable {variable!r} in "
                f"{', '.join(locations)}."
            )

        _validate_connection_units(nodes, system_name, errors, warnings)
        definitions.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "flow": local_graph,
                "nodes": nodes,
            }
        )

    _validate_global_outputs(output_locations, errors)
    _validate_cross_system_units(
        input_locations,
        output_locations,
        errors,
        warnings,
    )
    _validate_global_cycles(systems, definitions, errors)
    return ValidationReport(
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_block(node: Any, system_name: str, errors: list[str]) -> None:
    input_names = [port.name() for port in node.input_ports()]
    output_names = [port.name() for port in node.output_ports()]
    for port_name in input_names + output_names:
        if not port_name.isidentifier() or keyword.iskeyword(port_name):
            errors.append(
                f"System {system_name!r}: block {node.name()!r} has invalid "
                f"port name {port_name!r}."
            )
    duplicates = set(input_names) & set(output_names)
    if duplicates:
        errors.append(
            f"System {system_name!r}: block {node.name()!r} reuses "
            f"{', '.join(sorted(duplicates))} as input and output names."
        )

    source = _block_code(node)
    try:
        tree = ast.parse(source or "pass")
    except SyntaxError as exc:
        errors.append(
            f"System {system_name!r}: syntax error in {node.name()!r}, "
            f"line {exc.lineno}: {exc.msg}."
        )
        return

    assigned = {
        child.id
        for child in ast.walk(tree)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }
    missing = sorted(set(output_names) - assigned)
    uses_surrogate = _as_bool(_property(node, "use_surrogate", False))
    if missing and not uses_surrogate:
        errors.append(
            f"System {system_name!r}: block {node.name()!r} never assigns "
            f"output(s) {', '.join(missing)}."
        )
    if uses_surrogate and not _property(node, "surrogate_model_path"):
        errors.append(
            f"System {system_name!r}: block {node.name()!r} enables a "
            "surrogate but has no model path."
        )


def _validate_input_connections(
    node: Any,
    system_name: str,
    errors: list[str],
) -> None:
    for port in node.input_ports():
        connected = list(port.connected_ports())
        if not connected:
            errors.append(
                f"System {system_name!r}: node {node.name()!r} has "
                f"unconnected input {port.name()!r}."
            )
        elif len(connected) > 1:
            errors.append(
                f"System {system_name!r}: node {node.name()!r} input "
                f"{port.name()!r} has multiple sources."
            )


def _validate_connection_units(
    nodes: list[Any],
    system_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    for target in nodes:
        target_unit = str(_property(target, "unit", "-"))
        if not is_specified_unit(target_unit):
            continue
        for port in target.input_ports():
            for connected in port.connected_ports():
                source = connected.node()
                source_unit = str(_property(source, "unit", "-"))
                if not is_specified_unit(source_unit):
                    continue
                _compare_units(
                    source_unit,
                    target_unit,
                    (
                        f"System {system_name!r}: {source.name()!r} "
                        f"to {target.name()!r}"
                    ),
                    errors,
                    warnings,
                )


def _validate_global_outputs(
    locations: dict[str, list[tuple[int, str, str]]],
    errors: list[str],
) -> None:
    for variable, entries in sorted(locations.items()):
        systems = {system_index for system_index, _, _ in entries}
        if len(systems) > 1:
            names = ", ".join(system_name for _, system_name, _ in entries)
            errors.append(
                f"Output {variable!r} has multiple providers: {names}."
            )


def _validate_cross_system_units(
    inputs: dict[str, list[tuple[int, str, str]]],
    outputs: dict[str, list[tuple[int, str, str]]],
    errors: list[str],
    warnings: list[str],
) -> None:
    for variable in sorted(inputs.keys() & outputs.keys()):
        for source_index, source_name, source_unit in outputs[variable]:
            for target_index, target_name, target_unit in inputs[variable]:
                if source_index == target_index:
                    continue
                _compare_units(
                    source_unit,
                    target_unit,
                    (
                        f"Cross-system variable {variable!r}: "
                        f"{source_name!r} to {target_name!r}"
                    ),
                    errors,
                    warnings,
                )


def _compare_units(
    source: str,
    target: str,
    context: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not is_specified_unit(source) or not is_specified_unit(target):
        return
    try:
        compatible = units_compatible(source, target)
    except UnitError as exc:
        errors.append(f"{context} uses an invalid unit: {exc}")
        return
    if not compatible:
        errors.append(f"{context} has incompatible units {source!r} and {target!r}.")
    elif source != target:
        warnings.append(f"{context} converts {source!r} to {target!r}.")


def _validate_unit(
    unit: str,
    system_name: str,
    node_name: str,
    errors: list[str],
) -> None:
    if not is_specified_unit(unit):
        return
    try:
        units_compatible(unit, unit)
    except UnitError as exc:
        errors.append(
            f"System {system_name!r}: node {node_name!r} has invalid unit "
            f"{unit!r}: {exc}"
        )


def _validate_global_cycles(
    systems: Sequence[SystemRecord],
    definitions: list[dict[str, Any]],
    errors: list[str],
) -> None:
    variable_graph = nx.DiGraph()
    for system_index, definition in enumerate(definitions):
        nodes = definition.get("nodes", [])
        local_graph: nx.DiGraph = definition["flow"]
        input_nodes = {
            node.id: _property(node, "var_name")
            for node in nodes
            if _is_node(node, _INPUT_NODE) and _property(node, "var_name")
        }
        output_nodes = {
            node.id: _property(node, "var_name")
            for node in nodes
            if _is_node(node, _OUTPUT_NODE) and _property(node, "var_name")
        }
        for input_id, input_name in input_nodes.items():
            descendants = nx.descendants(local_graph, input_id)
            for output_id, output_name in output_nodes.items():
                if output_id in descendants:
                    variable_graph.add_edge(
                        (system_index, input_name, "input"),
                        (system_index, output_name, "output"),
                    )

    providers: dict[str, list[int]] = defaultdict(list)
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, definition in enumerate(definitions):
        for name in definition["outputs"]:
            providers[name].append(index)
        for name in definition["inputs"]:
            consumers[name].append(index)
    for name in providers.keys() & consumers.keys():
        for source in providers[name]:
            for target in consumers[name]:
                if source != target:
                    variable_graph.add_edge(
                        (source, name, "output"),
                        (target, name, "input"),
                    )

    try:
        cycle = nx.find_cycle(variable_graph)
    except nx.NetworkXNoCycle:
        return
    cycle_nodes = [edge[0] for edge in cycle]
    readable = " -> ".join(
        f"[{systems[index]['name']}] {variable}"
        for index, variable, _direction in cycle_nodes
    )
    errors.append(f"Cross-system circular dependency: {readable}.")


def _connection_graph(nodes: list[Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(node.id for node in nodes)
    for node in nodes:
        for port in node.output_ports():
            for connected in port.connected_ports():
                graph.add_edge(node.id, connected.node().id)
    return graph


def _block_code(node: Any) -> str:
    widget = node.get_widget("code_content")
    source = widget.get_value() if widget is not None else _property(
        node, "code_content", ""
    )
    if not source:
        source = _property(node, "code", "")
    return str(source or "")


def _node_kind(node: Any) -> str:
    if _is_node(node, _INPUT_NODE):
        return "input"
    if _is_node(node, _OUTPUT_NODE):
        return "output"
    if _is_node(node, _INTERMEDIATE_NODE):
        return "intermediate"
    if _is_node(node, _BLOCK_NODE):
        return "block"
    return "unknown"


def _property(node: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(node, "has_property") and not node.has_property(name):
            return default
        value = node.get_property(name)
    except (AttributeError, KeyError, RuntimeError):
        return default
    return default if value is None else value


def _is_node(node: Any, prefix: str) -> bool:
    return str(getattr(node, "type_", "")).startswith(prefix)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


__all__ = ["ValidationReport", "validate_systems"]
