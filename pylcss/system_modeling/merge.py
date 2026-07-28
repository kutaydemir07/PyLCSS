# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Merge independently compiled system models by their named interfaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

from .dependencies import analyze_model_dependencies
from .types import CompiledModel, InputSpec, OutputSpec
from .units import UnitError, conversion_parameters, is_specified_unit


class ModelMergeError(ValueError):
    """Raised when model interfaces cannot be merged unambiguously."""


@dataclass(frozen=True)
class VariableEndpoint:
    model_name: str
    direction: Literal["input", "output"]
    unit: str


@dataclass(frozen=True)
class VariableConnection:
    name: str
    providers: tuple[VariableEndpoint, ...]
    consumers: tuple[VariableEndpoint, ...]
    unit_issue: str | None = None


@dataclass(frozen=True)
class MergeSummary:
    connections: tuple[VariableConnection, ...]
    global_inputs: tuple[str, ...]
    global_outputs: tuple[str, ...]


@dataclass(frozen=True)
class _ExecutionStep:
    model_index: int
    valid_outputs: tuple[str, ...]


def analyze_merge(models: Sequence[CompiledModel]) -> MergeSummary:
    """Describe automatic name-based connections and unit concerns."""

    _validate_models(models)
    inputs, outputs = _collect_endpoints(models)
    global_inputs = tuple(sorted(inputs.keys() - outputs.keys()))
    global_outputs = tuple(sorted(outputs.keys() - inputs.keys()))
    connections: list[VariableConnection] = []
    for name in sorted(inputs.keys() & outputs.keys()):
        providers = tuple(outputs[name])
        consumers = tuple(inputs[name])
        issue = _unit_issue((*providers, *consumers))
        connections.append(
            VariableConnection(name, providers, consumers, issue)
        )
    return MergeSummary(tuple(connections), global_inputs, global_outputs)


def create_merged_model(models: Sequence[CompiledModel]) -> CompiledModel:
    """Create one source model with isolated namespaces for every subsystem."""

    _validate_models(models)
    all_inputs, all_outputs, provider_by_name = _collect_specs(models)
    input_names = set(all_inputs)
    output_names = set(all_outputs)
    global_input_names = sorted(input_names - output_names)
    global_output_names = sorted(output_names - input_names)
    if not global_output_names:
        raise ModelMergeError("The merged model has no externally visible outputs.")

    schedule = _execution_schedule(models, provider_by_name, global_input_names)
    source = _merged_source(
        models,
        schedule,
        provider_by_name,
        all_inputs,
        all_outputs,
        global_input_names,
        global_output_names,
    )
    return {
        "name": "Merged",
        "code": source,
        "inputs": [all_inputs[name][0].copy() for name in global_input_names],
        "outputs": [all_outputs[name].copy() for name in global_output_names],
    }


def _execution_schedule(
    models: Sequence[CompiledModel],
    provider_by_name: Mapping[str, int],
    global_inputs: Sequence[str],
) -> list[_ExecutionStep]:
    system_graph = nx.DiGraph()
    system_graph.add_nodes_from(range(len(models)))
    for consumer_index, model in enumerate(models):
        for item in model["inputs"]:
            provider_index = provider_by_name.get(item["name"])
            if provider_index is not None and provider_index != consumer_index:
                system_graph.add_edge(provider_index, consumer_index)

    if nx.is_directed_acyclic_graph(system_graph):
        return [
            _ExecutionStep(
                index,
                tuple(item["name"] for item in models[index]["outputs"]),
            )
            for index in nx.topological_sort(system_graph)
        ]

    internal_dependencies: list[set[tuple[str, str]]] = []
    variable_graph = nx.DiGraph()
    for index, model in enumerate(models):
        dependencies = analyze_model_dependencies(
            model["code"],
            (item["name"] for item in model["inputs"]),
            (item["name"] for item in model["outputs"]),
        )
        internal_dependencies.append(dependencies)
        for input_name, output_name in dependencies:
            variable_graph.add_edge(
                (index, "input", input_name),
                (index, "output", output_name),
            )
    for consumer_index, model in enumerate(models):
        for item in model["inputs"]:
            provider_index = provider_by_name.get(item["name"])
            if provider_index is not None and provider_index != consumer_index:
                variable_graph.add_edge(
                    (provider_index, "output", item["name"]),
                    (consumer_index, "input", item["name"]),
                )
    if not nx.is_directed_acyclic_graph(variable_graph):
        raise ModelMergeError("Models contain a true variable-level circular dependency.")

    available = set(global_inputs)
    pending = {
        item["name"] for model in models for item in model["outputs"]
    }
    schedule: list[_ExecutionStep] = []
    while pending:
        progress = False
        for index, model in enumerate(models):
            ready: list[str] = []
            for output in model["outputs"]:
                name = output["name"]
                if name not in pending:
                    continue
                required = {
                    input_name
                    for input_name, output_name in internal_dependencies[index]
                    if output_name == name
                }
                if required.issubset(available):
                    ready.append(name)
            if not ready:
                continue
            schedule.append(_ExecutionStep(index, tuple(ready)))
            available.update(ready)
            pending.difference_update(ready)
            progress = True
        if not progress:
            unresolved = ", ".join(sorted(pending))
            raise ModelMergeError(
                f"Cannot find a valid execution order for outputs: {unresolved}."
            )
    return schedule


def _merged_source(
    models: Sequence[CompiledModel],
    schedule: Sequence[_ExecutionStep],
    provider_by_name: Mapping[str, int],
    all_inputs: Mapping[str, list[InputSpec]],
    all_outputs: Mapping[str, OutputSpec],
    global_inputs: Sequence[str],
    global_outputs: Sequence[str],
) -> str:
    sources = [model["code"] for model in models]
    lines = [
        "# Generated by pylcss.system_modeling.merge.",
        f"_MODEL_SOURCES = {sources!r}",
        "",
        "def _load_model(source, index):",
        "    namespace = {",
        "        '__name__': f'_pylcss_merged_part_{index}',",
        "        '__file__': __file__,",
        "    }",
        "    exec(compile(source, f'<merged subsystem {index}>', 'exec'), namespace)",
        "    function = namespace.get('system_function')",
        "    if not callable(function):",
        "        candidates = [",
        "            value",
        "            for name, value in namespace.items()",
        "            if name.startswith('system_function_') and callable(value)",
        "        ]",
        "        if len(candidates) != 1:",
        "            raise ValueError(",
        "                f'Subsystem {index} does not define one model function.'",
        "            )",
        "        function = candidates[0]",
        "    return function",
        "",
        "_MODEL_FUNCTIONS = [",
        "    _load_model(source, index)",
        "    for index, source in enumerate(_MODEL_SOURCES)",
        "]",
        "",
        "def system_function(**kwargs):",
        f"    _expected = {set(global_inputs)!r}",
        "    _missing = _expected - kwargs.keys()",
        "    _unexpected = kwargs.keys() - _expected",
        "    if _missing or _unexpected:",
        "        raise TypeError(",
        "            f'Invalid merged-model arguments; missing={sorted(_missing)}, '",
        "            f'unexpected={sorted(_unexpected)}'",
        "        )",
        "    _values = dict(kwargs)",
    ]

    available = set(global_inputs)
    global_source_specs: dict[str, InputSpec | OutputSpec] = {
        name: all_inputs[name][0] for name in global_inputs
    }
    for step_number, step in enumerate(schedule):
        model = models[step.model_index]
        arguments: list[str] = []
        for input_spec in model["inputs"]:
            name = input_spec["name"]
            if name not in available:
                arguments.append(f"{name}=0.0")
                continue
            source_spec = (
                all_outputs[name]
                if name in provider_by_name
                else global_source_specs[name]
            )
            expression = f"_values[{name!r}]"
            expression = _conversion_expression(expression, source_spec, input_spec)
            arguments.append(f"{name}={expression}")
        lines.append(
            f"    _result_{step_number} = "
            f"_MODEL_FUNCTIONS[{step.model_index}]({', '.join(arguments)})"
        )
        lines.append(
            f"    if not isinstance(_result_{step_number}, dict):"
        )
        lines.append(
            f"        raise TypeError('Subsystem {step.model_index} did not return a dict.')"
        )
        for output_name in step.valid_outputs:
            lines.append(
                f"    _values[{output_name!r}] = "
                f"_result_{step_number}[{output_name!r}]"
            )
            available.add(output_name)

    return_items = ", ".join(
        f"{name!r}: _values[{name!r}]" for name in global_outputs
    )
    lines.append(f"    return {{{return_items}}}")
    source = "\n".join(lines).rstrip() + "\n"
    compile(source, "<merged system model>", "exec")
    return source


def _conversion_expression(
    expression: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> str:
    source_unit = str(source.get("unit", "-"))
    target_unit = str(target.get("unit", "-"))
    if (
        not is_specified_unit(source_unit)
        or not is_specified_unit(target_unit)
        or source_unit == target_unit
    ):
        return expression
    try:
        scale, offset = conversion_parameters(source_unit, target_unit)
    except UnitError as exc:
        raise ModelMergeError(str(exc)) from exc
    if offset:
        return f"(({expression}) * {scale!r} + {offset!r})"
    if scale != 1:
        return f"(({expression}) * {scale!r})"
    return expression


def _collect_specs(
    models: Sequence[CompiledModel],
) -> tuple[
    dict[str, list[InputSpec]],
    dict[str, OutputSpec],
    dict[str, int],
]:
    inputs: dict[str, list[InputSpec]] = defaultdict(list)
    outputs: dict[str, OutputSpec] = {}
    providers: dict[str, int] = {}
    for index, model in enumerate(models):
        for input_spec in model["inputs"]:
            inputs[input_spec["name"]].append(input_spec)
        for output_spec in model["outputs"]:
            name = output_spec["name"]
            if name in providers:
                first = models[providers[name]]["name"]
                raise ModelMergeError(
                    f"Output {name!r} has multiple providers: "
                    f"{first!r} and {model['name']!r}."
                )
            outputs[name] = output_spec
            providers[name] = index

    for name, occurrences in inputs.items():
        issue = _unit_issue(
            tuple(
                VariableEndpoint("", "input", str(item.get("unit", "-")))
                for item in occurrences
            )
        )
        if issue and issue.startswith("incompatible"):
            raise ModelMergeError(f"Global input {name!r} has {issue}.")
    return dict(inputs), outputs, providers


def _collect_endpoints(
    models: Sequence[CompiledModel],
) -> tuple[
    dict[str, list[VariableEndpoint]],
    dict[str, list[VariableEndpoint]],
]:
    inputs: dict[str, list[VariableEndpoint]] = defaultdict(list)
    outputs: dict[str, list[VariableEndpoint]] = defaultdict(list)
    for model in models:
        for input_spec in model["inputs"]:
            inputs[input_spec["name"]].append(
                VariableEndpoint(
                    model["name"],
                    "input",
                    str(input_spec.get("unit", "-")),
                )
            )
        for output_spec in model["outputs"]:
            outputs[output_spec["name"]].append(
                VariableEndpoint(
                    model["name"],
                    "output",
                    str(output_spec.get("unit", "-")),
                )
            )
    return dict(inputs), dict(outputs)


def _unit_issue(endpoints: Sequence[VariableEndpoint]) -> str | None:
    specified = [item.unit for item in endpoints if is_specified_unit(item.unit)]
    if len(set(specified)) <= 1:
        return None
    reference = specified[0]
    for unit in specified[1:]:
        try:
            conversion_parameters(reference, unit)
        except UnitError:
            return f"incompatible units: {', '.join(sorted(set(specified)))}"
    return f"unit conversion required: {', '.join(sorted(set(specified)))}"


def _validate_models(models: Sequence[CompiledModel]) -> None:
    if not models:
        raise ModelMergeError("At least one model is required.")
    model_names: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise TypeError(f"Model {index} must be a mapping.")
        missing = {"name", "code", "inputs", "outputs"} - model.keys()
        if missing:
            raise ModelMergeError(
                f"Model {index} is missing: {', '.join(sorted(missing))}."
            )
        name = model["name"]
        if not isinstance(name, str) or not name:
            raise ModelMergeError(f"Model {index} has no valid name.")
        if name in model_names:
            raise ModelMergeError(f"Duplicate model name: {name!r}.")
        model_names.add(name)
        for direction in ("inputs", "outputs"):
            variable_names: set[str] = set()
            for item in model[direction]:
                variable_name = item.get("name")
                if not isinstance(variable_name, str) or not variable_name:
                    raise ModelMergeError(
                        f"Model {name!r} has an unnamed {direction[:-1]}."
                    )
                if variable_name in variable_names:
                    raise ModelMergeError(
                        f"Model {name!r} has duplicate {direction} "
                        f"{variable_name!r}."
                    )
                variable_names.add(variable_name)


__all__ = [
    "MergeSummary",
    "ModelMergeError",
    "VariableConnection",
    "VariableEndpoint",
    "analyze_merge",
    "analyze_model_dependencies",
    "create_merged_model",
]
