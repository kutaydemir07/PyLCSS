# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Infer data dependencies from generated Python model functions."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence


def analyze_model_dependencies(
    model_code: str,
    input_names: Iterable[str],
    output_names: Iterable[str],
) -> set[tuple[str, str]]:
    """Conservatively infer input-to-output dependencies from model source."""

    inputs = tuple(input_names)
    outputs = tuple(output_names)
    fallback = {
        (input_name, output_name)
        for input_name in inputs
        for output_name in outputs
    }
    try:
        tree = ast.parse(model_code)
    except (SyntaxError, TypeError):
        return fallback

    dependencies: dict[str, set[str]] = {}
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        arguments = _argument_names(function)
        environment = {name: {name} for name in inputs if name in arguments}
        returned = _analyze_statements(function.body, environment)
        for output_name, sources in returned.items():
            if output_name in outputs:
                dependencies.setdefault(output_name, set()).update(
                    sources & set(inputs)
                )

    result: set[tuple[str, str]] = set()
    for output_name in outputs:
        output_sources = dependencies.get(output_name, set(inputs))
        result.update((source, output_name) for source in output_sources)
    return result


def _argument_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    arguments = function.args
    return {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }


def _analyze_statements(
    statements: Sequence[ast.stmt],
    environment: dict[str, set[str]],
) -> dict[str, set[str]]:
    returned: dict[str, set[str]] = {}
    for statement in statements:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                continue
            dependencies = _expression_dependencies(value, environment)
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                for name in _target_names(target):
                    environment[name] = set(dependencies)
        elif isinstance(statement, ast.AugAssign):
            dependencies = _expression_dependencies(statement.value, environment)
            for name in _target_names(statement.target):
                environment.setdefault(name, set()).update(dependencies)
        elif isinstance(statement, ast.Return) and isinstance(
            statement.value,
            ast.Dict,
        ):
            for key, value in zip(statement.value.keys, statement.value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    returned.setdefault(key.value, set()).update(
                        _expression_dependencies(value, environment)
                    )
        elif isinstance(
            statement,
            (ast.If, ast.For, ast.While, ast.Try, ast.With),
        ):
            nested_blocks = [
                value
                for value in vars(statement).values()
                if isinstance(value, list)
                and all(isinstance(item, ast.stmt) for item in value)
            ]
            for block in nested_blocks:
                branch_environment = {
                    name: set(values) for name, values in environment.items()
                }
                branch_returns = _analyze_statements(block, branch_environment)
                for name, values in branch_environment.items():
                    environment.setdefault(name, set()).update(values)
                for name, values in branch_returns.items():
                    returned.setdefault(name, set()).update(values)
    return returned


def _expression_dependencies(
    expression: ast.AST,
    environment: Mapping[str, set[str]],
) -> set[str]:
    dependencies: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            dependencies.update(environment.get(node.id, ()))
    return dependencies


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


__all__ = ["analyze_model_dependencies"]
