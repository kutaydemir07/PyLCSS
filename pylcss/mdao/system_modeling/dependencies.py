# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Read explicit graph dependencies without guessing arbitrary Python behavior."""

from __future__ import annotations

import ast
from collections.abc import Iterable


def output_dependencies(
    model_code: str, input_names: Iterable[str], output_names: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """Use compiler metadata, or conservatively require every input.

    Conditions, helper functions, mutation, and dynamic code can all affect an
    output. Source without the graph compiler's evaluator contract is therefore
    treated as an atomic calculation.
    """

    inputs = tuple(input_names)
    outputs = tuple(output_names)
    fallback = {name: inputs for name in outputs}
    try:
        tree = ast.parse(model_code)
        metadata = None
        evaluators = set()
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            names = {
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            if "_OUTPUT_INPUTS" in names:
                metadata = ast.literal_eval(statement.value)
            if "_OUTPUT_FUNCTIONS" in names and isinstance(statement.value, ast.Dict):
                evaluators = {ast.literal_eval(key) for key in statement.value.keys}
        if not isinstance(metadata, dict) or set(metadata) != set(outputs):
            return fallback
        if evaluators != set(outputs):
            return fallback
        for required in metadata.values():
            if not isinstance(required, (list, tuple)) or not all(
                isinstance(name, str) and name in inputs for name in required
            ):
                return fallback
        return {name: tuple(metadata[name]) for name in outputs}
    except (SyntaxError, TypeError, ValueError):
        return fallback


__all__ = ["output_dependencies"]
