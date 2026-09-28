# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compilation orchestration for PyLCSS system models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

from .compiler import GraphBuilder
from .types import CompiledModel, SystemRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompilationFailure:
    system_name: str
    reason: str


class ModelCompilationError(RuntimeError):
    """Raised after one or more graphs fail to compile."""

    def __init__(self, failures: Sequence[CompilationFailure]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(
            f"{failure.system_name}: {failure.reason}" for failure in self.failures
        )
        super().__init__(f"Could not compile {len(self.failures)} system(s): {details}")


def compile_systems(systems: Sequence[SystemRecord]) -> list[CompiledModel]:
    """Compile named graphs, raising one actionable aggregate error on failure."""

    models: list[CompiledModel] = []
    failures: list[CompilationFailure] = []
    global_names = {f"system_function_{index}" for index in range(len(systems))}

    for index, system in enumerate(systems):
        system_name = str(system.get("name") or f"System {index + 1}")
        try:
            builder = GraphBuilder(system["graph"])
            code, inputs, outputs = builder.build_system_model(
                function_name=f"system_function_{index}",
                global_reserved_names=global_names,
            )
            models.append(
                {
                    "name": system_name,
                    "code": code,
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )
        except Exception as exc:
            logger.exception("Failed to compile system %r", system_name)
            failures.append(CompilationFailure(system_name, str(exc)))

    if failures:
        raise ModelCompilationError(failures)
    return models


__all__ = [
    "CompilationFailure",
    "ModelCompilationError",
    "compile_systems",
]
