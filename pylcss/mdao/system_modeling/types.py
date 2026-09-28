# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared type definitions for the system-modeling domain."""

from collections.abc import Callable
from typing import Any, TypedDict


class _NamedVariable(TypedDict):
    name: str


class InputSpec(_NamedVariable, total=False):
    """Metadata for one public model input."""

    display_name: str
    unit: str
    min: int | float | str
    max: int | float | str
    type: str
    granularity: float


class OutputSpec(_NamedVariable, total=False):
    """Metadata for one public model output."""

    display_name: str
    unit: str
    req_min: int | float | str
    req_max: int | float | str
    minimize: bool
    maximize: bool
    show_in_legend: bool
    color: str | None


class CompiledModel(TypedDict):
    """Serializable source and metadata produced by the graph compiler."""

    name: str
    code: str
    inputs: list[InputSpec]
    outputs: list[OutputSpec]


class SystemRecord(TypedDict):
    """A named NodeGraphQt graph managed by the modeling interface."""

    name: str
    graph: Any


ModelCallable = Callable[..., dict[str, Any]]


__all__ = [
    "CompiledModel",
    "InputSpec",
    "ModelCallable",
    "OutputSpec",
    "SystemRecord",
]
