# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared node contracts and input-resolution utilities."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "CadQueryNode",
    "CancelCallback",
    "GraphLike",
    "InputPortLike",
    "NodeLike",
    "NodeResult",
    "OutputPortLike",
    "is_numeric",
    "is_shape",
    "resolve_all_inputs",
    "resolve_any_input",
    "resolve_numeric_input",
    "resolve_shape_input",
]

_BASE = "pylcss.design_studio.core.base_node"
_CONTRACTS = "pylcss.design_studio.core.contracts"
_LAZY_EXPORTS = {
    "CadQueryNode": (_BASE, "CadQueryNode"),
    "CancelCallback": (_CONTRACTS, "CancelCallback"),
    "GraphLike": (_CONTRACTS, "GraphLike"),
    "InputPortLike": (_CONTRACTS, "InputPortLike"),
    "NodeLike": (_CONTRACTS, "NodeLike"),
    "NodeResult": (_CONTRACTS, "NodeResult"),
    "OutputPortLike": (_CONTRACTS, "OutputPortLike"),
    "is_numeric": (_BASE, "is_numeric"),
    "is_shape": (_BASE, "is_shape"),
    "resolve_all_inputs": (_BASE, "resolve_all_inputs"),
    "resolve_any_input": (_BASE, "resolve_any_input"),
    "resolve_numeric_input": (_BASE, "resolve_numeric_input"),
    "resolve_shape_input": (_BASE, "resolve_shape_input"),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
