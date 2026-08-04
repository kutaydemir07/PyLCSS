# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Crash and impact simulation nodes."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "CrashSolverNode",
    "ImpactConditionNode",
]

_LAZY_EXPORTS = {
    "CrashSolverNode": (
        "pylcss.design_studio.crash.solver",
        "CrashSolverNode",
    ),
    "ImpactConditionNode": (
        "pylcss.design_studio.crash.conditions",
        "ImpactConditionNode",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
