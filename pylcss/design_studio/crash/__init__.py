# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Crash and impact simulation nodes."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "CrashMaterialNode",
    "CrashSolverNode",
    "ImpactConditionNode",
    "RunRadiossDeckNode",
]

_LAZY_EXPORTS = {
    "CrashMaterialNode": (
        "pylcss.design_studio.crash.materials",
        "CrashMaterialNode",
    ),
    "CrashSolverNode": (
        "pylcss.design_studio.crash.solver",
        "CrashSolverNode",
    ),
    "ImpactConditionNode": (
        "pylcss.design_studio.crash.conditions",
        "ImpactConditionNode",
    ),
    "RunRadiossDeckNode": (
        "pylcss.design_studio.crash.radioss_deck",
        "RunRadiossDeckNode",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
