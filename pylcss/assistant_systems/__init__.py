# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""LLM-powered engineering assistant for PyLCSS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylcss.assistant_systems.config import AssistantConfig, LLMControlConfig
    from pylcss.assistant_systems.core.manager import AssistantManager

__all__ = ["AssistantConfig", "AssistantManager", "LLMControlConfig"]


def __getattr__(name: str) -> object:
    """Load Qt-heavy assistant components only when requested."""
    if name == "AssistantManager":
        from pylcss.assistant_systems.core.manager import AssistantManager

        return AssistantManager
    if name in {"AssistantConfig", "LLMControlConfig"}:
        from pylcss.assistant_systems.config import (
            AssistantConfig,
            LLMControlConfig,
        )

        return {
            "AssistantConfig": AssistantConfig,
            "LLMControlConfig": LLMControlConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
