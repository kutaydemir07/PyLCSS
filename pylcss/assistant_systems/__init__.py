# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
AI Assistant System for PyLCSS.

Provides a text-driven, LLM-powered assistant (PydanticAI agent) that controls
the application by calling its tools.  Requests are entered as natural language
from the assistant side panel.
"""

from pylcss.assistant_systems.core.manager import AssistantManager
from pylcss.assistant_systems.config import AssistantConfig, LLMControlConfig

__all__ = [
    "AssistantManager",
    "AssistantConfig",
    "LLMControlConfig",
]
