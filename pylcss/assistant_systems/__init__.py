# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Assistant Control System for PyLCSS.

This module provides assistant control capabilities using:
- Offline speech recognition via Vosk for voice commands
- LLM-powered assistant
- System control via PyAutoGUI for mouse/keyboard simulation

Note: Camera-based head tracking has been removed.
"""

from pylcss.assistant_systems.core.manager import AssistantManager
from pylcss.assistant_systems.config import AssistantConfig, LLMControlConfig

__all__ = [
    "AssistantManager",
    "AssistantConfig",
    "LLMControlConfig",
]
