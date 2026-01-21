# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hands-Free Control System for PyLCSS.

This module provides hands-free control capabilities using:
- Offline speech recognition via Vosk for voice commands
- LLM-powered assistant via GPT@RUB API for intelligent graph building
- System control via PyAutoGUI for mouse/keyboard simulation

Note: Camera-based head tracking has been removed.
"""

from pylcss.hands_free.hands_free_manager import HandsFreeManager
from pylcss.hands_free.config import HandsFreeConfig, LLMControlConfig

__all__ = [
    "HandsFreeManager",
    "HandsFreeConfig",
    "LLMControlConfig",
]
