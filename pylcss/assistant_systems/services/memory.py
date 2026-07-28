# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Backward-compatible imports for assistant memory and key storage."""

from pylcss.assistant_systems.services.conversation_memory import (
    LEGACY_MEMORY_FILE,
    MEMORY_FILE,
    Conversation,
    ConversationMessage,
    LLMMemory,
)
from pylcss.assistant_systems.services.secure_storage import (
    ENCRYPTION_KEY_FILE,
    LEGACY_ENCRYPTION_KEY_FILE,
    SecureKeyStorage,
    get_secure_storage,
)

__all__ = [
    "Conversation",
    "ConversationMessage",
    "ENCRYPTION_KEY_FILE",
    "LEGACY_ENCRYPTION_KEY_FILE",
    "LEGACY_MEMORY_FILE",
    "LLMMemory",
    "MEMORY_FILE",
    "SecureKeyStorage",
    "get_secure_storage",
]
