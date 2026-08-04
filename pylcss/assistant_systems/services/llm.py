# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility imports for settings-dialog LLM provider clients."""

from pylcss.assistant_systems.services.anthropic_provider import AnthropicProvider
from pylcss.assistant_systems.services.google_provider import GoogleProvider
from pylcss.assistant_systems.services.llm_base import (
    AuthenticationError,
    ChatCompletion,
    LLMProvider,
    LLMProviderError,
    Message,
    ModelInfo,
    NetworkError,
    RateLimitError,
)
from pylcss.assistant_systems.services.llm_registry import (
    PROVIDER_DISPLAY_NAMES,
    PROVIDERS,
    get_available_providers,
    get_provider,
)
from pylcss.assistant_systems.services.local_provider import LocalProvider
from pylcss.assistant_systems.services.openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "AuthenticationError",
    "ChatCompletion",
    "GoogleProvider",
    "LLMProvider",
    "LLMProviderError",
    "LocalProvider",
    "Message",
    "ModelInfo",
    "NetworkError",
    "OpenAIProvider",
    "PROVIDER_DISPLAY_NAMES",
    "PROVIDERS",
    "RateLimitError",
    "get_available_providers",
    "get_provider",
]
