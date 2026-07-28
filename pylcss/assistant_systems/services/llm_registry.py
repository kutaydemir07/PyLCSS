# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Factory for settings-dialog LLM provider clients."""

from __future__ import annotations

from typing import Any

from pylcss.assistant_systems.services.anthropic_provider import AnthropicProvider
from pylcss.assistant_systems.services.google_provider import GoogleProvider
from pylcss.assistant_systems.services.llm_base import LLMProvider
from pylcss.assistant_systems.services.local_provider import LocalProvider
from pylcss.assistant_systems.services.openai_provider import OpenAIProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "local": LocalProvider,
}

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI (ChatGPT)",
    "anthropic": "Anthropic (Claude)",
    "google": "Google (Gemini)",
    "local": "Local",
}


def get_provider(name: str, api_key: str = "", **kwargs: Any) -> LLMProvider:
    """Create a provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}"
        )

    if name == "local":
        return LocalProvider(
            api_key=api_key,
            api_url=kwargs.get("local_api_url", ""),
            selected_model=kwargs.get("selected_model", ""),
        )

    return PROVIDERS[name](api_key=api_key, **kwargs)


def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return list(PROVIDERS.keys())
