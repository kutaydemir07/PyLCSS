# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared contracts for the settings-dialog LLM provider clients."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 2.0
RATE_LIMIT_MAX_DELAY = 60.0


MessageRole = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    """A chat message."""

    role: MessageRole
    content: str


@dataclass(slots=True)
class ChatCompletion:
    """Response from chat completion API."""

    content: str
    model: str
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass(slots=True)
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str = ""
    max_context_length: int = 4096


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class AuthenticationError(LLMProviderError):
    """Authentication failed (invalid API key)."""


class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""


class NetworkError(LLMProviderError):
    """Network error."""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"
    display_name: str = "Base Provider"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._messages: list[Message] = []
        self.temperature = 0.7
        self.max_tokens = 1000

    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self.api_key = api_key

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = []

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self._messages.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._messages = [m for m in self._messages if m.role != "system"]
        self._messages.insert(0, Message(role="system", content=prompt))

    def _discard_pending_user_message(self) -> None:
        """Roll back the user turn when a provider request fails."""
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()

    @abstractmethod
    def get_models(self) -> list[ModelInfo]:
        """Get available models for this provider."""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        user_message: str,
        model: str = "",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat message and get a response."""
        raise NotImplementedError

    def chat_async(
        self,
        user_message: str,
        on_complete: Callable[[ChatCompletion], None],
        on_error: Callable[[Exception], None],
        model: str = "",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> threading.Thread:
        """Send a chat message asynchronously."""

        def run() -> None:
            try:
                result = self.chat(
                    user_message, model=model, system_prompt=system_prompt, **kwargs
                )
                on_complete(result)
            except Exception as e:
                on_error(e)

        thread = threading.Thread(
            target=run,
            name=f"pylcss-{self.name}-request",
            daemon=True,
        )
        thread.start()
        return thread

    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            self.get_models()
            return True
        except Exception:
            return False
