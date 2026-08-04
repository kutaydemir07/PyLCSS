# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Anthropic provider client used by the assistant settings dialog."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pylcss.assistant_systems.services.llm_base import (
    AuthenticationError,
    ChatCompletion,
    LLMProvider,
    LLMProviderError,
    Message,
    ModelInfo,
    NetworkError,
    RATE_LIMIT_BASE_DELAY,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MAX_RETRIES,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    name = "anthropic"
    display_name = "Anthropic (Claude)"
    API_URL = "https://api.anthropic.com/v1"

    DEFAULT_MODELS = [
        ModelInfo(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            max_context_length=200000,
        ),
        ModelInfo(
            id="claude-3-5-haiku-20241022",
            name="Claude 3.5 Haiku",
            provider="anthropic",
            max_context_length=200000,
        ),
        ModelInfo(
            id="claude-3-opus-20240229",
            name="Claude 3 Opus",
            provider="anthropic",
            max_context_length=200000,
        ),
        ModelInfo(
            id="claude-3-sonnet-20240229",
            name="Claude 3 Sonnet",
            provider="anthropic",
            max_context_length=200000,
        ),
        ModelInfo(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider="anthropic",
            max_context_length=200000,
        ),
    ]

    def _make_request(self, endpoint: str, data: dict | None = None) -> dict:
        """Make an HTTP request to Anthropic API with rate limit retry."""
        if not self.api_key:
            raise AuthenticationError("Anthropic API key not set.")

        url = f"{self.API_URL}{endpoint}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                if data:
                    body = json.dumps(data).encode("utf-8")
                    request = Request(url, data=body, headers=headers, method="POST")
                else:
                    request = Request(url, headers=headers, method="GET")

                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))

            except HTTPError as e:
                if e.code == 401:
                    raise AuthenticationError("Invalid Anthropic API key.")
                elif e.code == 429:
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(
                            RATE_LIMIT_BASE_DELAY * (2**attempt), RATE_LIMIT_MAX_DELAY
                        )
                        logger.warning(
                            f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})..."
                        )
                        time.sleep(delay)
                        continue
                    raise RateLimitError(
                        f"Anthropic rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries."
                    )
                else:
                    error_body = e.read().decode("utf-8") if e.readable() else str(e)
                    raise LLMProviderError(
                        f"Anthropic API error ({e.code}): {error_body}"
                    )
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")

        raise RateLimitError("Anthropic rate limit exceeded after retries.")

    def get_models(self) -> list[ModelInfo]:
        """Get available Claude models."""
        return self.DEFAULT_MODELS.copy()

    def chat(
        self,
        user_message: str,
        model: str = "claude-3-5-sonnet-20241022",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat message to Claude."""
        if system_prompt:
            self.set_system_prompt(system_prompt)

        self._messages.append(Message(role="user", content=user_message))

        # Claude uses a different format - system is separate
        system_content = ""
        messages = []
        for m in self._messages:
            if m.role == "system":
                system_content = m.content
            else:
                messages.append({"role": m.role, "content": m.content})

        data = {
            "model": model or "claude-3-5-sonnet-20241022",
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": messages,
        }

        if system_content:
            data["system"] = system_content

        if "temperature" in kwargs or self.temperature != 0.7:
            data["temperature"] = kwargs.get("temperature", self.temperature)

        try:
            response = self._make_request("/messages", data)
            content_blocks = response.get("content")
            if not isinstance(content_blocks, list):
                raise LLMProviderError("Anthropic returned malformed message content.")
            text_parts = [
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if not all(isinstance(part, str) for part in text_parts):
                raise LLMProviderError("Anthropic returned non-text message content.")
            content = "".join(text_parts)
        except Exception:
            self._discard_pending_user_message()
            raise

        self._messages.append(Message(role="assistant", content=content))

        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="anthropic",
            usage=response.get("usage", {}),
            finish_reason=response.get("stop_reason", "stop"),
        )
