# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""OpenAI provider client used by the assistant settings dialog."""

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


class OpenAIProvider(LLMProvider):
    """OpenAI ChatGPT provider."""

    name = "openai"
    display_name = "OpenAI (ChatGPT)"
    API_URL = "https://api.openai.com/v1"

    DEFAULT_MODELS = [
        ModelInfo(
            id="gpt-4o", name="GPT-4o", provider="openai", max_context_length=128000
        ),
        ModelInfo(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider="openai",
            max_context_length=128000,
        ),
        ModelInfo(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            provider="openai",
            max_context_length=128000,
        ),
        ModelInfo(id="gpt-4", name="GPT-4", provider="openai", max_context_length=8192),
        ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="openai",
            max_context_length=16385,
        ),
    ]

    def _make_request(self, endpoint: str, data: dict | None = None) -> dict:
        """Make an HTTP request to OpenAI API with rate limit retry."""
        if not self.api_key:
            raise AuthenticationError("OpenAI API key not set.")

        url = f"{self.API_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
                    raise AuthenticationError("Invalid OpenAI API key.")
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
                        f"OpenAI rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries."
                    )
                else:
                    error_body = e.read().decode("utf-8") if e.readable() else str(e)
                    raise LLMProviderError(f"OpenAI API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")

        raise RateLimitError("OpenAI rate limit exceeded after retries.")

    def get_models(self) -> list[ModelInfo]:
        """Get available OpenAI models."""
        # Return curated list instead of all models
        return self.DEFAULT_MODELS.copy()

    def chat(
        self,
        user_message: str,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat message to OpenAI."""
        if system_prompt:
            self.set_system_prompt(system_prompt)

        self._messages.append(Message(role="user", content=user_message))

        messages = [{"role": m.role, "content": m.content} for m in self._messages]

        data = {
            "model": model or "gpt-4o",
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            response = self._make_request("/chat/completions", data)
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMProviderError("OpenAI returned no completion choices.")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise LLMProviderError("OpenAI returned a malformed completion choice.")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise LLMProviderError("OpenAI returned a malformed message.")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise LLMProviderError("OpenAI returned non-text message content.")
        except Exception:
            self._discard_pending_user_message()
            raise

        self._messages.append(Message(role="assistant", content=content))

        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="openai",
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )
