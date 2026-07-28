# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Local OpenAI-compatible provider client used by the assistant settings dialog."""

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


class LocalProvider(LLMProvider):
    """Local provider using OpenAI-compatible API (LM Studio)."""

    name = "local"
    display_name = "Local"
    DEFAULT_API_URL = "http://localhost:1234/v1"
    DEFAULT_API_KEY = ""

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
        selected_model: str = "",
    ) -> None:
        super().__init__(api_key=api_key)
        self.api_url = api_url or self.DEFAULT_API_URL
        self.selected_model = selected_model  # User's preferred model from config

    def _make_request(self, endpoint: str, data: dict | None = None) -> dict:
        """Make an HTTP request to Local API with rate limit retry."""
        if not self.api_key:
            self.api_key = ""

        url = f"{self.api_url}{endpoint}"
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
                    raise AuthenticationError("Invalid Local API key.")
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
                        f"Local rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries."
                    )
                else:
                    error_body = e.read().decode("utf-8") if e.readable() else str(e)
                    # Log the request details for debugging 400 errors
                    if e.code == 400 and data:
                        logger.error(
                            f"Local API 400 error - Request data: model={data.get('model')}, messages_count={len(data.get('messages', []))}"
                        )
                        # Log first message role/length for debugging
                        msgs = data.get("messages", [])
                        for i, msg in enumerate(msgs):
                            logger.error(
                                f"  Message {i}: role={msg.get('role')}, content_len={len(msg.get('content', ''))}"
                            )
                    raise LLMProviderError(f"Local API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error connecting to Local: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")

        raise RateLimitError("Local rate limit exceeded after retries.")

    def get_models(self) -> list[ModelInfo]:
        """Get available models from Local server."""
        try:
            response = self._make_request("/models")
            logger.info(f"Local API models response type: {type(response)}")
            if isinstance(response, list) and len(response) > 0:
                logger.info(f"First item sample: {response[0]}")

            models = []

            # Handle diverse response formats
            model_list = []
            if isinstance(response, dict):
                model_list = response.get("data", [])
            elif isinstance(response, list):
                model_list = response

            # Ensure model_list is actually a list
            if not isinstance(model_list, list):
                logger.warning(f"Unexpected model list format: {type(model_list)}")
                model_list = []

            for model_item in model_list:
                model_id = ""
                context_length = 4096

                if isinstance(model_item, dict):
                    # Check for 'id' OR 'name' (common in some custom APIs)
                    model_id = model_item.get("id", model_item.get("name", ""))
                    # Check for variations of context length
                    context_length = model_item.get(
                        "context_length", model_item.get("max_context_length", 4096)
                    )
                elif isinstance(model_item, str):
                    model_id = model_item

                if model_id:
                    models.append(
                        ModelInfo(
                            id=model_id,
                            name=model_id,
                            provider="local",
                            max_context_length=context_length,
                        )
                    )

            return models
        except Exception as e:
            logger.error(f"Error parsing Local/RUB models: {e}")
            raise

    def chat(
        self,
        user_message: str,
        model: str = "",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat message to Local."""
        if system_prompt:
            self.set_system_prompt(system_prompt)

        self._messages.append(Message(role="user", content=user_message))

        messages = [{"role": m.role, "content": m.content} for m in self._messages]

        # Get model: use specified > configured selected_model > first available
        if not model:
            if self.selected_model:
                model = self.selected_model
                logger.info(f"Local API: Using configured model: {model}")
            else:
                available_models = self.get_models()
                if available_models:
                    model = available_models[0].id
                    logger.info(
                        f"Local API: No model specified, using first available: {model}"
                    )
                else:
                    raise LLMProviderError("No models available on Local server.")
        else:
            logger.info(f"Local API: Using specified model: {model}")

        # Prepare request payload - minimal for RUB GPT / Azure OpenAI compatibility
        # Some endpoints reject extra parameters, so only send model and messages
        data = {
            "model": model,
            "messages": messages,
        }

        # Log the request for debugging
        logger.debug(
            f"Local API request: model={model}, messages_count={len(messages)}"
        )

        try:
            response = self._make_request("/chat/completions", data)
            if not isinstance(response, dict):
                raise LLMProviderError("Local API returned a non-object response.")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMProviderError("Local API returned no completion choices.")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise LLMProviderError("Local API returned a malformed choice.")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise LLMProviderError("Local API returned a malformed message.")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise LLMProviderError("Local API returned non-text message content.")
        except Exception:
            self._discard_pending_user_message()
            raise

        self._messages.append(Message(role="assistant", content=content))

        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="local",
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )
