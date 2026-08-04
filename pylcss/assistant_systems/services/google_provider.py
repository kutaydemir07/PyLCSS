# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Google provider client used by the assistant settings dialog."""

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


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    name = "google"
    display_name = "Google (Gemini)"
    API_URL = "https://generativelanguage.googleapis.com/v1beta"

    DEFAULT_MODELS = [
        ModelInfo(
            id="gemini-2.5-flash-lite",
            name="Gemini 2.5 Flash Lite",
            provider="google",
            max_context_length=1000000,
        ),
        ModelInfo(
            id="gemini-3-flash",
            name="Gemini 3 Flash",
            provider="google",
            max_context_length=1000000,
        ),
        ModelInfo(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="google",
            max_context_length=1000000,
        ),
    ]

    # Fallback order when a model hits rate limit
    FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]

    def _make_request(
        self, endpoint: str, data: dict | None = None, method: str = "POST"
    ) -> dict:
        """Make an HTTP request to Google AI API with rate limit retry."""
        if not self.api_key:
            raise AuthenticationError("Google API key not set.")

        url = f"{self.API_URL}{endpoint}?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                if data:
                    body = json.dumps(data).encode("utf-8")
                    request = Request(url, data=body, headers=headers, method=method)
                else:
                    request = Request(url, headers=headers, method="GET")

                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))

            except HTTPError as e:
                if e.code == 401 or e.code == 403:
                    raise AuthenticationError("Invalid Google API key.")
                elif e.code == 429:
                    # Rate limit - retry with exponential backoff
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(
                            RATE_LIMIT_BASE_DELAY * (2**attempt), RATE_LIMIT_MAX_DELAY
                        )
                        logger.warning(
                            f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError(
                            f"Google rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries."
                        )
                else:
                    error_body = e.read().decode("utf-8") if e.readable() else str(e)
                    raise LLMProviderError(f"Google API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")

        # Should not reach here, but just in case
        raise RateLimitError("Google rate limit exceeded after retries.")

    def get_models(self) -> list[ModelInfo]:
        """Get available Gemini models from API."""
        try:
            response = self._make_request("/models", method="GET")
            models = []

            for model_data in response.get("models", []):
                # Filter for models that support content generation
                supported_methods = model_data.get("supportedGenerationMethods", [])
                if "generateContent" not in supported_methods:
                    continue

                # Extract clean ID (remove 'models/' prefix if present)
                full_name = model_data.get("name", "")
                model_id = (
                    full_name.replace("models/", "")
                    if full_name.startswith("models/")
                    else full_name
                )

                models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_data.get("displayName", model_id),
                        provider="google",
                        max_context_length=model_data.get("inputTokenLimit", 32000),
                    )
                )

            if models:
                # Sort by name for better UX
                models.sort(key=lambda x: x.name, reverse=True)
                return models

        except AuthenticationError:
            raise
        except Exception as e:
            logger.warning("Failed to fetch Google models: %s", e)
            if isinstance(e, LLMProviderError):
                raise

        # An otherwise valid response can omit generation-capable models.
        return self.DEFAULT_MODELS.copy()

    def chat(
        self,
        user_message: str,
        model: str = "gemini-2.5-flash-lite",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat message to Gemini with automatic model fallback on rate limit."""
        if system_prompt:
            self.set_system_prompt(system_prompt)

        self._messages.append(Message(role="user", content=user_message))

        # Gemini uses different format
        contents = []
        system_instruction = None

        for m in self._messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Build list of models to try: requested model first, then fallbacks
        model_name = model or "gemini-2.5-flash-lite"
        models_to_try = [model_name]
        for fallback in self.FALLBACK_MODELS:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for try_model in models_to_try:
            try:
                response = self._make_request(
                    f"/models/{try_model}:generateContent", data
                )
                if not isinstance(response, dict):
                    raise LLMProviderError("Google returned a non-object response.")

                candidates = response.get("candidates", [])
                if not isinstance(candidates, list):
                    raise LLMProviderError("Google returned malformed candidates.")
                content = ""
                finish_reason = "stop"
                if candidates:
                    candidate = candidates[0]
                    if not isinstance(candidate, dict):
                        raise LLMProviderError("Google returned a malformed candidate.")
                    candidate_content = candidate.get("content", {})
                    if not isinstance(candidate_content, dict):
                        raise LLMProviderError(
                            "Google returned malformed message content."
                        )
                    parts = candidate_content.get("parts", [])
                    if not isinstance(parts, list):
                        raise LLMProviderError(
                            "Google returned malformed message parts."
                        )
                    for part in parts:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            if not isinstance(text, str):
                                raise LLMProviderError(
                                    "Google returned non-text message content."
                                )
                            content += text
                    finish_reason = str(candidate.get("finishReason", "STOP"))

                self._messages.append(Message(role="assistant", content=content))

                if try_model != model_name:
                    logger.info(f"Successfully fell back to model: {try_model}")

                return ChatCompletion(
                    content=content,
                    model=try_model,
                    provider="google",
                    usage=response.get("usageMetadata", {}),
                    finish_reason=finish_reason,
                )

            except RateLimitError as e:
                logger.warning(f"Model {try_model} rate limited, trying fallback...")
                last_error = e
                continue
            except LLMProviderError:
                self._discard_pending_user_message()
                raise

        # All models failed
        # Remove the user message since we couldn't get a response
        self._discard_pending_user_message()
        raise RateLimitError(f"All models rate limited. Last error: {last_error}")
