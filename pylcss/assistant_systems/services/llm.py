# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Multi-Provider LLM Client Abstraction.

Supports OpenAI (ChatGPT), Anthropic (Claude), and Google (Gemini).
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# Rate limit retry configuration
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 2.0  # seconds
RATE_LIMIT_MAX_DELAY = 60.0  # max wait time in seconds


@dataclass
class Message:
    """A chat message."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


@dataclass
class ChatCompletion:
    """Response from chat completion API."""
    content: str
    model: str
    provider: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    provider: str = ""
    max_context_length: int = 4096


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class AuthenticationError(LLMProviderError):
    """Authentication failed (invalid API key)."""
    pass


class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    pass


class NetworkError(LLMProviderError):
    """Network error."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    name: str = "base"
    display_name: str = "Base Provider"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._messages: List[Message] = []
        self.temperature = 0.7
        self.max_tokens = 1000
        
    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self.api_key = api_key
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = []
        
    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self._messages.copy()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._messages = [m for m in self._messages if m.role != "system"]
        self._messages.insert(0, Message(role="system", content=prompt))
    
    @abstractmethod
    def get_models(self) -> List[ModelInfo]:
        """Get available models for this provider."""
        pass
    
    @abstractmethod
    def chat(
        self,
        user_message: str,
        model: str = "",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """Send a chat message and get a response."""
        pass
    
    def chat_async(
        self,
        user_message: str,
        on_complete: Callable[[ChatCompletion], None],
        on_error: Callable[[Exception], None],
        model: str = "",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> threading.Thread:
        """Send a chat message asynchronously."""
        def run():
            try:
                result = self.chat(user_message, model=model, system_prompt=system_prompt, **kwargs)
                on_complete(result)
            except Exception as e:
                on_error(e)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            self.get_models()
            return True
        except Exception:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI ChatGPT provider."""
    
    name = "openai"
    display_name = "OpenAI (ChatGPT)"
    API_URL = "https://api.openai.com/v1"
    
    DEFAULT_MODELS = [
        ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai", max_context_length=128000),
        ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider="openai", max_context_length=128000),
        ModelInfo(id="gpt-4-turbo", name="GPT-4 Turbo", provider="openai", max_context_length=128000),
        ModelInfo(id="gpt-4", name="GPT-4", provider="openai", max_context_length=8192),
        ModelInfo(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider="openai", max_context_length=16385),
    ]
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
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
                    body = json.dumps(data).encode('utf-8')
                    request = Request(url, data=body, headers=headers, method='POST')
                else:
                    request = Request(url, headers=headers, method='GET')
                
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode('utf-8'))
                    
            except HTTPError as e:
                if e.code == 401:
                    raise AuthenticationError("Invalid OpenAI API key.")
                elif e.code == 429:
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY)
                        logger.warning(f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                        time.sleep(delay)
                        continue
                    raise RateLimitError(f"OpenAI rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries.")
                else:
                    error_body = e.read().decode('utf-8') if e.readable() else str(e)
                    raise LLMProviderError(f"OpenAI API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")
        
        raise RateLimitError("OpenAI rate limit exceeded after retries.")
    
    def get_models(self) -> List[ModelInfo]:
        """Get available OpenAI models."""
        # Return curated list instead of all models
        return self.DEFAULT_MODELS.copy()
    
    def chat(
        self,
        user_message: str,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        **kwargs
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
        
        response = self._make_request("/chat/completions", data)
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        self._messages.append(Message(role="assistant", content=content))
        
        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="openai",
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    name = "anthropic"
    display_name = "Anthropic (Claude)"
    API_URL = "https://api.anthropic.com/v1"
    
    DEFAULT_MODELS = [
        ModelInfo(id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", provider="anthropic", max_context_length=200000),
        ModelInfo(id="claude-3-5-haiku-20241022", name="Claude 3.5 Haiku", provider="anthropic", max_context_length=200000),
        ModelInfo(id="claude-3-opus-20240229", name="Claude 3 Opus", provider="anthropic", max_context_length=200000),
        ModelInfo(id="claude-3-sonnet-20240229", name="Claude 3 Sonnet", provider="anthropic", max_context_length=200000),
        ModelInfo(id="claude-3-haiku-20240307", name="Claude 3 Haiku", provider="anthropic", max_context_length=200000),
    ]
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
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
                    body = json.dumps(data).encode('utf-8')
                    request = Request(url, data=body, headers=headers, method='POST')
                else:
                    request = Request(url, headers=headers, method='GET')
                
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode('utf-8'))
                    
            except HTTPError as e:
                if e.code == 401:
                    raise AuthenticationError("Invalid Anthropic API key.")
                elif e.code == 429:
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY)
                        logger.warning(f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                        time.sleep(delay)
                        continue
                    raise RateLimitError(f"Anthropic rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries.")
                else:
                    error_body = e.read().decode('utf-8') if e.readable() else str(e)
                    raise LLMProviderError(f"Anthropic API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")
        
        raise RateLimitError("Anthropic rate limit exceeded after retries.")
    
    def get_models(self) -> List[ModelInfo]:
        """Get available Claude models."""
        return self.DEFAULT_MODELS.copy()
    
    def chat(
        self,
        user_message: str,
        model: str = "claude-3-5-sonnet-20241022",
        system_prompt: Optional[str] = None,
        **kwargs
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
        
        response = self._make_request("/messages", data)
        
        # Claude returns content as a list
        content_blocks = response.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")
        
        self._messages.append(Message(role="assistant", content=content))
        
        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="anthropic",
            usage=response.get("usage", {}),
            finish_reason=response.get("stop_reason", "stop"),
        )


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""
    
    name = "google"
    display_name = "Google (Gemini)"
    API_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    DEFAULT_MODELS = [
        ModelInfo(id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite", provider="google", max_context_length=1000000),
        ModelInfo(id="gemini-3-flash", name="Gemini 3 Flash", provider="google", max_context_length=1000000),
        ModelInfo(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider="google", max_context_length=1000000),
    ]
    
    # Fallback order when a model hits rate limit
    FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None, method: str = "POST") -> Dict:
        """Make an HTTP request to Google AI API with rate limit retry."""
        if not self.api_key:
            raise AuthenticationError("Google API key not set.")
            
        url = f"{self.API_URL}{endpoint}?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }
        
        last_error = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                if data:
                    body = json.dumps(data).encode('utf-8')
                    request = Request(url, data=body, headers=headers, method=method)
                else:
                    request = Request(url, headers=headers, method='GET')
                
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode('utf-8'))
                    
            except HTTPError as e:
                if e.code == 401 or e.code == 403:
                    raise AuthenticationError("Invalid Google API key.")
                elif e.code == 429:
                    # Rate limit - retry with exponential backoff
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY)
                        logger.warning(f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                        time.sleep(delay)
                        last_error = e
                        continue
                    else:
                        raise RateLimitError(f"Google rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries.")
                else:
                    error_body = e.read().decode('utf-8') if e.readable() else str(e)
                    raise LLMProviderError(f"Google API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")
        
        # Should not reach here, but just in case
        raise RateLimitError("Google rate limit exceeded after retries.")
    
    def get_models(self) -> List[ModelInfo]:
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
                model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                
                models.append(ModelInfo(
                    id=model_id,
                    name=model_data.get("displayName", model_id),
                    provider="google",
                    max_context_length=model_data.get("inputTokenLimit", 32000),
                ))
            
            if models:
                # Sort by name for better UX
                models.sort(key=lambda x: x.name, reverse=True)
                return models
                
        except AuthenticationError:
            # Re-raise auth errors so UI shows "Invalid Key"
            raise
        except Exception as e:
            logger.warning(f"Failed to fetch Google models: {e}")
            # If it's a critical error (like rate limit), maybe re-raise?
            # For now, if we can't list models, we can't verify the key works.
            # But to be safe for existing users with network issues, we could fallback.
            # However, the user specifically wants to fix the "fake success" issue.
            # So let's re-raise if it's an API error, but maybe fallback for generic network?
            # Actually, without models we can't chat anyway (names might be wrong).
            # So falling back to defaults is risky if defaults are wrong.
            if isinstance(e, LLMProviderError):
                raise
        
        # Fallback to default models ONLY if API didn't return valid model list but also didn't error (edge case)
        # or if we decide to be lenient on network errors. 
        # For this specific "invalid key" bug, we WANT it to fail if key is bad.
        # So we only fallback if we really have to.
        return self.DEFAULT_MODELS.copy()
    
    def chat(
        self,
        user_message: str,
        model: str = "gemini-2.5-flash-lite",
        system_prompt: Optional[str] = None,
        **kwargs
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
                contents.append({
                    "role": role,
                    "parts": [{"text": m.content}]
                })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
            }
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
                response = self._make_request(f"/models/{try_model}:generateContent", data)
                
                # Extract content from Gemini response
                candidates = response.get("candidates", [])
                content = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        content += part.get("text", "")
                
                self._messages.append(Message(role="assistant", content=content))
                
                if try_model != model_name:
                    logger.info(f"Successfully fell back to model: {try_model}")
                
                return ChatCompletion(
                    content=content,
                    model=try_model,
                    provider="google",
                    usage=response.get("usageMetadata", {}),
                    finish_reason=candidates[0].get("finishReason", "STOP") if candidates else "stop",
                )
                
            except RateLimitError as e:
                logger.warning(f"Model {try_model} rate limited, trying fallback...")
                last_error = e
                continue
            except LLMProviderError as e:
                # For non-rate-limit errors, don't try fallback
                raise
        
        # All models failed
        # Remove the user message since we couldn't get a response
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()
        raise RateLimitError(f"All models rate limited. Last error: {last_error}")



class LocalProvider(LLMProvider):
    """Local provider using OpenAI-compatible API (LM Studio)."""
    
    name = "local"
    display_name = "Local"
    DEFAULT_API_URL = "http://localhost:1234/v1"
    DEFAULT_API_KEY = ""
    
    def __init__(self, api_key: str = "", api_url: str = ""):
        super().__init__(api_key=api_key)
        self.api_url = api_url or self.DEFAULT_API_URL
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
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
                    body = json.dumps(data).encode('utf-8')
                    request = Request(url, data=body, headers=headers, method='POST')
                else:
                    request = Request(url, headers=headers, method='GET')
                
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode('utf-8'))
                    
            except HTTPError as e:
                if e.code == 401:
                    raise AuthenticationError("Invalid Local API key.")
                elif e.code == 429:
                    if attempt < RATE_LIMIT_MAX_RETRIES:
                        delay = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY)
                        logger.warning(f"Rate limit hit, waiting {delay:.1f}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                        time.sleep(delay)
                        continue
                    raise RateLimitError(f"Local rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries.")
                else:
                    error_body = e.read().decode('utf-8') if e.readable() else str(e)
                    raise LLMProviderError(f"Local API error ({e.code}): {error_body}")
            except URLError as e:
                raise NetworkError(f"Network error connecting to Local: {e.reason}")
            except Exception as e:
                raise LLMProviderError(f"Unexpected error: {e}")
        
        raise RateLimitError("Local rate limit exceeded after retries.")
    
    def get_models(self) -> List[ModelInfo]:
        """Get available models from Local server."""
        try:
            response = self._make_request("/models")
            models = []
            
            for model_data in response.get("data", []):
                model_id = model_data.get("id", "")
                models.append(ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider="local",
                    max_context_length=model_data.get("context_length", 4096),
                ))
            
            if models:
                return models
                
        except Exception as e:
            logger.warning(f"Failed to fetch Local models: {e}")
        
        # Return empty list if no models found
        return []
    
    def chat(
        self,
        user_message: str,
        model: str = "",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """Send a chat message to Local."""
        if system_prompt:
            self.set_system_prompt(system_prompt)
            
        self._messages.append(Message(role="user", content=user_message))
        
        messages = [{"role": m.role, "content": m.content} for m in self._messages]
        
        # Get model from available models if not specified
        if not model:
            available_models = self.get_models()
            if available_models:
                model = available_models[0].id
            else:
                raise LLMProviderError("No models available on Local server.")
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        response = self._make_request("/chat/completions", data)
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        self._messages.append(Message(role="assistant", content=content))
        
        return ChatCompletion(
            content=content,
            model=response.get("model", model),
            provider="local",
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )


# Provider registry
PROVIDERS: Dict[str, type] = {
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


def get_provider(name: str, api_key: str = "", **kwargs) -> LLMProvider:
    """Create a provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    
    if name == "local":
        return LocalProvider(api_key=api_key, api_url=kwargs.get("local_api_url", ""))
        
    return PROVIDERS[name](api_key=api_key, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available provider names."""
    return list(PROVIDERS.keys())

