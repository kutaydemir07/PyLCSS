# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Client for GPT@RUB API integration.

Provides an OpenAI-compatible client for interacting with the GPT@RUB API
to enable LLM-powered hands-free control of PyLCSS.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


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
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    provider: str = ""
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    max_context_length: int = 4096


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class AuthenticationError(LLMClientError):
    """Authentication failed (invalid token)."""
    pass


class RateLimitError(LLMClientError):
    """Rate limit exceeded (60 req/min)."""
    pass


class NetworkError(LLMClientError):
    """Network error (not on VPN, server unavailable)."""
    pass


class LLMClient:
    """
    Client for GPT@RUB API.
    
    Follows OpenAI API specification and is compatible with official OpenAI libraries.
    Note: Does not support streaming.
    """
    
    DEFAULT_API_URL = "https://gpt.ruhr-uni-bochum.de/external/v1"
    
    def __init__(
        self,
        access_token: str = "",
        api_url: str = DEFAULT_API_URL,
        model: str = "gpt-4.1-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the LLM client.
        
        Args:
            access_token: GPT@RUB access token from account settings
            api_url: Base URL for the API
            model: Default model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in the response
        """
        self.access_token = access_token
        self.api_url = api_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Conversation history
        self._messages: List[Message] = []
        
    def set_token(self, token: str) -> None:
        """Set the access token."""
        self.access_token = token
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages = []
        
    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self._messages.copy()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt (replaces any existing system message)."""
        # Remove existing system messages
        self._messages = [m for m in self._messages if m.role != "system"]
        # Add new system message at the beginning
        self._messages.insert(0, Message(role="system", content=prompt))
        
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Make an HTTP request to the API.
        
        Args:
            endpoint: API endpoint (e.g., '/models', '/chat/completions')
            data: Request body data (for POST requests)
            
        Returns:
            Parsed JSON response
            
        Raises:
            AuthenticationError: If token is invalid
            RateLimitError: If rate limit exceeded
            NetworkError: If network issues occur
            LLMClientError: For other API errors
        """
        if not self.access_token:
            raise AuthenticationError("Access token not set. Please configure your GPT@RUB token.")
            
        url = f"{self.api_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        
        try:
            if data:
                # POST request
                body = json.dumps(data).encode('utf-8')
                request = Request(url, data=body, headers=headers, method='POST')
            else:
                # GET request
                request = Request(url, headers=headers, method='GET')
            
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except HTTPError as e:
            if e.code == 401:
                raise AuthenticationError("Invalid access token. Please check your GPT@RUB token.")
            elif e.code == 429:
                raise RateLimitError("Rate limit exceeded (60 requests/minute). Please wait.")
            else:
                error_body = e.read().decode('utf-8') if e.readable() else str(e)
                raise LLMClientError(f"API error ({e.code}): {error_body}")
                
        except URLError as e:
            raise NetworkError(
                f"Network error: {e.reason}\n"
                "Make sure you are connected to RUB network or VPN."
            )
        except Exception as e:
            raise LLMClientError(f"Unexpected error: {e}")
            
    def get_models(self) -> List[ModelInfo]:
        """
        Get available models.
        
        Returns:
            List of ModelInfo objects
        """
        response = self._make_request("/models")
        models = []
        
        # Handle both formats: list directly or {"data": [...]}
        model_list = response if isinstance(response, list) else response.get("data", [])
        
        for model_data in model_list:
            models.append(ModelInfo(
                id=model_data.get("id", ""),
                name=model_data.get("name", model_data.get("id", "")),
                provider=model_data.get("provider", ""),
                cost_per_input_token=model_data.get("cost_per_input_token", 0),
                cost_per_output_token=model_data.get("cost_per_output_token", 0),
                max_context_length=model_data.get("max_context_length", 4096),
            ))
            
        return models
        
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        Send a chat message and get a response.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt (uses existing if not provided)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ChatCompletion object with the response
        """
        # Set system prompt if provided
        if system_prompt:
            self.set_system_prompt(system_prompt)
            
        # Add user message to history
        self._messages.append(Message(role="user", content=user_message))
        
        # Build request
        messages = [{"role": m.role, "content": m.content} for m in self._messages]
        
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            data["presence_penalty"] = kwargs["presence_penalty"]
            
        # Make request
        response = self._make_request("/chat/completions", data)
        
        # Parse response
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        
        # Add assistant response to history
        self._messages.append(Message(role="assistant", content=content))
        
        return ChatCompletion(
            content=content,
            model=response.get("model", self.model),
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )
        
    def chat_async(
        self,
        user_message: str,
        on_complete: Callable[[ChatCompletion], None],
        on_error: Callable[[Exception], None],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> threading.Thread:
        """
        Send a chat message asynchronously.
        
        Args:
            user_message: The user's message
            on_complete: Callback with ChatCompletion on success
            on_error: Callback with Exception on failure
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            The thread running the request
        """
        def run():
            try:
                result = self.chat(user_message, system_prompt=system_prompt, **kwargs)
                on_complete(result)
            except Exception as e:
                on_error(e)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
        
    def test_connection(self) -> bool:
        """
        Test the API connection.
        
        Returns:
            True if connection is successful
        """
        try:
            self.get_models()
            return True
        except Exception:
            return False
