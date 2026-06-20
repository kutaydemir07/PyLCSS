# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Configuration and settings for the AI assistant system.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
# Adjusted for location in pylcss/assistant_systems/config/
ASSISTANT_DIR = Path(__file__).parent.parent
CONFIG_FILE = ASSISTANT_DIR / "config" / "settings.json"


@dataclass
class LLMControlConfig:
    """Configuration for the LLM assistant with multi-provider support."""
    enabled: bool = True

    # Provider selection: openai, anthropic, google, local
    provider: str = "google"

    # Encrypted API keys (stored encrypted in settings.json)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    local_api_key: str = ""
    local_api_url: str = "http://localhost:1234/v1"

    # Model selection (per provider, use selected_model for active)
    selected_model: str = ""
    model: str = "gemini-2.5-flash-lite"  # Updated default

    # Generation settings
    auto_execute: bool = False  # Require confirmation before executing actions
    max_tokens: int = 1000
    temperature: float = 0.7

    # Memory settings (hidden but functional)
    memory_enabled: bool = True
    max_memory_messages: int = 20  # Context window for LLM

    # Agentic AI: route requests through PydanticAgentRunner (native
    # function-calling).  Turned on by default; turning it off falls back to
    # the legacy single-shot path used only by the chat dialog.
    agentic_mode: bool = True
    # NOTE: the legacy flags `use_critic_agent`, `validate_design_intent`,
    # `max_retries`, and `auto_save_workflows` were tied to the deleted
    # orchestrator/workflow stack and had no effect at runtime; they were
    # removed in the PydanticAI migration.  Old settings.json files that
    # still contain them load fine -- AssistantConfig.load filters them out.

    def get_api_key_for_provider(self, provider: str = "") -> str:
        """Get the API key for a specific provider."""
        provider = provider or self.provider
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "local": self.local_api_key,
        }
        return key_map.get(provider, "")

    def set_api_key_for_provider(self, provider: str, key: str) -> None:
        """Set the API key for a specific provider."""
        if provider == "openai":
            self.openai_api_key = key
        elif provider == "anthropic":
            self.anthropic_api_key = key
        elif provider == "google":
            self.google_api_key = key
        elif provider == "local":
            self.local_api_key = key


@dataclass
class AssistantConfig:
    """Main configuration container for the AI assistant system."""
    llm_control: LLMControlConfig = field(default_factory=LLMControlConfig)

    # General settings
    startup_enabled: bool = False  # Auto-start on PyLCSS launch
    overlay_enabled: bool = True   # Show status overlay

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        save_path = path or CONFIG_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved assistant config to {save_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AssistantConfig":
        """Load configuration from JSON file.

        Legacy ``head_tracking`` / ``voice_control`` blocks written by older
        (voice-enabled) builds are silently ignored so old settings.json files
        keep loading after the voice assistant was removed.
        """
        load_path = path or CONFIG_FILE

        if not load_path.exists():
            logger.info("No config file found, using defaults")
            return cls()

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load LLM config
            llm_data = data.get('llm_control', {})

            # Validate provider - ensure it is one of the supported ones
            valid_providers = ['openai', 'anthropic', 'google', 'local']
            loaded_provider = llm_data.get('provider', 'google')
            if loaded_provider not in valid_providers:
                llm_data['provider'] = 'google'

            # Generic filtering of unknown keys to prevent __init__ errors
            valid_keys = set(LLMControlConfig.__dataclass_fields__.keys())
            llm_data = {k: v for k, v in llm_data.items() if k in valid_keys}

            config = cls(
                llm_control=LLMControlConfig(**llm_data),
                startup_enabled=data.get('startup_enabled', False),
                overlay_enabled=data.get('overlay_enabled', True),
            )
            logger.info(f"Loaded assistant config from {load_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return cls()
