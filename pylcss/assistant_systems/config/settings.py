# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Configuration and settings for the Assistant Control System.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
# Adjusted for location in pylcss/assistant_systems/config/
ASSISTANT_DIR = Path(__file__).parent.parent
models_dir = ASSISTANT_DIR.parent.parent / "models"
CONFIG_FILE = ASSISTANT_DIR / "config" / "settings.json"


@dataclass
class HeadTrackingConfig:
    """Configuration for head tracking system (deprecated - camera tracking removed)."""
    enabled: bool = False  # Disabled - camera head tracking removed
    camera_index: int = 0
    sensitivity_x: float = 1.0  # Multiplier for horizontal movement (was 2.5)
    sensitivity_y: float = 0.8  # Multiplier for vertical movement (was 2.0)
    deadzone: float = 0.03  # Minimum movement threshold (0-1)
    smoothing: float = 0.5  # Smoothing factor (0 = no smoothing, 1 = max)
    invert_x: bool = False
    invert_y: bool = False
    blink_threshold: float = 0.21  # Eye aspect ratio threshold for blink
    blink_frames: int = 3  # Consecutive frames to trigger click
    

@dataclass
class VoiceControlConfig:
    """Configuration for voice control system."""
    enabled: bool = True
    model_path: str = "base.en"
    sample_rate: int = 16000
    hotword: str = "hey computer"  # Wake word
    hotword_enabled: bool = False  # Always listen by default
    command_timeout: float = 5.0  # Seconds to wait for command after hotword
    feedback_enabled: bool = True  # Audio feedback on command recognition
    confirm_actions: bool = False  # Require voice confirmation for actions
    input_device_index: Optional[int] = None  # Specific audio input device index


@dataclass
class LLMControlConfig:
    """Configuration for LLM assistant control with multi-provider support."""
    enabled: bool = True
    
    # Provider selection: openai, anthropic, google
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
    
    # Agentic AI settings
    agentic_mode: bool = True  # Use multi-agent system (Planner → Executor → Critic)
    use_critic_agent: bool = True  # Enable critic for validation
    validate_design_intent: bool = True  # Critic validates design intent, not just geometry
    max_retries: int = 3  # Max retry attempts for self-correction
    auto_save_workflows: bool = True  # Auto-save successful multi-step operations
    
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
    """Main configuration container for assistant system."""
    head_tracking: HeadTrackingConfig = field(default_factory=HeadTrackingConfig)
    voice_control: VoiceControlConfig = field(default_factory=VoiceControlConfig)
    llm_control: LLMControlConfig = field(default_factory=LLMControlConfig)
    
    # General settings
    startup_enabled: bool = False  # Auto-start on PyLCSS launch
    overlay_enabled: bool = True  # Show status overlay
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        save_path = path or CONFIG_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved assistant config to {save_path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AssistantConfig":
        """Load configuration from JSON file."""
        load_path = path or CONFIG_FILE
        
        if not load_path.exists():
            logger.info("No config file found, using defaults")
            return cls()
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Helper to resolve model path
            vc_data = data.get('voice_control', {})
            if 'model_path' in vc_data:
                model_path_str = vc_data['model_path']
                model_path = Path(model_path_str)
                
                # Logic:
                # 1. If absolute and exists, keep it.
                # 2. If relative, try resolving against project root.
                # 3. If still not found, fallback to default.
                
                resolved_path = model_path
                if not model_path.is_absolute():
                     project_root = models_dir.parent
                     candidate = project_root / model_path
                     if candidate.exists():
                         resolved_path = candidate
                     else:
                         # Try resolving inside models dir implicitly
                         candidate_in_models = models_dir / model_path
                         if candidate_in_models.exists():
                             resolved_path = candidate_in_models
                
                # If the path (absolute or resolved relative) doesn't exist, we just keep it string
                # Faster-Whisper handles download or finding it.
                vc_data['model_path'] = str(resolved_path)

            # Load LLM config
            llm_data = data.get('llm_control', {})
            
            # Validate provider - ensure it is one of the supported ones
            valid_providers = ['openai', 'anthropic', 'google', 'local']
            loaded_provider = llm_data.get('provider', 'google')
            if loaded_provider not in valid_providers:
                llm_data['provider'] = 'google'
            
            # Generic filtering of unknown keys to prevent __init__ errors
            valid_keys = set(LLMControlConfig.__dataclass_fields__.keys())
            keys_to_remove = [k for k in llm_data.keys() if k not in valid_keys]
            
            for k in keys_to_remove:
                del llm_data[k]
            
            config = cls(
                head_tracking=HeadTrackingConfig(**data.get('head_tracking', {})),
                voice_control=VoiceControlConfig(**vc_data),
                llm_control=LLMControlConfig(**llm_data),
                startup_enabled=data.get('startup_enabled', False),
                overlay_enabled=data.get('overlay_enabled', True),
            )
            logger.info(f"Loaded assistant config from {load_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return cls()


# Voice command definitions are intentionally empty.
# Speech is routed to the AI assistant as natural language instead of matching
# hardcoded command phrases. LLM actions still use CommandDispatcher internally.
VOICE_COMMANDS: Dict[str, Dict] = {}
COMMAND_ALIASES: Dict[str, str] = {}
