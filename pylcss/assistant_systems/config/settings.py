# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Configuration for the AI assistant."""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pylcss.assistant_systems.persistence import atomic_write_json, read_json_object

logger = logging.getLogger(__name__)


def user_config_dir() -> Path:
    """Return a writable per-user configuration directory.

    Installed application files can be read-only and must never contain user
    credentials or chat history.  Keep this helper Qt-independent so headless
    and command-line uses resolve the same location as the desktop app.
    """
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "PyLCSS"
        return Path.home() / "AppData" / "Local" / "PyLCSS"
    root = os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home() / ".config") / "pylcss"


LEGACY_CONFIG_FILE = Path(__file__).with_name("settings.json")
CONFIG_FILE = user_config_dir() / "assistant" / "settings.json"
ProviderName = Literal["openai", "anthropic", "google", "local"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = (
    "openai",
    "anthropic",
    "google",
    "local",
)


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded_float(
    value: object,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _read_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return default


@dataclass(slots=True)
class LLMControlConfig:
    """Configuration for the LLM assistant with multi-provider support."""

    enabled: bool = True

    provider: ProviderName = "google"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    local_api_key: str = ""
    local_api_url: str = "http://localhost:1234/v1"

    selected_model: str = ""
    model: str = "gemini-2.5-flash-lite"

    auto_execute: bool = False
    max_tokens: int = 1000
    temperature: float = 0.7

    memory_enabled: bool = True
    max_memory_messages: int = 20

    agentic_mode: bool = True

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            self.provider = "google"
        self.enabled = _read_bool(self.enabled, True)
        self.auto_execute = _read_bool(self.auto_execute, False)
        self.memory_enabled = _read_bool(self.memory_enabled, True)
        self.agentic_mode = _read_bool(self.agentic_mode, True)
        self.max_tokens = _bounded_int(self.max_tokens, 1000, 1, 1_000_000)
        self.temperature = _bounded_float(self.temperature, 0.7, 0.0, 2.0)
        self.max_memory_messages = _bounded_int(
            self.max_memory_messages,
            20,
            1,
            200,
        )
        for attribute in (
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "local_api_key",
            "selected_model",
            "model",
        ):
            if not isinstance(getattr(self, attribute), str):
                setattr(self, attribute, "")
        if not isinstance(self.local_api_url, str) or not self.local_api_url.strip():
            self.local_api_url = "http://localhost:1234/v1"

    def get_api_key_for_provider(self, provider: str = "") -> str:
        """Get the API key for a specific provider."""
        selected = provider or self.provider
        key_map: dict[str, str] = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "local": self.local_api_key,
        }
        return key_map.get(selected, "")

    def set_api_key_for_provider(self, provider: str, key: str) -> None:
        """Set the API key for a specific provider."""
        attribute = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "local": "local_api_key",
        }.get(provider)
        if attribute is None:
            raise ValueError(f"Unsupported LLM provider: {provider!r}")
        setattr(self, attribute, key)


@dataclass(slots=True)
class AssistantConfig:
    """Main configuration container for the AI assistant system."""

    llm_control: LLMControlConfig = field(default_factory=LLMControlConfig)
    startup_enabled: bool = False
    overlay_enabled: bool = True

    def save(self, path: Path | None = None) -> None:
        """Atomically save configuration to a per-user JSON file."""
        save_path = path or CONFIG_FILE
        payload = {
            "_copyright": "Copyright (c) 2026 Kutay Demir.",
            "_license": (
                "Licensed under the PolyForm Shield License 1.0.0. "
                "See LICENSE file for details."
            ),
            **asdict(self),
        }
        atomic_write_json(save_path, payload)
        logger.info("Saved assistant config to %s", save_path)

    @classmethod
    def load(cls, path: Path | None = None) -> "AssistantConfig":
        """Load configuration from JSON file.

        Legacy ``head_tracking`` / ``voice_control`` blocks written by older
        (voice-enabled) builds are silently ignored so old settings.json files
        keep loading after the voice assistant was removed.
        """
        load_path = path or CONFIG_FILE
        legacy_source = False
        if path is None and not load_path.exists() and LEGACY_CONFIG_FILE.exists():
            load_path = LEGACY_CONFIG_FILE
            legacy_source = True

        if not load_path.exists():
            logger.info("No config file found, using defaults")
            return cls()

        try:
            data = read_json_object(load_path)
            raw_llm_data = data.get("llm_control", {})
            if not isinstance(raw_llm_data, dict):
                raise ValueError("llm_control must be a JSON object")
            valid_keys = set(LLMControlConfig.__dataclass_fields__)
            llm_data = {
                key: value for key, value in raw_llm_data.items() if key in valid_keys
            }

            config = cls(
                llm_control=LLMControlConfig(**llm_data),
                startup_enabled=_read_bool(data.get("startup_enabled"), False),
                overlay_enabled=_read_bool(data.get("overlay_enabled"), True),
            )
            logger.info("Loaded assistant config from %s", load_path)
            if legacy_source:
                try:
                    config.save(CONFIG_FILE)
                    logger.info("Migrated assistant settings to %s", CONFIG_FILE)
                except Exception as exc:
                    logger.warning(
                        "Could not migrate legacy assistant settings: %s", exc
                    )
            return config
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load config (%s); using defaults", exc)
            return cls()
