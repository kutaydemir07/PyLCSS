# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Thread-safe persistence for assistant conversation history."""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pylcss.assistant_systems.config.settings import user_config_dir
from pylcss.assistant_systems.persistence import atomic_write_json, read_json_object

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]
_VALID_ROLES = frozenset({"system", "user", "assistant"})
_ASSISTANT_DATA_DIR = user_config_dir() / "assistant"
MEMORY_FILE = _ASSISTANT_DATA_DIR / "llm_memory.json"
LEGACY_MEMORY_FILE = Path(__file__).parent.parent / "config" / "llm_memory.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ConversationMessage:
    """One validated message in a stored conversation."""

    role: Role
    content: str
    timestamp: str = ""
    model: str = ""
    provider: str = ""

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"Unsupported conversation role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("Conversation content must be text")
        if not self.timestamp:
            self.timestamp = _timestamp()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        role = str(data.get("role", "user"))
        if role not in _VALID_ROLES:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        return cls(
            role=cast(Role, role),
            content=str(data.get("content", "")),
            timestamp=str(data.get("timestamp", "")),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "")),
        )


@dataclass(slots=True)
class Conversation:
    """A persisted assistant conversation."""

    id: str = ""
    title: str = ""
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    provider: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"conv_{uuid.uuid4().hex}"
        if not self.created_at:
            self.created_at = _timestamp()
        if not self.updated_at:
            self.updated_at = self.created_at

    def add_message(
        self,
        role: Role,
        content: str,
        model: str = "",
        provider: str = "",
    ) -> None:
        """Append a message and update the conversation metadata."""
        self.messages.append(
            ConversationMessage(
                role=role,
                content=content,
                model=model or self.model,
                provider=provider or self.provider,
            )
        )
        self.updated_at = _timestamp()
        if not self.title and role == "user":
            normalized = " ".join(content.split())
            self.title = normalized[:50] + ("..." if len(normalized) > 50 else "")

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Return provider-neutral role/content dictionaries."""
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [asdict(message) for message in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        conversation = cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
        )
        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("Conversation messages must be a list")
        for raw_message in raw_messages:
            if isinstance(raw_message, dict):
                conversation.messages.append(ConversationMessage.from_dict(raw_message))
        return conversation


class LLMMemory:
    """Manage bounded, persistent conversation history."""

    def __init__(
        self,
        storage_path: Path | None = None,
        max_conversations: int = 100,
    ) -> None:
        if max_conversations < 1:
            raise ValueError("max_conversations must be at least 1")
        self.storage_path = Path(storage_path) if storage_path else MEMORY_FILE
        self.max_conversations = max_conversations
        self._conversations: dict[str, Conversation] = {}
        self._current_conversation_id: str | None = None
        self._lock = threading.RLock()
        self._migrate_legacy_file(storage_path)
        self._load()

    def _migrate_legacy_file(self, explicit_path: Path | None) -> None:
        if (
            explicit_path is None
            and not self.storage_path.exists()
            and LEGACY_MEMORY_FILE.is_file()
        ):
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(LEGACY_MEMORY_FILE, self.storage_path)
            except OSError as exc:
                logger.warning("Could not migrate legacy assistant memory: %s", exc)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = read_json_object(self.storage_path)
        except (OSError, ValueError) as exc:
            logger.error("Could not load assistant memory: %s", exc)
            return

        raw_conversations = data.get("conversations", [])
        if not isinstance(raw_conversations, list):
            logger.error(
                "Ignoring malformed assistant memory: conversations is not a list"
            )
            return

        with self._lock:
            for raw_conversation in raw_conversations:
                if not isinstance(raw_conversation, dict):
                    continue
                try:
                    conversation = Conversation.from_dict(raw_conversation)
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed conversation: %s", exc)
                    continue
                self._conversations[conversation.id] = conversation

            current_id = data.get("current_conversation_id")
            if isinstance(current_id, str) and current_id in self._conversations:
                self._current_conversation_id = current_id
            self._prune()

    def _prune(self) -> None:
        if len(self._conversations) <= self.max_conversations:
            return
        recent = sorted(
            self._conversations.values(),
            key=lambda conversation: conversation.updated_at,
            reverse=True,
        )[: self.max_conversations]
        self._conversations = {conversation.id: conversation for conversation in recent}
        if self._current_conversation_id not in self._conversations:
            self._current_conversation_id = recent[0].id if recent else None

    def _save(self) -> None:
        with self._lock:
            self._prune()
            payload = {
                "_copyright": "Copyright (c) 2026 Kutay Demir.",
                "_license": (
                    "Licensed under the PolyForm Shield License 1.0.0. "
                    "See LICENSE file for details."
                ),
                "conversations": [
                    conversation.to_dict()
                    for conversation in self._conversations.values()
                ],
                "current_conversation_id": self._current_conversation_id,
            }
            try:
                atomic_write_json(self.storage_path, payload)
            except OSError as exc:
                logger.error("Could not save assistant memory: %s", exc)

    def new_conversation(self, provider: str = "", model: str = "") -> Conversation:
        with self._lock:
            conversation = Conversation(provider=provider, model=model)
            self._conversations[conversation.id] = conversation
            self._current_conversation_id = conversation.id
            self._save()
            return conversation

    def get_current_conversation(self) -> Conversation | None:
        with self._lock:
            if self._current_conversation_id is None:
                return None
            return self._conversations.get(self._current_conversation_id)

    def get_or_create_current(
        self, provider: str = "", model: str = ""
    ) -> Conversation:
        conversation = self.get_current_conversation()
        return conversation or self.new_conversation(provider, model)

    def add_message(
        self,
        role: Role,
        content: str,
        model: str = "",
        provider: str = "",
        conversation_id: str | None = None,
    ) -> None:
        with self._lock:
            target_id = conversation_id or self._current_conversation_id
            if target_id not in self._conversations:
                target_id = self.new_conversation(provider, model).id
            self._conversations[target_id].add_message(role, content, model, provider)
            self._save()

    def get_context_messages(self, max_messages: int = 20) -> list[dict[str, str]]:
        if max_messages < 1:
            return []
        with self._lock:
            conversation = self.get_current_conversation()
            if conversation is None:
                return []
            messages = conversation.get_messages_for_llm()
            system_messages = [
                message for message in messages if message["role"] == "system"
            ][:max_messages]
            remaining = max_messages - len(system_messages)
            if remaining == 0:
                return system_messages
            non_system_messages = [
                message for message in messages if message["role"] != "system"
            ]
            return system_messages + non_system_messages[-remaining:]

    def clear_current_conversation(self) -> None:
        with self._lock:
            if self._current_conversation_id is not None:
                self._conversations.pop(self._current_conversation_id, None)
                self._current_conversation_id = None
                self._save()

    def get_conversation_count(self) -> int:
        with self._lock:
            return len(self._conversations)

    def get_recent_conversations(self, limit: int = 10) -> list[Conversation]:
        if limit <= 0:
            return []
        with self._lock:
            return sorted(
                self._conversations.values(),
                key=lambda conversation: conversation.updated_at,
                reverse=True,
            )[:limit]

    def clear_all(self) -> None:
        with self._lock:
            self._conversations.clear()
            self._current_conversation_id = None
            self._save()
