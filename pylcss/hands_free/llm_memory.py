# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Encrypted API Key Storage and LLM Memory System.

Provides secure storage for API keys using encryption and
persistent conversation memory for LLM context.
"""

import base64
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Directory for memory storage
HANDS_FREE_DIR = Path(__file__).parent
MEMORY_FILE = HANDS_FREE_DIR / "llm_memory.json"
ENCRYPTION_KEY_FILE = HANDS_FREE_DIR / ".llm_key"


class SecureKeyStorage:
    """
    Secure storage for API keys using encryption.
    
    Uses a machine-specific key derived from hardware identifiers
    combined with a random salt for encryption.
    """
    
    def __init__(self):
        self._key = self._get_or_create_key()
    
    def _get_machine_id(self) -> str:
        """Get a machine-specific identifier."""
        try:
            # Try to get username + hostname as a base
            import socket
            machine_info = f"{os.getlogin()}@{socket.gethostname()}"
            return hashlib.sha256(machine_info.encode()).hexdigest()[:32]
        except Exception:
            # Fallback to a fixed string if we can't get machine info
            return "pylcss_default_key_base_2026"
    
    def _get_or_create_key(self) -> bytes:
        """Get or create the encryption key."""
        if ENCRYPTION_KEY_FILE.exists():
            try:
                with open(ENCRYPTION_KEY_FILE, 'rb') as f:
                    salt = f.read()
            except Exception:
                salt = os.urandom(32)
                self._save_salt(salt)
        else:
            salt = os.urandom(32)
            self._save_salt(salt)
        
        # Derive key from machine ID + salt
        machine_id = self._get_machine_id()
        key_material = machine_id.encode() + salt
        return hashlib.sha256(key_material).digest()
    
    def _save_salt(self, salt: bytes) -> None:
        """Save the salt to file."""
        try:
            with open(ENCRYPTION_KEY_FILE, 'wb') as f:
                f.write(salt)
        except Exception as e:
            logger.warning(f"Could not save encryption salt: {e}")
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string using XOR cipher with the derived key.
        
        This is a simple encryption suitable for local storage.
        For production, consider using Fernet from cryptography library.
        """
        if not plaintext:
            return ""
        
        # Simple XOR encryption with key cycling
        key_bytes = self._key
        plaintext_bytes = plaintext.encode('utf-8')
        
        encrypted = bytes([
            plaintext_bytes[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(plaintext_bytes))
        ])
        
        # Base64 encode for safe JSON storage
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string encrypted with encrypt().
        
        Falls back to returning the original string if decryption fails,
        which handles legacy plaintext keys.
        """
        if not ciphertext:
            return ""
        
        try:
            encrypted = base64.b64decode(ciphertext.encode('ascii'))
            key_bytes = self._key
            
            decrypted = bytes([
                encrypted[i] ^ key_bytes[i % len(key_bytes)]
                for i in range(len(encrypted))
            ])
            
            result = decrypted.decode('utf-8')
            # Validate it looks like an API key (non-empty, printable)
            if result and result.isprintable():
                return result
            else:
                # Decryption produced garbage, return original (plaintext key)
                logger.debug("Decrypted value not printable, using original as plaintext")
                return ciphertext
        except Exception as e:
            # Decryption failed - the ciphertext is likely a plaintext key
            logger.debug(f"Decryption failed (key likely plaintext): {e}")
            return ciphertext  # Return original as-is


# Global secure storage instance
_secure_storage: Optional[SecureKeyStorage] = None

def get_secure_storage() -> SecureKeyStorage:
    """Get the global secure storage instance."""
    global _secure_storage
    if _secure_storage is None:
        _secure_storage = SecureKeyStorage()
    return _secure_storage


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: str = ""
    model: str = ""
    provider: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Conversation:
    """A conversation session with messages."""
    id: str = ""
    title: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    provider: str = ""
    model: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def add_message(self, role: str, content: str, model: str = "", provider: str = "") -> None:
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            model=model or self.model,
            provider=provider or self.provider,
        ))
        self.updated_at = datetime.now().isoformat()
        
        # Auto-generate title from first user message
        if not self.title:
            for msg in self.messages:
                if msg.role == "user":
                    # Take first 50 chars of first user message
                    self.title = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
                    break
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages in LLM-compatible format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [asdict(m) for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        """Create from dictionary."""
        conv = cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )
        for msg_data in data.get("messages", []):
            conv.messages.append(ConversationMessage(**msg_data))
        return conv


class LLMMemory:
    """
    Persistent conversation memory manager.
    
    Stores conversations in JSON format for context preservation
    across sessions without visible UI.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, max_conversations: int = 100):
        self.storage_path = storage_path or MEMORY_FILE
        self.max_conversations = max_conversations
        self._conversations: Dict[str, Conversation] = {}
        self._current_conversation_id: Optional[str] = None
        self._load()
    
    def _load(self) -> None:
        """Load conversations from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for conv_data in data.get("conversations", []):
                conv = Conversation.from_dict(conv_data)
                self._conversations[conv.id] = conv
            
            self._current_conversation_id = data.get("current_conversation_id")
            logger.info(f"Loaded {len(self._conversations)} conversations from memory")
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def _save(self) -> None:
        """Save conversations to storage."""
        try:
            # Limit number of conversations
            if len(self._conversations) > self.max_conversations:
                # Sort by updated_at and keep most recent
                sorted_convs = sorted(
                    self._conversations.values(),
                    key=lambda c: c.updated_at,
                    reverse=True
                )
                self._conversations = {c.id: c for c in sorted_convs[:self.max_conversations]}
            
            data = {
                "conversations": [c.to_dict() for c in self._conversations.values()],
                "current_conversation_id": self._current_conversation_id,
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def new_conversation(self, provider: str = "", model: str = "") -> Conversation:
        """Create a new conversation."""
        conv = Conversation(provider=provider, model=model)
        self._conversations[conv.id] = conv
        self._current_conversation_id = conv.id
        self._save()
        logger.info(f"Created new conversation: {conv.id}")
        return conv
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current conversation."""
        if self._current_conversation_id:
            return self._conversations.get(self._current_conversation_id)
        return None
    
    def get_or_create_current(self, provider: str = "", model: str = "") -> Conversation:
        """Get current conversation or create a new one."""
        conv = self.get_current_conversation()
        if conv is None:
            conv = self.new_conversation(provider, model)
        return conv
    
    def add_message(
        self,
        role: str,
        content: str,
        model: str = "",
        provider: str = "",
        conversation_id: Optional[str] = None
    ) -> None:
        """Add a message to a conversation."""
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id or conv_id not in self._conversations:
            conv = self.new_conversation(provider, model)
            conv_id = conv.id
        
        self._conversations[conv_id].add_message(role, content, model, provider)
        self._save()
    
    def get_context_messages(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """
        Get recent messages for LLM context.
        
        Returns messages from current conversation, limited to max_messages.
        """
        conv = self.get_current_conversation()
        if not conv:
            return []
        
        messages = conv.get_messages_for_llm()
        
        # Keep system message + last N messages
        if len(messages) > max_messages:
            system_msgs = [m for m in messages if m["role"] == "system"]
            other_msgs = [m for m in messages if m["role"] != "system"]
            return system_msgs + other_msgs[-(max_messages - len(system_msgs)):]
        
        return messages
    
    def clear_current_conversation(self) -> None:
        """Clear current conversation and start fresh."""
        if self._current_conversation_id:
            if self._current_conversation_id in self._conversations:
                del self._conversations[self._current_conversation_id]
            self._current_conversation_id = None
            self._save()
    
    def get_conversation_count(self) -> int:
        """Get total number of stored conversations."""
        return len(self._conversations)
    
    def get_recent_conversations(self, limit: int = 10) -> List[Conversation]:
        """Get most recent conversations."""
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
        return sorted_convs[:limit]
    
    def clear_all(self) -> None:
        """Clear all stored conversations."""
        self._conversations = {}
        self._current_conversation_id = None
        self._save()
        logger.info("Cleared all conversation memory")

