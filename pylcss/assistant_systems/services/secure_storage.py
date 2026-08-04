# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Authenticated, machine-bound encryption for local assistant API keys."""

from __future__ import annotations

import base64
import getpass
import hashlib
import logging
import os
import platform
import shutil
import threading
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from pylcss.assistant_systems.config.settings import user_config_dir

logger = logging.getLogger(__name__)

_ASSISTANT_DATA_DIR = user_config_dir() / "assistant"
ENCRYPTION_KEY_FILE = _ASSISTANT_DATA_DIR / ".llm_key"
LEGACY_ENCRYPTION_KEY_FILE = Path(__file__).parent.parent / "config" / ".llm_key"
_CIPHERTEXT_PREFIX = "fernet:"
_SALT_BYTES = 32


class SecureKeyStorage:
    """Encrypt API keys at rest using authenticated Fernet encryption."""

    def __init__(self, salt_path: Path | None = None) -> None:
        self._salt_path = salt_path or ENCRYPTION_KEY_FILE
        self._salt = self._get_or_create_salt()
        material = self._machine_id().encode("utf-8") + self._salt
        key = base64.urlsafe_b64encode(hashlib.sha256(material).digest())
        self._fernet = Fernet(key)

    @staticmethod
    def _machine_id() -> str:
        identity = f"{getpass.getuser()}@{platform.node()}".strip("@")
        return identity or "pylcss-local-user"

    def _get_or_create_salt(self) -> bytes:
        if (
            self._salt_path == ENCRYPTION_KEY_FILE
            and not self._salt_path.exists()
            and LEGACY_ENCRYPTION_KEY_FILE.is_file()
        ):
            try:
                self._salt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(LEGACY_ENCRYPTION_KEY_FILE, self._salt_path)
            except OSError as exc:
                logger.warning("Could not migrate the legacy key salt: %s", exc)

        try:
            salt = self._salt_path.read_bytes()
            if len(salt) == _SALT_BYTES:
                return salt
            logger.warning("Ignoring an invalid assistant key salt")
        except FileNotFoundError:
            logger.debug("No assistant encryption salt exists yet")
        except OSError as exc:
            logger.warning("Could not read the assistant key salt: %s", exc)

        salt = os.urandom(_SALT_BYTES)
        try:
            self._write_salt(salt)
        except OSError as exc:
            logger.warning(
                "Could not persist the assistant encryption salt; "
                "keys will only be readable in this process: %s",
                exc,
            )
        return salt

    def _write_salt(self, salt: bytes) -> None:
        self._salt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._salt_path.with_name(f".{self._salt_path.name}.tmp")
        try:
            temporary.write_bytes(salt)
            if os.name != "nt":
                temporary.chmod(0o600)
            os.replace(temporary, self._salt_path)
        finally:
            temporary.unlink(missing_ok=True)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a value and mark it with the current storage format."""
        if not plaintext:
            return ""
        token = self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return f"{_CIPHERTEXT_PREFIX}{token}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt current values and read legacy XOR/plaintext values."""
        if not ciphertext:
            return ""
        if ciphertext.startswith(_CIPHERTEXT_PREFIX):
            token = ciphertext.removeprefix(_CIPHERTEXT_PREFIX)
            try:
                return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
            except (InvalidToken, UnicodeError, ValueError) as exc:
                raise ValueError(
                    "The stored API key cannot be decrypted on this machine"
                ) from exc
        return self._decrypt_legacy_or_plaintext(ciphertext)

    def _decrypt_legacy_or_plaintext(self, value: str) -> str:
        try:
            encrypted = base64.b64decode(value.encode("ascii"), validate=True)
            decrypted = bytes(
                byte ^ self._legacy_key[index % len(self._legacy_key)]
                for index, byte in enumerate(encrypted)
            ).decode("utf-8")
            if decrypted and decrypted.isprintable():
                return decrypted
        except (ValueError, UnicodeError):
            return value
        return value

    @property
    def _legacy_key(self) -> bytes:
        try:
            import socket

            identity = f"{os.getlogin()}@{socket.gethostname()}"
            machine_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
        except OSError:
            machine_id = "pylcss_default_key_base_2026"
        material = machine_id.encode("utf-8") + self._salt
        return hashlib.sha256(material).digest()


_storage: SecureKeyStorage | None = None
_storage_lock = threading.Lock()


def get_secure_storage() -> SecureKeyStorage:
    """Return the process-wide key storage instance."""
    global _storage
    with _storage_lock:
        if _storage is None:
            _storage = SecureKeyStorage()
        return _storage
