# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Primitive coercion helpers for persisted and UI configuration."""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_boolean(value: Any, label: str = "Value") -> bool:
    """Parse a strict boolean from a UI/JSON-compatible value."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{label} must be a boolean.")


__all__ = ["parse_boolean"]
