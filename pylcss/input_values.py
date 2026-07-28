# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Normalization helpers for values received from node-graph and UI inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def flatten_inputs(items: Iterable[Any] | None) -> list[Any]:
    """Flatten nested list/tuple inputs while preserving order and non-None values."""
    result: list[Any] = []
    for item in items or ():
        if isinstance(item, (list, tuple)):
            result.extend(flatten_inputs(item))
        elif item is not None:
            result.append(item)
    return result


def as_bool(value: Any) -> bool:
    """Interpret boolean-like UI values using the application's legacy rules."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    return bool(value)
