# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Crash-safe JSON helpers shared by application project components."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pylcss.io_manager._atomic import PathLike, atomic_text_writer

__all__ = ["atomic_json_dump", "load_json_object"]

JsonObject = dict[str, Any]


def atomic_json_dump(
    value: Any,
    path: PathLike,
    *,
    indent: int = 2,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = False,
) -> Path:
    """Atomically replace ``path`` with standards-compliant UTF-8 JSON."""
    target = Path(path).expanduser()
    with atomic_text_writer(target, encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=indent,
            ensure_ascii=False,
            default=default,
            allow_nan=allow_nan,
        )
        handle.write("\n")
    return target


def load_json_object(
    path: PathLike,
    *,
    required_keys: Iterable[str] = (),
) -> JsonObject:
    """Load a JSON object and reject malformed or wrong-kind project data."""
    target = Path(path).expanduser()
    try:
        with target.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target.name} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{target.name} must contain a JSON object.")
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(
            f"{target.name} is missing required field(s): {', '.join(missing)}."
        )
    return value
