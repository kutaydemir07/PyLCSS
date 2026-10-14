# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Small data helpers shared by crash processing and optional validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def as_array(value: object) -> np.ndarray:
    """Return a flat float array, or an empty array for unavailable input."""
    try:
        return np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def sha256_file(path: str | Path | None) -> str | None:
    """Fingerprint an existing file without loading it all into memory."""
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Normalize nested data, including NumPy non-finite values, for JSON."""
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def assessment_status(statuses: Iterable[str]) -> str:
    """Keep unavailable evidence distinct from measured failures and passes."""
    values = list(statuses)
    if not values:
        return "not_evaluated"
    if "fail" in values:
        return "fail"
    if "warning" in values:
        return "warning"
    if any(value != "pass" for value in values):
        return "unverified"
    return "pass"
