# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Parameter text shared by geometry nodes and the saved-study runtime."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import re
from typing import Any


PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SLOT_NAME_RE = re.compile(r"^param_([1-9]\d*)_name$")


def parse_parameter_text(text: str) -> dict[str, Any]:
    """Read literal dictionaries or name=value lines without executing code."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Parameter text must evaluate to a dict.")
        return {str(key): value for key, value in parsed.items()}

    params: dict[str, Any] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Parameter line {line_number} must use name=value.")
        name, raw_value = stripped.split("=", 1)
        name = name.strip()
        if not PARAMETER_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid parameter name on line {line_number}: {name!r}.")
        value = raw_value.strip()
        try:
            params[name] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                params[name] = float(value)
            except ValueError:
                params[name] = value
    return params


def format_parameter_text(params: Mapping[str, Any]) -> str:
    """Keep strings, booleans and containers intact when updating a scalar."""
    if any(not PARAMETER_NAME_RE.fullmatch(str(name)) for name in params):
        return repr(dict(params))
    return "\n".join(f"{name}={value!r}" for name, value in params.items())


def parameter_slot_indices(custom: Mapping[str, Any]) -> tuple[int, ...]:
    """Discover saved slots, respecting an explicit active socket count."""
    indices = sorted(
        int(match.group(1))
        for key in custom
        if (match := _SLOT_NAME_RE.fullmatch(str(key)))
    )
    try:
        count = max(0, int(custom["param_count"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return tuple(indices)
    return tuple(index for index in indices if index <= count)
