# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Small result type shared by third-party optimization adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import FloatArray


@dataclass
class SolverResult:
    """Minimal result returned by numerical backend adapters."""

    x: FloatArray | None
    fun: float
    message: str
    success: bool = True


__all__ = ["SolverResult"]
