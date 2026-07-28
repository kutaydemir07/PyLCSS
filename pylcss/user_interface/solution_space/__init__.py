# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Solution-space UI package with a lazy widget import."""

from __future__ import annotations

from typing import Any

__all__ = ["SolutionSpaceWidget"]


def __getattr__(name: str) -> Any:
    if name == "SolutionSpaceWidget":
        from .solution_space_widget import SolutionSpaceWidget

        return SolutionSpaceWidget
    raise AttributeError(name)
