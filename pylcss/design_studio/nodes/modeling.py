# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility facade for the former selection-node module name.

New code should import from :mod:`pylcss.design_studio.nodes.selection`.
"""

from __future__ import annotations

from pylcss.design_studio.nodes import selection as _selection

InteractiveSelectFaceNode = _selection.InteractiveSelectFaceNode
SelectFaceNode = _selection.SelectFaceNode

__all__ = ["InteractiveSelectFaceNode", "SelectFaceNode"]


def __getattr__(name: str) -> object:
    return getattr(_selection, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_selection)))
