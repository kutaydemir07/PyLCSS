# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Small PEP 562 helpers for lightweight package public APIs."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TypeAlias

LazyExport: TypeAlias = tuple[str, str]
LazyExports: TypeAlias = dict[str, LazyExport]


def load_attribute(
    name: str,
    exports: LazyExports,
    namespace: dict[str, object],
) -> object:
    """Load and cache one attribute declared by a package facade."""
    try:
        module_name, attribute_name = exports[name]
    except KeyError as exc:
        package_name = str(namespace.get("__name__", "module"))
        raise AttributeError(
            f"module {package_name!r} has no attribute {name!r}"
        ) from exc

    module: ModuleType = import_module(module_name)
    value = getattr(module, attribute_name)
    namespace[name] = value
    return value


def public_names(
    exports: LazyExports,
    namespace: dict[str, object],
) -> list[str]:
    """Return stable names for interactive discovery and documentation tools."""
    return sorted(set(namespace) | set(exports))
