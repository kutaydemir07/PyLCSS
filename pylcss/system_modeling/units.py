# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Unit validation and affine conversion helpers."""

from typing import Any

try:
    import pint as _pint
except ImportError:  # pragma: no cover - pint is a declared runtime dependency.
    _pint = None  # type: ignore[assignment]


class UnitError(ValueError):
    """Raised when a unit is invalid or two units are incompatible."""


UNIT_REGISTRY: Any | None = _pint.UnitRegistry() if _pint is not None else None
_UNSPECIFIED_UNITS = {"", "-", None}


def is_specified_unit(unit: str | None) -> bool:
    """Return whether *unit* describes a real unit rather than a placeholder."""

    return unit not in _UNSPECIFIED_UNITS


def _require_registry() -> Any:
    if UNIT_REGISTRY is None:
        raise UnitError("Unit support is unavailable because pint is not installed.")
    return UNIT_REGISTRY


def units_compatible(first: str, second: str) -> bool:
    """Return whether two specified unit expressions have equal dimensionality."""

    if not is_specified_unit(first) or not is_specified_unit(second):
        return True
    registry = _require_registry()
    try:
        first_quantity = registry.Quantity(1, first)
        second_quantity = registry.Quantity(1, second)
    except Exception as exc:
        raise UnitError(f"Invalid unit expression: {first!r} or {second!r}.") from exc
    return first_quantity.dimensionality == second_quantity.dimensionality


def conversion_parameters(source: str, target: str) -> tuple[float, float]:
    """Return ``scale, offset`` for ``target = source * scale + offset``."""

    if not is_specified_unit(source) or not is_specified_unit(target) or source == target:
        return 1.0, 0.0
    if not units_compatible(source, target):
        raise UnitError(f"Incompatible units: {source!r} and {target!r}.")

    registry = _require_registry()
    try:
        zero = registry.Quantity(0, source).to(target).magnitude
        one = registry.Quantity(1, source).to(target).magnitude
    except Exception as exc:
        raise UnitError(f"Cannot convert from {source!r} to {target!r}.") from exc
    return float(one - zero), float(zero)


def convert_value(value: Any, source: str, target: str) -> Any:
    """Convert a scalar or NumPy-compatible value between units."""

    scale, offset = conversion_parameters(source, target)
    return value * scale + offset


__all__ = [
    "UNIT_REGISTRY",
    "UnitError",
    "conversion_parameters",
    "convert_value",
    "is_specified_unit",
    "units_compatible",
]
