# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Unit validation and affine conversion helpers."""

from typing import Any

import pint


class UnitError(ValueError):
    """Raised when a unit is invalid or two units are incompatible."""


UNIT_REGISTRY = pint.UnitRegistry()
_UNSPECIFIED_UNITS = {"", "-", None}


def is_specified_unit(unit: str | None) -> bool:
    """Return whether *unit* describes a real unit rather than a placeholder."""

    return unit not in _UNSPECIFIED_UNITS


def units_compatible(first: str, second: str) -> bool:
    """Return whether two specified unit expressions have equal dimensionality."""

    if not is_specified_unit(first) or not is_specified_unit(second):
        return True
    registry = UNIT_REGISTRY
    try:
        first_quantity = registry.Quantity(1, first)
        second_quantity = registry.Quantity(1, second)
    except Exception as exc:
        raise UnitError(f"Invalid unit expression: {first!r} or {second!r}.") from exc
    return first_quantity.dimensionality == second_quantity.dimensionality


def conversion_parameters(source: str, target: str) -> tuple[float, float]:
    """Return ``scale, offset`` for ``target = source * scale + offset``."""

    if (
        not is_specified_unit(source)
        or not is_specified_unit(target)
        or source == target
    ):
        return 1.0, 0.0
    if not units_compatible(source, target):
        raise UnitError(f"Incompatible units: {source!r} and {target!r}.")

    registry = UNIT_REGISTRY
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


def conversion_expression(expression: str, source: str, target: str) -> str:
    """Apply the same unit conversion to a generated Python expression."""

    scale, offset = conversion_parameters(source, target)
    if offset:
        return f"(({expression}) * {scale!r} + {offset!r})"
    if scale != 1:
        return f"(({expression}) * {scale!r})"
    return expression


__all__ = [
    "UNIT_REGISTRY",
    "UnitError",
    "conversion_parameters",
    "conversion_expression",
    "convert_value",
    "is_specified_unit",
    "units_compatible",
]
