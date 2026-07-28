# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Boundary validation shared by solver-specific deck writers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pylcss.solver_backends.base import SolverBackendError


def finite_float(value: Any, *, label: str) -> float:
    """Convert a value to a finite float with a user-actionable error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise SolverBackendError(f"{label} must be finite.")
    return result


def positive_float(value: Any, *, label: str) -> float:
    """Return a finite value greater than zero."""
    result = finite_float(value, label=label)
    if result <= 0.0:
        raise SolverBackendError(f"{label} must be greater than zero.")
    return result


def nonnegative_float(value: Any, *, label: str) -> float:
    """Return a finite value greater than or equal to zero."""
    result = finite_float(value, label=label)
    if result < 0.0:
        raise SolverBackendError(f"{label} must be non-negative.")
    return result


def integer(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact integer within optional inclusive bounds."""
    numeric = finite_float(value, label=label)
    if not numeric.is_integer():
        raise SolverBackendError(f"{label} must be an integer.")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise SolverBackendError(f"{label} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise SolverBackendError(f"{label} must not exceed {maximum}.")
    return result


def record_list(value: Any, *, label: str) -> list[dict[str, Any]]:
    """Normalize a sequence of mapping records at a public API boundary."""
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SolverBackendError(f"{label} must be a sequence of dictionaries.")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise SolverBackendError(f"{label} item {index} must be a dictionary.")
        records.append(dict(item))
    return records


def validate_isotropic_material(
    material: Mapping[str, Any],
    *,
    require_yield: bool = False,
    validate_strain_rate: bool = False,
) -> None:
    """Validate the material values required by both external backends."""
    if not isinstance(material, Mapping):
        raise SolverBackendError("Material must be a dictionary.")
    positive_float(
        material.get("E", 210000.0),
        label="Material Young's modulus",
    )
    poissons_ratio = finite_float(
        material.get("nu", material.get("poissons_ratio", 0.3)),
        label="Material Poisson's ratio",
    )
    if not -1.0 < poissons_ratio < 0.5:
        raise SolverBackendError(
            "Material Poisson's ratio must be between -1.0 and 0.5."
        )
    positive_float(
        material.get("rho", material.get("density", 7.85e-9)),
        label="Material density",
    )
    yield_strength = nonnegative_float(
        material.get("yield_strength", 0.0) or 0.0,
        label="Material yield strength",
    )
    nonnegative_float(
        material.get("tangent_modulus", 0.0) or 0.0,
        label="Material tangent modulus",
    )
    nonnegative_float(
        material.get("failure_strain", 0.0) or 0.0,
        label="Material failure strain",
    )
    if validate_strain_rate:
        strain_rate_c = nonnegative_float(
            material.get("strain_rate_c", 0.0) or 0.0,
            label="Material Cowper-Symonds C",
        )
        strain_rate_p = nonnegative_float(
            material.get("strain_rate_p", 0.0) or 0.0,
            label="Material Cowper-Symonds P",
        )
        if (strain_rate_c == 0.0) != (strain_rate_p == 0.0):
            raise SolverBackendError(
                "Material Cowper-Symonds C and P must either both be zero or both "
                "be greater than zero."
            )
    if require_yield and yield_strength <= 0.0:
        raise SolverBackendError(
            "Nonlinear plastic analysis requires a positive material yield strength."
        )
