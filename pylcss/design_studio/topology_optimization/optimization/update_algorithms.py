# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Density update algorithms and passive-volume accounting."""

from __future__ import annotations

from typing import Optional

import numpy as np


def optimality_criteria_update(
    x: np.ndarray,
    gradient: np.ndarray,
    volume_fraction: float,
    active_mask: Optional[np.ndarray] = None,
    passive_density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the compliance-specific optimality-criteria density update."""
    active = (
        np.ones_like(x, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    x_active = np.asarray(x, dtype=float)[active]
    gradient_active = np.asarray(gradient, dtype=float)[active]
    if x_active.size == 0:
        updated = np.asarray(x, dtype=float).copy()
        if passive_density is not None:
            updated[~active] = np.asarray(passive_density, dtype=float)[~active]
        return updated

    target = float(volume_fraction) * float(x_active.size)
    move = 0.2
    lower = np.maximum(1e-3, x_active - move)
    upper = np.minimum(1.0, x_active + move)

    def candidate(multiplier: float) -> np.ndarray:
        ratio = np.maximum(-gradient_active / multiplier, 0.0)
        return np.clip(x_active * np.sqrt(ratio), lower, upper)

    lower_multiplier = upper_multiplier = 1e-9
    for _ in range(200):
        if float(np.sum(candidate(upper_multiplier))) <= target:
            break
        lower_multiplier = upper_multiplier
        upper_multiplier *= 2.0
    for _ in range(200):
        if float(np.sum(candidate(lower_multiplier))) >= target:
            break
        upper_multiplier = lower_multiplier
        lower_multiplier *= 0.5

    updated_active = candidate(upper_multiplier)
    while upper_multiplier - lower_multiplier > 1e-8 * (
        lower_multiplier + upper_multiplier
    ):
        midpoint = 0.5 * (lower_multiplier + upper_multiplier)
        updated_active = candidate(midpoint)
        if float(np.sum(updated_active)) > target:
            lower_multiplier = midpoint
        else:
            upper_multiplier = midpoint

    updated = np.asarray(x, dtype=float).copy()
    updated[active] = updated_active
    if passive_density is not None:
        updated[~active] = np.asarray(passive_density, dtype=float)[~active]
    return updated


def restore_active_volume(
    x: np.ndarray,
    active_mask: np.ndarray,
    volume_fraction: float,
    passive_density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Restore active volume after a non-volume-preserving projection."""
    updated = np.asarray(x, dtype=float).copy()
    active = np.asarray(active_mask, dtype=bool)
    if passive_density is not None:
        updated[~active] = np.asarray(passive_density, dtype=float)[~active]
    if not np.any(active):
        return updated

    target = float(np.clip(volume_fraction, 1e-3, 1.0)) * float(np.sum(active))
    lower, upper = -1.0, 1.0
    active_values = updated[active]
    tolerance = 1e-9 * max(target, 1.0)
    midpoint = 0.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        shifted = np.clip(active_values + midpoint, 1e-3, 1.0)
        current = float(np.sum(shifted))
        if abs(current - target) <= tolerance:
            break
        if current < target:
            lower = midpoint
        else:
            upper = midpoint
    updated[active] = np.clip(active_values + midpoint, 1e-3, 1.0)
    if passive_density is not None:
        updated[~active] = np.asarray(passive_density, dtype=float)[~active]
    return updated


def projected_gradient_update(
    x: np.ndarray,
    gradient: np.ndarray,
    volume_fraction: float,
    *,
    active_mask: Optional[np.ndarray] = None,
    passive_density: Optional[np.ndarray] = None,
    step: float = 0.15,
    move: float = 0.2,
) -> np.ndarray:
    """Apply a scaled projected-gradient step and exact volume projection."""
    values = np.asarray(x, dtype=float)
    grad = np.asarray(gradient, dtype=float)
    active = (
        np.ones_like(values, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if not np.any(active):
        updated = values.copy()
        if passive_density is not None:
            updated[~active] = np.asarray(passive_density, dtype=float)[~active]
        return updated

    finite_gradient = np.where(np.isfinite(grad[active]), grad[active], 0.0)
    scale = float(np.max(np.abs(finite_gradient)))
    if scale <= 1e-30:
        updated = values.copy()
    else:
        direction = np.zeros_like(values)
        direction[active] = -finite_gradient / scale
        lower = np.maximum(1e-3, values - float(move))
        upper = np.minimum(1.0, values + float(move))
        updated = np.clip(values + float(step) * direction, lower, upper)

    return restore_active_volume(
        updated,
        active,
        volume_fraction,
        passive_density=passive_density,
    )


def volume_budget_from_masks(
    volume_fraction: float,
    active_mask: np.ndarray,
    passive_density: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    *,
    min_density: float = 1e-3,
) -> dict[str, float | bool]:
    """Translate a source-domain target into an active-material budget."""
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    passive = np.asarray(passive_density, dtype=float).reshape(-1)
    if passive.size != active.size:
        passive = np.resize(passive, active.size)
    if source_mask is None:
        source = np.ones(active.size, dtype=bool)
    else:
        source = np.asarray(source_mask, dtype=bool).reshape(-1)
        if source.size != active.size:
            source = np.ones(active.size, dtype=bool)

    source_count = max(1, int(np.sum(source)))
    active_source = active & source
    passive_source = (~active) & source
    passive_outside = (~active) & (~source)

    active_count = int(np.sum(active_source))
    target_source_sum = float(np.clip(volume_fraction, min_density, 1.0)) * float(
        source_count
    )
    passive_source_sum = float(np.sum(passive[passive_source]))
    passive_outside_sum = float(np.sum(passive[passive_outside]))

    minimum_active_sum = float(min_density) * float(active_count)
    raw_active_sum = target_source_sum - passive_source_sum
    feasible_active_sum = float(
        np.clip(raw_active_sum, minimum_active_sum, float(active_count))
    )
    active_volume_fraction = (
        feasible_active_sum / float(active_count)
        if active_count > 0
        else float(min_density)
    )
    source_total_sum = passive_source_sum + feasible_active_sum
    flat_total_target = source_total_sum + passive_outside_sum
    minimum_source_sum = passive_source_sum + minimum_active_sum

    return {
        "active_volfrac": float(np.clip(active_volume_fraction, min_density, 1.0)),
        "flat_total_target": float(flat_total_target),
        "source_total_target": float(source_total_sum),
        "source_count": float(source_count),
        "active_count": float(active_count),
        "passive_source_sum": float(passive_source_sum),
        "min_source_volfrac": float(minimum_source_sum / float(source_count)),
        "target_was_clamped": bool(abs(feasible_active_sum - raw_active_sum) > 1e-9),
    }


# Private aliases preserve the established test and extension API.
_oc_update = optimality_criteria_update
_restore_active_volume = restore_active_volume
_projected_gradient_update = projected_gradient_update
_volume_budget_from_masks = volume_budget_from_masks
