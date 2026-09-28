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
    maximum_density: float = 1.0,
) -> np.ndarray:
    """Apply the compliance-specific optimality-criteria density update.

    ``maximum_density`` is the largest value a design variable may take. It is
    1.0 for a solid study, where the variable is a material indicator and
    saturating it means "solid here".

    A lattice study is the case it exists for. There the variable is not an
    indicator, it *is* the cell's relative density, and the homogenized law it
    drives was measured over a finite density range. Leaving the bound at 1.0
    lets the optimizer push most of the part past the top of that range, where
    the tabulated tensor is an extrapolation and the manufacturing step has no
    cell left to build: measured on the bundled studies, 35-71% of the envelope
    came back above the solid-transition level and the delivered "lattice" was
    83-90% dense. Capping the variable at the family's own maximum relative
    density keeps every element inside the law that was actually measured, and
    makes the delivered relative density the number the optimizer solved for.
    """
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

    ceiling = float(np.clip(maximum_density, 1e-2, 1.0))
    target = float(
        min(float(volume_fraction), ceiling)
    ) * float(x_active.size)
    move = 0.2
    lower = np.maximum(1e-3, x_active - move)
    upper = np.minimum(ceiling, x_active + move)
    # A restart or an initial design above the new ceiling would otherwise make
    # `lower > upper` and hand the bisection an empty interval.
    lower = np.minimum(lower, upper)

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
        # The passive material alone already meets or exceeds the whole volume
        # target, so the design variables have no budget left. This is not a
        # clamped target that still leaves a usable study — it is an infeasible
        # one, and the distinction matters because the solve does not fail
        # visibly: the optimizer simply cannot move, converges immediately on a
        # flat objective, and returns the passive regions as the "result". On
        # the bundled cold plate that produced a mounting frame and two module
        # pads standing apart with nothing between them, reported as a
        # converged design.
        "active_budget_exhausted": bool(
            raw_active_sum <= minimum_active_sum + 1e-9
        ),
    }


# Private aliases preserve the established test and extension API.
_oc_update = optimality_criteria_update
_restore_active_volume = restore_active_volume
_volume_budget_from_masks = volume_budget_from_masks
