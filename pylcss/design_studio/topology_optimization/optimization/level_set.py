# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Reaction-diffusion level-set updates for voxel topology optimization.

The design variable is a signed scalar field ``phi``. Material occupies the
positive phase and void occupies the negative phase. A smooth Heaviside map is
used only as an ersatz-material interface for finite-element analysis; the
evolving optimization variable remains the signed level-set field.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage


LEVEL_SET_BETA = 4.0
LEVEL_SET_MIN_DENSITY = 1e-3
LEVEL_SET_TIME_STEP = 0.22
LEVEL_SET_REGULARIZATION = 0.16


def level_set_heaviside(
    phi: np.ndarray,
    *,
    beta: float = LEVEL_SET_BETA,
    min_density: float = LEVEL_SET_MIN_DENSITY,
) -> np.ndarray:
    """Map a signed level-set field to an ersatz physical density."""
    values = np.asarray(phi, dtype=float)
    phase = 0.5 * (1.0 + np.tanh(float(beta) * values))
    floor = float(np.clip(min_density, 0.0, 1.0))
    return floor + (1.0 - floor) * phase


def level_set_heaviside_derivative(
    phi: np.ndarray,
    *,
    beta: float = LEVEL_SET_BETA,
    min_density: float = LEVEL_SET_MIN_DENSITY,
) -> np.ndarray:
    """Derivative of :func:`level_set_heaviside` with respect to ``phi``."""
    values = np.asarray(phi, dtype=float)
    floor = float(np.clip(min_density, 0.0, 1.0))
    tanh_value = np.tanh(float(beta) * values)
    return 0.5 * (1.0 - floor) * float(beta) * (1.0 - tanh_value**2)


def restore_level_set_volume(
    phi: np.ndarray,
    active_mask: np.ndarray,
    target_density: float,
    *,
    passive_phi: Optional[np.ndarray] = None,
    beta: float = LEVEL_SET_BETA,
    min_density: float = LEVEL_SET_MIN_DENSITY,
) -> np.ndarray:
    """Shift the interface so the active phase satisfies its volume budget."""
    values = np.asarray(phi, dtype=float).copy()
    active = np.asarray(active_mask, dtype=bool)
    if values.shape != active.shape:
        raise ValueError("Level-set field and active mask must have identical shapes.")
    if passive_phi is not None:
        passive = np.asarray(passive_phi, dtype=float)
        if passive.shape != values.shape:
            raise ValueError("Passive level-set field must match the design field.")
        values[~active] = passive[~active]
    if not np.any(active):
        return values

    target = float(np.clip(target_density, min_density, 1.0))
    base = values[active]
    lo, hi = -4.0, 4.0
    shift = 0.0
    for _ in range(80):
        shift = 0.5 * (lo + hi)
        density = level_set_heaviside(
            np.clip(base + shift, -1.0, 1.0),
            beta=beta,
            min_density=min_density,
        )
        current = float(np.mean(density))
        if abs(current - target) <= 1e-8:
            break
        if current < target:
            lo = shift
        else:
            hi = shift
    values[active] = np.clip(base + shift, -1.0, 1.0)
    if passive_phi is not None:
        values[~active] = np.asarray(passive_phi, dtype=float)[~active]
    return values


def initialize_level_set(
    shape: tuple[int, int, int],
    active_mask: np.ndarray,
    target_density: float,
    *,
    passive_phi: Optional[np.ndarray] = None,
    beta: float = LEVEL_SET_BETA,
) -> np.ndarray:
    """Create a deterministic, volume-correct signed field with seeded holes."""
    if len(shape) != 3 or min(int(value) for value in shape) < 1:
        raise ValueError("A 3-D positive grid shape is required for level-set setup.")
    active = np.asarray(active_mask, dtype=bool).reshape(shape)
    axes = [
        (np.arange(size, dtype=float) + 0.5) / float(size)
        for size in shape
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    seed = (
        0.10 * np.cos(2.0 * np.pi * xx)
        + 0.08 * np.cos(2.0 * np.pi * yy)
        + 0.06 * np.cos(2.0 * np.pi * zz)
    )
    field = np.asarray(seed, dtype=float)
    return restore_level_set_volume(
        field,
        active,
        target_density,
        passive_phi=passive_phi,
        beta=beta,
    )


def reaction_diffusion_level_set_update(
    phi: np.ndarray,
    objective_gradient: np.ndarray,
    active_mask: np.ndarray,
    target_density: float,
    *,
    passive_phi: Optional[np.ndarray] = None,
    beta: float = LEVEL_SET_BETA,
    time_step: float = LEVEL_SET_TIME_STEP,
    regularization: float = LEVEL_SET_REGULARIZATION,
) -> np.ndarray:
    """Advance the signed interface by a normalized reaction-diffusion step.

    Objective sensitivity supplies the reaction term. The Laplacian is the
    fictitious interface-energy regularizer described by the
    reaction-diffusion level-set method. A scalar interface shift then enforces
    the active material-volume constraint to numerical tolerance.
    """
    values = np.asarray(phi, dtype=float)
    gradient = np.asarray(objective_gradient, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if values.shape != gradient.shape or values.shape != active.shape:
        raise ValueError("Level-set field, gradient, and active mask must match.")
    if not np.any(active):
        return values.copy()

    finite_gradient = np.where(np.isfinite(gradient), gradient, 0.0)
    gradient_scale = float(np.max(np.abs(finite_gradient[active])))
    reaction = np.zeros_like(values)
    if gradient_scale > 1e-30:
        reaction[active] = finite_gradient[active] / gradient_scale

    diffusion = ndimage.laplace(values, mode="nearest")
    diffusion_scale = float(np.max(np.abs(diffusion[active])))
    if diffusion_scale > 1e-30:
        diffusion = diffusion / diffusion_scale
    else:
        diffusion = np.zeros_like(values)

    trial = values.copy()
    trial[active] = (
        values[active]
        - float(time_step) * reaction[active]
        + float(time_step) * float(regularization) * diffusion[active]
    )
    trial = np.clip(trial, -1.0, 1.0)
    return restore_level_set_volume(
        trial,
        active,
        target_density,
        passive_phi=passive_phi,
        beta=beta,
    )


__all__ = [
    "LEVEL_SET_BETA",
    "LEVEL_SET_MIN_DENSITY",
    "level_set_heaviside",
    "level_set_heaviside_derivative",
    "restore_level_set_volume",
    "initialize_level_set",
    "reaction_diffusion_level_set_update",
]
