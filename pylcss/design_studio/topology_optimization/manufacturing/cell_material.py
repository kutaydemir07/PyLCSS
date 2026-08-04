# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Homogenized material laws for the lattice cell families PyLCSS builds.

A lattice study is a two-scale problem. At the micro scale a unit cell has an
effective anisotropic stiffness; at the macro scale the part is optimized as a
graded continuum made of that effective material. This module is the bridge:
it turns :mod:`.homogenization` — which solves one cell — into a continuous,
differentiable law

    C^H(rho) = [C11(rho), C12(rho), C44(rho)]     per unit Young's modulus

that the part-scale sensitivity loop can call on every element of every
iteration, together with the exact derivatives dC/drho the adjoint needs.

Why this replaces ``E = E0 * rho^p``
-----------------------------------
A single isotropic power law cannot represent a lattice, for three reasons
that are measurable rather than stylistic:

* Two families at the same relative density have different stiffness. A
  gyroid sheet and an octet truss at rho = 0.3 differ by a large factor, and
  no choice of exponent fixes both at once.
* A lattice is not isotropic. The octet cell measures a Zener ratio near 1.6,
  meaning its shear stiffness is over 50% higher than any isotropic law with
  the same bulk response can produce. Optimizing against an isotropic
  surrogate therefore mis-ranks shear-dominated load paths.
* The power law has no correct exponent even in one direction. Stretch-
  dominated cells run near rho^1, bending-dominated ones near rho^2, and the
  real curve moves between those regimes as density rises.

Scope
-----
The four cubic families (gyroid, diamond, cubic, octet) are covered. Their
periodic cell is a cube and their homogenized tensor has cubic symmetry, so
three constants describe it exactly and the part-scale operator stays a
three-term sum. Honeycomb is deliberately excluded: it is an extruded prism
whose periodic cell is not a cube and whose tensor is strongly orthotropic, so
it keeps the continuum-density surrogate until the orthotropic assembly path
exists.

The laws are generated offline by ``scripts/build_lattice_material_database.py``
and shipped in :mod:`.cell_material_data`. Anything not in the shipped table —
an unusual Poisson ratio, a different cell resolution — is computed on demand
and cached under ``%LOCALAPPDATA%/PyLCSS``, which costs about a minute once.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .homogenization import homogenize_cell
from .lattice_families import (
    FAMILIES,
    HOMOGENIZED_FAMILIES,
    normalize_family_key,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_SUPERSAMPLE",
    "DATABASE_DENSITY_GRID",
    "DATABASE_POISSON",
    "DATABASE_RESOLUTION",
    "HOMOGENIZED_CELL_FAMILIES",
    "CellMaterialLaw",
    "build_cell_material_law",
    "cell_material_law",
    "solid_cubic_constants",
    "solve_cell_parameter_for_density",
    "unit_cell_occupancy",
]

# Families whose periodic cell is a cube and whose homogenized tensor is cubic.
# Declared once, in the registry, next to the geometry each one generates.
HOMOGENIZED_CELL_FAMILIES = HOMOGENIZED_FAMILIES

# Voxels per axis of the periodic cell used to build a law. 24 is where the
# measured constants stop moving by more than a few percent (checked against
# 32) while a full four-family sweep still finishes in about a minute.
DATABASE_RESOLUTION = 24

# Rasterization refinement used to turn the cell into partial volumes before
# it reaches the finite-element grid. See :func:`unit_cell_occupancy` for why a
# binary cell is not usable here.
CELL_SUPERSAMPLE = 4

# Base Poisson ratio of the shipped tables. C^H is linear in the base Young's
# modulus but not in the base Poisson ratio, so a study far from this value
# gets its own homogenization rather than a rescaled table.
DATABASE_POISSON = 0.30

# How far a study's Poisson ratio may sit from a tabulated one before the law
# is recomputed.
#
# Measured, not assumed. Moving the base ratio from 0.30 to 0.34 across the
# four families at rho = 0.15 and 0.40 moves C44 by under 2% — the struts carry
# axial load, so shear barely notices — but moves C11 by up to 4.4% and C12 by
# up to 13%. C12 is the sensitive constant, because it is the one that carries
# the transverse coupling the base ratio actually sets. A tolerance of 0.04 was
# therefore too loose to justify; 0.025 keeps the worst tabulated error near
# the few-percent level of the voxel rasterization itself.
#
# The band matters practically because it decides who pays for an on-demand
# homogenization. Tables are shipped at both 0.30 and 0.34 so that the common
# structural metals — steels and Inconel near 0.28-0.30, and aluminium,
# titanium and copper near 0.33-0.35 — all land on a tabulated ratio. Ti-6Al-4V
# at 0.342 is the case that motivated this: it is the most common metal AM
# lattice alloy and it fell just outside the original band.
POISSON_TOLERANCE = 0.025

# Base Poisson ratios the shipped tables are generated at.
DATABASE_POISSON_VALUES = (0.30, 0.34)

# Relative densities the law is sampled at. Clustered low, because that is
# where a lattice actually operates and where the curve bends most.
DATABASE_DENSITY_GRID = (
    0.05,
    0.08,
    0.12,
    0.16,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
)

# Below this the sampled tensor is dominated by the void stiffness floor and
# by disconnected rasterization, so the law extrapolates instead of sampling.
MINIMUM_SAMPLED_DENSITY = 0.02

_LAW_CACHE: dict[tuple[str, int, int], "CellMaterialLaw"] = {}


def normalize_cell_family(cell_type: str) -> str:
    """Return the canonical family key, or ``""`` if there is no such family."""
    key = normalize_family_key(cell_type)
    return "" if key == "solid" else key


def solid_cubic_constants(poisson: float) -> tuple[float, float, float]:
    """Return (C11, C12, C44) of the solid base material at unit stiffness.

    This anchors every law at rho = 1 analytically, so a lattice study whose
    density saturates reproduces the solid material exactly instead of
    inheriting whatever the densest resolvable cell happened to measure.
    """
    nu = float(poisson)
    factor = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (
        float(factor * (1.0 - nu)),
        float(factor * nu),
        float(1.0 / (2.0 * (1.0 + nu))),
    )


# ── unit-cell geometry ────────────────────────────────────────────────────────


def unit_cell_occupancy(
    cell_type: str,
    resolution: int,
    parameter: float,
    *,
    supersample: int = CELL_SUPERSAMPLE,
) -> np.ndarray:
    """Rasterize one periodic unit cell of ``cell_type`` as partial volumes.

    ``parameter`` is the family's own thickness control: the requested cell
    relative density for the TPMS sheets, and the member thickness as a
    fraction of the cell pitch for the strut families.

    The generators are the ones :mod:`.structures` uses at part scale, called
    with ``cell_size`` equal to the array size so exactly one period is
    sampled. That is the point of building the law this way — it describes the
    geometry PyLCSS goes on to manufacture, not an idealized textbook cell.

    The cell is rasterized ``supersample`` times finer than the returned array
    and box-averaged down, so each returned voxel carries the fraction of
    itself the cell occupies. A binary cell would not do: these cells are
    highly symmetric, so every strut crosses a voxel-centre threshold at the
    same radius and the achieved density jumps in steps. Measured on a 24-voxel
    octet cell the binary density leaps from 0.056 straight to 0.107, leaving a
    hole in the middle of the operating range that no bisection can enter.
    Partial volumes remove the steps, and :func:`homogenize_cell` already
    consumes a fractional field as a stiffness scale.
    """
    from . import structures

    family = normalize_cell_family(cell_type)
    if family not in HOMOGENIZED_CELL_FAMILIES:
        raise ValueError(
            f"{cell_type!r} has no homogenized cell law. Supported families: "
            + ", ".join(HOMOGENIZED_CELL_FAMILIES)
        )
    n = int(resolution)
    if n < 8:
        raise ValueError("A homogenized unit cell needs at least 8 voxels per axis.")
    factor = max(1, int(supersample))
    fine = n * factor
    shape = (fine, fine, fine)

    entry = FAMILIES[family]
    if entry.is_tpms:
        # Go through the calibrated implicit-field CDF directly. The part-scale
        # wrapper also applies a minimum printable wall, which would flatten
        # the low-density end of the law into a constant.
        band = float(
            structures._tpms_band_for_relative_density(
                np.asarray(float(parameter)), family, samples=fine
            )
        )
        implicit = structures._tpms_implicit(shape, family, float(fine))
        occupancy = (
            implicit <= band
            if entry.wall_mode == "skeletal"
            else np.abs(implicit) <= band
        )
    else:
        occupancy = structures._strut_lattice(
            shape,
            family,
            float(fine),
            float(parameter) * float(fine),
            None,
        )

    if factor == 1:
        return occupancy.astype(float)
    return (
        occupancy.astype(float)
        .reshape(n, factor, n, factor, n, factor)
        .mean(axis=(1, 3, 5))
    )


def _parameter_bounds(cell_type: str) -> tuple[float, float]:
    entry = FAMILIES.get(normalize_cell_family(cell_type))
    if entry is not None and entry.is_tpms:
        # The parameter is a requested density, so it shares its own range.
        return 0.01, 0.99
    # Member thickness as a fraction of the cell pitch. Half the pitch is the
    # point where opposing struts merge and the cell stops being a lattice.
    return 0.01, 0.50


def solve_cell_parameter_for_density(
    cell_type: str,
    resolution: int,
    target_density: float,
    *,
    tolerance: float = 2.5e-3,
    iterations: int = 40,
) -> tuple[float, float]:
    """Bisect the family's thickness control onto a target relative density.

    Returns ``(parameter, achieved_density)``. There is no usable closed form:
    node overlap, fillet-free intersections and voxel rasterization all break
    the analytic ``rho(r/a)`` relation, so the cell is built and measured. When
    the requested density is beyond what the family can reach, the achievable
    end of the range is returned and the caller decides what to do about it.
    """
    low, high = _parameter_bounds(cell_type)
    target = float(target_density)

    def measure(parameter: float) -> float:
        return float(unit_cell_occupancy(cell_type, resolution, parameter).mean())

    density_low, density_high = measure(low), measure(high)
    if target <= density_low:
        return low, density_low
    if target >= density_high:
        return high, density_high

    best_parameter, best_density = high, density_high
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        achieved = measure(middle)
        if abs(achieved - target) < abs(best_density - target):
            best_parameter, best_density = middle, achieved
        if abs(achieved - target) <= tolerance:
            break
        if achieved > target:
            high = middle
        else:
            low = middle
    return float(best_parameter), float(best_density)


# ── the law ───────────────────────────────────────────────────────────────────


def _monotone_log_curve(
    density: np.ndarray,
    values: np.ndarray,
) -> tuple[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], bool]:
    """Return an evaluator giving ``(value, d value / d rho)``.

    Interpolation happens on log(C) against log(rho). Every one of these
    constants is close to a power law over any decade of density, so the
    samples are nearly collinear there — which is what makes a shape-preserving
    cubic both accurate between samples and safe outside them. Extrapolation is
    linear in log-log using the end slope, i.e. it continues as the power law
    the last two samples imply, instead of letting a cubic turn over.

    Falls back to linear-in-rho interpolation if any sample is non-positive,
    which an auxetic or badly resolved cell can produce for C12.
    """
    from scipy.interpolate import PchipInterpolator

    density = np.asarray(density, dtype=float)
    values = np.asarray(values, dtype=float)
    positive = bool(np.all(values > 0.0))

    if positive:
        nodes = np.log(density)
        samples = np.log(values)
    else:
        nodes = density
        samples = values

    spline = PchipInterpolator(nodes, samples, extrapolate=False)
    slope_spline = spline.derivative()
    first_node, last_node = float(nodes[0]), float(nodes[-1])
    first_value, last_value = float(samples[0]), float(samples[-1])
    first_slope = float(slope_spline(first_node))
    last_slope = float(slope_spline(last_node))

    def evaluate(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho = np.asarray(rho, dtype=float)
        node = np.log(rho) if positive else rho
        interior = (node >= first_node) & (node <= last_node)
        result = np.empty_like(node)
        slope = np.empty_like(node)
        if np.any(interior):
            result[interior] = spline(node[interior])
            slope[interior] = slope_spline(node[interior])
        below = node < first_node
        if np.any(below):
            result[below] = first_value + first_slope * (node[below] - first_node)
            slope[below] = first_slope
        above = node > last_node
        if np.any(above):
            result[above] = last_value + last_slope * (node[above] - last_node)
            slope[above] = last_slope
        if positive:
            value = np.exp(result)
            # d C / d rho = C * d(log C)/d(log rho) / rho
            return value, value * slope / rho
        return result, slope

    return evaluate, positive


@dataclass(frozen=True)
class CellMaterialLaw:
    """A differentiable homogenized law ``rho -> (C11, C12, C44)``.

    All constants are per unit base Young's modulus; a two-phase solid/void
    composite is exactly linear in the base modulus, so the part-scale caller
    multiplies by ``E0``. They are *not* linear in the base Poisson ratio,
    which is why ``base_poisson`` is recorded and checked.
    """

    cell_type: str
    resolution: int
    base_poisson: float
    relative_density: np.ndarray
    c11: np.ndarray
    c12: np.ndarray
    c44: np.ndarray
    # The family's own thickness control that produced each sample, so the
    # de-homogenization step can invert density back to geometry.
    cell_parameter: np.ndarray
    # Highest density reached by a real cell before the analytic solid anchor.
    maximum_lattice_density: float
    # Worst deviation from cubic symmetry over the samples, as a fraction.
    cubic_symmetry_residual: float

    def __post_init__(self) -> None:
        density = np.asarray(self.relative_density, dtype=float)
        if density.ndim != 1 or density.size < 3:
            raise ValueError("A cell material law needs at least three samples.")
        if np.any(np.diff(density) <= 0.0):
            raise ValueError("Cell material law densities must strictly increase.")
        for name in ("c11", "c12", "c44", "cell_parameter"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != density.shape:
                raise ValueError(f"{name} must have one value per sampled density.")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "relative_density", density)
        # A cubic elastic tensor is positive definite only when C44 > 0,
        # C11 - C12 > 0, and C11 + 2*C12 > 0.  Every homogenized sample obeys
        # those conditions, but the three log curves are extrapolated
        # independently below the first measured density.  On a strongly
        # bending-dominated cell (notably BCC) their extrapolated C11 and C12
        # can cross by a few parts in 1e5 near rho=1e-3.  The resulting tiny
        # negative shear mode is enough to give the part-scale stiffness
        # matrix a zero pivot after Heaviside projection.
        #
        # Preserve the weakest *measured* normal-mode margin during that
        # extrapolation.  This is preferable to a fixed epsilon: it is
        # dimensionless, family-specific, continuous at the crossing, and
        # scales toward zero with the material law.
        normal_scale = np.maximum.reduce(
            [np.abs(self.c11), np.abs(self.c12), np.full_like(self.c11, 1e-30)]
        )
        normal_margin = (self.c11 - self.c12) / normal_scale
        if (
            np.any(self.c44 <= 0.0)
            or np.any(self.c11 + 2.0 * self.c12 <= 0.0)
            or np.any(normal_margin <= 0.0)
        ):
            raise ValueError(
                "A sampled homogenized cubic tensor is not positive definite."
            )
        object.__setattr__(
            self,
            "_minimum_normal_stability_ratio",
            float(np.clip(np.min(normal_margin), 1e-4, 0.95)),
        )
        object.__setattr__(
            self,
            "_curves",
            {
                "c11": _monotone_log_curve(density, self.c11)[0],
                "c12": _monotone_log_curve(density, self.c12)[0],
                "c44": _monotone_log_curve(density, self.c44)[0],
            },
        )

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        density: np.ndarray,
        *,
        young: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(C11, C12, C44)`` for a density field, scaled by ``young``."""
        rho = np.clip(np.asarray(density, dtype=float), 1e-6, 1.0)
        curves = self._curves  # type: ignore[attr-defined]
        values = [
            float(young) * curves[name](rho)[0] for name in ("c11", "c12", "c44")
        ]
        ratio = float(
            getattr(self, "_minimum_normal_stability_ratio", 1.0e-4)
        )
        # C11 >= C12/(1-ratio) is equivalent to
        # (C11-C12)/C11 >= ratio for the positive sampled curves.
        stable_c11 = values[1] / max(1.0 - ratio, 1.0e-6)
        values[0] = np.maximum(values[0], stable_c11)
        return tuple(values)  # type: ignore[return-value]

    def evaluate_with_gradient(
        self,
        density: np.ndarray,
        *,
        young: float = 1.0,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        """Return ``((C11, C12, C44), (dC11, dC12, dC44))`` against density.

        The derivatives are the whole reason the law is interpolated rather
        than looked up: without them the two-scale problem has no gradient and
        cannot be driven by MMA.
        """
        raw = np.asarray(density, dtype=float)
        rho = np.clip(raw, 1e-6, 1.0)
        # Where the clip is active the constants are frozen, so their derivative
        # is zero; otherwise the optimizer sees stiffness it cannot change. The
        # upper bound stays live at exactly 1.0 so a saturated element can still
        # be driven back down.
        live = (raw > 1e-6) & (raw <= 1.0)
        curves = self._curves  # type: ignore[attr-defined]
        values: list[np.ndarray] = []
        gradients: list[np.ndarray] = []
        for name in ("c11", "c12", "c44"):
            value, gradient = curves[name](rho)
            values.append(float(young) * value)
            gradients.append(float(young) * gradient * live)
        ratio = float(
            getattr(self, "_minimum_normal_stability_ratio", 1.0e-4)
        )
        scale = max(1.0 - ratio, 1.0e-6)
        stable_c11 = values[1] / scale
        clamped = values[0] < stable_c11
        values[0] = np.where(clamped, stable_c11, values[0])
        gradients[0] = np.where(
            clamped,
            gradients[1] / scale,
            gradients[0],
        )
        return tuple(values), tuple(gradients)

    def young_isotropic(
        self,
        density: np.ndarray,
        *,
        young: float = 1.0,
    ) -> np.ndarray:
        """Voigt-Reuss-Hill isotropic-equivalent modulus of the cell.

        Used for reporting and for the stress surrogate, which still wants one
        scalar modulus. It is an average of a genuinely anisotropic tensor, so
        it is a summary of the law, never a replacement for it.
        """
        c11, c12, c44 = self.evaluate(density, young=young)
        bulk = (c11 + 2.0 * c12) / 3.0
        shear_voigt = (c11 - c12 + 3.0 * c44) / 5.0
        denominator = 4.0 * c44 + 3.0 * (c11 - c12)
        shear_reuss = np.where(
            np.abs(denominator) > 1e-30,
            5.0 * c44 * (c11 - c12) / np.where(np.abs(denominator) > 1e-30, denominator, 1.0),
            0.0,
        )
        shear = 0.5 * (shear_voigt + np.maximum(shear_reuss, 0.0))
        return np.where(
            (bulk > 0.0) & (shear > 0.0),
            9.0 * bulk * shear / np.maximum(3.0 * bulk + shear, 1e-30),
            0.0,
        )

    def zener_ratio(self, density: np.ndarray) -> np.ndarray:
        """Anisotropy of the cell at a density; 1.0 would be isotropic."""
        c11, c12, c44 = self.evaluate(density)
        difference = c11 - c12
        return np.where(
            np.abs(difference) > 1e-30,
            2.0 * c44 / np.where(np.abs(difference) > 1e-30, difference, 1.0),
            np.inf,
        )

    def cell_parameter_for_density(self, density: np.ndarray) -> np.ndarray:
        """Invert density back to the family's thickness control.

        This is the de-homogenization map. It is measured, not analytic, which
        is what makes a requested relative density actually come out of the
        generated geometry.
        """
        rho = np.clip(np.asarray(density, dtype=float), 1e-6, 1.0)
        return np.interp(
            rho,
            self.relative_density,
            self.cell_parameter,
            left=float(self.cell_parameter[0]),
            right=float(self.cell_parameter[-1]),
        )

    # ── reporting ────────────────────────────────────────────────────────────

    def equivalent_power_law(
        self,
        density_range: tuple[float, float] = (0.10, 0.60),
    ) -> tuple[float, float]:
        """Least-squares ``E/E0 = a * rho^b`` fit over a working density band.

        Reported so a study can state how far the isotropic power law it
        replaces would have been, and to give a sanity check against the
        Gibson-Ashby exponents (near 1 for stretch-dominated cells, near 2 for
        bending-dominated ones).
        """
        low, high = (float(value) for value in density_range)
        rho = np.geomspace(max(low, 1e-4), min(high, 1.0), 24)
        modulus = self.young_isotropic(rho)
        usable = modulus > 0.0
        if int(np.count_nonzero(usable)) < 2:
            return 0.0, 0.0
        slope, intercept = np.polyfit(
            np.log(rho[usable]), np.log(modulus[usable]), 1
        )
        return float(np.exp(intercept)), float(slope)

    def diagnostics(self) -> dict[str, object]:
        """Return a JSON-friendly summary for the study report."""
        coefficient, exponent = self.equivalent_power_law()
        return {
            "cell_type": self.cell_type,
            "method": (
                "periodic unit-cell numerical homogenization, six unit "
                "macroscopic strains, cubic reduction"
            ),
            "unit_cell_resolution": int(self.resolution),
            "base_poisson_ratio": float(self.base_poisson),
            "sample_count": int(self.relative_density.size),
            "density_range": [
                float(self.relative_density[0]),
                float(self.relative_density[-1]),
            ],
            "maximum_lattice_density": float(self.maximum_lattice_density),
            "cubic_symmetry_residual": float(self.cubic_symmetry_residual),
            "zener_ratio_at_0p3": float(self.zener_ratio(np.asarray(0.3))),
            "equivalent_power_law_coefficient": coefficient,
            "equivalent_power_law_exponent": exponent,
        }

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_type": self.cell_type,
            "resolution": int(self.resolution),
            "base_poisson": float(self.base_poisson),
            "relative_density": [float(v) for v in self.relative_density],
            "c11": [float(v) for v in self.c11],
            "c12": [float(v) for v in self.c12],
            "c44": [float(v) for v in self.c44],
            "cell_parameter": [float(v) for v in self.cell_parameter],
            "maximum_lattice_density": float(self.maximum_lattice_density),
            "cubic_symmetry_residual": float(self.cubic_symmetry_residual),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CellMaterialLaw":
        return cls(
            cell_type=str(payload["cell_type"]),
            resolution=int(payload["resolution"]),  # type: ignore[arg-type]
            base_poisson=float(payload["base_poisson"]),  # type: ignore[arg-type]
            relative_density=np.asarray(payload["relative_density"], dtype=float),
            c11=np.asarray(payload["c11"], dtype=float),
            c12=np.asarray(payload["c12"], dtype=float),
            c44=np.asarray(payload["c44"], dtype=float),
            cell_parameter=np.asarray(payload["cell_parameter"], dtype=float),
            maximum_lattice_density=float(payload["maximum_lattice_density"]),  # type: ignore[arg-type]
            cubic_symmetry_residual=float(
                payload["cubic_symmetry_residual"]  # type: ignore[arg-type]
            ),
        )


# ── construction ──────────────────────────────────────────────────────────────


def build_cell_material_law(
    cell_type: str,
    *,
    poisson: float = DATABASE_POISSON,
    resolution: int = DATABASE_RESOLUTION,
    density_grid: tuple[float, ...] = DATABASE_DENSITY_GRID,
    progress: Callable[[str], None] | None = None,
) -> CellMaterialLaw:
    """Homogenize one family across a density sweep and return its law.

    This is the expensive path — one periodic six-load-case solve per sample.
    It runs offline through ``scripts/build_lattice_material_database.py`` and
    on demand only when the shipped table does not cover a study.
    """
    family = normalize_cell_family(cell_type)
    if family not in HOMOGENIZED_CELL_FAMILIES:
        raise ValueError(
            f"{cell_type!r} has no homogenized cell law. Supported families: "
            + ", ".join(HOMOGENIZED_CELL_FAMILIES)
        )

    densities: list[float] = []
    parameters: list[float] = []
    constants: list[tuple[float, float, float]] = []
    residual = 0.0
    for requested in density_grid:
        parameter, achieved = solve_cell_parameter_for_density(
            family, resolution, requested
        )
        if achieved < MINIMUM_SAMPLED_DENSITY:
            continue
        # A family that has run out of thickness reports the same density for
        # every further request; one sample at the ceiling is enough.
        if densities and achieved <= densities[-1] + 1e-4:
            continue
        occupancy = unit_cell_occupancy(family, resolution, parameter)
        elasticity = homogenize_cell(
            occupancy.astype(float), young=1.0, poisson=float(poisson)
        )
        densities.append(float(elasticity.relative_density))
        parameters.append(float(parameter))
        constants.append(
            (float(elasticity.c11), float(elasticity.c12), float(elasticity.c44))
        )
        residual = max(residual, float(elasticity.cubic_symmetry_residual))
        if progress is not None:
            progress(
                f"{family}: rho={elasticity.relative_density:.4f} "
                f"C11={elasticity.c11:.5f} C12={elasticity.c12:.5f} "
                f"C44={elasticity.c44:.5f} "
                f"zener={elasticity.zener_ratio:.3f}"
            )

    if len(densities) < 3:
        raise RuntimeError(
            f"Homogenizing the {family} family produced only {len(densities)} "
            "usable samples; the cell is not resolved at this resolution."
        )

    maximum_lattice_density = float(densities[-1])
    # Anchor the solid end analytically. Without it the law would extrapolate a
    # power law past the densest resolvable cell and a saturated study would
    # not recover the base material.
    solid = solid_cubic_constants(poisson)
    if maximum_lattice_density < 1.0 - 1e-6:
        densities.append(1.0)
        parameters.append(float(_parameter_bounds(family)[1]))
        constants.append(solid)

    array = np.asarray(constants, dtype=float)
    return CellMaterialLaw(
        cell_type=family,
        resolution=int(resolution),
        base_poisson=float(poisson),
        relative_density=np.asarray(densities, dtype=float),
        c11=array[:, 0],
        c12=array[:, 1],
        c44=array[:, 2],
        cell_parameter=np.asarray(parameters, dtype=float),
        maximum_lattice_density=maximum_lattice_density,
        cubic_symmetry_residual=residual,
    )


# ── lookup ────────────────────────────────────────────────────────────────────


def _cache_directory() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    base = Path(root) if root else Path.home() / "AppData" / "Local"
    return base / "PyLCSS" / "lattice_material"


def _cache_path(cell_type: str, poisson: float, resolution: int) -> Path:
    key = f"{cell_type}_nu{round(float(poisson) * 100):03d}_n{int(resolution)}.json"
    return _cache_directory() / key


def cell_material_law(
    cell_type: str,
    *,
    poisson: float = DATABASE_POISSON,
    resolution: int = DATABASE_RESOLUTION,
    allow_build: bool = True,
) -> CellMaterialLaw | None:
    """Return the homogenized law for a family, or ``None`` if unavailable.

    Resolution order: process memory, the shipped table, the on-disk cache,
    then a fresh homogenization. ``None`` means the family has no law at all
    (honeycomb, solid) — callers fall back to the isotropic surrogate and
    must say so in their report.
    """
    family = normalize_cell_family(cell_type)
    if family not in HOMOGENIZED_CELL_FAMILIES:
        return None

    memory_key = (family, round(float(poisson) * 1000), int(resolution))
    cached = _LAW_CACHE.get(memory_key)
    if cached is not None:
        return cached

    shipped = _shipped_law(family, poisson, resolution)
    if shipped is not None:
        _LAW_CACHE[memory_key] = shipped
        return shipped

    path = _cache_path(family, poisson, resolution)
    try:
        if path.is_file():
            law = CellMaterialLaw.from_dict(json.loads(path.read_text("utf-8")))
            _LAW_CACHE[memory_key] = law
            return law
    except Exception:
        logger.warning("Ignoring unreadable lattice material cache %s.", path)

    if not allow_build:
        return None

    logger.warning(
        "No tabulated %s cell law for Poisson %.3f at resolution %d. "
        "Homogenizing it now; this takes about a minute and is then cached.",
        family,
        float(poisson),
        int(resolution),
    )
    law = build_cell_material_law(
        family, poisson=poisson, resolution=resolution
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(law.to_dict(), indent=1), encoding="utf-8")
    except OSError:
        logger.warning("Could not cache the %s cell law to %s.", family, path)
    _LAW_CACHE[memory_key] = law
    return law


def _shipped_law(
    family: str,
    poisson: float,
    resolution: int,
) -> CellMaterialLaw | None:
    """Return the pre-generated law nearest this study's base Poisson ratio.

    Entries are selected on the ``base_poisson`` recorded inside each payload
    rather than on the dictionary key, so a table generated at a new ratio is
    picked up without any change here and older single-ratio tables keep
    working.
    """
    if int(resolution) != DATABASE_RESOLUTION:
        return None
    if not math.isfinite(float(poisson)):
        return None
    try:
        from .cell_material_data import SHIPPED_CELL_LAWS
    except Exception:
        logger.warning("The shipped lattice material table could not be imported.")
        return None

    candidates = [
        payload
        for payload in SHIPPED_CELL_LAWS.values()
        if str(payload.get("cell_type", "")) == family
    ]
    if not candidates:
        return None
    nearest = min(
        candidates,
        key=lambda payload: abs(float(payload.get("base_poisson", 0.0)) - float(poisson)),
    )
    offset = abs(float(nearest.get("base_poisson", 0.0)) - float(poisson))
    if offset > POISSON_TOLERANCE:
        return None
    try:
        return CellMaterialLaw.from_dict(dict(nearest))
    except Exception:
        logger.warning("The shipped %s cell law is malformed; ignoring it.", family)
        return None
