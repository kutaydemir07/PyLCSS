# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Optimization-aware printable structures for topology-optimized designs.

Gyroid and Diamond modes use the optimized continuum density as a local
relative-density design field.  The explicit TPMS wall band is sized locally
from that field instead of inserting one constant-thickness lattice after the
solve.  Independent re-analysis remains mandatory because homogenization is a
macro-scale approximation of the manufactured cell geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_MODE_ALIASES = {
    "": "solid",
    "none": "solid",
    "solid": "solid",
    "solid envelope": "solid",
    "ribs": "topology_ribs",
    "rib": "topology_ribs",
    "rib network": "topology_ribs",
    "topology-following ribs": "topology_ribs",
    "topology ribs": "topology_ribs",
    "gyroid": "gyroid",
    "gyroid lattice": "gyroid",
    "diamond": "diamond",
    "diamond lattice": "diamond",
}


@dataclass(frozen=True)
class ManufacturingStructureOptions:
    """Controls for the explicit optimized manufacturing geometry."""

    mode: str = "solid"
    cell_size_voxels: float = 6.0
    member_thickness_voxels: float = 1.0
    skin_thickness_voxels: float = 0.75
    variable_density: bool = True
    minimum_relative_density: float = 0.12
    maximum_relative_density: float = 0.90
    solid_transition_density: float = 0.92

    def __post_init__(self) -> None:
        normalized = str(self.mode or "solid").strip().lower().replace("_", " ")
        mode = _MODE_ALIASES.get(normalized)
        if mode is None:
            raise ValueError(
                "Structure mode must be Solid Envelope, Topology-Following Ribs, "
                "Gyroid Lattice, or Diamond Lattice."
            )
        object.__setattr__(self, "mode", mode)
        for name in (
            "cell_size_voxels",
            "member_thickness_voxels",
            "skin_thickness_voxels",
            "minimum_relative_density",
            "maximum_relative_density",
            "solid_transition_density",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "variable_density", bool(self.variable_density))
        if self.mode in {"gyroid", "diamond"} and self.cell_size_voxels < 3.0:
            raise ValueError("A lattice cell must span at least 3 voxels.")
        if self.mode != "solid" and self.member_thickness_voxels <= 0.0:
            raise ValueError("A rib or lattice member thickness must be positive.")
        if not (
            0.0 < self.minimum_relative_density
            < self.maximum_relative_density
            <= self.solid_transition_density
            <= 1.0
        ):
            raise ValueError(
                "Lattice densities require 0 < minimum < maximum <= solid "
                "transition <= 1."
            )

    @property
    def display_name(self) -> str:
        return {
            "solid": "Solid Envelope",
            "topology_ribs": "Topology-Following Ribs",
            "gyroid": "Gyroid Lattice",
            "diamond": "Diamond Lattice",
        }[self.mode]


def _boundary_skin(envelope: np.ndarray, thickness: float) -> np.ndarray:
    if thickness <= 0.0 or not np.any(envelope):
        return np.zeros_like(envelope, dtype=bool)
    from scipy import ndimage as ndi

    inside_distance = ndi.distance_transform_edt(envelope)
    return envelope & (inside_distance <= max(1.0, float(thickness)))


def _topology_ribs(
    envelope: np.ndarray,
    member_radius: float,
    cell_size: float,
) -> np.ndarray:
    """Return a topology-following medial rib network.

    Lee's 3-D skeleton follows the optimized load-path envelope.  The fallback
    uses three orthogonal rib families so geometry generation remains available
    when scikit-image's skeletonizer is not installed.
    """
    from scipy import ndimage as ndi

    skeleton = None
    try:
        from skimage.morphology import skeletonize

        skeleton = np.asarray(skeletonize(envelope, method="lee"), dtype=bool)
    except Exception:
        pass
    if skeleton is None or not np.any(skeleton):
        coordinates = np.indices(envelope.shape, dtype=float)
        period = max(3.0, float(cell_size))
        distance_to_plane = [
            np.minimum(np.mod(axis + 0.5, period), period - np.mod(axis + 0.5, period))
            for axis in coordinates
        ]
        skeleton = np.logical_or.reduce(
            [distance <= 0.5 for distance in distance_to_plane]
        )
        skeleton &= envelope

    distance_to_skeleton = ndi.distance_transform_edt(~skeleton)
    return envelope & (distance_to_skeleton <= max(0.5, float(member_radius)))


def _tpms_implicit(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
) -> np.ndarray:
    coordinates = np.indices(shape, dtype=float)
    phase = [
        2.0 * np.pi * (axis + 0.5) / max(float(cell_size), 1e-9)
        for axis in coordinates
    ]
    x, y, z = phase
    if mode == "gyroid":
        return (
            np.sin(x) * np.cos(y)
            + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x)
        )
    return (
        np.sin(x) * np.sin(y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(z)
        + np.cos(x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(y) * np.sin(z)
    )


_TPMS_BAND_LUT: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _tpms_band_for_relative_density(
    relative_density: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Map requested cell-relative density to an implicit sheet half-band.

    The lookup table is the numerical cumulative distribution of ``|f|`` over
    one periodic cell.  Consequently a requested relative density of ``r``
    occupies approximately ``r`` of a resolved unit cell before skin and
    attachment solids are added.
    """
    if mode not in _TPMS_BAND_LUT:
        samples = 64
        phase = (
            2.0
            * np.pi
            * (np.arange(samples, dtype=float) + 0.5)
            / float(samples)
        )
        x, y, z = np.meshgrid(phase, phase, phase, indexing="ij")
        if mode == "gyroid":
            implicit = (
                np.sin(x) * np.cos(y)
                + np.sin(y) * np.cos(z)
                + np.sin(z) * np.cos(x)
            )
        else:
            implicit = (
                np.sin(x) * np.sin(y) * np.sin(z)
                + np.sin(x) * np.cos(y) * np.cos(z)
                + np.cos(x) * np.sin(y) * np.cos(z)
                + np.cos(x) * np.cos(y) * np.sin(z)
            )
        fractions = np.linspace(0.0, 1.0, 513)
        bands = np.quantile(np.abs(implicit).ravel(), fractions)
        _TPMS_BAND_LUT[mode] = (fractions, bands)
    fractions, bands = _TPMS_BAND_LUT[mode]
    return np.interp(
        np.clip(np.asarray(relative_density, dtype=float), 0.0, 1.0),
        fractions,
        bands,
    )


def _periodic_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None = None,
) -> np.ndarray:
    implicit = _tpms_implicit(shape, mode, cell_size)

    minimum_band = float(
        np.pi * float(member_thickness) / max(float(cell_size), 1e-9)
    )
    if relative_density is None:
        band: np.ndarray | float = max(0.12, minimum_band)
    else:
        band = np.maximum(
            _tpms_band_for_relative_density(relative_density, mode),
            minimum_band,
        )
    band = np.clip(band, 0.01, float(np.max(np.abs(implicit))) + 1e-9)
    return np.abs(implicit) <= band


def build_manufacturing_field(
    density: np.ndarray,
    cutoff: float,
    options: ManufacturingStructureOptions | None = None,
    *,
    resolution_scale: float = 1.0,
    passive_solid_mask: np.ndarray | None = None,
    passive_void_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a solid, ribbed, or TPMS lattice field within the density envelope.

    ``resolution_scale`` converts values expressed in source voxel units to a
    supersampled recovery grid.  The returned field is binary and has the same
    shape as ``density``.
    """
    opts = options or ManufacturingStructureOptions()
    field = np.nan_to_num(
        np.asarray(density, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if field.ndim != 3:
        raise ValueError("Manufacturing structure input must be a 3-D density field.")

    envelope = field >= float(np.clip(cutoff, 1e-6, 0.999999))
    if opts.mode == "solid" or not np.any(envelope):
        manufactured = envelope
    else:
        scale = max(float(resolution_scale), 1e-9)
        cell_size = max(3.0, opts.cell_size_voxels * scale)
        member = max(0.5, opts.member_thickness_voxels * scale)
        skin = _boundary_skin(envelope, opts.skin_thickness_voxels * scale)
        solid_zone = np.zeros_like(envelope, dtype=bool)

        if opts.mode == "topology_ribs":
            core = _topology_ribs(envelope, 0.5 * member, cell_size)
        else:
            if opts.variable_density:
                # The macro optimizer's physical density is the desired local
                # lattice relative density inside the selected topology
                # envelope.  Never expand that envelope down to the minimum
                # lattice density: doing so turns low-density solver haze into
                # printable material that is absent from the Density view.
                # High-density zones remain solid and the intermediate zone
                # receives a locally sized TPMS sheet.
                solid_zone = field >= opts.solid_transition_density
                target_relative_density = np.clip(
                    field,
                    opts.minimum_relative_density,
                    opts.maximum_relative_density,
                )
            else:
                target_relative_density = None
            core = envelope & _periodic_lattice(
                tuple(int(v) for v in field.shape),
                opts.mode,
                cell_size,
                member,
                relative_density=target_relative_density,
            )

        manufactured = envelope & (skin | core | solid_zone)

    if passive_solid_mask is not None:
        keep = np.asarray(passive_solid_mask, dtype=bool)
        if keep.shape != field.shape:
            raise ValueError("Passive-solid mask shape must match the density field.")
        manufactured |= keep
    if passive_void_mask is not None:
        cut = np.asarray(passive_void_mask, dtype=bool)
        if cut.shape != field.shape:
            raise ValueError("Passive-void mask shape must match the density field.")
        manufactured &= ~cut
    return manufactured.astype(float)


def passive_region_masks(
    shape: tuple[int, int, int],
    *,
    solid_boxes: Any = (),
    void_boxes: Any = (),
    solid_cylinders: Any = (),
    void_cylinders: Any = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize fractional passive regions for manufacturing verification."""
    dims = tuple(max(1, int(v)) for v in shape)
    coordinates = [
        (np.arange(n, dtype=float) + 0.5) / float(n)
        for n in dims
    ]
    x, y, z = np.meshgrid(*coordinates, indexing="ij")
    grids = (x, y, z)

    def boxes_mask(boxes: Any) -> np.ndarray:
        result = np.zeros(dims, dtype=bool)
        for box in boxes or ():
            if len(box) < 6:
                continue
            values = [float(v) for v in box[:6]]
            result |= (
                (x >= min(values[0], values[1]))
                & (x <= max(values[0], values[1]))
                & (y >= min(values[2], values[3]))
                & (y <= max(values[2], values[3]))
                & (z >= min(values[4], values[5]))
                & (z <= max(values[4], values[5]))
            )
        return result

    def cylinders_mask(cylinders: Any) -> np.ndarray:
        result = np.zeros(dims, dtype=bool)
        axis_index = {"x": 0, "y": 1, "z": 2}
        radial_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
        for cylinder in cylinders or ():
            if len(cylinder) < 6:
                continue
            axis = str(cylinder[0] or "z").lower()
            if axis not in axis_index:
                continue
            c0, c1 = float(cylinder[1]), float(cylinder[2])
            lo, hi = sorted((float(cylinder[3]), float(cylinder[4])))
            r0 = float(cylinder[5])
            r1 = float(cylinder[6]) if len(cylinder) > 6 else r0
            a0, a1 = radial_axes[axis]
            radial = (
                ((grids[a0] - c0) / max(r0, 1e-12)) ** 2
                + ((grids[a1] - c1) / max(r1, 1e-12)) ** 2
                <= 1.0
            )
            axial = (grids[axis_index[axis]] >= lo) & (
                grids[axis_index[axis]] <= hi
            )
            result |= radial & axial
        return result

    solid = boxes_mask(solid_boxes) | cylinders_mask(solid_cylinders)
    void = boxes_mask(void_boxes) | cylinders_mask(void_cylinders)
    solid &= ~void
    return solid, void


def structure_options_from_values(
    mode: Any,
    cell_size_voxels: Any,
    member_thickness_voxels: Any,
    skin_thickness_voxels: Any,
    variable_density: Any = True,
    minimum_relative_density: Any = 0.12,
    maximum_relative_density: Any = 0.90,
    solid_transition_density: Any = 0.92,
) -> ManufacturingStructureOptions:
    """Parse graph-property values with one consistent validation path."""
    return ManufacturingStructureOptions(
        mode=str(mode or "Solid Envelope"),
        cell_size_voxels=float(cell_size_voxels or 6.0),
        member_thickness_voxels=float(member_thickness_voxels or 1.0),
        skin_thickness_voxels=float(skin_thickness_voxels or 0.0),
        variable_density=bool(variable_density),
        minimum_relative_density=float(minimum_relative_density or 0.12),
        maximum_relative_density=float(maximum_relative_density or 0.90),
        solid_transition_density=float(solid_transition_density or 0.92),
    )
