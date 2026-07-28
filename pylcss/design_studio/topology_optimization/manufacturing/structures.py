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
    "honeycomb": "honeycomb",
    "honeycomb lattice": "honeycomb",
    "cubic": "cubic",
    "cubic lattice": "cubic",
    "octet": "octet",
    "octet truss": "octet",
    "octet truss lattice": "octet",
}


@dataclass(frozen=True)
class ManufacturingStructureOptions:
    """Controls for the explicit optimized manufacturing geometry."""

    mode: str = "solid"
    cell_size_voxels: float = 8.0
    member_thickness_voxels: float = 1.0
    skin_thickness_voxels: float = 0.75
    variable_density: bool = True
    minimum_relative_density: float = 0.15
    maximum_relative_density: float = 0.60
    solid_transition_density: float = 0.92

    def __post_init__(self) -> None:
        normalized = str(self.mode or "solid").strip().lower().replace("_", " ")
        mode = _MODE_ALIASES.get(normalized)
        if mode is None:
            raise ValueError(
                "Structure mode must be Solid Envelope, Topology-Following Ribs, "
                "Gyroid, Diamond, Honeycomb, Cubic, or Octet Truss."
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
        if self.mode in {
            "gyroid", "diamond", "honeycomb", "cubic", "octet"
        } and self.cell_size_voxels < 3.0:
            raise ValueError("A lattice cell must span at least 3 voxels.")
        if self.mode != "solid" and self.member_thickness_voxels <= 0.0:
            raise ValueError("A rib or lattice member thickness must be positive.")
        if (
            self.mode in {"gyroid", "diamond", "honeycomb", "cubic", "octet"}
            and self.cell_size_voxels
            < 4.0 * self.member_thickness_voxels
        ):
            raise ValueError(
                "Lattice cell size must be at least four times the member "
                "thickness so the cell openings remain resolved."
            )
        if (
            self.mode in {"gyroid", "diamond", "honeycomb", "cubic", "octet"}
            and self.skin_thickness_voxels > 0.25 * self.cell_size_voxels
        ):
            raise ValueError(
                "Lattice skin thickness must not exceed one quarter of the "
                "cell size."
            )
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
            "honeycomb": "Honeycomb Lattice",
            "cubic": "Cubic Lattice",
            "octet": "Octet Truss Lattice",
        }[self.mode]


def _boundary_skin(envelope: np.ndarray, thickness: float) -> np.ndarray:
    if thickness <= 0.0 or not np.any(envelope):
        return np.zeros_like(envelope, dtype=bool)
    from scipy import ndimage as ndi

    # Pad with void so a domain that fills the whole array still gets a skin
    # on all six exterior faces. scipy's EDT otherwise has no explicit
    # out-of-array background and biases the skin toward one corner.
    padded = np.pad(
        np.asarray(envelope, dtype=bool),
        pad_width=1,
        mode="constant",
        constant_values=False,
    )
    inside_distance = ndi.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
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


_TPMS_BAND_LUT: dict[tuple[str, int], np.ndarray] = {}


def _tpms_band_for_relative_density(
    relative_density: np.ndarray,
    mode: str,
    samples: int = 64,
) -> np.ndarray:
    """Map requested cell-relative density to an implicit sheet half-band.

    The lookup table is the numerical cumulative distribution of ``|f|`` over
    one periodic cell.  Consequently a requested relative density of ``r``
    occupies approximately ``r`` of a resolved unit cell before skin and
    attachment solids are added.
    """
    # Calibrate at the actual resolved points per cell. A continuum 64³ lookup
    # badly overfilled a Diamond sheet sampled on only 6–8 voxels per cell.
    samples = int(np.clip(round(samples), 3, 64))
    key = (mode, samples)
    if key not in _TPMS_BAND_LUT:
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
        _TPMS_BAND_LUT[key] = np.sort(np.abs(implicit).ravel())
    bands = _TPMS_BAND_LUT[key]
    requested = np.clip(
        np.asarray(relative_density, dtype=float), 0.0, 1.0
    )
    # Select the first discrete threshold that contains at least the requested
    # cell fraction. Interpolating repeated quantiles is unreliable on coarse
    # 6--8 voxel cells and previously collapsed nominal 20--35% sheets to a
    # few isolated voxels.
    indices = np.ceil(requested * bands.size).astype(int) - 1
    indices = np.clip(indices, 0, bands.size - 1)
    return bands[indices] + 32.0 * np.finfo(float).eps


def _periodic_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None = None,
) -> np.ndarray:
    implicit = _tpms_implicit(shape, mode, cell_size)

    # Convert the minimum printable sheet thickness to a cell-relative volume
    # fraction, then use the same calibrated implicit-field CDF as the target
    # density. A raw π*t/cell band strongly overfilled Diamond cells (often
    # >80% solid for a one-voxel wall in an 8-voxel cell).
    minimum_fraction = float(np.clip(
        2.0 * float(member_thickness) / max(float(cell_size), 1e-9),
        0.02,
        0.80,
    ))
    minimum_band = float(_tpms_band_for_relative_density(
        np.asarray(minimum_fraction), mode, samples=int(round(cell_size))
    ))
    if relative_density is None:
        band: np.ndarray | float = max(0.12, minimum_band)
    else:
        band = np.maximum(
            _tpms_band_for_relative_density(
                relative_density,
                mode,
                samples=int(round(cell_size)),
            ),
            # Calibrate against the actual voxel resolution of each cell.
            minimum_band,
        )
    band = np.clip(band, 0.01, float(np.max(np.abs(implicit))) + 1e-9)
    return np.abs(implicit) <= band


def _strut_radius_field(
    member_thickness: float,
    relative_density: np.ndarray | None,
) -> np.ndarray | float:
    radius = max(0.25, 0.5 * float(member_thickness))
    if relative_density is None:
        return radius
    # Strut volume scales approximately with radius squared. This local sizing
    # is a manufacturing interpretation, not a homogenized unit-cell solve.
    scale = np.sqrt(
        np.clip(np.asarray(relative_density, dtype=float), 0.05, 1.0) / 0.35
    )
    return radius * np.clip(scale, 0.45, 1.8)


def _cubic_or_octet_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None,
) -> np.ndarray:
    coordinates = np.moveaxis(
        np.indices(shape, dtype=float) + 0.5, 0, -1
    )
    period = max(float(cell_size), 1e-9)
    unit = np.mod(coordinates / period, 1.0)
    radius = _strut_radius_field(member_thickness, relative_density) / period

    if mode == "cubic":
        distance_to_grid = np.minimum(unit, 1.0 - unit)
        distance_sq = np.minimum.reduce([
            distance_to_grid[..., 1] ** 2 + distance_to_grid[..., 2] ** 2,
            distance_to_grid[..., 0] ** 2 + distance_to_grid[..., 2] ** 2,
            distance_to_grid[..., 0] ** 2 + distance_to_grid[..., 1] ** 2,
        ])
        # A one-voxel member whose centreline lies on a cell boundary otherwise
        # misses every voxel centre. Include the in-plane voxel footprint.
        raster_radius = np.asarray(radius) + 0.25 / period
        return distance_sq <= raster_radius ** 2

    # An octet unit cell connects each cube corner to the four corners of
    # every face through face-diagonal struts. Repeating this cell produces
    # the tetrahedral/octahedral truss network.
    corners = np.asarray([
        (x, y, z)
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
        for z in (0.0, 1.0)
    ])
    face_centers = np.asarray([
        (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
        (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
        (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
    ])
    distance_sq = np.full(shape, np.inf, dtype=float)
    for center in face_centers:
        fixed_axis = int(np.argmax(np.abs(center - 0.5)))
        face_value = center[fixed_axis]
        for corner in corners:
            if corner[fixed_axis] != face_value:
                continue
            segment = corner - center
            projection = np.sum((unit - center) * segment, axis=-1)
            projection /= float(np.dot(segment, segment))
            projection = np.clip(projection, 0.0, 1.0)
            delta = unit - (
                center + projection[..., None] * segment
            )
            distance_sq = np.minimum(
                distance_sq, np.sum(delta * delta, axis=-1)
            )
    # A diagonal centreline can cross a voxel without passing close to the
    # voxel centre. Rasterize the finite member against the voxel's half
    # diagonal; otherwise a nominal one-voxel octet can collapse to a handful
    # of isolated samples or even an empty-looking field.
    raster_radius = np.asarray(radius) + (0.5 / period)
    return distance_sq <= raster_radius ** 2


def _honeycomb_lattice(
    shape: tuple[int, int, int],
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None,
) -> np.ndarray:
    """Extruded regular-hexagonal walls with cell axes parallel to Z."""
    x, y, _ = np.indices(shape, dtype=float) + 0.5
    circumradius = max(1.5, 0.5 * float(cell_size))

    # Map to the nearest flat-top hex center using axial/cube rounding.
    q = (2.0 / 3.0) * x / circumradius
    r = (
        (-1.0 / 3.0) * x
        + (np.sqrt(3.0) / 3.0) * y
    ) / circumradius
    cube_x, cube_z = q, r
    cube_y = -cube_x - cube_z
    rx, ry, rz = np.rint(cube_x), np.rint(cube_y), np.rint(cube_z)
    dx, dy, dz = np.abs(rx - cube_x), np.abs(ry - cube_y), np.abs(rz - cube_z)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = (~fix_x) & (dy > dz)
    fix_z = ~(fix_x | fix_y)
    rx[fix_x] = -ry[fix_x] - rz[fix_x]
    ry[fix_y] = -rx[fix_y] - rz[fix_y]
    rz[fix_z] = -rx[fix_z] - ry[fix_z]

    center_x = circumradius * 1.5 * rx
    center_y = circumradius * np.sqrt(3.0) * (rz + 0.5 * rx)
    local_x, local_y = x - center_x, y - center_y
    inradius = np.sqrt(3.0) * 0.5 * circumradius
    projections = np.stack([
        np.abs(np.cos(angle) * local_x + np.sin(angle) * local_y)
        for angle in (np.pi / 6.0, np.pi / 2.0, 5.0 * np.pi / 6.0)
    ])
    distance_to_wall = np.maximum(
        0.0, inradius - np.max(projections, axis=0)
    )
    half_wall = _strut_radius_field(
        member_thickness, relative_density
    )
    return distance_to_wall <= np.asarray(half_wall)


def _remove_floating_lattice(
    envelope: np.ndarray,
    core: np.ndarray,
    skin: np.ndarray,
    member_thickness: float,
) -> np.ndarray:
    """Keep lattice members that form a load path to the envelope boundary.

    Sampling an implicit cell family on a coarse voxel grid can create tiny
    islands between the periodic network and the skin.  A one-member-radius
    attachment band closes sub-voxel raster gaps, then binary propagation
    removes any genuinely floating islands.  When skin is disabled, the
    envelope boundary is used only as the attachment seed and is not added to
    the returned structure.
    """
    from scipy import ndimage as ndi

    envelope = np.asarray(envelope, dtype=bool)
    core = np.asarray(core, dtype=bool) & envelope
    skin = np.asarray(skin, dtype=bool) & envelope
    if not np.any(core):
        return core

    envelope_boundary = envelope & ~ndi.binary_erosion(
        envelope,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    labels, count = ndi.label(
        core, structure=ndi.generate_binary_structure(3, 1)
    )
    if count <= 1:
        return core

    # The outer skin is itself connected and provides the intended attachment
    # between separate periodic wall/strut families. Keep only lattice
    # components that actually touch it; do not thicken all members merely to
    # close a rasterization gap.
    attachment_target = skin if np.any(skin) else envelope_boundary
    touching = np.unique(labels[attachment_target & (labels > 0)])
    if touching.size:
        kept = np.isin(labels, touching)
        if np.any(skin):
            return kept
        kept_labels, kept_count = ndi.label(
            kept, structure=ndi.generate_binary_structure(3, 1)
        )
        if kept_count == 1:
            return kept
        sizes = np.bincount(kept_labels.ravel())
        sizes[0] = 0
        return kept_labels == int(np.argmax(sizes))

    # A phase can place every centreline between boundary voxel centres. Keep
    # the largest resolved network instead of returning an empty structure.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


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
        occupied = np.argwhere(envelope)
        extents = (
            np.ptp(occupied, axis=0) + 1
            if len(occupied)
            else np.asarray(field.shape, dtype=int)
        )
        # Avoid a single oversized cell swallowing a thin engineering part.
        # Two cells across the second-largest envelope dimension is a useful
        # lower bound for a recognizable, connected 3-D lattice.
        second_largest = float(np.sort(np.maximum(extents, 1))[-2])
        largest_useful_cell = max(3.0, 0.5 * second_largest)
        cell_size = min(
            max(3.0, opts.cell_size_voxels * scale),
            largest_useful_cell,
        )
        member = min(
            max(0.75, opts.member_thickness_voxels * scale),
            0.35 * cell_size,
        )
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
            shape = tuple(int(v) for v in field.shape)
            if opts.mode in {"gyroid", "diamond"}:
                lattice = _periodic_lattice(
                    shape,
                    opts.mode,
                    cell_size,
                    member,
                    relative_density=target_relative_density,
                )
            elif opts.mode in {"cubic", "octet"}:
                lattice = _cubic_or_octet_lattice(
                    shape,
                    opts.mode,
                    cell_size,
                    member,
                    target_relative_density,
                )
            else:
                lattice = _honeycomb_lattice(
                    shape,
                    cell_size,
                    member,
                    target_relative_density,
                )
            core = envelope & lattice
            core = _remove_floating_lattice(
                envelope,
                core | solid_zone,
                skin,
                member,
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
    minimum_relative_density: Any = 0.15,
    maximum_relative_density: Any = 0.60,
    solid_transition_density: Any = 0.92,
) -> ManufacturingStructureOptions:
    """Parse graph-property values with one consistent validation path."""
    return ManufacturingStructureOptions(
        mode=str(mode or "Solid Envelope"),
        cell_size_voxels=float(cell_size_voxels or 8.0),
        member_thickness_voxels=float(member_thickness_voxels or 1.0),
        skin_thickness_voxels=float(skin_thickness_voxels or 0.0),
        variable_density=bool(variable_density),
        minimum_relative_density=float(minimum_relative_density or 0.15),
        maximum_relative_density=float(maximum_relative_density or 0.60),
        solid_transition_density=float(solid_transition_density or 0.92),
    )
