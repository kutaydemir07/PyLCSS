# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Explicit lattice geometry inside a topology envelope.

The desktop workflow optimizes a solid SIMP field before generating cells.
This module builds uniform or density-graded cells, preserves passive regions,
and reports connectivity and envelope coverage. Available material laws
calibrate strut thickness; TPMS walls use implicit-field occupancy calibration.
The resulting geometry requires its own structural verification.
Cell definitions and resolution guidance come from :mod:`.lattice_families`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Any

import numpy as np

from . import lattice_families as families
from .lattice_families import FAMILIES, LatticeFamily, normalize_family_key
from .member_sizing import OptimizedMemberPlan, rasterize_member_plan

logger = logging.getLogger(__name__)


def _resolve_mode(value: Any) -> str:
    """Return the canonical family key for a stored or displayed mode."""
    return normalize_family_key(value)


@dataclass(frozen=True)
class ManufacturingStructureOptions:
    """Controls for the explicit optimized manufacturing geometry."""

    mode: str = "solid"
    cell_size_voxels: float = 8.0
    member_thickness_voxels: float = 1.0
    skin_thickness_voxels: float = 0.0
    variable_density: bool = True
    minimum_relative_density: float = 0.15
    maximum_relative_density: float = 0.60
    solid_transition_density: float = 0.92
    # Optional whole-envelope occupancy target, fitted by building trials.
    target_relative_density: float = 0.0
    # Scale density-derived strut radii while retaining the minimum diameter.
    member_scale: float = 1.0
    # Off-pattern ligaments are a geometry change that requires explicit intent.
    repair_connections: bool = False

    def __post_init__(self) -> None:
        mode = _resolve_mode(self.mode)
        if not mode:
            raise ValueError(
                "Structure mode must be Solid Envelope or one of: "
                + ", ".join(family.display_name for family in FAMILIES.values())
                + "."
            )
        object.__setattr__(self, "mode", mode)
        for name in (
            "cell_size_voxels",
            "member_thickness_voxels",
            "skin_thickness_voxels",
            "minimum_relative_density",
            "maximum_relative_density",
            "solid_transition_density",
            "target_relative_density",
            "member_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "variable_density", bool(self.variable_density))
        object.__setattr__(self, "repair_connections", bool(self.repair_connections))
        is_lattice = self.mode != "solid"
        if not is_lattice:
            return
        if self.cell_size_voxels <= 0.0:
            raise ValueError("A lattice cell size must be positive.")
        if self.member_thickness_voxels <= 0.0:
            raise ValueError("A lattice member thickness must be positive.")
        if self.member_scale <= 0.0:
            raise ValueError("Lattice member scale must be positive.")
        if self.target_relative_density > 1.0:
            raise ValueError(
                "Target relative density must be between 0 and 1; "
                "0 sizes the structure from member thickness."
            )
        if self.variable_density and not (
            0.0
            < self.minimum_relative_density
            <= self.maximum_relative_density
            <= self.solid_transition_density
            <= 1.0
        ):
            raise ValueError(
                "Lattice densities require 0 < minimum <= maximum <= solid "
                "transition <= 1."
            )

    @property
    def family(self) -> LatticeFamily | None:
        """Registry entry for this mode; ``None`` for the solid envelope."""
        return FAMILIES.get(self.mode)

    @property
    def display_name(self) -> str:
        family = self.family
        return family.display_name if family is not None else "Solid Envelope"


def _envelope_thickness_voxels(envelope: np.ndarray) -> float:
    """Estimate representative envelope rib thickness in voxels.

    Return twice the median interior distance to a free surface. Padding with
    void includes the array's exterior faces in the measurement.
    """
    from scipy import ndimage as ndi

    solid = np.asarray(envelope, dtype=bool)
    if not np.any(solid):
        return 0.0
    padded = np.pad(solid, pad_width=1, mode="constant", constant_values=False)
    inside = ndi.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
    return float(2.0 * np.median(inside[solid]))


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


def _tpms_implicit(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Evaluate a family's level-set function over a voxel block."""
    return families.tpms_implicit_field(shape, mode, cell_size, origin=origin)


_TPMS_BAND_LUT: dict[tuple[str, int], np.ndarray] = {}


def _tpms_band_for_relative_density(
    relative_density: np.ndarray,
    mode: str,
    samples: int = 64,
) -> np.ndarray:
    """Convert requested occupancy to a calibrated implicit-field threshold.

    Cache sorted samples from one periodic cell at the requested resolution.
    Sheets use absolute field values; skeletal networks use signed values.
    Choose the first threshold containing at least the requested fraction.
    """
    # Calibrate at the resolved sampling density, including discrete ties.
    samples = int(np.clip(round(samples), 3, 64))
    key = (mode, samples)
    if key not in _TPMS_BAND_LUT:
        implicit = families.tpms_unit_cell_field(mode, samples)
        family = FAMILIES.get(mode)
        skeletal = family is not None and family.wall_mode == "skeletal"
        values = implicit if skeletal else np.abs(implicit)
        _TPMS_BAND_LUT[key] = np.sort(values.ravel())
    bands = _TPMS_BAND_LUT[key]
    requested = np.clip(np.asarray(relative_density, dtype=float), 0.0, 1.0)
    # Quantile interpolation across repeated values can miss thin walls.
    indices = np.ceil(requested * bands.size).astype(int) - 1
    indices = np.clip(indices, 0, bands.size - 1)
    return bands[indices] + 32.0 * np.finfo(float).eps


def _periodic_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None = None,
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    implicit = _tpms_implicit(shape, mode, cell_size, origin=origin)
    family = FAMILIES.get(mode)
    skeletal = family is not None and family.wall_mode == "skeletal"

    # Approximate thin-wall occupancy as specific area * thickness / pitch.
    # Skeletal ligaments use half the sheet conversion.
    wall_to_fraction = families.tpms_specific_surface_area(mode)
    if skeletal:
        wall_to_fraction *= 0.5
    minimum_fraction = float(
        np.clip(
            wall_to_fraction * float(member_thickness) / max(float(cell_size), 1e-9),
            0.02,
            0.80,
        )
    )
    minimum_band = float(
        _tpms_band_for_relative_density(
            np.asarray(minimum_fraction), mode, samples=int(round(cell_size))
        )
    )
    if relative_density is None:
        # Uniform cells use the 10% baseline or the larger wall-derived floor.
        default = _tpms_band_for_relative_density(
            np.asarray(0.10), mode, samples=int(round(cell_size))
        )
        band: np.ndarray | float = max(float(default), minimum_band)
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
    extreme = float(np.max(np.abs(implicit))) + 1e-9
    if skeletal:
        # A signed level; it legitimately runs negative, and the useful range
        # is bounded by the surface's own extremes rather than by zero.
        return implicit <= np.clip(band, -extreme, extreme)
    band = np.clip(band, 0.01, extreme)
    return np.abs(implicit) <= band


def dehomogenizing_cell_law(mode: str) -> Any:
    """Return an available geometry-calibration law without building a new one."""
    from .cell_material import cell_material_law

    return cell_material_law(mode, allow_build=False)


_STRUT_DENSITY_CURVE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _strut_density_curve(
    mode: str, cell_size: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Measure and cache a strut family's density-to-radius calibration.

    Threshold one unit-cell distance field over a radius sweep, including the
    raster allowance, and retain strictly increasing occupancy samples.
    Return densities and radii expressed as fractions of cell pitch.
    """
    samples = int(np.clip(round(float(cell_size)), 12, 40))
    key = (str(mode), samples)
    cached = _STRUT_DENSITY_CURVE.get(key)
    if cached is not None:
        return cached
    try:
        distance = families.strut_unit_cell_distance(mode, samples)
    except (ValueError, KeyError):
        return None
    radii = np.linspace(0.0, families.MAXIMUM_MEMBER_RADIUS, 96)
    densities = np.asarray(
        [
            float(
                families.strut_occupancy(distance, float(radius), float(samples)).mean()
            )
            for radius in radii
        ],
        dtype=float,
    )
    # Keep the strictly increasing part so the inverse is single valued. The
    # curve saturates once opposing members merge, and interpolating inside
    # that plateau would return an arbitrary radius from it.
    keep = np.concatenate(([True], np.diff(densities) > 1e-6))
    densities, radii = densities[keep], radii[keep]
    if densities.size < 2:
        return None
    result = (densities, radii)
    _STRUT_DENSITY_CURVE[key] = result
    return result


def _strut_radius_field(
    member_thickness: float,
    relative_density: np.ndarray | None,
    *,
    mode: str = "",
    cell_size: float = 0.0,
    member_scale: float = 1.0,
) -> np.ndarray | float:
    """Convert requested local density to a strut radius in voxels.

    Use the material law's measured thickness parameters when available,
    otherwise a measured radius curve. The explicit member thickness remains
    a minimum diameter; member_scale adjusts the density-derived radius.
    Only an unavailable calibration uses the square-root approximation.
    """
    radius = max(0.25, 0.5 * float(member_thickness))
    if relative_density is None:
        return radius

    density = np.clip(np.asarray(relative_density, dtype=float), 0.02, 1.0)
    if mode and cell_size > 0.0:
        law = dehomogenizing_cell_law(mode)
        if law is not None:
            # The law stores member thickness as a fraction of the cell pitch,
            # so it transfers to any cell size and any grid resolution.
            thickness = law.cell_parameter_for_density(density) * float(cell_size)
            return np.maximum(radius, 0.5 * thickness * float(member_scale))
        curve = _strut_density_curve(mode, cell_size)
        if curve is not None:
            densities, radii = curve
            # Radii are fractions of the cell pitch, so this transfers to any
            # cell size and any grid resolution, exactly as the law does.
            measured = np.interp(density, densities, radii) * float(cell_size)
            return np.maximum(radius, measured * float(member_scale))

    scale = np.sqrt(density / 0.35)
    return radius * np.clip(scale, 1.0, 1.8)


def _strut_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None,
    member_scale: float = 1.0,
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize the family edge list using constant or density-derived radii.

    A mask restricts distance evaluation to selected voxels; the returned mask
    still has the full block shape.
    """
    period = max(float(cell_size), 1e-9)
    radius = (
        _strut_radius_field(
            member_thickness,
            relative_density,
            mode=mode,
            cell_size=period,
            member_scale=member_scale,
        )
        / period
    )
    margin = families.strut_raster_margin(float(np.max(radius)), period)

    if mask is None:
        distance = families.strut_distance_field(
            shape, mode, period, origin=origin, margin=margin
        )
        return families.strut_occupancy(distance, radius, period)

    selected = np.asarray(mask, dtype=bool)
    distance = families.strut_distance_field(
        shape, mode, period, origin=origin, margin=margin, mask=selected
    )
    local_radius = (
        radius[selected] if np.ndim(radius) else radius  # type: ignore[index]
    )
    result = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    result[selected] = families.strut_occupancy(distance, local_radius, period)
    return result


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
    r = ((-1.0 / 3.0) * x + (np.sqrt(3.0) / 3.0) * y) / circumradius
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
    projections = np.stack(
        [
            np.abs(np.cos(angle) * local_x + np.sin(angle) * local_y)
            for angle in (np.pi / 6.0, np.pi / 2.0, 5.0 * np.pi / 6.0)
        ]
    )
    distance_to_wall = np.maximum(0.0, inradius - np.max(projections, axis=0))
    if relative_density is not None:
        dens = np.clip(np.asarray(relative_density, dtype=float), 0.02, 0.90)
        half_wall = np.maximum(
            max(0.25, 0.5 * float(member_thickness)),
            (np.sqrt(3.0) / 4.0) * circumradius * dens,
        )
    else:
        half_wall = max(0.25, 0.5 * float(member_thickness))
    return distance_to_wall <= np.asarray(half_wall)


def _remove_floating_lattice(
    envelope: np.ndarray,
    core: np.ndarray,
    skin: np.ndarray,
    member_thickness: float,
) -> np.ndarray:
    """Retain lattice components touching the skin or envelope boundary.

    Without a skin, the boundary is only an attachment seed. If no component
    touches an attachment seed, retain the largest component as a fallback.
    Disconnected boundary fragments remain available for optional repair.
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
    labels, count = ndi.label(core, structure=ndi.generate_binary_structure(3, 3))
    if count <= 1:
        return core

    # The outer skin is itself connected and provides the intended attachment
    # between separate periodic wall/strut families. Without one, the envelope
    # boundary is the equivalent seed: a fragment that reaches it is trimmed
    # lattice, not raster debris floating in a pore.
    attachment_target = skin if np.any(skin) else envelope_boundary
    touching = np.unique(labels[attachment_target & (labels > 0)])
    touching = touching[touching > 0]
    if touching.size:
        return np.isin(labels, touching)

    # A phase can place every centreline between boundary voxel centres. Keep
    # the largest resolved network instead of returning an empty structure.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _connect_lattice_components(
    envelope: np.ndarray,
    manufactured: np.ndarray,
    member_thickness: float,
    *,
    maximum_components: int = 20_000,
    maximum_rounds: int = 4,
) -> tuple[np.ndarray, int]:
    """Attempt to join voxel fragments within each envelope body.

    Connect fragments to the largest component with thickened straight lines
    clipped to that body. Repeat within the cost limits and return the field
    and number of attempted ligaments. Connectivity diagnostics report any
    remaining disconnection; clipping a straight line does not guarantee a path.
    """
    from scipy import ndimage as ndi

    connectivity = ndi.generate_binary_structure(3, 3)
    allowed = np.asarray(envelope, dtype=bool)
    # A copy, because the ligaments below are OR-ed in place and `np.asarray`
    # hands back the caller's own array when it is already boolean.
    result = np.array(manufactured, dtype=bool, copy=True)
    if not np.any(result):
        return result, 0

    envelope_labels, envelope_count = ndi.label(allowed, structure=connectivity)
    radius = max(1, int(np.ceil(0.5 * float(member_thickness))))
    bridges = 0

    for body_id in range(1, int(envelope_count) + 1):
        body = envelope_labels == body_id
        for _ in range(max(1, int(maximum_rounds))):
            labels, count = ndi.label(result & body, structure=connectivity)
            if count <= 1 or count > int(maximum_components):
                break
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            trunk_id = int(np.argmax(sizes))
            trunk = labels == trunk_id
            # One distance transform serves every fragment in this round: it
            # gives each void voxel its distance to the trunk and the index of
            # the trunk voxel that is nearest to it.
            distance, nearest = ndi.distance_transform_edt(~trunk, return_indices=True)
            # `minimum_position` finds each fragment's closest voxel to the
            # trunk in a single labelled pass. Scanning `labels == fragment_id`
            # per fragment instead is a full array traversal each time, which on
            # a supersampled grid with a hundred fragments dominates the whole
            # build.
            fragment_ids = [
                index for index in range(1, int(count) + 1) if index != trunk_id
            ]
            starts = ndi.minimum_position(distance, labels=labels, index=fragment_ids)
            addition = np.zeros_like(result, dtype=bool)
            for start in starts:
                start = np.asarray(start, dtype=int)
                end = nearest[(slice(None), *tuple(int(v) for v in start))]
                segment_length = int(
                    max(2, np.ceil(np.linalg.norm(end.astype(float) - start)) + 1)
                )
                samples = np.rint(np.linspace(start, end, segment_length)).astype(int)
                samples = np.clip(
                    samples,
                    np.zeros(3, dtype=int),
                    np.asarray(result.shape, dtype=int) - 1,
                )
                addition[tuple(samples.T)] = True
                bridges += 1
            if not np.any(addition):
                break
            addition = ndi.binary_dilation(
                addition, structure=connectivity, iterations=radius
            )
            # A ligament may not shortcut through a designed opening, and may
            # not leave the body it is repairing.
            result |= addition & body
    return result, bridges


def _keep_load_bearing_components(
    manufactured: np.ndarray,
    member_thickness: float,
) -> np.ndarray:
    """Remove disconnected raster specks smaller than a member-sized volume.

    This is a geometric size filter, not a structural load-path analysis.
    If all components are below the floor, retain the largest. Passive solids
    are added after cleanup so preserved interfaces are retained.
    """
    from scipy import ndimage as ndi

    source = np.asarray(manufactured, dtype=bool)
    labels, count = ndi.label(source, structure=ndi.generate_binary_structure(3, 3))
    if count <= 1:
        return source

    radius = max(0.5, 0.5 * float(member_thickness))
    minimum_voxels = max(2, int(round(4.0 / 3.0 * math.pi * radius**3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= minimum_voxels
    keep[0] = False
    if not np.any(keep):
        # Nothing clears the floor. Return the largest rather than nothing, so
        # an under-resolved study still shows what it built.
        keep = np.zeros_like(sizes, dtype=bool)
        keep[int(np.argmax(sizes))] = True
    return keep[labels]


def _attach_passive_interfaces(
    envelope: np.ndarray,
    manufactured: np.ndarray,
    passive_solid: np.ndarray,
    member_thickness: float,
) -> np.ndarray:
    """Attempt to attach preserved interfaces to nearby lattice material.

    Thicken a straight connection and clip it to the allowed envelope.
    Final connectivity checks determine whether the attachment succeeded.
    """
    from scipy import ndimage as ndi

    allowed = np.asarray(envelope, dtype=bool)
    result = np.asarray(manufactured, dtype=bool).copy()
    passive = np.asarray(passive_solid, dtype=bool) & allowed
    if not np.any(result) or not np.any(passive):
        return result

    connectivity = ndi.generate_binary_structure(3, 3)
    labels, count = ndi.label(passive, structure=connectivity)
    dilation_iterations = max(1, int(np.ceil(0.5 * float(member_thickness))))

    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        if np.any(ndi.binary_dilation(component, structure=connectivity) & result):
            result |= component
            continue

        target = result & ~component
        if not np.any(target):
            continue
        distance, nearest = ndi.distance_transform_edt(
            ~target,
            return_indices=True,
        )
        component_points = np.argwhere(component)
        component_distances = distance[component]
        start = component_points[int(np.argmin(component_distances))]
        end = nearest[(slice(None), *tuple(int(v) for v in start))]
        segment_length = int(
            max(2, np.ceil(np.linalg.norm(end.astype(float) - start)) + 1)
        )
        samples = np.rint(np.linspace(start, end, segment_length)).astype(int)
        samples = np.clip(
            samples,
            np.zeros(3, dtype=int),
            np.asarray(result.shape, dtype=int) - 1,
        )
        bridge = np.zeros_like(result, dtype=bool)
        bridge[tuple(samples.T)] = True
        bridge = ndi.binary_dilation(
            bridge,
            structure=connectivity,
            iterations=dilation_iterations,
        )
        # Do not manufacture a shortcut through a designed opening.
        bridge &= allowed
        result |= component | bridge

    return result


#: What the most recent lattice build did to stay connected, for the caller to
#: report. The build returns a bare array and is called from inside the
#: relative-density bisection, so there is no result dict to thread this
#: through; the values below describe the last completed build, which is the
#: one the study delivers. Mirrors ``surface_recovery.LAST_LATTICE_SIZING``.
LAST_STRUCTURE_DIAGNOSTICS: dict[str, Any] = {}


def _record_structure_diagnostics(
    envelope: np.ndarray,
    manufactured: np.ndarray,
    *,
    bridge_count: int,
    cell_size: float,
    member: float,
    display_name: str,
    repaired_voxels: int = 0,
) -> None:
    """Record how much of the envelope the built lattice reached."""
    from scipy import ndimage as ndi

    occupied = int(np.count_nonzero(envelope))
    built = int(np.count_nonzero(manufactured))
    _, components = ndi.label(
        np.asarray(manufactured, dtype=bool),
        structure=ndi.generate_binary_structure(3, 3),
    )
    _, envelope_bodies = ndi.label(
        np.asarray(envelope, dtype=bool),
        structure=ndi.generate_binary_structure(3, 3),
    )
    # Cell-sized block pooling estimates how much of the envelope received
    # lattice material without a costly whole-grid dilation.
    reach = 0.0
    if built and occupied:
        from skimage.measure import block_reduce

        block = max(1, int(round(cell_size)))
        envelope_blocks = block_reduce(
            np.asarray(envelope, dtype=bool), (block,) * 3, np.max
        )
        material_blocks = block_reduce(
            np.asarray(manufactured, dtype=bool), (block,) * 3, np.max
        )
        covered = int(np.count_nonzero(envelope_blocks & material_blocks))
        total = int(np.count_nonzero(envelope_blocks))
        reach = float(covered) / float(max(total, 1))
    LAST_STRUCTURE_DIAGNOSTICS.clear()
    LAST_STRUCTURE_DIAGNOSTICS.update(
        display_name=str(display_name),
        envelope_voxels=occupied,
        manufactured_voxels=built,
        relative_density=float(built) / float(max(occupied, 1)),
        envelope_reach=reach,
        component_count=int(components),
        envelope_body_count=int(envelope_bodies),
        connecting_ligaments=int(bridge_count),
        # Report added material as a fraction of the delivered geometry.
        repaired_voxels=int(repaired_voxels),
        repaired_fraction=(float(repaired_voxels) / float(built) if built else 0.0),
        resolved_cell_voxels=float(cell_size),
        resolved_member_voxels=float(member),
        envelope_thickness_voxels=_envelope_thickness_voxels(envelope),
    )
    if built and repaired_voxels and float(repaired_voxels) / float(built) > 0.01:
        logger.warning(
            "%s needed %d connecting ligaments to stay in one piece, adding "
            "%.1f%% of the delivered part as off-pattern material. The cell "
            "does not fit this envelope; reduce the cell pitch or refine the "
            "grid.",
            display_name,
            int(bridge_count),
            100.0 * float(repaired_voxels) / float(built),
        )
    if reach and reach < 0.90:
        logger.warning(
            "%s reached only %.0f%% of the optimized envelope at a resolved "
            "cell pitch of %.1f voxels. The Density view shows material this "
            "lattice does not fill; reduce the cell pitch or refine the grid.",
            display_name,
            100.0 * reach,
            cell_size,
        )


def build_manufacturing_field(
    density: np.ndarray,
    cutoff: float,
    options: ManufacturingStructureOptions | None = None,
    *,
    resolution_scale: float = 1.0,
    passive_solid_mask: np.ndarray | None = None,
    passive_void_mask: np.ndarray | None = None,
    member_plan: OptimizedMemberPlan | None = None,
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
    passive_solid = np.zeros_like(envelope, dtype=bool)
    if passive_solid_mask is not None:
        passive_solid = np.asarray(passive_solid_mask, dtype=bool)
        if passive_solid.shape != field.shape:
            raise ValueError("Passive-solid mask shape must match the density field.")
    passive_void = np.zeros_like(envelope, dtype=bool)
    if passive_void_mask is not None:
        passive_void = np.asarray(passive_void_mask, dtype=bool)
        if passive_void.shape != field.shape:
            raise ValueError("Passive-void mask shape must match the density field.")

    if opts.mode == "solid" or not np.any(envelope):
        manufactured = envelope
    else:
        scale = max(float(resolution_scale), 1e-9)
        # Physical cell and member sizes are engineering inputs. A poor fit
        # must be reported, not repaired by silently changing those inputs.
        cell_size = opts.cell_size_voxels * scale
        member = opts.member_thickness_voxels * scale
        skin = _boundary_skin(
            envelope,
            opts.skin_thickness_voxels * scale,
        )
        solid_zone = np.zeros_like(envelope, dtype=bool)

        if opts.variable_density:
            # The macro optimizer's physical density is the desired local
            # lattice relative density inside the selected topology envelope.
            # Never expand that envelope down to the minimum lattice density:
            # doing so turns low-density solver haze into printable material
            # that is absent from the Density view. High-density zones remain
            # solid and the intermediate zone receives a locally sized cell.
            solid_zone = field >= opts.solid_transition_density
            target_relative_density = np.clip(
                field,
                opts.minimum_relative_density,
                opts.maximum_relative_density,
            )
        else:
            target_relative_density = (
                np.full_like(field, float(opts.target_relative_density))
                if float(opts.target_relative_density or 0.0) > 0.0
                else None
            )
        shape = tuple(int(v) for v in field.shape)
        family = opts.family
        if family is not None and family.is_tpms:
            lattice = _periodic_lattice(
                shape,
                opts.mode,
                cell_size,
                member,
                relative_density=target_relative_density,
            )
        elif family is not None and family.is_strut and member_plan is not None:
            lattice = rasterize_member_plan(envelope, member_plan)
        elif family is not None and family.is_strut:
            # Only the envelope can hold lattice; the distance evaluation is
            # the dominant cost of a strut family, so do not spend it on voxels
            # that are about to be intersected away.
            lattice = _strut_lattice(
                shape,
                opts.mode,
                cell_size,
                member,
                target_relative_density,
                opts.member_scale,
                mask=envelope,
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
        # Rejoin what trimming to the envelope broke apart, then drop only what
        # is too small to be a member. Doing it in this order matters: the
        # connectivity repair has to see every fragment, because a fragment is
        # exactly what it exists to reattach.
        before_repair = manufactured
        if opts.repair_connections and not np.all(envelope):
            manufactured, bridge_count = _connect_lattice_components(
                envelope,
                manufactured,
                member,
            )
            repaired_voxels = int(np.count_nonzero(manufactured & ~before_repair))
        else:
            bridge_count = 0
            repaired_voxels = 0
        manufactured = _keep_load_bearing_components(manufactured, member)
        _record_structure_diagnostics(
            envelope,
            manufactured,
            bridge_count=bridge_count,
            repaired_voxels=repaired_voxels,
            cell_size=cell_size,
            member=member,
            display_name=opts.display_name,
        )

    if opts.mode != "solid" and opts.repair_connections:
        manufactured = _attach_passive_interfaces(
            envelope | passive_solid,
            manufactured,
            passive_solid,
            opts.member_thickness_voxels * max(float(resolution_scale), 1e-9),
        )
    manufactured |= passive_solid
    manufactured &= ~passive_void
    return manufactured.astype(float)


def cell_fit_warning(
    options: ManufacturingStructureOptions | None,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    """Warn when cell pitch exceeds representative envelope rib thickness.

    This assesses whether the part can hold a cell, independently of whether
    the build grid can resolve it. Include explicit repair statistics.
    """
    if options is None or options.mode == "solid" or not diagnostics:
        return None
    try:
        cell = float(diagnostics.get("resolved_cell_voxels") or 0.0)
        thickness = float(diagnostics.get("envelope_thickness_voxels") or 0.0)
        fragments = int(diagnostics.get("connecting_ligaments") or 0)
        repaired = float(diagnostics.get("repaired_fraction") or 0.0)
    except (TypeError, ValueError):
        return None
    if cell <= 0.0 or thickness <= 0.0 or cell <= thickness:
        return None
    repair_note = (
        f"Explicit connection repair added {fragments} ligament(s), contributing "
        f"{100.0 * repaired:.1f}% of the delivered part outside the cell pattern. "
        if fragments
        else "Connection repair added no ligaments. "
    )
    return (
        f"{options.display_name} builds a {cell:.0f}-voxel cell inside an "
        f"envelope whose ribs are a median {thickness:.0f} voxels thick, so the "
        "cell cannot tile within the material it is filling. "
        + repair_note
        + "Reduce the "
        "cell pitch, refine the analysis grid so a finer pitch resolves, or "
        "choose a family with a lower resolution floor."
    )


def cell_resolution_warning(
    options: ManufacturingStructureOptions | None,
    resolution_scale: float = 1.0,
) -> str | None:
    """Warn when the resolved pitch is below the family's sampling guidance."""
    if options is None or options.mode == "solid":
        return None
    family = options.family
    if family is None:
        return None
    resolved = options.cell_size_voxels * max(float(resolution_scale), 1e-9)
    required = float(family.minimum_cell_voxels)
    if resolved >= required:
        return None
    return (
        f"{family.display_name} resolves to {resolved:.1f} voxels per cell on "
        f"the build grid, below the {required:.0f} this family needs to stay "
        "connected. Increase the cell pitch, or refine the analysis grid or "
        "surface quality, and re-check the manufactured component count."
    )


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
    coordinates = [(np.arange(n, dtype=float) + 0.5) / float(n) for n in dims]
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
            radial = ((grids[a0] - c0) / max(r0, 1e-12)) ** 2 + (
                (grids[a1] - c1) / max(r1, 1e-12)
            ) ** 2 <= 1.0
            axial = (grids[axis_index[axis]] >= lo) & (grids[axis_index[axis]] <= hi)
            result |= radial & axial
        return result

    solid = boxes_mask(solid_boxes) | cylinders_mask(solid_cylinders)
    void = boxes_mask(void_boxes) | cylinders_mask(void_cylinders)
    solid &= ~void
    return solid, void


def achieved_relative_density(
    manufactured: np.ndarray,
    density: np.ndarray,
    cutoff: float,
) -> float:
    """Fraction of the optimized envelope that the built structure occupies."""
    envelope = np.asarray(density, dtype=float) >= float(
        np.clip(cutoff, 1e-6, 0.999999)
    )
    occupied = int(np.count_nonzero(envelope))
    if not occupied:
        return 0.0
    built = np.asarray(manufactured)
    built = built if built.dtype == bool else built >= 0.5
    return float(np.count_nonzero(built & envelope)) / float(occupied)


def resolve_target_relative_density(
    density: np.ndarray,
    cutoff: float,
    options: ManufacturingStructureOptions,
    *,
    resolution_scale: float = 1.0,
    passive_solid_mask: np.ndarray | None = None,
    passive_void_mask: np.ndarray | None = None,
    member_plan: OptimizedMemberPlan | None = None,
    tolerance: float = 0.01,
    max_evaluations: int = 14,
) -> tuple[ManufacturingStructureOptions, float]:
    """Fit the built envelope occupancy by measuring trial geometries.

    Search member_scale for graded struts with an available material law;
    otherwise search member thickness. Trials run on the recovery grid and
    include skin, cleanup, and passive regions. Return the closest sampled
    occupancy when rasterization or minimum features prevent an exact fit.
    """
    target = float(options.target_relative_density)
    if target <= 0.0 or options.mode == "solid":
        return options, float("nan")

    scale = max(float(resolution_scale), 1e-9)
    family = options.family
    dehomogenized = bool(
        options.variable_density
        and family is not None
        and family.is_strut
        and member_plan is None
        and dehomogenizing_cell_law(options.mode) is not None
    )

    if dehomogenized:
        control = "member_scale"
        # A de-homogenized member is already the right size; this only has to
        # cover trimming and skin effects, so a factor of a few either way is
        # the whole useful range.
        low, high = 0.25, 4.0
    else:
        control = "member_thickness_voxels"
        # Search through a full pitch and measure when each family fills.
        high = max(options.cell_size_voxels, options.member_thickness_voxels)
        low = min(0.5 / scale, 0.5 * high)
    if not (high > low > 0.0):
        return replace(options, target_relative_density=0.0), float("nan")

    def _build(value: float) -> tuple[np.ndarray, float]:
        trial = replace(
            options,
            target_relative_density=0.0,
            **{control: float(value)},
        )
        built = build_manufacturing_field(
            density,
            cutoff,
            trial,
            resolution_scale=scale,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
            member_plan=member_plan,
        )
        return built, achieved_relative_density(built, density, cutoff)

    best_value, best_achieved = high, 0.0
    _, high_density = _build(high)
    if high_density <= target:
        # Even the thickest member the cell can hold stays under target; that is
        # the honest answer, and reporting it lets the caller say so.
        return (
            replace(
                options,
                target_relative_density=0.0,
                **{control: high},
            ),
            high_density,
        )
    best_value, best_achieved = high, high_density

    for _ in range(int(max_evaluations)):
        middle = 0.5 * (low + high)
        _, reached = _build(middle)
        if abs(reached - target) < abs(best_achieved - target):
            best_value, best_achieved = middle, reached
        if abs(reached - target) <= float(tolerance):
            break
        if reached > target:
            high = middle
        else:
            low = middle

    return (
        replace(
            options,
            target_relative_density=0.0,
            **{control: float(best_value)},
        ),
        float(best_achieved),
    )
