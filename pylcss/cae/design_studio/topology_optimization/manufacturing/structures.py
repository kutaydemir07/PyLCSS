# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Optimization-aware printable structures for topology-optimized designs.

Every mode here uses the optimized continuum density as a local relative-density
design field, so the explicit wall or strut is sized locally from that field
instead of inserting one constant-thickness lattice after the solve.

The families with a cubic effective tensor are de-homogenized from the measured
cell laws in :mod:`.cell_material`, which are the same laws the part-scale solve
was driven by. The rest — honeycomb, and the surfaces whose symmetry is not
cubic — keep an isotropic continuum interpretation. Independent re-analysis of
the explicit manufactured geometry is mandatory either way.

Which families exist, what each one's geometry is, and which of them carry a
measured law is not decided here: it all comes from :mod:`.lattice_families`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Any

import numpy as np

from . import lattice_families as families
from .lattice_families import FAMILIES, LatticeFamily, normalize_family_key, PUBLIC_LATTICE_FAMILY_NAMES
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
    skin_thickness_voxels: float = 0.75
    variable_density: bool = True
    minimum_relative_density: float = 0.15
    maximum_relative_density: float = 0.60
    solid_transition_density: float = 0.92
    # Fraction of the envelope the built structure should occupy. Zero keeps
    # ``member_thickness_voxels`` as the control. Cell pitch and member
    # thickness are the two numbers a printer's capability is stated in, but
    # they map onto mass in a way nobody can predict by eye: an 8 mm octet cell
    # with 1.6 mm struts measures 34% relative density -- a perforated solid,
    # not the open lattice those numbers suggest. Stating the target directly
    # and solving for the thickness is how a mass budget is actually met.
    target_relative_density: float = 0.0
    # Uniform thickening of a de-homogenized strut lattice. On that path the
    # cell law sets the member thickness from the local density, so member
    # thickness is no longer a free control and this is what a whole-part mass
    # budget has to move instead. 1.0 builds exactly what the law asks for.
    member_scale: float = 1.0

    def __post_init__(self) -> None:
        mode = _resolve_mode(self.mode)
        if not mode:
            raise ValueError(
                "Structure mode must be Solid Envelope or one of: "
                + ", ".join(
                    family.display_name for family in FAMILIES.values()
                )
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
        if not (0.05 <= self.member_scale <= 20.0):
            raise ValueError("Lattice member scale must be between 0.05 and 20.")
        if self.target_relative_density and not (
            0.02 <= self.target_relative_density <= 0.95
        ):
            raise ValueError(
                "Target relative density must be between 0.02 and 0.95, or 0 to "
                "size the structure from an explicit member thickness instead."
            )
        object.__setattr__(self, "variable_density", bool(self.variable_density))
        is_lattice = self.mode != "solid"
        if is_lattice and self.cell_size_voxels < 3.0:
            raise ValueError("A lattice cell must span at least 3 voxels.")
        if is_lattice and self.member_thickness_voxels <= 0.0:
            raise ValueError("A lattice member thickness must be positive.")
        if is_lattice and self.cell_size_voxels < 4.0 * self.member_thickness_voxels:
            raise ValueError(
                "Lattice cell size must be at least four times the member "
                "thickness so the cell openings remain resolved."
            )
        if is_lattice and self.skin_thickness_voxels > 0.25 * self.cell_size_voxels:
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
    def family(self) -> LatticeFamily | None:
        """Registry entry for this mode; ``None`` for the solid envelope."""
        return FAMILIES.get(self.mode)

    @property
    def display_name(self) -> str:
        family = self.family
        return family.display_name if family is not None else "Solid Envelope"


def _envelope_thickness_voxels(envelope: np.ndarray) -> float:
    """Representative wall thickness of an optimized envelope, in voxels.

    A periodic cell has to fit inside the material it is filling. The bounding
    box says nothing about that — an optimized result is a set of ribs whose
    thickness is a small fraction of the box — so this measures the ribs.

    The statistic is the material-weighted median of twice the interior
    distance transform: for every solid voxel, twice its distance to the
    nearest free surface is the local wall thickness, and the median over the
    material is the thickness that describes most of the part. A minimum would
    be set by a single tapering rib tip and would collapse the pitch for the
    whole study; a mean is pulled up by the few thick blocks around the load
    and support interfaces, which are exactly the regions that do *not* need a
    lattice fitted into them.
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
    """Map requested cell-relative density to an implicit level threshold.

    The lookup table is the numerical cumulative distribution of the level-set
    value over one periodic cell.  Consequently a requested relative density of
    ``r`` occupies approximately ``r`` of a resolved unit cell before skin and
    attachment solids are added.

    Which value is tabulated is what separates the two wall interpretations. A
    sheet is the shell ``|f| <= t`` around the surface, so its table is the
    distribution of ``|f|`` and the threshold is a half-band. A skeletal
    network is one side of it, ``f <= t``, so its table is the distribution of
    the signed ``f`` and the threshold is a level — which for these surfaces is
    negative over most of the useful density range. Sharing one calibration
    routine is what keeps a requested density meaning the same thing in both.
    """
    # Calibrate at the actual resolved points per cell. A continuum 64³ lookup
    # badly overfilled a Diamond sheet sampled on only 6–8 voxels per cell.
    samples = int(np.clip(round(samples), 3, 64))
    key = (mode, samples)
    if key not in _TPMS_BAND_LUT:
        implicit = families.tpms_unit_cell_field(mode, samples)
        family = FAMILIES.get(mode)
        skeletal = family is not None and family.wall_mode == "skeletal"
        values = implicit if skeletal else np.abs(implicit)
        _TPMS_BAND_LUT[key] = np.sort(values.ravel())
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
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    implicit = _tpms_implicit(shape, mode, cell_size, origin=origin)
    family = FAMILIES.get(mode)
    skeletal = family is not None and family.wall_mode == "skeletal"

    # Convert the minimum printable wall thickness to a cell-relative volume
    # fraction, then use the same calibrated implicit-field CDF as the target
    # density. A raw π*t/cell band strongly overfilled Diamond cells (often
    # >80% solid for a one-voxel wall in an 8-voxel cell).
    #
    # The conversion is the surface's own specific area: a sheet of thickness
    # ``t`` around a mid-surface of area ``A`` per unit cell fills ``A*t/a`` of
    # that cell. ``A`` is measured from this family's field
    # (:func:`~.lattice_families.tpms_specific_surface_area`), not assumed.
    # A single shared factor of 2 was wrong by roughly +55% for the gyroid and
    # -15% for the Schwarz P, in opposite directions, so a stated printable
    # wall built a different thickness on every surface.
    #
    # A skeletal ligament of thickness t is one solid body of width t rather
    # than two shells of t/2 either side of the surface, and its solid already
    # occupies half the cell at the surface itself, so the same printable wall
    # buys roughly half the cell fraction it does for a sheet.
    wall_to_fraction = families.tpms_specific_surface_area(mode)
    if skeletal:
        wall_to_fraction *= 0.5
    minimum_fraction = float(np.clip(
        wall_to_fraction
        * float(member_thickness)
        / max(float(cell_size), 1e-9),
        0.02,
        0.80,
    ))
    minimum_band = float(_tpms_band_for_relative_density(
        np.asarray(minimum_fraction), mode, samples=int(round(cell_size))
    ))
    if relative_density is None:
        # A calibrated floor, not a fixed level. The previous constant 0.12 was
        # tied to the gyroid/diamond amplitude and meant nothing for a surface
        # like Neovius, whose level-set runs to about 7; worse, at the common
        # 8-voxel cell it selected *no* voxels at all for either family, so a
        # uniform-density lattice survived only because the printable-wall
        # floor below happened to be larger. Stating it as a density gives the
        # same thin default sheet at fine resolution and cannot collapse.
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
    """Return the homogenized law that sizes ``mode``, or ``None``.

    One predicate for "does this family de-homogenize from a measured law",
    so the sizing code and the mass-budget search cannot disagree about which
    control actually moves the built density.
    """
    from .cell_material import cell_material_law

    return cell_material_law(mode, allow_build=False)


_STRUT_DENSITY_CURVE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _strut_density_curve(
    mode: str, cell_size: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Measured ``(relative density, member radius)`` for one strut family.

    The strut counterpart of the TPMS level-set CDF, and it exists for the same
    reason: a requested relative density has to select the geometry that
    actually measures that density, on this family, at this resolution.

    Without it the fallback was ``r = r0 * sqrt(rho / 0.35)`` clipped to a
    narrow band — an approximation that carries no information about the
    family at all, so the same request built wildly different parts. Measured
    on the bundled crush block, a BCC lattice asked for 10-50% relative density
    delivered 11% and fragmented into 223 pieces, because the requested density
    never reached the radius: only the member thickness did, and that had
    become a resolution floor rather than a design control.

    One distance field per (family, resolution), thresholded across the radius
    range through the same :func:`~.lattice_families.strut_occupancy` the part
    is rasterized with, so the raster allowance is included rather than being a
    systematic offset between the curve and the build.
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
                families.strut_occupancy(
                    distance, float(radius), float(samples)
                ).mean()
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
    """Return the local strut radius, in voxels, for a requested density field.

    This is the de-homogenization map for the strut families: it converts the
    macro optimizer's relative density back into the geometry that measures
    that density. When the family has a homogenized cell law, the conversion
    uses the measured density-to-thickness relation the law was built from,
    which is the same relation that produced the stiffness the optimizer was
    driven by. Anything else would generate geometry the macro solve did not
    describe.

    On that path the law fixes the *distribution* of material and
    ``member_scale`` sets its *level*: the requested member thickness no
    longer means anything, because the density field already determines the
    thickness everywhere. A whole-part mass budget therefore has to move
    ``member_scale``, which is what :func:`resolve_target_relative_density`
    searches on.

    Without a shipped law it uses the family's measured density-radius curve
    (:func:`_strut_density_curve`), which is the same inversion the law itself
    was built from. Only if that measurement is unavailable does it fall back
    to the old approximation — strut volume scaling with radius squared about a
    35% reference — which is clipped tightly because outside a narrow band it
    stops resembling the real relation at all.
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
            return np.maximum(0.25, 0.5 * thickness * float(member_scale))
        curve = _strut_density_curve(mode, cell_size)
        if curve is not None:
            densities, radii = curve
            # Radii are fractions of the cell pitch, so this transfers to any
            # cell size and any grid resolution, exactly as the law does.
            measured = np.interp(density, densities, radii) * float(cell_size)
            return np.maximum(0.25, measured * float(member_scale))

    scale = np.sqrt(density / 0.35)
    return radius * np.clip(scale, 0.45, 1.8)


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
    """Rasterize any strut family from its registry edge list.

    One distance evaluation against the family's centrelines, thresholded at
    the local member radius. This replaces the hand-written cubic and octet
    rasterizers and reproduces both of them voxel for voxel, which is what let
    the four families added around them share the calibration those two were
    tuned with.

    With ``mask`` the distance is evaluated only on the selected voxels and the
    result is written back into a full array. A lattice exists only inside the
    optimized envelope, so on a typical part that is most of the work skipped.
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
    result[selected] = families.strut_occupancy(
        distance, local_radius, period
    )
    return result


def _cubic_or_octet_lattice(
    shape: tuple[int, int, int],
    mode: str,
    cell_size: float,
    member_thickness: float,
    relative_density: np.ndarray | None,
    member_scale: float = 1.0,
) -> np.ndarray:
    """Backwards-compatible alias for :func:`_strut_lattice`."""
    return _strut_lattice(
        shape,
        mode,
        cell_size,
        member_thickness,
        relative_density,
        member_scale,
    )


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
    """Keep lattice members that reach the envelope boundary.

    Sampling an implicit cell family on a coarse voxel grid can create tiny
    islands between the periodic network and the skin.  Binary propagation from
    the attachment surface removes any genuinely floating island.  When skin is
    disabled, the envelope boundary is used only as the attachment seed and is
    not added to the returned structure.

    What this must not do is keep only the *largest* piece.  A periodic network
    trimmed to an arbitrary optimized envelope fragments wherever the envelope
    is thinner than the resolved cell, and those fragments are real lattice
    standing on the real load path.  The no-skin branch used to collapse them to
    the single largest component, and on the bundled rocker link that deleted
    96% of a no-skin Octet lattice -- the Density view showed the whole part and
    the Manufactured Mesh showed a corner of it.  Every fragment that reaches
    the envelope boundary is therefore kept, with or without a skin, and
    :func:`_connect_lattice_components` joins them into one body afterwards.
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
    """Join lattice fragments with short ligaments routed inside the envelope.

    Trimming a periodic cell family to an optimized envelope fragments it
    wherever the envelope is thinner than the resolved cell pitch. The fragments
    are all standing on the optimizer's own load path -- the Density view shows
    one connected envelope -- so the manufacturable repair is to connect them,
    not to delete every fragment but one.

    Each bridge is one member thick and is clipped to the envelope, so this can
    only add material the optimizer already asked for. It never crosses a
    designed void, which is what keeps the repair honest: a study whose density
    field is genuinely in two pieces still comes out in two pieces and still
    fails the downstream connectivity gate. Fragments in *different* envelope
    bodies are never joined, for the same reason.

    Returns the connected structure and the number of ligaments added. This is
    the same technique :func:`_attach_passive_interfaces` uses to reach a
    preserved interface, generalized to the lattice itself.

    ``maximum_components`` is a cost ceiling, and it used to sit at 512 — below
    the fragment counts this routine is most needed for. A badly under-resolved
    octet on the bundled crush block arrived here in 724 pieces, tripped the
    ceiling, returned immediately with zero ligaments, and the speck cull then
    reported 88 disconnected load-bearing bodies as the delivered part. The
    ceiling is now far above any repairable case, because the per-round cost is
    two whole-array passes regardless of how many fragments they find; the
    per-fragment work is a short line segment. A study that still exceeds it is
    not under-resolved, it is broken, and it leaves through the connectivity
    gate rather than through a silent early return.
    """
    from scipy import ndimage as ndi

    connectivity = ndi.generate_binary_structure(3, 1)
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
            distance, nearest = ndi.distance_transform_edt(
                ~trunk, return_indices=True
            )
            # `minimum_position` finds each fragment's closest voxel to the
            # trunk in a single labelled pass. Scanning `labels == fragment_id`
            # per fragment instead is a full array traversal each time, which on
            # a supersampled grid with a hundred fragments dominates the whole
            # build.
            fragment_ids = [
                index for index in range(1, int(count) + 1) if index != trunk_id
            ]
            starts = ndi.minimum_position(
                distance, labels=labels, index=fragment_ids
            )
            addition = np.zeros_like(result, dtype=bool)
            for start in starts:
                start = np.asarray(start, dtype=int)
                end = nearest[(slice(None), *tuple(int(v) for v in start))]
                segment_length = int(
                    max(2, np.ceil(np.linalg.norm(end.astype(float) - start)) + 1)
                )
                samples = np.rint(
                    np.linspace(start, end, segment_length)
                ).astype(int)
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
    """Remove sub-resolution raster specks from the manufactured structure.

    A one-voxel skin sampled on a curved boundary can contain isolated corner
    voxels even when the lattice core itself is connected. Those islands become
    separate shells in the recovered STL and cannot carry load. Explicit passive
    solids are added after this cleanup, so a user-preserved attachment is never
    silently removed.

    This deliberately no longer keeps only the largest component per source
    body. That rule was written for stray skin corners and is far too blunt for
    a trimmed periodic network, which legitimately fragments: on the bundled
    rocker link it cut a no-skin Octet lattice to 3.6% of its envelope while the
    Density view still showed the whole part. :func:`_connect_lattice_components`
    joins the fragments instead, and anything it cannot join is reported by the
    study's connectivity gate rather than quietly deleted here.

    A speck is a component too small to contain one member of the requested
    thickness, so removing it cannot remove a resolved feature.
    """
    from scipy import ndimage as ndi

    source = np.asarray(manufactured, dtype=bool)
    labels, count = ndi.label(
        source, structure=ndi.generate_binary_structure(3, 1)
    )
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
    """Bridge preserved interfaces to the nearest resolved lattice member.

    A graph-sized member plan can reach a bearing sleeve analytically while
    the final voxel raster misses it by one or two samples. Adding the passive
    sleeve after rasterization then creates the misleading result of an
    isolated, perfectly preserved interface. Build one short, physical
    attachment strut inside the topology envelope for each such component.
    The bridge never crosses a topology void, so a genuinely disconnected
    design still fails the downstream connectivity gate.
    """
    from scipy import ndimage as ndi

    allowed = np.asarray(envelope, dtype=bool)
    result = np.asarray(manufactured, dtype=bool).copy()
    passive = np.asarray(passive_solid, dtype=bool) & allowed
    if not np.any(result) or not np.any(passive):
        return result

    connectivity = ndi.generate_binary_structure(3, 1)
    labels, count = ndi.label(passive, structure=connectivity)
    dilation_iterations = max(1, int(np.ceil(0.5 * float(member_thickness))))

    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        if np.any(
            ndi.binary_dilation(component, structure=connectivity)
            & result
        ):
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
        samples = np.rint(
            np.linspace(start, end, segment_length)
        ).astype(int)
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
        structure=ndi.generate_binary_structure(3, 1),
    )
    _, envelope_bodies = ndi.label(
        np.asarray(envelope, dtype=bool),
        structure=ndi.generate_binary_structure(3, 1),
    )
    # A lattice reaching only a corner of its envelope is the failure this
    # measurement exists to catch: the relative density still looks plausible,
    # because it is measured against the whole envelope, but most of the part is
    # not there. Ask instead whether every cell-sized block of the envelope
    # actually received some material.
    #
    # Block reduction rather than a dilation or a distance transform: this runs
    # on every trial of the relative-density bisection, and at a resolved pitch
    # of 35 voxels a dilation of that radius costs seconds per call on a
    # supersampled grid while a max-pool over the same blocks costs milliseconds
    # and answers the same question.
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
        # What fraction of the delivered part the connectivity repair invented.
        # A ligament is routed straight from a fragment to the nearest trunk
        # voxel, so it does not lie on the cell pattern: it reads as material
        # in the middle of a pore, and at any real quantity it is the visible
        # sign that the cell does not fit the envelope. Reported rather than
        # left implicit in a ligament count, because a count means nothing
        # without the size of the part it was added to.
        repaired_voxels=int(repaired_voxels),
        repaired_fraction=(
            float(repaired_voxels) / float(built) if built else 0.0
        ),
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
        # The bounding box is not what limits the pitch, though: an optimized
        # envelope is a set of ribs a small fraction of that box thick, and a
        # cell that does not fit *inside a rib* is the failure this bound was
        # meant to prevent. Measure the envelope's own wall thickness and keep
        # the pitch inside it.
        #
        # This is why BCC used to come apart on thin sections where Octet did
        # not, which reads as a family bug and is not one: BCC's only interior
        # node is the body centre, so once a rib is thinner than a cell the
        # centre falls outside the envelope and nothing anchors the eight
        # struts that meet there. Octet carries its struts on the cell faces
        # and still catches a plane of them. Measured on the bundled crush
        # block at a 16 voxel pitch, BCC filled 47% of the sub-half-cell
        # regions against Octet's 73%, and needed 48 repair ligaments against
        # 2. Sizing the pitch to the part rather than to its bounding box fixes
        # both families at once, instead of special-casing one.
        thickness = _envelope_thickness_voxels(envelope)
        if thickness > 0.0:
            largest_useful_cell = min(largest_useful_cell, thickness)
        # Shrinking the pitch to fit a rib is only worth doing while the cell
        # is still resolved. Below the family's own floor the surface pinches
        # off at its necks and the connectivity cull removes most of it, which
        # is a worse result than a cell that is honestly too large for the rib.
        # When the two bounds cross, the pitch stays at the floor and
        # `cell_resolution_warning` is what tells the user the grid, not the
        # pitch, is what has to change.
        family_floor = opts.family
        resolution_floor = max(
            3.0,
            float(family_floor.minimum_cell_voxels)
            if family_floor is not None
            else 3.0,
        )
        cell_size = min(
            max(3.0, opts.cell_size_voxels * scale),
            max(largest_useful_cell, resolution_floor),
        )
        member = min(
            max(0.75, opts.member_thickness_voxels * scale),
            0.35 * cell_size,
        )
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
            target_relative_density = None
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
        if not np.all(envelope):
            manufactured, bridge_count = _connect_lattice_components(
                envelope,
                manufactured,
                member,
            )
            repaired_voxels = int(
                np.count_nonzero(manufactured & ~before_repair)
            )
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

    if opts.mode != "solid":
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
    """Report a cell too large for the ribs of the envelope it was trimmed to.

    Distinct from :func:`cell_resolution_warning`, and it catches a case that
    one cannot. Resolution asks whether the *grid* can represent the cell;
    this asks whether the *part* can hold it. An optimized envelope is a set of
    ribs, and a cell wider than a rib cannot tile inside one: the periodic
    network arrives in hundreds of pieces, and the connectivity repair then
    rebuilds a load path out of straight off-pattern ligaments.

    On the bundled payload fitting an octet cell resolved to 16 voxels against
    ribs a median 9.5 voxels thick, fragmented into 356 pieces, and needed 981
    ligaments — 3.2% of the delivered part invented by the repair. Every
    individual check passed: the pitch met the family's resolution floor, the
    envelope reach was 99%, and the component count came back at 1. The part
    was still wrong, and this is the question none of them asked.
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
    return (
        f"{options.display_name} builds a {cell:.0f}-voxel cell inside an "
        f"envelope whose ribs are a median {thickness:.0f} voxels thick, so the "
        f"cell cannot tile within the material it is filling. The trimmed "
        f"network came apart and {fragments} connecting ligament(s) were added "
        f"to restore one load path, contributing {100.0 * repaired:.1f}% of the "
        "delivered part as geometry that is not on the cell pattern. Reduce the "
        "cell pitch, refine the analysis grid so a finer pitch resolves, or "
        "choose a family with a lower resolution floor."
    )


def cell_resolution_warning(
    options: ManufacturingStructureOptions | None,
    resolution_scale: float = 1.0,
) -> str | None:
    """Report a cell pitch too coarse for the family to stay connected.

    A periodic surface sampled below its own neck width does not come out
    thinner — it comes out in pieces, and the connectivity cull then removes
    most of it. The resulting body has a plausible-looking relative density and
    no load path, which is the failure mode worth naming explicitly rather than
    leaving to the component count. Each family carries its own measured floor;
    see :attr:`LatticeFamily.minimum_cell_voxels`.
    """
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
    target_relative_density: Any = 0.0,
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
        target_relative_density=float(target_relative_density or 0.0),
    )


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
    """Size the member thickness so the built structure hits its target density.

    Returns the options to build with and the relative density achieved. There
    is no closed form to invert here: the built field is the intersection of a
    periodic lattice with an arbitrary optimized envelope, then culled of
    members that carry no load path, so the only reliable relation between
    thickness and mass is to build it and measure. Relative density rises
    monotonically with member thickness, which is what makes bisection sound.

    Bisecting at the *recovery* resolution rather than the analysis resolution
    is deliberate: the two disagree by about 20% because the recovery grid is
    supersampled anisotropically, and the recovered solid is the one that gets
    manufactured.

    Which control is searched depends on how the lattice is sized. A family
    that de-homogenizes from a measured cell law takes its member thickness
    from the local density, so thickness is not free and moving it changes
    nothing; the uniform ``member_scale`` is searched instead. Everything else
    searches the member thickness as before.
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
        # Search only the band a member is actually allowed to occupy. The
        # option validation rejects a member thicker than a quarter of the cell
        # so the cell openings stay resolved, and build_manufacturing_field
        # floors the resolved member at 0.75 recovery voxels, below which the
        # grid cannot represent it.
        high = 0.25 * options.cell_size_voxels
        low = min(0.75 / scale, 0.5 * high)
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
