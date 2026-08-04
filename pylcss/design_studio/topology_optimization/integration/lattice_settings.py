# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Resolve guided or manual lattice manufacturing settings.

Lattice dimensions are coupled to the analysis grid: a cell needs at least
three voxels, and each wall/member and the optional skin may occupy at most one
quarter of the cell pitch.  Keeping this resolution in one place lets the
inspector describe the same limits that execution validates.
"""

from __future__ import annotations

import logging
from typing import Any

from ..manufacturing.lattice_families import family_for
from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    structure_options_from_values,
)
from .voxelization import lattice_voxels_from_length, voxel_size_from_bounds

logger = logging.getLogger(__name__)

GUIDED_LATTICE_SETUP = "Guided"
MANUAL_LATTICE_SETUP = "Manual"

#: A lattice cell must stay at least this many members across, or its openings
#: stop being resolved and the cell becomes a perforated solid. Mirrors the
#: bound `ManufacturingStructureOptions.__post_init__` validates.
MINIMUM_CELL_TO_MEMBER_RATIO = 4.0


def lattice_setup_is_guided(value: Any) -> bool:
    """Return whether lattice manufacturing dimensions are automatic."""
    return str(value or GUIDED_LATTICE_SETUP).strip().lower() != "manual"


def minimum_feature_voxels(
    minimum_feature_size: Any,
    voxel_size: Any,
) -> float:
    """Convert a printable feature size in model units to analysis voxels."""
    try:
        physical = float(minimum_feature_size or 0.0)
        edge = float(voxel_size or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if physical <= 0.0 or edge <= 0.0:
        return 0.0
    return physical / edge


def node_minimum_feature_size(node: Any) -> float:
    """Return the printable feature floor a study states, in model units.

    The topology node already carries this as the thinnest member the design
    may contain. For a lattice study that is the same process capability — a
    minimum printable wall or strut — so it governs the cell as well as the
    envelope rather than being restated as a second number.
    """
    try:
        return max(0.0, float(node.get_property("minimum_member_size_mm") or 0.0))
    except (TypeError, ValueError):
        return 0.0


#: Supersampling the recovery grid is expected to apply to a cell pitch stated
#: in analysis voxels. It is not a guess about a particular study: recovery now
#: refines the grid until the selected family's own resolved floor is met (see
#: `surface_recovery`), and the point budget on a typical part allows 3-5x. It
#: is used here only to state the guided pitch in the right units.
EXPECTED_RECOVERY_SUPERSAMPLE = 4.0

#: Finest cell pitch, in analysis voxels, a guided study will ask for. Below
#: this the macro density field has no room to grade from cell to cell.
MINIMUM_GUIDED_CELL_VOXELS = 3.5


def guided_lattice_voxel_dimensions(
    mode: Any,
    *,
    voxel_size: Any = None,
    minimum_feature_size: Any = 0.0,
) -> tuple[float, float, float]:
    """Return a connected, family-resolved cell, member, and skin size.

    The proportions are voxel-denominated, and a voxel is a different physical
    length on every part and every grid. On its own that lets a guided study
    produce a wall or strut below anything the process can build, without ever
    stating a number in model units. ``minimum_feature_size`` is the printable
    floor in model units: when it is given together with the grid's
    ``voxel_size``, the member is raised to meet it and the cell pitch grows
    with it so the openings stay resolved.
    """
    family = family_for(mode)
    family_minimum = (
        float(family.minimum_cell_voxels) if family is not None else 8.0
    )
    # `minimum_cell_voxels` is a floor on the *resolved* build grid, and this
    # function returns a pitch in *analysis* voxels. Using one as the other is
    # a unit error, and it makes the cell coarse by the whole supersampling
    # factor: on the bundled crush block an octet came out at 16 analysis
    # voxels, an 82 mm pitch with only four cells across a 320 mm part, when
    # the resolved requirement was already met at 5.3. Divide by the
    # supersampling the recovery grid supplies and the pitch lands where the
    # family actually needs it — about 27 mm here, twelve cells across.
    cell_size = max(
        MINIMUM_GUIDED_CELL_VOXELS,
        family_minimum / EXPECTED_RECOVERY_SUPERSAMPLE,
    )
    # The thinnest wall the resolved build grid can carry, not a fixed
    # proportion of the pitch. This number is not cosmetic: it sets the *floor*
    # on relative density, because a wall of thickness ``t`` around a surface of
    # specific area ``A`` cannot occupy less than ``A*t/pitch`` of its cell. At
    # the 8:1 proportion this used to carry, that floor was 39% of a gyroid
    # cell — so a guided study whose optimizer asked for a 15% cell built a 39%
    # one instead, and the delivered part was far denser than the solve. A
    # 24:1 proportion is about one voxel of wall on the supersampled recovery
    # grid, which puts the floor near 13% and under the whole working range, so
    # the requested density is what governs the built geometry.
    #
    # A stated process capability still outranks this, below: that floor is a
    # real constraint and is meant to bind.
    member_size = max(0.5, cell_size / 24.0)
    member_floor = minimum_feature_voxels(minimum_feature_size, voxel_size)
    if member_floor > member_size:
        member_size = member_floor
        # A cell must stay at least four members across, which is the same
        # bound `ManufacturingStructureOptions` validates. Widening the pitch
        # rather than clipping the member keeps the printable floor exact.
        cell_size = max(cell_size, MINIMUM_CELL_TO_MEMBER_RATIO * member_size)
    # No skin. A guided lattice study used to wrap its result in a solid
    # boundary shell, and that shell is not a detail: it seals the cell family
    # inside an opaque solid, so the manufactured mesh shows no lattice at all,
    # and on a powder-bed process it traps unfused powder with no evacuation
    # path. The study already emitted both of those as warnings while building
    # it anyway. A skin is a legitimate design choice — it is still available in
    # Manual mode — but it cannot be what a study does by default.
    skin_size = 0.0
    return cell_size, member_size, skin_size


#: Absolute floor on an infill cell pitch, in build voxels. Mirrors the bound
#: `ManufacturingStructureOptions` validates.
#:
#: The infill node sizes its own grid from the requested pitch, so this is a
#: backstop rather than the working limit: it is reached only when the
#: build-quality budget caps the grid below what the family asked for, and the
#: family's own `minimum_cell_voxels` is what governs before that.
MINIMUM_INFILL_CELL_VOXELS = 3.0


def _characteristic_bounds_span(bounds: Any) -> float:
    """Representative size of the body, in model units.

    The geometric mean of the three bounding-box dimensions rather than the
    smallest. "Twelve cells across the part" is what a fineness setting means
    to the person choosing it, and on a plate-like body the smallest dimension
    is the thickness — measuring against it asks for a pitch an order of
    magnitude finer than intended and lands under the grid resolution.
    """
    import numpy as np

    try:
        lower = np.asarray(bounds[0], dtype=float).reshape(-1)[:3]
        upper = np.asarray(bounds[1], dtype=float).reshape(-1)[:3]
    except Exception:
        return 0.0
    span = np.abs(upper - lower)
    if span.size < 3 or not np.all(np.isfinite(span)) or np.any(span <= 0.0):
        return 0.0
    return float(np.cbrt(np.prod(span)))


def _infill_structure_options(
    node: Any,
    mode: Any,
    voxel_size: Any,
    bounds: Any,
    resolve_cell_size: Any,
) -> ManufacturingStructureOptions:
    """Manufacturing options for a uniform lattice infill of a solid body."""
    cell_physical = float(resolve_cell_size(_characteristic_bounds_span(bounds)))
    cell_voxels = lattice_voxels_from_length(cell_physical, voxel_size, 8.0)
    if cell_voxels < MINIMUM_INFILL_CELL_VOXELS:
        # Only reachable when the build-quality budget capped the grid far
        # below the pitch: `LatticeInfillNode.resolve_build_plan` sizes the
        # voxel from the pitch and grows the pitch rather than thinning the
        # wall, so a planned grid lands at the family's own floor by
        # construction. Clamping is the only honest option here — the
        # grid defines the envelope every later stage is trimmed against — but
        # it coarsens what was asked for, so say what happened and what fixes
        # it. This node runs no FE solve, so a finer grid costs only memory.
        try:
            edge = float(voxel_size or 0.0)
        except (TypeError, ValueError):
            edge = 0.0
        achievable = MINIMUM_INFILL_CELL_VOXELS * edge
        logger.warning(
            "Lattice infill asked for a %.3g cell on a build grid whose voxel "
            "is %.3g, which is %.1f voxels per cell; a cell needs at least "
            "%.0f. Building at %.3g instead. Raise Build Quality or the cell "
            "pitch — this node runs no FE solve, so a finer grid costs only "
            "memory and triangles.",
            cell_physical,
            edge,
            cell_voxels,
            MINIMUM_INFILL_CELL_VOXELS,
            achievable,
        )
        cell_voxels = MINIMUM_INFILL_CELL_VOXELS
    # The wall is only a starting point: the requested relative density is met
    # by searching this thickness and measuring what gets built, so it has to
    # be small enough not to be the binding constraint at the thinnest density
    # the user might ask for.
    member_voxels = max(0.5, cell_voxels / 24.0)
    member_floor = minimum_feature_voxels(
        node_minimum_feature_size(node), voxel_size
    )
    if member_floor > member_voxels:
        member_voxels = member_floor
        cell_voxels = max(
            cell_voxels, MINIMUM_CELL_TO_MEMBER_RATIO * member_voxels
        )
    skin_voxels = lattice_voxels_from_length(
        node.get_property("infill_skin_thickness_mm"),
        voxel_size,
        0.0,
        allow_zero=True,
    )
    skin_voxels = min(float(skin_voxels), 0.25 * cell_voxels)
    target = float(node.get_property("lattice_target_relative_density") or 0.0)
    return structure_options_from_values(
        mode,
        cell_voxels,
        member_voxels,
        skin_voxels,
        # Uniform, not graded: there is no macro density field to grade from.
        False,
        0.15,
        0.60,
        0.92,
        target,
    )


def resolve_lattice_structure_options(
    node: Any,
    shape: tuple[int, int, int],
    bounds: Any,
) -> ManufacturingStructureOptions:
    """Resolve and validate the node's lattice manufacturing interpretation."""
    mode = node.get_property("structure_mode")
    voxel_size = voxel_size_from_bounds(shape, bounds)
    guided = lattice_setup_is_guided(
        node.get_property("lattice_settings_mode")
    )

    infill_cell_size = getattr(node, "resolve_infill_cell_size", None)
    if callable(infill_cell_size) and family_for(mode) is not None:
        # Lattice infill: a uniform fill of an existing body. The pitch is
        # stated against the part, the density is stated directly, and none of
        # the graded-field controls apply because there is no graded field.
        return _infill_structure_options(
            node, mode, voxel_size, bounds, infill_cell_size
        )

    if guided and family_for(mode) is not None:
        cell_size, member_size, skin_size = guided_lattice_voxel_dimensions(
            mode,
            voxel_size=voxel_size,
            minimum_feature_size=node_minimum_feature_size(node),
        )
        variable_density = True
        # The published working range for a graded TPMS or strut lattice is
        # roughly 10-50% relative density; above that the cell stops being a
        # lattice and becomes a perforated solid, and it leaves the density
        # band the shipped cell laws were measured over. These two numbers now
        # bound the design variable itself (see
        # `TopologyOptVoxelProblem.lattice_maximum_density`), so the ceiling is
        # what the optimizer solves against rather than a clamp applied to a
        # saturated field afterwards.
        minimum_density = 0.10
        maximum_density = 0.50
        # Only a preserved passive interface reaches this now, which is the
        # intent: a bearing seat or bolt pad is solid, the lattice is not.
        solid_transition = 0.95
        target_density = 0.0
    else:
        raw_skin_voxels = node.get_property(
            "structure_skin_thickness_voxels"
        )
        try:
            skin_fallback = (
                0.75
                if raw_skin_voxels is None
                or str(raw_skin_voxels).strip() == ""
                else float(raw_skin_voxels)
            )
        except (TypeError, ValueError):
            skin_fallback = 0.75
        cell_size = lattice_voxels_from_length(
            node.get_property("lattice_cell_size_mm"),
            voxel_size,
            node.get_property("structure_cell_size_voxels") or 8.0,
        )
        member_size = lattice_voxels_from_length(
            node.get_property("lattice_member_thickness_mm"),
            voxel_size,
            node.get_property("structure_member_thickness_voxels") or 1.0,
        )
        skin_size = lattice_voxels_from_length(
            node.get_property("lattice_skin_thickness_mm"),
            voxel_size,
            skin_fallback,
            allow_zero=True,
        )
        variable_density = node.get_property("lattice_variable_density")
        minimum_density = node.get_property(
            "lattice_min_relative_density"
        )
        maximum_density = node.get_property(
            "lattice_max_relative_density"
        )
        solid_transition = node.get_property(
            "lattice_solid_transition_density"
        )
        target_density = node.get_property(
            "lattice_target_relative_density"
        )

    return structure_options_from_values(
        mode,
        cell_size,
        member_size,
        skin_size,
        variable_density,
        minimum_density,
        maximum_density,
        solid_transition,
        target_density,
    )


def lattice_dimension_range_text(
    mode: Any,
    voxel_size: float | None,
    *,
    cell_size: float | None = None,
) -> str:
    """Describe accepted and recommended manual dimensions for the active grid."""
    family = family_for(mode)
    recommended_voxels = max(
        3.0,
        float(family.minimum_cell_voxels) if family is not None else 3.0,
    )
    family_name = family.display_name if family is not None else "this family"
    if voxel_size is None or voxel_size <= 0.0:
        text = (
            "Manual range: cell pitch >= 3 analysis voxels; "
            f"{family_name} is recommended at >= {recommended_voxels:g} "
            "voxels per cell. Wall/member must be > 0 and no more than 25% "
            "of the pitch; skin may be 0 to 25% of the pitch."
        )
    else:
        hard_minimum = 3.0 * float(voxel_size)
        recommended = recommended_voxels * float(voxel_size)
        text = (
            f"Manual range on this grid ({float(voxel_size):.3g} per voxel): "
            f"cell pitch >= {hard_minimum:.3g}; {family_name} is recommended "
            f"at >= {recommended:.3g} ({recommended_voxels:g} voxels). "
            "Wall/member must be > 0 and no more than 25% of the pitch; "
            "skin may be 0 to 25% of the pitch."
        )
    text += (
        " Relative densities must satisfy 0 < minimum < maximum <= "
        "solid transition <= 1. Target density may be 0 (use member "
        "thickness) or 0.02 to 0.95."
    )
    try:
        current_cell_size = float(cell_size)
    except (TypeError, ValueError):
        current_cell_size = 0.0
    if current_cell_size > 0.0:
        text += (
            f" At the current {current_cell_size:.3g} pitch, the "
            f"wall/member maximum is {0.25 * current_cell_size:.3g}."
        )
    return text
