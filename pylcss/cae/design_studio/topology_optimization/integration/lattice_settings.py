# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Resolve physical dimensions for uniform infill or a solid topology result."""

from __future__ import annotations

from typing import Any

import numpy as np
from pylcss.input_values import as_bool

from ..manufacturing.lattice_families import family_for
from ..manufacturing.structures import ManufacturingStructureOptions
from .voxelization import voxel_size_from_bounds


def _characteristic_bounds_span(bounds: Any) -> float:
    try:
        lower, upper = (np.asarray(value, dtype=float)[:3] for value in bounds)
        span = upper - lower
    except (TypeError, ValueError):
        return 0.0
    if span.shape != (3,) or not np.all(np.isfinite(span)) or np.any(span <= 0.0):
        return 0.0
    return float(np.cbrt(np.prod(span)))


def _infill_structure_options(node, mode, voxel_size, bounds, resolve_cell_size):
    if voxel_size is None or not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("Lattice infill needs a positive physical voxel size.")
    cell_physical = float(resolve_cell_size(_characteristic_bounds_span(bounds)))
    if not np.isfinite(cell_physical) or cell_physical <= 0.0:
        raise ValueError("Lattice cell pitch must be finite and positive.")
    cell = cell_physical / voxel_size
    raw_thickness = node.get_property("infill_skin_thickness_mm")
    skin_mode = node.get_property("infill_skin_mode")
    if skin_mode is None:
        thickness = float(raw_thickness or 0.0)
        skin_mode = "Manual" if thickness != 0 else "None"
    if skin_mode == "None":
        skin = 0.0
    elif skin_mode == "Automatic":
        skin = max(1.5, cell / 40.0)
    elif skin_mode == "Manual":
        thickness = float(raw_thickness or 0.0)
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Enter a positive skin thickness or choose None.")
        skin = thickness / voxel_size
    else:
        raise ValueError("Skin mode must be None, Automatic, or Manual.")
    return ManufacturingStructureOptions(
        mode=str(mode),
        cell_size_voxels=cell,
        member_thickness_voxels=max(0.5, cell / 24.0),
        skin_thickness_voxels=skin,
        variable_density=False,
        target_relative_density=float(
            node.get_property("lattice_target_relative_density") or 0.0
        ),
        repair_connections=as_bool(node.get_property("repair_lattice_connections")),
    )


def resolve_lattice_structure_options(
    node, shape, bounds
) -> ManufacturingStructureOptions:
    resolve_cell_size = getattr(node, "resolve_infill_cell_size", None)
    if not callable(resolve_cell_size):
        return ManufacturingStructureOptions()
    mode = node.get_property("structure_mode")
    if family_for(mode) is None:
        raise ValueError("Choose a supported lattice family.")
    return _infill_structure_options(
        node,
        mode,
        voxel_size_from_bounds(shape, bounds),
        bounds,
        resolve_cell_size,
    )
