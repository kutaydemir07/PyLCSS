# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility facade for CAD, cache, and spatial geometry utilities."""

from .cad_geometry import (
    CadGeometry,
    _coerce_cells,
    _coerce_points,
    _extract_nodal_fields,
    _normalize_nodal_array,
    cad_evaluate_geometry,
)
from .geometry_cache import GeometryCache, evaluate_with_cache
from .spatial import (
    TRIMESH_AVAILABLE,
    _tetra_to_surface,
    _volume_to_surface,
    compute_sdf,
    make_background_grid,
    normalize_grid_coordinates,
)

__all__ = [
    "TRIMESH_AVAILABLE",
    "CadGeometry",
    "GeometryCache",
    "_coerce_cells",
    "_coerce_points",
    "_extract_nodal_fields",
    "_normalize_nodal_array",
    "_tetra_to_surface",
    "_volume_to_surface",
    "cad_evaluate_geometry",
    "compute_sdf",
    "evaluate_with_cache",
    "make_background_grid",
    "normalize_grid_coordinates",
]
