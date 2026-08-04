# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Manufacturing constraints and explicit lattice representations."""

from .cell_material import (
    HOMOGENIZED_CELL_FAMILIES,
    CellMaterialLaw,
    cell_material_law,
)
from .homogenization import CubicElasticity, homogenize_cell
from .lattice_families import (
    FAMILIES,
    PUBLIC_FAMILIES,
    PUBLIC_FAMILY_KEYS,
    PUBLIC_LATTICE_FAMILY_NAMES,
    STRUT_FAMILIES,
    TPMS_FAMILIES,
    LatticeFamily,
    family_for,
    normalize_family_key,
)
from .member_sizing import (
    OptimizedMemberPlan,
    optimize_lattice_members,
    rasterize_member_plan,
)
from .structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
    cell_fit_warning,
    cell_resolution_warning,
    passive_region_masks,
)

__all__ = [
    "FAMILIES",
    "HOMOGENIZED_CELL_FAMILIES",
    "PUBLIC_FAMILIES",
    "PUBLIC_FAMILY_KEYS",
    "STRUT_FAMILIES",
    "TPMS_FAMILIES",
    "CellMaterialLaw",
    "CubicElasticity",
    "LatticeFamily",
    "ManufacturingStructureOptions",
    "OptimizedMemberPlan",
    "build_manufacturing_field",
    "cell_material_law",
    "cell_fit_warning",
    "cell_resolution_warning",
    "family_for",
    "homogenize_cell",
    "normalize_family_key",
    "optimize_lattice_members",
    "passive_region_masks",
    "rasterize_member_plan",
]
