# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Manufacturing constraints and explicit rib/lattice representations."""

from .structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
    passive_region_masks,
)

__all__ = [
    "ManufacturingStructureOptions",
    "build_manufacturing_field",
    "passive_region_masks",
]
