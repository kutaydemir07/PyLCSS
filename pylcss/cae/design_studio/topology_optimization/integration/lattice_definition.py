# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Cell-family and density properties for direct geometric infill."""

from ..manufacturing.lattice_families import PUBLIC_LATTICE_FAMILY_NAMES

DEFAULT_LATTICE_FAMILY = "Gyroid Lattice"


class LatticeDefinitionMixin:
    def __init__(self) -> None:
        super().__init__()
        self.create_property(
            "structure_mode",
            DEFAULT_LATTICE_FAMILY,
            widget_type="combo",
            items=list(PUBLIC_LATTICE_FAMILY_NAMES),
        )
        self.create_property(
            "lattice_target_relative_density", 0.25, widget_type="float"
        )
        self.create_property("repair_lattice_connections", False, widget_type="bool")
