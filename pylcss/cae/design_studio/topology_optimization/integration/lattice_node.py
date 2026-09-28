# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for variable-density lattice optimization."""

from __future__ import annotations

import logging
from typing import Any

from ..manufacturing import PUBLIC_FAMILIES, PUBLIC_FAMILY_KEYS
from ..configuration.presets import LATTICE_DESIGN_GOALS, industrial_lattice_defaults
from .topology_node import TopologyOptVoxelNode

logger = logging.getLogger(__name__)

# The picker offers the production core only. Every retired specialist family
# stays loadable, because the value is resolved through the family registry
# rather than through this list.
PUBLIC_LATTICE_FAMILY_NAMES: tuple[str, ...] = tuple(
    PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS
)
DEFAULT_LATTICE_FAMILY = "Gyroid Lattice"


class LatticeOptVoxelNode(TopologyOptVoxelNode):
    """Variable-density lattice optimization on a structured voxel grid.

    A lattice study is not a rendering option applied to a finished topology
    result — it is a different design problem. SIMP penalization and the
    Heaviside projection exist to drive the density to 0/1, while a graded
    lattice *is* the intermediate density, so the two work against each other:
    a penalized, projected field reaches rho ~ 1 throughout the structure,
    every voxel clears the solid-transition threshold, and no lattice is
    generated at all. This node therefore keeps the same graph inputs and the
    same solver as :class:`TopologyOptVoxelNode` but solves the homogenized
    cell law of the selected family, retains the graded field, and reconstructs
    the explicit manufactured cells from it.

    Every graph port, boundary-condition mapping, and execution stage is
    inherited unchanged; what this node adds is the manufacturing definition of
    the cell — family, pitch, wall/member thickness, skin, and the mass budget.
    """

    __identifier__ = "com.cad.sim.lattice_voxel"
    NODE_NAME = "Lattice Solver"

    def __init__(self) -> None:
        super().__init__()
        self.set_property("physics_mode", "Structural")
        self.set_property("design_goal", LATTICE_DESIGN_GOALS[0])
        # The cell family is the study definition, so unlike the topology node
        # there is no "Solid Envelope" entry: a lattice study that builds a
        # solid is a topology study, and it has its own node.
        self.create_property(
            "structure_mode",
            DEFAULT_LATTICE_FAMILY,
            widget_type="combo",
            items=list(PUBLIC_LATTICE_FAMILY_NAMES),
        )
        self.create_property(
            "lattice_settings_mode",
            "Guided",
            widget_type="combo",
            items=["Guided", "Manual"],
        )
        self.create_property("structure_cell_size_voxels", 8.0, widget_type="float")
        self.create_property(
            "structure_member_thickness_voxels",
            1.0,
            widget_type="float",
        )
        self.create_property(
            "structure_skin_thickness_voxels", 0.75, widget_type="float"
        )
        # Physical lattice dimensions, in model units. These are what a printer
        # capability (minimum feature, powder removal) is stated in, and unlike
        # the voxel-denominated values above they do not change meaning when
        # the grid is resized by the quality preset or a different part.
        # Zero means "use the voxel value", so existing studies are unchanged.
        self.create_property("lattice_cell_size_mm", 0.0, widget_type="float")
        self.create_property(
            "lattice_member_thickness_mm", 0.0, widget_type="float"
        )
        self.create_property(
            "lattice_skin_thickness_mm", 0.0, widget_type="float"
        )
        # The mass budget for the manufactured lattice, as a fraction of the
        # optimized envelope. This is the primary lattice control: cell pitch
        # and member thickness are printer capabilities, but only the relative
        # density says what the part will weigh, and it cannot be predicted from
        # the other two by eye. Zero keeps the explicit member thickness.
        self.create_property(
            "lattice_target_relative_density", 0.0, widget_type="float"
        )
        self.create_property("lattice_variable_density", True, widget_type="bool")
        self.create_property("lattice_min_relative_density", 0.15, widget_type="float")
        self.create_property("lattice_max_relative_density", 0.60, widget_type="float")
        self.create_property(
            "lattice_solid_transition_density", 0.92, widget_type="float"
        )
        self.create_property(
            "lattice_porosity",
            "Conservative",
            widget_type="combo",
            items=[
                "Conservative",
                "Balanced (Concept)",
                "Maximum Porosity (Concept)",
            ],
        )
        self.create_property(
            "optimize_lattice_members", True, widget_type="bool"
        )
        self.create_property(
            "lattice_max_member_thickness_voxels",
            3.0,
            widget_type="float",
        )
        self.create_property(
            "lattice_member_sizing_iterations",
            35,
            widget_type="int",
        )
        self.create_property(
            "lattice_buckling_length_factor",
            1.0,
            widget_type="float",
        )
        # A lattice has no automatic CAD replacement to wait for, so its own
        # geometry is the useful default view.
        self.set_property("visualization", "CAD")
        self.set_property("cad_export_filename", "lattice_optimized.step")

    def apply_guided_defaults(self) -> dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_lattice_defaults(
            self.get_property("design_goal"),
            self.get_property("manufacturing_process"),
            nelx=self.get_property("nelx") or 30,
            nely=self.get_property("nely") or 20,
            nelz=self.get_property("nelz") or 10,
            structure_mode=self.get_property("structure_mode"),
        )
        for key, value in settings.items():
            if self.has_property(key):
                self.set_property(key, value)
        return settings
