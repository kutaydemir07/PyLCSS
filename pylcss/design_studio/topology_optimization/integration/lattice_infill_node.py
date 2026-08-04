# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for lattice infill without optimization."""

from __future__ import annotations

import logging
from typing import Any

from ..manufacturing import PUBLIC_FAMILIES, PUBLIC_FAMILY_KEYS
from .lattice_node import LatticeOptVoxelNode

logger = logging.getLogger(__name__)

PUBLIC_LATTICE_FAMILY_NAMES: tuple[str, ...] = tuple(
    PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS
)

#: Cell pitch, as a fraction of the body's shortest bounding-box dimension,
#: for each automatic fineness step. Stated against the *part* rather than
#: against the analysis grid because that is the only thing a user can judge
#: by eye: "Fine" puts roughly sixteen cells across the thinnest direction.
AUTOMATIC_FINENESS_FRACTIONS: dict[str, float] = {
    "Coarse": 1.0 / 6.0,
    "Medium": 1.0 / 10.0,
    "Fine": 1.0 / 16.0,
    "Very Fine": 1.0 / 24.0,
}
DEFAULT_FINENESS = "Medium"


class LatticeInfillNode(LatticeOptVoxelNode):
    """Fill an existing solid with a lattice. No optimization.

    The lattice study next door answers "where should material go, and how
    dense should the cell be there" — it needs loads, supports and a volume
    budget, and it spends minutes solving. This node answers a different and
    much more common question: "take this body and make it a lattice." There
    is no load case, no objective and no solve; the density field is uniform
    over the whole input solid and the manufacturing stage does all the work.

    Because the field is uniform, none of the graded-lattice machinery applies:
    ``lattice_variable_density`` is off, so there is no solid-transition zone
    and no de-homogenization from a macro density. The cell pitch and the
    relative density are stated directly and are the only two controls that
    matter. Relative density is met by searching the wall/strut thickness and
    measuring what was built, the same bisection a mass budget uses on the
    optimizing node, so the number delivered is the number requested rather
    than whatever a nominal thickness happened to produce.

    The optimizer's density ceiling does not apply here either. On a lattice
    *study* that ceiling bounds the design variable, because the variable is a
    cell relative density and the homogenized tensor was only measured over a
    finite range. Here the field is not a design variable — it is the constant
    1.0 that says "this is all part" — and the requested lattice density is
    carried separately, so the full printable range stays available.
    """

    __identifier__ = "com.cad.sim.lattice_infill"
    NODE_NAME = "Lattice Infill"

    def __init__(self) -> None:
        super().__init__()
        # No boundary conditions: this node does not solve anything, so a
        # load or support port would only invite a connection that is silently
        # ignored. `design_domain` is the body to fill.
        self.set_port_deletion_allowed(True)
        for port_name in ("supports", "loads"):
            try:
                self.delete_input(port_name)
            except Exception:  # pragma: no cover - port set varies by version
                logger.debug("Lattice infill: no %s port to remove.", port_name)
        self.set_port_deletion_allowed(False)

        self.set_property("structure_mode", PUBLIC_LATTICE_FAMILY_NAMES[0])
        # A uniform infill, not a graded one. Everything the graded path adds —
        # the solid-transition zone, the min/max density band, the macro
        # de-homogenization — describes a field this node does not have.
        self.set_property("lattice_variable_density", False)
        # The envelope is the whole input body: density 1.0 everywhere inside
        # it, so the cutoff only has to separate body from background.
        self.set_property("volfrac", 1.0)
        self.set_property("density_cutoff", 0.5)
        self.set_property("lattice_target_relative_density", 0.25)
        self.set_property("optimize_lattice_members", False)
        self.set_property("validate_after_optimize", False)
        self.set_property("visualization", "Manufactured Mesh")
        self.set_property("cad_export_filename", "lattice_infill.step")
        self.set_property("nelx", 40)
        self.set_property("nely", 40)
        self.set_property("nelz", 40)

        self.create_property(
            "infill_cell_size_mode",
            "Automatic",
            widget_type="combo",
            items=["Automatic", "Manual"],
        )
        self.create_property(
            "infill_fineness",
            DEFAULT_FINENESS,
            widget_type="combo",
            items=list(AUTOMATIC_FINENESS_FRACTIONS),
        )
        # Solid boundary wall around the lattice, in model units. Zero leaves
        # the cells open at the trimmed boundary, which is what a powder
        # process needs; a closed skin seals the lattice and traps powder, so
        # it stays an explicit choice rather than a default.
        self.create_property("infill_skin_thickness_mm", 0.0, widget_type="float")

    # ── settings ─────────────────────────────────────────────────────────────

    def resolve_infill_cell_size(self, shortest_span: float) -> float:
        """Return the cell pitch in model units for the current settings.

        ``shortest_span`` is the smallest bounding-box dimension of the body
        being filled. Automatic fineness is stated against it, because a pitch
        that is sensible for a 300 mm bracket is meaningless on a 20 mm boss
        and the user has no way to convert between the two by eye.
        """
        manual = str(
            self.get_property("infill_cell_size_mode") or "Automatic"
        ).strip().lower() == "manual"
        if manual:
            requested = float(self.get_property("lattice_cell_size_mm") or 0.0)
            if requested > 0.0:
                return requested
            logger.warning(
                "Lattice infill is set to Manual with no cell size; using the "
                "automatic pitch instead."
            )
        fraction = AUTOMATIC_FINENESS_FRACTIONS.get(
            str(self.get_property("infill_fineness") or DEFAULT_FINENESS).strip(),
            AUTOMATIC_FINENESS_FRACTIONS[DEFAULT_FINENESS],
        )
        span = float(shortest_span)
        if not (span > 0.0):
            raise ValueError(
                "Lattice infill needs a 3-D body to measure its cell size "
                "against. Connect a CAD solid to design_domain."
            )
        return span * fraction

    def apply_guided_defaults(self) -> dict[str, Any]:
        """No guided solver defaults: there is no solve to configure."""
        return {}

    def run(
        self,
        progress_callback: Any = None,
        cancel_callback: Any = None,
        **kwargs: object,
    ) -> dict[str, Any] | None:
        """Generate the lattice infill without running a topology optimization solver."""
        import time
        import numpy as np
        from .execution_setup import prepare_topology_study
        from .execution_output import (
            build_topology_output,
            finalize_topology_output,
        )
        from ..optimization.results import TopologyOptVoxelResult

        study = prepare_topology_study(self)
        if study is None:
            return None

        design_domain = np.asarray(study.design_domain, dtype=float)
        result = TopologyOptVoxelResult(
            density=design_domain,
            solve_time_s=0.0,
            n_iter=0,
            converged=True,
            message="Lattice infill generated successfully.",
            optimizer_used="None (Direct Infill)",
            formulation_used="Uniform Lattice Infill",
            recovery_density=design_domain,
            recovery_cutoff=0.5,
        )

        recovery_started = time.perf_counter()
        output_context = build_topology_output(self, result, study)
        if output_context is None:
            return None
        recovery_time_s = time.perf_counter() - recovery_started
        finalization_started = time.perf_counter()
        output = finalize_topology_output(self, output_context, cancel_callback)
        output["timing"] = {
            "optimization_s": 0.0,
            "recovery_s": float(recovery_time_s),
            "validation_and_cad_s": float(
                time.perf_counter() - finalization_started
            ),
        }
        return output

