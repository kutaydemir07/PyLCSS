# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for lattice infill without optimization."""

from __future__ import annotations

import logging
import math
from typing import Any

from ..manufacturing import PUBLIC_FAMILIES, PUBLIC_FAMILY_KEYS
from ..manufacturing.lattice_families import family_for
from .lattice_node import LatticeOptVoxelNode

logger = logging.getLogger(__name__)

PUBLIC_LATTICE_FAMILY_NAMES: tuple[str, ...] = tuple(
    PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS
)

#: Cell pitch, as a fraction of the body's characteristic size, for each
#: automatic fineness step. Stated against the *part* rather than against a
#: voxel grid because that is the only thing a user can judge by eye: "Fine"
#: puts roughly five cells across the part.
#:
#: The ceiling on these is not a matter of taste. Rasterizing a lattice needs
#: ``r`` voxels per cell to hold a representable wall (:meth:`required_cell_voxels`),
#: so on a budget of ``B`` voxels the cells that fit across a body of volume
#: ``V`` come out at ``V**(1/3) / (r * (V/B)**(1/3))`` — which is ``B**(1/3)/r``,
#: independent of the body. A 25% gyroid needs r=24, so the most any part can
#: ever carry is 4.2 cells at Standard, 5.2 at High and 6.3 at Professional.
#:
#: The previous ladder asked for 6/10/16/24 cells. Not one of those was
#: reachable at any build quality, so every setting was silently coarsened to
#: the same pitch and the control did nothing — which is exactly how it read.
#: These four are chosen to sit under the ceiling instead: Coarse and Medium
#: build at Standard, Fine needs High, Very Fine needs Professional. Strut
#: families resolve at a thinner wall and reach all four sooner.
AUTOMATIC_FINENESS_FRACTIONS: dict[str, float] = {
    "Coarse": 1.0 / 3.0,
    "Medium": 1.0 / 4.0,
    "Fine": 1.0 / 5.0,
    "Very Fine": 1.0 / 6.0,
}
DEFAULT_FINENESS = "Medium"

#: Total voxels the build grid may use, per build-quality preset.
#:
#: This node runs no FE solve, so the grid is a rasterization budget and
#: nothing else — it costs memory and triangles, not solver time. The numbers
#: mirror the recovery point caps in `surface_recovery` so the grid the lattice
#: is built on is the grid the surface is extracted from, with no second
#: supersampling stage in between.
BUILD_VOXEL_BUDGET: dict[str, int] = {
    "Standard": 1_000_000,
    "High": 2_000_000,
    "Professional": 3_500_000,
}
DEFAULT_BUILD_QUALITY = "High"

#: Wall/strut thickness, in build voxels, the grid is sized to deliver at the
#: requested relative density.
#:
#: This is the number that decides whether the part weighs what it was asked
#: to. The relative-density search measures the *voxel* field, while what ships
#: is the isosurface through it, and the two only agree once the wall is thick
#: enough for the contour to resolve. Measured on a 25% gyroid in a 100x60x40
#: block, holding everything else fixed:
#:
#:     wall 1.05 voxels -> voxel 23.1%, delivered mesh 11.0%   (2x light)
#:     wall 1.24 voxels -> voxel 22.2%, delivered mesh 22.2%   (exact)
#:     wall 1.48 voxels -> voxel 24.5%, delivered mesh 24.5%   (exact)
#:
#: It is a cliff, not a gradient, and on the wrong side of it every reported
#: mass is wrong by a factor of two while the geometry still looks like a
#: lattice. 1.5 sits clear of the crossing with room for the search to move the
#: wall in both directions.
MINIMUM_BUILD_WALL_VOXELS = 1.5

#: Measured wall-to-pitch constant `k` in `wall = k * pitch * relative_density`,
#: which inverts to the pitch a target density needs. Calibrated by building
#: each family at 15/25/40% on a 24-voxel pitch and reading back the thickness
#: the density search settled on: gyroid 0.254-0.266, Schwarz primitive
#: 0.335-0.345, honeycomb 0.391-0.420, octet 0.437-0.573, BCC 0.625-0.938. The
#: values below take the low end of each class, because under-estimating `k`
#: refines the grid further than needed while over-estimating it delivers a
#: wall the raster cannot hold.
WALL_PITCH_CONSTANT_STRUT = 0.44
WALL_PITCH_CONSTANT_SURFACE = 0.25


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

    Two things follow from being a modeling operation rather than a study, and
    both are why this node does not simply inherit the topology node's inputs:

    *No material.* Nothing here reads a stiffness, a density or an allowable —
    the output is geometry. Material belongs to whatever *analyses* the result
    downstream, which is a separate operation, exactly as it is in nTop and in
    Fusion's volumetric lattice. The port is removed rather than left connected
    to nothing.

    *No analysis grid.* The optimizing node's ``nelx/nely/nelz`` size an FE
    problem. Here there is no FE problem, and a grid chosen for one is the
    wrong constraint on a cell pitch: it capped the pitch at three voxels, so
    "Fine" and "Very Fine" both silently fell back to the same coarse cell and
    the field the results were measured on could not represent the cell at all.
    The grid is derived instead — from the requested pitch, the family's own
    resolution floor, and the wall thickness the requested density needs — and
    capped by a build-quality budget. See :meth:`resolve_build_grid`.
    """

    __identifier__ = "com.cad.sim.lattice_infill"
    NODE_NAME = "Lattice Infill"

    def __init__(self) -> None:
        super().__init__()
        # No boundary conditions and no material: this node does not solve
        # anything, so a load, support or material port would only invite a
        # connection that is silently ignored. `design_domain` is the body to
        # fill and is the only input.
        self.set_port_deletion_allowed(True)
        for port_name in ("supports", "loads", "material"):
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
        # Expert, so no guided defaults rewrite the settings below and no
        # guided lattice interpretation is applied to a uniform field.
        self.set_property("workflow_mode", "Expert")
        self.set_property("lattice_settings_mode", "Manual")

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
        # The pitch in model units, used when the mode is Manual. Deliberately
        # its own property rather than the optimizing node's
        # `lattice_cell_size_mm`: that one is denominated against an analysis
        # grid this node does not have, and sharing it made the inspector write
        # one key while the node read another, so a manual pitch did nothing.
        self.create_property("infill_cell_size_mm", 0.0, widget_type="float")
        # Solid boundary wall around the lattice, in model units. Zero leaves
        # the cells open at the trimmed boundary, which is what a powder
        # process needs; a closed skin seals the lattice and traps powder, so
        # it stays an explicit choice rather than a default.
        self.create_property("infill_skin_thickness_mm", 0.0, widget_type="float")
        self.create_property(
            "infill_build_quality",
            DEFAULT_BUILD_QUALITY,
            widget_type="combo",
            items=list(BUILD_VOXEL_BUDGET),
        )

    # ── settings ─────────────────────────────────────────────────────────────

    def resolve_infill_cell_size(self, characteristic_span: float) -> float:
        """Return the cell pitch this infill will actually build, in model units.

        Normally the requested pitch. When the build-quality budget cannot
        carry it — see :meth:`resolve_build_plan` — the pitch the plan settled
        on is returned instead, so the manufacturing options and the grid
        describe the same lattice rather than disagreeing by whatever the
        budget removed.
        """
        planned = getattr(self, "_resolved_infill_cell_size", None)
        if planned is not None:
            return float(planned)
        return self.requested_infill_cell_size(characteristic_span)

    def requested_infill_cell_size(self, characteristic_span: float) -> float:
        """Return the cell pitch the current settings ask for, in model units.

        ``characteristic_span`` is a representative size of the body being
        filled. Automatic fineness is stated against it, because a pitch that
        is sensible for a 300 mm bracket is meaningless on a 20 mm boss and the
        user has no way to convert between the two by eye.
        """
        manual = str(
            self.get_property("infill_cell_size_mode") or "Automatic"
        ).strip().lower() == "manual"
        if manual:
            requested = float(self.get_property("infill_cell_size_mm") or 0.0)
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
        span = float(characteristic_span)
        if not (span > 0.0):
            raise ValueError(
                "Lattice infill needs a 3-D body to measure its cell size "
                "against. Connect a CAD solid to design_domain."
            )
        return span * fraction

    def required_cell_voxels(self) -> float:
        """Build voxels per cell the current family and density need.

        Two independent floors. The family's own ``minimum_cell_voxels`` is
        where its openings stop being resolved and the periodic surface pinches
        into fragments. The second is the wall the requested relative density
        implies: the density search measures the voxel field, what ships is the
        isosurface through it, and below roughly 1.25 voxels of wall the
        contour cannot follow the raster and the delivered part comes back at
        about half the mass the search settled on. See
        :data:`MINIMUM_BUILD_WALL_VOXELS`.
        """
        family = family_for(self.get_property("structure_mode"))
        family_floor = float(family.minimum_cell_voxels) if family is not None else 8.0
        constant = (
            WALL_PITCH_CONSTANT_STRUT
            if family is not None and family.is_strut
            else WALL_PITCH_CONSTANT_SURFACE
        )
        try:
            density = float(
                self.get_property("lattice_target_relative_density") or 0.0
            )
        except (TypeError, ValueError):
            density = 0.0
        if not (density > 0.0):
            return family_floor
        wall_floor = MINIMUM_BUILD_WALL_VOXELS / (constant * density)
        return max(family_floor, wall_floor)

    def resolve_build_plan(
        self,
        span: tuple[float, float, float],
        requested_cell_size: float,
    ) -> tuple[float, tuple[int, int, int], bool]:
        """Choose the cell pitch and the voxel grid together.

        Returns ``(cell_size, grid, coarsened)``.

        The grid follows the cell: enough voxels per cell for the family to
        stay connected and for the requested density's wall to be
        representable, then as many cells as the body holds.

        When that exceeds the build-quality budget, something has to give, and
        which one is not a free choice. Holding the pitch and shrinking the
        voxel thins the wall below what the isosurface can resolve, and the
        part then ships at roughly half the requested relative density while
        still looking like a correct lattice — a silent 2x mass error (see
        :data:`MINIMUM_BUILD_WALL_VOXELS`). Holding the wall and growing the
        pitch delivers the requested density in fewer, larger cells, which is
        wrong in a way the user can see and that this reports. Mass is the
        number a lattice exists to control, so the pitch is what gives.
        """
        extents = [float(value) for value in span[:3]]
        pitch = float(requested_cell_size)
        if pitch <= 0.0 or not all(value > 0.0 for value in extents):
            raise ValueError(
                "Lattice infill needs a 3-D body with a positive cell pitch."
            )
        voxels_per_cell = max(self.required_cell_voxels(), 1.0)
        voxel = pitch / voxels_per_cell
        counts = [max(4, int(math.ceil(value / voxel))) for value in extents]

        quality = str(
            self.get_property("infill_build_quality") or DEFAULT_BUILD_QUALITY
        ).strip()
        budget = BUILD_VOXEL_BUDGET.get(
            quality, BUILD_VOXEL_BUDGET[DEFAULT_BUILD_QUALITY]
        )
        total = counts[0] * counts[1] * counts[2]
        if total <= budget:
            return pitch, (counts[0], counts[1], counts[2]), False

        # Spend the whole budget on a uniform, cubic voxel, then let the pitch
        # be whatever that voxel can carry the wall at.
        budget_voxel = (
            (extents[0] * extents[1] * extents[2]) / float(budget)
        ) ** (1.0 / 3.0)
        counts = [
            max(4, int(round(value / budget_voxel))) for value in extents
        ]
        coarsened_pitch = voxels_per_cell * budget_voxel
        logger.warning(
            "Lattice infill needs a %dx%dx%d grid to build a %.3g cell at "
            "%.0f%% relative density, which is past the %s budget of %d "
            "voxels. Building a %.3g cell on %dx%dx%d instead, so the "
            "requested density is still delivered. Raise Build Quality for a "
            "finer cell.",
            *[int(math.ceil(value / voxel)) for value in extents],
            pitch,
            100.0
            * float(self.get_property("lattice_target_relative_density") or 0.0),
            quality,
            budget,
            coarsened_pitch,
            *counts,
        )
        return coarsened_pitch, (counts[0], counts[1], counts[2]), True

    def apply_guided_defaults(self) -> dict[str, Any]:
        """No guided solver defaults: there is no solve to configure."""
        return {}

    # ── execution ────────────────────────────────────────────────────────────

    def _apply_build_grid(self) -> None:
        """Derive and store the cell pitch and build grid from the body."""
        import numpy as np

        from .geometry_mapping import _mesh_bounds
        from .lattice_settings import _characteristic_bounds_span

        # Cleared first: a stale plan from the previous run would otherwise be
        # returned by `resolve_infill_cell_size` for a different body or a
        # different density.
        self._resolved_infill_cell_size = None
        self._infill_pitch_coarsened = False
        bounds = _mesh_bounds(self.get_input_value("design_domain", None))
        if bounds is None:
            # Leave whatever grid is stored; `prepare_topology_study` reports
            # the missing or unusable design domain with its own message.
            return
        lower = np.asarray(bounds[0], dtype=float)[:3]
        upper = np.asarray(bounds[1], dtype=float)[:3]
        span = tuple(float(value) for value in np.maximum(upper - lower, 0.0))
        quality_preset = str(
            self.get_property("infill_build_quality") or DEFAULT_BUILD_QUALITY
        ).strip()
        if quality_preset not in BUILD_VOXEL_BUDGET:
            quality_preset = DEFAULT_BUILD_QUALITY
        requested = self.requested_infill_cell_size(
            _characteristic_bounds_span(bounds)
        )
        cell_size, (nelx, nely, nelz), coarsened = self.resolve_build_plan(
            span, requested
        )
        self._resolved_infill_cell_size = float(cell_size)
        self._infill_pitch_coarsened = bool(coarsened)
        self._requested_infill_cell_size = float(requested)
        self.set_property("nelx", int(nelx))
        self.set_property("nely", int(nely))
        self.set_property("nelz", int(nelz))
        # Surface recovery supersamples this grid again, under its own point
        # cap, and that cap is what the delivered mesh is actually built on.
        # Left independent, a Standard build was contoured at the Professional
        # cap and came back 2.1M triangles and a minute long for a three-cell
        # lattice — the control the user set governed half the chain. One
        # setting, both stages.
        self.set_property("surface_quality", quality_preset)
        logger.info(
            "Lattice infill build grid: %dx%dx%d voxels for a %.3g pitch "
            "(%.1f voxels per cell).",
            nelx,
            nely,
            nelz,
            cell_size,
            cell_size / (span[0] / nelx) if span[0] > 0.0 else 0.0,
        )

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

        # Building an explicit lattice is minutes of rasterizing, contouring
        # and mesh cleanup with no iteration to report against, so a silent
        # progress bar is indistinguishable from a hang. Name the stage.
        def stage(text: str) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback({"status_only": True, "stage": text}, None, 0, 0)
            except Exception:
                logger.debug("Lattice infill progress reporting failed.")

        stage("Sizing the build grid")
        try:
            self._apply_build_grid()
        except ValueError as exc:
            self.set_error(str(exc))
            return None

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

        stage("Building the lattice and recovering its surface")
        recovery_started = time.perf_counter()
        output_context = build_topology_output(self, result, study)
        if output_context is None:
            return None
        recovery_time_s = time.perf_counter() - recovery_started
        stage("Checking the manufactured geometry")
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
