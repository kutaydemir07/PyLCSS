# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""NodeGraphQt orchestration for lattice infill without optimization."""

from __future__ import annotations

import logging
import math
from typing import Any

from pylcss.input_values import as_bool
from ..manufacturing.lattice_families import family_for
from .lattice_definition import LatticeDefinitionMixin
from .geometry_node import GeometryStudyNode

logger = logging.getLogger(__name__)

# Requested pitch as a fraction of the body's characteristic span. The actual
# integer grid and any pitch adjustment are resolved against the build budget.
AUTOMATIC_FINENESS_FRACTIONS: dict[str, float] = {
    "Coarse": 1.0 / 3.0,
    "Medium": 1.0 / 4.0,
    "Fine": 1.0 / 5.0,
    "Very Fine": 1.0 / 6.0,
}
DEFAULT_FINENESS = "Medium"

# Rasterization budgets; these are application resource presets.
BUILD_VOXEL_BUDGET: dict[str, int] = {
    "Standard": 1_000_000,
    "High": 2_000_000,
    "Professional": 3_500_000,
}
DEFAULT_BUILD_QUALITY = "High"

# Empirical raster-resolution allowances, checked against the delivered mesh.
# They are not printer capabilities or constants established by the AMR paper.
MINIMUM_BUILD_WALL_VOXELS = 1.5

# Conservative wall-to-pitch estimates for the initial build-resolution plan.
WALL_PITCH_CONSTANT_STRUT = 0.44
WALL_PITCH_CONSTANT_SURFACE = 0.25


class LatticeInfillNode(LatticeDefinitionMixin, GeometryStudyNode):
    """Generate uniform lattice geometry inside an existing body.

    This node has no material, support, load, or optimizer controls. Recovery
    sizes walls/struts toward the relative-density target and reports the
    achieved density and geometry checks. It performs no mechanical solve.
    """

    __identifier__ = "com.cad.sim.lattice_infill"
    NODE_NAME = "Lattice Infill"

    def __init__(self) -> None:
        super().__init__()
        self.set_property("density_cutoff", 0.5)
        self.set_property("visualization", "CAD")
        self.set_property("cad_export_filename", "lattice_infill.stl")
        self.set_property("cad_export_format", "STL")
        for axis in ("nelx", "nely", "nelz"):
            self.create_property(axis, 4, widget_type="int")
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
        self.create_property("infill_cell_size_mm", 10.0, widget_type="float")
        self.create_property("infill_allow_pitch_coarsening", False, widget_type="bool")
        self.create_property(
            "infill_skin_mode",
            "None",
            widget_type="combo",
            items=["None", "Automatic", "Manual"],
        )
        self.create_property("infill_skin_thickness_mm", 0.0, widget_type="float")
        self.create_property(
            "infill_build_quality",
            DEFAULT_BUILD_QUALITY,
            widget_type="combo",
            items=list(BUILD_VOXEL_BUDGET),
        )

    # â”€â”€ settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        manual = (
            str(self.get_property("infill_cell_size_mode") or "Automatic")
            .strip()
            .lower()
            == "manual"
        )
        if manual:
            requested = float(self.get_property("infill_cell_size_mm") or 0.0)
            if math.isfinite(requested) and requested > 0.0:
                return requested
            raise ValueError("Manual infill needs a finite positive cell size.")
        fraction = AUTOMATIC_FINENESS_FRACTIONS.get(
            str(self.get_property("infill_fineness") or DEFAULT_FINENESS).strip(),
            AUTOMATIC_FINENESS_FRACTIONS[DEFAULT_FINENESS],
        )
        span = float(characteristic_span)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError(
                "Lattice infill needs a 3-D body to measure its cell size "
                "against. Connect a CAD solid to design_domain."
            )
        return span * fraction

    def required_cell_voxels(self) -> float:
        """Build voxels per cell the current family and density need.

        Use the larger of the family's opening-resolution floor and an
        empirical wall-resolution allowance at the requested density. These
        planning estimates do not guarantee delivered wall thickness or mass;
        the recovered geometry reports its measured density and connectivity.
        """
        family = family_for(self.get_property("structure_mode"))
        if family is None:
            raise ValueError("Choose a supported lattice family.")
        family_floor = float(family.minimum_cell_voxels)
        constant = (
            WALL_PITCH_CONSTANT_STRUT
            if family is not None and family.is_strut
            else WALL_PITCH_CONSTANT_SURFACE
        )
        try:
            density = float(self.get_property("lattice_target_relative_density") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Infill relative density must be numeric.") from exc
        if not math.isfinite(density) or not 0.0 < density <= 1.0:
            raise ValueError(
                "Infill relative density must be greater than 0 and at most 1."
            )
        wall_floor = MINIMUM_BUILD_WALL_VOXELS / (constant * density)
        return max(family_floor, wall_floor)

    def resolve_build_plan(
        self,
        span: tuple[float, float, float],
        requested_cell_size: float,
    ) -> tuple[float, tuple[int, int, int], bool]:
        """Return (pitch, grid, coarsened), with integer counts within budget.

        Automatic pitch may grow to preserve wall resolution. Manual pitch
        grows only when Allow Larger Cells is explicitly enabled.
        """
        extents = tuple(float(value) for value in span)
        pitch = float(requested_cell_size)
        if (
            len(extents) != 3
            or not math.isfinite(pitch)
            or pitch <= 0.0
            or any(not math.isfinite(value) or value <= 0.0 for value in extents)
        ):
            raise ValueError(
                "Lattice infill needs finite positive 3-D bounds and cell pitch."
            )
        voxels_per_cell = self.required_cell_voxels()
        voxel = pitch / voxels_per_cell

        def grid(edge):
            return tuple(max(4, math.ceil(value / edge)) for value in extents)

        quality = str(
            self.get_property("infill_build_quality") or DEFAULT_BUILD_QUALITY
        )
        if quality not in BUILD_VOXEL_BUDGET:
            raise ValueError("Choose Standard, High, or Professional Build Quality.")
        budget = BUILD_VOXEL_BUDGET[quality]
        counts = grid(voxel)
        if math.prod(counts) <= budget:
            return pitch, counts, False
        manual = (
            str(self.get_property("infill_cell_size_mode") or "Automatic") == "Manual"
        )
        if manual and not as_bool(self.get_property("infill_allow_pitch_coarsening")):
            raise ValueError(
                f"The requested {pitch:g} cell needs {math.prod(counts):,} build voxels; "
                f"{quality} allows {budget:,}. Raise Build Quality or enable Allow Larger Cells."
            )
        lower, upper = voxel, max(extents)
        for _ in range(60):
            candidate = (lower + upper) / 2.0
            if math.prod(grid(candidate)) > budget:
                lower = candidate
            else:
                upper = candidate
        counts = grid(upper)
        if math.prod(counts) > budget:
            raise ValueError(
                "The selected build quality cannot accommodate the minimum grid."
            )
        coarsened_pitch = voxels_per_cell * upper
        logger.warning(
            "Infill pitch increased from %.4g to %.4g to fit %s (%d voxels).",
            pitch,
            coarsened_pitch,
            quality,
            math.prod(counts),
        )
        return coarsened_pitch, counts, True

    def apply_guided_defaults(self) -> dict[str, Any]:
        """No guided solver defaults: there is no solve to configure."""
        return {}

    # â”€â”€ execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        requested = self.requested_infill_cell_size(_characteristic_bounds_span(bounds))
        cell_size, (nelx, nely, nelz), coarsened = self.resolve_build_plan(
            span, requested
        )
        self._resolved_infill_cell_size = float(cell_size)
        self._resolved_infill_voxel_size = (
            float(cell_size) / self.required_cell_voxels()
        )
        self._infill_pitch_coarsened = bool(coarsened)
        self._requested_infill_cell_size = float(requested)
        self.set_property("nelx", int(nelx))
        self.set_property("nely", int(nely))
        self.set_property("nelz", int(nelz))
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
        self.clear_error()
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
        output = finalize_topology_output(self, output_context)
        output["timing"] = {
            "optimization_s": 0.0,
            "recovery_s": float(recovery_time_s),
            "validation_and_cad_s": float(time.perf_counter() - finalization_started),
        }
        return output
