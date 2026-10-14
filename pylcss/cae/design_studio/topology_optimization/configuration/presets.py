# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Engineering-intent presets mapped to low-level voxel-optimizer controls."""
from __future__ import annotations

from math import prod
from typing import Any, Dict, Tuple

INDUSTRIAL_WORKFLOW_MODES = ('Guided', 'Expert')
INDUSTRIAL_DESIGN_GOALS = (
    'Lightweight Stiffness',
    'Minimum Mass Under Stress',
)
# Retired goals. Each existed only to flip a setting that is now reachable
# directly: both forced worst-case load aggregation, and the multibody variant
# additionally switched on joint validation, which is now driven by whether the
# graph actually contains joints. Saved projects keep their behaviour by
# migrating the setting instead of the name.
LEGACY_DESIGN_GOALS = {
    'load-case envelope': 'Lightweight Stiffness',
    'multibody load envelope': 'Lightweight Stiffness',
}
RETIRED_DESIGN_GOALS = {
    'thermal conduction': 'Lightweight Stiffness',
    'coupled structural + thermal': 'Lightweight Stiffness',
}


def migrate_design_goal(value: Any) -> Tuple[str, str | None]:
    """Map a retired goal onto its replacement plus an aggregation override."""
    key = str(value or '').strip().lower()
    replacement = LEGACY_DESIGN_GOALS.get(key)
    if replacement is not None:
        return replacement, 'Worst Case'
    replacement = RETIRED_DESIGN_GOALS.get(key)
    if replacement is None:
        return str(value or 'Lightweight Stiffness'), None
    return replacement, None
INDUSTRIAL_MANUFACTURING_PROCESSES = (
    'Normal',
)
EXTRUDED_MANUFACTURING_PROCESSES: Tuple[str, ...] = ()
SYMMETRIC_MANUFACTURING_PROCESSES: Tuple[str, ...] = ()
# Default Fine grid budget used to seed legacy/manual element counts.
# Runtime automatic meshing uses geometric resolution in `voxel_meshing`;
# its configurable resource safeguard is independent of the chosen pitch.
_GUIDED_TARGET_MAX_CELLS_PER_AXIS = 220
_GUIDED_TOTAL_CELL_BUDGET = 40_000


def _choice(value: Any, choices: Tuple[str, ...], default: str) -> str:
    text = str(value or '').strip().lower().replace('_', ' ').replace('-', ' ')
    if text in ('none', 'unrestricted', 'standard', 'normal', ''):
        return 'Normal' if 'Normal' in choices else default
    for choice in choices:
        normalized = choice.lower().replace('_', ' ').replace('-', ' ')
        if text == normalized:
            return choice
    return default


# Normal production grids are solved with a recursive geometric-multigrid CG,
# and each level of that hierarchy halves every axis. The number of levels a grid
# supports is therefore set by how many factors of two its *smallest* axis has:
# a 12-element axis stops the hierarchy after two levels, a 16-element one
# carries it to four. Since the coarse-level direct solve is what dominates a
# shallow hierarchy, aligning preset grids to a multiple of eight buys several
# times the solve speed for a few percent more cells.
_MULTIGRID_GRID_ALIGNMENT = 8

# Program controlled minimum member size, in element lengths: 3x the mesh
# element size, the established density-method default. A density filter of
# radius R holds members about 2R across, so the radius is half of it.
# Kept here as a literal rather than imported from
# `..integration.voxelization`, which is the runtime consumer of this module —
# these presets must stay importable without pulling in numba and the solver.
PROGRAM_CONTROLLED_MEMBER_ELEMENTS = 3.0
PROGRAM_CONTROLLED_RMIN = PROGRAM_CONTROLLED_MEMBER_ELEMENTS / 2.0


def _align_axis(value: int, minimum: int) -> int:
    """Round one axis onto the multigrid alignment, or make it even below it."""
    if value < _MULTIGRID_GRID_ALIGNMENT:
        # A genuinely thin axis (a plate or a 2-D-like study) must not be
        # inflated to 8 cells just to gain a level; that would cost more than
        # the deeper hierarchy saves. Keep it even so at least one level works.
        return max(minimum, value + value % 2 if value > 1 else value)
    aligned = int(round(value / _MULTIGRID_GRID_ALIGNMENT)) * _MULTIGRID_GRID_ALIGNMENT
    return max(_MULTIGRID_GRID_ALIGNMENT, aligned)


def _scaled_grid(
    nelx: Any,
    nely: Any,
    nelz: Any,
    target_max: int,
    max_cells: int,
) -> Tuple[int, int, int]:
    try:
        dims = [max(1, int(round(float(v)))) for v in (nelx, nely, nelz)]
    except Exception:
        dims = [30, 20, 10]
    current_max = max(dims) or 1
    scale = float(target_max) / float(current_max)
    scaled = [max(1, int(round(v * scale))) for v in dims]
    minima = (4, 2, 1)
    scaled = [max(minima[i], scaled[i]) for i in range(3)]
    # Fit the cell budget by scaling all three axes together. Shedding cells
    # from only the longest axis, one step at a time, drives every axis toward
    # equality: a 60x12x40 upright resolved to 48x48x48 -- a cube holding none
    # of the part's proportions, with voxels four times longer on one axis than
    # another, which is what terraced the recovered surface.
    total = int(prod(scaled))
    if total > int(max_cells):
        budget_scale = (float(max_cells) / float(total)) ** (1.0 / 3.0)
        scaled = [
            max(minima[i], int(round(scaled[i] * budget_scale))) for i in range(3)
        ]
    scaled = [_align_axis(scaled[i], minima[i]) for i in range(3)]
    # Alignment rounds to the nearest step, so it can re-cross the budget by a
    # few percent. Step by the alignment here too, so the correction cannot
    # undo it.
    while int(prod(scaled)) > int(max_cells):
        axis = max(range(3), key=lambda index: scaled[index])
        step = (
            _MULTIGRID_GRID_ALIGNMENT
            if scaled[axis] > _MULTIGRID_GRID_ALIGNMENT
            else 2
        )
        if scaled[axis] - step < max(minima[axis], 2):
            break
        scaled[axis] = scaled[axis] - step
    return int(scaled[0]), int(scaled[1]), int(scaled[2])


def industrial_topopt_defaults(
    design_goal: Any = 'Lightweight Stiffness',
    manufacturing_process: Any = 'Normal',
    *,
    nelx: Any = 30,
    nely: Any = 20,
    nelz: Any = 10,
) -> Dict[str, Any]:
    """Map engineering intent to the low-level topology optimizer controls.

    Industrial tools expose a short study definition and derive numerical
    controls from it. This helper keeps that policy testable and reusable by
    the Qt property panel without tying the solver backend to Qt.
    """
    goal = _choice(design_goal, INDUSTRIAL_DESIGN_GOALS, 'Lightweight Stiffness')

    # Mesh resolution is selected independently in the voxel meshing controls.
    # These counts seed older/manual studies; the runtime sizes automatic
    # grids using the selected mesh budget and actual geometry. A goal preset
    # does not change voxel size, mesh quality, or optional refinement.
    out_nelx, out_nely, out_nelz = _scaled_grid(
        nelx,
        nely,
        nelz,
        _GUIDED_TARGET_MAX_CELLS_PER_AXIS,
        _GUIDED_TOTAL_CELL_BUDGET,
    )
    # The geometry-extraction threshold is a convention, not a quality knob:
    # 0.3 is the established iso value for density-method shape recovery, so a
    # study exported from PyLCSS encloses the same material as the same study
    # run elsewhere. It is unrelated to the lower density bound the solver uses
    # internally to keep the stiffness matrix non-singular (`Emin`/rho_min,
    # typically 1e-3) — that bound never reaches recovery.
    RECOVERY_ISO_VALUE = 0.30
    quality_settings = dict(
        max_iter=120,
        tol=0.0035,
        density_cutoff=RECOVERY_ISO_VALUE,
    )

    settings: Dict[str, Any] = {
        'nelx': out_nelx,
        'nely': out_nely,
        'nelz': out_nelz,
        'rmin': PROGRAM_CONTROLLED_RMIN,
        'penal': 3.0,
        'formulation': 'Density (SIMP)',
        'optimizer': 'Auto',
        'physics_mode': 'Structural',
        # `load_aggregation` is deliberately absent: it is a user decision about
        # the load envelope, not something a fidelity or process preset owns.
        'stress_constraint': False,
        # The density field is the result an optimization engineer needs to
        # judge first.  CAD remains available as a result view, but a plain
        # shaded solid hides intermediate density and the retained/removed
        # material decision.
        'visualization': 'CAD',
        **quality_settings,
    }

    if goal == 'Minimum Mass Under Stress':
        settings.update(
            optimizer='GCMMA',
            stress_constraint=True,
            yield_stress=250.0,
            max_iter=max(int(settings['max_iter']), 120),
            tol=min(float(settings['tol']), 0.005),
        )
    return settings


