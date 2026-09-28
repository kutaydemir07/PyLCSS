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
LATTICE_DESIGN_GOALS = INDUSTRIAL_DESIGN_GOALS
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
    'None',
    'Additive',
    'Cast / Moulded',
    'Extruded',
    'Symmetric',
    'Additive + Symmetric',
    'Extruded + Symmetric',
)
# Only a swept profile has an exact editable B-rep, so these are the presets
# that hand a study a CAD body. Every other process is still a valid study;
# its result is delivered as the recovered surface.
EXTRUDED_MANUFACTURING_PROCESSES = ('Extruded', 'Extruded + Symmetric')
SYMMETRIC_MANUFACTURING_PROCESSES = (
    'Symmetric',
    'Additive + Symmetric',
    'Extruded + Symmetric',
)
# Guided discretization ceiling. Not a fidelity choice: the grid is sized from
# the study's physical feature lengths, and these two only stop a large model
# from being refined past what is solvable in a sitting.
#
# The cell budget tracks `_guided_cell_budget`, which is what actually sizes a
# guided solve. This copy seeds the node's displayed element counts, and those
# become the real grid the moment a study is switched to Expert -- so a
# divergence here would hand Expert a silently different study from the one
# Guided just ran.
_GUIDED_TARGET_MAX_CELLS_PER_AXIS = 220
_GUIDED_TOTAL_CELL_BUDGET = 40_000


def _choice(value: Any, choices: Tuple[str, ...], default: str) -> str:
    text = str(value or '').strip().lower().replace('_', ' ').replace('-', ' ')
    if text == 'thermo mechanical':
        text = 'coupled structural + thermal'
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
    manufacturing_process: Any = 'None',
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
    manufacturing = _choice(manufacturing_process, INDUSTRIAL_MANUFACTURING_PROCESSES, 'None')

    # There is no fidelity dropdown. What used to be "Analysis Resolution"
    # bundled six settings across three unrelated concerns -- discretization,
    # convergence and post-processing -- behind one marketing word, and the
    # concern it appeared to own was not even its own: the guided grid is
    # derived from the physical feature lengths (minimum member and void size,
    # keep-out cylinder diameters), which are manufacturing capabilities rather
    # than numerical preferences. What is left is a cost ceiling, and that is
    # computed from the study itself -- load-case count, physics, joints.
    # Convergence and output quality are simply held at the best setting: a
    # solver should converge, not be told to stop early.
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
        mesh_decimate_ratio=1.00,
        surface_quality='Professional',
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
        'symmetry': 'None',
        'extrusion': 'None',
        'overhang_build_axis': 'None',
        'pull_out_axis': 'None',
        'max_member_size_voxels': 0.0,
        'maximum_member_size_mm': 0.0,
        'pattern_repeat': 1,
        'pattern_axis': 'Y',
        'print_ready_mesh': True,
        # The density field is the result an optimization engineer needs to
        # judge first.  CAD remains available as a result view, but a plain
        # shaded solid hides intermediate density and the retained/removed
        # material decision.
        'visualization': 'Density',
        'surface_recovery_method': 'Volume-Preserving SDF (VTK)',
        'exclusion_scope': 'All Loads and Supports',
        'exclusion_thickness_mode': 'Program Controlled',
        'exclusion_thickness_mm': 2.0,
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
    elif goal == 'Thermal Conduction':
        settings.update(
            physics_mode='Thermal',
            optimizer='Auto',
        )
    elif goal == 'Coupled Structural + Thermal':
        settings.update(
            physics_mode='Coupled Structural + Thermal',
            optimizer='MMA',
            max_iter=max(int(settings['max_iter']), 120),
        )
    if manufacturing in ('Additive', 'Additive + Symmetric'):
        # No cutoff bump: raising the recovery iso value to shed thin features
        # is a post-hoc trim of a design the optimizer already committed to, and
        # it silently discards stiffness the compliance was measured with. The
        # printable length scale belongs to `minimum_member_size_mm`, which acts
        # during the solve, so the recovered shape stays the analyzed one.
        settings.update(
            overhang_build_axis='+Y',
            print_ready_mesh=True,
        )
    if manufacturing == 'Cast / Moulded':
        # A one-sided tool pulled in +Z. Undercut removal is what makes an
        # optimized rib pattern on a housing actually demouldable.
        settings.update(pull_out_axis='+Z', print_ready_mesh=True)
    if manufacturing in EXTRUDED_MANUFACTURING_PROCESSES:
        settings.update(extrusion='Z', print_ready_mesh=True)
    if manufacturing in SYMMETRIC_MANUFACTURING_PROCESSES:
        # Mirroring across the extrusion axis is vacuous: the density is
        # averaged along that axis anyway, so the mirror has nothing to do.
        # An extruded symmetric part has to mirror inside its profile.
        settings.update(
            symmetry='X'
            if manufacturing in EXTRUDED_MANUFACTURING_PROCESSES
            else 'Z'
        )

    return settings


# The cell family a lattice study falls back to when none was saved. Gyroid is
# the general-purpose default: self-supporting in every build orientation, with
# no unsupported horizontal members and no stress-concentrating strut joints.
DEFAULT_LATTICE_STRUCTURE_MODE = 'Gyroid Lattice'


def industrial_lattice_defaults(
    design_goal: Any = 'Lightweight Stiffness',
    manufacturing_process: Any = 'None',
    *,
    nelx: Any = 30,
    nely: Any = 20,
    nelz: Any = 10,
    structure_mode: Any = DEFAULT_LATTICE_STRUCTURE_MODE,
) -> Dict[str, Any]:
    """Map engineering intent to the lattice optimizer controls.

    A lattice study shares the topology study's discretization, physics and
    manufacturing-constraint policy, so this extends that mapping rather than
    restating it. What it adds is the manufacturing definition of the cell.
    The selected family is intent and is never overwritten by a preset; only
    an empty or unrecognised value falls back to the default.
    """
    settings = industrial_topopt_defaults(
        design_goal,
        manufacturing_process,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
    )
    settings.update(
        {
            'structure_mode': (
                str(structure_mode).strip()
                or DEFAULT_LATTICE_STRUCTURE_MODE
            )
            if structure_mode
            else DEFAULT_LATTICE_STRUCTURE_MODE,
            'lattice_settings_mode': 'Guided',
            'structure_cell_size_voxels': 8.0,
            'structure_member_thickness_voxels': 1.0,
            'structure_skin_thickness_voxels': 0.75,
            'lattice_porosity': 'Conservative',
            # The manufactured cells are the result, and unlike a solid
            # envelope they are not replaced by a reconstructed CAD body a
            # moment later.
            'visualization': 'CAD',
        }
    )
    return settings


