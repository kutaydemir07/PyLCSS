# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Engineering-intent presets mapped to low-level voxel-optimiser controls."""
from __future__ import annotations

from typing import Any, Dict, Tuple

INDUSTRIAL_WORKFLOW_MODES = ('Guided',)
INDUSTRIAL_DESIGN_GOALS = (
    'Lightweight Stiffness',
    'Minimum Mass Under Stress',
)
INDUSTRIAL_MANUFACTURING_PROCESSES = (
    'None',
    'Additive',
    'Extruded',
    'Symmetric',
    'Additive + Symmetric',
)


def _choice(value: Any, choices: Tuple[str, ...], default: str) -> str:
    text = str(value or '').strip().lower().replace('_', ' ').replace('-', ' ')
    for choice in choices:
        normalized = choice.lower().replace('_', ' ').replace('-', ' ')
        if text == normalized:
            return choice
    return default


def _scaled_grid(nelx: Any, nely: Any, nelz: Any, target_max: int) -> Tuple[int, int, int]:
    try:
        dims = [max(1, int(round(float(v)))) for v in (nelx, nely, nelz)]
    except Exception:
        dims = [30, 20, 10]
    current_max = max(dims) or 1
    scale = float(target_max) / float(current_max)
    scaled = [max(1, int(round(v * scale))) for v in dims]
    scaled[0] = max(4, scaled[0])
    scaled[1] = max(2, scaled[1])
    scaled[2] = max(1, scaled[2])
    return int(scaled[0]), int(scaled[1]), int(scaled[2])


def industrial_topopt_defaults(
    design_goal: Any = 'Lightweight Stiffness',
    quality_preset: Any = 'Balanced',
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
    # Kept as an argument for older saved graphs/API calls, but guided TopOpt
    # now has one automatic quality policy instead of user-facing presets.
    _ = quality_preset
    manufacturing = _choice(manufacturing_process, INDUSTRIAL_MANUFACTURING_PROCESSES, 'None')

    target_max = 100
    out_nelx, out_nely, out_nelz = _scaled_grid(nelx, nely, nelz, target_max)
    max_dim = max(out_nelx, out_nely, out_nelz)
    filter_ratio = 0.030
    quality_settings = dict(max_iter=100, tol=0.0050, density_cutoff=0.45, mesh_decimate_ratio=1.00)

    settings: Dict[str, Any] = {
        'nelx': out_nelx,
        'nely': out_nely,
        'nelz': out_nelz,
        'rmin': round(max(1.2, min(5.0, max_dim * filter_ratio)), 2),
        'penal': 3.0,
        'optimizer': 'OC',
        'stress_constraint': False,
        'symmetry': 'None',
        'extrusion': 'None',
        'overhang_build_axis': 'None',
        'max_member_size_voxels': 0.0,
        'pattern_repeat': 1,
        'pattern_axis': 'Y',
        'print_ready_mesh': False,
        **quality_settings,
    }

    if goal == 'Minimum Mass Under Stress':
        settings.update(
            optimizer='MMA',
            stress_constraint=True,
            max_iter=max(int(settings['max_iter']), 120),
            tol=min(float(settings['tol']), 0.005),
        )

    if manufacturing in ('Additive', 'Additive + Symmetric'):
        settings.update(
            overhang_build_axis='+Y',
            density_cutoff=max(float(settings['density_cutoff']), 0.45),
            print_ready_mesh=True,
        )
    if manufacturing == 'Extruded':
        settings.update(extrusion='Z', print_ready_mesh=True)
    if manufacturing in ('Symmetric', 'Additive + Symmetric'):
        settings.update(symmetry='Z')

    return settings


