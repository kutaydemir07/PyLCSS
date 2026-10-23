# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Locate internal spline lines that permit or exhibit tangent discontinuity."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def inspect_internal_spline_continuity(surface):
    """Inspect unit-weight tensor B-splines, without changing their geometry.

    A degree-p knot of multiplicity m guarantees only C^(p-m). Geometry may
    be smoother than this representation guarantees, so actual one-sided
    tangent-plane angles are sampled separately on every opposing knot span.
    These measurements are not a global G1 or curvature-continuity proof.
    """
    weights = getattr(surface, 'weights', None)
    if weights is not None and not np.all(np.asarray(weights) == 1.):
        raise ValueError('Internal continuity inspection requires a unit-weight B-spline')
    controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    axes = [(np.asarray(surface.knotvector_u), surface.degree_u),
            (np.asarray(surface.knotvector_v), surface.degree_v)]
    basis = [BSpline(knots, np.eye(controls.shape[axis]), degree)
             for axis, (knots, degree) in enumerate(axes)]
    lines = []
    for axis, (knots, degree) in enumerate(axes):
        values, multiplicities = np.unique(knots, return_counts=True)
        other = np.unique(axes[1 - axis][0])
        sites = .5 * (other[:-1] + other[1:])
        for index in np.flatnonzero((multiplicities >= degree) & (values > 0) & (values < 1)):
            knot = values[index]
            offset = min(knot - values[index - 1], values[index + 1] - knot) * 1e-7
            normals, positions, undefined = [], [], 0
            for parameter in (knot - offset, knot + offset):
                uv = np.column_stack((np.full(len(sites), parameter), sites))
                if axis == 1:
                    uv = uv[:, ::-1]
                u, v = basis[0](uv[:, 0]), basis[1](uv[:, 1])
                du = np.einsum('ni,nj,ijc->nc', basis[0](uv[:, 0], nu=1), v, controls)
                dv = np.einsum('ni,nj,ijc->nc', u, basis[1](uv[:, 1], nu=1), controls)
                cross = np.cross(du, dv)
                lengths = np.linalg.norm(cross, axis=1)
                undefined += int(np.count_nonzero(lengths <= 1e-14))
                normals.append(np.divide(cross, lengths[:, None],
                    out=np.full_like(cross, np.nan), where=lengths[:, None] > 1e-14))
                positions.append(np.einsum('ni,nj,ijc->nc', u, v, controls))
            dot = np.abs(np.einsum('ij,ij->i', *normals))
            angles = np.rad2deg(np.arccos(np.clip(dot[np.isfinite(dot)], 0, 1)))
            lines.append({'axis': 'uv'[axis], 'knot_parameter': float(knot),
                          'degree': int(degree), 'multiplicity': int(multiplicities[index]),
                          'guaranteed_parametric_continuity_order': int(degree - multiplicities[index]),
                          'sample_count': len(sites), 'undefined_one_sided_normal_count': undefined,
                          'maximum_sampled_tangent_plane_jump_degrees': float(angles.max()) if len(angles) else None,
                          'maximum_probe_position_gap_model_units': float(np.linalg.norm(positions[0] - positions[1], axis=1).max())})
    angles = [row['maximum_sampled_tangent_plane_jump_degrees'] for row in lines
              if row['maximum_sampled_tangent_plane_jump_degrees'] is not None]
    return {'method': 'Knot-multiplicity analysis and one-sided analytic surface derivatives',
            'geometry_changed': False, 'global_G1_continuity_certified': False,
            'angle_definition': 'Acute angle between unoriented tangent planes',
            'C0_knot_line_count': len(lines),
            'lines_with_sampled_jump_over_1_degree': sum(angle > 1 for angle in angles),
            'lines_with_sampled_jump_over_5_degrees': sum(angle > 5 for angle in angles),
            'maximum_sampled_tangent_plane_jump_degrees': max(angles, default=0.),
            'lines': lines}
