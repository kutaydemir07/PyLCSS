# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Reduce manufacturing meshes without changing closed shells or cavities."""
from __future__ import annotations

import numpy as np
import trimesh

from .cad_diagnostics import assess_sampled_mesh_deviation
from .mesh_topology import mesh_topology_report, topology_error


def compact_mesh(vertices, faces, *, tolerance, target_reduction=0.7, samples=512):
    """Try two quadric reductions, accepting only measured geometry fidelity.

    Tolerance is in model units. The error check is deterministic and sampled,
    not a certified Hausdorff bound. A rejected simplification retains the
    input mesh, including every enclosed cavity and disconnected body.
    """
    source = trimesh.Trimesh(vertices, faces, process=False)
    report = {'method': 'Quadric edge collapse with topology and sampled error checks',
              'input_triangles': len(faces), 'output_triangles': len(faces),
              'tolerance': float(tolerance), 'reduced': False, 'rejected_attempts': []}
    if len(faces) < 256 or tolerance <= 0 or target_reduction <= 0:
        return np.asarray(vertices), np.asarray(faces), report
    original = mesh_topology_report(vertices, faces)
    if not original['is_closed'] or source.volume <= 0:
        report['reason'] = 'Reduction requires a closed, oriented material boundary'
        return np.asarray(vertices), np.asarray(faces), report

    def signature(mesh):
        return sorted((int(body.euler_number), int(np.sign(body.volume)))
                      for body in mesh.split(only_watertight=False))

    shells = signature(source)
    for reduction in (target_reduction, target_reduction*0.5):
        try:
            candidate = source.simplify_quadric_decimation(percent=float(reduction), aggression=3)
            error = topology_error(original, mesh_topology_report(candidate.vertices, candidate.faces))
            if error or signature(candidate) != shells:
                raise ValueError(error or 'Reduction changed cavity orientation or shell genus')
            volume_delta = abs(candidate.volume-source.volume)/source.volume
            if volume_delta > 0.02:
                raise ValueError(f'Reduction changed material volume by {volume_delta:.1%}')
            deviation = assess_sampled_mesh_deviation(source.vertices, source.faces,
                                                     candidate.vertices, candidate.faces, samples=samples)
            if deviation['maximum_distance'] > tolerance:
                raise ValueError(f'Sampled surface error exceeds {tolerance:g}')
            if len(candidate.faces) >= len(source.faces):
                break
            report.update(reduced=True, output_triangles=len(candidate.faces),
                          relative_volume_delta=float(volume_delta), sampled_deviation=deviation)
            return np.asarray(candidate.vertices), np.asarray(candidate.faces), report
        except (ImportError, ValueError, RuntimeError) as error:
            report['rejected_attempts'].append(str(error))
    return np.asarray(vertices), np.asarray(faces), report
