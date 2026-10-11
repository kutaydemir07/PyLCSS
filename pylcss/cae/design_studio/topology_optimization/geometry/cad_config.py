# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Bounded, in-process surface recovery and CAD reconstruction settings."""
from dataclasses import dataclass
import math


CAD_RECONSTRUCTION_REVISION = 2


def cached_cad_shape(payload, strategy):
    """Reuse only a verified body from this reconstruction revision/method."""
    report = payload.get('cad_reconstruction')
    if (not isinstance(report, dict) or report.get('backend') != 'in-process'
            or report.get('revision') != CAD_RECONSTRUCTION_REVISION
            or report.get('requested_strategy') != strategy or not report.get('valid')):
        return None
    return payload.get('cad_shape') or payload.get('shape')


@dataclass(frozen=True)
class SurfaceRecoveryConfig:
    density_cutoff: float = 0.5
    density_smooth_sigma: float = 0.5
    mesh_laplacian_smoothing_iters: int = 2
    mesh_laplacian_lambda: float = 0.3


@dataclass(frozen=True)
class CadReconstructionConfig:
    target_triangles: int = 160
    maximum_patches: int = 12000
    maximum_refinements: int = 5
    samples_per_axis: int = 8
    validation_samples: int = 1000
    fairness: float = 1e-4
    tangent_stiffness: float = 10.0
    constraint_tolerance: float = 1e-8
    constraint_iterations: int = 80
    crease_angle_deg: float = 55.0
    beam_fuse: bool = False
    maximum_fused_solids: int = 250
    maximum_chart_patches: int = 64
    maximum_chart_spans: int = 16

    def __post_init__(self):
        for name in ('target_triangles', 'maximum_patches', 'maximum_refinements',
                     'samples_per_axis', 'validation_samples', 'constraint_iterations',
                     'maximum_fused_solids', 'maximum_chart_patches', 'maximum_chart_spans'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.samples_per_axis < 6 or self.maximum_patches < 12:
            raise ValueError('Quintic fitting requires at least six samples per axis and twelve patches')
        if not math.isfinite(self.crease_angle_deg) or not 0 < self.crease_angle_deg <= 180:
            raise ValueError('crease_angle_deg must be finite and in (0, 180]')
        if not math.isfinite(self.fairness) or self.fairness < 0:
            raise ValueError('fairness must be finite and nonnegative')
        if not math.isfinite(self.tangent_stiffness) or self.tangent_stiffness < 0:
            raise ValueError('tangent_stiffness must be finite and nonnegative')
        if not math.isfinite(self.constraint_tolerance) or self.constraint_tolerance <= 0:
            raise ValueError('constraint_tolerance must be finite and positive')
