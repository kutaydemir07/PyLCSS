# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Accuracy and sampling settings for in-process geometry recovery."""
from dataclasses import dataclass
import math


CAD_RECONSTRUCTION_REVISION = 10


def cached_cad_shape(payload, strategy):
    """Reuse only a verified body from this reconstruction revision/method."""
    from .lattice_cad import topopt_export_formats
    if 'STEP' not in topopt_export_formats(payload):
        return None
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
    samples_per_axis: int = 8
    validation_samples: int = 1000
    fairness: float = 0.01
    crease_angle_deg: float = 55.0
    beam_fuse: bool = False

    def __post_init__(self):
        for name in ('target_triangles', 'samples_per_axis', 'validation_samples'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.samples_per_axis < 4:
            raise ValueError('Cubic fitting requires four samples per axis')
        if not math.isfinite(self.crease_angle_deg) or not 0 < self.crease_angle_deg <= 180:
            raise ValueError('crease_angle_deg must be finite and in (0, 180]')
        if not math.isfinite(self.fairness) or self.fairness < 0:
            raise ValueError('fairness must be finite and nonnegative')
