# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Accuracy and sampling settings for in-process geometry recovery."""
from dataclasses import dataclass
import math


CAD_RECONSTRUCTION_REVISION = 36


def cached_cad_shape(payload, strategy):
    """Reuse current CAD for display, including an incomplete surface model.

    Solid-only consumers must also call ``require_step_export_ready`` with
    the reconstruction report; a useful preview need not enclose material.
    """
    from .lattice_cad import topopt_export_formats
    if 'STEP' not in topopt_export_formats(payload):
        return None
    report = payload.get('cad_reconstruction')
    if (not isinstance(report, dict) or report.get('backend') != 'in-process'
            or report.get('revision') != CAD_RECONSTRUCTION_REVISION
            or report.get('requested_strategy') != strategy
            or report.get('preview_only') or report.get('validation_status') == 'unverified'):
        return None
    if report.get('paper_pipeline'):
        if not report.get('cad_face_count') or not report.get('display_ready'):
            return None
    elif not report.get('valid') or report.get('export_ready') is False:
        return None
    return payload.get('cad_shape') or payload.get('shape')


def require_step_export_ready(report):
    """Reject known incomplete reconstruction before starting STEP transfer.

    Prefer the latest failed candidate's diagnosis and layout IDs, retaining
    the initial integration evidence in the report. Missing legacy metadata
    still reaches the exporter's native closed-solid check.
    """
    if not isinstance(report, dict):
        return
    incomplete = (
        report.get('is_solid') is False or report.get('export_ready') is False
        or report.get('valid') is False or report.get('failed') is True
        or report.get('validation_status') in {'incomplete', 'failed', 'unverified'}
    )
    if not incomplete:
        return
    stages = report.get('stages') or report.get('stage_reports') or {}
    integration = stages.get('solid_integration') or {}
    retry = stages.get('local_chart_retry') or {}
    latest = (report.get('latest_retry_failure') or integration.get('latest_retry_failure')
              or retry.get('latest_retry_failure') or {})
    reason = latest.get('reason') or integration.get('reason') or report.get('error')
    if not reason:
        warnings = report.get('warnings') or ()
        reason = warnings if isinstance(warnings, str) else '; '.join(map(str, warnings))
    if not reason:
        reason = 'The fitted surfaces have not produced a verified closed solid.'
    rejected = latest.get('rejected_layout_ids') if latest else integration.get('rejected_layout_ids')
    if rejected:
        reason = f"{reason} (layouts {', '.join(map(str, rejected))})"
    if report.get('is_solid') is False:
        status = 'The current CAD result contains fitted surfaces only.'
    else:
        status = 'The current CAD result is not ready for solid export.'
    message = (
        f'STEP export requires a valid closed CAD solid. {status}\n'
        f'Solid construction: {reason}'
    )
    if not latest and retry.get('accepted') is False and retry.get('reason'):
        retry_reason = retry['reason']
        retry_rejected = retry.get('rejected_layout_ids')
        if retry_rejected:
            retry_reason = f"{retry_reason} (layouts {', '.join(map(str, retry_rejected))})"
        message += f'\nLatest chart retry: {retry_reason}'
    checkpoint = report.get('failure_checkpoint_dir')
    if checkpoint:
        message += f'\nDiagnostics saved to: {checkpoint}'
    raise RuntimeError(message)


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
