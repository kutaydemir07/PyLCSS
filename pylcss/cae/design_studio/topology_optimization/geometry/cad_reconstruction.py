# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""In-process CAD reconstruction for continuum, beam and implicit lattices."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import cadquery as cq
import numpy as np

from .cad_config import CadReconstructionConfig, CAD_RECONSTRUCTION_REVISION
from ._compat import warn_retired_options
from .cad_diagnostics import _export_cad_animation_mesh
from .cad_features import (_payload_bounds, _apply_passive_regions_to_step,
                           _attach_passive_regions_to_lattice)
from .cad_mesh_repair import _extract_recovered_mesh
from .spline_recovery import SurfaceFitError
from .lattice_cad import (beam_lattice_from_member_plan, beam_lattice_from_structure,
                          beam_lattice_solid, beam_lattice_triangles,
                          lattice_cad_strategy, topopt_export_formats)
from .occ_shapes import _unify_same_domain_shape
from .step_export import verify_step_roundtrip, assert_unchanged_surface_geometry, export_cad_model


class SmoothCadUnavailable(SurfaceFitError):
    """A recovered boundary could not be converted at the source resolution."""


def normalize_cad_strategy(value):
    """Map persisted method names to the current geometry strategies."""
    key = str(value or 'auto').strip().lower()
    if key in {'beam', 'analytic beam lattice'}:
        return 'beam'
    if key in {'beam_fused', 'fused beam lattice'}:
        return 'beam_fused'
    if key in {'spline', 'smooth spline', 'smooth freeform'} or key.startswith('smooth'):
        return 'spline'
    if key in {'tpms', 'tpms spline'}:
        return 'tpms'
    if key in {'amrto', 'amrto bicubic nurbs'}:
        return 'amrto'
    if key in {'auto', 'auto (amrto)',
               'recovered shape', 'faceted'}:
        return 'auto'
    raise ValueError(f'Unknown CAD reconstruction strategy: {value}')


def _structure_options(payload, supplied):
    from ..manufacturing.structures import ManufacturingStructureOptions
    recovered = payload.get('recovered_shape') or payload
    value = recovered.get('resolved_structure_options') or supplied or payload.get('structure_options')
    if isinstance(value, dict):
        names = {field.name for field in fields(ManufacturingStructureOptions)}
        return ManufacturingStructureOptions(**{key: val for key, val in value.items() if key in names})
    if isinstance(value, str):
        return ManufacturingStructureOptions(mode=value)
    return value


def build_topopt_beam_lattice(payload, structure_options, member_plan=None):
    """Build model-space members from the solved sizing plan or density field."""
    bounds = _payload_bounds(payload)
    if bounds is None:
        field = payload.get('density')
        if field is None:
            field = payload.get('recovery_density')
        if field is None:
            raise ValueError('Beam export requires physical study bounds')
        bounds = (np.zeros(3), np.asarray(np.shape(field), dtype=float))
    if member_plan is not None:
        return beam_lattice_from_member_plan(member_plan, bounds)
    density = payload.get('density')
    if density is None:
        density = payload.get('recovery_density')
    if density is None:
        raise ValueError('An analytic lattice body needs the optimized density field')
    cutoff = float(payload.get('density_cutoff') or payload.get('recovery_cutoff') or 0.5)
    return beam_lattice_from_structure(np.asarray(density, dtype=float), cutoff,
                                      structure_options, bounds,
                                      passive_solid_mask=payload.get('passive_solid_mask'),
                                      passive_void_mask=payload.get('passive_void_mask'))


def _verify_transfer(shape, folder=None):
    """Verify the exact geometry that CAD consumers will receive."""
    def transfer(path):
        export_cad_model(shape, (), path, verify=False)
        restored, report = verify_step_roundtrip(path, return_shape=True)
        report.update(assert_unchanged_surface_geometry(shape, restored))
        return restored, report
    if folder is not None:
        return transfer(folder / 'cad.step')
    with TemporaryDirectory(prefix='pylcss_cad_') as directory:
        return transfer(Path(directory) / 'cad.step')


def reconstruct_topopt_cad(
    payload: Any, *, source_geometry='Recovered Shape', reconstruction_strategy='auto',
    sew_tolerance=1e-3, relative_sew_tolerance=1e-6,
    fit_tolerance=0., relative_fit_tolerance=5e-3, crease_angle_deg=None,
    maximum_volume_delta=0.05, maximum_relative_deviation=0.01,
    max_faces=None, repair_mesh=True, merge_angle_deg=0.,
    void_axial_margin=0.02, void_radial_margin=0., return_report=False,
    structure_options=None, member_plan=None, export_steps_dir=None, failure_checkpoint_dir=None,
    reconstruction_config: CadReconstructionConfig | None = None,
    progress_callback=None, quad_layout=None, mechanical_validation=None, amrto_config=None,
    preview_only=False, **legacy_options,
):
    """Return AMRTO CAD with solid integration, or analytic beam lattice CAD.

    Continuum boundaries use the five paper stages, common seam constraints
    and the original non-design CAD interfaces. Each patch is fitted once.
    Historical optional keyword arguments are accepted for saved callers;
    they cannot reactivate a removed reconstruction engine.
    """
    if not isinstance(payload, dict):
        raise TypeError('CAD reconstruction needs a topology result dictionary')
    warn_retired_options(legacy_options, operation='CAD reconstruction')
    if max_faces is not None and (int(max_faces) != max_faces or max_faces < 1):
        raise ValueError('max_faces must be a positive integer')
    config = reconstruction_config or CadReconstructionConfig()
    if crease_angle_deg is not None:
        config = replace(config, crease_angle_deg=float(crease_angle_deg))
    for name, value in (('sew_tolerance', sew_tolerance), ('relative_sew_tolerance', relative_sew_tolerance),
                        ('fit_tolerance', fit_tolerance), ('relative_fit_tolerance', relative_fit_tolerance),
                        ('maximum_volume_delta', maximum_volume_delta),
                        ('merge_angle_deg', merge_angle_deg),
                        ('maximum_relative_deviation', maximum_relative_deviation)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    strategy = normalize_cad_strategy(reconstruction_strategy)
    if max_faces is not None and strategy == 'spline':
        config = replace(config, target_triangles=int(max_faces))
    options = _structure_options(payload, structure_options)
    kind = lattice_cad_strategy(options)
    if kind == 'beam' and ('STEP' not in topopt_export_formats(options)
                           or 'STEP' not in topopt_export_formats(payload)):
        raise ValueError('Analytic member STEP would omit skin or solid features; '
                         'export the complete body as STL.')
    plan = member_plan if member_plan is not None else payload.get('member_plan')
    folder = Path(export_steps_dir) if export_steps_dir else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    stages = {}
    def progress(name):
        if progress_callback:
            progress_callback({'type': 'topopt_progress', 'status_only': True,
                               'stage': f'CAD reconstruction — {name}'}, None, 0, 0)
    report = {'requested_strategy': strategy, 'source_geometry': source_geometry,
              'backend': 'in-process', 'revision': CAD_RECONSTRUCTION_REVISION, 'stage_reports': stages,
              'configuration': asdict(config)}
    if strategy in {'beam', 'beam_fused'} and kind != 'beam':
        raise ValueError('Analytic beam export requires a strut lattice family')
    if kind != 'beam' or strategy not in {'auto', 'beam', 'beam_fused'}:
        # Viewing and export share the five-stage fit and the same downstream
        # stitching/exact-interface integration. No second fitting loop.
        from .amrto_reconstruction import reconstruct_topopt_cad as reconstruct_amrto
        from .amrto_surface import cached_surface_layout
        amrto_payload = dict(payload)
        cache = cached_surface_layout(payload)
        if isinstance(cache, dict) and cache.get('reports', {}).get('quadrangulation', {}).get('paper_pipeline'):
            amrto_payload['amrto_cached_quadrangulation'] = {
                'vertices': cache['vertices'], 'quads': cache['quads'],
                'report': {**cache['reports']['quadrangulation'],
                           'source_geometry_sha256': cache['design_geometry_sha256'],
                           'requested_settings': cache['requested_settings']}}
        paper_model, paper_report = reconstruct_amrto(
            amrto_payload, reconstruction_strategy='amrto',
            sew_tolerance=sew_tolerance, relative_sew_tolerance=relative_sew_tolerance,
            crease_angle_deg=crease_angle_deg, max_faces=max_faces,
            repair_mesh=repair_mesh, return_report=True, amrto_config=amrto_config,
            maximum_volume_delta=maximum_volume_delta,
            maximum_relative_deviation=maximum_relative_deviation,
            void_axial_margin=void_axial_margin, void_radial_margin=void_radial_margin,
            export_steps_dir=folder, failure_checkpoint_dir=failure_checkpoint_dir,
            progress_callback=progress_callback)
        paper_report['requested_strategy'] = strategy
        return (paper_model, paper_report) if return_report else paper_model

    if kind == 'beam' and strategy in {'auto', 'beam', 'beam_fused'}:
        progress('Building analytic lattice members')
        lattice = build_topopt_beam_lattice(payload, options, plan)
        workplane, beam_report = beam_lattice_solid(
            lattice, fuse=strategy == 'beam_fused' or config.beam_fuse,
            return_report=True)
        if beam_report['fused']:
            shape = _apply_passive_regions_to_step(workplane.val(), payload,
                                                   void_axial_margin=void_axial_margin,
                                                   void_radial_margin=void_radial_margin)
            passive = {'mode': 'Analytic booleans on the fused lattice; hardware stays separate'}
        else:
            shape, passive = _attach_passive_regions_to_lattice(
                workplane.val(), payload, void_axial_margin=void_axial_margin,
                void_radial_margin=void_radial_margin)
        report.update(beam_report, beam_lattice=lattice.diagnostics(), passive_features=passive)
        report['volume_semantics'] = ('Sum of member solids; overlapping joints are counted repeatedly'
                                    if not beam_report['fused'] else 'Boolean union material volume')
    progress('Checking the recovered CAD solid')
    if merge_angle_deg and 'amrto' not in stages:
        shape = _unify_same_domain_shape(shape, merge_angle_deg=merge_angle_deg)
    if not shape.isValid() or not shape.Solids():
        raise SmoothCadUnavailable('Reconstruction did not produce valid material solids')
    # STEP reimport is done when a file is exported. A CAD preview does not
    # need a temporary export/import of every lattice member.
    transfer = report.get('step_roundtrip', {'status': 'Checked on STEP export'})
    if folder is not None:
        shape, transfer = _verify_transfer(shape, folder)
    report.update(step_roundtrip=transfer, valid=bool(shape.isValid()),
                  is_solid=bool(shape.Solids()), cad_volume=float(shape.Volume(tol=1e-6)),
                  cad_face_count=len(shape.Faces()), solid_count=len(shape.Solids()))
    report['grouping'] = {'product': 'Recovered design', 'part': 'Material',
                          'solid_count': len(shape.Solids())}
    request = mechanical_validation if mechanical_validation is not None else payload.get('cad_mechanical_validation')
    if request is not None:
        from ..verification.cad_performance import validate_cad_performance
        progress('Comparing recovered and reconstructed mechanical performance')
        recovered = _extract_recovered_mesh(payload)
        if recovered is None and kind == 'beam':
            tri_v, tri_f = beam_lattice_triangles(lattice)
            if len(tri_v) and len(tri_f):
                recovered = (tri_v, tri_f)
        if recovered is None:
            raise SmoothCadUnavailable('Mechanical comparison requires a recovered reference surface')
        performance = validate_cad_performance(recovered, shape, **request)
        report['mechanical_validation'] = performance
        if not performance['passed']:
            report['valid'] = False
            if folder:
                (folder / 'cad_stage_reports.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise SmoothCadUnavailable('Reconstructed CAD failed mechanical performance preservation: '
                                       + performance['reason'])
    else:
        report['mechanical_validation'] = {'status': 'not run', 'passed': None,
                                            'reason': 'Material, supports and loads were not supplied for CAD comparison'}
    if folder:
        box = shape.BoundingBox()
        report['animation_mesh'] = _export_cad_animation_mesh(
            folder, 'cad', shape, max(np.linalg.norm((box.xlen, box.ylen, box.zlen)) * 1e-3, 1e-7))
    report['elapsed_seconds'] = perf_counter() - start
    surface_layout = report.pop('_surface_layout', None)
    if folder:
        (folder / 'cad_stage_reports.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if surface_layout is not None:
        report['_surface_layout'] = surface_layout
    workplane = cq.Workplane(obj=shape)
    return (workplane, report) if return_report else workplane


__all__ = ['SmoothCadUnavailable', 'CadReconstructionConfig', 'build_topopt_beam_lattice',
           'normalize_cad_strategy', 'reconstruct_topopt_cad', 'export_cad_model']
