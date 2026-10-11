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
import trimesh

from .cad_config import CadReconstructionConfig, CAD_RECONSTRUCTION_REVISION
from .cad_diagnostics import (_export_stage_mesh, _export_cad_animation_mesh,
                              assess_sampled_mesh_deviation, assess_sampled_surface_deviation)
from .cad_features import (_payload_bounds, _apply_passive_regions_to_step,
                           _attach_passive_regions_to_lattice)
from .cad_mesh_repair import _extract_recovered_mesh, _effective_tolerance
from .g1_surface import (SurfaceFitError, build_quad_cage, fit_patches, patch_mesh,
                         assemble_solid, _shell_signature)
from .lattice_cad import (beam_lattice_from_member_plan, beam_lattice_from_structure,
                          beam_lattice_solid, lattice_cad_strategy)
from .mesh_surface import preprocess_and_repair_mesh
from .step_export import verify_step_roundtrip, assert_unchanged_surface_geometry, export_cad_model


class SmoothCadUnavailable(SurfaceFitError):
    """A recovered boundary could not be converted within the requested limits."""


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
    # Historical reconstruction selections are migrated; no old engine runs.
    if key in {'auto', 'amrto', 'auto (amrto)', 'amrto bicubic nurbs',
               'recovered shape', 'faceted'}:
        return 'auto'
    raise ValueError(f'Unknown CAD reconstruction strategy: {value}')


def _structure_options(payload, supplied):
    from ..manufacturing.structures import ManufacturingStructureOptions
    recovered = payload.get('recovered_shape') or {}
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
    fit_tolerance=0., relative_fit_tolerance=1e-3, crease_angle_deg=None,
    maximum_volume_delta=0.05, maximum_relative_deviation=0.01,
    max_faces=None, repair_mesh=True, validate_watertight=True, validate_brep=True,
    void_axial_margin=0.02, void_radial_margin=0., return_report=False,
    structure_options=None, member_plan=None, export_steps_dir=None,
    reconstruction_config: CadReconstructionConfig | None = None,
    progress_callback=None, **legacy_options,
):
    """Return a verified STEP-ready Workplane and optional acceptance report.

    Beam families use analytic cylinders/cones. Continuum and implicit/sheet
    lattices use a topology-checked quad cage and coupled degree-five G1 fit.
    A failed fit is retried with a finer cage, up to explicit size limits.
    Historical optional keyword arguments are accepted for saved callers;
    they cannot reactivate a removed reconstruction engine.
    """
    if not isinstance(payload, dict):
        raise TypeError('CAD reconstruction needs a topology result dictionary')
    config = reconstruction_config or CadReconstructionConfig()
    if crease_angle_deg is not None:
        config = replace(config, crease_angle_deg=float(crease_angle_deg))
    if max_faces is not None:
        config = replace(config, maximum_patches=int(max_faces))
    for name, value in (('sew_tolerance', sew_tolerance), ('relative_sew_tolerance', relative_sew_tolerance),
                        ('fit_tolerance', fit_tolerance), ('relative_fit_tolerance', relative_fit_tolerance),
                        ('maximum_volume_delta', maximum_volume_delta),
                        ('maximum_relative_deviation', maximum_relative_deviation)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    strategy = normalize_cad_strategy(reconstruction_strategy)
    options = _structure_options(payload, structure_options)
    kind = lattice_cad_strategy(options)
    is_tpms = bool(options is not None and options.family is not None and options.family.is_tpms)
    if kind == 'isosurface' and reconstruction_config is None:
        config = replace(config, target_triangles=640, maximum_patches=12000,
                         fairness=.01, tangent_stiffness=100., maximum_refinements=4)
        if max_faces is not None:
            config = replace(config, maximum_patches=int(max_faces))
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
    if strategy == 'tpms' and kind != 'isosurface':
        raise ValueError('TPMS spline export requires an implicit or sheet lattice')

    if kind == 'beam' and strategy in {'auto', 'beam', 'beam_fused'}:
        progress('Building analytic lattice members')
        lattice = build_topopt_beam_lattice(payload, options, plan)
        workplane, beam_report = beam_lattice_solid(
            lattice, fuse=strategy == 'beam_fused' or config.beam_fuse,
            maximum_fused_solids=config.maximum_fused_solids, return_report=True)
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
    else:
        progress('Preparing recovered boundary')
        recovered = _extract_recovered_mesh(payload)
        if recovered is None:
            from .surface_recovery import _recover_voxel_shape
            field = payload.get('recovery_density')
            if field is None:
                field = payload.get('density')
            if field is None:
                raise SmoothCadUnavailable('No recovered boundary or voxel density was supplied')
            regions = payload.get('passive_regions') or {}
            boundary = _recover_voxel_shape(np.asarray(field), _payload_bounds(payload),
                                            float(payload.get('recovery_cutoff') or payload.get('density_cutoff') or .5),
                                            structure_options=options, member_plan=plan,
                                            source_mask=payload.get('source_mask'),
                                            passive_solid_mask=payload.get('passive_solid_mask'),
                                            passive_void_mask=payload.get('passive_void_mask'),
                                            extrusion_axis=payload.get('extrusion_axis') or 'none',
                                            **{name: regions.get(name) or () for name in
                                               ('solid_boxes', 'void_boxes', 'solid_cylinders',
                                                'void_cylinders', 'joint_pin_cylinders')})
            if boundary is None:
                raise SmoothCadUnavailable('Voxel recovery did not produce a boundary')
            recovered = _extract_recovered_mesh(boundary)
        vertices, faces = recovered
        if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError('Recovered boundary has invalid coordinates or triangle indices')
        if repair_mesh:
            vertices, faces = preprocess_and_repair_mesh(vertices, faces, laplacian_iters=0)
        source = trimesh.Trimesh(vertices, faces, process=False)
        if not source.is_watertight or not source.is_winding_consistent or source.volume <= 0:
            raise SmoothCadUnavailable('Recovered material boundary must be closed with outward material orientation')
        diagonal = float(np.linalg.norm(source.extents))
        sew = _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance)
        fit_limit = max(float(fit_tolerance), float(relative_fit_tolerance) * diagonal)
        deviation_limit = float(maximum_relative_deviation) * diagonal
        stages['surface_recovery'] = {'vertex_count': len(vertices), 'triangle_count': len(faces),
                                      'material_volume': float(source.volume)}
        if folder:
            _export_stage_mesh(folder, 'recovered_boundary', [('source', vertices, faces)])
        failures = []
        shape = None
        initial_target = config.target_triangles
        signature = _shell_signature(source)
        genus = sum(max(0, 1 - chi / 2) for chi, _ in signature)
        if reconstruction_config is None:
            # Handles and disconnected shells need enough charts regardless
            # of whether the boundary came from topopt or a lattice field.
            initial_target = max(initial_target, int(800 * genus), 160 * len(signature))
            if genus:
                config = replace(config, fairness=max(config.fairness, .01),
                                 tangent_stiffness=max(config.tangent_stiffness, 100.))
        report['configuration'] = asdict(config)
        stages['surface_recovery'].update(boundary_shell_count=len(signature), total_genus=genus)
        previous_profile = None
        # Multiply handled voxel boundaries are especially susceptible to
        # curvature being mistaken for a branching crease graph. Auto starts
        # with smooth wall charts there; explicit feature settings stay exact.
        smooth_feature_retry = (reconstruction_config is None and crease_angle_deg is None and genus >= 2)
        stages['attempts'] = []
        for attempt in range(config.maximum_refinements):
            target = min(config.maximum_patches // 3, initial_target * 2**attempt)
            implicit_features = is_tpms or smooth_feature_retry
            profile = (target, implicit_features, config.fairness, config.tangent_stiffness)
            if profile == previous_profile:
                break
            previous_profile = profile
            attempt_start = perf_counter()
            attempt_report = {'number': attempt + 1, 'target_triangles': target,
                              'fairness': config.fairness, 'tangent_stiffness': config.tangent_stiffness,
                              'crease_policy': 'Clipping planes' if implicit_features else 'Dihedral features'}
            stages['attempts'].append(attempt_report)
            progress(f'Fitting G1 splines (attempt {attempt + 1})')
            try:
                cage_vertices, quads, cage_report = build_quad_cage(source, target, config.maximum_patches,
                                                                   config.crease_angle_deg,
                                                                   implicit_surface=implicit_features)
                cage_report['crease_policy'] = attempt_report['crease_policy']
                poles, fit_report = fit_patches(source, cage_vertices, quads, config,
                                               crease_edges=cage_report['crease_edges'])
                sampled_vertices, sampled_faces = patch_mesh(poles)
                deviation = assess_sampled_mesh_deviation(vertices, faces, sampled_vertices,
                                                         sampled_faces, samples=config.validation_samples)
                if deviation_limit and deviation['maximum_distance'] > deviation_limit:
                    raise SurfaceFitError(f'Sampled geometric deviation {deviation["maximum_distance"]:.6g} '
                                          f'exceeds {deviation_limit:.6g}')
                if fit_limit and fit_report['fitting_rms_distance'] > fit_limit:
                    raise SurfaceFitError(f'Fitting RMS distance exceeds {fit_limit:.6g}')
                shape, assembly = assemble_solid(
                    poles, sew, quads=quads, crease_edges=cage_report['crease_edges'],
                    maximum_chart_patches=config.maximum_chart_patches,
                    maximum_chart_spans=config.maximum_chart_spans)
                if _shell_signature(trimesh.Trimesh(sampled_vertices, sampled_faces, process=True)) != _shell_signature(source):
                    raise SurfaceFitError('Fitted tessellation changed boundary-shell topology')
                # Adaptive integration splits at internal spline knots. OCC's
                # fixed-order default can mismeasure consolidated surfaces.
                delta = abs(shape.Volume(tol=1e-6) - source.volume) / source.volume
                if maximum_volume_delta and delta > maximum_volume_delta:
                    raise SurfaceFitError(f'Fitting changed material volume by {delta:.1%}')
                report.update(assembly, method='TPMS G1 spline' if is_tpms else 'Sparse G1 spline',
                              smooth=True, editable=True, relative_volume_delta=float(delta),
                              sampled_surface_deviation=deviation, continuity=fit_report,
                              attempts=attempt + 1, rejected_attempts=failures)
                stages.update(quad_cage=cage_report, spline_fit=fit_report)
                cage_report['requested_cage_triangles'] = target
                if kind == 'isosurface':
                    report['lattice_surface'] = {'family': options.mode if options else None,
                                                'source': 'Actual recovered manufacturing boundary',
                                                'grading_and_clipping_retained': True}
                if folder:
                    _export_stage_mesh(folder, 'quad_cage', [('cage', cage_vertices, quads)])
                    _export_stage_mesh(folder, 'fitted_surface', [('splines', sampled_vertices, sampled_faces)])
                attempt_report.update(accepted=True, elapsed_seconds=perf_counter() - attempt_start)
                break
            except SurfaceFitError as error:
                attempt_report.update(accepted=False, reason=str(error),
                                      elapsed_seconds=perf_counter() - attempt_start)
                failures.append(f'Attempt {attempt + 1} ({target} cage triangles): {error}')
                progress(f'Fit rejected: {error}')
                if reconstruction_config is None and len(getattr(error, 'failed_patches', [])):
                    # Retry a folded general surface with the stabilizing
                    # tangent/fairness profile too. It still has to satisfy
                    # the same source-distance and material-volume limits.
                    config = replace(config, fairness=max(config.fairness, .01),
                                     tangent_stiffness=max(config.tangent_stiffness, 100.))
                    report['configuration'] = asdict(config)
                    # A coarse dihedral graph may interpret curvature as a
                    # branching sharp feature and create incompatible tangent
                    # sectors. Auto can retry smoother charts while retaining
                    # explicit clipping planes and all source acceptance gates.
                    if crease_angle_deg is None:
                        smooth_feature_retry = True
                shape = None
        if shape is None:
            if folder:
                report.update(valid=False, rejected_attempts=failures,
                              elapsed_seconds=perf_counter() - start)
                (folder / 'cad_stage_reports.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise SmoothCadUnavailable('Smooth CAD reconstruction exhausted its bounded refinements: '
                                       + '; '.join(failures))
        shape = _apply_passive_regions_to_step(shape, payload,
                                               void_axial_margin=void_axial_margin,
                                               void_radial_margin=void_radial_margin)
        report['continuity_scope'] = 'Uncreased fitted patch seams before analytic passive feature booleans'
    progress('Verifying STEP geometry')
    shape, transfer = _verify_transfer(shape, folder)
    report.update(step_roundtrip=transfer, valid=bool(shape.isValid()),
                  is_solid=bool(shape.Solids()), cad_volume=float(shape.Volume(tol=1e-6)),
                  cad_face_count=len(shape.Faces()), solid_count=len(shape.Solids()))
    if folder:
        box = shape.BoundingBox()
        report['animation_mesh'] = _export_cad_animation_mesh(
            folder, 'cad', shape, max(np.linalg.norm((box.xlen, box.ylen, box.zlen)) * 1e-3, 1e-7))
    report['elapsed_seconds'] = perf_counter() - start
    if folder:
        (folder / 'cad_stage_reports.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    workplane = cq.Workplane(obj=shape)
    return (workplane, report) if return_report else workplane


__all__ = ['SmoothCadUnavailable', 'CadReconstructionConfig', 'build_topopt_beam_lattice',
           'normalize_cad_strategy', 'reconstruct_topopt_cad', 'export_cad_model']
