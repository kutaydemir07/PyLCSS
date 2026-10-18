# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""The five AMRTO stages and the downstream solid CAD integration.

Ren et al., CMAME 435 (2025), 117673, Sections 2.3-2.7 and 3.1.
Section 2.7 stitches the fitted surfaces before volumetric analysis. Common
boundary constraints automate that integration; a failed solid may receive
one explicitly reported local chart retry without changing the source mesh.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time

import cadquery as cq
import numpy as np
import trimesh
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from .amrto_common import AMRTOConfig
from .amrto_tri_mesh import generate_topopt_triangular_mesh, preprocess_and_repair_mesh
from .amrto_quadrangulation import quadrangulate_paper_mesh
from .amrto_layout_partition import PaperLayoutError, divide_paper_sparse_quad_layouts
from .amrto_parameterization import parameterize_paper_layout, adaptive_sample_layout
from .amrto_nurbs_fitting import fit_paper_nurbs_surface_for_layout, nurbs_surface_to_occ_face
from .amrto_surface import geometry_fingerprint, layout_display_data, add_source_groups
from .cad_config import CAD_RECONSTRUCTION_REVISION
from .cad_diagnostics import _export_stage_mesh, _export_cad_animation_mesh
from .cad_mesh_repair import _extract_recovered_mesh, _extract_amrto_design_mesh, _extract_supplied_quad_mesh, _effective_tolerance
from .cad_features import _payload_bounds, restore_paper_non_design_regions
from .amrto_shared_curves import prepare_shared_boundary_curves
from .amrto_solid import stitch_paper_surfaces, split_closed_paper_faces
from .occ_shapes import _assess_brep_self_intersections, _solid_volume
from .mesh_topology import triangulate_quad_faces
from .step_export import export_cad_model

logger = logging.getLogger(__name__)


class SmoothCadUnavailable(RuntimeError):
    """The supplied data could not produce any CAD surfaces."""


def build_topopt_beam_lattice(payload, structure_options, member_plan=None):
    from .cad_reconstruction import build_topopt_beam_lattice as build
    return build(payload, structure_options, member_plan)


def _fit_paper_layout(task):
    layout, cfg = task
    scope = threadpool_limits(limits=1, user_api='blas') if threadpool_limits else nullcontext()
    with scope:
        fit_paper_nurbs_surface_for_layout(
            layout, degree_u=cfg.nurbs_degree_u, degree_v=cfg.nurbs_degree_v,
            k_type=cfg.ctrlpts_ratio_kt, min_ctrlpts=cfg.ctrlpts_min,
            curvature_ck=cfg.multires_curvature_ck, curvature_cp=cfg.multires_curvature_cp,
            multires_enabled=cfg.multires_enabled, compression_ratio=cfg.multires_compression_ratio,
            multires_layers=cfg.multires_layers,
            maximum_knot_removal_attempts=cfg.multires_maximum_knot_removal_attempts,
            join_boundaries=cfg.stitch_surface_boundaries,
            maximum_source_deviation=getattr(layout, 'solid_source_fit_limit', None),
            maximum_multires_displacement=getattr(layout, 'solid_fit_tolerance', None))
    return layout


def _fit_paper_layouts(layouts, cfg):
    workers = min(max(1, int(cfg.fitting_workers)), os.cpu_count() or 1, len(layouts), 61)
    if workers == 1 or getattr(__import__('sys'), 'frozen', False):
        yield from map(_fit_paper_layout, ((layout, cfg) for layout in layouts))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            yield from pool.map(_fit_paper_layout, ((layout, cfg) for layout in layouts), chunksize=1)


def _assemble_paper_cad(layouts, tolerance):
    """Assemble the paper's independent natural-boundary spline faces unchanged."""
    faces = [cq.Shape.cast(nurbs_surface_to_occ_face(layout.nurbs_surface, tol=tolerance))
             for layout in layouts]
    if not faces:
        raise SmoothCadUnavailable('AMRTO did not generate any NURBS patches')
    # Retain all completed fits here. Sewing and exact solid features belong
    # to the separate CAD integration below.
    return cq.Compound.makeCompound(faces)


def _complete_paper_solid(shape, payload, *, tolerance, source_scale,
                          source_volume, maximum_volume_delta,
                          void_axial_margin, void_radial_margin, progress_callback=None):
    """Join once, restore exact CAD interfaces and retain surfaces on failure."""
    surface_model = shape
    report = {'status': 'surface_only', 'is_solid': False, 'fitting_attempts': 0}
    def progress(stage):
        if callable(progress_callback):
            progress_callback(stage)
    try:
        progress('Stitching fitted CAD surfaces')
        shape, report['stitching'] = stitch_paper_surfaces(
            shape, tolerance=tolerance, source_scale=source_scale,
            source_volume=source_volume,
            maximum_volume_relative_error=maximum_volume_delta)
        if not report['stitching']['is_solid']:
            report['reason'] = 'The fitted surfaces did not close within the stitching tolerance'
            return surface_model, report
        progress('Preparing exact CAD seams')
        shape, report['closed_surface_seams'] = split_closed_paper_faces(shape)
        progress('Restoring exact non-design geometry')
        try:
            shape, report['non_design'] = restore_paper_non_design_regions(
                shape, payload, void_axial_margin=void_axial_margin,
                void_radial_margin=void_radial_margin)
        except MemoryError:
            raise
        except Exception as nd_err:
            logger.warning('Could not apply exact non-design regions onto CAD solid: %s', nd_err)
            report['non_design'] = {'applied': False, 'error': str(nd_err)}
            # A closed fitted body is not the requested CAD when its original
            # bores, interfaces or source boundary could not be restored.
            # Keep the fitted surfaces through the outer failure handler.
            raise RuntimeError(f'Exact non-design restoration failed: {nd_err}') from nd_err
        progress('Checking final CAD solid')
        report['intersections'] = _assess_brep_self_intersections(shape)
        if not report['intersections']['passed']:
            raise RuntimeError('The stitched CAD body has unresolved geometric intersections')
        volume = _solid_volume(shape, relative_tolerance=1e-6)
        if not shape.isValid() or not shape.Solids() or volume is None or volume <= 0:
            raise RuntimeError('Solid CAD construction did not retain valid enclosed material')
        report.update(status='closed_solid', is_solid=True, cad_volume=volume,
                      solid_count=len(shape.Solids()), face_count=len(shape.Faces()))
        return shape, report
    except MemoryError:
        raise
    except Exception as error:
        # Keep the complete fitted model visible. Do not publish a solid-shaped
        # but unusable intermediate to FEA or lattice consumers.
        logger.warning('AMRTO solid integration incomplete: %s', error)
        report['reason'] = str(error)
        return surface_model, report


def _write_parameterized_layout_checkpoint(layouts, folder: Path) -> None:
    """Retain exact source/UV correspondence without exporting display meshes."""
    arrays = {}
    for layout in layouts:
        prefix = f"layout_{int(layout.layout_id)}_"
        arrays[prefix + "vertices"] = layout.vertices
        arrays[prefix + "triangles"] = layout.triangles
        arrays[prefix + "quads"] = layout.quads
        arrays[prefix + "uv"] = layout.parameter_coords
        arrays[prefix + "corners"] = np.asarray(layout.corner_indices, dtype=np.int64)
        arrays[prefix + "sampled_grid_3d"] = np.asarray(layout.sampled_grid_3d).reshape(*layout.sampled_shape, 3)
        arrays[prefix + "sampled_shape"] = np.asarray(layout.sampled_shape, dtype=np.int64)
        for index, path in enumerate(layout.boundary_paths):
            arrays[prefix + f"boundary_{index}"] = np.asarray(path, dtype=np.int64)
        if hasattr(layout, "source_vertex_indices"):
            arrays[prefix + "source_vertex_indices"] = layout.source_vertex_indices
    np.savez_compressed(folder / "step4_parameterized_layouts.npz", **arrays)


def _write_incomplete_cad_checkpoint(folder, vertices, triangles, shape, report):
    """Retain portable failure evidence without losing the fitted CAD view."""
    errors = []
    saved = []

    def retain(name, write):
        try:
            write()
            saved.append(name)
        except Exception as error:
            # Diagnostics are best effort: even a native BREP writer failure
            # must not replace the completed surface model with a failed job.
            errors.append(f'{name}: {error}')
            report['failure_checkpoint_error'] = '; '.join(errors)
            logger.warning('Could not retain AMRTO %s in %s', name, folder, exc_info=True)

    try:
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        report['failure_checkpoint_error'] = str(error)
        logger.warning('Could not create AMRTO failure checkpoint %s', folder, exc_info=True)
        return
    report['failure_checkpoint_dir'] = str(folder)
    retain('source_mesh.npz', lambda: np.savez_compressed(
        folder / 'source_mesh.npz', vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(triangles, dtype=np.int64)))

    def write_brep():
        from OCP.BRepTools import BRepTools
        from OCP.TopTools import TopTools_FormatVersion_VERSION_1
        if not BRepTools.Write_s(shape.wrapped, str(folder / 'fitted_surfaces.brep'),
                                False, False, TopTools_FormatVersion_VERSION_1):
            raise OSError('The CAD surface BREP writer did not produce a checkpoint')

    retain('fitted_surfaces.brep', write_brep)
    # Save before _surface_layout attaches the large in-memory display cache.
    # The integration report contains the precise rejection reason.
    retain('cad_stage_reports.json', lambda: (folder / 'cad_stage_reports.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'))
    if not saved:
        report.pop('failure_checkpoint_dir', None)


def _sampled_layout_checkpoints(layouts, folder: Optional[Path]) -> dict:
    """Show the unchanged source surface and save sampling as a point overlay.

    Connecting a uniform UV grid in 3D creates chords across source corners and
    creases. Parameterization does not move the source mesh, so the stage-4
    surface must retain its triangles rather than display that approximation.
    """
    grid_info, source_meshes = [], []
    for layout in layouts:
        grid = np.asarray(layout.sampled_grid_3d, dtype=float).reshape(*layout.sampled_shape, 3)
        if min(grid.shape[:2]) < 2 or not np.isfinite(grid).all():
            raise RuntimeError(f"AMRTO stage 4 layout {layout.layout_id} has no sampled surface")
        n_u, n_v, _ = grid.shape
        grid_info.append({"layout_id": int(layout.layout_id), "grid_shape": [n_u, n_v],
                          "point_count": n_u * n_v,
                          "parameterization": getattr(layout, "parameterization_report", {})})
        source_meshes.append((f"parameterized_patch_{layout.layout_id}", layout.vertices, layout.triangles))
    if folder is not None:
        _export_stage_mesh(folder, "step4_sampled_surface", source_meshes)
        with (folder / "step4_sampled_points.obj").open("w", encoding="ascii") as stream:
            stream.write("# AMRTO inverse-map samples; point overlay on the unchanged source surface\n")
            offset = 0
            for layout in layouts:
                samples = np.asarray(layout.sampled_grid_3d, dtype=float).reshape(-1, 3)
                stream.write(f"g sampled_patch_{layout.layout_id}\n")
                for vertex in samples:
                    stream.write("v {:.12g} {:.12g} {:.12g}\n".format(*vertex))
                stream.write("p " + " ".join(str(offset + index + 1) for index in range(len(samples))) + "\n")
                offset += len(samples)
        (folder / "step4_sampled_grids.json").write_text(json.dumps(grid_info, indent=2), encoding="utf-8")
        _write_parameterized_layout_checkpoint(layouts, folder)
    branches = {}
    for grid in grid_info:
        method = grid["parameterization"].get("branch", grid["parameterization"].get("method", "unknown"))
        branches[method] = branches.get(method, 0) + 1
    return {"method": "Per-layout parameterization and Eq. 14 sampling", "branch_counts": branches,
            "sampled_grids": grid_info,
            "surface_geometry": "Unchanged source triangles with retained boundaries and creases",
            "sampling_geometry": "Inverse-map points saved separately for overlay; no interpolated sample-grid faces",
            "physical_source_geometry_unchanged": True,
            "surface_checkpoint": "step4_sampled_surface.obj" if folder is not None else None,
            "sample_overlay_checkpoint": "step4_sampled_points.obj" if folder is not None else None}


def reconstruct_topopt_cad(
    payload, *, reconstruction_strategy='auto', sew_tolerance=1e-3,
    relative_sew_tolerance=1e-6, fit_tolerance=0., relative_fit_tolerance=1e-3,
    crease_angle_deg=None, maximum_volume_delta=.05, maximum_relative_deviation=.01,
    merge_angle_deg=0., max_faces=None, repair_mesh=True, validate_watertight=True,
    validate_brep=True, preview_only=False, void_axial_margin=.02, void_radial_margin=0.,
    return_report=False, structure_options=None, member_plan=None, export_steps_dir=None,
    failure_checkpoint_dir=None, amrto_config=None, progress_callback=None, **legacy_options,
):
    """Run each paper stage once, then construct the application's CAD solid.

    Historical validation/preview keywords remain accepted for saved callers;
    they do not select an alternative fit, introduce retries, or gate the CAD
    result. Mathematical iterations within Algorithm 1 and Algorithms 2/3 remain.
    """
    if not isinstance(payload, dict):
        raise TypeError('CAD reconstruction payload must be a dictionary')
    from .lattice_cad import lattice_cad_strategy
    options = structure_options if structure_options is not None else payload.get('structure_options')
    if lattice_cad_strategy(options) == 'beam' and str(reconstruction_strategy).lower() != 'amrto':
        from .cad_reconstruction import reconstruct_topopt_cad as reconstruct_beam
        return reconstruct_beam(payload, reconstruction_strategy='beam', return_report=return_report,
                                structure_options=options, member_plan=member_plan,
                                export_steps_dir=export_steps_dir, progress_callback=progress_callback)
    cfg = amrto_config or AMRTOConfig()
    if max_faces is not None and (int(max_faces) != max_faces or max_faces < 1):
        raise ValueError('max_faces must be a positive integer')
    crease = float(cfg.quad_crease_deg if crease_angle_deg is None else crease_angle_deg)
    folder_value = export_steps_dir if export_steps_dir is not None else payload.get('export_steps_dir')
    folder = Path(folder_value) if folder_value is not None else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    stages = {'configuration': {**asdict(cfg), 'paper_pipeline': True}}
    started = stage_started = time.perf_counter()

    def progress(stage, completed=0, total=0):
        if callable(progress_callback):
            progress_callback({'type': 'topopt_progress', 'status_only': True,
                               'stage': f'CAD reconstruction — {stage}'}, None, completed, total)

    def finish(name, data):
        nonlocal stage_started
        now = time.perf_counter()
        stages[name] = {**data, 'elapsed_s': now-stage_started}
        stages.setdefault('timings_s', {})[name] = now-stage_started
        stages['timings_s']['total'] = now-started
        stage_started = now
        logger.info('AMRTO %s completed in %.3f s', name, stages[name]['elapsed_s'])
        if folder:
            (folder/'amrto_stage_reports.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')

    # 1. Isosurface/triangular mesh and the paper's routine smoothing.
    progress('Step 1/5 — Preparing triangular mesh')
    supplied = _extract_supplied_quad_mesh(payload)
    recovered = _extract_amrto_design_mesh(payload)
    if recovered is None and supplied is not None:
        recovered = supplied[0], triangulate_quad_faces(*supplied)
    if recovered is not None:
        vertices, triangles = recovered
        if repair_mesh:
            vertices, triangles = preprocess_and_repair_mesh(vertices, triangles, laplacian_iters=0)
        step1 = {'source': 'Supplied triangular boundary; already smoothed'}
        if isinstance(payload.get('recovered_shape'), dict) and payload['recovered_shape'].get('amrto_design_surface'):
            step1.update(source='Smoothed design isosurface before prescribed-geometry restoration',
                         exact_non_design_restoration_stage='Downstream CAD solid assembly',
                         engineering_extension='Analytic prescribed regions are restored after NURBS fitting')
    else:
        # A completed topology solve explicitly records that recovery was
        # attempted. Re-contouring its raw field here would bypass the source
        # and bore constraints which rejected that manufactured boundary.
        # Standalone pseudo-density callers have no such failed recovery and
        # retain the paper's direct-density entry point below.
        if payload.get('surface_recovery_requested'):
            raise SmoothCadUnavailable(
                'The topology result has no valid recovered surface. Regenerate '
                'the constrained topology surface and resolve its recovery error '
                'before CAD reconstruction; raw density cannot bypass failed '
                'source-geometry and bore preservation.'
            )
        field = payload.get('recovery_density')
        cutoff = payload.get('recovery_cutoff')
        if field is None:
            field, cutoff = payload.get('density'), payload.get('density_cutoff')
        if field is None:
            raise ValueError('AMRTO requires a recovered mesh or pseudo-density field')
        vertices, triangles, step1 = generate_topopt_triangular_mesh(
            np.asarray(field, dtype=float), cutoff=float(cfg.density_cutoff if cutoff is None else cutoff),
            bounds=_payload_bounds(payload), config=cfg)
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles, np.int64)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError('AMRTO triangular input must contain finite vertices and faces')
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    step1.update(vertex_count=len(vertices), face_count=len(triangles), is_watertight=bool(mesh.is_watertight))
    if folder:
        _export_stage_mesh(folder, 'step1_mesh', [('triangles', vertices, triangles)])
    finish('step1', step1)

    # 2. One semi-regular pure-quad Instant Meshes reconstruction.
    progress('Step 2/5 — Quadrangulation')
    requested = int(max_faces if max_faces is not None else cfg.target_quad_faces)
    settings = {'target_faces': requested, 'crease_deg': crease, 'paper_pipeline': True}
    fingerprint = geometry_fingerprint(vertices, triangles)
    cached = payload.get('amrto_cached_quadrangulation')
    if supplied is not None:
        quad_vertices, quads = supplied
        step2 = {'quadrangulator': 'Supplied pure quad mesh', 'intermediate_input': True}
    elif (isinstance(cached, dict) and cached.get('report', {}).get('paper_pipeline')
          and cached['report'].get('source_geometry_sha256') == fingerprint
          and cached['report'].get('requested_settings') == settings):
        quad_vertices, quads = np.asarray(cached['vertices'], float), np.asarray(cached['quads'], np.int64)
        step2 = {**cached['report'], 'reused_paper_quadrangulation': True}
    else:
        quad_vertices, quads, step2 = quadrangulate_paper_mesh(
            vertices, triangles, replace(cfg, target_quad_faces=requested, quad_crease_deg=crease))
    step2.update(paper_pipeline=True, source_geometry_sha256=fingerprint, requested_settings=settings)
    if folder:
        _export_stage_mesh(folder, 'step2_quads', [('quads', quad_vertices, quads)])
    finish('step2', step2)

    # 3. Generalized motorcycle graph layouts, including Appendix D topology handling.
    progress('Step 3/5 — Sparse quadrilateral layouts')
    try:
        layouts, step3 = divide_paper_sparse_quad_layouts(
            quad_vertices, quads, layout_folder=payload.get('amrto_layouts_dir'))
    except PaperLayoutError as error:
        # Retain native stderr and topology diagnostics before temporary GMCG
        # files disappear; an executable exit code alone cannot explain failure.
        finish('step3', {**error.report, 'status': 'failed'})
        raise
    if not layouts:
        raise SmoothCadUnavailable('The generalized motorcycle graph produced no layouts')
    if folder:
        _export_stage_mesh(folder, 'step3_layouts', [
            (f'layout_{layout.layout_id}', layout.vertices, layout.quads) for layout in layouts])
    finish('step3', step3)

    # 4. Native Scaffold-Map rectangles, otherwise Algorithm 1; Eqs. 14/15 sampling.
    parameterizations = []
    for index, layout in enumerate(layouts, 1):
        progress(f'Step 4/5 — Parameterization and sampling {index}/{len(layouts)}', index, len(layouts))
        parameterize_paper_layout(layout, cfg)
        adaptive_sample_layout(layout, k_sample=cfg.sampling_rate_k, d_min=cfg.sampling_min_d,
                               d_max=cfg.sampling_max_d)
        parameterizations.append({'layout_id': int(layout.layout_id),
                                  'sampled_shape': list(layout.sampled_shape),
                                  **layout.parameterization_report})
    step4 = {'method': 'Section 2.6; Algorithm 1; Eqs. 14 and 15',
             'patch_count': len(layouts), 'parameterizations': parameterizations}
    if folder:
        step4.update(_sampled_layout_checkpoints(layouts, folder))
    finish('step4', step4)

    # Section 2.7's stitching needs consistent physical curves at T-junctions.
    # Fit those curves once; do not re-partition or couple a tangent network.
    source_scale = float(np.max(np.ptp(vertices, axis=0)))
    fitting_cfg = cfg
    boundary_rejection = None
    if cfg.stitch_surface_boundaries:
        progress('Preparing common CAD boundary curves')
        bounds = _payload_bounds(payload)
        bounds = tuple(np.asarray(side, float) for side in bounds) if bounds is not None else None
        boundary_tolerance = max(source_scale * cfg.boundary_fit_relative_tolerance, 1e-9)
        # Preparation can reject a boundary proposal before any surface fit.
        # Retain the clean Stage4 charts so that rejection still produces one
        # complete, independently fitted surface model for the CAD view.
        clean_layouts = deepcopy(layouts)
        try:
            boundary_report = prepare_shared_boundary_curves(
                layouts, boundary_tolerance, bounds=bounds,
                certify_stock_hull=bounds is not None,
                blocked_junction_positions={tuple(map(float, point)) for layout in layouts for point in layout.vertices})
        except ValueError as error:
            layouts = clean_layouts
            fitting_cfg = replace(cfg, stitch_surface_boundaries=False)
            boundary_rejection = str(error)
            boundary_report = {**getattr(error, 'report', {}), 'accepted': False,
                               'reason': boundary_rejection,
                               'output': 'One independent fit per original Stage4 chart; solid construction incomplete'}
        boundary_report['engineering_extension'] = True
        if boundary_rejection is None:
            for layout in layouts:
                layout.source_box_bounds = bounds
                layout.solid_fit_tolerance = boundary_tolerance
                layout.solid_source_fit_limit = max(source_scale * float(maximum_relative_deviation), 1e-12)
        finish('solid_boundary_curves', boundary_report)

    # 5. Independent chord-length bicubic least squares, then Algorithms 2 and 3.
    fitted = []
    for index, layout in enumerate(_fit_paper_layouts(layouts, fitting_cfg), 1):
        fitted.append(layout)
        progress(f'Step 5/5 — Fitting NURBS {index}/{len(layouts)}', index, len(layouts))
    shape = _assemble_paper_cad(fitted, _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance))
    control_count = sum(layout.nurbs_surface.ctrlpts_size_u*layout.nurbs_surface.ctrlpts_size_v for layout in fitted)
    step5 = {'method': ('Section 2.7 fitting with common seams and bounded source constraints; '
                       'Section 3.1 Algorithms 2 and 3' if fitting_cfg.stitch_surface_boundaries else
                       'Section 2.7 chord-length least squares; Section 3.1 Algorithms 2 and 3'),
             'solid_fitting_engineering_extension': bool(fitting_cfg.stitch_surface_boundaries),
             'nurbs_patch_count': len(fitted), 'control_point_count': control_count,
             'fitting_attempts': 1, 'refinement_passes': 1, 'patch_partition_unchanged': True,
             'patch_reports': [{'layout_id': int(layout.layout_id), **layout.fitting_report,
                                'multiresolution': layout.multiresolution_report} for layout in fitted]}
    if folder:
        export_cad_model(shape, export_path=folder/'step5_cad.step', require_solid=False)
        step5['cad_checkpoint_file'] = 'step5_cad.step'
        step5['cad_checkpoint_current'] = True
        step5['animation_mesh'] = _export_cad_animation_mesh(
            folder, 'step5_cad', shape, max(float(np.linalg.norm(np.ptp(vertices, axis=0)))*1e-3, 1e-7))
    finish('step5', step5)
    progress('Stitching CAD and restoring exact non-design geometry')
    reference_volume = abs(float(mesh.volume)) if mesh.is_watertight and mesh.is_winding_consistent else None
    fit_warnings = []
    if boundary_rejection is not None:
        fit_warnings.append(f'Common CAD boundary preparation: {boundary_rejection}')
    rejected = [int(layout.layout_id) for layout in fitted
                if cfg.stitch_surface_boundaries
                and not layout.fitting_report.get('source_fit_acceptance', {}).get('accepted', False)]
    if rejected:
        fit_warnings.append(f'Fitted surfaces had localized deviation/orientation warnings (layouts {", ".join(map(str, rejected))})')

    effective_sew_tol = _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance)
    intrinsic_preflight = []
    if rejected:
        from .amrto_intrinsic import preflight_layout_intrinsic_fold
        progress('Checking rejected fitted patches before native solid operations')
        for layout in fitted:
            if int(layout.layout_id) not in rejected:
                continue
            check = preflight_layout_intrinsic_fold(layout, tolerance=min(effective_sew_tol, 1e-4))
            intrinsic_preflight.append(check)
            if check['definitive_fold']:
                break
    if any(check['definitive_fold'] for check in intrinsic_preflight):
        solid_report = {'status': 'surface_only', 'is_solid': False, 'fitting_attempts': 0,
            'reason': 'A rejected fitted patch has an intrinsic C1 surface self-intersection',
            'intrinsic_patch_preflight': intrinsic_preflight,
            'native_solid_operations_skipped_for_proven_intrinsic_fold': True}
    else:
        shape, solid_report = _complete_paper_solid(
            shape, payload, tolerance=min(effective_sew_tol, 1e-4),
            source_scale=source_scale, source_volume=reference_volume,
            maximum_volume_delta=maximum_volume_delta,
            void_axial_margin=void_axial_margin, void_radial_margin=void_radial_margin,
            progress_callback=progress)
        if intrinsic_preflight:
            solid_report['intrinsic_patch_preflight'] = intrinsic_preflight
    finish('solid_integration', solid_report)
    if not solid_report['is_solid'] and rejected and boundary_rejection is None and cfg.stitch_surface_boundaries:
        from .amrto_chart_retry import propose_failed_chart_retry, refine_failed_chart_retry
        retry_report = {'accepted': False, 'engineering_extension': True,
                        'paper_algorithm': False, 'initial_solid_integration': deepcopy(solid_report),
                        'initial_rejected_layout_ids': rejected.copy(),
                        'initial_patch_reports': step5['patch_reports']}
        try:
            progress('Retrying rejected CAD charts on the unchanged source mesh')
            retry_layouts, tangent_edges, proposal = propose_failed_chart_retry(clean_layouts, rejected, cfg)
            retry_report.update(proposal)
            retry_report['refinement_rounds'] = []
            for refinement_round in range(3):
                # Keep pristine source UVs for the next proposal. Shared seam
                # preparation/fitting can change their parameter clocks.
                retry_clean = deepcopy(retry_layouts)
                retry_report['boundary_curves'] = prepare_shared_boundary_curves(
                    retry_layouts, boundary_tolerance, bounds=bounds, certify_stock_hull=bounds is not None,
                    blocked_junction_positions={tuple(map(float, point)) for layout in retry_layouts for point in layout.vertices},
                    tangent_preserving_edges=tangent_edges)
                for layout in retry_layouts:
                    layout.source_box_bounds = bounds
                    layout.solid_fit_tolerance = boundary_tolerance
                    layout.solid_source_fit_limit = max(source_scale * float(maximum_relative_deviation), 1e-12)
                retry_fitted = []
                for index, layout in enumerate(_fit_paper_layouts(retry_layouts, cfg), 1):
                    retry_fitted.append(layout)
                    progress(f'Refitting CAD charts {index}/{len(retry_layouts)}', index, len(retry_layouts))
                retry_rejected = [int(layout.layout_id) for layout in retry_fitted
                                  if not layout.fitting_report.get('source_fit_acceptance', {}).get('accepted', False)]
                retry_report['attempts'] = refinement_round + 1
                retry_report['rejected_layout_ids'] = retry_rejected
                retry_report['patch_reports'] = [
                    {'layout_id': int(layout.layout_id), **layout.fitting_report,
                     'multiresolution': layout.multiresolution_report} for layout in retry_fitted]
                if not retry_rejected or refinement_round == 2:
                    break
                progress('Subdividing rejected CAD charts while retaining every source quad')
                retry_layouts, added_edges, refinement = refine_failed_chart_retry(
                    retry_clean, retry_rejected, cfg, round_index=refinement_round+1)
                retry_report['refinement_rounds'].append(refinement)
                if not refinement['applied']:
                    break
                tangent_edges.update(added_edges)
                if refinement['subdivision']['applied']:
                    retry_report['patch_partition_unchanged'] = False
            if retry_rejected:
                retry_report['reason'] = 'The bounded chart retry did not pass every original source-fit check'
            else:
                retry_surface = _assemble_paper_cad(retry_fitted, effective_sew_tol)
                if folder:
                    try:
                        candidate_file = folder/'retry_fitted_surfaces.brep'
                        retry_surface.exportBrep(str(candidate_file))
                        retry_report['candidate_checkpoint_file'] = candidate_file.name
                        retry_report['candidate_checkpoint_is_export_ready_solid'] = False
                    except MemoryError:
                        raise
                    except Exception as error:
                        retry_report['candidate_checkpoint_error'] = str(error)
                retry_shape, retry_solid = _complete_paper_solid(
                    retry_surface, payload, tolerance=min(effective_sew_tol, 1e-4),
                    source_scale=source_scale, source_volume=reference_volume,
                    maximum_volume_delta=maximum_volume_delta,
                    void_axial_margin=void_axial_margin, void_radial_margin=void_radial_margin,
                    progress_callback=progress)
                retry_report['solid_integration'] = retry_solid
                if retry_solid['is_solid']:
                    shape, solid_report, fitted = retry_shape, retry_solid, retry_fitted
                    rejected, fit_warnings = [], []
                    retry_report['accepted'] = True
                    control_count = sum(layout.nurbs_surface.ctrlpts_size_u * layout.nurbs_surface.ctrlpts_size_v
                                        for layout in fitted)
                    step4 = {**step4, 'local_chart_retry': True, 'patch_count': len(fitted), 'parameterizations': [
                        {'layout_id': int(layout.layout_id), 'sampled_shape': list(layout.sampled_shape),
                         **layout.parameterization_report} for layout in fitted]}
                    step5 = {**step5, 'fitting_attempts': 1+retry_report['attempts'],
                             'refinement_passes': 1+retry_report['attempts'], 'nurbs_patch_count': len(fitted),
                             'patch_partition_unchanged': retry_report['patch_partition_unchanged'],
                             'control_point_count': control_count, 'patch_reports': retry_report['patch_reports']}
                    if folder:
                        # Checkpoint errors do not reverse an already validated
                        # geometry decision or reinstate stale stage metadata.
                        step4 = {key: value for key, value in step4.items()
                                 if key in ('method', 'patch_count', 'parameterizations', 'local_chart_retry')}
                        step5.pop('animation_mesh', None)
                        step5.update(cad_checkpoint_file='step5_retry_cad.step', cad_checkpoint_current=False)
                        try:
                            step4.update(_sampled_layout_checkpoints(fitted, folder))
                            export_cad_model(retry_surface, export_path=folder/'step5_retry_cad.step', require_solid=False)
                            step5['cad_checkpoint_current'] = True
                            step5['animation_mesh'] = _export_cad_animation_mesh(
                                folder, 'step5_cad', retry_surface, max(float(np.linalg.norm(np.ptp(vertices, axis=0)))*1e-3, 1e-7))
                        except MemoryError:
                            raise
                        except Exception as error:
                            retry_report['checkpoint_error'] = str(error)
                            logger.warning('Accepted chart retry checkpoint could not be refreshed: %s', error)
                    # Preserve first-attempt stage timings. The complete retry
                    # has its own timer, rather than attributing its native
                    # assembly and fitting time to parameterization.
                    for name, data in (('step4', step4), ('step5', step5)):
                        stages[name] = {**data, 'elapsed_s': stages[name]['elapsed_s'],
                                        'timing_scope': 'Initial attempt; retry work is recorded in local_chart_retry'}
                else:
                    retry_report['reason'] = retry_solid.get('reason', 'Chart retry did not produce an accepted solid')
        except MemoryError:
            raise
        except Exception as error:
            retry_report['reason'] = str(error)
            logger.warning('AMRTO bounded chart retry rejected: %s', error)
        finish('local_chart_retry', retry_report)
    if fit_warnings:
        solid_report['fit_warnings'] = fit_warnings
        if rejected:
            solid_report['rejected_layout_ids'] = rejected
    stages['solid_integration'] = {**solid_report,
        'elapsed_s': stages['solid_integration']['elapsed_s'],
        'timing_scope': 'Initial attempt; any retry work is recorded in local_chart_retry'}
    solid_ready = bool(solid_report['is_solid'])
    grouping_report = step3
    accepted_retry = stages.get('local_chart_retry', {})
    if accepted_retry.get('accepted') and not accepted_retry.get('patch_partition_unchanged', True):
        grouping_report = {
            'method': 'Native sparse layouts with bounded engineering source-quad subdivision',
            'engineering_extension': True, 'paper_stage_3': False,
            'initial_paper_layout_count': len(layouts), 'layout_count': len(fitted),
            'initial_paper_stage3': step3,
            'refinement_rounds': accepted_retry['refinement_rounds'],
            'source_quad_coverage_exactly_once': True,
            'final_source_parents': [{'layout_id': int(layout.layout_id),
                'source_parent_id': int(getattr(layout, 'source_chart_parent_id', layout.layout_id)),
                'subdivision_generation': int(getattr(layout, 'source_chart_subdivision_generation', 0))}
                for layout in fitted]}
        if folder:
            try:
                _export_stage_mesh(folder, 'retry_layouts', [
                    (f'layout_{layout.layout_id}', layout.vertices, layout.quads) for layout in fitted])
            except MemoryError:
                raise
            except Exception as error:
                grouping_report['checkpoint_error'] = str(error)
        stages['effective_layout_grouping'] = grouping_report
    display_layout = layout_display_data(quad_vertices, quads, fitted,
        reports={'quadrangulation': step2, 'grouping': grouping_report})
    display_source = _extract_recovered_mesh(payload)
    if display_source is None:
        display_source = vertices, triangles
    display_layout.update(source_geometry_sha256=geometry_fingerprint(*display_source),
                          design_geometry_sha256=fingerprint, requested_settings=settings)
    add_source_groups(display_layout, *display_source)
    report = {
        'method': 'AMRTO five-step reconstruction (Ren et al., 2025)', 'paper_pipeline': True,
        'backend': 'in-process', 'revision': CAD_RECONSTRUCTION_REVISION,
        'requested_strategy': str(reconstruction_strategy), 'smooth': True, 'editable': True,
        'valid': bool(shape.isValid()), 'is_solid': solid_ready,
        'cad_volume': solid_report.get('cad_volume'), 'cad_face_count': len(shape.Faces()),
        'cad_surface_types': {kind: sum(face.geomType() == kind for face in shape.Faces())
                              for kind in {face.geomType() for face in shape.Faces()}}, 'step3_layouts': len(layouts),
        'effective_fitting_layouts': len(fitted),
        'control_point_count': control_count, 'step5_nurbs_faces': len(fitted),
        'validation_status': 'passed' if solid_ready else 'incomplete',
        'export_ready': solid_ready, 'display_ready': True,
        'qa_geometry_source': 'Stitched fitted NURBS with original CAD non-design interfaces' if solid_ready else 'Fitted NURBS surfaces',
        'warnings': [] if solid_ready else [solid_report.get('reason', 'Solid CAD construction is incomplete')],
        'stages': stages, 'elapsed_seconds': time.perf_counter()-started,
    }
    if not solid_ready and failure_checkpoint_dir is not None:
        _write_incomplete_cad_checkpoint(failure_checkpoint_dir, vertices, triangles, shape, report)
    if folder:
        (folder/'cad_stage_reports.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['_surface_layout'] = display_layout
    workplane = cq.Workplane().newObject([shape])
    return (workplane, report) if return_report else workplane


__all__ = ['SmoothCadUnavailable', 'build_topopt_beam_lattice', 'reconstruct_topopt_cad']
