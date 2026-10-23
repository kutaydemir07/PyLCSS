# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""On-demand CalculiX analysis of the completed TopOpt CAD solid."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from ..geometry.lattice_cad import topopt_export_formats
from ..configuration.fea_meshing import CadFeaMeshOptions, FEA_QUALITY_OPTIONS
from ._common import cad_tet_mesh, validate_boundary_definitions, validated_cad_shape


def _lattice_input_fingerprint(geometry_hash, material, supports, loads, options, surface_tolerance=0.):
    """Only reuse a response when geometry, material, BCs and meshing agree."""
    import hashlib
    import json
    from collections.abc import Mapping
    def canonical(value):
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, np.ndarray):
            return canonical(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, float, int, bool)):
            return value
        shape = value.val() if callable(getattr(value, 'val', None)) else value
        if callable(getattr(shape, 'exportBrep', None)):
            from ..geometry.cad_fingerprint import cad_shape_fingerprint
            return {'brep_sha256': cad_shape_fingerprint(shape)}
        raise TypeError('Unsupported analysis input for response reuse')
    try:
        settings = options.as_dict()
        settings.pop('element_order')  # Same corner mesh can be promoted to quadratic.
        backend=options.solver_backend
        if backend=='Automatic':
            from pylcss.cae.solver_backends.sparse_static import sparse_static_available
            backend='Sparse Direct' if sparse_static_available() else 'CalculiX'
        data = canonical(['pylcss-topopt-linear-fea-v2',backend,geometry_hash, material, supports, loads, settings, surface_tolerance])
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, RuntimeError):
        return None  # Unsupported inputs disable reuse, never weaken the key.


def is_lattice_analysis(result):
    return bool(result.get('generated_lattice') or result.get('study_kind') == 'post_topopt_lattice')


def topopt_fea_unavailable_reason(result):
    """Share the geometry eligibility check between the action and the solver."""
    if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
        return "Run TopOpt"
    if str(result.get("physics_mode") or "structural").lower() != "structural":
        return "Unsupported study physics"
    if (result.get("multibody") or {}).get("global_joints"):
        return "Connector analysis required"
    if is_lattice_analysis(result):
        surface = result.get('recovered_shape')
        if not isinstance(surface, dict) or any(
            surface.get(key) is None or not np.asarray(surface[key]).size for key in ('vertices', 'faces')
        ):
            return 'Generate the lattice surface first'
        return ''
    if "STEP" not in topopt_export_formats(result):
        return "STL lattice"
    if result.get("cad_shape") is None and result.get("shape") is None:
        return "Build CAD"
    report = result.get("cad_reconstruction") or {}
    if report.get("paper_pipeline") and report.get("is_solid") is False:
        return "CAD surfaces need stitching into a solid"
    return ""


def run_topopt_fea(
    result,
    material,
    supports,
    loads,
    *,
    quality="Standard",
    mesh_options=None,
    progress_callback=None,
    cancel_callback=None,
    initial_analysis=None,
):
    """Solve the current CAD or resolved lattice, using geometric boundary conditions."""
    from pylcss.cae.solver_backends import ExternalRunConfig, run_calculix_static

    reason = topopt_fea_unavailable_reason(result)
    if reason:
        raise ValueError(f"CAD FEA unavailable: {reason}.")
    material, supports, loads = validate_boundary_definitions(
        material,
        supports,
        loads,
        context="CAD FEA",
    )
    if quality not in FEA_QUALITY_OPTIONS:
        raise ValueError("Choose Standard or Mesh Convergence FEA quality.")
    options = CadFeaMeshOptions(**(mesh_options or {}))

    def cancelled():
        return cancel_callback is not None and cancel_callback()

    if cancelled():
        return None
    lattice = is_lattice_analysis(result)
    target = 'Lattice material' if lattice else 'CAD'
    if lattice:
        from ..geometry.lattice_tet_mesh import lattice_tet_mesh
        shape = result['recovered_shape']
        spans = np.ptp(np.asarray(shape['vertices'], dtype=float), axis=0)
    else:
        shape = result.get("cad_shape")
        if shape is None:
            shape = result.get("shape")
        shape = validated_cad_shape(shape)
        bounds = shape.BoundingBox()
        spans = np.array([bounds.xlen, bounds.ylen, bounds.zlen], dtype=float)
    size, minimum_size = options.resolve_sizes(spans)
    surface_tolerance = options.surface_tolerance_mm
    if lattice and surface_tolerance == 0.:
        generated = result.get('generated_lattice') or {}
        minimum_member = float(generated.get('minimum_member_thickness_mm') or generated.get('min_thickness_mm') or 0.)
        recovery = shape.get('surface_recovery_report') or {}
        spacing = recovery.get('sampling_spacing')
        # Reserve a small fraction of the physical wall and contour resolution
        # for the analysis approximation; retain numerical-only preparation
        # when provenance is absent. The exported source remains unchanged.
        if minimum_member > 0 and spacing is not None and np.asarray(spacing).size == 3:
            surface_tolerance = min(.02*minimum_member, .1*float(np.min(spacing)), .05*size)
    levels = []
    started = perf_counter()
    input_hash = geometry_hash = None
    base_mesh = None
    reuse_response = False
    if initial_analysis is None:
        initial_analysis=result.get('validation')
    if lattice:
        from ..geometry.lattice_tet_mesh import lattice_surface_fingerprint, reuse_lattice_mesh, stress_refinement_elements
        geometry_hash = lattice_surface_fingerprint(shape)
    else:
        from ..geometry.cad_fingerprint import cad_shape_fingerprint
        geometry_hash=cad_shape_fingerprint(shape)
    input_hash = _lattice_input_fingerprint(geometry_hash, material, supports, loads, options, surface_tolerance)
    if (input_hash is not None and isinstance(initial_analysis, dict)
                and initial_analysis.get('analysis_input_sha256') == input_hash
                and initial_analysis.get('analysis_geometry_sha256') == geometry_hash
                and initial_analysis.get('source') == ('topopt_lattice_fea' if lattice else 'topopt_cad_fea')
                and len((initial_analysis.get('convergence_study') or {}).get('levels', [])) >= 1):
            possible = initial_analysis.get('mesh')
            mesh_report=getattr(possible,'surface_meshing_report' if lattice else 'cad_meshing_report',{})
            if (possible is not None and mesh_report.get('passed')
                    and (not lattice or mesh_report.get('source_surface_sha256') == geometry_hash)):
                if lattice:base_mesh = possible
                reuse_response = initial_analysis.get('element_order') == options.element_order
    analysis = None
    maximum_levels=4 if quality=='Mesh Convergence' else 1
    reference_size=(initial_analysis['convergence_study']['levels'][-1]['maximum_element_size']
                    if reuse_response else size)
    refinement_stop_reason=None
    with TemporaryDirectory(prefix="pylcss_topopt_cad_fea_") as directory:
        for level in range(maximum_levels):
            if cancelled():
                return None
            level_size = size / (2**level)
            if not lattice:
                level_size=(reference_size*.8**level if reuse_response else
                            size if level==0 else .5*size*.8**(level-1))
            level_minimum = minimum_size*level_size/size
            if level == 0 and reuse_response:
                analysis = dict(initial_analysis)
                entry = dict(initial_analysis['convergence_study']['levels'][-1])
                entry['reused_from_previous_study_level_count']=len(initial_analysis['convergence_study']['levels'])
                entry['refine']=1
                entry.update(response_reused=True, meshing_seconds=0., solve_seconds=0.)
                levels.append(entry)
                continue
            meshing_started = perf_counter()
            if not lattice and level>0 and analysis is not None:
                history=levels if len(levels)>=2 else (initial_analysis or {}).get('convergence_study',{}).get('levels',levels)
                exponent=3.
                if len(history)>=2:
                    older,newer=history[-2:]
                    ratio=older['maximum_element_size']/newer['maximum_element_size']
                    if ratio>1.:
                        exponent=float(np.clip(np.log(newer['n_elements']/older['n_elements'])/np.log(ratio),1.,3.))
                last=levels[-1]
                predicted=int(np.ceil(last['n_elements']*(last['maximum_element_size']/level_size)**exponent))
                if predicted>options.max_elements:
                    refinement_stop_reason=(f'Next CAD refinement is estimated at {predicted:,} elements; '
                        f'limit is {options.max_elements:,}. Current analysis retained without a convergence pass.')
                    break
            if progress_callback is not None:
                progress_callback('Refining lattice stress variations and hotspots' if lattice and level > 0 else
                                  f'Meshing {target}, level {level+1} (maximum size {level_size:.4g} mm)')
            mesher = lattice_tet_mesh if lattice else cad_tet_mesh
            marked = stress_refinement_elements(analysis) if lattice and level > 0 and analysis is not None else None
            try:
                mesh = (reuse_lattice_mesh(base_mesh, element_order=options.element_order,
                                      max_elements=options.max_elements, refine=level > 0, marked_elements=marked)
                    if lattice and base_mesh is not None else mesher(
                shape,
                size if lattice else level_size,
                directory,
                minimum_size=level_minimum,
                grading=options.mesh_grading,
                element_order=options.element_order,
                max_elements=options.max_elements,
                **({'refinement_level': level, 'cancel_callback': cancel_callback} if lattice else {}),
                **({'surface_tolerance_mm': surface_tolerance} if lattice else {}),
                ))
            except ValueError as error:
                if level>0 and analysis is not None and 'elements' in str(error) and ('limit' in str(error) or 'exceed' in str(error)):
                    refinement_stop_reason=str(error)
                    break
                raise
            if lattice and level == 0:
                base_mesh = mesh
            if cancelled():
                return None
            mesh.enable_geometry_selection_cache = True
            if not lattice:
                mesh.geometry_selection_tolerance_mm = max(size*.05, float(np.max(spans))*1e-8)
                mesh.cad_face_selection_by_facets=True
                mesh.cad_face_corner_tolerance_mm=max(float(np.max(spans))*1e-8,1e-6)
            if lattice:
                recovery = shape.get('surface_recovery_report') or {}
                accuracy = recovery.get('envelope_accuracy') or {}
                preparation = mesh.surface_meshing_report['analysis_surface_preparation']
                tolerance = (float(accuracy.get('contour_cell_half_diagonal') or 0.)
                             + float(recovery.get('source_tessellation_tolerance') or 0.)
                             + preparation['maximum_original_vertex_displacement_mm'])
                mesh.geometry_selection_tolerance_mm = max(tolerance, size * 1e-3)
                mesh.surface_meshing_report['CAD_face_selection'] = {
                    'boundary_nodes_only': True,
                    'distance_tolerance_mm': mesh.geometry_selection_tolerance_mm,
                    'basis': 'Recorded contour/tessellation error plus bounded FEA preparation; coordinate conditions remain explicit',
                }
            if progress_callback is not None:
                progress_callback(
                    f"Solving linear solid FEA, level {level + 1} "
                    f"({mesh.p.shape[1]:,} nodes, {mesh.t.shape[1]:,} elements)"
                )
            meshing_seconds = perf_counter() - meshing_started
            solve_started = perf_counter()
            analysis = None
            if options.solver_backend != 'CalculiX':
                from pylcss.cae.solver_backends.sparse_static import run_sparse_static
                analysis = run_sparse_static(mesh, material, supports, loads, cancel_callback=cancel_callback)
                if analysis is None and options.solver_backend == 'Sparse Direct':
                    raise ValueError('Sparse Direct requires MKL PARDISO and supported linear solid loads/constraints. Choose Automatic or CalculiX for other models.')
            if analysis is None:
                analysis = run_calculix_static(
                    mesh, material, supports, loads,
                    ExternalRunConfig(
                        run_solver=True,
                        job_name=f"pylcss_topopt_cad_fea_L{level + 1}",
                        cancel_callback=cancel_callback,
                    ),
                )
            meshing_report = getattr(mesh, 'surface_meshing_report' if lattice else 'cad_meshing_report', {})
            native_warnings = meshing_report.get("native_warnings", [])
            if native_warnings:
                analysis.setdefault("warnings", []).extend(
                    f"{target} mesh, level {level + 1}: {warning}" for warning in native_warnings
                )
            if cancelled():
                return None
            if analysis.get("mesh") is None or any(
                analysis.get(key) is None
                or np.asarray(analysis[key]).size == 0
                or not np.all(np.isfinite(analysis[key]))
                for key in ("stress", "displacement")
            ):
                raise RuntimeError(
                    analysis.get("message")
                    or "CalculiX produced no stress/displacement results."
                )
            from .fea_hotspots import stress_hotspots
            analysis['element_order']=options.element_order
            analysis['stress_hotspots']=stress_hotspots(
                {**analysis,'element_order':options.element_order},max(level_size,1e-12))
            levels.append(
                {
                    "refine": level + 1,
                    "n_nodes": int(mesh.p.shape[1]),
                    "n_elements": int(mesh.t.shape[1]),
                    "maximum_element_size": size if lattice else level_size,
                    'refinement_strategy': ('Stress-guided local conforming refinement' if lattice and level > 0
                                            else 'Native CAD remeshing with smaller elements' if level>0
                                            else 'Initial volume mesh'),
                    "minimum_element_size": level_minimum,
                    **({'surface_meshing': meshing_report} if lattice else
                       {'cad_meshing': getattr(mesh, 'cad_meshing_report', {'backend': 'Netgen'})}),
                    "meshing": meshing_report,
                    "compliance": analysis.get("compliance"),
                    "peak_von_mises": analysis.get(
                        "peak_stress_nodal", analysis.get("max_stress_gauss")
                    ),
                    "solved": True,
                    'meshing_seconds': meshing_seconds,
                    'solve_seconds': perf_counter() - solve_started,
                }
            )
            if lattice:
                base_mesh=mesh
            if level>0:
                previous,current=levels[-2:]
                compliance_pair=[previous.get('compliance'),current.get('compliance')]
                stress_pair=[previous.get('peak_von_mises'),current.get('peak_von_mises')]
                if (all(value is not None for value in compliance_pair+stress_pair)
                        and np.isfinite(compliance_pair+stress_pair).all()
                        and compliance_pair[1]>0 and stress_pair[1]>0
                        and abs(compliance_pair[1]-compliance_pair[0])/compliance_pair[1]<=options.convergence_tolerance
                        and abs(stress_pair[1]-stress_pair[0])/stress_pair[1]<=options.convergence_tolerance):
                    break
    change = None
    stress_change = None
    if len(levels) >= 2:
        previous, current = (entry["compliance"] for entry in levels[-2:])
        if (
            previous is not None
            and current is not None
            and np.isfinite([previous, current]).all()
            and current > 0
        ):
            change = abs(current - previous) / current
        previous, current = (entry['peak_von_mises'] for entry in levels[-2:])
        if previous is not None and current is not None and np.isfinite([previous, current]).all() and current > 0:
            stress_change = abs(current-previous)/current
    converged=(change<=options.convergence_tolerance and stress_change is not None
               and stress_change<=options.convergence_tolerance) if change is not None else None
    if quality=='Mesh Convergence' and not converged and refinement_stop_reason is None:
        refinement_stop_reason='Maximum refinement levels reached without meeting both stress and compliance tolerances.'
    analysis.update(
        analysis_geometry_sha256=geometry_hash,analysis_input_sha256=input_hash,
        buckling_assessed=False,fatigue_assessed=False,
        validation_scope='Resolved linear-elastic solid stress and displacement; no buckling or fatigue analysis',
        analysis_target="Resolved lattice material" if lattice else "CAD solid",
        source="topopt_lattice_fea" if lattice else "topopt_cad_fea",
        element_order=options.element_order,
        mesh_settings={**options.as_dict(), "resolved_element_size_mm": size,
                       'resolved_surface_tolerance_mm': surface_tolerance if lattice else None},
        elapsed_seconds=perf_counter() - started,
        convergence_study={
            "element_order": options.element_order,
            "levels": levels,
            'maximum_levels':maximum_levels,
            'stop_reason':refinement_stop_reason,
            "tolerance": options.convergence_tolerance,
            "compliance_rel_change": change,
            "stress_rel_change": stress_change,
            'required_metrics':['compliance','peak_von_mises'],
            "converged":converged,
        },
    )
    if lattice:
        analysis.update(analysis_geometry_sha256=geometry_hash, analysis_input_sha256=input_hash,
                        buckling_assessed=False, fatigue_assessed=False,
                        validation_scope='Resolved linear-elastic lattice stress and displacement; no buckling or fatigue analysis')
    return analysis
