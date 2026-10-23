# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded reinforcement proposals driven by resolved lattice FEA.

This is a generate/solve/redesign search, not a homogenized material law or a
minimum-mass optimizer. Neither density nor cell refinement is assumed to
improve strength: each changed geometry must be analysed independently.
"""
from __future__ import annotations

import numpy as np
from time import perf_counter
from scipy.ndimage import maximum_filter


def stress_reinforcement_proposal(analysis, density, bounds, *, allowable_stress,
                                   minimum_density, maximum_density, nominal_pitch,
                                   design_mask, previous_refinement=None,
                                   displacement_limit=0.):
    """Map actual microstress to physical sizing cells and propose reinforcement."""
    density = np.asarray(density, dtype=float)
    design = np.asarray(design_mask, dtype=bool)
    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    points = np.asarray(analysis['mesh'].p, dtype=float).T
    stress = np.asarray(analysis['stress'], dtype=float).reshape(-1)
    if (analysis.get('source') != 'topopt_lattice_fea'
            or density.ndim != 3 or design.shape != density.shape
            or points.shape != (len(stress), 3) or not np.isfinite(stress).all()
            or np.any(stress < 0) or allowable_stress <= 0):
        raise ValueError('Reinforcement needs resolved nodal microstress and a matching physical sizing grid.')
    spacing = (upper-lower)/np.asarray(density.shape)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Lattice reinforcement needs finite increasing physical bounds.')
    rounding = np.max(upper-lower) * 1e-7
    if np.any(points < lower-rounding) or np.any(points > upper+rounding):
        raise ValueError('Resolved lattice stress coordinates fall outside the physical sizing bounds.')
    indices = np.floor((points-lower)/spacing).astype(np.int64)
    indices = np.clip(indices, 0, np.asarray(density.shape)-1)
    demand = np.zeros(density.shape, dtype=float)
    np.maximum.at(demand, tuple(indices.T), stress/allowable_stress)
    # A maximum over a half-cell neighbourhood spreads a hotspot to the
    # material controlled by that cell. It never averages down a peak stress.
    radius = np.maximum(1, np.ceil(.5*nominal_pitch/spacing).astype(int))
    demand = maximum_filter(demand, size=tuple(2*radius+1), mode='nearest')
    displacement = float(analysis.get('peak_displacement') or 0.)
    displacement_utilization = displacement/displacement_limit if displacement_limit > 0 else 0.
    drive = np.maximum(demand, displacement_utilization)
    # Damped numerical search step only; sqrt(utilization) is NOT presented
    # as a calibrated stress-density law. Reanalysis decides its usefulness.
    multiplier = np.where(drive > 1., np.clip(np.sqrt(np.maximum(drive, 1.)), 1.05, 1.5), 1.)
    proposed = density.copy()
    proposed[design] = np.clip(density[design]*multiplier[design], minimum_density, maximum_density)
    refinement = (demand > 1.) & design
    if previous_refinement is not None:
        previous = np.asarray(previous_refinement, dtype=bool)
        if previous.shape != density.shape:
            raise ValueError('Previous refinement mask does not match the sizing grid.')
        refinement |= previous
    return proposed, refinement, {
        'method': 'Conservative physical-bin nodal stress maxima and bounded reinforcement proposals',
        'stress_source': 'Resolved lattice material, not original solid-envelope stress',
        'maximum_stress_utilization': float(stress.max(initial=0.)/allowable_stress),
        'displacement_utilization': displacement_utilization,
        'changed_density_voxels': int(np.count_nonzero(proposed[design] > density[design]+1e-12)),
        'requested_refined_voxels': int(np.count_nonzero(refinement)),
        'minimum_member_thickness_changed': False,
        'minimum_overstressed_density_multiplier': 1.05,
        'optimal_cell_size_claimed': False, 'candidate_reanalysis_required': True,
    }


def generate_stress_guided_lattice(result, settings, *, cancel_callback=None, progress_callback=None):
    from .post_lattice import _generate_lattice_once, topology_source
    from ..verification.fea_analysis import run_topopt_fea
    from ..verification._common import validate_boundary_definitions, finite_nonnegative, validation_integer
    from ..verification.quality_gate import update_verification_status
    from ..geometry.lattice_cache import LatticeGeometryCache

    inputs = settings.get('_mechanical_inputs') or {}
    material, supports, loads = validate_boundary_definitions(inputs.get('material'), inputs.get('supports'),
        inputs.get('loads'), context='FEA-guided lattice sizing')
    yield_strength = finite_nonnegative(material.get('yield_strength', material.get('yield_stress', 0.)),
                                       name='Lattice material yield strength', positive=True)
    material['yield_strength'] = yield_strength
    factor = finite_nonnegative(inputs.get('safety_factor', 1.), name='Lattice yield safety factor', positive=True)
    if factor < 1:
        raise ValueError('Lattice yield safety factor must be at least 1.')
    displacement_limit = finite_nonnegative(inputs.get('displacement_limit', 0.), name='Lattice displacement limit')
    count = settings.get('post_lattice_sizing_iterations')
    iterations = validation_integer(3 if count is None else count, name='Lattice sizing iterations')
    source = topology_source(result)
    bounds = (source['bounds']['min'], source['bounds']['max'])
    relative, refinement, best = None, None, None
    previous_relative, previous_refinement = None, None
    best_rank = (-1, -np.inf)
    geometry_cache = LatticeGeometryCache()
    started = perf_counter()
    history, status = [], 'iteration_limit'
    errors = []
    # The thin-wall probes showed large C3D4 stress errors. Keep the existing
    # quadratic default; try linear guidance only when the user selected it.
    use_preliminary = (inputs.get('mesh_options') or {}).get('element_order', 'C3D10') == 'C3D4'

    def progress(label):
        if callable(cancel_callback) and cancel_callback():
            raise InterruptedError('FEA-guided lattice sizing cancelled.')
        if callable(progress_callback):
            progress_callback({'status_only': True, 'stage': label}, None, 0, 0)

    def utilization(analysis):
        return (float(analysis['peak_stress_nodal'])/(yield_strength/factor),
                float(analysis['peak_displacement'])/displacement_limit if displacement_limit > 0 else 0.)

    def alternative_after_failure(record, iteration):
        nonlocal relative, refinement
        if iteration+1 >= iterations or previous_relative is None:
            return False
        changed_refinement = refinement is not None and np.any(refinement) and (
            previous_refinement is None or not np.array_equal(refinement, previous_refinement))
        if changed_refinement:
            refinement = None if previous_refinement is None else previous_refinement.copy()
            record['next_geometry_variant'] = 'Reinforced density with the previous cell layout; new cell refinement omitted'
            return True
        if relative is not None and np.any(np.abs(relative-previous_relative) > 1e-12):
            relative = .5*(relative+previous_relative)
            record['next_geometry_variant'] = 'Half reinforcement step within the same density/member bounds'
            return True
        return False

    def analyse(candidate, record, stage, *, order='C3D4', previous=None, convergence=False):
        options = {**(inputs.get('mesh_options') or {}), 'element_order': order}
        analysis = run_topopt_fea(candidate, material, supports, loads,
            quality='Mesh Convergence' if convergence else 'Standard', mesh_options=options,
            initial_analysis=previous, cancel_callback=cancel_callback,
            progress_callback=lambda label: progress(f'Candidate {record["iteration"]}, {stage}: {label}'))
        if analysis is None:
            raise InterruptedError('FEA-guided lattice sizing cancelled.')
        stress_util, disp_util = utilization(analysis)
        record['analyses'].append(dict(stage=stage, element_order=order,
            seconds=analysis['elapsed_seconds'], peak_stress_mpa=float(analysis['peak_stress_nodal']),
            n_elements=int(analysis['mesh'].t.shape[1]), n_nodes=int(analysis['mesh'].p.shape[1]),
            mesh_quality=getattr(analysis['mesh'], 'quality_report', {}),
            equilibrium_error_pct=analysis.get('equilibrium_error_pct'),
            mesh_levels=analysis['convergence_study']['levels']))
        record.update(resolved_lattice_FEA_completed=True, maximum_stress_utilization=stress_util,
                      displacement_utilization=disp_util, peak_stress_mpa=float(analysis['peak_stress_nodal']),
                      peak_displacement_mm=float(analysis['peak_displacement']), element_order=order,
                      force_equilibrium_balanced=bool(analysis.get('equilibrium_balanced')))
        candidate['validation'] = analysis
        candidate['fea_analysis_status'] = 'completed'
        # This status requires convergence, so preliminary linear or quadratic
        # responses alone cannot mark a candidate verified.
        update_verification_status(candidate, material=material, safety_factor=factor,
                                   displacement_limit=displacement_limit)
        progress(f'Candidate {record["iteration"]}, {stage}: stress {analysis["peak_stress_nodal"]:.4g} MPa '
                 f'(allowable {yield_strength/factor:.4g} MPa)')
        return analysis

    from .homogenized_sizing import pre_size_homogenized_lattice
    try:
        relative, refinement, homogenized_report = pre_size_homogenized_lattice(
            source, settings, material, supports, loads, geometry_cache=geometry_cache,
            allowable_stress=yield_strength/factor, displacement_limit=displacement_limit,
            iterations=iterations, progress=progress)
    except InterruptedError:
        raise
    except Exception as error:
        relative, refinement = None, None
        homogenized_report = dict(performed=False, reason=str(error),
                                 fallback='Resolved material workflow retained', resolved_lattice_strength_verified=False)

    for iteration in range(iterations):
        progress(f'FEA-guided lattice: building candidate {iteration+1}/{iterations}')
        geometry_started = perf_counter()
        record = {'iteration': iteration+1, 'analyses': []}
        history.append(record)
        try:
            candidate = _generate_lattice_once(source, settings, cancel_callback=cancel_callback,
                progress_callback=progress_callback, relative_density_override=relative,
                stress_refinement_mask=refinement, geometry_cache=geometry_cache)
        except InterruptedError:
            raise
        except Exception as error:
            if best is None:
                raise
            record['geometry_error'] = str(error)
            errors.append(str(error))
            status = 'geometry_candidate_rejected'
            if alternative_after_failure(record, iteration):
                continue
            break
        if best is None:
            best = candidate
        record.update(volume_mm3=float(candidate['volume']), geometry_seconds=perf_counter()-geometry_started)
        try:
            analysis = analyse(candidate, record, 'preliminary solid FEA' if use_preliminary else 'quadratic solid FEA',
                               order='C3D4' if use_preliminary else 'C3D10')
            # Linear tetrahedra guide changes only. Always check the last
            # candidate with quadratic elements, even if it misses the target.
            if (use_preliminary and (iteration == 0 or max(utilization(analysis)) <= 1. or iteration == iterations-1)
                    and analysis.get('equilibrium_balanced')):
                preliminary = analysis
                analysis = analyse(candidate, record, 'quadratic solid FEA', order='C3D10', previous=analysis)
                differences = {key: abs(float(analysis[key])-float(preliminary[key]))/max(abs(float(analysis[key])), 1e-30)
                               for key in ('peak_stress_nodal', 'peak_displacement', 'compliance')}
                record['linear_quadratic_relative_differences'] = differences
                # A lower-order model is useful only if this measured comparison
                # agrees within the requested accuracy. Thin TPMS walls commonly
                # fail this test; retain quadratic solves in that case.
                tolerance = float((inputs.get('mesh_options') or {}).get('convergence_tolerance', .05))
                if max(differences.values()) > tolerance:
                    use_preliminary = False
                    record['preliminary_FEA_disabled'] = 'Linear/quadratic disagreement exceeds the configured convergence tolerance'
            if (analysis.get('element_order') == 'C3D10' and max(utilization(analysis)) <= 1.
                    and analysis.get('equilibrium_balanced')):
                progress('Checking lattice stress and compliance mesh convergence')
                analysis = analyse(candidate, record, 'quadratic mesh convergence', order='C3D10',
                                   previous=analysis, convergence=True)
                record['convergence_study'] = analysis['convergence_study']
            rank = (int(analysis['element_order'] == 'C3D10') + int(bool(candidate['verification_passed'])),
                    -max(utilization(analysis)))
            if rank > best_rank:
                best, best_rank = candidate, rank
            if not analysis.get('equilibrium_balanced'):
                status = 'force_equilibrium_not_demonstrated'
                break
            if candidate['verification_passed']:
                status = 'linear_static_targets_and_mesh_checks_passed'
                break
            if (analysis.get('convergence_study') or {}).get('converged') is False:
                status = 'mesh_convergence_not_demonstrated'
                break
            if iteration == iterations-1:
                break
            previous_relative = candidate['recovery_density'].copy()
            previous_refinement = None if refinement is None else refinement.copy()
            relative, new_refinement, proposal = stress_reinforcement_proposal(
                analysis, candidate['recovery_density'], bounds,
                allowable_stress=yield_strength/factor,
                minimum_density=float(candidate['structure_options'].minimum_relative_density),
                maximum_density=float(candidate['structure_options'].maximum_relative_density),
                nominal_pitch=float(candidate['generated_lattice']['cell_size_mm']),
                design_mask=np.asarray(candidate['design_domain']) & ~np.asarray(candidate['passive_solid_mask']),
                previous_refinement=refinement, displacement_limit=displacement_limit)
            record['next_proposal'] = proposal
            if proposal['changed_density_voxels'] == 0 and refinement is not None and np.array_equal(new_refinement, refinement):
                status = 'no_further_reinforcement_within_configured_limits'
                break
            refinement = new_refinement
        except InterruptedError:
            raise
        except Exception as error:
            record['analysis_error'] = str(error)
            errors.append(str(error))
            status = 'analysis_or_convergence_check_failed'
            if alternative_after_failure(record, iteration):
                continue
            break
    best['lattice_optimization'] = {
        'method': 'Bounded FEA-guided reinforcement of explicit lattice geometry', 'status': status,
        'iterations': history, 'minimum_mass_optimum_claimed': False,
        'original_solid_stress_used_as_lattice_strength': False,
        'yield_safety_factor': factor, 'allowable_stress_mpa': yield_strength/factor,
        'displacement_limit_mm': displacement_limit, 'buckling_assessed': False, 'fatigue_assessed': False,
        'verified': bool(best.get('verification_passed')), 'errors': errors,
        'seconds': perf_counter()-started, 'geometry_cache': geometry_cache.report(),
        'homogenized_sizing': homogenized_report,
        'analysis_policy': 'Quadratic solid FEA by default; selected preliminary linear FEA is compared with quadratic FEA and disabled if inaccurate; final verification requires quadratic FEA and mesh convergence',
    }
    best['generated_lattice']['method'] = 'FEA-guided thickness and local cell-refinement candidates, independently reanalysed'
    best['generated_lattice']['fit'].update(sizing_basis='Resolved lattice FEA feedback within geometric/manufacturing bounds',
        load_based_strength_sizing=any(row.get('resolved_lattice_FEA_completed') for row in history))
    if not best.get('verification_passed'):
        best['warnings'].append(f'FEA-guided lattice sizing did not complete its linear-static checks: {status}.')
    return best
