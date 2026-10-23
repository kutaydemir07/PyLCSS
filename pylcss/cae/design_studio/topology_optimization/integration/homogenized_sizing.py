# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Cheap periodic-continuum proposals before constructing explicit lattice cells.

Effective stress divided by density is a solid-phase average, a necessary
yield check rather than a peak microstress prediction. This stage can guide
stiffness/density changes, but cannot pass the resolved-lattice quality gate.
"""
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter, binary_erosion


def _resample(field, shape, *, order=1):
    coordinates = np.meshgrid(*[(np.arange(n)+.5)*old/n-.5 for old, n in zip(field.shape, shape)], indexing='ij')
    return map_coordinates(np.asarray(field, dtype=float), coordinates, order=order, mode='nearest')


def cubic_constitutive_basis():
    c11, c12, c44 = np.zeros((3, 6, 6))
    c11[:3, :3] = np.eye(3)
    c12[:3, :3] = np.ones((3, 3))-np.eye(3)
    c44[3:, 3:] = np.eye(3)
    return np.array([c11, c12, c44])


def _selected_nodes(mesh, record):
    from pylcss.cae.solver_backends.selection import dict_geometries, nodes_matching_geometries, nodes_matching_condition
    geometries = dict_geometries(record)
    if geometries:
        return nodes_matching_geometries(mesh, geometries)
    if record.get('condition'):
        return nodes_matching_condition(mesh, record['condition'])
    raise ValueError('Homogenized sizing needs geometric or coordinate boundary definitions.')


def pre_size_homogenized_lattice(source, settings, material, supports, loads, *,
                                 geometry_cache, allowable_stress, displacement_limit,
                                 iterations, progress, maximum_elements=16000):
    from .post_lattice import prepare_lattice_field, lattice_base_geometry, lattice_design_mask
    from ..manufacturing.cell_material import cell_material_law, solid_cubic_constants
    from ..optimization.pymoto_runtime import import_pymoto
    from ..optimization.matrix_free import MatrixFreeElasticity, kernel_threads
    pym = import_pymoto()
    from pymoto.modules.assembly import get_B

    started = perf_counter()
    report = dict(method='Periodic cubic continuum before explicit geometry', performed=False,
        resolved_lattice_strength_verified=False, peak_microstress_predicted=False, iterations=[])
    def skip(reason):
        report.update(reason=reason, seconds=perf_counter()-started)
        return None, None, report
    base, _ = lattice_base_geometry(source)
    _, values, density, bounds, _, design, solid, void, options = prepare_lattice_field(
        source, settings, base_geometry=base, geometry_cache=geometry_cache)
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    spans = upper-lower
    fine_shape = density.shape
    pitch = float(options.cell_size_voxels*np.mean(spans/np.asarray(fine_shape)))
    report['nominal_cells_across_bounds'] = (spans/pitch).tolist()
    # A one-cell-thick cut lattice has strong boundary effects; retain its
    # resolved workflow instead of pretending it is a periodic bulk solid.
    if float(np.min(spans/pitch)) < 3.:
        return skip('Fewer than three nominal cells across a part dimension; use resolved material FEA.')
    poisson = float(material.get('nu', material.get('poissons_ratio', .3)))
    law = cell_material_law(options.mode, poisson=poisson, allow_build=False)
    if law is None:
        return skip('No cached periodic material tensor for this family/Poisson ratio; no hidden cell-table build started.')
    if any(str(load.get('type', 'force')).lower() != 'force' for load in loads):
        return skip('This coarse proposal stage supports resultant forces; other load types retain resolved FEA.')
    if any(np.any(np.asarray(record.get('displacement', [0., 0., 0.]))) for record in supports):
        return skip('Prescribed nonzero motions retain the resolved workflow.')
    factor = min(1., (maximum_elements/float(np.prod(fine_shape)))**(1/3))
    shape = tuple(max(2, int(np.floor(n*factor))) for n in fine_shape)
    while np.prod(shape) > maximum_elements:
        index = int(np.argmax(shape))
        shape = tuple(n-1 if i == index else n for i, n in enumerate(shape))
    spacing = spans/np.asarray(shape)
    domain = pym.VoxelDomain(*shape, *spacing)
    occupancy = lattice_design_mask(base, shape, bounds, geometry_cache)
    prescribed = _resample(solid, shape, order=0) > .5
    empty = _resample(void, shape, order=0) > .5
    active = (occupancy | prescribed) & ~empty
    if not np.any(active):
        return skip('No retained material on the coarse physical grid.')
    node_ids = domain.conn[active.ravel(order='F')]
    present = np.unique(node_ids)
    boundary_cells = active & ~binary_erosion(active)
    boundary_nodes = np.unique(domain.conn[boundary_cells.ravel(order='F')])
    mesh = SimpleNamespace(p=domain.get_node_position()+lower[:, None],
        boundary_selection_nodes=boundary_nodes,
        geometry_selection_tolerance_mm=float(np.linalg.norm(spacing)/2))
    fixed = []
    for record in supports:
        nodes = np.intersect1d(_selected_nodes(mesh, record), present)
        if not len(nodes):
            return skip('A support is unresolved on the coarse grid; retain its geometric resolved analysis.')
        dofs = np.asarray(record.get('fixed_dofs', [0, 1, 2]), dtype=int)
        if np.any((dofs < 0) | (dofs > 2)):
            raise ValueError('Support components must be X, Y or Z.')
        fixed.extend((nodes[:, None]*3+dofs).ravel())
    fixed = np.unique(fixed).astype(int)
    force = np.zeros(3*domain.nnodes)
    for record in loads:
        nodes = np.intersect1d(_selected_nodes(mesh, record), present)
        vector = np.asarray(record.get('vector'), dtype=float)
        if not len(nodes) or vector.shape != (3,) or not np.isfinite(vector).all():
            return skip('A resultant force is unresolved on the coarse grid.')
        # Same total resultant; the coarse application footprint is approximate
        # and is explicitly superseded by geometric final-load integration.
        for component in range(3):
            force[3*nodes+component] += vector[component]/len(nodes)
    if np.any(np.abs(force[fixed]) > 0):
        return skip('A coarse force overlaps a constrained DOF; retain resolved boundary selection.')
    elasticity = MatrixFreeElasticity(domain, cubic_constitutive_basis(), fixed)
    young = float(material.get('E', 210000.))
    rho = _resample(density, shape)
    rho[prescribed] = 1.; rho[~active] = 0.
    initial = None
    best = rho.copy(); best_utilization = np.inf
    solid_constants = np.asarray(solid_cubic_constants(poisson))*young
    b = get_B(domain.eval_shape_fun_der(np.zeros(3)))
    allowed = float(allowable_stress)
    for iteration in range(iterations):
        progress(f'Homogenized lattice sizing {iteration+1}/{iterations}; {domain.nel:,} elements')
        coefficients = np.array(law.evaluate(rho.ravel(order='F'), young=young))
        mask = active.ravel(order='F')
        pad = prescribed.ravel(order='F') & mask
        coefficients[:, ~mask] = solid_constants[:, None]*1e-9
        coefficients[:, pad] = solid_constants[:, None]
        elasticity.update(coefficients)
        initial = elasticity.solve(force, initial)
        with kernel_threads():
            displacement = initial.reshape(-1, 3)
            element_u = displacement[domain.conn].reshape(domain.nel, 24)
            strain = element_u@b.T
        stress = np.zeros_like(strain)
        stress[:, :3] = coefficients[0, :, None]*strain[:, :3]
        stress[:, :3] += coefficients[1, :, None]*(strain[:, :3].sum(axis=1)[:, None]-strain[:, :3])
        stress[:, 3:] = coefficients[2, :, None]*strain[:, 3:]
        vm = np.sqrt(np.maximum(0., .5*((stress[:, 0]-stress[:, 1])**2+(stress[:, 1]-stress[:, 2])**2
                    +(stress[:, 2]-stress[:, 0])**2)+3*np.sum(stress[:, 3:]**2, axis=1)))
        phase_average = np.zeros(domain.nel)
        phase_average[mask] = vm[mask]/np.maximum(rho.ravel(order='F')[mask], 1e-6)
        local = phase_average.reshape(shape, order='F')/allowed
        peak_displacement = float(np.linalg.norm(displacement[present], axis=1).max())
        displacement_util = peak_displacement/displacement_limit if displacement_limit > 0 else 0.
        score = max(float(local[active].max()), displacement_util)
        report['iterations'].append(dict(iteration=iteration+1, phase_average_von_mises_mpa=float(phase_average.max()),
            peak_displacement_mm=peak_displacement, compliance=float(force@initial), utilization=score,
            mean_relative_density=float(rho[active].mean()), resolved_strength_pass=False))
        if score < best_utilization:
            best, best_utilization = rho.copy(), score
        if score <= 1.:
            break
        drive = np.maximum(local, displacement_util)
        multiplier = np.where(drive > 1., np.clip(np.sqrt(np.maximum(drive, 1.)), 1.05, 1.5), 1.)
        changeable = active & ~prescribed
        updated = rho.copy()
        updated[changeable] = np.clip(rho[changeable]*multiplier[changeable],
                                    options.minimum_relative_density, options.maximum_relative_density)
        if np.allclose(updated, rho, rtol=0., atol=1e-12):
            break
        rho = updated
    output = _resample(best, fine_shape)
    output = np.clip(output, options.minimum_relative_density, options.maximum_relative_density)
    output[solid] = 1.; output[void | (~design & ~solid)] = 0.
    report.update(performed=True, seconds=perf_counter()-started, grid_shape=list(shape), elements=int(domain.nel),
        material_law=law.diagnostics(), solver=elasticity.statistics,
        stress_scope='Solid-phase average necessary yield condition; no junction or boundary microstress prediction',
        final_resolved_FEA_required=True,
        limitations=['Periodic bulk approximation', 'Coarse force footprints', 'No cell buckling or boundary stress prediction'])
    return output, None, report
