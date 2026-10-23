# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Linear Tet4/Tet10 elasticity using compiled assembly and MKL PARDISO.

Boundary sets, forces, pressures and gravity use the existing CalculiX exporter,
so geometry selections and resultants stay consistent. Unsupported models and
curved isoparametric elements retain the external solver workflow.
"""
from time import perf_counter
import os

import numpy as np
from scipy import sparse


def sparse_static_available():
    from importlib.util import find_spec
    if find_spec('pypardiso') is None:
        return False
    try:
        import pypardiso  # Import verifies its required native MKL runtime.
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _boundary_vectors(mesh, material, supports, loads, warnings):
    from .calculix_deck import _build_sets_and_step
    model, step = _build_sets_and_step(mesh, supports, loads, warnings)
    sets = {}
    surfaces = {}
    state = None
    for line in model:
        if line.startswith('*NSET'):
            name = line.split('NSET=')[1].split(',')[0].strip()
            sets[name] = []; state = name
        elif line.startswith('*SURFACE') and 'TYPE=ELEMENT' in line:
            name = line.split('NAME=')[1].split(',')[0].strip()
            surfaces[name] = []; state = ('surface', name)
        elif line.startswith('*'):
            raise NotImplementedError('This constraint definition needs the external solver.')
        elif state is not None:
            if isinstance(state, tuple):
                element_id, face = [value.strip() for value in line.split(',')]
                surfaces[state[1]].append((int(element_id)-1, int(face[1:])-1))
            else:
                sets[state].extend(int(value.strip())-1 for value in line.split(',') if value.strip())
    force = np.zeros(3*mesh.p.shape[1])
    prescribed = {}
    state = None
    for line in step:
        if line.startswith('*'):
            state = line.split(',')[0].upper()
            if state in ('*DSLOAD', '*MPC', '*EQUATION', '*TRANSFORM'):
                raise NotImplementedError('Distributed loads or coupled constraints need the external solver.')
            continue
        tokens = [value.strip() for value in line.split(',')]
        if state == '*BOUNDARY':
            nodes = sets[tokens[0]] if tokens[0] in sets else [int(tokens[0])-1]
            for node in nodes:
                for component in range(int(tokens[1])-1, int(tokens[2])):
                    dof = 3*node+component; value = float(tokens[3])
                    if dof in prescribed and prescribed[dof] != value:
                        raise ValueError('Conflicting prescribed displacement values.')
                    prescribed[dof] = value
        elif state == '*CLOAD':
            nodes = sets[tokens[0]] if tokens[0] in sets else [int(tokens[0])-1]
            for node in nodes:
                force[3*node+int(tokens[1])-1] += float(tokens[2])
        elif state == '*DLOAD':
            from .mesh import mesh_to_tet4, mesh_to_tet10, tet10_connectivity
            quadratic = tet10_connectivity(mesh) is not None
            points, cells = mesh_to_tet10(mesh, warnings) if quadratic else mesh_to_tet4(mesh, warnings)
            if tokens[1] == 'GRAV' and tokens[0] == 'EALL':
                xyz = points[cells[:, :4]]
                volume = np.abs(np.linalg.det(np.stack((xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0],
                                                        xyz[:, 3]-xyz[:, 0]), axis=-1)))/6
                acceleration = float(tokens[2])*np.array([float(value) for value in tokens[3:6]])
                density = float(material.get('rho', material.get('density', 7.85e-9)))
                factors = np.r_[np.full(4, -.05), np.full(6, .2)] if quadratic else np.full(4, .25)
                for local_node, factor in enumerate(factors):
                    for component in range(3):
                        np.add.at(force, 3*cells[:, local_node]+component, volume*density*acceleration[component]*factor)
            elif tokens[1] == 'P' and tokens[0] in surfaces:
                local_faces = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
                opposite = (3, 2, 0, 1)
                midpoint_faces = ((4, 5, 6), (4, 8, 7), (5, 9, 8), (6, 9, 7))
                for element_id, face_id in surfaces[tokens[0]]:
                    cell = cells[element_id]
                    xyz = points[cell[list(local_faces[face_id])]]
                    normal_area = .5*np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
                    if normal_area@(points[cell[opposite[face_id]]]-xyz.mean(axis=0)) < 0:
                        normal_area = -normal_area
                    # Positive pressure pushes into the material.
                    nodal_force = float(tokens[2])*normal_area/3
                    local_nodes = midpoint_faces[face_id] if quadratic else local_faces[face_id]
                    for node in cell[list(local_nodes)]:
                        force[3*node:3*node+3] += nodal_force
            else:
                raise NotImplementedError('This distributed load needs the external solver.')
    fixed = np.asarray(sorted(prescribed), dtype=int)
    return force, fixed, np.array([prescribed[dof] for dof in fixed])


def run_sparse_static(mesh, material, supports, loads, *, cancel_callback=None):
    """Return None for an unsupported loading/backend, never a fake solution."""
    if not sparse_static_available() or any(str(load.get('type', 'force')).lower() not in
            ('force', 'pressure', 'gravity', 'bearing', 'bearing_load') for load in loads):
        return None
    from skfem import ElementTetP1, ElementTetP2
    from skfem.models.elasticity import lame_parameters
    from skfem.quadrature import get_quadrature_tet
    from .tet_assembly import stiffness_triplets, recover_gauss_stress
    from pylcss.cae.design_studio.topology_optimization.optimization.matrix_free import kernel_threads
    from pypardiso import PyPardisoSolver
    from threadpoolctl import threadpool_limits
    from .mesh import tet10_connectivity, straight_sided_tetrahedra
    from .validation import validate_isotropic_material
    validate_isotropic_material(material)
    def check_cancel():
        if callable(cancel_callback) and cancel_callback():
            raise InterruptedError('Sparse static analysis cancelled.')
    check_cancel()
    warnings = []
    try:
        force, fixed, prescribed = _boundary_vectors(mesh, material, supports, loads, warnings)
    except NotImplementedError:
        return None
    start = perf_counter()
    quadratic = tet10_connectivity(mesh) is not None
    if not straight_sided_tetrahedra(mesh):
        return None  # Curved isoparametric solids retain their native solver.
    element = ElementTetP2() if quadratic else ElementTetP1()
    sites_q, weights_q = get_quadrature_tet(2)
    derivatives = np.array([element.lbasis(sites_q, node)[1].T for node in range(len(element.doflocs))]).transpose(1, 0, 2).copy()
    conn = mesh.dofs.element_dofs if quadratic else mesh.t
    if not np.array_equal(conn[:4], mesh.t[:4]):
        raise ValueError('Element DOFs do not match material tetrahedron corners.')
    cells = np.ascontiguousarray(conn.T, dtype=np.int64)
    points = np.ascontiguousarray(mesh.p.T)
    n_dofs = 3*mesh.p.shape[1]
    young = float(material.get('E', 210000.))
    poisson = float(material.get('nu', material.get('poissons_ratio', .3)))
    lame, shear = lame_parameters(young, poisson)
    # Batches bound temporary element tensors instead of allocating 900*N
    # triplets for the whole quadratic model at once.
    stiffness = None
    for first in range(0, mesh.t.shape[1], 12000):
        check_cancel()
        with kernel_threads():
            rows, columns, values = stiffness_triplets(points, cells[first:first+12000], derivatives,
                                                      weights_q, lame, shear)
        block = sparse.coo_matrix((values, (rows, columns)), shape=(n_dofs, n_dofs)).tocsr()
        stiffness = block if stiffness is None else stiffness+block
    assembly_seconds = perf_counter()-start
    check_cancel()
    displacement = np.zeros(n_dofs)
    displacement[fixed] = prescribed
    free = np.setdiff1d(np.arange(n_dofs), fixed)
    if not len(free):
        raise ValueError('The model has no free displacement DOFs.')
    rhs = force[free]-stiffness[free][:, fixed]@prescribed
    reduced = stiffness[free][:, free].tocsr()
    # PARDISO's real SPD mode stores the upper triangle only.
    upper = sparse.triu(reduced, format='csr')
    upper.sort_indices()
    solver = PyPardisoSolver(mtype=2)
    solve_started = perf_counter()
    try:
        with threadpool_limits(limits=min(8, os.cpu_count() or 1)):
            displacement[free] = solver.solve(upper, rhs)
    finally:
        solver.free_memory(everything=True)
    solve_seconds = perf_counter()-solve_started
    check_cancel()
    residual = stiffness@displacement-force
    residual_relative = float(np.linalg.norm(residual[free])/max(np.linalg.norm(rhs), 1e-30))
    if not np.isfinite(displacement).all() or residual_relative > 1e-6:
        raise RuntimeError(f'Sparse static solve failed its independent residual check ({residual_relative:.3g}).')
    reaction = np.zeros_like(force); reaction[fixed] = residual[fixed]
    applied_force = force.reshape(-1, 3).sum(axis=0)
    reaction_force = reaction.reshape(-1, 3).sum(axis=0)
    balance = applied_force+reaction_force
    equilibrium_error = float(100*np.linalg.norm(balance)/max(np.linalg.norm(applied_force), np.linalg.norm(reaction_force), 1e-30))
    # Linear stress on a straight-sided quadratic tetrahedron is recovered at
    # every element node. Display/feedback retain the maximum incident value,
    # so averaging across elements cannot hide a local stress concentration.
    with kernel_threads():
        stress_q, energy_elements, volumes = recover_gauss_stress(points, cells, derivatives, weights_q,
                                                                 displacement, lame, shear)
    gauss = np.column_stack((np.ones(sites_q.shape[1]), sites_q.T))
    sites = np.column_stack((np.ones(len(element.doflocs)), element.doflocs))
    extrapolation = sites@np.linalg.inv(gauss)
    local = np.einsum('aq,eqc->eac', extrapolation, stress_q, optimize=True)
    vm = np.sqrt(np.maximum(0., .5*((local[..., 0]-local[..., 1])**2+(local[..., 1]-local[..., 2])**2
         +(local[..., 2]-local[..., 0])**2)+3*np.sum(local[..., 3:]**2, axis=-1)))
    nodal_stress = np.zeros(mesh.p.shape[1])
    np.maximum.at(nodal_stress, conn.T.ravel(), vm.ravel())
    tensor_sum = np.zeros((mesh.p.shape[1], 6)); weight = np.zeros(mesh.p.shape[1])
    for index in range(conn.shape[0]):
        np.add.at(tensor_sum, conn[index], local[:, index]*volumes[:, None])
        np.add.at(weight, conn[index], volumes)
    tensor_sum /= np.maximum(weight[:, None], 1e-30)
    energy = float(energy_elements.sum())
    nodal_energy = np.zeros(mesh.p.shape[1]); energy_density = energy_elements/volumes
    for index in range(conn.shape[0]):
        np.add.at(nodal_energy, conn[index], energy_density*volumes)
    nodal_energy /= np.maximum(weight, 1e-30)
    yield_strength = float(material.get('yield_strength', material.get('yield_stress', 0.)) or 0.)
    maximum_displacement = float(np.linalg.norm(displacement.reshape(-1, 3), axis=1).max())
    characteristic_length = max(float(np.ptp(mesh.p, axis=1).max()), 1e-9)
    auto_scale = float(np.clip(.05*characteristic_length/maximum_displacement, 1., 200.)) if maximum_displacement > 1e-9 else 1.
    safety = yield_strength/np.maximum(nodal_stress, 1e-30) if yield_strength > 0 else None
    gauss_vm = np.sqrt(np.maximum(0., .5*((stress_q[..., 0]-stress_q[..., 1])**2+(stress_q[..., 1]-stress_q[..., 2])**2
                     +(stress_q[..., 2]-stress_q[..., 0])**2)+3*np.sum(stress_q[..., 3:]**2, axis=-1)))
    return dict(type='fea', backend='scikit-fem / MKL PARDISO', mesh=mesh, displacement=displacement,
        stress=nodal_stress, stress_tensor=tensor_sum, peak_stress_nodal=float(nodal_stress.max()),
        max_stress_gauss=float(gauss_vm.max()), peak_displacement=maximum_displacement,
        stress_location='Envelope of unaveraged element-nodal von Mises stresses',
        stress_tensor_location='Volume-averaged incident element-nodal tensor for component display',
        applied_force=applied_force, reaction_force=reaction_force, residual_force=balance,
        applied_magnitude=float(np.linalg.norm(applied_force)), reaction_magnitude=float(np.linalg.norm(reaction_force)),
        residual_magnitude=float(np.linalg.norm(balance)), equilibrium_error_pct=equilibrium_error,
        equilibrium_balanced=equilibrium_error < 1., reaction_node_ids=np.unique(fixed//3),
        nodal_force=reaction.reshape(-1, 3), ener_nodal=nodal_energy, element_ener=energy_density,
        applied_nodal_force=force.reshape(-1,3),
        element_volumes=volumes, volume=float(volumes.sum()), strain_energy=energy, compliance=2*energy,
        yield_strength=yield_strength, safety_factor=safety,
        min_safety_factor=float(safety.min()) if safety is not None else None,
        auto_deformation_scale=auto_scale, deformation_scale=auto_scale, visualization_mode='Von Mises Stress',
        solver_elapsed_s=solve_seconds, assembly_seconds=assembly_seconds, elapsed_seconds=perf_counter()-start,
        linear_residual_relative=residual_relative, warnings=warnings, external_status='completed', analysis_type='Linear',
        message='Linear solid analysis completed; independent residual and force balance recorded.')
