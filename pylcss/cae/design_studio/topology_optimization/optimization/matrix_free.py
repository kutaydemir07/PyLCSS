# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Compiled Hex8 matrix actions with symmetric geometric multigrid and adjoints.

Only the small coarsest operator is assembled. Coarse material coefficients
are volume averaged for preconditioning; residuals and derivatives always use
the original fine-grid operator. Non-dyadic grids keep their physical bounds.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
import os

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


@njit(cache=True, parallel=True)
def _action(conn, adjacent, matrices, scales, vector):
    result = np.zeros(len(vector))
    for node in prange(len(adjacent)):
        for corner in range(8):
            element = adjacent[node, corner]
            if element < 0:
                continue
            for component in range(3):
                row = 3*corner+component
                value = 0.
                for material in range(len(matrices)):
                    term = 0.
                    for column in range(24):
                        term += matrices[material, row, column] * vector[3*conn[element, column//3]+column%3]
                    value += scales[material, element]*term
                result[3*node+component] += value
    return result


@njit(cache=True, parallel=True)
def _diagonal(adjacent, matrices, scales, row_norms):
    result = np.zeros(3*len(adjacent))
    bound = np.zeros(3*len(adjacent))
    for node in prange(len(adjacent)):
        for corner in range(8):
            element = adjacent[node, corner]
            if element >= 0:
                for component in range(3):
                    row = 3*corner+component
                    for material in range(len(matrices)):
                        result[3*node+component] += scales[material, element]*matrices[material, row, row]
                        bound[3*node+component] += abs(scales[material, element])*row_norms[material, row]
    return result, bound


@njit(cache=True, parallel=True)
def _energy_gradient(conn, matrices, left, right):
    result = np.zeros((len(matrices), len(conn)))
    for element in prange(len(conn)):
        for material in range(len(matrices)):
            value = 0.
            for i in range(24):
                inner = 0.
                for j in range(24):
                    inner += matrices[material, i, j]*right[3*conn[element, j//3]+j%3]
                value += left[3*conn[element, i//3]+i%3]*inner
            result[material, element] = -value
    return result


@contextmanager
def kernel_threads():
    """Bound compilation/runtime threads and avoid nested BLAS oversubscription."""
    from threadpoolctl import threadpool_limits
    previous = get_num_threads()
    set_num_threads(min(previous, 8, os.cpu_count() or 1))
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            yield
    finally:
        set_num_threads(previous)


def element_matrices(domain, constitutive):
    from pymoto.modules.assembly import get_B
    matrices = np.zeros((len(constitutive), 24, 24))
    weight = float(np.prod(domain.element_size/2))
    for node in domain.node_numbering:
        position = np.asarray(node)*domain.element_size/(2*np.sqrt(3.))
        b = get_B(domain.eval_shape_fun_der(position))
        for index, tensor in enumerate(constitutive):
            matrices[index] += weight*(b.T@tensor@b)
    return matrices


def _interpolation(fine, coarse):
    matrices = []
    for nf, nc in zip(fine, coarse):
        sites = np.linspace(0., float(nc), nf+1)
        left = np.minimum(np.floor(sites).astype(int), nc-1)
        fraction = sites-left
        rows = np.repeat(np.arange(nf+1), 2)
        columns = np.column_stack((left, left+1)).ravel()
        values = np.column_stack((1-fraction, fraction)).ravel()
        matrices.append(sparse.csr_matrix((values, (rows, columns)), shape=(nf+1, nc+1)))
    # pyMOTO uses x fastest, then y, z; components are interleaved.
    return sparse.kron(sparse.kron(sparse.kron(matrices[2], matrices[1]), matrices[0]),
                       sparse.eye(3), format='csr')


def _average_axis(nfine, ncoarse):
    edges = np.linspace(0., float(nfine), ncoarse+1)
    lo, hi = np.arange(nfine), np.arange(1, nfine+1)
    weights = np.maximum(0., np.minimum(edges[1:, None], hi)-np.maximum(edges[:-1, None], lo))
    return weights/(edges[1:]-edges[:-1])[:, None]


class MatrixFreeElasticity:
    """One geometry/BC operator, with mutable constitutive scales per element."""
    def __init__(self, domain, constitutive, fixed, *, constant=None, tolerance=1e-6,
                 max_iterations=5000, coarse_elements=512, cancel_callback=None):
        from .pymoto_runtime import import_pymoto
        self.domain = domain
        self.fixed = np.asarray(fixed, dtype=np.int64)
        self.tolerance, self.max_iterations = float(tolerance), int(max_iterations)
        self.cancel_callback = cancel_callback
        self.conn = np.ascontiguousarray(domain.conn, dtype=np.int64)
        self.matrices = element_matrices(domain, constitutive)
        self.row_norms = np.abs(self.matrices).sum(axis=2)
        self.adjacent = np.full((domain.nnodes, 8), -1, dtype=np.int64)
        for corner in range(8):
            self.adjacent[self.conn[:, corner], corner] = np.arange(domain.nel)
        self.shape = (3*domain.nnodes,)*2
        self.bc_diagonal = float(np.max(np.abs(self.matrices)))
        self.constant = None if constant is None or not getattr(constant, 'nnz', 0) else constant.tocsr()
        self.child = self.transfer = None
        dims = (int(domain.nelx), int(domain.nely), int(domain.nelz))
        if domain.nel > coarse_elements:
            coarse = tuple(max(1, (count+1)//2) for count in dims)
            if coarse != dims:
                units = np.asarray(dims)*domain.element_size/np.asarray(coarse)
                child_domain = import_pymoto().VoxelDomain(*coarse, *units)
                self.transfer = _interpolation(dims, coarse)
                # Dirichlet values remain exact on the fine operator. Remove
                # coarse corrections at constrained fine DOFs.
                if len(self.fixed):
                    free = np.ones(self.shape[0]); free[self.fixed] = 0.
                    self.transfer = sparse.diags(free)@self.transfer
                coords = np.asarray(domain.get_node_indices(self.fixed//3)).T
                mapped = np.rint(coords*np.asarray(coarse)/np.asarray(dims)).astype(int)
                child_nodes = child_domain.get_nodenumber(*mapped.T)
                child_fixed = np.unique(3*child_nodes+self.fixed%3)
                coarse_free = np.ones(3*child_domain.nnodes)
                coarse_free[child_fixed] = 0.
                # Fixed coarse basis functions must not move adjacent free
                # fine nodes. Masking fine boundary rows alone is insufficient
                # and made the preconditioner depend badly on material units.
                self.transfer = self.transfer@sparse.diags(coarse_free)
                child_constant = None if self.constant is None else self.transfer.T@self.constant@self.transfer
                self.child = MatrixFreeElasticity(child_domain, constitutive, child_fixed,
                    constant=child_constant, tolerance=tolerance, max_iterations=max_iterations,
                    coarse_elements=coarse_elements)
                self.averages = tuple(_average_axis(nf, nc) for nf, nc in zip(dims, coarse))
        self.operator = LinearOperator(self.shape, matvec=self.matvec, dtype=float)
        self.preconditioner = LinearOperator(self.shape, matvec=self.cycle, dtype=float)
        self.statistics = dict(backend='Compiled matrix-free Hex8 with geometric multigrid',
                               fine_stiffness_assembled=self.child is None, solves=[], setup_seconds=0.)

    def update(self, scales):
        started = perf_counter()
        values = np.ascontiguousarray(np.atleast_2d(scales), dtype=float)
        if hasattr(self, 'scales') and np.array_equal(values, self.scales):
            return
        self.scales = values.copy()
        if self.scales.shape != (len(self.matrices), self.domain.nel) or not np.isfinite(self.scales).all():
            raise ValueError('Matrix-free material fields must match the physical element grid.')
        with kernel_threads():
            self.diagonal, bound = _diagonal(self.adjacent, self.matrices, self.scales, self.row_norms)
        self.diagonal[self.fixed] = self.bc_diagonal
        bound[self.fixed] = self.bc_diagonal
        if self.constant is not None:
            self.diagonal += self.constant.diagonal()
            bound += np.asarray(abs(self.constant).sum(axis=1)).ravel()
        if np.any(self.diagonal <= 0):
            raise ValueError('Matrix-free elasticity needs positive diagonal stiffness.')
        # Gershgorin bounds D^-1 A. Keep the symmetric Jacobi smoother
        # positive for auxetic/nearly incompressible and anisotropic laws too.
        self.smoothing_weight = min(.5, 1.8/float(np.max(bound/self.diagonal)))
        if self.child is not None:
            fields = self.scales[:, self.domain.elements]
            ax, ay, az = self.averages
            fields = np.einsum('ai,mijk->majk', ax, fields, optimize=True)
            fields = np.einsum('bj,majk->mabk', ay, fields, optimize=True)
            fields = np.einsum('ck,mabk->mabc', az, fields, optimize=True)
            child_scales = np.empty((len(fields), self.child.domain.nel))
            child_scales[:, self.child.domain.elements.ravel()] = fields.reshape(len(fields), -1)
            self.child.update(child_scales)
        else:
            dofs = (self.conn[:, :, None]*3+np.arange(3)).reshape(-1, 24)
            rows = np.repeat(dofs, 24, axis=1).ravel()
            columns = np.tile(dofs, (1, 24)).ravel()
            values = np.einsum('me,mij->eij', self.scales, self.matrices).ravel()
            matrix = sparse.coo_matrix((values, (rows, columns)), shape=self.shape).tocsc()
            free = np.ones(self.shape[0]); free[self.fixed] = 0.
            matrix = sparse.diags(free)@matrix@sparse.diags(free)
            matrix += sparse.diags((1-free)*self.bc_diagonal)
            if self.constant is not None:
                matrix += self.constant
            # Stabilize the preconditioner only when coarsening merges distinct
            # anchors. This term never enters the fine residual or sensitivities.
            matrix += sparse.eye(self.shape[0], format='csc')*float(self.diagonal.max())*1e-12
            self.factor = splu(matrix.tocsc())
        self.statistics['setup_seconds'] += perf_counter()-started

    def matvec(self, vector):
        raw = np.asarray(vector, dtype=float)
        values = raw.copy(); values[self.fixed] = 0.
        result = _action(self.conn, self.adjacent, self.matrices, self.scales, values)
        result[self.fixed] = self.bc_diagonal*raw[self.fixed]
        if self.constant is not None:
            result += self.constant@raw
        return result

    def cycle(self, rhs):
        if self.child is None:
            return self.factor.solve(rhs)
        weight = self.smoothing_weight
        u = weight*rhs/self.diagonal
        u += weight*(rhs-self.matvec(u))/self.diagonal
        residual = rhs-self.matvec(u)
        u += self.transfer@self.child.cycle(self.transfer.T@residual)
        u += weight*(rhs-self.matvec(u))/self.diagonal
        u += weight*(rhs-self.matvec(u))/self.diagonal
        return u

    def solve(self, rhs, initial=None):
        rhs = np.asarray(rhs, dtype=float)
        started = perf_counter(); count = [0]
        def iteration_done(_):
            count[0] += 1
            if callable(self.cancel_callback) and self.cancel_callback():
                raise InterruptedError('Matrix-free structural solve cancelled.')
        with kernel_threads():
            solution, status = cg(self.operator, rhs, x0=initial, M=self.preconditioner,
                rtol=self.tolerance, atol=0., maxiter=self.max_iterations,
                callback=iteration_done)
            relative = float(np.linalg.norm(self.matvec(solution)-rhs)/max(np.linalg.norm(rhs), 1e-30))
        self.statistics['solves'].append(dict(seconds=perf_counter()-started, iterations=count[0],
                                              relative_residual=relative))
        if status != 0 or not np.isfinite(solution).all() or relative > self.tolerance*1.05:
            raise RuntimeError(f'Matrix-free elasticity did not converge: status {status}, residual {relative:.3g}.')
        return solution

    def gradient(self, left, right):
        left, right = np.asarray(left).copy(), np.asarray(right).copy()
        left[self.fixed] = 0.; right[self.fixed] = 0.
        with kernel_threads():
            return _energy_gradient(self.conn, self.matrices, left, right)


def make_elastic_solve_module(pym):
    class ElasticSolve(pym.Module):
        def __init__(self, domain, force, fixed, poisson, constant=None, cancel_callback=None, elasticity=None):
            from pymoto.modules.assembly import get_D
            super().__init__()
            self.force = np.asarray(force, dtype=float)
            self.elasticity = (elasticity if elasticity is not None else
                MatrixFreeElasticity(domain, [get_D(1., poisson, '3d')], fixed,
                                     constant=constant, cancel_callback=cancel_callback))
            self.solution = None

        def __call__(self, material):
            self.elasticity.update(material)
            self.solution = self.elasticity.solve(self.force, self.solution)
            return self.solution

        def _sensitivity(self, derivative):
            derivative = np.asarray(derivative, dtype=float)
            factor = float(derivative@self.force/max(self.force@self.force, 1e-30))
            if np.linalg.norm(derivative-factor*self.force) <= 1e-12*max(np.linalg.norm(derivative), 1e-30):
                adjoint = factor*self.solution
            else:
                adjoint = self.elasticity.solve(derivative)
            return [self.elasticity.gradient(adjoint, self.solution)[0]]
    return ElasticSolve
