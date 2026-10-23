# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Compiled straight-sided Tet4/Tet10 stiffness and stress integration."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _geometry(points, conn, derivatives):
    a, b, c, d = points[conn[0]], points[conn[1]], points[conn[2]], points[conn[3]]
    x, y, z = b-a, c-a, d-a
    first = np.cross(y, z)
    determinant = x@first
    inverse = np.empty((3, 3))
    inverse[0] = first/determinant
    inverse[1] = np.cross(z, x)/determinant
    inverse[2] = np.cross(x, y)/determinant
    gradient = np.empty_like(derivatives)
    for q in range(len(derivatives)):
        for node in range(len(conn)):
            for component in range(3):
                gradient[q, node, component] = 0.
                for axis in range(3):
                    gradient[q, node, component] += derivatives[q, node, axis]*inverse[axis, component]
    return gradient, abs(determinant)


@njit(cache=True, parallel=True)
def stiffness_triplets(points, cells, derivatives, weights, lame, shear):
    width = 3*cells.shape[1]
    count = len(cells)*width*width
    rows, columns = np.empty(count, dtype=np.int32), np.empty(count, dtype=np.int32)
    values = np.empty(count)
    for element in prange(len(cells)):
        conn = cells[element]
        gradients, determinant = _geometry(points, conn, derivatives)
        for a in range(len(conn)):
            for b in range(len(conn)):
                for i in range(3):
                    for j in range(3):
                        value = 0.
                        for q in range(len(weights)):
                            first, second = gradients[q, a], gradients[q, b]
                            term = lame*first[i]*second[j]+shear*first[j]*second[i]
                            if i == j:
                                term += shear*(first@second)
                            value += term*weights[q]*determinant
                        index = element*width*width+(3*a+i)*width+3*b+j
                        rows[index], columns[index], values[index] = 3*conn[a]+i, 3*conn[b]+j, value
    return rows, columns, values


@njit(cache=True, parallel=True)
def recover_gauss_stress(points, cells, derivatives, weights, displacement, lame, shear):
    stress = np.zeros((len(cells), len(weights), 6))
    energy, volumes = np.zeros(len(cells)), np.zeros(len(cells))
    for element in prange(len(cells)):
        conn = cells[element]
        gradients, determinant = _geometry(points, conn, derivatives)
        volumes[element] = determinant*weights.sum()
        for q in range(len(weights)):
            gradient = np.zeros((3, 3))
            for node in range(len(conn)):
                for i in range(3):
                    for j in range(3):
                        gradient[i, j] += displacement[3*conn[node]+i]*gradients[q, node, j]
            trace = gradient[0, 0]+gradient[1, 1]+gradient[2, 2]
            for i in range(3):
                stress[element, q, i] = lame*trace+2*shear*gradient[i, i]
            stress[element, q, 3] = shear*(gradient[0, 1]+gradient[1, 0])
            stress[element, q, 4] = shear*(gradient[1, 2]+gradient[2, 1])
            stress[element, q, 5] = shear*(gradient[2, 0]+gradient[0, 2])
            work = 0.
            for i in range(3):
                work += stress[element, q, i]*gradient[i, i]
            work += stress[element, q, 3]*(gradient[0, 1]+gradient[1, 0])
            work += stress[element, q, 4]*(gradient[1, 2]+gradient[2, 1])
            work += stress[element, q, 5]*(gradient[2, 0]+gradient[0, 2])
            energy[element] += .5*work*weights[q]*determinant
    return stress, energy, volumes
