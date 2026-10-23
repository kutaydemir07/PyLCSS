# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Compiled inverse barycentric sampling of unchanged CAD source charts."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def sample_chart(vertices, triangles, uv, grid_u, grid_v):
    sampled = np.zeros((len(grid_u), len(grid_v), 3))
    violations = np.full((len(grid_u), len(grid_v)), np.inf)
    tolerance = 1e-5
    for triangle in triangles:
        a, b, c = uv[triangle[0]], uv[triangle[1]], uv[triangle[2]]
        denominator = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if abs(denominator) <= 1e-14:
            raise ValueError('Collapsed parameter triangle in inverse sampling')
        i0 = np.searchsorted(grid_u, min(a[0], b[0], c[0])-tolerance)
        i1 = np.searchsorted(grid_u, max(a[0], b[0], c[0])+tolerance, side='right')
        j0 = np.searchsorted(grid_v, min(a[1], b[1], c[1])-tolerance)
        j1 = np.searchsorted(grid_v, max(a[1], b[1], c[1])+tolerance, side='right')
        for i in range(i0, i1):
            for j in range(j0, j1):
                beta = ((grid_u[i]-a[0])*(c[1]-a[1])-(grid_v[j]-a[1])*(c[0]-a[0]))/denominator
                gamma = ((b[0]-a[0])*(grid_v[j]-a[1])-(b[1]-a[1])*(grid_u[i]-a[0]))/denominator
                alpha = 1-beta-gamma
                if min(alpha, beta, gamma) < -tolerance:
                    continue
                violation = max(-alpha, 0.)+max(-beta, 0.)+max(-gamma, 0.)
                if violation >= violations[i, j]:
                    continue
                alpha, beta, gamma = max(alpha, 0.), max(beta, 0.), max(gamma, 0.)
                total = alpha+beta+gamma
                for k in range(3):
                    sampled[i, j, k] = (alpha*vertices[triangle[0], k]+beta*vertices[triangle[1], k]
                                        +gamma*vertices[triangle[2], k])/total
                violations[i, j] = violation
    # Preserve the existing nearest barycentric projection for sites outside
    # the chart. The fallback uses triangles, never nearest source vertices.
    for i in range(len(grid_u)):
        for j in range(len(grid_v)):
            if np.isfinite(violations[i, j]):
                continue
            best = np.inf
            for triangle in triangles:
                a, b, c = uv[triangle[0]], uv[triangle[1]], uv[triangle[2]]
                denominator = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
                beta = ((grid_u[i]-a[0])*(c[1]-a[1])-(grid_v[j]-a[1])*(c[0]-a[0]))/denominator
                gamma = ((b[0]-a[0])*(grid_v[j]-a[1])-(b[1]-a[1])*(grid_u[i]-a[0]))/denominator
                alpha = max(1-beta-gamma, 0.); beta, gamma = max(beta, 0.), max(gamma, 0.)
                total = alpha+beta+gamma
                if total <= 0:
                    continue
                px = (alpha*a[0]+beta*b[0]+gamma*c[0])/total
                py = (alpha*a[1]+beta*b[1]+gamma*c[1])/total
                distance = (px-grid_u[i])**2+(py-grid_v[j])**2
                if distance < best:
                    best = distance
                    for k in range(3):
                        sampled[i, j, k] = (alpha*vertices[triangle[0], k]+beta*vertices[triangle[1], k]
                                            +gamma*vertices[triangle[2], k])/total
            if not np.isfinite(best):
                raise ValueError('Inverse sampling misses a source chart point')
    return sampled
