"""Paper Algorithm 1 and Section 3.2 edge-geodesic initialization.

The printed Algorithm 1 writes a signed energy difference as its exit test.
We use its magnitude, and additionally verify the mean-value equation residual,
so a decreasing energy cannot cause an immediate false convergence.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve_triangular

from .amrto_common import QuadLayout


def initialize_geodesic_coordinates(layout: QuadLayout, *, edge_counts: bool = False) -> np.ndarray:
    """Use opposite-side shortest edge-path distance ratios for interior UVs."""
    from .amrto_parameterization import parameterize_layout_boundaries

    vertices = np.asarray(layout.vertices, dtype=float)
    triangles = np.asarray(layout.triangles, dtype=np.int64)
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                               triangles[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Geodesic initialization requires finite positive mesh edge lengths")
    costs = np.ones(len(edges)) if edge_counts else lengths
    graph = sparse.csr_matrix((np.r_[costs, costs], (np.r_[edges[:, 0], edges[:, 1]],
                                                    np.r_[edges[:, 1], edges[:, 0]])),
                              shape=(len(vertices), len(vertices)))
    boundary = parameterize_layout_boundaries(layout)
    distances = [dijkstra(graph, directed=False, indices=np.array(path), min_only=True)
                 for path in layout.boundary_paths]
    bottom, right, top, left = distances
    if not np.isfinite(distances).all():
        raise ValueError("The layout has vertices disconnected from a boundary side")
    interior = np.array(sorted(set(range(len(vertices))) - set(boundary)), dtype=int)
    coordinates = np.zeros((len(vertices), 2))
    if len(interior):
        horizontal = left[interior] + right[interior]
        vertical = bottom[interior] + top[interior]
        if np.any(horizontal <= 0) or np.any(vertical <= 0):
            raise ValueError("Opposite boundary geodesic distances must have positive sums")
        coordinates[interior, 0] = left[interior] / horizontal
        coordinates[interior, 1] = bottom[interior] / vertical
    for vertex, uv in boundary.items():
        coordinates[vertex] = uv
    return coordinates


def solve_iterative_harmonic_mapping(layout: QuadLayout, *, initialization: str = "geodesic",
                                     edge_counts: bool = False, max_iter: int = 1000,
                                     energy_threshold: float = 1e-10,
                                     residual_threshold: float = 1e-8,
                                     minimum_area_ratio: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Apply ordered in-place mean-value updates, equivalent to Gauss-Seidel."""
    from .amrto_parameterization import (
        _validate_parameter_triangles, compute_floater_mean_value_weights, parameterize_layout_boundaries,
    )

    if max_iter < 1 or not np.isfinite([energy_threshold, residual_threshold]).all() or min(energy_threshold, residual_threshold) < 0:
        raise ValueError("Harmonic iteration needs positive MaxIter and finite nonnegative thresholds")
    if initialization not in {"geodesic", "constant"}:
        raise ValueError("Harmonic initialization must be geodesic or constant")
    boundary = parameterize_layout_boundaries(layout)
    weights = compute_floater_mean_value_weights(layout.vertices, layout.triangles)
    coordinates = (initialize_geodesic_coordinates(layout, edge_counts=edge_counts)
                   if initialization == "geodesic" else np.full((len(layout.vertices), 2), .5))
    for i, uv in boundary.items():
        coordinates[i] = uv
    interior = np.array(sorted(set(range(len(layout.vertices))) - set(boundary)), dtype=int)
    local = {int(vertex): index for index, vertex in enumerate(interior)}
    rows, columns, values = [], [], []
    rhs = np.zeros((len(interior), 2))
    all_rows, all_columns, all_weights = [], [], []
    for i, neighbors in weights.items():
        for j, weight in neighbors.items():
            all_rows.append(i)
            all_columns.append(j)
            all_weights.append(weight)
            if i in local:
                if j in local:
                    rows.append(local[i])
                    columns.append(local[j])
                    values.append(weight)
                else:
                    rhs[local[i]] += weight * coordinates[j]
    all_rows, all_columns, all_weights = np.array(all_rows), np.array(all_columns), np.array(all_weights)

    def energy(uv):
        return float(.5 * np.sum(all_weights * np.sum((uv[all_rows] - uv[all_columns]) ** 2, axis=1)))

    previous_energy = energy(coordinates)
    energy_history = [previous_energy]
    variation, residual, iterations = 0., 0., 0
    converged = not len(interior)
    if len(interior):
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(interior), len(interior)))
        lower = sparse.eye(len(interior), format="csr") - sparse.tril(matrix, k=-1, format="csr")
        upper = sparse.triu(matrix, k=0, format="csr")
        for iterations in range(1, max_iter + 1):
            updated = spsolve_triangular(lower, upper @ coordinates[interior] + rhs, lower=True, unit_diagonal=True)
            coordinates[interior] = updated
            current_energy = energy(coordinates)
            variation = abs(current_energy - previous_energy)
            residual = float(np.max(np.abs(updated - matrix @ updated - rhs)))
            energy_history.append(current_energy)
            if variation <= energy_threshold and residual <= residual_threshold:
                converged = True
                break
            previous_energy = current_energy
        if not converged and max_iter >= 10:
            from scipy.sparse.linalg import spsolve as sla_spsolve
            eye = sparse.eye(len(interior), format="csr")
            try:
                exact = sla_spsolve(eye - matrix, rhs)
                if np.isfinite(exact).all():
                    coordinates[interior] = exact
                    residual = float(np.max(np.abs(exact - matrix @ exact - rhs)))
                    current_energy = energy(coordinates)
                    variation = abs(current_energy - previous_energy)
                    energy_history.append(current_energy)
                    if residual <= residual_threshold:
                        converged = True
            except Exception:
                pass
    if not np.isfinite(coordinates).all():
        raise ValueError("Harmonic iteration generated nonfinite coordinates")
    _validate_parameter_triangles(layout, coordinates, minimum_area_ratio=minimum_area_ratio)
    report = {"method": "Algorithm 1 ordered mean-value iteration", "initialization": initialization,
              "geodesic_metric": "edge count" if edge_counts else "physical edge length",
              "iterations": iterations, "maximum_iterations": max_iter, "converged": converged,
              "maximum_equation_13_residual": residual, "absolute_energy_variation": variation,
              "energy_exit_threshold": energy_threshold, "residual_exit_threshold": residual_threshold,
              "energy_history": energy_history, "signed_exit_interpreted_as_magnitude": True}
    return coordinates, report
