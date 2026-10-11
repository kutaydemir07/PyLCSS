# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Integrate selected triangles/lines onto the analysis grid without padding."""

from __future__ import annotations

import numpy as np

from ..models.study import BoundaryMesh


def sample_boundary(boundary: BoundaryMesh, resolution: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample at half a cell spacing; preserve each cell's integrated force.

    These points lie on the selected geometry. Rounding to analysis nodes
    approximates that geometry at mesh resolution, without filling its bounds.
    """
    vertices = np.asarray(boundary.vertices, dtype=float)
    points_out, forces_out = [], []
    sample_count = 0
    for index, cell in enumerate(np.asarray(boundary.cells, dtype=int)):
        points = vertices[cell]
        edges = points - np.roll(points, 1, axis=0)
        divisions = max(1, int(np.ceil(2.0 * np.max(np.linalg.norm(edges * resolution, axis=1)))))
        sample_count += divisions ** 2 if len(cell) == 3 else divisions
        if sample_count > 2_000_000:
            raise ValueError("Selected boundary exceeds the 2,000,000-sample integration budget on this grid; reduce analysis resolution.")
        if len(cell) == 3:
            i, j = np.indices((divisions, divisions))
            first = i + j < divisions
            second = i + j < divisions - 1
            a = np.concatenate(((i[first] + 1 / 3) / divisions, (i[second] + 2 / 3) / divisions))
            b = np.concatenate(((j[first] + 1 / 3) / divisions, (j[second] + 2 / 3) / divisions))
            samples = points[0] + a[:, None] * (points[1] - points[0]) + b[:, None] * (points[2] - points[0])
        elif len(cell) == 2:
            fractions = (np.arange(divisions) + 0.5) / divisions
            samples = points[0] + fractions[:, None] * (points[1] - points[0])
        else:
            samples = points[:1]
        points_out.append(samples)
        if boundary.cell_forces is not None:
            share = np.asarray(boundary.cell_forces[index], dtype=float) / len(samples)
            forces_out.append(np.broadcast_to(share, (len(samples), 3)))
        else:
            points_out.append(points)
    return np.concatenate(points_out), np.concatenate(forces_out) if forces_out else None
