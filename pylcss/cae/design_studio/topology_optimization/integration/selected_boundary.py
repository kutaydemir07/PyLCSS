# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Map CAD selections without turning their bounding boxes into volumes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.study import BoundaryMesh
from .geometry_mapping import _bbox_tuple, _entry_bboxes, _fraction_box


def selection_regions(entry: dict, bounds: tuple[np.ndarray, np.ndarray], *, exact_boxes: bool = True) -> list:
    """Return exact axis-aligned boxes or tessellated selected boundaries.

    A box is exact only for a rectangular axis-aligned face, straight
    axis-aligned edge, or vertex. Trimmed, curved and angled selections use
    their geometry. Bounding-box metadata alone cannot describe those shapes.
    """
    mins, maxs = (np.asarray(value, dtype=float)[:3] for value in bounds)
    span = np.maximum(maxs - mins, 1e-12)
    tolerance = float(np.max(span)) * 1e-7

    def validate_bounds(bbox: tuple) -> None:
        lower, upper = np.asarray(bbox[::2]), np.asarray(bbox[1::2])
        if not np.all(np.isfinite(bbox)) or np.any(lower > upper):
            raise ValueError("Selected boundary has invalid bounds.")
        if np.any(lower < mins - tolerance) or np.any(upper > maxs + tolerance):
            raise ValueError("Selected boundary lies outside the design domain.")

    geometries = [geometry for geometry in entry.get("geometries") or [] if geometry is not None]
    if not geometries and entry.get("geometry") is not None:
        geometries = [entry["geometry"]]
    if not geometries:
        boxes = list(dict.fromkeys(_entry_bboxes(entry)))
        for box in boxes:
            validate_bounds(box)
        if any(all(box[i + 1] - box[i] > 1e-9 for i in (0, 2, 4)) for box in boxes):
            raise ValueError("A curved or angled selection needs its CAD geometry; its bounding box is a volume.")
        return [_fraction_box(box, bounds) for box in boxes]

    regions = []
    for geometry in geometries:
        try:
            bbox = _bbox_tuple(geometry.BoundingBox())
            kind = geometry.ShapeType()
        except AttributeError as exc:
            raise ValueError("Selected boundary needs a CAD face, edge or vertex.") from exc
        if bbox is None:
            raise ValueError("Selected boundary has no valid CAD bounds.")
        extents = np.asarray([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]])
        dimension = int(np.count_nonzero(extents > tolerance))
        exact = kind == "Vertex"
        if kind == "Edge" and dimension == 1:
            exact = np.isclose(float(geometry.Length()), float(np.max(extents)), rtol=1e-7)
        if kind == "Face" and dimension == 2:
            area = float(np.prod(extents[extents > tolerance]))
            exact = np.isclose(float(geometry.Area()), area, rtol=1e-7)
        if exact and exact_boxes:
            validate_bounds(bbox)
            regions.append(_fraction_box(bbox, bounds))
            continue

        chord_tolerance = max(float(np.max(span)) * 1e-4, 1e-6)
        if kind == "Face":
            vertices, cells = geometry.tessellate(chord_tolerance)
        elif kind == "Edge":
            vertices, _ = geometry.sample(chord_tolerance)
            cells = [(i, i + 1) for i in range(len(vertices) - 1)]
        else:
            raise ValueError(f"Select a CAD face, edge or vertex, rather than {kind!r}.")
        points = np.asarray([vertex.toTuple() for vertex in vertices], dtype=float)
        cells = np.asarray(cells, dtype=np.int64)
        if not len(points) or not len(cells):
            raise ValueError("Selected CAD boundary could not be sampled.")
        if np.any(points < mins - tolerance) or np.any(points > maxs + tolerance):
            raise ValueError("Selected boundary lies outside the design domain.")
        physical_cells = points[cells]
        if cells.shape[1] == 3:
            area_vectors = 0.5 * np.cross(
                physical_cells[:, 1] - physical_cells[:, 0],
                physical_cells[:, 2] - physical_cells[:, 0],
            )
            measures = np.linalg.norm(area_vectors, axis=1)
        else:
            area_vectors = None
            measures = np.linalg.norm(physical_cells[:, 1] - physical_cells[:, 0], axis=1)
        valid = measures > 1e-15
        if not np.any(valid):
            raise ValueError("Selected CAD boundary has zero measure.")
        regions.append((
            BoundaryMesh(vertices=np.clip((points - mins) / span, 0.0, 1.0), cells=cells[valid]),
            measures[valid],
            area_vectors[valid] if area_vectors is not None else None,
        ))
    return regions


def region_measure(region: tuple, bounds: tuple[np.ndarray, np.ndarray]) -> float:
    if isinstance(region[0], BoundaryMesh):
        return float(np.sum(region[1]))
    span = np.asarray(bounds[1])[:3] - np.asarray(bounds[0])[:3]
    extents = (np.asarray(region[1::2]) - np.asarray(region[::2])) * span
    return float(np.prod(extents[extents > 1e-9]))


def boundary_with_force(region: tuple, vector: Any = None, pressure: float | None = None) -> BoundaryMesh:
    boundary, measures, area_vectors = region
    if pressure is not None:
        if area_vectors is None:
            raise ValueError("Pressure requires a selected CAD face.")
        boundary.cell_forces = float(pressure) * area_vectors
    else:
        boundary.cell_forces = measures[:, None] / float(np.sum(measures)) * np.asarray(vector, dtype=float)
    return boundary

