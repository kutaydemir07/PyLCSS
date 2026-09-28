# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recognize analytic regions in a recovered topology surface.

A topology result is not a smooth blob. The webs and flanges a load path
settles into are flat, and the interfaces and service passages that survive
from the source CAD are exactly planar or cylindrical. Fitting freeform
patches to all of it discards that: the press-crown reconstruction produced
3000 Bezier faces with no planar or cylindrical face at all.

This module is the recognition stage that has to come first. It grows regions
over the triangle adjacency graph, accepting a neighbour only when its normal
agrees with the region normal and its vertices stay within a distance band of
the region plane -- the standard formulation used for reverse engineering
(PCL region growing, Polylidar3D's dominant-normal partitioning), with the
plane refitted by area-weighted least squares once the region has stopped
growing.

It is deliberately pure NumPy and free of OpenCASCADE: which surface each
region *is* can then be tested without a CAD kernel, and the B-rep builder can
consume the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


__all__ = [
    "CylindricalRegion",
    "PlanarRegion",
    "SurfaceSegmentation",
    "segment_cylindrical_regions",
    "segment_planar_regions",
    "snap_surface_to_planar_regions",
]


@dataclass(frozen=True)
class PlanarRegion:
    """One maximal set of adjacent triangles sharing a fitted plane."""

    face_indices: np.ndarray
    origin: np.ndarray
    normal: np.ndarray
    area: float
    max_deviation: float

    def signed_distances(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of each point from the fitted plane."""
        return (np.asarray(points, dtype=float) - self.origin) @ self.normal


@dataclass(frozen=True)
class SurfaceSegmentation:
    """Planar regions found in a surface, plus what they did not cover."""

    regions: list[PlanarRegion]
    total_area: float
    planar_area: float
    face_region: np.ndarray  # -1 where a face belongs to no accepted region

    @property
    def planar_area_fraction(self) -> float:
        """Share of surface area explained by accepted planar regions."""
        if self.total_area <= 0.0:
            return 0.0
        return float(self.planar_area / self.total_area)


@dataclass(frozen=True)
class CylindricalRegion:
    """One connected set of triangles sharing a fitted cylinder."""

    face_indices: np.ndarray
    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius: float
    area: float
    max_deviation: float

    def radial_distances(self, points: np.ndarray) -> np.ndarray:
        """Distance of each point from the cylinder axis."""
        offset = np.asarray(points, dtype=float) - self.axis_point
        along = offset @ self.axis_direction
        radial = offset - along[:, None] * self.axis_direction
        return np.linalg.norm(radial, axis=1)


def _face_normals_and_areas(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit face normals and triangle areas; degenerate faces get zero."""
    triangle = vertices[faces]
    cross = np.cross(
        triangle[:, 1, :] - triangle[:, 0, :],
        triangle[:, 2, :] - triangle[:, 0, :],
    )
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    safe = np.where(twice_area > 0.0, twice_area, 1.0)
    normals = cross / safe[:, None]
    normals[twice_area <= 0.0] = 0.0
    return normals, areas


def _face_adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR-style neighbour offsets/indices over shared triangle edges.

    Built with a sort rather than a Python dictionary: a recovered surface runs
    to hundreds of thousands of triangles, and the edge table is the only part
    of region growing that would otherwise dominate the cost.
    """
    face_count = len(faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    owners = np.tile(np.arange(face_count, dtype=np.int64), 3)

    _unique, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    # Only manifold edges couple two faces. A boundary edge has one owner and a
    # non-manifold edge has three or more; neither defines a unique neighbour.
    order = np.argsort(inverse, kind="stable")
    sorted_edge_ids = inverse[order]
    sorted_owners = owners[order]
    starts = np.searchsorted(sorted_edge_ids, np.arange(len(counts)), side="left")
    manifold = np.flatnonzero(counts == 2)
    if len(manifold) == 0:
        offsets = np.zeros(face_count + 1, dtype=np.int64)
        return offsets, np.zeros(0, dtype=np.int64)

    left = sorted_owners[starts[manifold]]
    right = sorted_owners[starts[manifold] + 1]
    source = np.concatenate([left, right])
    target = np.concatenate([right, left])

    degree = np.bincount(source, minlength=face_count)
    offsets = np.zeros(face_count + 1, dtype=np.int64)
    np.cumsum(degree, out=offsets[1:])
    order = np.argsort(source, kind="stable")
    return offsets, target[order].astype(np.int64)


def _fit_plane(
    points: np.ndarray,
    weights: np.ndarray,
    fallback_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted least-squares plane through a region's vertices."""
    total = float(np.sum(weights))
    if total <= 0.0 or len(points) < 3:
        return points.mean(axis=0), fallback_normal
    origin = (points * weights[:, None]).sum(axis=0) / total
    centered = (points - origin) * np.sqrt(weights)[:, None]
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return origin, fallback_normal
    normal = np.asarray(vh[-1], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return origin, fallback_normal
    normal = normal / norm
    # Keep the fitted normal on the same side as the triangles it came from,
    # so a region normal always points out of the body like its faces do.
    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal
    return origin, normal


def segment_planar_regions(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    angle_tolerance_deg: float = 5.0,
    distance_tolerance: float | None = None,
    minimum_area: float | None = None,
    maximum_flatness_ratio: float = 0.02,
    band_saturation_ratio: float = 0.6,
    minimum_border_crease_deg: float = 9.0,
) -> SurfaceSegmentation:
    """Group triangles into planar regions and fit each region's plane.

    ``distance_tolerance`` defaults to half the median edge length, which keeps
    one voxel step from being absorbed into its neighbouring face while still
    tolerating the sub-voxel ripple a smoothed isosurface carries.
    ``minimum_area`` defaults to one part in a thousand of the surface, below
    which a "plane" is a tessellation artifact rather than a face.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] < 3
        or faces.ndim != 2
        or faces.shape[1] < 3
        or len(faces) == 0
    ):
        return SurfaceSegmentation([], 0.0, 0.0, np.zeros(0, dtype=np.int64))
    vertices = vertices[:, :3]
    faces = faces[:, :3]
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        return SurfaceSegmentation([], 0.0, 0.0, np.zeros(0, dtype=np.int64))

    normals, areas = _face_normals_and_areas(vertices, faces)
    total_area = float(np.sum(areas))

    if distance_tolerance is None:
        triangle = vertices[faces]
        edge_lengths = np.concatenate(
            [
                np.linalg.norm(triangle[:, 1, :] - triangle[:, 0, :], axis=1),
                np.linalg.norm(triangle[:, 2, :] - triangle[:, 1, :], axis=1),
                np.linalg.norm(triangle[:, 0, :] - triangle[:, 2, :], axis=1),
            ]
        )
        edge_lengths = edge_lengths[edge_lengths > 0.0]
        median_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
        distance_tolerance = 0.5 * median_edge
    if minimum_area is None:
        minimum_area = 1.0e-3 * total_area

    offsets, neighbours = _face_adjacency(faces)
    minimum_dot = math.cos(math.radians(float(angle_tolerance_deg)))

    face_region = np.full(len(faces), -1, dtype=np.int64)
    visited = np.zeros(len(faces), dtype=bool)
    visited[areas <= 0.0] = True
    seed_order = np.argsort(areas)[::-1]

    regions: list[PlanarRegion] = []
    planar_area = 0.0
    for seed in seed_order:
        if visited[seed]:
            continue
        seed_normal = normals[seed]
        seed_origin = vertices[faces[seed]].mean(axis=0)
        visited[seed] = True
        member_faces = [int(seed)]
        stack = [int(seed)]
        while stack:
            current = stack.pop()
            for slot in range(offsets[current], offsets[current + 1]):
                candidate = int(neighbours[slot])
                if visited[candidate]:
                    continue
                if float(np.dot(normals[candidate], seed_normal)) < minimum_dot:
                    continue
                offset = (vertices[faces[candidate]] - seed_origin) @ seed_normal
                if float(np.max(np.abs(offset))) > distance_tolerance:
                    continue
                visited[candidate] = True
                member_faces.append(candidate)
                stack.append(candidate)

        member_indices = np.asarray(member_faces, dtype=np.int64)
        region_area = float(np.sum(areas[member_indices]))
        if region_area < minimum_area:
            # Left unassigned rather than forced into a plane; the caller can
            # hand what remains to a curved-surface or freeform stage.
            face_region[member_indices] = -1
            continue

        member_vertex_ids = np.unique(faces[member_indices].reshape(-1))
        member_points = vertices[member_vertex_ids]
        vertex_weight = np.zeros(len(vertices), dtype=float)
        np.add.at(
            vertex_weight,
            faces[member_indices].reshape(-1),
            np.repeat(areas[member_indices] / 3.0, 3),
        )
        origin, normal = _fit_plane(
            member_points,
            vertex_weight[member_vertex_ids],
            seed_normal,
        )
        deviation = float(np.max(np.abs((member_points - origin) @ normal)))
        # A patch on a curved surface passes the angle and band tests as long
        # as it stays small, so those two alone carve a sphere into hundreds of
        # "planes": at 5 degrees a truncated sphere produced 498 of them, and
        # snapping to invented planes deformed the body until it would no
        # longer close. Two scale-free tests separate a face from curvature.
        # A real plane is flat relative to its own size:
        if deviation > float(maximum_flatness_ratio) * math.sqrt(region_area):
            face_region[member_indices] = -1
            continue
        # ...and it did not need the whole growth band to stay together. A
        # curvature patch grows until it saturates the band, which is exactly
        # what makes it stop; a genuine face never approaches it. This test is
        # independent of how finely the surface was tessellated, so it does not
        # punish a coarse mesh whose face really is two triangles.
        if deviation > float(band_saturation_ratio) * float(distance_tolerance):
            face_region[member_indices] = -1
            continue
        # Finally: a face ends somewhere. Where a real one stops there is a
        # crease, and where a curvature patch stops the surface simply carries
        # on bending at the same rate. This is the only test of the three that
        # is not local, and it is the one that actually separates them -- a
        # patch small enough is always flat enough to pass the other two.
        #
        # Measured at the 75th percentile rather than the median, because a
        # voxel isosurface arrives smoothed and its creases are rounded. On the
        # press crown the real faces (top face, four column pads) show a median
        # border angle of only 6.6-7.0 degrees against 2.1-5.4 for the
        # artifacts -- too close to separate -- while their upper quartile is
        # 10-15 degrees against 6-8. The crease is still there; it just no
        # longer runs the whole way round.
        member_mask = np.zeros(len(faces), dtype=bool)
        member_mask[member_indices] = True
        outside_normals = [
            normals[int(neighbours[slot])]
            for face_index in member_indices
            for slot in range(offsets[face_index], offsets[face_index + 1])
            if not member_mask[int(neighbours[slot])]
        ]
        if outside_normals:
            crossing = np.degrees(
                np.arccos(
                    np.clip(np.asarray(outside_normals) @ seed_normal, -1.0, 1.0)
                )
            )
            if float(np.percentile(crossing, 75.0)) < float(
                minimum_border_crease_deg
            ):
                face_region[member_indices] = -1
                continue

        face_region[member_indices] = len(regions)
        regions.append(
            PlanarRegion(
                face_indices=member_indices,
                origin=origin,
                normal=normal,
                area=region_area,
                max_deviation=deviation,
            )
        )
        planar_area += region_area

    return SurfaceSegmentation(
        regions=regions,
        total_area=total_area,
        planar_area=planar_area,
        face_region=face_region,
    )


def _fit_circle_2d(points_2d: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit, weighted; returns centre and radius."""
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    target = x * x + y * y
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    try:
        solution, *_ = np.linalg.lstsq(
            design * sqrt_w[:, None], target * sqrt_w, rcond=None
        )
    except np.linalg.LinAlgError:
        return np.zeros(2), 0.0
    centre = np.asarray(solution[:2], dtype=float)
    squared = float(solution[2]) + float(centre @ centre)
    return centre, math.sqrt(max(squared, 0.0))


def segment_cylindrical_regions(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    candidate_faces: Sequence[int] | None = None,
    radial_tolerance: float | None = None,
    minimum_area: float | None = None,
    minimum_normal_alignment: float = 0.90,
) -> list[CylindricalRegion]:
    """Grow cylindrical regions out of the triangles plane fitting left behind.

    A cylinder's surface normals are all perpendicular to its axis, so the
    axis is the direction the normals never point along -- the smallest right
    singular vector of the area-weighted normal matrix. With the axis known the
    problem drops to a 2-D circle fit.

    The growth has to be driven by cylinder consistency, not by connectivity.
    Everything a plane pass rejects on a real result is one single connected
    sheet, so collecting connected components hands the fitter the whole body
    at once and every candidate is rejected. Instead each seed estimates a
    cylinder from a small local patch, and the region then grows only over
    faces that sit on that radius and whose normals point radially -- the local
    seeding that Efficient RANSAC gets from random sampling, taken from the
    adjacency graph instead.
    """
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    if len(faces) == 0:
        return []
    normals, areas = _face_normals_and_areas(vertices, faces)
    total_area = float(np.sum(areas))
    if candidate_faces is None:
        candidate_pool = np.arange(len(faces), dtype=np.int64)
    else:
        candidate_pool = np.asarray(candidate_faces, dtype=np.int64)
    if len(candidate_pool) == 0:
        return []
    if minimum_area is None:
        minimum_area = 1.0e-3 * total_area

    offsets, neighbours = _face_adjacency(faces)
    eligible = np.zeros(len(faces), dtype=bool)
    eligible[candidate_pool] = True
    eligible[areas <= 0.0] = False

    if radial_tolerance is None:
        triangle = vertices[faces]
        edge_lengths = np.linalg.norm(
            triangle[:, 1, :] - triangle[:, 0, :], axis=1
        )
        edge_lengths = edge_lengths[edge_lengths > 0.0]
        radial_tolerance = (
            float(np.median(edge_lengths)) if edge_lengths.size else 1.0
        )

    def _fit_cylinder(member: np.ndarray):
        """Return (axis_point, axis, radius, max_radial_deviation) or None."""
        if len(member) < 4:
            return None
        weighted_normals = normals[member] * areas[member][:, None]
        try:
            _u, _s, vh = np.linalg.svd(weighted_normals, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        axis = np.asarray(vh[-1], dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return None
        axis = axis / axis_norm
        vertex_ids = np.unique(faces[member].reshape(-1))
        points = vertices[vertex_ids]
        helper = (
            np.array([0.0, 0.0, 1.0])
            if abs(float(axis[2])) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        basis_u = np.cross(axis, helper)
        basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
        basis_v = np.cross(axis, basis_u)
        origin = points.mean(axis=0)
        local = points - origin
        planar = np.column_stack([local @ basis_u, local @ basis_v])
        centre_2d, radius = _fit_circle_2d(planar, np.ones(len(planar)))
        if not np.isfinite(radius) or radius <= 0.0:
            return None
        deviation = float(
            np.max(np.abs(np.linalg.norm(planar - centre_2d, axis=1) - radius))
        )
        axis_point = origin + centre_2d[0] * basis_u + centre_2d[1] * basis_v
        return axis_point, axis, float(radius), deviation

    def _radially_consistent(candidate: int, axis_point, axis, radius) -> bool:
        """True when a face lies on the cylinder and faces away from its axis."""
        triangle_points = vertices[faces[candidate]]
        offset = triangle_points - axis_point
        radial = offset - (offset @ axis)[:, None] * axis
        lengths = np.linalg.norm(radial, axis=1)
        if float(np.max(np.abs(lengths - radius))) > radial_tolerance:
            return False
        centroid_offset = triangle_points.mean(axis=0) - axis_point
        centroid_radial = centroid_offset - float(centroid_offset @ axis) * axis
        length = float(np.linalg.norm(centroid_radial))
        if length <= 1e-9:
            return False
        alignment = abs(
            float(normals[candidate] @ (centroid_radial / length))
        )
        return alignment >= float(minimum_normal_alignment)

    visited = ~eligible
    regions: list[CylindricalRegion] = []
    for seed in candidate_pool[np.argsort(areas[candidate_pool])[::-1]]:
        if visited[seed]:
            continue
        # Seed patch: a few adjacency rings, enough to estimate an axis and a
        # radius but small enough that it cannot span unrelated geometry.
        patch = [int(seed)]
        frontier = [int(seed)]
        in_patch = {int(seed)}
        while frontier and len(patch) < 48:
            current = frontier.pop(0)
            for slot in range(offsets[current], offsets[current + 1]):
                candidate = int(neighbours[slot])
                if candidate in in_patch or visited[candidate]:
                    continue
                in_patch.add(candidate)
                patch.append(candidate)
                frontier.append(candidate)
        estimate = _fit_cylinder(np.asarray(patch, dtype=np.int64))
        if estimate is None:
            visited[seed] = True
            continue
        axis_point, axis, radius, _deviation = estimate
        if not _radially_consistent(int(seed), axis_point, axis, radius):
            visited[seed] = True
            continue

        member_faces = [int(seed)]
        claimed = {int(seed)}
        stack = [int(seed)]
        while stack:
            current = stack.pop()
            for slot in range(offsets[current], offsets[current + 1]):
                candidate = int(neighbours[slot])
                if candidate in claimed or visited[candidate]:
                    continue
                if not _radially_consistent(candidate, axis_point, axis, radius):
                    continue
                claimed.add(candidate)
                member_faces.append(candidate)
                stack.append(candidate)

        member = np.asarray(member_faces, dtype=np.int64)
        region_area = float(np.sum(areas[member]))
        if region_area < minimum_area or len(member) < 8:
            visited[seed] = True
            continue

        refit = _fit_cylinder(member)
        if refit is None:
            visited[seed] = True
            continue
        axis_point, axis, radius, deviation = refit
        if deviation > radial_tolerance:
            visited[seed] = True
            continue

        visited[member] = True
        regions.append(
            CylindricalRegion(
                face_indices=member,
                axis_point=axis_point,
                axis_direction=axis,
                radius=float(radius),
                area=region_area,
                max_deviation=deviation,
            )
        )
    return regions


def snap_surface_to_planar_regions(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    segmentation: SurfaceSegmentation,
    *,
    maximum_displacement: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Move vertices onto the planes their region was fitted to.

    A recovered isosurface carries sub-voxel ripple, so a face that *is* flat
    arrives a few millimetres wavy. Every later stage inherits that: a patch
    fitter spends resolution reproducing the ripple, and a column pad that the
    press actually bolts to comes out curved.

    Snapping resolves each vertex against every planar region meeting at it --
    one plane projects, two planes give an edge line, three or more give a
    corner -- so shared boundaries stay shared and the regions keep meeting
    exactly where they met before. A vertex that would have to travel further
    than ``maximum_displacement`` is left alone: that is a sign its region
    assignment is wrong, and moving it would cut into the body.
    """
    vertices = np.asarray(vertices, dtype=float)[:, :3].copy()
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    regions = segmentation.regions
    if not regions or len(faces) == 0:
        return vertices, {
            "snapped_vertices": 0.0,
            "skipped_vertices": 0.0,
            "max_displacement": 0.0,
            "mean_displacement": 0.0,
        }

    # The cap is per vertex, not global. A vertex was accepted into its region
    # because it already lay inside that region's band, so snapping it must
    # move it by at most that much; a global cap taken from the worst region in
    # the model lets a vertex on a tight region travel far enough to distort
    # the body instead of flattening it.
    global_cap = (
        float(maximum_displacement)
        if maximum_displacement is not None
        else float("inf")
    )

    # Which planar regions touch each vertex.
    vertex_regions: dict[int, set[int]] = {}
    for index, region in enumerate(regions):
        for vertex_id in np.unique(faces[region.face_indices].reshape(-1)):
            vertex_regions.setdefault(int(vertex_id), set()).add(index)

    displacements: list[float] = []
    skipped = 0
    for vertex_id, region_ids in vertex_regions.items():
        original = vertices[vertex_id]
        members = [regions[index] for index in sorted(region_ids)]
        normals = np.asarray([region.normal for region in members], dtype=float)
        offsets = np.asarray(
            [float(region.normal @ region.origin) for region in members],
            dtype=float,
        )
        if len(members) == 1:
            target = original - float(
                normals[0] @ original - offsets[0]
            ) * normals[0]
        else:
            # Least-norm correction that satisfies every plane it can. Two
            # planes leave the shared edge free, three pin a corner, and a
            # rank-deficient set (nearly parallel planes) falls back to the
            # least-squares point rather than exploding.
            residual = offsets - normals @ original
            try:
                correction, *_ = np.linalg.lstsq(normals, residual, rcond=None)
            except np.linalg.LinAlgError:
                skipped += 1
                continue
            target = original + correction
        local_cap = min(
            global_cap,
            1.5 * max(region.max_deviation for region in members),
        )
        travel = float(np.linalg.norm(target - original))
        if not np.isfinite(travel) or travel > local_cap:
            skipped += 1
            continue
        vertices[vertex_id] = target
        displacements.append(travel)

    return vertices, {
        "snapped_vertices": float(len(displacements)),
        "skipped_vertices": float(skipped),
        "max_displacement": float(max(displacements)) if displacements else 0.0,
        "mean_displacement": (
            float(np.mean(displacements)) if displacements else 0.0
        ),
    }
