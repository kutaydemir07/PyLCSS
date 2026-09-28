# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Guarded analytic CAD reconstruction for beam-like topology results.

Professional topology-to-CAD systems do not force every result through one
surface representation.  Slender branches have a much better description: a
curve skeleton, a radius field, exact cylinders/cones along the branches, and
rounded analytic joints.  This module builds that representation directly from
the density grid and accepts it only when it reproduces the recovered mesh.

The route is deliberately conservative.  Plates, shells, rectangular ribs,
and complicated junctions fail the proxy, topology, volume, or symmetric
surface-deviation checks and continue to the freeform subdivision fitter.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import product
import logging
from typing import Any, Optional, Sequence

import numpy as np

from .lattice_cad import BeamLattice, _fuse_solids, _member_solid
from .occ_shapes import _assert_valid_occ_shape, _solid_volume
from .spline_brep import _mesh_volume, _sampled_surface_deviation

logger = logging.getLogger(__name__)


def _payload_field_and_bounds(
    payload: Any,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Read the recovery field, level, and physical bounds from a result."""
    if not isinstance(payload, dict):
        raise RuntimeError("Analytic branch reconstruction needs a topology payload.")

    field = payload.get("recovery_density")
    cutoff = payload.get("recovery_cutoff")
    if field is None:
        field = payload.get("manufactured_density")
        cutoff = payload.get("density_cutoff")
    if field is None:
        field = payload.get("density")
        cutoff = payload.get("density_cutoff")
    grid = np.asarray(field, dtype=float) if field is not None else np.zeros(0)
    if grid.ndim != 3 or min(grid.shape, default=0) < 3:
        raise RuntimeError(
            "Analytic branch reconstruction needs a three-dimensional density field."
        )

    bounds = payload.get("bounds")
    if isinstance(bounds, dict):
        lower = np.asarray(bounds.get("min"), dtype=float).reshape(-1)
        upper = np.asarray(bounds.get("max"), dtype=float).reshape(-1)
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        lower = np.asarray(bounds[0], dtype=float).reshape(-1)
        upper = np.asarray(bounds[1], dtype=float).reshape(-1)
    else:
        raise RuntimeError("Analytic branch reconstruction needs physical bounds.")
    if (
        lower.size < 3
        or upper.size < 3
        or not np.all(np.isfinite(lower[:3]))
        or not np.all(np.isfinite(upper[:3]))
        or np.any(upper[:3] <= lower[:3])
    ):
        raise RuntimeError("Analytic branch reconstruction received invalid bounds.")

    level = float(0.5 if cutoff is None else cutoff)
    if not np.isfinite(level):
        level = 0.5
    return grid, level, lower[:3], upper[:3]


def _skeleton_graph(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Thin a volume and return its 26-neighbour voxel graph."""
    from skimage.morphology import skeletonize

    skeleton = np.asarray(skeletonize(np.asarray(mask, dtype=bool)), dtype=bool)
    points = np.argwhere(skeleton).astype(np.int64, copy=False)
    if len(points) < 2:
        raise RuntimeError("The material mask has no usable curve skeleton.")

    lookup = {tuple(int(value) for value in point): index for index, point in enumerate(points)}
    offsets = tuple(
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
    )
    adjacency: list[tuple[int, ...]] = []
    for point in points:
        neighbours = []
        for offset in offsets:
            key = tuple(int(point[axis]) + int(offset[axis]) for axis in range(3))
            neighbour = lookup.get(key)
            if neighbour is not None:
                neighbours.append(int(neighbour))
        adjacency.append(tuple(sorted(set(neighbours))))
    return points, tuple(adjacency)


def _graph_components(adjacency: Sequence[Sequence[int]]) -> list[set[int]]:
    unseen = set(range(len(adjacency)))
    components: list[set[int]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _farthest_graph_node(
    seed: int,
    component: set[int],
    adjacency: Sequence[Sequence[int]],
) -> int:
    distance = {int(seed): 0}
    queue: deque[int] = deque([int(seed)])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour not in component or neighbour in distance:
                continue
            distance[int(neighbour)] = distance[current] + 1
            queue.append(int(neighbour))
    return max(distance, key=distance.get)


def _trace_skeleton_paths(
    points: np.ndarray,
    adjacency: Sequence[Sequence[int]],
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Collapse junction voxels and trace maximal degree-two skeleton paths."""
    degrees = np.asarray([len(values) for values in adjacency], dtype=np.int64)
    critical = {int(index) for index in np.flatnonzero(degrees != 2)}

    # A closed ring has degree two everywhere.  Add two opposite anchors so it
    # becomes two paths without changing its loop topology.
    for component in _graph_components(adjacency):
        if component.isdisjoint(critical):
            first = min(component)
            opposite = _farthest_graph_node(first, component, adjacency)
            critical.update((first, opposite))

    if not critical:
        raise RuntimeError("The curve skeleton has no traceable branch anchors.")

    unclustered = set(critical)
    clusters: list[set[int]] = []
    while unclustered:
        seed = unclustered.pop()
        cluster = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unclustered:
                    unclustered.remove(neighbour)
                    cluster.add(neighbour)
                    queue.append(neighbour)
        clusters.append(cluster)

    cluster_of: dict[int, int] = {}
    for cluster_index, cluster in enumerate(clusters):
        for voxel in cluster:
            cluster_of[int(voxel)] = int(cluster_index)
    cluster_points = np.asarray(
        [np.mean(points[sorted(cluster)], axis=0) for cluster in clusters],
        dtype=float,
    )

    visited: set[tuple[int, int]] = set()
    paths: list[tuple[int, int, np.ndarray]] = []
    for start_cluster, cluster in enumerate(clusters):
        for start_voxel in sorted(cluster):
            for neighbour in adjacency[start_voxel]:
                if cluster_of.get(neighbour) == start_cluster:
                    continue
                edge = tuple(sorted((int(start_voxel), int(neighbour))))
                if edge in visited:
                    continue
                visited.add(edge)
                chain = [int(start_voxel), int(neighbour)]
                previous, current = int(start_voxel), int(neighbour)
                while current not in cluster_of:
                    onward = [
                        int(value)
                        for value in adjacency[current]
                        if int(value) != previous
                    ]
                    if len(onward) != 1:
                        break
                    following = onward[0]
                    next_edge = tuple(sorted((current, following)))
                    if next_edge in visited:
                        break
                    visited.add(next_edge)
                    chain.append(following)
                    previous, current = current, following
                end_cluster = cluster_of.get(current)
                if end_cluster is None or end_cluster == start_cluster:
                    continue
                interior = points[np.asarray(chain[1:-1], dtype=np.int64)]
                paths.append((int(start_cluster), int(end_cluster), interior))

    if not paths:
        raise RuntimeError("The curve skeleton produced no branch paths.")
    return cluster_points, paths


def _closest_surface_points(
    mesh: Any,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    query = np.asarray(points, dtype=float)
    try:
        closest, distances, _ = trimesh.proximity.closest_point(mesh, query)
    except Exception:
        closest, distances, _ = trimesh.proximity.closest_point_naive(mesh, query)
    return np.asarray(closest, dtype=float), np.asarray(distances, dtype=float)


def _profile_keep_indices(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    spatial_tolerance: float,
    radius_tolerance: float,
) -> np.ndarray:
    """RDP simplification in joint centreline/radius space."""
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(points) <= 2:
        return np.arange(len(points), dtype=np.int64)

    kept = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, finish = stack.pop()
        if finish <= start + 1:
            continue
        chord = points[finish] - points[start]
        denominator = float(np.dot(chord, chord))
        segment = points[start + 1 : finish]
        if denominator <= 1e-30:
            parameter = np.zeros(len(segment), dtype=float)
            spatial_error = np.linalg.norm(segment - points[start], axis=1)
        else:
            parameter = np.clip(
                np.einsum("ij,j->i", segment - points[start], chord) / denominator,
                0.0,
                1.0,
            )
            projection = points[start] + parameter[:, None] * chord
            spatial_error = np.linalg.norm(segment - projection, axis=1)
        expected_radius = (
            (1.0 - parameter) * radii[start] + parameter * radii[finish]
        )
        radius_error = np.abs(radii[start + 1 : finish] - expected_radius)
        score = np.maximum(
            spatial_error / max(float(spatial_tolerance), 1e-12),
            radius_error / max(float(radius_tolerance), 1e-12),
        )
        local = int(np.argmax(score))
        if float(score[local]) <= 1.0:
            continue
        split = start + 1 + local
        kept.add(split)
        stack.extend(((start, split), (split, finish)))
    return np.asarray(sorted(kept), dtype=np.int64)


def _cross_section_radius_estimates(
    surface_vertices: np.ndarray,
    surface_tree: Any,
    path_points: np.ndarray,
    closest_radii: np.ndarray,
    *,
    cell: np.ndarray,
) -> np.ndarray:
    """Refine inscribed distances with robust local cross-section radii.

    Point-to-surface distance systematically underestimates a radius on a
    marching-cubes staircase.  A median radial projection of nearby recovered
    surface vertices is a much better circle/cone fit, while the conservative
    closest distance remains the fallback when a section is undersampled.
    """
    vertices = np.asarray(surface_vertices, dtype=float)
    points = np.asarray(path_points, dtype=float)
    estimates = np.asarray(closest_radii, dtype=float).copy()
    pitch = float(np.linalg.norm(cell))
    for index, point in enumerate(points):
        if len(points) == 2:
            tangent = points[1] - points[0]
        elif index == 0:
            tangent = points[1] - points[0]
        elif index == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[index + 1] - points[index - 1]
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1e-12:
            continue
        tangent /= tangent_length
        original = float(estimates[index])
        slab = max(1.75 * pitch, 0.45 * original)
        search_radius = max(2.0 * original, 3.0 * pitch)
        local_ids = surface_tree.query_ball_point(
            point,
            float(np.hypot(slab, search_radius)),
        )
        if len(local_ids) < 8:
            continue
        delta = vertices[np.asarray(local_ids, dtype=np.int64)] - point
        axial = np.abs(np.einsum("ij,j->i", delta, tangent))
        radial = np.linalg.norm(
            delta - np.einsum("ij,j->i", delta, tangent)[:, None] * tangent,
            axis=1,
        )
        section = radial[
            (axial <= slab)
            & (radial >= 0.25 * original)
            & (radial <= search_radius)
        ]
        if len(section) < 8:
            continue
        robust_radius = float(np.median(section))
        estimates[index] = np.clip(
            robust_radius,
            original,
            max(1.4 * original, original + pitch),
        )
    return estimates


def _prepare_branch_profiles(
    cluster_voxels: np.ndarray,
    raw_paths: Sequence[tuple[int, int, np.ndarray]],
    *,
    lower: np.ndarray,
    cell: np.ndarray,
    source_mesh: Any,
    maximum_members: int,
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, np.ndarray]], dict[str, Any]]:
    """Convert skeleton paths to simplified model-space centreline profiles."""
    cluster_points = lower + (np.asarray(cluster_voxels) + 0.5) * cell
    incidence: dict[int, int] = defaultdict(int)
    for start, finish, _ in raw_paths:
        incidence[int(start)] += 1
        incidence[int(finish)] += 1

    surface_vertices = np.asarray(source_mesh.vertices, dtype=float)
    from scipy.spatial import cKDTree

    surface_tree = cKDTree(surface_vertices)
    profiles: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for start, finish, interior in raw_paths:
        middle = lower + (np.asarray(interior, dtype=float) + 0.5) * cell
        path_points = np.vstack((cluster_points[start], middle, cluster_points[finish]))
        closest, radii = _closest_surface_points(source_mesh, path_points)
        if len(path_points) < 2 or not np.all(np.isfinite(radii)):
            continue
        radii = _cross_section_radius_estimates(
            surface_vertices,
            surface_tree,
            path_points,
            radii,
            cell=cell,
        )

        # Near a planar branch end the closest surface is the cap, so raw
        # point-to-surface distance decreases toward zero even though the true
        # cross-section radius is constant.  Detect that axial nearest-point
        # regime and continue the first radial estimate through it.
        for order in (np.arange(len(path_points)), np.arange(len(path_points) - 1, -1, -1)):
            if len(order) < 3:
                continue
            inward = path_points[int(order[1])] - path_points[int(order[0])]
            inward_length = float(np.linalg.norm(inward))
            if inward_length <= 1e-12:
                continue
            inward /= inward_length
            cap_count = 0
            for ordered_index in order[:-1]:
                surface_vector = closest[int(ordered_index)] - path_points[int(ordered_index)]
                surface_distance = float(np.linalg.norm(surface_vector))
                if surface_distance <= 1e-12:
                    cap_count += 1
                    continue
                axial_alignment = abs(float(np.dot(surface_vector, inward))) / surface_distance
                if axial_alignment < 0.75:
                    break
                cap_count += 1
            if 0 < cap_count < len(order) - 1:
                radial_start = cap_count
                radial_finish = min(len(order), cap_count + 3)
                reference = float(
                    np.median(radii[order[radial_start:radial_finish]])
                )
                radii[order[:cap_count]] = reference

        # Digital thinning retracts an endpoint by roughly half the local
        # thickness.  Recover the cap centre from the extreme axial projection
        # of nearby surface vertices.  This is substantially more stable than
        # a single closest-point query on oblique voxel caps.
        for end_index, cluster_id in (
            (0, start),
            (-1, finish),
        ):
            if incidence[int(cluster_id)] != 1:
                continue
            tangent_span = min(6, len(path_points) - 1)
            tangent_anchor = (
                tangent_span if end_index == 0 else len(path_points) - 1 - tangent_span
            )
            outward = path_points[end_index] - path_points[tangent_anchor]
            outward_length = float(np.linalg.norm(outward))
            if outward_length <= 1e-12:
                continue
            outward /= outward_length
            interior_radius = float(
                np.median(radii[1 : min(len(radii), 4)])
                if end_index == 0
                else np.median(radii[max(0, len(radii) - 4) : -1])
            )
            cap_search_radius = float(
                np.hypot(2.5 * interior_radius, 1.35 * interior_radius)
            )
            cap_ids = surface_tree.query_ball_point(
                path_points[end_index],
                cap_search_radius,
            )
            surface_delta = (
                surface_vertices[np.asarray(cap_ids, dtype=np.int64)]
                - path_points[end_index]
            )
            axial = np.einsum("ij,j->i", surface_delta, outward)
            radial = np.linalg.norm(
                surface_delta - axial[:, None] * outward,
                axis=1,
            )
            cap_candidates = axial[
                (axial > 0.0)
                & (axial <= 2.5 * interior_radius)
                & (radial <= 1.35 * interior_radius)
            ]
            if len(cap_candidates) >= 6:
                cap_distance = float(np.percentile(cap_candidates, 95.0))
                path_points[end_index] += cap_distance * outward
                radii[end_index] = interior_radius

        if len(radii) >= 3:
            from scipy.ndimage import median_filter

            radii = median_filter(radii, size=3, mode="nearest")
        minimum_radius = max(0.20 * float(np.min(cell)), 1e-7)
        radii = np.maximum(np.asarray(radii, dtype=float), minimum_radius)
        profiles.append((int(start), int(finish), path_points, radii))

    if not profiles:
        raise RuntimeError("No skeleton branch retained a usable radius profile.")

    all_radii = np.concatenate([profile[3] for profile in profiles])
    base_spatial = max(float(np.linalg.norm(cell)), 0.003 * float(np.linalg.norm(np.ptp(cluster_points, axis=0))))
    base_radius = max(0.35 * float(np.min(cell)), 0.08 * float(np.median(all_radii)))
    selected: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    selected_tolerance = base_spatial
    for scale in (1.0, 1.6, 2.5, 4.0, 6.0):
        candidate_profiles = []
        member_count = 0
        for start, finish, path_points, radii in profiles:
            keep = _profile_keep_indices(
                path_points,
                radii,
                spatial_tolerance=base_spatial * scale,
                radius_tolerance=base_radius * scale,
            )
            reduced_points = path_points[keep]
            reduced_radii = radii[keep]
            candidate_profiles.append(
                (start, finish, reduced_points, reduced_radii)
            )
            member_count += max(0, len(reduced_points) - 1)
        selected = candidate_profiles
        selected_tolerance = base_spatial * scale
        if member_count <= int(maximum_members):
            break

    member_count = sum(max(0, len(profile[2]) - 1) for profile in selected)
    if member_count > int(maximum_members):
        raise RuntimeError(
            f"The curve skeleton needs {member_count} analytic members, above "
            f"the guarded {int(maximum_members)}-member limit."
        )
    return cluster_points, selected, {
        "raw_skeleton_voxel_count": int(
            sum(len(profile[2]) for profile in profiles)
        ),
        "skeleton_branch_count": int(len(profiles)),
        "centerline_simplification_tolerance": float(selected_tolerance),
    }


def _profiles_to_lattice(
    cluster_points: np.ndarray,
    profiles: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
) -> BeamLattice:
    nodes = [np.asarray(point, dtype=float) for point in cluster_points]
    beams: list[tuple[int, int]] = []
    beam_radii: list[tuple[float, float]] = []
    cluster_updates: dict[int, list[np.ndarray]] = defaultdict(list)

    for start, finish, points, radii in profiles:
        cluster_updates[int(start)].append(np.asarray(points[0], dtype=float))
        cluster_updates[int(finish)].append(np.asarray(points[-1], dtype=float))
        path_ids = [int(start)]
        for point in points[1:-1]:
            path_ids.append(len(nodes))
            nodes.append(np.asarray(point, dtype=float))
        path_ids.append(int(finish))
        for index in range(len(path_ids) - 1):
            if float(np.linalg.norm(points[index + 1] - points[index])) <= 1e-9:
                continue
            beams.append((path_ids[index], path_ids[index + 1]))
            beam_radii.append((float(radii[index]), float(radii[index + 1])))

    for cluster_id, values in cluster_updates.items():
        nodes[int(cluster_id)] = np.mean(values, axis=0)
    return BeamLattice(
        nodes=np.asarray(nodes, dtype=float),
        beams=np.asarray(beams, dtype=np.int64),
        radii=np.asarray(beam_radii, dtype=float),
        family="Recovered topology branch network",
        source="topology-preserving density skeleton and recovered-surface radii",
    )


def _proxy_surface_deviation(
    source_vertices: np.ndarray,
    lattice: BeamLattice,
    *,
    sample_limit: int = 2500,
) -> float:
    """Cheap one-way rejection bound before OpenCascade boolean work."""
    points = np.asarray(source_vertices, dtype=float)
    if len(points) > int(sample_limit):
        indices = np.linspace(0, len(points) - 1, int(sample_limit), dtype=int)
        points = points[indices]
    best = np.full(len(points), np.inf, dtype=float)
    for beam_index, (start_id, finish_id) in enumerate(lattice.beams):
        start = lattice.nodes[int(start_id)]
        finish = lattice.nodes[int(finish_id)]
        axis = finish - start
        length = float(np.linalg.norm(axis))
        if length <= 1e-12:
            continue
        direction = axis / length
        delta = points - start
        axial = np.einsum("ij,j->i", delta, direction)
        parameter = np.clip(axial / length, 0.0, 1.0)
        radial = np.linalg.norm(
            delta - np.clip(axial, 0.0, length)[:, None] * direction,
            axis=1,
        )
        radius = (
            (1.0 - parameter) * lattice.radii[beam_index, 0]
            + parameter * lattice.radii[beam_index, 1]
        )
        side = np.abs(radial - radius)
        before = axial < 0.0
        after = axial > length
        side[before] = np.where(
            radial[before] <= lattice.radii[beam_index, 0],
            -axial[before],
            np.hypot(-axial[before], radial[before] - lattice.radii[beam_index, 0]),
        )
        side[after] = np.where(
            radial[after] <= lattice.radii[beam_index, 1],
            axial[after] - length,
            np.hypot(
                axial[after] - length,
                radial[after] - lattice.radii[beam_index, 1],
            ),
        )
        # The endpoint disks are real candidate faces too.  In particular,
        # tessellated source cylinders commonly contain a centre vertex on
        # each cap; measuring that vertex against the lateral surface alone
        # would report a false error equal to the whole branch radius.
        inside_start_disk = radial <= lattice.radii[beam_index, 0]
        inside_finish_disk = radial <= lattice.radii[beam_index, 1]
        side = np.minimum(
            side,
            np.where(inside_start_disk, np.abs(axial), np.inf),
        )
        side = np.minimum(
            side,
            np.where(inside_finish_disk, np.abs(length - axial), np.inf),
        )
        best = np.minimum(best, side)
    node_degree = np.zeros(len(lattice.nodes), dtype=np.int64)
    for start, finish in lattice.beams:
        node_degree[int(start)] += 1
        node_degree[int(finish)] += 1
    for node_id in np.flatnonzero(node_degree >= 3):
        sphere_distance = np.abs(
            np.linalg.norm(points - lattice.nodes[int(node_id)], axis=1)
            - lattice.node_radii[int(node_id)]
        )
        best = np.minimum(best, sphere_distance)
    return float(np.max(best, initial=0.0))


def _analytic_branch_solid(
    lattice: BeamLattice,
    *,
    fuzzy_value: float,
) -> tuple[Any, int]:
    """Build one fused solid with planar ends and spherical internal joints."""
    import cadquery as cq
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    solids = []
    for index, (start_id, finish_id) in enumerate(lattice.beams):
        solids.append(
            _member_solid(
                lattice.nodes[int(start_id)],
                lattice.nodes[int(finish_id)],
                float(lattice.radii[index, 0]),
                float(lattice.radii[index, 1]),
            )
        )

    node_degree = np.zeros(len(lattice.nodes), dtype=np.int64)
    for start, finish in lattice.beams:
        node_degree[int(start)] += 1
        node_degree[int(finish)] += 1
    joint_count = 0
    node_radii = lattice.node_radii
    # Only true branch junctions need a ball.  Degree-two profile samples are
    # already joined by their adjacent cylinders/cones; a sphere there would
    # create an unintended bulge and can even sit outside a tapered segment.
    for node_id in np.flatnonzero(node_degree >= 3):
        radius = float(node_radii[int(node_id)])
        solids.append(
            BRepPrimAPI_MakeSphere(
                gp_Pnt(*(float(value) for value in lattice.nodes[int(node_id)])),
                radius,
            ).Shape()
        )
        joint_count += 1

    if not solids:
        raise RuntimeError("The curve skeleton produced no analytic solids.")
    fused = solids[0] if len(solids) == 1 else _fuse_solids(
        solids,
        fuzzy_value=max(float(fuzzy_value), 0.0),
    )
    shape = cq.Shape.cast(fused)
    # OCC's multi-tool general fuse occasionally returns an empty, formally
    # valid compound for strongly overlapping tapered members.  Retry that
    # specific failure pairwise, keeping the operation bounded by the guarded
    # member limit above.
    if shape is None or (not shape.Solids() and len(solids) > 1):
        shape = cq.Shape.cast(solids[0])
        for tool in solids[1:]:
            shape = shape.fuse(
                cq.Shape.cast(tool),
                tol=max(float(fuzzy_value), 1e-9),
            )
    if shape is None or not shape.isValid() or not shape.Solids():
        raise RuntimeError("The analytic branch union is not a valid B-rep.")
    return shape, int(joint_count)


def _mesh_signature(mesh: Any) -> tuple[bool, int, int]:
    return bool(mesh.is_watertight), int(mesh.euler_number), int(mesh.body_count)


def _lattice_euler_number(lattice: BeamLattice) -> tuple[int, int]:
    """Return surface Euler number and component count of a thickened graph."""
    used = {int(value) for value in np.asarray(lattice.beams).reshape(-1)}
    adjacency: dict[int, set[int]] = {node: set() for node in used}
    for start, finish in lattice.beams:
        adjacency[int(start)].add(int(finish))
        adjacency[int(finish)].add(int(start))
    components = 0
    unseen = set(used)
    while unseen:
        components += 1
        queue = [unseen.pop()]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    # A regular neighbourhood of a graph is a handlebody.  Its genus is the
    # graph cycle rank E - V + C, hence chi(boundary) = 2C - 2g.
    cycle_rank = len(lattice.beams) - len(used) + components
    return int(2 * components - 2 * cycle_rank), int(components)


def _candidate_brep_signature(
    shape: Any,
    lattice: BeamLattice,
) -> tuple[bool, int, int]:
    """Validate closure/components and use the branch graph for its genus."""
    solids = list(shape.Solids())
    if not solids and str(shape.ShapeType()).lower() == "solid":
        solids = [shape]
    closed_valid = bool(solids) and all(
        bool(solid.isValid()) and bool(solid.Shells()) for solid in solids
    )
    euler_number, graph_components = _lattice_euler_number(lattice)
    closed_valid = closed_valid and len(solids) == graph_components
    return closed_valid, int(euler_number), int(len(solids))


def _skeleton_branch_brep(
    payload: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    absolute_fit_tolerance: float,
    relative_fit_tolerance: float,
    maximum_volume_delta: float,
    maximum_relative_deviation: float,
    maximum_members: int = 64,
    validate_brep: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct and validate an exact cylinder/cone branch network."""
    import trimesh

    source_vertices = np.asarray(vertices, dtype=float)[:, :3]
    source_faces = np.asarray(faces, dtype=np.int64)[:, :3]
    grid, cutoff, lower, upper = _payload_field_and_bounds(payload)
    mask = np.asarray(grid >= cutoff, dtype=bool)
    if not mask.any() or mask.all():
        raise RuntimeError("The density field has no material/void interface.")

    diagonal = float(np.linalg.norm(upper - lower))
    cell = (upper - lower) / np.asarray(grid.shape, dtype=float)
    fit_tolerance = max(
        float(absolute_fit_tolerance or 0.0),
        float(relative_fit_tolerance or 0.0) * diagonal,
        1e-7,
    )
    deviation_limit = max(
        4.0 * fit_tolerance,
        float(maximum_relative_deviation) * diagonal,
    )

    source_mesh = trimesh.Trimesh(
        vertices=source_vertices,
        faces=source_faces,
        process=False,
    )
    if not source_mesh.is_watertight:
        raise RuntimeError("Analytic branch reconstruction needs a watertight mesh.")

    skeleton_points, adjacency = _skeleton_graph(mask)
    cluster_points, raw_paths = _trace_skeleton_paths(skeleton_points, adjacency)
    cluster_points, profiles, profile_report = _prepare_branch_profiles(
        cluster_points,
        raw_paths,
        lower=lower,
        cell=cell,
        source_mesh=source_mesh,
        maximum_members=int(maximum_members),
    )
    lattice = _profiles_to_lattice(cluster_points, profiles)
    if not len(lattice.beams):
        raise RuntimeError("The curve skeleton produced no analytic members.")

    proxy_deviation = _proxy_surface_deviation(source_vertices, lattice)
    proxy_limit = max(3.0 * deviation_limit, 0.03 * diagonal)
    if not np.isfinite(proxy_deviation) or proxy_deviation > proxy_limit:
        raise RuntimeError(
            "The result is not sufficiently beam-like for analytic branch "
            f"reconstruction (proxy deviation {proxy_deviation:g}, limit "
            f"{proxy_limit:g})."
        )

    candidate, joint_count = _analytic_branch_solid(
        lattice,
        fuzzy_value=max(1e-8, 1e-7 * diagonal),
    )
    if validate_brep:
        _assert_valid_occ_shape(candidate, label="analytic topology branch body")

    source_signature = _mesh_signature(source_mesh)
    candidate_signature = _candidate_brep_signature(candidate, lattice)
    if candidate_signature != source_signature:
        raise RuntimeError(
            "Analytic branch reconstruction changed watertightness, component "
            f"count, or genus ({source_signature} -> {candidate_signature})."
        )

    source_volume = _mesh_volume(source_vertices, source_faces)
    candidate_volume = _solid_volume(candidate)
    volume_delta: Optional[float] = None
    if source_volume is not None and candidate_volume is not None:
        volume_delta = (float(candidate_volume) - source_volume) / source_volume
        if not np.isfinite(volume_delta) or abs(volume_delta) > float(
            maximum_volume_delta
        ):
            raise RuntimeError(
                "Analytic branch reconstruction changed enclosed volume by "
                f"{volume_delta:+.2%}; the limit is "
                f"{float(maximum_volume_delta):.2%}."
            )

    sampled_deviation = _sampled_surface_deviation(
        source_vertices,
        source_faces,
        candidate,
        tessellation_tolerance=max(0.5 * fit_tolerance, 2e-4 * diagonal),
    )
    if not np.isfinite(sampled_deviation) or sampled_deviation > deviation_limit:
        raise RuntimeError(
            "Analytic branch reconstruction sampled surface deviation is "
            f"{sampled_deviation:g}, above the {deviation_limit:g} limit."
        )

    surface_types: dict[str, int] = defaultdict(int)
    for face in candidate.Faces():
        surface_types[str(face.geomType()).upper()] += 1
    report = {
        "method": "Analytic Branch Network",
        "representation": (
            "exact cylinders/cones with planar end caps and spherical rounded joints"
        ),
        "editable": True,
        "smooth": True,
        "fallback_used": False,
        "source_triangle_count": int(len(source_faces)),
        "skeleton_voxel_count": int(len(skeleton_points)),
        **profile_report,
        **lattice.diagnostics(),
        "joint_ball_count": int(joint_count),
        "cad_face_count": int(len(candidate.Faces())),
        "cad_surface_type_counts": dict(sorted(surface_types.items())),
        "proxy_surface_deviation": float(proxy_deviation),
        "fit_tolerance": float(fit_tolerance),
        "max_sampled_surface_deviation": float(sampled_deviation),
        "maximum_allowed_surface_deviation": float(deviation_limit),
        "source_mesh_volume": source_volume,
        "cad_volume_before_feature_healing": candidate_volume,
        "volume_delta_pct": (
            float(volume_delta * 100.0) if volume_delta is not None else None
        ),
        "source_topology_signature": source_signature,
        "cad_topology_signature": candidate_signature,
        "continuity": (
            "analytic cylinders/cones; spherical internal joints provide rounded "
            "branch transitions"
        ),
    }
    logger.info(
        "Topology CAD: accepted analytic skeleton reconstruction with %d members, "
        "%d rounded joints, %d CAD faces, deviation=%g, volume delta=%s.",
        len(lattice.beams),
        joint_count,
        len(candidate.Faces()),
        sampled_deviation,
        f"{volume_delta:+.2%}" if volume_delta is not None else "unknown",
    )
    return candidate, report


__all__ = [
    "_profile_keep_indices",
    "_skeleton_branch_brep",
    "_skeleton_graph",
    "_trace_skeleton_paths",
]
