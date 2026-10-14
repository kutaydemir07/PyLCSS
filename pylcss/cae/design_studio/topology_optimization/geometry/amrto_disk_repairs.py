# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Geometry-preserving mesh treatments from AMRTO Appendix D, pp. 30–31.

Improvement 1 cuts between boundary loops along shortest physical mesh-edge
paths and splits disconnected vertex fans. Improvement 4 inserts internal
vertices to avoid triangles whose three corners lie on one square side.
Unsupported handles remain explicit; cutting holes alone cannot remove genus.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import heapq

import numpy as np

from .amrto_common import QuadLayout


@dataclass
class DiskRepairResult:
    vertices: np.ndarray
    faces: np.ndarray
    source_vertex_indices: np.ndarray
    source_face_indices: np.ndarray
    report: dict

    @property
    def quads(self) -> np.ndarray:
        return self.faces


def _edge_owners(faces):
    owners = defaultdict(list)
    for face_index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            owners[tuple(sorted((int(a), int(b))))].append(face_index)
    return owners


def _components(neighbors, indices):
    remaining, result = set(indices), []
    while remaining:
        seed = min(remaining)
        reached, queue = {seed}, [seed]
        remaining.remove(seed)
        while queue:
            for neighbor in neighbors.get(queue.pop(), ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    reached.add(neighbor)
                    queue.append(neighbor)
        result.append(sorted(reached))
    return result


def _face_components(faces):
    neighbors = defaultdict(set)
    for owners in _edge_owners(faces).values():
        if len(owners) == 2:
            a, b = owners
            neighbors[a].add(b)
            neighbors[b].add(a)
    return _components(neighbors, range(len(faces)))


def _boundary_loops(faces):
    adjacency = defaultdict(set)
    for (a, b), owners in _edge_owners(faces).items():
        if len(owners) > 2:
            raise ValueError("Appendix D repair cannot accept nonmanifold edges")
        if len(owners) == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Patch boundary is not a collection of simple loops")
    loops, remaining = [], set(adjacency)
    while remaining:
        start = min(remaining)
        previous, current = start, min(adjacency[start])
        loop = [start]
        while current != start:
            if current in loop:
                raise ValueError("Patch boundary revisits a vertex")
            loop.append(current)
            following = min(adjacency[current] - {previous})
            previous, current = current, following
        remaining.difference_update(loop)
        loops.append(loop)
    return loops


def _topology(faces):
    owners = _edge_owners(faces)
    loops = _boundary_loops(faces)
    chi = len(np.unique(faces)) - len(owners) + len(faces)
    components = len(_face_components(faces))
    return {"euler_characteristic": chi, "boundary_loop_count": len(loops),
            "edge_connected_component_count": components,
            "is_disk": chi == 1 and len(loops) == 1 and components == 1}


def _split_vertex_fans(vertices, faces, source_indices, cut_edges=()):
    """Duplicate only sectors disconnected in the incident face-fan graph."""
    original = np.asarray(faces, dtype=np.int64)
    repaired = original.copy()
    points, source = list(np.asarray(vertices)), list(np.asarray(source_indices, dtype=np.int64))
    owners = _edge_owners(original)
    cuts = set(cut_edges)
    incident = defaultdict(set)
    neighbors = defaultdict(lambda: defaultdict(set))
    for (a, b), edge_faces in owners.items():
        if len(edge_faces) > 2:
            raise ValueError("Appendix D repair cannot split a nonmanifold edge")
        incident[a].update(edge_faces)
        incident[b].update(edge_faces)
        if len(edge_faces) == 2 and (a, b) not in cuts:
            first, second = edge_faces
            for vertex in (a, b):
                neighbors[vertex][first].add(second)
                neighbors[vertex][second].add(first)
    splits = []
    for vertex, face_indices in sorted(incident.items()):
        sectors = _components(neighbors[vertex], face_indices)
        if len(sectors) <= 1:
            continue
        copies = [vertex]
        for sector in sectors[1:]:
            new_vertex = len(points)
            points.append(np.asarray(vertices)[vertex].copy())
            source.append(int(source_indices[vertex]))
            copies.append(new_vertex)
            for face_index in sector:
                repaired[face_index, original[face_index] == vertex] = new_vertex
        splits.append({"vertex": vertex, "source_vertex": int(source_indices[vertex]),
                       "copies": copies, "sector_count": len(sectors)})
    return np.asarray(points, dtype=float), repaired, np.asarray(source, dtype=np.int64), splits


def _shortest_boundary_cut(vertices, faces, loops):
    """Multi-source Dijkstra stops at the nearest distinct boundary loop."""
    graph = defaultdict(list)
    for a, b in _edge_owners(faces):
        length = float(np.linalg.norm(vertices[a]-vertices[b]))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("A geodesic cut requires positive physical edge lengths")
        graph[a].append((b, length))
        graph[b].append((a, length))
    starts = set(loops[0])
    targets = set(vertex for loop in loops[1:] for vertex in loop)
    distances, parents = {i: 0. for i in starts}, {}
    queue = [(0., i) for i in sorted(starts)]
    heapq.heapify(queue)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        if vertex in targets:
            path = [vertex]
            while path[-1] not in starts:
                path.append(parents[path[-1]])
            return list(reversed(path)), distance
        for neighbor, length in sorted(graph[vertex]):
            candidate = distance + length
            if candidate < distances.get(neighbor, float('inf')):
                distances[neighbor] = candidate
                parents[neighbor] = vertex
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError("No mesh-edge geodesic connects the boundary loops")


def repair_quad_patch_to_disks(vertices, quads, *, source_vertex_indices=None, max_cuts=16):
    """Return coincident seam copies and unchanged quads with explicit coverage.

    The list also retains unsupported higher-genus parts with ``is_disk=False``;
    callers must use a reported further GMCG/partition treatment for those parts.
    Source face indices refer to the supplied quad array and occur exactly once.
    """
    vertices, quads = np.asarray(vertices, dtype=float), np.asarray(quads)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or quads.ndim != 2 or quads.shape[1] != 4 or not len(quads)
            or not np.issubdtype(quads.dtype, np.integer) or quads.min() < 0
            or quads.max() >= len(vertices) or any(len(set(face)) != 4 for face in quads)):
        raise ValueError("Appendix D repair requires finite vertices and nondegenerate indexed quads")
    if not isinstance(max_cuts, int) or max_cuts < 0:
        raise ValueError("The geodesic-cut budget must be a nonnegative integer")
    quads = quads.astype(np.int64, copy=False)
    source = (np.arange(len(vertices)) if source_vertex_indices is None
              else np.asarray(source_vertex_indices, dtype=np.int64))
    if source.shape != (len(vertices),):
        raise ValueError("Source vertex correspondence must cover every input vertex")
    split_vertices, split_quads, split_source, coincident_splits = _split_vertex_fans(vertices, quads, source)
    results, cuts, covered = [], [], []
    for component_index, face_indices in enumerate(_face_components(split_quads)):
        faces = split_quads[face_indices].copy()
        used = np.unique(faces)
        positions = split_vertices[used].copy()
        mapping = {int(old): i for i, old in enumerate(used)}
        faces = np.array([[mapping[int(i)] for i in face] for face in faces], dtype=np.int64)
        source_ids = split_source[used].copy()
        initial = _topology(faces)
        while len(cuts) < max_cuts:
            loops = _boundary_loops(faces)
            if len(loops) <= 1:
                break
            path, length = _shortest_boundary_cut(positions, faces, loops)
            path_source = source_ids[path].tolist()
            cut_edges = [tuple(sorted(edge)) for edge in zip(path[:-1], path[1:])]
            positions, faces, source_ids, sectors = _split_vertex_fans(positions, faces, source_ids, cut_edges)
            after_loops = _boundary_loops(faces)
            if len(after_loops) != len(loops)-1:
                raise ValueError("Geodesic seam did not merge exactly two boundary loops")
            cuts.append({"component": component_index, "path": path, "source_vertex_path": path_source,
                         "physical_length": length, "metric": "Shortest physical mesh-edge geodesic",
                         "boundary_loops_before": len(loops), "boundary_loops_after": len(after_loops),
                         "duplicated_vertex_count": sum(len(s['copies'])-1 for s in sectors)})
        topology = _topology(faces)
        face_sources = np.asarray(face_indices, dtype=np.int64)
        # Reindexing and seam copies may never alter physical polygons.
        if not np.array_equal(positions[faces], vertices[quads[face_sources]]):
            raise ValueError("Appendix D seam changed physical quad geometry")
        if not np.array_equal(source_ids[faces], source[quads[face_sources]]):
            raise ValueError("Appendix D seam lost source vertex correspondence")
        report = {"initial_topology": initial, **topology,
                  "unsupported_reason": None if topology['is_disk'] else
                  "Holes-to-boundary cuts do not remove handles, or the cut budget was exhausted"}
        results.append(DiskRepairResult(positions, faces, source_ids, face_sources, report))
        covered.extend(face_sources.tolist())
    if sorted(covered) != list(range(len(quads))):
        raise ValueError("Appendix D repairs did not retain every input quad exactly once")
    report = {"method": "Appendix D Improvement 1: shortest mesh-edge seams and coincident vertex fan splitting",
              "paper_D1_executed": bool(cuts or coincident_splits),
              "paper_D1_validated": all(result.report['is_disk'] for result in results),
              "coincident_vertex_splits": coincident_splits, "geodesic_cuts": cuts,
              "new_vertex_count": sum(len(result.vertices) for result in results)-len(np.unique(quads)),
              "input_quad_count": len(quads), "covered_quad_count": len(covered),
              "physical_geometry_unchanged": True, "source_quad_coverage_exactly_once": True,
              "remaining_non_disk_part_count": sum(not result.report['is_disk'] for result in results)}
    return results, report


def repair_same_side_boundary_triangles(layout: QuadLayout):
    """Improvement 4 using interior-chord midpoints and both incident triangles.

    A midpoint of a topologically interior chord joins both sides of that chord,
    unlike an isolated face centroid surrounded only by fixed collinear UVs.
    Split all same-side interior chords, keeping all four boundary paths and the
    quad array unchanged. New geometry has synthetic source ID -1 plus provenance.
    """
    vertices, triangles = np.asarray(layout.vertices, dtype=float), np.asarray(layout.triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0
            or triangles.max() >= len(vertices)
            or any(len(set(triangle)) != 3 for triangle in triangles)):
        raise ValueError("Improvement 4 requires finite vertices and indexed nondegenerate triangles")
    triangles = triangles.astype(np.int64, copy=False)
    sides = [set(path) for path in layout.boundary_paths]
    if len(sides) != 4 or any(len(side) < 2 or min(side) < 0 or max(side) >= len(vertices) for side in sides):
        raise ValueError("Improvement 4 requires four square boundary categories")
    source = np.asarray(getattr(layout, 'source_vertex_indices', np.arange(len(vertices))), dtype=np.int64)
    previous_triangle_sources = np.asarray(getattr(layout, 'source_triangle_indices', np.arange(len(triangles))))
    if source.shape != (len(vertices),) or previous_triangle_sources.shape != (len(triangles),):
        raise ValueError("Improvement 4 source correspondences must cover every vertex and triangle")
    bad = [i for i, triangle in enumerate(triangles) if any(set(triangle) <= side for side in sides)]
    owners = _edge_owners(triangles)
    if any(len(faces) > 2 for faces in owners.values()):
        raise ValueError("Improvement 4 requires manifold triangle edges")
    original_points = vertices[triangles]
    original_normals = np.cross(original_points[:, 1]-original_points[:, 0], original_points[:, 2]-original_points[:, 0])
    if not bad:
        # With no subdivision the area/orientation check compares each source
        # triangle with itself. Retain that degeneracy guard, but avoid copying
        # the complete chart and rebuilding its unchanged geometry on every
        # prospective clustering step.
        areas = np.linalg.norm(original_normals, axis=1)
        if (not np.allclose(areas, areas, rtol=1e-12, atol=1e-14)
                or np.any(np.einsum('ij,ij->i', original_normals, original_normals) <= 0)):
            raise ValueError("Improvement 4 subdivision changed triangle area or orientation")
        report = {"method": "Appendix D Improvement 4: interior-chord midpoint subdivision of both incident triangles",
                  "paper_D4_executed": False, "paper_D4_condition_satisfied": True,
                  "same_side_triangle_count_before": 0, "same_side_triangle_count_after": 0,
                  "new_vertex_count": 0, "split_interior_chord_count": 0,
                  "source_triangle_area_and_orientation_preserved": True,
                  "quad_array_unchanged": True, "synthetic_source_vertex_id": -1, "provenance": []}
        if not hasattr(layout, 'appendix_D4_report'):
            layout.appendix_D4_report = report
        return layout, report
    chords = sorted(edge for edge, faces in owners.items()
                    if bad and len(faces) == 2 and any(set(edge) <= side for side in sides))
    points, faces = list(vertices), triangles.tolist()
    triangle_sources = list(range(len(triangles)))
    provenance = []
    for a, b in chords:
        new_vertex = len(points)
        points.append((vertices[a]+vertices[b])*.5)
        replacement, replacement_sources = [], []
        split_count = 0
        for triangle, original_index in zip(faces, triangle_sources):
            for offset in range(3):
                first, second, opposite = [triangle[(offset+j) % 3] for j in range(3)]
                if {first, second} == {a, b}:
                    replacement.extend(([first, new_vertex, opposite], [new_vertex, second, opposite]))
                    replacement_sources.extend((original_index, original_index))
                    split_count += 1
                    break
            else:
                replacement.append(triangle)
                replacement_sources.append(original_index)
        if split_count != 2:
            raise ValueError("Improvement 4 chord did not have exactly two incident triangles")
        faces, triangle_sources = replacement, replacement_sources
        provenance.append({"vertex": new_vertex, "parent_vertices": [a, b], "weights": [.5, .5],
                           "source_vertex": -1, "physical_point": np.asarray(points[-1]).tolist()})
    result_vertices, result_triangles = np.asarray(points), np.asarray(faces, dtype=np.int64)
    triangle_sources = np.asarray(triangle_sources, dtype=np.int64)
    remaining = [i for i, triangle in enumerate(result_triangles) if any(set(triangle) <= side for side in sides)]
    if remaining:
        raise ValueError("Improvement 4 left a same-side boundary triangle without an interior chord")
    child_points = result_vertices[result_triangles]
    child_normals = np.cross(child_points[:, 1]-child_points[:, 0], child_points[:, 2]-child_points[:, 0])
    areas = np.zeros(len(triangles))
    np.add.at(areas, triangle_sources, np.linalg.norm(child_normals, axis=1))
    if (not np.allclose(areas, np.linalg.norm(original_normals, axis=1), rtol=1e-12, atol=1e-14)
            or np.any(np.einsum('ij,ij->i', child_normals, original_normals[triangle_sources]) <= 0)):
        raise ValueError("Improvement 4 subdivision changed triangle area or orientation")
    result = replace(layout, vertices=result_vertices, triangles=result_triangles,
                     parameter_coords=None, sampled_grid_3d=None, sampled_shape=(0, 0), nurbs_surface=None)
    for key, value in vars(layout).items():
        if key not in layout.__dataclass_fields__ and key not in {
            'parameterization_report', 'parameter_area_ratio_minimum', 'source_vertex_indices',
            'source_triangle_indices', 'synthetic_vertex_provenance',
        }:
            setattr(result, key, value)
    result.source_vertex_indices = np.r_[source, np.full(len(provenance), -1, dtype=np.int64)]
    result.source_triangle_indices = previous_triangle_sources[triangle_sources]
    result.synthetic_vertex_provenance = list(getattr(layout, 'synthetic_vertex_provenance', [])) + provenance
    report = {"method": "Appendix D Improvement 4: interior-chord midpoint subdivision of both incident triangles",
              "paper_D4_executed": bool(provenance), "paper_D4_condition_satisfied": not remaining,
              "same_side_triangle_count_before": len(bad), "same_side_triangle_count_after": len(remaining),
              "new_vertex_count": len(provenance), "split_interior_chord_count": len(chords),
              "source_triangle_area_and_orientation_preserved": True,
              "quad_array_unchanged": layout.quads is None or np.array_equal(layout.quads, result.quads),
              "synthetic_source_vertex_id": -1, "provenance": provenance}
    result.appendix_D4_report = report
    return result, report
