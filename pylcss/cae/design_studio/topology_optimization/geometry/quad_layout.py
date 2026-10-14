# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Feature/curvature-aware triangle pairing for compact surface recovery.

This is a bounded native layout optimizer, not an implementation of Instant
Meshes. Existing all-quad networks are retained; unmatched polygons still need
conforming subdivision. The resulting polygon network is subdivided conformingly when needed.
"""
from __future__ import annotations

import numpy as np

from .mesh_topology import mesh_topology_report, topology_error, triangulate_quad_faces


def _curvature_directions(mesh):
    """Estimate principal directions from the change of vertex normals."""
    directions = np.zeros_like(mesh.vertices)
    anisotropy = np.zeros(len(mesh.vertices))
    for vertex, ring in enumerate(mesh.vertex_neighbors):
        normal = mesh.vertex_normals[vertex]
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, axis)
        first /= max(np.linalg.norm(first), 1e-30)
        frame = np.stack((first, np.cross(normal, first)), axis=1)
        delta = mesh.vertices[ring] - mesh.vertices[vertex]
        coordinates = delta @ frame
        lengths = np.linalg.norm(delta, axis=1)
        if len(ring) < 3 or np.any(lengths <= 0):
            continue
        change = (mesh.vertex_normals[ring] - normal) @ frame
        shape = np.linalg.lstsq(coordinates / lengths[:, None],
                                change / lengths[:, None], rcond=None)[0]
        values, vectors = np.linalg.eigh((shape + shape.T) * .5)
        anisotropy[vertex] = abs(values[1] - values[0]) / max(np.max(np.abs(values)), 1e-12)
        directions[vertex] = frame @ vectors[:, 1]
    return directions, anisotropy


def pair_triangles(mesh, crease_angle_deg=55.):
    """Optimize quad quality, principal-direction alignment and pairing coverage."""
    directions, anisotropy = _curvature_directions(mesh)
    valence = np.bincount(mesh.faces.ravel(), minlength=len(mesh.vertices))
    candidates = {}
    area_scale = max(float(np.linalg.norm(mesh.extents))**2, 1e-30)
    for faces, edge, angle in zip(mesh.face_adjacency, mesh.face_adjacency_edges,
                                  mesh.face_adjacency_angles):
        if angle >= np.deg2rad(min(crease_angle_deg, 70.)) or np.any(valence[edge] <= 3):
            continue
        sides = [(int(a), int(b)) for face in faces for a, b in
                 zip(mesh.faces[face], np.roll(mesh.faces[face], -1))
                 if set((int(a), int(b))) != set(edge)]
        successor = dict(sides)
        quad = [sides[0][0]]
        for _ in range(3):
            quad.append(successor[quad[-1]])
        points = mesh.vertices[quad]
        vectors = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(vectors, axis=1)
        normal = mesh.face_normals[faces].sum(axis=0)
        turns = np.cross(vectors, np.roll(vectors, -1, axis=0)) @ normal
        if np.any(turns <= area_scale * 1e-12) or np.any(lengths <= 0):
            continue
        cosines = np.sum(vectors * np.roll(vectors, 1, axis=0), axis=1) / (lengths*np.roll(lengths, 1))
        alignment = []
        for i, vertex in enumerate(quad):
            cosine = abs(float((vectors[i] / lengths[i]) @ directions[vertex]))
            alignment.append(anisotropy[vertex] * min(cosine**2, 1-cosine**2))
        score = float(np.max(np.abs(cosines)) + .2*(lengths.max()/lengths.min()-1)
                      + angle + .5*np.mean(alignment))
        candidates[tuple(sorted(map(int, faces)))] = (score, tuple(quad), tuple(map(int, edge)))
    # Sorted local matching has O(F log F) cost; a global blossom matching
    # becomes cubic and is unnecessary for geometry recovery.
    paired, selected = set(), []
    for pair in sorted(candidates, key=lambda key: (candidates[key][0], key)):
        if not paired.intersection(pair):
            paired.update(pair)
            selected.append(pair)
    remaining_valence = valence.copy()
    for pair in selected:
        remaining_valence[list(candidates[pair][2])] -= 1
    # Matching is over faces; independently prevent valence-two tangent fans.
    retained = set(selected)
    for pair in sorted(selected, key=lambda key: (-candidates[key][0], key)):
        edge = candidates[pair][2]
        if np.any(remaining_valence[list(edge)] < 3):
            retained.remove(pair)
            remaining_valence[list(edge)] += 1
    paired = {face for pair in retained for face in pair}
    polygons = [candidates[pair][1] for pair in sorted(retained)]
    polygons.extend(tuple(face) for i, face in enumerate(mesh.faces) if i not in paired)
    return polygons


def layout_quality(vertices, quads):
    points = np.asarray(vertices)[quads]
    lengths = np.linalg.norm(np.roll(points, -1, axis=1)-points, axis=2)
    valence = np.bincount(np.asarray(quads).ravel(), minlength=len(vertices))
    used = valence > 0
    return {'regular_vertex_count': int(np.sum(valence == 4)),
            'extraordinary_vertex_count': int(np.sum(used & (valence != 4))),
            'maximum_edge_aspect_ratio': float(np.max(lengths.max(axis=1)/np.maximum(lengths.min(axis=1), 1e-30))),
            'valence_histogram': {str(n): int(np.sum(valence == n)) for n in np.unique(valence[used])}}


def compact_quad_network(mesh):
    """MeshLab pairing, accepted only for convex topology-preserving quads."""
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                                  face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    meshes.meshing_tri_to_quad_by_smart_triangle_pairing()
    result = meshes.current_mesh()
    polygons = result.polygonal_face_list()
    if not all(len(p) == 4 for p in polygons):
        return None
    vertices, quads = result.vertex_matrix(), np.asarray(polygons, dtype=np.int64)
    # Keep only convex quads. A nonconvex pairing can fold a parameter map,
    # even though the triangle mesh itself remains a valid manifold.
    xyz = vertices[quads]
    edges = np.roll(xyz,-1,axis=1)-xyz
    normals = np.cross(xyz,np.roll(xyz,-1,axis=1)).sum(axis=1)
    turns = np.einsum('fkc,fc->fk',np.cross(edges,np.roll(edges,-1,axis=1)),normals)
    if np.any(turns <= 1e-14*np.linalg.norm(normals,axis=1)[:,None]**2):
        return None
    used = np.unique(quads)
    quads = np.searchsorted(used, quads)
    vertices = vertices[used]
    error = topology_error(mesh_topology_report(mesh.vertices, mesh.faces), mesh_topology_report(vertices, quads))
    if error:
        return None
    return vertices, quads, 0


def validate_quad_layout(source, vertices, quads):
    """Validate a caller-supplied cage without silently repairing its topology."""
    vertices = np.asarray(vertices, dtype=float)
    raw = np.asarray(quads)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or raw.ndim != 2 or raw.shape[1] != 4 or not len(raw)
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('A quad layout requires finite XYZ vertices and integer four-corner faces')
    quads = raw.astype(np.int64)
    if np.any(quads < 0) or np.any(quads >= len(vertices)):
        raise ValueError('Quad layout has invalid vertex indices')
    if np.any(np.diff(np.sort(quads, axis=1), axis=1) == 0):
        raise ValueError('A quad face repeats a corner')
    error = topology_error(mesh_topology_report(source.vertices, source.faces),
                           mesh_topology_report(vertices, quads))
    if error:
        raise ValueError(error)
    triangles = triangulate_quad_faces(vertices, quads)
    import trimesh
    cage = trimesh.Trimesh(vertices, triangles, process=False)
    def signature(mesh):
        return sorted((int(part.euler_number), 1 if part.volume > 0 else -1)
                      for part in mesh.split(only_watertight=False))
    if signature(cage) != signature(source):
        raise ValueError('Quad layout changed boundary-shell orientation or individual genus')
    return vertices, quads

