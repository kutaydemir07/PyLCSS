# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Feature-aware source normal filtering without moving source geometry.

Physical XYZ ownership spans artificial chart boundaries. Vertex normal fans
remain separate across protected edges, sharp source dihedrals and nonmanifold
edges. Fixed graph diffusion filters vectors on those fans, not XYZ positions.
This is a bounded engineering normal-field filter, not a claim to reproduce
Kawasaki et al.'s complete spline reconstruction algorithm.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class SourceNormalField:
    """Read-only per-triangle corner targets and physical weighting metadata."""
    corner_normals: list
    report: dict

    def sample(self, layout, index, grid_u, grid_v):
        """Barycentric interpolation respects distinct normal fans at a crease."""
        from .uv_sampling import sample_chart
        faces = np.asarray(layout.triangles, dtype=np.int64)
        uv = np.asarray(layout.parameter_coords, float)[faces].reshape(-1, 2)
        normals = np.asarray(self.corner_normals[index], float).reshape(-1, 3)
        cells = np.arange(len(normals), dtype=np.int64).reshape(-1, 3)
        values = sample_chart(normals, cells, uv, np.asarray(grid_u, float), np.asarray(grid_v, float))
        lengths = np.linalg.norm(values, axis=2)
        if not np.isfinite(values).all() or np.any(lengths <= 1e-12):
            raise ValueError('Interpolated source normal field has a degenerate target')
        return values/lengths[:, :, None]


def build_source_normal_field(layouts, *, iterations=5, relaxation=.5,
                              source_crease_angle=30., spatial_sigma_mm=None,
                              maximum_faces=250000):
    """Build one physical normal field for all charts, preserving crease fans.

    Five fixed, lazy random-walk diffusion steps are the default. Positive
    distance and angular conductances, normalization each step and relaxation
    at most one half bound the operation. The spatial Gaussian radius is in
    physical millimetres; its default is twice the median source edge length.
    Every source position and chart UV is only read. No nearest-point welding,
    rounding or connectivity inferred from geometric proximity is performed.
    """
    started = perf_counter()
    if (iterations not in range(0, 21) or not np.isfinite(relaxation) or not 0 < relaxation <= .5
            or not 0 < source_crease_angle < 90 or maximum_faces < 1
            or (spatial_sigma_mm is not None and (not np.isfinite(spatial_sigma_mm) or spatial_sigma_mm <= 0))):
        raise ValueError('Require bounded normal-field iterations, relaxation and physical scales')
    if sum(len(layout.triangles) for layout in layouts) > maximum_faces:
        raise ValueError('Explicit source-normal-field face budget exceeded')
    vertex_lookup, vertices, face_lookup, faces, mappings = {}, [], {}, [], []
    for layout in layouts:
        vertex_ids = []
        for point in np.asarray(layout.vertices, float):
            if not np.isfinite(point).all():
                raise ValueError('Normal-field source coordinates must be finite')
            key = tuple(map(float, point))
            if key not in vertex_lookup:
                vertex_lookup[key] = len(vertices); vertices.append(point.copy())
            vertex_ids.append(vertex_lookup[key])
        ids = np.asarray(vertex_ids, dtype=np.int64)
        mapped = []
        for triangle in np.asarray(layout.triangles, dtype=np.int64):
            current = ids[triangle]
            key = tuple(sorted(map(int, current)))
            if key not in face_lookup:
                face_lookup[key] = len(faces); faces.append(current)
            face_id = face_lookup[key]
            corner_ids = [int(np.flatnonzero(np.asarray(faces[face_id]) == vertex)[0]) for vertex in current]
            mapped.append([3*face_id+corner for corner in corner_ids])
        mappings.append(np.asarray(mapped, dtype=np.int64))
    if not faces:
        return SourceNormalField([np.empty((0, 3, 3)) for _ in layouts],
            {'source_positions_unchanged': True, 'face_count': 0, 'seconds': perf_counter()-started})
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    xyz = vertices[faces]
    area_vectors = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    twice_area = np.linalg.norm(area_vectors, axis=1)
    if np.any(twice_area <= 1e-14):
        raise ValueError('Normal-field source has a degenerate physical triangle')
    face_normals = area_vectors/twice_area[:, None]
    protected = set()
    for layout in layouts:
        for edge in getattr(layout, 'protected_source_crease_edges', ()):
            ids = [vertex_lookup.get(tuple(map(float, point))) for point in edge]
            if any(index is None for index in ids):
                raise ValueError('A protected source crease is not represented by source vertices')
            protected.add(tuple(sorted(ids)))
    edge_faces = defaultdict(list)
    for face_id, triangle in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_faces[tuple(sorted((int(triangle[a]), int(triangle[b]))))].append((face_id, a, b))
    # Resolve winding from shared half-edge orientation, independently of
    # normal angle. Aligning signs by abs(normal dot) would incorrectly smooth
    # a genuinely folded 170-degree crease as though it were a ten-degree turn.
    adjacency = [[] for _ in faces]
    for owners in edge_faces.values():
        if len(owners) != 2: continue
        a, b = owners[0], owners[1]
        same_direction = faces[a[0], a[1]] == faces[b[0], b[1]]
        relative = -1 if same_direction else 1
        adjacency[a[0]].append((b[0], relative)); adjacency[b[0]].append((a[0], relative))
    sign = np.zeros(len(faces), dtype=np.int8); conflicts = 0
    for seed in range(len(faces)):
        if sign[seed]: continue
        sign[seed] = 1; queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor, relative in adjacency[current]:
                value = sign[current]*relative
                if not sign[neighbor]:
                    sign[neighbor] = value; queue.append(neighbor)
                elif sign[neighbor] != value:
                    conflicts += 1
    if conflicts:
        raise ValueError('Source normal field requires an orientable manifold adjacency')
    face_normals *= sign[:, None]
    corner_row, corner_column = [], []
    blocked = defaultdict(int)
    for edge, owners in edge_faces.items():
        if edge in protected:
            blocked['explicit_protected_edges'] += 1; continue
        if len(owners) != 2:
            blocked['boundary_or_nonmanifold_edges'] += 1; continue
        a, b = owners[0][0], owners[1][0]
        dot = float(face_normals[a]@face_normals[b])
        if dot < np.cos(np.deg2rad(source_crease_angle)):
            blocked['inferred_source_crease_edges'] += 1; continue
        for vertex in edge:
            ca = 3*a+int(np.flatnonzero(faces[a] == vertex)[0])
            cb = 3*b+int(np.flatnonzero(faces[b] == vertex)[0])
            corner_row.extend((ca, cb)); corner_column.extend((cb, ca))
    corner_graph = sparse.coo_matrix((np.ones(len(corner_row)), (corner_row, corner_column)),
                                     shape=(3*len(faces), 3*len(faces))).tocsr()
    fan_count, fan_ids = connected_components(corner_graph, directed=False)
    corner_areas = np.repeat(twice_area, 3)
    raw = np.repeat(face_normals, 3, axis=0)
    accum = np.column_stack([np.bincount(fan_ids, weights=corner_areas*raw[:, axis], minlength=fan_count)
                             for axis in range(3)])
    lengths = np.linalg.norm(accum, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('A physical vertex normal fan has no stable normal')
    initial = accum/lengths[:, None]
    fans = fan_ids.reshape(-1, 3)
    first_corner = np.full(fan_count, 3*len(faces), dtype=np.int64)
    np.minimum.at(first_corner, fan_ids, np.arange(len(fan_ids)))
    fan_positions = vertices[faces.ravel()[first_corner]]
    edges = np.vstack((fans[:, [0, 1]], fans[:, [1, 2]], fans[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    distance = np.linalg.norm(fan_positions[edges[:, 0]]-fan_positions[edges[:, 1]], axis=1)
    keep = (distance > 0) & (edges[:, 0] != edges[:, 1])
    edges, distance = edges[keep], distance[keep]
    median_edge = float(np.median(distance)) if len(distance) else 1.
    sigma = 2*median_edge if spatial_sigma_mm is None else float(spatial_sigma_mm)
    dot = np.clip(np.einsum('nc,nc->n', initial[edges[:, 0]], initial[edges[:, 1]]), -1, 1)
    angular_sigma = np.deg2rad(source_crease_angle)
    weight = np.exp(-.5*(distance/sigma)**2-.5*(np.arccos(dot)/angular_sigma)**2)
    weight /= np.maximum(distance, median_edge*1e-6)
    row, column = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    graph = sparse.coo_matrix((np.r_[weight, weight], (row, column)), shape=(fan_count, fan_count)).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    transition = sparse.diags(np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0))@graph
    filtered = initial.copy()
    for _ in range(iterations):
        update = (1-relaxation)*filtered+relaxation*(transition@filtered)
        update[degree == 0] = filtered[degree == 0]
        norm = np.linalg.norm(update, axis=1)
        if np.any(norm <= 1e-12):
            raise ValueError('Normal diffusion produced a degenerate fan')
        filtered = update/norm[:, None]
    def energy(value):
        return float(np.sum(weight*np.sum((value[edges[:, 0]]-value[edges[:, 1]])**2, axis=1)))
    movement = np.rad2deg(np.arccos(np.clip(np.einsum('nc,nc->n', initial, filtered), -1, 1)))
    corner_filtered = filtered[fan_ids]
    result = [corner_filtered[mapping] for mapping in mappings]
    for value in result:
        value.setflags(write=False)
    report = {'method': 'Feature-split physical vertex-fan normal diffusion',
        'engineering_extension': True, 'source_positions_unchanged': True, 'source_UV_unchanged': True,
        'physical_vertex_ownership': 'Exact XYZ tuples across all charts; no proximity welding',
        'vertex_count': int(len(vertices)), 'face_count': int(len(faces)), 'normal_fan_count': int(fan_count),
        'maximum_source_faces': int(maximum_faces),
        'iterations': iterations, 'relaxation': relaxation, 'source_crease_angle_degrees': source_crease_angle,
        'spatial_sigma_mm': sigma, 'median_source_edge_mm': median_edge,
        'blocked_edges': dict(blocked), 'normal_graph_energy_before': energy(initial),
        'normal_graph_energy_after': energy(filtered),
        'maximum_vertex_normal_rotation_degrees': float(movement.max(initial=0.)),
        'seconds': perf_counter()-started}
    return SourceNormalField(result, report)
