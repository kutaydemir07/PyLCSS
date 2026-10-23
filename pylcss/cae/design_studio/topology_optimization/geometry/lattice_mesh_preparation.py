# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Remove numerical short edges from an FEA copy of a closed lattice surface."""
from __future__ import annotations

import numpy as np
import heapq


def prepare_lattice_analysis_surface(vertices, faces, displacement_limit):
    """Apply checked endpoint collapses, with a bound from original vertices.

    Every retained triangle is the affine image of a source triangle. Removed
    triangles collapse onto retained mesh edges. The closed-manifold link
    condition, absence of duplicate faces, local orientation and displacement
    bound are required at each collapse. The exported source is unchanged.
    """
    import trimesh
    from .mesh_screening import _intersecting_triangles
    original = np.asarray(vertices, dtype=float)
    cells = np.asarray(faces, dtype=np.int64).copy()
    if not np.isfinite(displacement_limit) or displacement_limit <= 0:
        raise ValueError('Lattice analysis surface preparation needs a positive physical bound.')
    parent = np.arange(len(original))
    members = {i: [i] for i in range(len(original))}
    incident = [set() for _ in original]
    for index, face in enumerate(cells):
        for vertex in face:
            incident[vertex].add(index)
    active = np.ones(len(cells), dtype=bool)
    source = trimesh.Trimesh(original, cells.copy(), process=False)
    source_volume=float(source.volume)
    from trimesh.util import bounds_tree
    source_triangles=original[cells]
    spatial_index=bounds_tree(np.column_stack((source_triangles.min(axis=1)-displacement_limit,
                                               source_triangles.max(axis=1)+displacement_limit)))
    intersection_tolerance=max(float(np.ptp(original,axis=0).max())*1e-10,1e-12)
    edges = source.edges_unique
    lengths = source.edges_unique_length
    candidates = np.flatnonzero(lengths <= displacement_limit)
    versions = np.zeros(len(original), dtype=np.int64)
    queue = [(float(lengths[i]), int(edges[i, 0]), int(edges[i, 1]), 0, 0) for i in candidates]
    heapq.heapify(queue)
    queued = {(a, b, va, vb) for _, a, b, va, vb in queue}

    def root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return int(vertex)

    collapsed = 0
    while queue:
        _, a, b, va, vb = heapq.heappop(queue)
        queued.discard((a, b, va, vb))
        if root(a) != a or root(b) != b or versions[a] != va or versions[b] != vb:
            continue
        ring = sorted(incident[a] | incident[b])
        old = cells[ring]
        edge_faces = [index for index in ring if a in cells[index] and b in cells[index]]
        if len(edge_faces) != 2:
            continue
        neighbors_a = set(cells[list(incident[a])].ravel()) - {a}
        neighbors_b = set(cells[list(incident[b])].ravel()) - {b}
        opposite = set(cells[edge_faces].ravel()) - {a, b}
        if neighbors_a & neighbors_b != opposite:
            continue
        cluster = members[a] + members[b]
        choices = sorted((float(np.linalg.norm(original[cluster] - original[keep], axis=1).max()), keep)
                         for keep in (a, b))
        for distance, keep in choices:
            if distance > displacement_limit:
                continue
            drop = b if keep == a else a
            proposed = old.copy()
            proposed[proposed == drop] = keep
            valid = np.array([len(set(face)) == 3 for face in proposed])
            retained = proposed[valid]
            if len(np.unique(np.sort(retained, axis=1), axis=0)) != len(retained):
                continue
            before = original[old[valid]]
            after = original[retained]
            cross_before = np.cross(before[:, 1]-before[:, 0], before[:, 2]-before[:, 0])
            cross_after = np.cross(after[:, 1]-after[:, 0], after[:, 2]-after[:, 0])
            if np.any(np.einsum('ij,ij->i', cross_before, cross_after) <= 0):
                continue
            pair_a,pair_b=np.triu_indices(len(retained),1)
            if len(pair_a) and np.any(_intersecting_triangles(after[pair_a],after[pair_b],
                    intersection_tolerance,include_boundary_contacts=False)):
                continue
            # The original-face boxes expanded by the certified displacement
            # remain a conservative index after every collapse. This checks
            # nearby disconnected walls as well as the topological one-ring.
            bounds=np.r_[after.min(axis=(0,1))-intersection_tolerance,
                         after.max(axis=(0,1))+intersection_tolerance]
            outside={i for i in spatial_index.intersection(bounds) if active[i] and i not in ring}
            if outside:
                other_cells=cells[sorted(outside)]
                other=original[other_cells]
                lo,hi=after.min(axis=1),after.max(axis=1)
                overlaps=np.all((hi[:,None]>=other.min(axis=1)[None]) &
                                (other.max(axis=1)[None]>=lo[:,None]),axis=2)
                ia,ib=np.nonzero(overlaps)
                if len(ia):
                    adjacent=np.any(retained[ia,:,None]==other_cells[ib,None,:],axis=(1,2))
                    bad=False
                    for shared in (False,True):
                        ids=adjacent==shared
                        if np.any(ids) and np.any(_intersecting_triangles(after[ia[ids]],other[ib[ids]],
                                intersection_tolerance,include_boundary_contacts=not shared)):
                            bad=True;break
                    if bad:continue
            # A non-local duplicate cannot contain the retained endpoint:
            # every face incident to either endpoint is already in this ring.
            for index in ring:
                for vertex in cells[index]:
                    incident[vertex].discard(index)
            for local, index in enumerate(ring):
                if not valid[local]:
                    active[index] = False
                    continue
                cells[index] = proposed[local]
                for vertex in cells[index]:
                    incident[vertex].add(index)
            parent[drop] = keep
            members[keep] = cluster
            del members[drop]
            collapsed += 1
            # A collapse changes the link and orientation tests of nearby
            # edges. Revisit those edges, including newly created short edges;
            # a single pass leaves numerical slivers that become legal later.
            affected = np.unique(old)
            versions[affected] += 1
            neighbors = set()
            for vertex in affected:
                neighbors.update(incident[vertex])
            for index in sorted(neighbors):
                face = cells[index]
                for first, second in zip(face, np.roll(face, -1)):
                    first, second = sorted((int(first), int(second)))
                    length = float(np.linalg.norm(original[first]-original[second]))
                    stamp = (first, second, int(versions[first]), int(versions[second]))
                    if length <= displacement_limit and stamp not in queued:
                        heapq.heappush(queue, (length, *stamp)); queued.add(stamp)
            break
    representatives = np.array([root(i) for i in range(len(original))])
    displacement = np.linalg.norm(original[representatives] - original, axis=1)
    kept = cells[active]
    used = np.unique(kept)
    output_faces = np.searchsorted(used, kept)
    output_vertices = original[used].copy()
    result = trimesh.Trimesh(output_vertices, output_faces, process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('Analysis surface preparation did not preserve a closed oriented manifold.')
    return output_vertices, output_faces, {
        'method': 'Topology-preserving endpoint collapse of numerical short edges on an FEA copy',
        'source_export_geometry_unchanged': True, 'collapsed_edge_count': collapsed,
        'original_vertex_count': len(original), 'prepared_vertex_count': len(output_vertices),
        'displacement_limit_mm': float(displacement_limit),
        'maximum_original_vertex_displacement_mm': float(displacement.max(initial=0.)),
        'triangle_correspondence_displacement_bound_mm': float(displacement.max(initial=0.)),
        'link_condition_checked': True, 'local_face_orientation_preserved': True,
        'changed_neighborhood_edges_revisited': True,
        'local_triangle_interior_overlap_checked': True,
        'spatial_triangle_contacts_checked': True,
        'material_volume_relative_change': abs(float(result.volume)-source_volume)/abs(source_volume),
    }
