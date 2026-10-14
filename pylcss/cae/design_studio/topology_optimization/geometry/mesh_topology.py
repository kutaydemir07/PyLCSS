# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Combinatorial boundary-shell checks independent of the fitter."""
from typing import Optional
import numpy as np


def triangulate_quad_faces(vertices, quads):
    """Appendix D2 diagonals, avoiding tiny triangles in nearly triangular quads.

    Output order is the first triangle of every quad followed by the second,
    so a quad's group ID can be tiled twice for display/measurement.
    """
    quads = np.asarray(quads, dtype=np.int64)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError('Quadrilateral faces must have shape (N, 4)')
    points = np.asarray(vertices)[quads]
    def area(a, b, c):
        return np.linalg.norm(np.cross(points[:, b] - points[:, a], points[:, c] - points[:, a]), axis=1)
    first_diagonal = np.abs(area(0, 1, 2) - area(0, 2, 3)) < np.abs(area(0, 1, 3) - area(1, 2, 3))
    first = np.where(first_diagonal[:, None], quads[:, [0, 1, 2]], quads[:, [0, 1, 3]])
    second = np.where(first_diagonal[:, None], quads[:, [0, 2, 3]], quads[:, [1, 2, 3]])
    return np.concatenate((first, second))

def mesh_topology_report(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Combinatorial closure, connected components, orientation and degeneracy."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=2).reshape(-1, 2)
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    orientation = np.bincount(inverse, weights=signs, minlength=len(unique))
    used = np.unique(faces)
    local_edges = np.searchsorted(used, unique)
    graph = coo_matrix((np.ones(len(local_edges)), (local_edges[:, 0], local_edges[:, 1])),
                       shape=(len(used), len(used)))
    components, labels = connected_components(graph, directed=False, return_labels=True)
    component_vertices = np.bincount(labels, minlength=components)
    component_edges = np.bincount(labels[local_edges[:, 0]], minlength=components)
    face_components = labels[np.searchsorted(used, faces[:, 0])]
    component_faces = np.bincount(face_components, minlength=components)
    component_euler = np.sort(component_vertices - component_edges + component_faces)
    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if faces.shape[1] == 4:
        normals += np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0])
    threshold = np.finfo(float).eps * max(float(np.max(np.ptp(vertices, axis=0))) ** 2, np.finfo(float).tiny)
    degenerate = int(np.count_nonzero(np.linalg.norm(normals, axis=1) <= threshold))
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    winding = int(np.count_nonzero((counts == 2) & (orientation != 0)))
    return {"boundary_edge_count": boundary, "nonmanifold_edge_count": nonmanifold,
            "winding_conflict_count": winding, "degenerate_face_count": degenerate,
            "connected_components": int(components), "euler_characteristic": int(len(used) - len(unique) + len(faces)),
            "component_euler_characteristics": [int(value) for value in component_euler],
            "is_closed": boundary == 0 and nonmanifold == 0 and degenerate == 0 and winding == 0}


def topology_error(source: dict, output: dict) -> Optional[str]:
    if output["nonmanifold_edge_count"] or output["winding_conflict_count"] or output["degenerate_face_count"]:
        return "Surface conversion produced nonmanifold, inconsistently oriented or degenerate faces"
    if source["is_closed"]:
        if not output["is_closed"]:
            return f"Surface conversion opened {output['boundary_edge_count']} boundary edges in a closed input"
        if output["connected_components"] != source["connected_components"]:
            return "Surface conversion changed the number of connected components"
        if output["euler_characteristic"] != source["euler_characteristic"]:
            return "Surface conversion changed the Euler characteristic (surface genus)"
        if (source.get("component_euler_characteristics") is not None and
                output.get("component_euler_characteristics") != source["component_euler_characteristics"]):
            return "Surface conversion changed the genus of individual closed components"
    return None

