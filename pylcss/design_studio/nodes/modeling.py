# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Modeling Nodes - 3D Operations and Transformations.
"""

import cadquery as cq
from pylcss.design_studio.core.base_node import (
    CadQueryNode,
    resolve_any_input,
    resolve_numeric_input,
    resolve_shape_input,
)
import math
import logging

logger = logging.getLogger(__name__)


_SELECTOR_TYPE_ALIASES = {
    "direction": "Direction",
    "nearesttopoint": "NearestToPoint",
    "nearest point": "NearestToPoint",
    "nearest_point": "NearestToPoint",
    "index": "Index",
    "face index": "Index",
    "face_index": "Index",
    "largest area": "Largest Area",
    "largest_area": "Largest Area",
    "tag": "Tag",
    "box": "Box",
    "bounding box": "Box",
    "bounding_box": "Box",
    "coordinate range": "Coordinate Range",
    "range expression": "Coordinate Range",
    "range_expression": "Coordinate Range",
}


_DIRECTION_ALIASES = {
    "+X": ">X",
    "-X": "<X",
    "+Y": ">Y",
    "-Y": "<Y",
    "+Z": ">Z",
    "-Z": "<Z",
    "X+": ">X",
    "X-": "<X",
    "Y+": ">Y",
    "Y-": "<Y",
    "Z+": ">Z",
    "Z-": "<Z",
}


def _canonical_selector_type(value):
    text = str(value or "Direction").strip()
    return _SELECTOR_TYPE_ALIASES.get(text.lower(), text)


def _canonical_face_direction(value):
    text = str(value or ">Z").strip().upper()
    return _DIRECTION_ALIASES.get(text, text)


def _face_summary(face):
    if isinstance(face, dict):
        return {
            "center": face.get("center"),
            "bbox": face.get("bbox"),
            "node_count": face.get("node_count"),
        }
    try:
        c = face.Center()
        bb = face.BoundingBox()
        return {
            "center": [float(c.x), float(c.y), float(c.z)],
            "bbox": {
                "xmin": float(bb.xmin), "xmax": float(bb.xmax),
                "ymin": float(bb.ymin), "ymax": float(bb.ymax),
                "zmin": float(bb.zmin), "zmax": float(bb.zmax),
            },
            "area": float(face.Area()),
        }
    except Exception:
        return {}


def _selection_payload(workplane, faces, selector_type):
    faces = list(faces or [])
    return {
        "workplane": workplane,
        "face": faces[0] if faces else None,
        "faces": faces,
        "selector_type": selector_type,
        "face_count": len(faces),
        "face_summaries": [_face_summary(face) for face in faces[:12]],
    }


_MESH_FACE_DIRECTIONS = ('<X', '>X', '<Y', '>Y', '<Z', '>Z')
_MESH_COMPONENT_INDEX_BASE = 1000
_MESH_COMPONENT_INDEX_STRIDE = 1000


def _mesh_component_stored_index(direction_index, component_index):
    return (
        _MESH_COMPONENT_INDEX_BASE
        + int(direction_index) * _MESH_COMPONENT_INDEX_STRIDE
        + int(component_index)
    )


def _mesh_points(mesh_like):
    """Return mesh node coordinates as a (3, n) array for mesh-backed selection."""
    try:
        import numpy as np

        if isinstance(mesh_like, dict):
            if mesh_like.get("mesh") is not None:
                return _mesh_points(mesh_like.get("mesh"))
            if mesh_like.get("vertices") is not None:
                pts = np.asarray(mesh_like.get("vertices"), dtype=float)
                if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
                    return pts[:, :3].T
            return None
        if hasattr(mesh_like, "p"):
            pts = np.asarray(mesh_like.p, dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] > 0:
                return pts[:3, :]
    except Exception:
        return None
    return None


def _mesh_selection_payload(mesh_like, node_ids, selector_type, label, surface_faces=None):
    """Build a SelectFace-compatible payload for selected mesh nodes."""
    import numpy as np

    p = _mesh_points(mesh_like)
    if p is None:
        return None
    ids = np.asarray(node_ids, dtype=int)
    ids = ids[(ids >= 0) & (ids < p.shape[1])]
    ids = np.unique(ids)
    if ids.size == 0:
        return None

    pts = p[:, ids].T
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    centroid = pts.mean(axis=0)
    selection = {
        "mesh_selection": True,
        "selector_type": selector_type,
        "label": label,
        "node_ids": [int(v) for v in ids.tolist()],
        "node_count": int(ids.size),
        "center": [float(v) for v in center.tolist()],
        "centroid": [float(v) for v in centroid.tolist()],
        "points": [[float(x) for x in center.tolist()]],
        "bbox": {
            "xmin": float(mins[0]), "xmax": float(maxs[0]),
            "ymin": float(mins[1]), "ymax": float(maxs[1]),
            "zmin": float(mins[2]), "zmax": float(maxs[2]),
        },
    }

    if surface_faces is not None:
        try:
            face_arr = np.asarray(surface_faces, dtype=int)
            if face_arr.ndim == 2 and face_arr.shape[1] >= 3 and face_arr.size:
                face_arr = face_arr[:, :3]
                valid = np.all((face_arr >= 0) & (face_arr < p.shape[1]), axis=1)
                face_arr = face_arr[valid]
                if face_arr.size:
                    tri_pts = p[:, face_arr].transpose(1, 2, 0)
                    tri_centers = tri_pts.mean(axis=1)
                    tri_areas = 0.5 * np.linalg.norm(
                        np.cross(tri_pts[:, 1, :] - tri_pts[:, 0, :],
                                 tri_pts[:, 2, :] - tri_pts[:, 0, :]),
                        axis=1,
                    )
                    order = np.argsort(tri_areas)[::-1]
                    diag = float(np.linalg.norm(maxs - mins))
                    min_spacing = max(0.25, 0.12 * diag)
                    sample_points = []
                    for tri_idx in order:
                        point = np.asarray(tri_centers[int(tri_idx)], dtype=float)
                        if all(np.linalg.norm(point - prev) >= min_spacing for prev in sample_points):
                            sample_points.append(point)
                        if len(sample_points) >= 6:
                            break
                    if sample_points:
                        selection["points"] = [
                            [float(v) for v in point.tolist()] for point in sample_points
                        ]
                    unique_nodes, inverse = np.unique(face_arr.reshape(-1), return_inverse=True)
                    vertices = p[:, unique_nodes].T
                    triangles = inverse.reshape((-1, 3))
                    selection["surface_node_ids"] = [int(v) for v in unique_nodes.tolist()]
                    selection["surface_vertices"] = [
                        [float(x), float(y), float(z)] for x, y, z in vertices.tolist()
                    ]
                    selection["surface_triangles"] = [
                        [int(a), int(b), int(c)] for a, b, c in triangles.tolist()
                    ]
        except Exception:
            pass
    return _selection_payload(None, [selection], selector_type)


def _mesh_axis_tolerance(p):
    import numpy as np

    spans = np.max(p, axis=1) - np.min(p, axis=1)
    positive = spans[spans > 1e-9]
    ref = float(np.max(positive)) if positive.size else 1.0
    return max(1e-6, 0.0025 * ref)


def _mesh_tetrahedra(mesh_like):
    """Return tetra connectivity as a (4, n) array when available."""
    try:
        import numpy as np

        if isinstance(mesh_like, dict):
            if mesh_like.get("mesh") is not None:
                return _mesh_tetrahedra(mesh_like.get("mesh"))
            for key in ("t", "elements", "tetrahedra", "cells"):
                if mesh_like.get(key) is None:
                    continue
                arr = np.asarray(mesh_like.get(key), dtype=int)
                if arr.ndim != 2:
                    continue
                if arr.shape[0] >= 4:
                    return arr[:4, :]
                if arr.shape[1] >= 4:
                    return arr[:, :4].T
        if hasattr(mesh_like, "t"):
            arr = np.asarray(mesh_like.t, dtype=int)
            if arr.ndim == 2 and arr.shape[0] >= 4:
                return arr[:4, :]
            if arr.ndim == 2 and arr.shape[1] >= 4:
                return arr[:, :4].T
    except Exception:
        return None
    return None


def _mesh_boundary_face_data(mesh_like):
    """Extract exterior triangle faces, centers, normals and an edge scale."""
    try:
        import numpy as np

        p = _mesh_points(mesh_like)
        t = _mesh_tetrahedra(mesh_like)
        if p is None or t is None or t.size == 0:
            return None

        pts = p.T
        n_nodes = p.shape[1]
        t = t[:, np.all((t >= 0) & (t < n_nodes), axis=0)]
        if t.size == 0:
            return None

        face_patterns = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        face_counts = {}
        face_nodes = {}
        for elem in t.T:
            for pattern in face_patterns:
                face = tuple(int(elem[i]) for i in pattern)
                key = tuple(sorted(face))
                face_counts[key] = face_counts.get(key, 0) + 1
                face_nodes[key] = face

        boundary = [face_nodes[key] for key, count in face_counts.items() if count == 1]
        if not boundary:
            return None

        faces = np.asarray(boundary, dtype=int)
        tri = pts[faces]
        centers = tri.mean(axis=1)
        normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
        areas2 = np.linalg.norm(normals, axis=1)
        valid = areas2 > 1e-12
        if not np.any(valid):
            return None
        tri = tri[valid]
        faces = faces[valid]
        centers = centers[valid]
        normals = normals[valid]
        areas2 = areas2[valid]
        normals = normals / areas2[:, None]

        body_center = pts.mean(axis=0)
        inward = np.einsum("ij,ij->i", normals, centers - body_center) < 0.0
        normals[inward] *= -1.0

        edge_lengths = np.concatenate([
            np.linalg.norm(tri[:, 1, :] - tri[:, 0, :], axis=1),
            np.linalg.norm(tri[:, 2, :] - tri[:, 1, :], axis=1),
            np.linalg.norm(tri[:, 0, :] - tri[:, 2, :], axis=1),
        ])
        edge_lengths = edge_lengths[edge_lengths > 1e-9]
        edge_scale = float(np.median(edge_lengths)) if edge_lengths.size else _mesh_axis_tolerance(p)

        return {
            "faces": faces,
            "centers": centers,
            "normals": normals,
            "areas": 0.5 * areas2,
            "edge_scale": edge_scale,
        }
    except Exception:
        return None


def _mesh_direction_node_ids(mesh_like, selector):
    """Select exterior nodes on the end face implied by a Direction selector."""
    ids, _faces = _mesh_direction_selection(mesh_like, selector)
    return ids


def _mesh_direction_selection(mesh_like, selector):
    """Select exterior nodes and surface triangles for a directional mesh face."""
    import numpy as np

    p = _mesh_points(mesh_like)
    if p is None:
        return np.array([], dtype=int), None

    axis = {'X': 0, 'Y': 1, 'Z': 2}.get(selector[-1], 2)
    sign = -1.0 if selector.startswith('<') else 1.0
    data = _mesh_boundary_face_data(mesh_like)
    if data is None:
        tol = _mesh_axis_tolerance(p)
        limit = float(np.min(p[axis]) if sign < 0.0 else np.max(p[axis]))
        if sign < 0.0:
            return np.where(p[axis] <= limit + tol)[0].astype(int), None
        return np.where(p[axis] >= limit - tol)[0].astype(int), None

    centers = data["centers"]
    normals = data["normals"]
    faces = data["faces"]
    axis_span = float(np.max(p[axis]) - np.min(p[axis]))
    edge_scale = float(data.get("edge_scale") or 0.0)
    tol = max(
        1e-6,
        0.005 * axis_span,
        min(3.0 * edge_scale, 0.08 * axis_span) if axis_span > 1e-9 else 3.0 * edge_scale,
    )
    direction = np.zeros(3, dtype=float)
    direction[axis] = sign

    normal_mask = np.dot(normals, direction) >= 0.30
    if np.any(normal_mask):
        candidate_centers = centers[normal_mask]
        extreme = float(np.min(candidate_centers[:, axis]) if sign < 0.0 else np.max(candidate_centers[:, axis]))
        if sign < 0.0:
            near_mask = centers[:, axis] <= extreme + tol
        else:
            near_mask = centers[:, axis] >= extreme - tol
        mask = normal_mask & near_mask
    else:
        extreme = float(np.min(centers[:, axis]) if sign < 0.0 else np.max(centers[:, axis]))
        if sign < 0.0:
            mask = centers[:, axis] <= extreme + tol
        else:
            mask = centers[:, axis] >= extreme - tol

    surface_faces = faces[mask] if np.any(mask) else np.empty((0, 3), dtype=int)
    ids = np.unique(surface_faces.reshape(-1)) if surface_faces.size else np.array([], dtype=int)
    if ids.size == 0:
        tol = _mesh_axis_tolerance(p)
        limit = float(np.min(p[axis]) if sign < 0.0 else np.max(p[axis]))
        if sign < 0.0:
            ids = np.where(p[axis] <= limit + tol)[0]
        else:
            ids = np.where(p[axis] >= limit - tol)[0]
        surface_faces = None
    return ids.astype(int), surface_faces


def _split_surface_face_components(surface_faces):
    """Split selected exterior triangles into connected surface patches."""
    import numpy as np

    face_arr = np.asarray(surface_faces, dtype=int)
    if face_arr.ndim != 2 or face_arr.shape[1] < 3 or face_arr.size == 0:
        return []
    face_arr = face_arr[:, :3]

    edge_to_faces = {}
    for face_idx, tri in enumerate(face_arr):
        a, b, c = [int(v) for v in tri[:3]]
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_to_faces.setdefault(key, []).append(face_idx)

    adjacency = [set() for _ in range(len(face_arr))]
    for owners in edge_to_faces.values():
        if len(owners) < 2:
            continue
        for owner in owners:
            adjacency[owner].update(v for v in owners if v != owner)

    components = []
    visited = np.zeros(len(face_arr), dtype=bool)
    for start in range(len(face_arr)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        current = []
        while stack:
            idx = stack.pop()
            current.append(idx)
            for nxt in adjacency[idx]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(face_arr[np.asarray(current, dtype=int)])
    return components


def _surface_component_metrics(mesh_like, surface_faces):
    import numpy as np

    p = _mesh_points(mesh_like)
    face_arr = np.asarray(surface_faces, dtype=int)
    if p is None or face_arr.ndim != 2 or face_arr.shape[1] < 3 or face_arr.size == 0:
        return 0.0, np.zeros(3, dtype=float)
    face_arr = face_arr[:, :3]
    valid = np.all((face_arr >= 0) & (face_arr < p.shape[1]), axis=1)
    face_arr = face_arr[valid]
    if face_arr.size == 0:
        return 0.0, np.zeros(3, dtype=float)
    tri_pts = p[:, face_arr].transpose(1, 2, 0)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri_pts[:, 1, :] - tri_pts[:, 0, :],
                 tri_pts[:, 2, :] - tri_pts[:, 0, :]),
        axis=1,
    )
    total_area = float(np.sum(areas))
    if total_area > 1e-12:
        center = np.sum(tri_pts.mean(axis=1) * areas[:, None], axis=0) / total_area
    else:
        center = tri_pts.reshape((-1, 3)).mean(axis=0)
    return total_area, center


def _mesh_direction_components(mesh_like, selector):
    """Return connected patch components for one virtual mesh direction."""
    import numpy as np

    _ids, surface_faces = _mesh_direction_selection(mesh_like, selector)
    if surface_faces is None or len(surface_faces) == 0:
        return []

    components = _split_surface_face_components(surface_faces)
    decorated = []
    for component in components:
        area, center = _surface_component_metrics(mesh_like, component)
        if area <= 1e-12:
            continue
        decorated.append((area, center, np.asarray(component, dtype=int)))
    decorated.sort(
        key=lambda item: (
            -float(item[0]),
            float(item[1][0]),
            float(item[1][1]),
            float(item[1][2]),
        )
    )
    return [component for _area, _center, component in decorated]


def _mesh_component_selection(mesh_like, stored_index):
    """Resolve a stored connected-patch index to nodes and exterior triangles."""
    import numpy as np

    local = int(stored_index) - _MESH_COMPONENT_INDEX_BASE
    if local < 0:
        return None, np.array([], dtype=int), None
    direction_index = local // _MESH_COMPONENT_INDEX_STRIDE
    component_index = local % _MESH_COMPONENT_INDEX_STRIDE
    if direction_index < 0 or direction_index >= len(_MESH_FACE_DIRECTIONS):
        return None, np.array([], dtype=int), None

    selector = _MESH_FACE_DIRECTIONS[int(direction_index)]
    components = _mesh_direction_components(mesh_like, selector)
    if component_index < 0 or component_index >= len(components):
        return selector, np.array([], dtype=int), None

    surface_faces = components[int(component_index)]
    ids = np.unique(surface_faces.reshape(-1)) if surface_faces.size else np.array([], dtype=int)
    return selector, ids.astype(int), surface_faces

class SelectFaceNode(CadQueryNode):
    """Select a face robustly based on geometric properties."""
    __identifier__ = 'com.cad.select_face'
    NODE_NAME = 'Select Face'

    def __init__(self):
        super(SelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))
        self.create_property('selector_type', 'Direction', items=['Direction', 'NearestToPoint', 'Index', 'Largest Area', 'Tag', 'Box', 'Coordinate Range'], widget_type='combo')
        self.create_property('direction', '>Z', widget_type='string')
        self.create_property('near_x', 0.0, widget_type='float')
        self.create_property('near_y', 0.0, widget_type='float')
        self.create_property('near_z', 0.0, widget_type='float')
        
        # New Box properties
        self.create_property('box_min_x', -10.0, widget_type='float')
        self.create_property('box_max_x', 10.0, widget_type='float')
        self.create_property('box_min_y', -10.0, widget_type='float')
        self.create_property('box_max_y', 10.0, widget_type='float')
        self.create_property('box_min_z', -10.0, widget_type='float')
        self.create_property('box_max_z', 10.0, widget_type='float')
        
        # New Coordinate Range property
        self.create_property('range_expr', '(x > 0) & (y < 20)', widget_type='string')
        
        self.create_property('face_index', 0, widget_type='int')
        self.create_property('tag', 'top', widget_type='string')

    def run(self):
        raw_input = resolve_any_input(self.get_input('shape'))
        method = _canonical_selector_type(self.get_property('selector_type'))

        mesh_points = _mesh_points(raw_input)
        if mesh_points is not None:
            try:
                import numpy as np

                p = mesh_points
                tol = _mesh_axis_tolerance(p)
                ids = np.array([], dtype=int)
                label = method

                if method == 'Direction':
                    selector = _canonical_face_direction(self.get_property('direction'))
                    ids, surface_faces = _mesh_direction_selection(raw_input, selector)
                    label = selector

                elif method == 'NearestToPoint':
                    pt = np.asarray([
                        float(self.get_property('near_x') or 0.0),
                        float(self.get_property('near_y') or 0.0),
                        float(self.get_property('near_z') or 0.0),
                    ])
                    dist = np.linalg.norm(p.T - pt, axis=1)
                    nearest = int(np.argmin(dist))
                    band = max(tol, float(dist[nearest]) + 1e-9)
                    ids = np.where(dist <= band)[0]
                    label = f"nearest {pt.tolist()}"

                elif method == 'Index':
                    idx = int(self.get_property('face_index') or 0) % len(_MESH_FACE_DIRECTIONS)
                    selector = _MESH_FACE_DIRECTIONS[idx]
                    ids, surface_faces = _mesh_direction_selection(raw_input, selector)
                    label = selector

                elif method == 'Largest Area':
                    best = None
                    for selector in _MESH_FACE_DIRECTIONS:
                        current, current_faces = _mesh_direction_selection(raw_input, selector)
                        if best is None or len(current) > len(best[1]):
                            best = (selector, current, current_faces)
                    if best is not None:
                        label, ids, surface_faces = best

                elif method == 'Box':
                    min_pt = np.asarray([
                        float(self.get_property('box_min_x') or 0.0),
                        float(self.get_property('box_min_y') or 0.0),
                        float(self.get_property('box_min_z') or 0.0),
                    ])
                    max_pt = np.asarray([
                        float(self.get_property('box_max_x') or 0.0),
                        float(self.get_property('box_max_y') or 0.0),
                        float(self.get_property('box_max_z') or 0.0),
                    ])
                    lo = np.minimum(min_pt, max_pt)
                    hi = np.maximum(min_pt, max_pt)
                    mask = np.all((p.T >= lo) & (p.T <= hi), axis=1)
                    ids = np.where(mask)[0]
                    label = "box"

                elif method == 'Coordinate Range':
                    expr = str(self.get_property('range_expr') or '').strip()
                    from pylcss.solver_backends.common import nodes_matching_condition

                    class _MeshProxy:
                        pass

                    proxy = _MeshProxy()
                    proxy.p = p
                    ids = nodes_matching_condition(proxy, expr, label="Select Face mesh range")
                    label = expr

                payload = _mesh_selection_payload(
                    raw_input, ids, method, label,
                    surface_faces=locals().get('surface_faces'),
                )
                if payload is None:
                    self.set_error("No mesh nodes matched the selector")
                    return None
                return payload
            except Exception as e:
                logger.error("SelectFaceNode (%s): mesh selection failed: %s", self.NODE_NAME, e)
                self.set_error(f"Mesh face selection failed: {e}")
                return None

        if raw_input is not None and any(
            hasattr(raw_input, attr)
            for attr in ('val', 'tessellate', 'faces', 'extrude', 'edges', 'toCompound', 'add')
        ):
            shape_input = raw_input
        else:
            shape_input = resolve_shape_input(self.get_input('shape'))
        if not shape_input:
            return None

        # Convert Assembly to Compound if needed
        if hasattr(shape_input, 'toCompound'):
            try:
                shape_val = shape_input.toCompound()
            except Exception:
                shape_val = shape_input
        else:
            shape_val = shape_input

        # Wrap in a Workplane to ensure .faces() returns a Workplane object with .vals()
        if isinstance(shape_val, cq.Workplane):
            obj = shape_val
        else:
            obj = cq.Workplane("XY").newObject([shape_val])

        try:
            if method == 'Direction':
                selector = _canonical_face_direction(self.get_property('direction'))
                face_selection = obj.faces(selector)
                faces = face_selection.vals()
                logger.debug("SelectFaceNode (%s): Direction %s found %d faces", self.NODE_NAME, selector, len(faces))
                if not faces:
                    self.set_error("No faces found with selector")
                    return None

                try:
                    wp = face_selection.workplane()
                except Exception:
                    wp = None
                return _selection_payload(wp, faces, method)

            elif method == 'NearestToPoint':
                pt = (self.get_property('near_x'), self.get_property('near_y'), self.get_property('near_z'))
                logger.debug("SelectFaceNode (%s): Point=%s", self.NODE_NAME, pt)
                face_selection = obj.faces(cq.NearestToPointSelector(pt))
                faces = face_selection.vals()
                logger.debug("SelectFaceNode (%s): NearestToPoint found %d faces", self.NODE_NAME, len(faces))
                if not faces:
                    self.set_error("No faces found near point")
                    return None

                try:
                    wp = face_selection.workplane()
                except Exception:
                    wp = None
                return _selection_payload(wp, faces, method)

            elif method == 'Index':
                idx = int(self.get_property('face_index'))
                all_faces = obj.faces().vals()
                logger.debug("SelectFaceNode (%s): Index=%d, Total faces=%d", self.NODE_NAME, idx, len(all_faces))
                if 0 <= idx < len(all_faces):
                    face = all_faces[idx]
                    wp = obj.newObject([face]).workplane()
                    return _selection_payload(wp, [face], method)
                else:
                    self.set_error(f"Face index {idx} out of range")
                    return None

            elif method == 'Largest Area':
                all_faces = obj.faces().vals()
                if not all_faces:
                    logger.debug("SelectFaceNode (%s): NO FACES FOUND AT ALL", self.NODE_NAME)
                    return None
                sorted_faces = sorted(all_faces, key=lambda f: f.Area(), reverse=True)
                largest_face = sorted_faces[0]
                wp = obj.newObject([largest_face]).workplane()
                return _selection_payload(wp, [largest_face], method)

            elif method == 'Tag':
                tag_name = self.get_property('tag')
                face_selection = obj.faces(tag=tag_name)
                faces = face_selection.vals()
                logger.debug("SelectFaceNode (%s): Tag %s found %d faces", self.NODE_NAME, tag_name, len(faces))
                if not faces:
                    return None
                return _selection_payload(face_selection.workplane(), faces, method)

            elif method == 'Box':
                # Custom Box Selector
                min_pt = (self.get_property('box_min_x'), self.get_property('box_min_y'), self.get_property('box_min_z'))
                max_pt = (self.get_property('box_max_x'), self.get_property('box_max_y'), self.get_property('box_max_z'))

                # Check center of faces against box
                def in_box(f):
                    c = f.Center()
                    return (min_pt[0] <= c.x <= max_pt[0] and
                            min_pt[1] <= c.y <= max_pt[1] and
                            min_pt[2] <= c.z <= max_pt[2])

                all_faces = obj.faces().vals()
                faces = [f for f in all_faces if in_box(f)]
                logger.debug("SelectFaceNode (%s): Box found %d faces", self.NODE_NAME, len(faces))
                if not faces:
                    return None

                new_wp = obj.newObject(faces)
                return _selection_payload(new_wp, faces, method)

            elif method == 'Coordinate Range':
                # Selector by simpleeval expression on face center
                expr = self.get_property('range_expr')
                all_faces = obj.faces().vals()
                faces = []

                # Try to use simpleeval if available
                try:
                    from simpleeval import simple_eval
                except ImportError:
                    simple_eval = None

                for f in all_faces:
                    c = f.Center()
                    try:
                        if simple_eval:
                            res = simple_eval(expr, names={'x': c.x, 'y': c.y, 'z': c.z})
                        else:
                            res = eval(expr, {"__builtins__": None}, {'x': c.x, 'y': c.y, 'z': c.z})
                        if res:
                            faces.append(f)
                    except Exception:
                        continue

                logger.debug("SelectFaceNode (%s): Coordinate Range found %d faces", self.NODE_NAME, len(faces))
                if not faces:
                    return None

                new_wp = obj.newObject(faces)
                return _selection_payload(new_wp, faces, method)

        except Exception as e:
            logger.error("SelectFaceNode (%s): %s", self.NODE_NAME, e)
            self.set_error(f"Face selection failed: {e}")
            return None

class InteractiveSelectFaceNode(CadQueryNode):
    """
    Select faces by interactively clicking them in the 3D viewport.

    This node stores a list of face indices (integers) chosen by the user
    when they click 'Pick Faces in 3D Viewer' in the Properties Panel.
    Its output is identical to SelectFaceNode — a dict with keys
    ``{'workplane', 'face', 'faces'}`` — so it is a drop-in replacement
    for any downstream FEA node.
    """
    __identifier__ = 'com.cad.select_face_interactive'
    NODE_NAME = 'Select Face (Interactive)'

    def __init__(self):
        super(InteractiveSelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))

        # Comma-separated face indices, e.g. "0,2,5"
        # Updated programmatically by the Properties Panel picking session.
        self.create_property('picked_face_indices', '', widget_type='string')
        # Human-readable label shown in the Properties Panel.
        self.create_property('selection_label', 'No faces selected', widget_type='string')

    # ------------------------------------------------------------------
    # Public helper: called by the Properties Panel after picking
    # ------------------------------------------------------------------
    def set_picked_faces(self, face_indices):
        """Store a list of face indices and update the label."""
        indices_str = ','.join(str(i) for i in face_indices)
        self.set_property('picked_face_indices', indices_str)
        n = len(face_indices)
        if n == 0:
            label = 'No faces selected'
        elif n == 1:
            label = f'1 face selected  (idx: {face_indices[0]})'
        else:
            label = f'{n} faces selected  (idx: {", ".join(str(i) for i in face_indices)})'
        self.set_property('selection_label', label)

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------
    def run(self):
        # Parse stored indices
        raw = self.get_property('picked_face_indices') or ''
        face_indices = []
        for tok in raw.split(','):
            tok = tok.strip()
            if tok.isdigit():
                face_indices.append(int(tok))

        if not face_indices:
            self.set_error('No faces picked yet — click "Pick Faces in 3D Viewer"')
            return None

        raw_input = resolve_any_input(self.get_input('shape'))
        if _mesh_points(raw_input) is not None:
            selected = []
            for idx in face_indices:
                if idx >= _MESH_COMPONENT_INDEX_BASE:
                    selector, ids, surface_faces = _mesh_component_selection(raw_input, idx)
                    if selector is None or ids.size == 0:
                        logger.warning(
                            "InteractiveSelectFaceNode: mesh patch index %s could not "
                            "be resolved on this mesh - skipped",
                            idx,
                        )
                        continue
                    label = f"patch {idx} / {selector}"

                elif 0 <= idx < len(_MESH_FACE_DIRECTIONS):
                    selector = _MESH_FACE_DIRECTIONS[idx]
                    label = f"idx {idx} / {selector}"
                    # Legacy saved projects used 0..5 to mean whole virtual
                    # directional faces.  Keep that behavior for old examples,
                    # while new GUI picks store connected patch ids >= 1000.
                    ids, surface_faces = _mesh_direction_selection(raw_input, selector)
                else:
                    logger.warning(
                        "InteractiveSelectFaceNode: mesh face index %s out of range "
                        "(valid 0..%s or connected patch ids >= %s) - skipped",
                        idx,
                        len(_MESH_FACE_DIRECTIONS) - 1,
                        _MESH_COMPONENT_INDEX_BASE,
                    )
                    continue

                payload = _mesh_selection_payload(
                    raw_input, ids, "Interactive", label,
                    surface_faces=surface_faces,
                )
                if isinstance(payload, dict):
                    selected.extend([f for f in payload.get("faces") or [] if f is not None])

            if not selected:
                self.set_error("None of the stored mesh face indices matched this mesh")
                return None
            return _selection_payload(None, selected, "Interactive")

        shape_input = resolve_shape_input(self.get_input('shape'))
        if not shape_input:
            return None

        # Resolve shape
        if hasattr(shape_input, 'toCompound'):
            try:
                shape_val = shape_input.toCompound()
            except Exception:
                shape_val = shape_input
        else:
            shape_val = shape_input

        if isinstance(shape_val, cq.Workplane):
            obj = shape_val
        else:
            obj = cq.Workplane("XY").newObject([shape_val])

        try:
            all_faces = obj.faces().vals()
        except Exception as e:
            self.set_error(f"Cannot enumerate faces: {e}")
            return None

        selected = []
        for idx in face_indices:
            if 0 <= idx < len(all_faces):
                selected.append(all_faces[idx])
            else:
                logger.warning(f"InteractiveSelectFaceNode: face index {idx} out of range "
                               f"({len(all_faces)} faces total) — skipped")

        if not selected:
            self.set_error(f"None of the stored face indices are valid for this shape "
                           f"(shape has {len(all_faces)} faces)")
            return None

        try:
            wp = obj.newObject(selected).workplane()
        except Exception:
            wp = None

        return _selection_payload(wp, selected, "Interactive")
