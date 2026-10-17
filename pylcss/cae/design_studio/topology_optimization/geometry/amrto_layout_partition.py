# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Step 3: Sparse quadrilateral layouts division.

Reference:
Section 2.5: Generation of sparse quadrilateral layouts (step 3)
In AMRTO (Ren et al., 2025):
"This Section takes the semi-regular mesh generated in Section 2.4 as input to
create sparse quadrilateral layouts using the generalized motorcycle graphs method...
Each ideal layout is a simply connected region with a boundary and is topologically
equivalent to a disk."
"""

from __future__ import annotations

import glob
import heapq
import logging
import math
import json
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pylcss.process_utils import headless_subprocess_kwargs

from .mesh_topology import triangulate_quad_faces
from .amrto_common import QuadLayout, get_gmcg_headless_executable
from .amrto_disk_repairs import repair_quad_patch_to_disks, repair_same_side_boundary_triangles

logger = logging.getLogger(__name__)


def triangle_area_3d(cor_1: np.ndarray, cor_2: np.ndarray, cor_3: np.ndarray) -> float:
    """Calculate the area of a 3D spatial triangle."""
    len_a = np.linalg.norm(cor_1 - cor_2)
    len_b = np.linalg.norm(cor_1 - cor_3)
    len_c = np.linalg.norm(cor_3 - cor_2)
    p0 = (len_a + len_b + len_c) / 2.0
    val = p0 * (p0 - len_a) * (p0 - len_b) * (p0 - len_c)
    return math.sqrt(max(0.0, val))


def triangulate_quads_clean(
    verts: np.ndarray,
    quads_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appendix D Improvement 2: choose the diagonal with least area imbalance.

    Boundary ears use the same rule. Improvement 4 subsequently prevents their
    three corners from collapsing onto one square side, without altering quads.
    """
    verts = np.asarray(verts, dtype=float)
    quads_arr = np.asarray(quads_arr, dtype=np.int64)
    tri_arr = triangulate_quad_faces(verts, quads_arr)
    tri_arr = tri_arr.reshape(2, len(quads_arr), 3).transpose(1, 0, 2).reshape(-1, 3)
    used_v = np.unique(tri_arr)
    v_map = {old: new for new, old in enumerate(used_v)}
    local_tris = np.asarray([[v_map[int(v)] for v in t] for t in tri_arr], dtype=np.int64)
    local_quads = np.asarray([[v_map[int(v)] for v in q] for q in quads_arr], dtype=np.int64)
    return used_v, local_tris, local_quads


def load_layout_from_obj(obj_path: str | Path, layout_id: int = 0,
                         *, repair_boundary_triangles: bool = True) -> Optional[QuadLayout]:
    """Load a sparse quad layout OBJ file formatted as in AMRTO output_quad."""
    p = Path(obj_path)
    if not p.is_file():
        return None

    raw_points = []
    pts_vt = []
    quads = []
    vertex_uv = {}

    with open(p, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                raw_points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt":
                pts_vt.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                q_idx = []
                for chunk in parts[1:]:
                    tokens = chunk.split("/")
                    tok = tokens[0]
                    if tok:
                        index = int(tok)
                        index = index - 1 if index > 0 else len(raw_points) + index
                        q_idx.append(index)
                        if len(tokens) > 1 and tokens[1]:
                            vt_index = int(tokens[1])
                            vt_index = vt_index - 1 if vt_index > 0 else len(pts_vt) + vt_index
                            vertex_uv[index] = pts_vt[vt_index]
                if len(q_idx) == 4:
                    if len(set(q_idx)) != 4:
                        raise ValueError(f"Degenerate quad in layout {p}")
                    quads.append(q_idx)
                else:
                    raise ValueError(f"Layout {p} must contain quadrilateral faces")

    if not raw_points or not quads:
        return None

    points_quad = np.asarray(raw_points, dtype=float)
    quads_arr = np.asarray(quads, dtype=np.int64)

    # Convert quads into triangles using area-minimizing diagonal split (as in AMRTO PYTOCAD)
    tri_arr = triangulate_quad_faces(points_quad, quads_arr)

    # Re-index unique vertices used by this patch
    used_v = np.unique(tri_arr)
    v_map = {old: new for new, old in enumerate(used_v)}
    local_verts = points_quad[used_v]
    local_tris = np.asarray([[v_map[int(v)] for v in t] for t in tri_arr], dtype=np.int64)
    local_quads = np.asarray([[v_map[int(v)] for v in q] for q in quads_arr], dtype=np.int64)

    layout = QuadLayout(
        layout_id=layout_id,
        vertices=local_verts,
        triangles=local_tris,
        quads=local_quads,
    )
    layout.source_vertex_indices = used_v

    # Find boundary edges & corners
    aligned_uv = [vertex_uv.get(int(v), [np.nan, np.nan]) for v in used_v]
    extract_layout_boundary_and_corners(layout, pts_vt=aligned_uv)
    if repair_boundary_triangles:
        layout = _apply_D4(layout)
    return layout


def _boundary_loop(triangles: np.ndarray, vertex_count: int) -> list[int]:
    """Validate a connected manifold disk and trace its only boundary loop."""
    edge_faces = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        if len(set(triangle)) != 3:
            raise ValueError("A layout contains a degenerate triangle")
        for u, v in zip(triangle, np.roll(triangle, -1)):
            edge_faces[tuple(sorted((int(u), int(v))))].append(face_index)
    if not edge_faces or any(len(owners) > 2 for owners in edge_faces.values()):
        raise ValueError("A layout is empty or nonmanifold")
    used = set(int(v) for v in triangles.flat)
    if len(used) != vertex_count or len(used) - len(edge_faces) + len(triangles) != 1:
        raise ValueError("A layout must have disk topology (Euler characteristic 1)")
    neighbors = defaultdict(set)
    for owners in edge_faces.values():
        if len(owners) == 2:
            neighbors[owners[0]].add(owners[1])
            neighbors[owners[1]].add(owners[0])
    reached, queue = {0}, deque([0])
    while queue:
        for neighbor in neighbors[queue.popleft()]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    if len(reached) != len(triangles):
        raise ValueError("A layout must be connected through its edges")
    boundary_adj = defaultdict(list)
    for (u, v), owners in edge_faces.items():
        if len(owners) == 1:
            boundary_adj[u].append(v)
            boundary_adj[v].append(u)
    if len(boundary_adj) < 4 or any(len(nbrs) != 2 for nbrs in boundary_adj.values()):
        raise ValueError("A quadrilateral layout requires one simple boundary with at least four vertices")
    start = min(boundary_adj)
    # Follow the input winding where it is consistent, preserving outward normals.
    first = None
    for triangle in triangles:
        for u, v in zip(triangle, np.roll(triangle, -1)):
            if int(u) == start and int(v) in boundary_adj[start]:
                first = int(v)
                break
        if first is not None:
            break
    previous, current = start, first if first is not None else boundary_adj[start][0]
    loop = [start]
    while current != start:
        if current in loop:
            raise ValueError("A layout boundary self-intersects topologically")
        loop.append(current)
        candidates = boundary_adj[current]
        following = candidates[0] if candidates[0] != previous else candidates[1]
        previous, current = current, following
    if len(loop) != len(boundary_adj):
        raise ValueError("A layout contains more than one boundary loop")
    return loop


def partition_nondisk_patch_into_disks(
    verts: np.ndarray,
    patch_quads: np.ndarray,
    depth: int = 0,
    max_depth: int = 4,
) -> list[np.ndarray]:
    """Non-paper PCA partition fallback for topology unsupported by D1 cuts.

    This changes patch boundaries, retaining physical quads. It is explicitly
    distinct from Appendix D Improvement 1's coincident shortest-path seams.
    """
    patch_quads = np.asarray(patch_quads, dtype=np.int64)
    if len(patch_quads) == 0:
        return []
    try:
        used, local_tris, _ = triangulate_quads_clean(verts, patch_quads)
        _boundary_loop(local_tris, len(used))
        return [patch_quads]
    except Exception:
        pass

    if depth >= max_depth or len(patch_quads) < 8:
        return [patch_quads]

    centers = verts[patch_quads].mean(axis=1)
    cov = np.cov(centers.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    best_split = None
    for ax_idx in [2, 1, 0]:
        axis = eigvecs[:, ax_idx]
        proj = centers @ axis
        median = np.median(proj)
        m1 = proj <= median
        m2 = ~m1
        if m1.sum() == 0 or m2.sum() == 0:
            continue
        try:
            used1, tris1, _ = triangulate_quads_clean(verts, patch_quads[m1])
            _boundary_loop(tris1, len(used1))
            d1 = True
        except Exception:
            d1 = False
        try:
            used2, tris2, _ = triangulate_quads_clean(verts, patch_quads[m2])
            _boundary_loop(tris2, len(used2))
            d2 = True
        except Exception:
            d2 = False

        if d1 and d2:
            return [patch_quads[m1], patch_quads[m2]]
        if best_split is None and m1.sum() > 0 and m2.sum() > 0:
            best_split = (patch_quads[m1], patch_quads[m2])

    if best_split is not None:
        res = []
        res.extend(partition_nondisk_patch_into_disks(verts, best_split[0], depth + 1, max_depth))
        res.extend(partition_nondisk_patch_into_disks(verts, best_split[1], depth + 1, max_depth))
        return res

    return [patch_quads]


def _required_square_corners(triangles: np.ndarray, loop: list[int]) -> set[int]:
    """Boundary triangle ears must become square corners to retain positive area."""
    boundary_edges = {tuple(sorted((u, v))) for u, v in zip(loop, loop[1:] + loop[:1])}
    required = set()
    for triangle in triangles:
        edges = [tuple(sorted((int(u), int(v)))) for u, v in zip(triangle, np.roll(triangle, -1))]
        on_boundary = [edge for edge in edges if edge in boundary_edges]
        if len(on_boundary) == 2:
            required.update(set(on_boundary[0]).intersection(on_boundary[1]))
    return required


def extract_layout_boundary_and_corners(
    layout: QuadLayout,
    pts_vt: Optional[list] = None,
    used_indices: Optional[np.ndarray] = None,
) -> None:
    """Find four distinct corners and paths which cover the complete disk boundary."""
    verts = layout.vertices
    loop = _boundary_loop(layout.triangles, len(verts))
    corners = None
    if pts_vt is not None and len(pts_vt):
        vt_arr = np.asarray(pts_vt, dtype=float)
        if used_indices is not None:
            vt_arr = vt_arr[used_indices]
        if vt_arr.shape == (len(verts), 2) and np.isfinite(vt_arr).all():
            boundary_uv = vt_arr[loop]
            low, high = boundary_uv.min(axis=0), boundary_uv.max(axis=0)
            span = high - low
            if (span > 0).all():
                normalized = (boundary_uv - low) / span
                targets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
                selected = [loop[int(np.argmin(np.sum((normalized - t) ** 2, axis=1)))] for t in targets]
                if len(set(selected)) == 4:
                    corners = selected
    if corners is None:
        # Prefer geometric turns, particularly the four corners of an actual rectangle.
        points = verts[loop]
        incoming = points - np.roll(points, 1, axis=0)
        outgoing = np.roll(points, -1, axis=0) - points
        denominator = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        cosine = np.divide(np.einsum("ij,ij->i", incoming, outgoing), denominator,
                           out=np.ones(len(loop)), where=denominator > 0)
        turns = np.arccos(np.clip(cosine, -1, 1))
        sharp = np.flatnonzero(turns > np.deg2rad(60))  # Raised from 35°; smooth surfaces need a higher threshold
        if len(sharp) == 4:
            corners = [loop[int(i)] for i in sharp]
        else:
            cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(outgoing, axis=1))]
            positions = [0]
            for fraction in (0.25, 0.5, 0.75):
                candidates = [i for i in range(1, len(loop)) if i not in positions]
                positions.append(min(candidates, key=lambda i: abs(cumulative[i] - fraction * cumulative[-1])))
            corners = [loop[i] for i in positions]
        required = _required_square_corners(layout.triangles, loop)
        if len(required) > 4:
            req_list = [v for v in loop if v in required]
            step = len(req_list) / 4.0
            corners = [req_list[int(round(i * step)) % len(req_list)] for i in range(4)]
        else:
            selected = [v for v in loop if v in required]
            for corner in corners:
                if len(selected) == 4:
                    break
                if corner not in selected:
                    selected.append(corner)
            corners = selected
        if len(set(corners)) < 4:
            step = len(loop) / 4.0
            corners = [loop[int(round(i * step)) % len(loop)] for i in range(4)]
    # Always use the cyclic order of the traced boundary; UV index order may be reversed.
    start_corner = corners[0]
    corner_set = set(corners)
    start = loop.index(start_corner)
    loop = loop[start:] + loop[:start]
    corners = [v for v in loop if v in corner_set]
    positions = [loop.index(v) for v in corners]
    paths = []
    for i in range(4):
        end = positions[i + 1] if i < 3 else len(loop)
        path = loop[positions[i]:end + 1] if i < 3 else loop[positions[i]:] + [loop[0]]
        paths.append(path)
    layout.corner_indices = corners
    layout.boundary_paths = paths


def _make_layout(vertices: np.ndarray, quads: np.ndarray, face_ids: list[int], layout_id: int,
                 *, triangulation=None, repair_boundary_triangles: bool = True) -> QuadLayout:
    sub_quads = quads[face_ids]
    used, local_tris, local_quads = (triangulate_quads_clean(vertices, sub_quads)
                                   if triangulation is None else triangulation)
    layout = QuadLayout(layout_id, vertices[used].copy(), local_tris, quads=local_quads)
    layout.source_vertex_indices = used
    extract_layout_boundary_and_corners(layout)
    layout.source_face_indices = list(face_ids)
    return _apply_D4(layout) if repair_boundary_triangles else layout


def _apply_D4(layout: QuadLayout) -> QuadLayout:
    """Treat boundary ears after corner selection; preserve valid layout identity."""
    repaired, report = repair_same_side_boundary_triangles(layout)
    if report["paper_D4_executed"]:
        return repaired
    if not hasattr(layout, "appendix_D4_report"):
        layout.appendix_D4_report = report
    return layout


def _remap_source_ids(local_ids: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
    """Preserve synthetic -1 interior IDs rather than treating them as an index."""
    local_ids = np.asarray(local_ids, dtype=np.int64)
    mapped = np.full(local_ids.shape, -1, dtype=np.int64)
    original = local_ids >= 0
    if np.any(local_ids[original] >= len(source_ids)):
        raise ValueError("A layout source vertex index lies outside its parent patch")
    mapped[original] = np.asarray(source_ids, dtype=np.int64)[local_ids[original]]
    return mapped


def _inherit_layout_correspondence(piece: QuadLayout, parent: QuadLayout) -> None:
    """Map only source mesh vertices; newly added D4 interiors keep ID -1."""
    parent_ids = np.asarray(getattr(parent, "source_vertex_indices", np.arange(len(parent.vertices))),
                            dtype=np.int64)
    piece.source_vertex_indices = _remap_source_ids(piece.source_vertex_indices, parent_ids)
    if getattr(parent, "require_exact_boundaries", False):
        piece.require_exact_boundaries = True
    if hasattr(parent, "source_face_indices") and hasattr(piece, "source_face_indices"):
        piece.source_face_indices = np.asarray(parent.source_face_indices)[piece.source_face_indices].tolist()
    if hasattr(parent, "protected_source_crease_edges"):
        boundary_edges = {tuple(sorted((tuple(map(float, piece.vertices[first])),
                                        tuple(map(float, piece.vertices[second])))))
                          for path in piece.boundary_paths for first, second in zip(path, path[1:])}
        piece.protected_source_crease_edges = sorted(
            boundary_edges.intersection(parent.protected_source_crease_edges))
        piece.source_crease_minimum_normal_dot = parent.source_crease_minimum_normal_dot


def _validate_source_quad_coverage(layouts, vertices, quads):
    """Only original quad corners need global IDs; synthetic interiors are extra."""
    exported = []
    for layout in layouts:
        ids = np.asarray(layout.source_vertex_indices, dtype=np.int64)
        if ids.shape != (len(layout.vertices),):
            raise ValueError("Disk partition source correspondence must cover every layout vertex")
        original = ids >= 0
        if (np.any(ids[original] >= len(vertices))
                or np.any(ids < -1)
                or not np.array_equal(layout.vertices[original], np.asarray(vertices)[ids[original]])):
            raise ValueError("Disk partition changed source vertex correspondence or coordinates")
        if np.any(ids[np.asarray(layout.quads)] < 0):
            raise ValueError("A synthetic interior vertex replaced an original quad corner")
        boundary = [vertex for path in layout.boundary_paths for vertex in path]
        if np.any(ids[boundary] < 0):
            raise ValueError("A synthetic interior vertex entered the shared boundary graph")
        exported.extend(tuple(sorted(map(int, ids[quad]))) for quad in layout.quads)
        _boundary_loop(layout.triangles, len(layout.vertices))
    if Counter(exported) != Counter(tuple(sorted(map(int, quad))) for quad in quads):
        raise ValueError("Disk layouts do not cover every input quad exactly once")
    return len(exported)


def _appendix_report(layouts):
    reports = [getattr(layout, "appendix_D4_report", {}) for layout in layouts]
    return {"paper_D2_executed": True,
            "triangulation_method": "Appendix D2 minimum difference of physical triangle areas",
            "paper_D4_executed": any(report.get("paper_D4_executed", False) for report in reports),
            "paper_D4_condition_satisfied": all(report.get("paper_D4_condition_satisfied", False) for report in reports),
            "D4_new_vertex_count": sum(report.get("new_vertex_count", 0) for report in reports),
            "D4_patch_reports": [{"patch": layout.layout_id, **report} for layout, report in zip(layouts, reports)
                                 if report.get("paper_D4_executed", False)],
            "paper_D3_executed": False,
            "D3_reason": "No curvature-selective patch simplification with fixed shared boundaries is implemented"}


def auto_partition_quad_mesh_into_layouts(
    vertices: np.ndarray,
    quads: np.ndarray,
    target_patch_size: int = 16,
    control_point_budget: int = 2500,
    fast_disk_growth: bool = False,
) -> List[QuadLayout]:
    """Topology-checked fallback; every input quad belongs to exactly one disk patch.

    This is a conservative clustering fallback, not a generalized motorcycle graph.
    """
    vertices = np.asarray(vertices, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    if not len(quads):
        return []
    if quads.ndim != 2 or quads.shape[1] != 4 or quads.min() < 0 or quads.max() >= len(vertices):
        raise ValueError("Invalid quadrilateral face indices")
    if any(len(set(q)) != 4 for q in quads):
        raise ValueError("A pure quad mesh cannot contain repeated face vertices")
    # D2 depends only on the unchanged four physical vertices of each quad.
    # Reuse those exact triangle pairs while trying growth candidates; every
    # candidate still receives the same boundary, corner and budget checks.
    used_all, triangles_all, _ = triangulate_quads_clean(vertices, quads)
    triangle_pairs = used_all[triangles_all].reshape(len(quads), 2, 3)

    def selected_triangulation(face_ids):
        selected_quads = quads[face_ids]
        used = np.unique(selected_quads)
        return (used, np.searchsorted(used, triangle_pairs[face_ids]).reshape(-1, 3),
                np.searchsorted(used, selected_quads))

    target_patch_size = max(1, int(target_patch_size))
    edge_to_quad = defaultdict(list)
    for q_index, quad in enumerate(quads):
        for u, v in zip(quad, np.roll(quad, -1)):
            edge_to_quad[tuple(sorted((int(u), int(v))))].append(q_index)
    neighbors = defaultdict(set)
    points = vertices[quads]
    normals = (np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
               + np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    if (lengths <= 0).any():
        raise ValueError("The quad mesh contains a zero-area face")
    normals /= lengths[:, None]
    for owners in edge_to_quad.values():
        if len(owners) > 2:
            raise ValueError("The quad mesh contains a nonmanifold edge")
        if len(owners) == 2:
            # Keep strong creases as patch boundaries instead of smoothing a
            # cuboid corner into one spline across several orthogonal faces.
            # Topology-optimized surfaces are smooth continuum geometry; only
            # suppress adjacency across genuine hard corners (> ~90 degrees).
            if np.dot(normals[owners[0]], normals[owners[1]]) < 0.0:
                continue
            neighbors[owners[0]].add(owners[1])
            neighbors[owners[1]].add(owners[0])
    visited, layouts = set(), []
    for start in range(len(quads)):
        if start in visited:
            continue
        patch, queued, queue = [], {start}, deque([start])
        patch_edges, patch_vertices = Counter(), set()
        while queue and len(patch) < target_patch_size:
            current = queue.popleft()
            if current in visited:
                continue
            candidate = patch + [current]
            try:
                if fast_disk_growth:
                    candidate_edges = patch_edges.copy()
                    candidate_edges.update(tuple(sorted((int(u), int(v))))
                                           for u, v in zip(quads[current], np.roll(quads[current], -1)))
                    candidate_vertices = patch_vertices.union(map(int, quads[current]))
                    boundary_degrees = Counter(vertex for edge, count in candidate_edges.items()
                                               if count == 1 for vertex in edge)
                    if (len(candidate_vertices) - len(candidate_edges) + len(candidate) != 1
                            or len(boundary_degrees) < 4 or any(count != 2 for count in boundary_degrees.values())):
                        raise ValueError("Candidate is not a manifold disk")
                else:
                    triangulation = selected_triangulation(candidate)
                    used_c, triangles, _ = triangulation
                    _boundary_loop(triangles, len(used_c))
                    prospective = _make_layout(vertices, quads, candidate, 0, triangulation=triangulation,
                                               repair_boundary_triangles=False)
                    paths = prospective.boundary_paths
                    poles_u = max(6, 4 + 3 * (len(paths[0]) + len(paths[2]) - 4))
                    poles_v = max(6, 4 + 3 * (len(paths[1]) + len(paths[3]) - 4))
                    if poles_u * poles_v > control_point_budget:
                        raise ValueError("Candidate exceeds the boundary-constrained control-net budget")
                    # D4 preserves graph sides, so it cannot change this budget.
                    # Reject oversized candidates before constructing synthetic
                    # interiors; accepted candidates retain every D4 guard.
                    _apply_D4(prospective)
            except ValueError:
                if not patch:
                    raise
                continue
            patch.append(current)
            if fast_disk_growth:
                patch_edges, patch_vertices = candidate_edges, candidate_vertices
            # Mark only faces actually included, not pending neighbors at the patch limit.
            visited.add(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor not in visited and neighbor not in queued:
                    queued.add(neighbor)
                    queue.append(neighbor)
        try:
            layout = _make_layout(vertices, quads, patch, len(layouts),
                                  triangulation=selected_triangulation(patch))
            paths = layout.boundary_paths
            poles_u = max(6, 4 + 3 * (len(paths[0]) + len(paths[2]) - 4))
            poles_v = max(6, 4 + 3 * (len(paths[1]) + len(paths[3]) - 4))
            if poles_u * poles_v > control_point_budget:
                raise ValueError("Patch exceeds the conservative control-net estimate")
        except ValueError:
            if not fast_disk_growth or len(patch) == 1:
                raise
            pieces = auto_partition_quad_mesh_into_layouts(
                vertices, quads[patch], target_patch_size=max(1, len(patch) // 2),
                control_point_budget=control_point_budget, fast_disk_growth=True)
            for piece in pieces:
                piece.source_face_indices = [patch[index] for index in piece.source_face_indices]
                piece.layout_id = len(layouts)
                layouts.append(piece)
            continue
        side_sets = [set(path) for path in layout.boundary_paths]
        collapsed = any(any(set(triangle).issubset(side) for side in side_sets)
                        for triangle in layout.triangles)
        if collapsed:
            if len(patch) > 1:
                # A BFS visitation prefix is connected, but its remaining
                # suffix need not be. Repartition the unchanged faces using
                # the same disk checks instead of slicing arbitrary IDs.
                pieces = auto_partition_quad_mesh_into_layouts(
                    vertices, quads[patch], target_patch_size=max(1, len(patch) // 2),
                    control_point_budget=control_point_budget, fast_disk_growth=fast_disk_growth)
                for piece in pieces:
                    piece.source_face_indices = [patch[index] for index in piece.source_face_indices]
                    piece.layout_id = len(layouts)
                    layouts.append(piece)
            else:
                layouts.append(layout)
        else:
            layouts.append(layout)
    covered = [face for layout in layouts for face in layout.source_face_indices]
    if sorted(covered) != list(range(len(quads))):
        raise RuntimeError("Sparse layout partition did not cover every quad exactly once")
    return layouts


def load_sparse_quad_layouts(
    layout_folder: Optional[str | Path] = None,
) -> List[QuadLayout]:
    """Scan and load pre-existing AMRTO quad layouts from disk."""
    search_dirs = []
    if layout_folder:
        search_dirs.append(Path(layout_folder))


    for d in search_dirs:
        if d.is_dir():
            files = sorted(glob.glob(str(d / "quad_*.obj")), key=lambda p: int(Path(p).stem.split("_")[-1]))
            if files:
                logger.info("Loading %d sparse quad layouts from %s", len(files), d)
                layouts = []
                for idx, f in enumerate(files):
                    lay = load_layout_from_obj(f, layout_id=idx)
                    if lay is not None and len(lay.vertices) >= 4:
                        layouts.append(lay)
                if layouts:
                    return layouts
    return []


def quad_mesh_from_layout_folder(folder):
    """Reassemble supplied chart quads without remeshing their shared geometry."""
    layouts = load_sparse_quad_layouts(folder)
    if not layouts:
        raise ValueError(f'No valid sparse quadrilateral layouts found in {folder}')
    vertices, faces = [], []
    offset = 0
    for layout in layouts:
        vertices.append(layout.vertices)
        faces.append(layout.quads + offset)
        offset += len(layout.vertices)
    points = np.concatenate(vertices)
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    quads = inverse[np.concatenate(faces)]
    if len({tuple(sorted(map(int, q))) for q in quads}) != len(quads):
        raise ValueError('Supplied layouts contain duplicate source quads')
    return unique, quads


def _match_layout_to_source(layout, vertices):
    """Restore global IDs/coordinates, retaining synthetic interior vertices."""
    from scipy.spatial import cKDTree

    ids = np.asarray(layout.source_vertex_indices, dtype=np.int64).copy()
    original = ids >= 0
    tolerance = max(1e-9, float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-6)
    points = layout.vertices[original]
    if not (np.all(ids[original] < len(vertices))
            and np.all(np.linalg.norm(vertices[ids[original]] - points, axis=1) <= tolerance)):
        distances, mapped = cKDTree(vertices).query(points)
        if np.any(distances > tolerance):
            raise ValueError('Supplied layouts do not match the source quad vertices')
        ids[original] = mapped
    layout.source_vertex_indices = ids
    layout.vertices[original] = vertices[ids[original]]


def _self_seam_source_ids(layout: QuadLayout) -> list[int]:
    """A CAD chart must not own both coincident sides of a D1 cut seam."""
    source = np.asarray(layout.source_vertex_indices, dtype=np.int64)
    boundary = set(vertex for path in layout.boundary_paths for vertex in path)
    counts = Counter(int(source[vertex]) for vertex in boundary if source[vertex] >= 0)
    return sorted(source_id for source_id, count in counts.items() if count > 1)


def _redivide_D1_self_seams(layouts, *, executable=None, maximum_native_passes=2, timeout_s=60.0):
    """Try repeated native GMCG, then explicitly split any surviving self-seam.

    Disk topology alone is insufficient for CAD sewing: two coincident boundary
    edges of the same spline face do not become an ordinary inter-face seam.
    Appendix D1 permits further GMCG division. Inputs retain coincident vertices
    as distinct mesh indices and exclude unused D4 interior sampling vertices.
    """
    if (not isinstance(maximum_native_passes, int) or maximum_native_passes < 0 or maximum_native_passes > 8
            or not np.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("D1 redivision requires bounded native passes and a positive timeout")
    executable = Path(executable) if executable is not None else get_gmcg_headless_executable()
    pending = [(layout, 0) for layout in reversed(layouts)]
    result, attempts, fallback_reports = [], [], []
    while pending:
        layout, passes = pending.pop()
        seam_ids = _self_seam_source_ids(layout)
        if not seam_ids:
            result.append(layout)
            continue
        children = None
        native_available = bool(executable and executable.is_file() and "headless" in executable.stem.lower())
        if passes < maximum_native_passes and native_available:
            attempt = {"pass": passes + 1, "input_quad_count": len(layout.quads),
                       "self_seam_source_vertex_count_before": len(seam_ids),
                       "executable": str(executable), "timeout_s": timeout_s,
                       "executed": True, "native_solver_completed": False}
            try:
                used = np.unique(layout.quads)
                vertices = layout.vertices[used]
                quads = np.searchsorted(used, layout.quads)
                with tempfile.TemporaryDirectory(prefix="pylcss_gmcg_D1_") as folder:
                    root = Path(folder)
                    input_obj = root / "disk.obj"
                    with input_obj.open("w", encoding="ascii") as stream:
                        for vertex in vertices:
                            stream.write("v " + " ".join(f"{float(x):.17g}" for x in vertex) + "\n")
                        for quad in quads:
                            stream.write("f " + " ".join(str(int(i) + 1) for i in quad) + "\n")
                    subprocess.run([str(executable), str(input_obj), str(root / "layouts")],
                                   check=True, capture_output=True, timeout=timeout_s,
                                   **headless_subprocess_kwargs())
                    children, native_report = _load_indexed_gmcg_layouts(
                        root / "layouts" / "quad_patch_indices.json", vertices, quads, 25,
                        gmcg_executable=executable, redivide_D1=False)
                face_lookup = defaultdict(deque)
                for index, face in enumerate(quads):
                    face_lookup[tuple(sorted(map(int, face)))].append(index)
                for child in children:
                    child_faces = [face_lookup[tuple(sorted(map(int, child.source_vertex_indices[quad])))].popleft()
                                   for quad in child.quads]
                    child.source_face_indices = np.asarray(layout.source_face_indices)[child_faces].tolist()
                    child.source_vertex_indices = _remap_source_ids(
                        child.source_vertex_indices, layout.source_vertex_indices[used])
                remaining = sum(bool(_self_seam_source_ids(child)) for child in children)
                attempt.update(native_solver_completed=True, output_layout_count=len(children),
                               remaining_self_seam_chart_count=remaining,
                               paper_stage_3=native_report["paper_stage_3"],
                               native_report=native_report)
                # An unchanged chart would produce the same graph again. Stop
                # this branch rather than retrying deterministic identical input.
                no_progress = (len(children) == 1 and len(children[0].quads) == len(layout.quads)
                               and len(_self_seam_source_ids(children[0])) >= len(seam_ids))
                attempt["no_progress"] = no_progress
                if no_progress:
                    children = None
                elif not native_report["paper_stage_3"]:
                    fallback_reports.append({"method": "Native output required non-paper corner/partition treatment",
                                             "native_pass": passes + 1})
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                stderr = getattr(error, "stderr", None)
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                attempt.update(failure=f"{type(error).__name__}: {error}",
                               native_error=stderr[-2000:] if stderr else None)
                children = None
            attempts.append(attempt)
        if children is None:
            if len(layout.quads) < 2:
                raise ValueError("A single source quad cannot retain duplicated D1 boundary vertex IDs")
            children = auto_partition_quad_mesh_into_layouts(
                layout.vertices, layout.quads, target_patch_size=max(1, len(layout.quads) // 2),
                control_point_budget=100000, fast_disk_growth=True)
            for child in children:
                _inherit_layout_correspondence(child, layout)
                if len(child.quads) >= len(layout.quads):
                    raise ValueError("D1 self-seam chart subdivision did not reduce the source quad count")
            fallback_reports.append({"method": "Non-paper disk clustering to separate both sides of a D1 CAD self-seam",
                                     "input_quad_count": len(layout.quads), "replacement_count": len(children),
                                     "self_seam_source_vertex_count_before": len(seam_ids),
                                     "native_unavailable_or_pass_limit": not native_available or passes >= maximum_native_passes})
            next_passes = maximum_native_passes
        else:
            next_passes = passes + 1
        pending.extend((child, next_passes) for child in reversed(children))
    before = Counter(face for layout in layouts for face in layout.source_face_indices)
    after = Counter(face for layout in result for face in layout.source_face_indices)
    if before != after or any(_self_seam_source_ids(layout) for layout in result):
        raise ValueError("D1 redivision changed quad coverage or retained a same-chart CAD self-seam")
    return result, {"method": "Appendix D1 repeated native GMCG followed by explicitly reported chart splitting when required",
                    "maximum_native_passes": maximum_native_passes, "native_attempts": attempts,
                    "native_execution_count": len(attempts), "fallback_reports": fallback_reports,
                    "non_paper_fallback": bool(fallback_reports), "remaining_self_seam_chart_count": 0,
                    "source_quad_coverage_exactly_once": True}


def _repair_gmcg_patch(vertices, patch_quads, patch_id, reason, target_patch_size,
                       *, gmcg_executable=None, redivide_D1=True):
    """Attempt paper D1 seams before explicitly reported non-paper partitioning."""
    pieces, fallback_methods = [], []
    try:
        parts, D1_report = repair_quad_patch_to_disks(vertices, patch_quads)
    except ValueError as exc:
        parts = []
        D1_report = {"paper_D1_executed": False, "paper_D1_validated": False,
                     "failure_reason": str(exc), "remaining_non_disk_part_count": 1}

    def append_part(part_vertices, part_quads, source_ids, face_indices, D1_topology):
        if D1_topology.get("is_disk", False):
            layout = _make_layout(part_vertices, part_quads, list(range(len(part_quads))), len(pieces))
            local_pieces = [layout]
        else:
            fallback_methods.append("Non-paper PCA bisection and topology-checked clustering")
            local_pieces = []
            for disk_quads in partition_nondisk_patch_into_disks(part_vertices, part_quads):
                try:
                    layout = _make_layout(part_vertices, disk_quads, list(range(len(disk_quads))), len(pieces))
                    # Recover the unchanged input face IDs for each PCA subset.
                    lookup = defaultdict(deque)
                    for index, face in enumerate(part_quads):
                        lookup[tuple(map(int, face))].append(index)
                    layout.source_face_indices = [lookup[tuple(map(int, face))].popleft() for face in disk_quads]
                    local_pieces.append(layout)
                except ValueError:
                    sub_pieces = auto_partition_quad_mesh_into_layouts(
                        part_vertices, disk_quads, target_patch_size=max(25, target_patch_size))
                    lookup = defaultdict(deque)
                    for index, face in enumerate(part_quads):
                        lookup[tuple(map(int, face))].append(index)
                    subset_faces = [lookup[tuple(map(int, face))].popleft() for face in disk_quads]
                    for piece in sub_pieces:
                        piece.source_face_indices = [subset_faces[index] for index in piece.source_face_indices]
                    local_pieces.extend(sub_pieces)
        for piece in local_pieces:
            piece.source_vertex_indices = _remap_source_ids(piece.source_vertex_indices, source_ids)
            piece.source_face_indices = np.asarray(face_indices)[piece.source_face_indices].tolist()
            piece.gmcg_repaired = True
            piece.gmcg_patch_id = patch_id
            piece.appendix_D1_report = {**D1_report, "part_topology": D1_topology}
            pieces.append(piece)

    for part in parts:
        append_part(part.vertices, part.quads, part.source_vertex_indices, part.source_face_indices, part.report)
    if not parts:
        append_part(vertices, patch_quads, np.arange(len(vertices)), np.arange(len(patch_quads)), {})
    redivision_report = None
    if redivide_D1 and any(_self_seam_source_ids(piece) for piece in pieces):
        pieces, redivision_report = _redivide_D1_self_seams(pieces, executable=gmcg_executable)
        if redivision_report["non_paper_fallback"]:
            fallback_methods.append("Non-paper chart division after repeated GMCG retained or failed a D1 CAD self-seam")
            # The now-unnecessary coincident cut can force an extra tiny chart.
            # Compare a partition of the unchanged original patch only after
            # retaining the actual D1/native evidence. Select it only if it
            # uses fewer validated charts and still covers every source quad.
            comparison = {"method": "Non-paper disk partition of the unchanged original patch",
                          "cut_disk_layout_count": len(pieces), "selected": False}
            try:
                original_pieces = auto_partition_quad_mesh_into_layouts(
                    vertices, patch_quads, target_patch_size=max(1, len(patch_quads) // 2),
                    control_point_budget=100000, fast_disk_growth=True)
                if len(original_pieces) < len(pieces):
                    original_pieces, original_map_repairs = _repair_parameterizable_layouts(
                        original_pieces, target_patch_size)
                    _validate_source_quad_coverage(original_pieces, vertices, patch_quads)
                    face_lookup = defaultdict(deque)
                    for index, face in enumerate(patch_quads):
                        face_lookup[tuple(sorted(map(int, face)))].append(index)
                    for piece in original_pieces:
                        piece.source_face_indices = [face_lookup[tuple(sorted(map(int, piece.source_vertex_indices[face])))].popleft()
                                                     for face in piece.quads]
                    comparison.update(original_patch_layout_count=len(original_pieces),
                                      parameterization_repairs=original_map_repairs,
                                      remaining_self_seam_chart_count=sum(bool(_self_seam_source_ids(p)) for p in original_pieces))
                    if (len(original_pieces) < len(pieces)
                            and comparison["remaining_self_seam_chart_count"] == 0):
                        pieces = original_pieces
                        comparison["selected"] = True
                else:
                    comparison["original_patch_layout_count"] = len(original_pieces)
            except (ValueError, IndexError) as error:
                comparison["failure"] = f"{type(error).__name__}: {error}"
            redivision_report["original_patch_partition_comparison"] = comparison
        for piece in pieces:
            piece.gmcg_repaired = True
            piece.gmcg_patch_id = patch_id
            piece.appendix_D1_report = {**D1_report, "chart_redivision": redivision_report}
    # An already-disk patch with invalid/missing graph sides uses geometric corner
    # reselection, not a successful paper D1 operation.
    if not D1_report.get("paper_D1_executed", False):
        fallback_methods.append("Non-paper geometric corner selection after invalid graph sides")
    covered = [index for piece in pieces for index in piece.source_face_indices]
    if sorted(covered) != list(range(len(patch_quads))):
        raise ValueError("GMCG patch repair changed its original quad coverage")
    return pieces, {"patch": patch_id, "reason": reason, "replacement_count": len(pieces),
                    "appendix_D1": D1_report, "paper_D1_executed": D1_report.get("paper_D1_executed", False),
                    "paper_D1_topology_validated": D1_report.get("paper_D1_validated", False),
                    "paper_D1_validated": D1_report.get("paper_D1_validated", False) and not fallback_methods,
                    "D1_chart_redivision": redivision_report,
                    "non_paper_fallback": bool(fallback_methods),
                    "fallback_methods": sorted(set(fallback_methods)),
                    "coverage_validation": "Every original patch quad occurs exactly once; unchanged physical geometry"}


def divide_sparse_quad_layouts(
    vertices: np.ndarray,
    quads: np.ndarray,
    layout_folder: Optional[str | Path] = None,
    target_patch_size: int = 25,
    gmcg_executable: Optional[str | Path] = None,
    force_bounded_partition: bool = False,
) -> Tuple[List[QuadLayout], dict]:
    """Step 3 entry point: Produce sparse quadrilateral layouts."""
    # If layout_folder is explicitly provided, load pre-existing layouts
    if layout_folder:
        existing = load_sparse_quad_layouts(layout_folder)
        if existing:
            for layout in existing:
                _match_layout_to_source(layout, np.asarray(vertices))
            existing, map_repairs = _repair_parameterizable_layouts(existing, target_patch_size)
            covered = _validate_source_quad_coverage(existing, vertices, quads)
            report = {
                "step": 3,
                "source": f"pre-computed layouts from {layout_folder}",
                "layout_count": len(existing),
                "covered_quad_count": covered,
                "coverage_validation": "Every source quad occurs exactly once",
                "paper_stage_3": False,
                "method_provenance": "Precomputed layout generation method has not been verified",
                "parameterization_repairs": map_repairs,
                **_appendix_report(existing),
            }
            return existing, report
        raise ValueError(f"No valid sparse quadrilateral layouts found in {layout_folder}")

    if force_bounded_partition:
        # Triangle midpoint subdivision creates a valence-three singularity
        # at every centroid. GMCG would turn that artifact into tens of
        # thousands of tiny surfaces. Cluster unchanged faces into checked
        # disks instead, explicitly reporting the alternative algorithm.
        return _bounded_subdivision_layouts(vertices, quads, target_patch_size)

    executable = Path(gmcg_executable) if gmcg_executable else get_gmcg_headless_executable()
    fallback_reason = "A GMCG headless executable is unavailable"
    # The supplied GMCG_revision executable is interactive. Its CLI cannot be
    # assumed equivalent to the explicitly supported headless wrapper.
    if executable and executable.is_file() and "headless" in executable.stem.lower():
        try:
            with tempfile.TemporaryDirectory(prefix="pylcss_gmcg_") as folder:
                root = Path(folder)
                input_obj = root / "input.obj"
                with input_obj.open("w", encoding="ascii") as stream:
                    for vertex in vertices:
                        stream.write("v " + " ".join(f"{float(x):.17g}" for x in vertex) + "\n")
                    for quad in quads:
                        stream.write("f " + " ".join(str(int(i) + 1) for i in quad) + "\n")
                subprocess.run([str(executable), str(input_obj), str(root / "layouts")],
                               check=True, capture_output=True, timeout=180,
                               **headless_subprocess_kwargs())
                output_root = root / "layouts"
                indexed_output = output_root / "quad_patch_indices.json"
                if indexed_output.is_file():
                    return _load_indexed_gmcg_layouts(indexed_output, vertices, quads, target_patch_size,
                                                     gmcg_executable=executable)
                # Appendix D permits splitting non-disk GMCG patches. Preserve
                # the valid sparse patches instead of discarding the entire graph.
                from .amrto_quadrangulation import load_quad_mesh_from_obj
                layouts = []
                repaired_patches = []
                exported_faces = []
                files = sorted((output_root / "output_quad").glob("quad_*.obj"),
                               key=lambda p: int(p.stem.split("_")[-1]))
                for file in files:
                    original_id = int(file.stem.split("_")[-1])
                    patch_vertices, patch_quads = load_quad_mesh_from_obj(file)
                    precision_tolerance = max(float(np.max(np.abs(vertices))), 1.0) * 1e-5
                    if (patch_vertices.shape != vertices.shape or
                            not np.allclose(patch_vertices, vertices, rtol=0, atol=precision_tolerance)):
                        raise ValueError("GMCG changed the global vertex correspondence")
                    exported_faces.extend(tuple(sorted(map(int, face))) for face in patch_quads)
                    try:
                        layout = load_layout_from_obj(file, layout_id=original_id,
                                                      repair_boundary_triangles=False)
                        if layout is None:
                            raise ValueError("Empty GMCG patch")
                        # Restore the original coordinates; GMCG's OBJ writer uses
                        # reduced float precision, but preserves global vertex IDs.
                        layout.vertices = vertices[layout.source_vertex_indices].copy()
                        layouts.append(layout)
                    except ValueError as exc:
                        repaired, repair_report = _repair_gmcg_patch(
                            vertices, patch_quads, original_id, str(exc), target_patch_size,
                            gmcg_executable=executable)
                        layouts.extend(repaired)
                        repaired_patches.append(repair_report)
                if not layouts:
                    raise ValueError("GMCG produced no patch files")
                boundaries = json.loads((output_root / "patch_boundaries.json").read_text())
                if Counter(exported_faces) != Counter(tuple(sorted(map(int, face))) for face in quads):
                    raise ValueError("GMCG layouts do not cover the input quad mesh")
                for layout in layouts:
                    if getattr(layout, "gmcg_repaired", False):
                        continue
                    patch = boundaries[layout.layout_id]
                    uv = np.full((len(layout.vertices), 2), np.nan)
                    local_ids = {int(global_id): local_id for local_id, global_id
                                 in enumerate(layout.source_vertex_indices)}
                    for edge in patch["edges"]:
                        indices = [local_ids[int(i)] for i in edge["vertices"]]
                        uv[indices] = np.asarray(edge["uv"], dtype=float)
                    # Texture coordinates are per halfedge in GMCG; the boundary
                    # export provides the patch-local correspondence explicitly.
                    loop = _boundary_loop(layout.triangles, len(layout.vertices))
                    if not np.isfinite(uv[loop]).all():
                        raise ValueError("GMCG boundary UV correspondence is incomplete")
                    uv[np.isnan(uv)] = 0.0
                    extract_layout_boundary_and_corners(layout, pts_vt=uv)
                layouts, map_repairs = _repair_parameterizable_layouts(layouts, target_patch_size)
                _validate_source_quad_coverage(layouts, vertices, quads)
                for new_id, layout in enumerate(layouts):
                    layout.layout_id = new_id
                return layouts, {
                    "step": 3, "source": "Generalized Motorcycle Graph (headless)",
                    "layout_count": len(layouts), "input_quad_count": len(quads),
                    "covered_quad_count": sum(len(layout.quads) for layout in layouts),
                    "paper_stage_3": not any(r["non_paper_fallback"] for r in repaired_patches)
                                     and not any(r.get("non_paper_fallback", False) for r in map_repairs),
                    "gmcg_patch_count": len(files),
                    "non_disk_patch_repairs": repaired_patches,
                    "parameterization_repairs": map_repairs,
                    "coverage_validation": "Every global input quad occurs exactly once",
                    "repair_method": "Appendix D1 coincident geodesic seams; explicit non-paper partition only when required",
                    **_appendix_report(layouts),
                }
        except Exception as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or b"").decode("utf-8", errors="replace")[-1000:]
                fallback_reason = f"{executable.name} exited {exc.returncode}: {details}".rstrip(": ")
            else:
                fallback_reason = str(exc)
            logger.warning("GMCG layout generation failed (%s); using disk partitioner", exc)

    # Otherwise auto-partition the quad mesh
    _FALLBACK_CTRL_BUDGET = 40000  # Raised from 2500; larger patches need more control budget
    layouts = auto_partition_quad_mesh_into_layouts(
        vertices,
        quads,
        target_patch_size=target_patch_size,
        control_point_budget=_FALLBACK_CTRL_BUDGET,
    )
    layouts, map_repairs = _repair_parameterizable_layouts(layouts, target_patch_size)
    _validate_source_quad_coverage(layouts, vertices, quads)
    for layout_id, layout in enumerate(layouts):
        layout.layout_id = layout_id
    report = {
        "step": 3,
        "source": "automatic quad partitioner",
        "layout_count": len(layouts),
        "input_quad_count": len(quads),
        "covered_quad_count": sum(len(layout.quads) for layout in layouts),
        "paper_stage_3": False,
        "fallback_control_point_budget": _FALLBACK_CTRL_BUDGET,
        "fallback_reason": fallback_reason,
        "parameterization_repairs": map_repairs,
        **_appendix_report(layouts),
    }
    return layouts, report


def _bounded_subdivision_layouts(vertices: np.ndarray, quads: np.ndarray,
                                 target_patch_size: int) -> Tuple[List[QuadLayout], dict]:
    target_patch_size = min(100, max(8, int(target_patch_size)))
    # Smooth shared boundary curves do not need a C0 knot at every source
    # edge. Retain a finite conservative estimate while avoiding unnecessary
    # fragmentation from the former piecewise-linear boundary budget.
    control_point_budget = 10000
    layouts = auto_partition_quad_mesh_into_layouts(vertices, quads, target_patch_size=target_patch_size,
                                                  control_point_budget=control_point_budget, fast_disk_growth=True)
    layouts, repairs = _repair_parameterizable_layouts(layouts, target_patch_size)
    for layout_id, layout in enumerate(layouts):
        layout.layout_id = layout_id
    covered_count = _validate_source_quad_coverage(layouts, vertices, quads)
    return layouts, {
        "step": 3, "source": "Bounded disk partition of triangle-subdivision quadrangulation",
        "layout_count": len(layouts), "input_quad_count": len(quads),
        "covered_quad_count": covered_count, "target_patch_faces": target_patch_size,
        "maximum_patch_faces": max((len(layout.quads) for layout in layouts), default=0),
        "fallback_control_point_budget": control_point_budget,
        "disk_growth": "Incremental quad Euler and boundary-degree checks; full disk and square-map validation after clustering",
        "patch_face_count_percentiles": np.percentile([len(layout.quads) for layout in layouts], [0, 50, 90, 100]).tolist(),
        "paper_stage_3": False,
        "fallback_reason": "Triangle midpoint subdivision has artificial valence-three centroids",
        "parameterization_repairs": repairs,
        "coverage_validation": "Every global input quad occurs exactly once; source coordinates unchanged",
        **_appendix_report(layouts),
    }


def _load_indexed_gmcg_layouts(path: Path, vertices: np.ndarray, quads: np.ndarray,
                              target_patch_size: int, *, gmcg_executable=None,
                              redivide_D1=True) -> Tuple[List[QuadLayout], dict]:
    """Consume graph side paths directly, without global texture optimization."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("vertex_count") != len(vertices) or data.get("face_count") != len(quads):
        raise ValueError("GMCG changed the input mesh size")
    layouts, repaired, exported = [], [], []
    for patch in data["patches"]:
        patch_id = int(patch["patch"])
        patch_quads = np.asarray(patch["quads"], dtype=np.int64)
        if (patch_quads.ndim != 2 or patch_quads.shape[1] != 4 or not len(patch_quads)
                or patch_quads.min() < 0 or patch_quads.max() >= len(vertices)):
            raise ValueError("GMCG exported invalid global face indices")
        exported.extend(tuple(sorted(map(int, face))) for face in patch_quads)
        try:
            used, triangles, local_quads = triangulate_quads_clean(vertices, patch_quads)
            layout = QuadLayout(patch_id, vertices[used].copy(), triangles, quads=local_quads)
            layout.source_vertex_indices = used
            loop = _boundary_loop(triangles, len(used))
            source_ids = {int(global_id): local_id for local_id, global_id in enumerate(used)}
            paths = [[source_ids[int(i)] for i in side] for side in patch["sides"]]
            if (len(paths) != 4 or any(len(side) < 2 for side in paths)
                    or any(paths[i][-1] != paths[(i + 1) % 4][0] for i in range(4))
                    or len({side[0] for side in paths}) != 4):
                raise ValueError("GMCG patch does not have four connected graph sides")
            boundary = Counter(tuple(sorted((a, b))) for a, b in zip(loop, loop[1:] + loop[:1]))
            side_edges = Counter(tuple(sorted((a, b))) for side in paths for a, b in zip(side, side[1:]))
            if side_edges != boundary:
                raise ValueError("GMCG graph sides do not cover the complete disk boundary")
            directed = set(zip(loop, loop[1:] + loop[:1]))
            if (paths[0][0], paths[0][1]) not in directed:
                paths = [list(reversed(side)) for side in reversed(paths)]
            layout.boundary_paths = paths
            layout.corner_indices = [side[0] for side in paths]
            layouts.append(layout)
        except (ValueError, KeyError) as exc:
            pieces, repair_report = _repair_gmcg_patch(
                vertices, patch_quads, patch_id, str(exc), target_patch_size,
                gmcg_executable=gmcg_executable, redivide_D1=redivide_D1)
            layouts.extend(pieces)
            repaired.append(repair_report)
    if Counter(exported) != Counter(tuple(sorted(map(int, face))) for face in quads):
        raise ValueError("GMCG layouts do not cover every input quad exactly once")
    layouts, map_repairs = _repair_parameterizable_layouts(layouts, target_patch_size)
    covered_count = _validate_source_quad_coverage(layouts, vertices, quads)
    for layout_id, layout in enumerate(layouts):
        layout.layout_id = layout_id
    return layouts, {
        "step": 3, "source": "Generalized Motorcycle Graph (headless)",
        "boundary_source": "Ordered graph halfarc sides; no global texture optimization",
        "layout_count": len(layouts), "gmcg_patch_count": len(data["patches"]),
        "input_quad_count": len(quads), "covered_quad_count": covered_count,
        "non_disk_patch_repairs": repaired, "parameterization_repairs": map_repairs,
        "paper_stage_3": not any(r["non_paper_fallback"] for r in repaired)
                         and not any(r.get("non_paper_fallback", False) for r in map_repairs),
        "coverage_validation": "Every global input quad occurs exactly once",
        "repair_method": "Appendix D1 coincident geodesic seams; explicit non-paper partition only when required",
        **_appendix_report(layouts),
    }


class PaperLayoutError(ValueError):
    """A native paper layout could not be consumed without changing method."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _paper_layout_from_graph_patch(vertices, patch):
    """Consume native GMCG sides, removing retraced interior edge excursions.

    GMCG can export a side walk containing A-B-A excursions through an interior
    edge. Cancel opposite traversals of the same edge, preserve the four native
    corners, and validate that the resulting sides cover the disk boundary.
    """
    patch_quads = np.asarray(patch["quads"], dtype=np.int64)
    if (patch_quads.ndim != 2 or patch_quads.shape[1] != 4 or not len(patch_quads)
            or patch_quads.min() < 0 or patch_quads.max() >= len(vertices)):
        raise ValueError("GMCG exported invalid global quad indices")
    used, triangles, quads = triangulate_quads_clean(vertices, patch_quads)
    layout = QuadLayout(int(patch["patch"]), vertices[used].copy(), triangles, quads=quads)
    layout.source_vertex_indices = used
    loop = _boundary_loop(triangles, len(used))
    source_ids = {int(global_id): local_id for local_id, global_id in enumerate(used)}
    paths, side_repairs = [], []
    for side_index, side in enumerate(patch["sides"]):
        cleaned = []
        for raw_source in side:
            local_vertex = source_ids[int(raw_source)]
            if len(cleaned) >= 2 and local_vertex == cleaned[-2]:
                cleaned.pop()
            else:
                cleaned.append(local_vertex)
        paths.append(cleaned)
        if len(cleaned) != len(side):
            side_repairs.append({"side": side_index,
                                 "original_source_path": list(map(int, side)),
                                 "source_path": [int(used[v]) for v in cleaned],
                                 "removed_edge_traversal_count": len(side) - len(cleaned)})
    if (len(paths) != 4 or any(len(side) < 2 for side in paths)
            or len({side[0] for side in paths}) != 4
            or any(paths[index][-1] != paths[(index + 1) % 4][0] for index in range(4))):
        raise ValueError("GMCG did not provide four connected graph sides")
    boundary = Counter(tuple(sorted((a, b))) for a, b in zip(loop, loop[1:] + loop[:1]))
    sides = Counter(tuple(sorted((a, b))) for side in paths for a, b in zip(side, side[1:]))
    if sides != boundary:
        raise ValueError("GMCG graph sides do not cover the complete layout boundary")
    directed = set(zip(loop, loop[1:] + loop[:1]))
    if (paths[0][0], paths[0][1]) not in directed:
        paths = [list(reversed(side)) for side in reversed(paths)]
    layout.boundary_paths = paths
    layout.corner_indices = [side[0] for side in paths]
    if side_repairs:
        layout.native_graph_side_repair = {
            "patch": int(patch["patch"]),
            "method": "Cancel opposite traversals of retraced native GMCG edges",
            "native_corners_preserved": True, "source_quads_unchanged": True,
            "complete_disk_boundary_validated": True, "side_repairs": side_repairs}
    return layout


def _paper_D1_annular_layouts(vertices, patch):
    """Divide a genus-0 two-loop annular chart into self-seam-free disks via D1 cuts."""
    from .amrto_disk_repairs import (
        _boundary_loops, _face_components, _shortest_boundary_cut, _split_vertex_fans, _topology,
    )

    patch_quads = np.asarray(patch["quads"], dtype=np.int64)
    used = np.unique(patch_quads)
    local_map = {int(old): index for index, old in enumerate(used)}
    local_quads = np.asarray([[local_map[int(v)] for v in quad] for quad in patch_quads], dtype=np.int64)
    local_verts = np.asarray(vertices, dtype=float)[used].copy()
    initial_topology = _topology(local_quads)
    if (initial_topology["euler_characteristic"] != 0
            or initial_topology["boundary_loop_count"] != 2
            or initial_topology["edge_connected_component_count"] != 1):
        return None
    loops = sorted(_boundary_loops(local_quads), key=len)
    inner_loop, outer_loop = loops[0], loops[1]
    quad_deg = Counter(int(v) for quad in local_quads for v in quad)
    reflex_outer = [v for v in outer_loop if quad_deg[v] >= 3]
    convex_outer = [v for v in outer_loop if quad_deg[v] == 1]
    native_corners = {int(side[0]) for side in patch.get("sides", ()) if len(side)}

    quad_edge_counts = Counter(
        tuple(sorted((int(u), int(v))))
        for quad in patch_quads for u, v in zip(quad, np.roll(quad, -1))
    )
    cleaned_sides = []
    for side in patch.get("sides", ()):
        cleaned = []
        for raw in side:
            val = int(raw)
            if len(cleaned) >= 2 and val == cleaned[-2]:
                cleaned.pop()
            else:
                cleaned.append(val)
        cleaned_sides.append(cleaned)
    side_directed = [(a, b) for side in cleaned_sides for a, b in zip(side, side[1:])]
    side_counts = Counter(tuple(sorted(edge)) for edge in side_directed)
    seam_edges = sorted(
        edge for edge, count in side_counts.items()
        if count == 2 and quad_edge_counts.get(edge) == 2
        and side_directed.count(edge) == 1
        and side_directed.count((edge[1], edge[0])) == 1
    )

    cuts, cut_edges, strip_corner_sets = [], [], []
    ring_comp = list(range(len(local_quads)))
    if reflex_outer and len(convex_outer) >= 2:
        local_ec = Counter(
            tuple(sorted((int(u), int(v))))
            for quad in local_quads for u, v in zip(quad, np.roll(quad, -1))
        )
        int_adj = defaultdict(list)
        for (u, v), count in local_ec.items():
            if count == 2:
                w = float(np.linalg.norm(local_verts[u] - local_verts[v]))
                int_adj[u].append((w, v))
                int_adj[v].append((w, u))
        outer_set, inner_set = set(outer_loop), set(inner_loop)
        best_neck = None
        for r in reflex_outer:
            dist, prev, pq = {r: 0.0}, {}, [(0.0, r)]
            c_cand, l_cand = None, None
            while pq:
                d_u, u = heapq.heappop(pq)
                if d_u > dist[u]:
                    continue
                if u != r and u in outer_set:
                    c_cand = [u]
                    while c_cand[-1] != r:
                        c_cand.append(prev[c_cand[-1]])
                    c_cand.reverse()
                    l_cand = d_u
                    break
                if u != r and u in inner_set:
                    continue
                for w, v in int_adj[u]:
                    if v not in dist or d_u + w < dist[v]:
                        dist[v] = d_u + w
                        prev[v] = u
                        heapq.heappush(pq, (d_u + w, v))
            if c_cand is None:
                continue
            cand_edges = [tuple(sorted((a, b))) for a, b in zip(c_cand, c_cand[1:])]
            _, test_faces, _, _ = _split_vertex_fans(local_verts, local_quads, used, cand_edges)
            comps = _face_components(test_faces)
            if len(comps) != 2:
                continue
            disk_comps = [comp for comp in comps if _topology(local_quads[comp])["is_disk"]]
            ring_comps = [comp for comp in comps if not _topology(local_quads[comp])["is_disk"]]
            if len(disk_comps) != 1 or len(ring_comps) != 1:
                continue
            r_comp = ring_comps[0]
            r_loops = sorted(_boundary_loops(local_quads[r_comp]), key=len)
            if len(r_loops) != 2:
                continue
            r_deg = Counter(int(v) for quad in local_quads[r_comp] for v in quad)
            rem_reflex = sum(1 for v in r_loops[1] if r_deg[v] >= 3)
            score = (rem_reflex, float(l_cand))
            if best_neck is None or score < best_neck[0]:
                best_neck = (score, c_cand, l_cand, cand_edges, disk_comps[0], r_comp, r_loops)
        if best_neck is not None:
            _, c_strip, l_strip, c_strip_edges, d_comp, ring_comp, ring_loops = best_neck
            cuts.append({
                "component": 0, "path": list(map(int, c_strip)),
                "source_vertex_path": [int(used[v]) for v in c_strip],
                "physical_length": float(l_strip),
                "metric": "Shortest physical mesh-edge geodesic across peninsula neck",
            })
            cut_edges.extend(c_strip_edges)
            neck_ends = {int(used[c_strip[0]]), int(used[c_strip[-1]])}
            sub_loop_src = {int(used[v]) for v in _boundary_loops(local_quads[d_comp])[0]}
            tip_corners = (sub_loop_src & native_corners) - neck_ends
            if len(tip_corners) != 2:
                tip_corners = {int(used[v]) for v in convex_outer if int(used[v]) in sub_loop_src} - neck_ends
            c_set = neck_ends | tip_corners
            strip_corner_sets.append((set(d_comp), c_set))
            inner_loop, outer_loop = ring_loops[0], ring_loops[1]

    cut1 = None
    if seam_edges:
        adj = defaultdict(list)
        for u, v in seam_edges:
            if u in local_map and v in local_map:
                lu, lv = local_map[u], local_map[v]
                adj[lu].append(lv)
                adj[lv].append(lu)
        starts = [v for v in adj if v in set(inner_loop)]
        ends = [v for v in adj if v in set(outer_loop)]
        if len(starts) == 1 and len(ends) == 1:
            path = [starts[0]]
            while path[-1] != ends[0]:
                nxt = [n for n in adj[path[-1]] if len(path) < 2 or n != path[-2]]
                if len(nxt) != 1:
                    break
                path.append(nxt[0])
            if path[-1] == ends[0]:
                cut1 = path
    if cut1 is None:
        cut1, _ = _shortest_boundary_cut(local_verts, local_quads[ring_comp], [inner_loop, outer_loop])
    if cut1[0] not in set(inner_loop):
        cut1 = list(reversed(cut1))
    l1 = float(sum(np.linalg.norm(local_verts[a] - local_verts[b]) for a, b in zip(cut1, cut1[1:])))
    cuts.append({
        "component": 0, "path": list(map(int, cut1)),
        "source_vertex_path": [int(used[v]) for v in cut1],
        "physical_length": l1,
        "metric": "Shortest physical mesh-edge geodesic",
    })
    cut_edges.extend(tuple(sorted((a, b))) for a, b in zip(cut1, cut1[1:]))

    u1, v1 = cut1[0], cut1[-1]
    idx_u1 = inner_loop.index(u1)
    rot_in = inner_loop[idx_u1:] + inner_loop[:idx_u1]
    u2 = rot_in[len(rot_in) // 2]
    ring_sub_quads = [quad for quad in local_quads[ring_comp] if not (set(quad) & set(cut1))]
    if not ring_sub_quads:
        return None
    cut2, l2 = _shortest_boundary_cut(
        local_verts, np.asarray(ring_sub_quads, dtype=np.int64),
        [[u2], [v for v in outer_loop if v not in set(cut1)]])
    v2 = cut2[-1]
    cuts.append({
        "component": 0, "path": list(map(int, cut2)),
        "source_vertex_path": [int(used[v]) for v in cut2],
        "physical_length": float(l2),
        "metric": "Shortest physical mesh-edge geodesic",
    })
    cut_edges.extend(tuple(sorted((a, b))) for a, b in zip(cut2, cut2[1:]))

    ring_corners = {int(used[u1]), int(used[v1]), int(used[u2]), int(used[v2])}
    pos, faces, src_ids, coincident_splits = _split_vertex_fans(local_verts, local_quads, used, cut_edges)
    layouts = []
    report = {
        "method": "Appendix D Improvement 1: shortest mesh-edge seams and coincident vertex fan splitting",
        "paper_D1_executed": True, "paper_D1_validated": True,
        "coincident_vertex_splits": coincident_splits, "geodesic_cuts": cuts,
        "new_vertex_count": int(len(pos) - len(used)),
        "input_quad_count": int(len(patch_quads)), "covered_quad_count": int(len(patch_quads)),
        "physical_geometry_unchanged": True, "source_quad_coverage_exactly_once": True,
        "remaining_non_disk_part_count": 0,
    }
    for comp in _face_components(faces):
        u2_idx, tris, c_qs = triangulate_quads_clean(pos, faces[comp])
        layout = QuadLayout(int(patch["patch"]), pos[u2_idx].copy(), tris, quads=c_qs)
        layout.source_vertex_indices = src_ids[u2_idx]
        layout.source_face_indices = list(comp)
        loop = _boundary_loop(tris, len(u2_idx))
        target_corners = next((cs for fset, cs in strip_corner_sets if fset == set(comp)), ring_corners)
        corners = [v for v in loop if int(layout.source_vertex_indices[v]) in target_corners]
        if len(corners) != 4:
            return None
        start = loop.index(corners[0])
        loop = loop[start:] + loop[:start]
        corners = [v for v in loop if v in set(corners)]
        positions = [loop.index(v) for v in corners]
        layout.boundary_paths = [
            loop[positions[i]:positions[i + 1] + 1] if i < 3 else loop[positions[3]:] + [loop[0]]
            for i in range(4)
        ]
        layout.corner_indices = corners
        if _self_seam_source_ids(layout):
            return None
        layout.appendix_D1_report = report
        layouts.append(layout)
    return layouts, report


def _paper_D1_layouts(vertices, patch):
    """Cut a cylindrical chart at geodesic seams; their ends supply four corners.

    Unlike geometric corner reselection or PCA clustering, these corners are
    supplied directly by the Appendix D1 seam endpoints (and any retained
    native corners on reflex-bounded peninsulas). Complex cuts that do not
    supply four corners remain an explicit unsupported case.
    """
    annular = _paper_D1_annular_layouts(vertices, patch)
    if annular is not None:
        return annular
    patch_quads = np.asarray(patch["quads"], dtype=np.int64)
    parts, report = repair_quad_patch_to_disks(vertices, patch_quads)
    if not report.get("paper_D1_executed") or not report.get("paper_D1_validated"):
        raise ValueError("The native chart requires further GMCG division; Appendix D1 did not produce disks")
    cut_endpoints = {int(path[index]) for cut in report["geodesic_cuts"]
                     for path in [cut["source_vertex_path"]] for index in (0, -1)}
    native_corners = {int(side[0]) for side in patch["sides"] if len(side)}
    layouts = []
    for part in parts:
        used, triangles, quads = triangulate_quads_clean(part.vertices, part.quads)
        layout = QuadLayout(int(patch["patch"]), part.vertices[used].copy(), triangles, quads=quads)
        layout.source_vertex_indices = part.source_vertex_indices[used]
        loop = _boundary_loop(triangles, len(used))
        corners = [vertex for vertex in loop if int(layout.source_vertex_indices[vertex]) in cut_endpoints]
        if len(corners) != 4:
            corners = [vertex for vertex in loop if int(layout.source_vertex_indices[vertex]) in native_corners]
        if len(corners) == 4:
            start = loop.index(corners[0])
            loop = loop[start:] + loop[:start]
            corners = [vertex for vertex in loop if vertex in set(corners)]
            positions = [loop.index(vertex) for vertex in corners]
            layout.boundary_paths = [loop[positions[index]:positions[index + 1] + 1]
                                     if index < 3 else loop[positions[index]:] + [loop[0]]
                                     for index in range(4)]
            layout.corner_indices = corners
        else:
            extract_layout_boundary_and_corners(layout)
        layout.source_face_indices = part.source_face_indices.tolist()
        layout.appendix_D1_report = report
        layouts.append(layout)
    return layouts, report


def _load_paper_indexed_gmcg_layouts(path: Path, vertices, quads):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("vertex_count") != len(vertices) or data.get("face_count") != len(quads):
        raise ValueError("GMCG changed the input mesh size")
    source_faces = defaultdict(deque)
    for face_id, quad in enumerate(quads):
        source_faces[tuple(sorted(map(int, quad)))].append(face_id)
    layouts, D1_reports = [], []
    for patch in data["patches"]:
        global_quads = np.asarray(patch["quads"], dtype=np.int64)
        patch_face_ids = []
        for quad in global_quads:
            key = tuple(sorted(map(int, quad)))
            if not source_faces[key]:
                raise ValueError("GMCG duplicated or invented a source quad")
            patch_face_ids.append(source_faces[key].popleft())
        try:
            pieces = [_paper_layout_from_graph_patch(vertices, patch)]
            pieces[0].source_face_indices = patch_face_ids
        except (ValueError, KeyError) as graph_error:
            try:
                pieces, D1_report = _paper_D1_layouts(vertices, patch)
            except ValueError as error:
                raise PaperLayoutError(
                    f"AMRTO Step 3 native patch {patch['patch']} cannot be processed by Appendix D1: {error}",
                    {"step": 3, "paper_stage_3": False, "native_patch": patch["patch"],
                     "graph_error": str(graph_error), "appendix_D1_error": str(error)}) from error
            for piece in pieces:
                piece.source_face_indices = [patch_face_ids[index] for index in piece.source_face_indices]
            D1_reports.append({"patch": patch["patch"], **D1_report})
        layouts.extend(_apply_D4(piece) for piece in pieces)
    covered = _validate_source_quad_coverage(layouts, vertices, quads)
    for layout_id, layout in enumerate(layouts):
        layout.layout_id = layout_id
    return layouts, {"step": 3, "source": "Generalized Motorcycle Graph (headless)",
                     "paper_stage_3": True, "paper_section": "2.5 and Appendix D",
                     "layout_count": len(layouts), "gmcg_patch_count": len(data["patches"]),
                     "input_quad_count": len(quads), "covered_quad_count": covered,
                     "boundary_source": "Native graph sides; Appendix D1 seam endpoints where required",
                     "appendix_D1": D1_reports, "parameterization_repairs": [],
                     "native_graph_side_repairs": [layout.native_graph_side_repair for layout in layouts
                                                   if hasattr(layout, "native_graph_side_repair")],
                     "coverage_validation": "Every source quad occurs exactly once; source coordinates unchanged",
                     **_appendix_report(layouts)}


def divide_paper_sparse_quad_layouts(vertices, quads, layout_folder=None, *, gmcg_executable=None):
    """Paper Step 3 with no distortion preflight, clustering or fit-driven cuts."""
    vertices, quads = np.asarray(vertices, dtype=float), np.asarray(quads, dtype=np.int64)
    if layout_folder:
        layouts = load_sparse_quad_layouts(layout_folder)
        if not layouts:
            raise ValueError(f"No sparse quadrilateral layouts found in {layout_folder}")
        for layout in layouts:
            _match_layout_to_source(layout, vertices)
        covered = _validate_source_quad_coverage(layouts, vertices, quads)
        return layouts, {"step": 3, "source": str(layout_folder), "layout_count": len(layouts),
                         "covered_quad_count": covered, "paper_stage_3": False,
                         "method_provenance": "User-supplied layouts; generation method not verified",
                         "parameterization_repairs": [], **_appendix_report(layouts)}
    executable = Path(gmcg_executable) if gmcg_executable else get_gmcg_headless_executable()
    if executable is None or not executable.is_file():
        raise PaperLayoutError("AMRTO Step 3 requires the native GMCG headless executable",
                               {"step": 3, "paper_stage_3": False, "native_execution_count": 0})
    with tempfile.TemporaryDirectory(prefix="pylcss_amrto_step3_") as folder:
        root = Path(folder)
        source = root / "source.obj"
        with source.open("w", encoding="ascii") as stream:
            for vertex in vertices:
                stream.write("v " + " ".join(f"{float(value):.17g}" for value in vertex) + "\n")
            for quad in quads:
                stream.write("f " + " ".join(str(int(index) + 1) for index in quad) + "\n")
        try:
            subprocess.run([str(executable), str(source), str(root / "layouts")],
                           check=True, capture_output=True, timeout=180,
                           **headless_subprocess_kwargs())
            layouts, report = _load_paper_indexed_gmcg_layouts(
                root / "layouts" / "quad_patch_indices.json", vertices, quads)
        except PaperLayoutError:
            raise
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            details = getattr(error, "stderr", b"") or b""
            if isinstance(details, bytes):
                details = details.decode("utf-8", errors="replace")
            raise PaperLayoutError("AMRTO native GMCG execution failed: " + str(error),
                                   {"step": 3, "paper_stage_3": False, "native_execution_count": 1,
                                    "failure": str(error), "native_error": details[-2000:]}) from error
    report.update(native_execution_count=1, executable=str(executable))
    return layouts, report


def subdivide_fitting_layouts(layouts: List[QuadLayout], failed_ids: set[int]) -> Tuple[List[QuadLayout], list[dict]]:
    """Refine only failed fits, preserving every original quad and vertex ID."""
    result, repairs = [], []
    for layout in layouts:
        if layout.layout_id not in failed_ids or len(layout.quads) <= 1:
            result.append(layout)
            continue
        pieces = auto_partition_quad_mesh_into_layouts(
            layout.vertices, layout.quads, target_patch_size=max(1, len(layout.quads) // 2))
        for piece in pieces:
            _inherit_layout_correspondence(piece, layout)
        pieces, map_repairs = _repair_parameterizable_layouts(pieces, max(1, len(layout.quads) // 2))
        if len(pieces) <= 1:
            raise ValueError("Fitting refinement did not subdivide the failed patch")
        result.extend(pieces)
        repairs.append({"patch": layout.layout_id, "reason": "Independent spline position, source orientation or B-rep self-intersection validation failed",
                        "position_passed": bool(getattr(layout, "fitting_report", {}).get("dense_map_residual_satisfied", True)),
                        "self_intersection_passed": not getattr(layout, "fitting_report", {}).get("brep_self_intersection", False),
                        "orientation_passed": getattr(layout, "fitting_report", {}).get("surface_orientation", {}).get("accepted"),
                        "replacement_count": len(pieces), "map_repairs": map_repairs})
    before = Counter(tuple(sorted(tuple(vertex) for vertex in layout.vertices[face]))
                     for layout in layouts for face in layout.quads)
    after = Counter(tuple(sorted(tuple(vertex) for vertex in layout.vertices[face]))
                    for layout in result for face in layout.quads)
    if before != after:
        raise ValueError("Fitting refinement changed source quad coverage")
    for new_id, layout in enumerate(result):
        layout.layout_id = new_id
    return result, repairs


def _repair_parameterizable_layouts(layouts: List[QuadLayout], target_patch_size: int):
    """Repair collapsed or compressed square maps before spline fitting."""
    from .amrto_parameterization import discrete_harmonic_mapping

    result, repairs = [], []
    for layout in layouts:
        original_layout = layout
        try:
            layout = _apply_D4(layout)
        except ValueError as exc:
            # Invalid sides must be reselected below; no D4 success is claimed.
            D4_failure = str(exc)
        else:
            D4_failure = None
        source_ids = np.asarray(getattr(layout, "source_vertex_indices", np.arange(len(layout.vertices))), dtype=np.int64)
        try:
            discrete_harmonic_mapping(layout)
            result.append(layout)
            if layout is not original_layout:
                repairs.append({"patch": layout.layout_id,
                                "reason": "Triangles had all three vertices on the same square boundary side",
                                "method": "Appendix D4 added interior vertices without changing graph sides",
                                "paper_D4_executed": True, "non_paper_fallback": False,
                                "appendix_D4": layout.appendix_D4_report})
            continue
        except ValueError as exc:
            original_error = str(exc)
        try:
            extract_layout_boundary_and_corners(layout)
            layout = _apply_D4(layout)
            discrete_harmonic_mapping(layout)
            result.append(layout)
            repairs.append({"patch": layout.layout_id, "reason": original_error,
                            "method": "Geometric square corners with Appendix D4 interior-vertex treatment",
                            "non_paper_fallback": True,
                            "appendix_D4": getattr(layout, "appendix_D4_report", {}),
                            "original_D4_failure": D4_failure})
            continue
        except ValueError:
            pass
        source_ids = np.asarray(getattr(layout, "source_vertex_indices", np.arange(len(layout.vertices))), dtype=np.int64)
        bisected = partition_nondisk_patch_into_disks(layout.vertices, layout.quads)
        if len(bisected) > 1:
            validated = []
            for b_quads in bisected:
                try:
                    u_d, tri_d, lq_d = triangulate_quads_clean(layout.vertices, b_quads)
                    lay_d = QuadLayout(len(result) + len(validated), layout.vertices[u_d].copy(), tri_d, quads=lq_d)
                    lay_d.source_vertex_indices = source_ids[u_d]
                    extract_layout_boundary_and_corners(lay_d)
                    lay_d = _apply_D4(lay_d)
                    discrete_harmonic_mapping(lay_d)
                    validated.append(lay_d)
                except Exception:
                    validated = []
                    break
            if len(validated) == len(bisected):
                result.extend(validated)
                repairs.append({"patch": layout.layout_id, "reason": original_error,
                                "method": "Local partition into parameterizable disks",
                                "non_paper_fallback": True,
                                "replacement_count": len(validated)})
                continue
        validated = []
        pending = [layout]
        layout.source_vertex_indices = source_ids
        while pending:
            parent = pending.pop()
            # Force progress even when this already-disk patch is smaller than
            # the normal clustering target. Do not immediately fragment the
            # entire failed patch into individual quads.
            split_size = max(1, min(target_patch_size, len(parent.quads) // 2))
            pieces = auto_partition_quad_mesh_into_layouts(
                parent.vertices, parent.quads, target_patch_size=split_size)
            for piece in pieces:
                _inherit_layout_correspondence(piece, parent)
                try:
                    discrete_harmonic_mapping(piece)
                    validated.append(piece)
                    continue
                except ValueError:
                    pass
                if len(piece.quads) >= len(parent.quads):
                    raise ValueError("Parameterization repair did not reduce the failed patch")
                pending.append(piece)
        result.extend(validated)
        repairs.append({"patch": layout.layout_id, "reason": original_error,
                        "method": "Local partition into parameterizable disks",
                        "non_paper_fallback": True,
                        "replacement_count": len(validated)})
    return result, repairs
