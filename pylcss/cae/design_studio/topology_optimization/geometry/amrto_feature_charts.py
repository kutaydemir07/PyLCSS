# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Preserve source creases as chart boundaries before parameterization.

This is an explicit production extension to the native GMCG partition. It
removes adjacency across actual sharp source edges. Nonseparating creases also
require separate charts: walking around a crease endpoint must not put both
sides of the crease back into the same smooth spline.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np

from .amrto_disk_repairs import repair_quad_patch_to_disks
from .amrto_layout_partition import (
    _boundary_loop, _inherit_layout_correspondence, _make_layout,
    _redivide_D1_self_seams, _remap_source_ids,
)


def _crease_components(vertices, quads, minimum_normal_dot):
    points = vertices[quads]
    normals = (np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) +
               np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 0):
        raise ValueError("Source crease charts require nondegenerate finite quad normals")
    normals /= lengths[:, None]
    owners = defaultdict(list)
    for index, face in enumerate(quads):
        for first, second in zip(face, np.roll(face, -1)):
            owners[tuple(sorted((int(first), int(second))))].append(index)
    neighbors = [set() for _ in quads]
    cuts = []
    for edge, faces in owners.items():
        if len(faces) > 2:
            raise ValueError("Source crease charts require manifold quad edges")
        if len(faces) != 2:
            continue
        first, second = faces
        dot = float(np.clip(normals[first] @ normals[second], -1., 1.))
        if dot < minimum_normal_dot:
            cuts.append({"edge": list(edge), "source_faces": [first, second], "normal_dot": dot,
                         "dihedral_degrees": float(np.rad2deg(np.arccos(dot)))})
        else:
            neighbors[first].add(second)
            neighbors[second].add(first)
    visited, components = set(), []
    for seed in range(len(quads)):
        if seed in visited:
            continue
        pending, component = [seed], []
        visited.add(seed)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    return components, cuts


def _expose_nonseparating_creases(quads, components, cuts):
    """Grow connected regions without ever including both sides of a cut.

    Removing an edge in the dual graph is insufficient when an alternate path
    connects its two incident faces. Each region therefore excludes every
    sharp neighbor of every face it contains. Additional smooth seams are
    allowed; all physical faces and their coordinates remain unchanged.
    """
    sharp_neighbors = defaultdict(set)
    for cut in cuts:
        first, second = cut["source_faces"]
        sharp_neighbors[first].add(second)
        sharp_neighbors[second].add(first)
    owners = defaultdict(list)
    for index, face in enumerate(quads):
        for first, second in zip(face, np.roll(face, -1)):
            owners[tuple(sorted((int(first), int(second))))].append(index)
    neighbors = defaultdict(set)
    for faces in owners.values():
        if len(faces) == 2:
            first, second = faces
            if second not in sharp_neighbors[first]:
                neighbors[first].add(second)
                neighbors[second].add(first)
    output, repaired_count = [], 0
    for component in components:
        remaining = set(component)
        if not any(sharp_neighbors[face].intersection(remaining) for face in component):
            output.append(component)
            continue
        repaired_count += 1
        for seed in component:
            if seed not in remaining:
                continue
            queue, queued = deque([seed]), {seed}
            region, forbidden = [], set()
            while queue:
                current = queue.popleft()
                if current not in remaining or current in forbidden:
                    continue
                region.append(current)
                remaining.remove(current)
                forbidden.update(sharp_neighbors[current])
                for neighbor in sorted(neighbors[current]):
                    if neighbor in remaining and neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)
            output.append(sorted(region))
    return output, repaired_count


def _protected_source_crease_edges(layouts, minimum_normal_dot):
    """Find sharp physical edges, including edges already between charts."""
    owners = defaultdict(list)
    for layout in layouts:
        points = np.asarray(layout.vertices)[np.asarray(layout.quads)]
        normals = (np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) +
                   np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for face, normal in zip(points, normals):
            for first, second in zip(face, np.roll(face, -1, axis=0)):
                edge = tuple(sorted((tuple(map(float, first)), tuple(map(float, second)))))
                owners[edge].append(normal)
    return {edge for edge, normals in owners.items() if len(normals) > 1
            and np.min(np.asarray(normals) @ np.asarray(normals).T) < minimum_normal_dot}


def split_layouts_at_source_creases(layouts, *, minimum_normal_dot: float = float(np.cos(np.deg2rad(30.0)))):
    """Return source-preserving charts and reports without moving source geometry.

    New charts require fresh ScaffoldMap/harmonic parameterization and fitting.
    Smooth unchanged charts retain their existing objects and coordinates. D1
    topology repair is attempted only if a connected feature component is not
    already a disk. Unsupported topology fails rather than deleting geometry.
    """
    if not np.isfinite(minimum_normal_dot) or not -1 < minimum_normal_dot < 1:
        raise ValueError("Require a finite source-normal threshold strictly between -1 and 1")
    output, repairs = [], []
    for parent in layouts:
        vertices, quads = np.asarray(parent.vertices, dtype=float), np.asarray(parent.quads)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or quads.ndim != 2 or quads.shape[1] != 4 or not len(quads)
                or not np.issubdtype(quads.dtype, np.integer)
                or quads.min() < 0 or quads.max() >= len(vertices)
                or any(len(set(face)) != 4 for face in quads)):
            raise ValueError("Source crease charts require finite indexed pure quadrilateral geometry")
        components, cuts = _crease_components(vertices, quads, minimum_normal_dot)
        components, nonseparating_count = _expose_nonseparating_creases(quads, components, cuts)
        if len(components) == 1:
            output.append(parent)
            continue
        pieces, D1_reports = [], []
        for face_ids in components:
            try:
                component = [_make_layout(vertices, quads, face_ids, parent.layout_id)]
            except ValueError as original_error:
                parts, report = repair_quad_patch_to_disks(vertices, quads[face_ids])
                if not report["paper_D1_validated"]:
                    raise ValueError(f"Source crease component cannot be repaired to disks: {original_error}")
                component = []
                for part in parts:
                    child = _make_layout(part.vertices, part.quads, list(range(len(part.quads))), parent.layout_id)
                    child.source_vertex_indices = _remap_source_ids(child.source_vertex_indices, part.source_vertex_indices)
                    child.source_face_indices = np.asarray(face_ids)[part.source_face_indices[child.source_face_indices]].tolist()
                    component.append(child)
                component, redivision = _redivide_D1_self_seams(component)
                D1_reports.append({"appendix_D1": report, "self_seam_redivision": redivision,
                                   "original_component_error": str(original_error)})
            for child in component:
                _inherit_layout_correspondence(child, parent)
                _boundary_loop(child.triangles, len(child.vertices))
                if (child.parameter_coords is not None or child.sampled_grid_3d is not None
                        or child.nurbs_surface is not None):
                    raise ValueError("A changed feature chart retained stale parameterization or fitting")
                child.source_crease_parent_layout_id = int(parent.layout_id)
                pieces.append(child)
        before = Counter(tuple(sorted(tuple(point) for point in vertices[quad])) for quad in quads)
        after = Counter(tuple(sorted(tuple(point) for point in child.vertices[quad]))
                        for child in pieces for quad in child.quads)
        if before != after or sum(len(child.quads) for child in pieces) != len(quads):
            raise ValueError("Source crease chart division changed physical quad coverage")
        if hasattr(parent, "source_face_indices") and Counter(parent.source_face_indices) != Counter(
                face for child in pieces for face in child.source_face_indices):
            raise ValueError("Source crease chart division changed source face correspondence")
        # This is the actual feature invariant, including open crease networks.
        # A disk/topology check alone cannot establish it.
        if any(_crease_components(child.vertices, child.quads, minimum_normal_dot)[1]
               for child in pieces):
            raise ValueError("Source crease chart division left a sharp edge inside a smooth chart")
        repairs.append({"patch": int(parent.layout_id), "changed": True,
                        "method": "Connected source-face regions with both sides of every sharp edge separated",
                        "non_paper_extension": True, "minimum_normal_dot": float(minimum_normal_dot),
                        "minimum_dihedral_degrees": float(np.rad2deg(np.arccos(minimum_normal_dot))),
                        "source_crease_edge_count": len(cuts), "source_crease_edges": cuts,
                        "input_quad_count": len(quads), "replacement_count": len(pieces),
                        "component_quad_counts": [len(child.quads) for child in pieces],
                        "nonseparating_crease_component_count": nonseparating_count,
                        "remaining_interior_crease_edge_count": 0,
                        "greedy_control_budget_applied": False, "physical_geometry_unchanged": True,
                        "source_quad_coverage_exactly_once": True, "parameterization_rebuilt_required": True,
                        "appendix_D1_component_repairs": D1_reports})
        output.extend(pieces)
    # Detect creases across original chart boundaries as well as the edges just
    # exposed above. Physical coordinate keys match both sides and T junctions;
    # attaching only a child's boundary edges avoids protecting added smooth seams.
    protected_edges = _protected_source_crease_edges(output, minimum_normal_dot)
    for layout in output:
        boundary_edges = {tuple(sorted((tuple(map(float, layout.vertices[first])),
                                        tuple(map(float, layout.vertices[second])))))
                          for path in layout.boundary_paths for first, second in zip(path, path[1:])}
        layout.protected_source_crease_edges = sorted(protected_edges.intersection(boundary_edges))
        layout.source_crease_minimum_normal_dot = float(minimum_normal_dot)
    # Unchanged charts retain their cached fit/UV. Callers assign consecutive IDs
    # when they actually integrate a changed set.
    return output, repairs
