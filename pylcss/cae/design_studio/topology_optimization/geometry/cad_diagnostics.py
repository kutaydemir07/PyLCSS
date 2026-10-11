# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Geometry checkpoints and independent sampled error measurements."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from .cad_tessellation import tessellate_cad_absolute

def _export_stage_mesh(
    folder: Path,
    name: str,
    patches: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Save polygon OBJ groups and their triangulated animation surface."""
    import trimesh

    folder.mkdir(parents=True, exist_ok=True)
    all_vertices, all_triangles = [], []
    offset = 0
    with (folder / f"{name}.obj").open("w", encoding="ascii") as stream:
        stream.write(f"# CAD {name}\n")
        for label, patch_vertices, polygons in patches:
            patch_vertices = np.asarray(patch_vertices, dtype=float).reshape(-1, 3)
            polygons = np.asarray(polygons, dtype=np.int64)
            if polygons.ndim != 2 or polygons.shape[1] < 3 or len(polygons) == 0:
                raise RuntimeError(f"Animation stage {name}, patch {label}, has no faces.")
            if not np.isfinite(patch_vertices).all() or not (
                (polygons >= 0).all() and (polygons < len(patch_vertices)).all()
            ):
                raise RuntimeError(f"Animation stage {name}, patch {label}, is invalid.")
            stream.write(f"g {label}\n")
            for vertex in patch_vertices:
                stream.write("v {:.12g} {:.12g} {:.12g}\n".format(*vertex))
            for polygon in polygons:
                stream.write("f " + " ".join(str(int(i) + offset + 1) for i in polygon) + "\n")
            triangles = np.concatenate(
                [polygons[:, [0, i, i + 1]] for i in range(1, polygons.shape[1] - 1)],
                axis=0,
            )
            all_vertices.append(patch_vertices)
            all_triangles.append(triangles + offset)
            offset += len(patch_vertices)
    vertices, triangles = np.concatenate(all_vertices), np.concatenate(all_triangles)
    trimesh.Trimesh(vertices=vertices, faces=triangles, process=False).export(
        folder / f"{name}.stl"
    )
    obj = folder / f"{name}.obj"
    stat = obj.stat()
    # The exporter already checked every index and coordinate. Preserve that
    # inventory so batch manifests do not reparse million-face OBJ files.
    (folder / f"{name}.mesh.json").write_text(json.dumps({
        "export_validated": True, "obj_size": stat.st_size, "obj_mtime_ns": stat.st_mtime_ns,
        "vertices": len(vertices), "triangles": len(triangles),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "watertight": None, "volume": None}, indent=2), encoding="utf-8")
    return vertices, triangles


def _discard_isolated_opposite_fins(mesh):
    """Remove only a zero-volume doubled fin from an otherwise closed mesh."""
    import trimesh
    report = {"isolated_opposite_triangle_pairs_removed": 0, "opposite_fin_area_removed": 0.0}
    if mesh.is_watertight:
        return mesh, report
    _, face_inverse, face_counts = np.unique(np.sort(mesh.faces, axis=1), axis=0,
                                             return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(face_counts == 2)
    if not len(duplicate_groups):
        return mesh, report
    vertex_incidence = np.bincount(mesh.faces.ravel(), minlength=len(mesh.vertices))
    edges, edge_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    incidence = {tuple(edge): int(count) for edge, count in zip(edges, edge_counts)}
    removed = []
    for group in duplicate_groups:
        pair = np.flatnonzero(face_inverse == group)
        first, second = mesh.faces[pair]
        if not any(np.array_equal(first, np.roll(second[::-1], shift)) for shift in range(3)):
            continue
        apex = first[vertex_incidence[first] == 2]
        if len(apex) != 1:
            continue
        base = first[first != apex[0]]
        if (incidence[tuple(sorted(map(int, base)))] != 4
                or any(incidence[tuple(sorted((int(apex[0]), int(vertex))))] != 2 for vertex in base)):
            continue
        removed.extend(map(int, pair))
    if not removed:
        return mesh, report
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[removed] = False
    candidate = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces[keep].copy(), process=False)
    candidate.remove_unreferenced_vertices()
    if (not candidate.is_watertight or not candidate.is_winding_consistent
            or abs(candidate.volume - mesh.volume) > max(1e-12, abs(mesh.volume) * 1e-10)):
        return mesh, report
    report.update(isolated_opposite_triangle_pairs_removed=len(removed) // 2,
                  opposite_fin_area_removed=float(mesh.area_faces[removed].sum()))
    return candidate, report


def _export_cad_animation_mesh(folder: Path, name: str, shape: Any, tolerance: float) -> dict[str, Any]:
    import trimesh

    angular_tolerance = 0.5
    vertices, triangles = tessellate_cad_absolute(shape, tolerance, angular_tolerance)
    if not len(vertices) or not len(triangles):
        raise RuntimeError("CAD tessellation produced no animation surface.")
    # OCC tessellates each face separately. Weld coincident boundary vertices
    # so a closed CAD body's animation mesh shares its actual seam topology.
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles, dtype=np.int64), process=True)
    solids = shape.Solids() if hasattr(shape, "Solids") else []
    closed_cad = bool(solids) and bool(shape.isValid()) and (
        len(shape.Faces()) == sum(len(solid.Faces()) for solid in solids)
    )
    repair = {"closed_cad_guard": closed_cad, "collapsed_index_triangles_removed": 0,
              "tiny_triangular_holes_capped": 0, "maximum_capped_hole_diameter": 0.0,
              "float32_vertex_groups_collapsed": 0,
              "maximum_float32_collapse_diameter": 0.0,
              "isolated_opposite_triangle_pairs_removed": 0, "opposite_fin_area_removed": 0.0,
              "repair_limit": float(tolerance) * 1e-3}
    if closed_cad:
        repeated = ((mesh.faces[:, 0] == mesh.faces[:, 1])
                    | (mesh.faces[:, 1] == mesh.faces[:, 2])
                    | (mesh.faces[:, 2] == mesh.faces[:, 0]))
        repair["collapsed_index_triangles_removed"] = int(repeated.sum())
        if repeated.any():
            mesh.update_faces(~repeated)
            mesh.remove_unreferenced_vertices()
        mesh, fin_report = _discard_isolated_opposite_fins(mesh)
        repair.update(fin_report)
        # OCC occasionally omits a nearly collinear triangle at a sewn seam.
        # Restore only isolated triangular loops far below tessellation error;
        # source surfaces and non-solid checkpoints never enter this repair.
        unique, inverse, counts = np.unique(mesh.edges_sorted, axis=0,
                                           return_inverse=True, return_counts=True)
        boundary = mesh.edges[counts[inverse] == 1]
        adjacency: dict[int, set[int]] = {}
        for start, end in boundary:
            adjacency.setdefault(int(start), set()).add(int(end))
            adjacency.setdefault(int(end), set()).add(int(start))
        visited: set[int] = set()
        caps = []
        for start in adjacency:
            if start in visited:
                continue
            component, pending = set(), [start]
            while pending:
                vertex = pending.pop()
                if vertex in component:
                    continue
                component.add(vertex)
                pending.extend(adjacency[vertex] - component)
            visited.update(component)
            if len(component) != 3 or any(len(adjacency[vertex]) != 2 for vertex in component):
                continue
            edges = [tuple(int(value) for value in edge) for edge in boundary
                     if int(edge[0]) in component]
            if len(edges) != 3:
                continue
            positions = mesh.vertices[sorted(component)]
            diameter = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max())
            if diameter > repair["repair_limit"]:
                continue
            first, second = edges[0]
            third = next(iter(component - {first, second}))
            face = (second, first, third)
            if {(face[i], face[(i + 1) % 3]) for i in range(3)} != {(b, a) for a, b in edges}:
                continue
            caps.append(face)
            repair["maximum_capped_hole_diameter"] = max(repair["maximum_capped_hole_diameter"], diameter)
        if caps:
            mesh.faces = np.vstack([mesh.faces, np.asarray(caps, dtype=np.int64)])
            repair["tiny_triangular_holes_capped"] = len(caps)
        # Binary STL uses float32 coordinates. If that precision merges a tiny
        # seam, collapse just those vertex groups and accept only a closed,
        # consistently wound result. This does not quantize other vertices.
        if mesh.is_watertight:
            rounded = np.asarray(mesh.vertices, dtype=np.float32)
            _, float_inverse, float_counts = np.unique(rounded, axis=0,
                                                       return_inverse=True, return_counts=True)
            mapping = np.arange(len(mesh.vertices))
            collapsed_groups, maximum_diameter = 0, 0.0
            for group in np.flatnonzero(float_counts > 1):
                indices = np.flatnonzero(float_inverse == group)
                positions = mesh.vertices[indices]
                diameter = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max())
                if diameter <= repair["repair_limit"]:
                    mapping[indices] = indices[0]
                    collapsed_groups += 1
                    maximum_diameter = max(maximum_diameter, diameter)
            if collapsed_groups:
                faces = mapping[mesh.faces]
                repeated = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                            | (faces[:, 2] == faces[:, 0]))
                candidate = trimesh.Trimesh(mesh.vertices.copy(), faces[~repeated], process=True)
                if candidate.is_watertight and candidate.is_winding_consistent:
                    mesh = candidate
                    repair["float32_vertex_groups_collapsed"] = collapsed_groups
                    repair["maximum_float32_collapse_diameter"] = maximum_diameter
                    repair["collapsed_index_triangles_removed"] += int(repeated.sum())
    _export_stage_mesh(folder, name, [("cad", np.asarray(mesh.vertices), np.asarray(mesh.faces))])
    metadata_path = folder / f"{name}.mesh.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(watertight=bool(mesh.is_watertight), volume=float(abs(mesh.volume)))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"vertex_count": len(mesh.vertices), "triangle_count": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight), "tessellation_tolerance": float(tolerance),
            "angular_tolerance": angular_tolerance, "deflection_is_relative": False,
            "animation_seam_repair": repair}


def assess_sampled_surface_deviation(vertices: np.ndarray, faces: np.ndarray,
                                     shape: Any, tolerance: float, samples: int = 2000) -> dict:
    """Deterministic two-way surface sampling; this is not a Hausdorff proof."""
    target_vertices, triangles = tessellate_cad_absolute(shape, tolerance, 0.5)
    if len(triangles) == 0 or len(target_vertices) == 0:
        raise RuntimeError("Cannot assess source deviation against an empty CAD tessellation")
    result = assess_sampled_mesh_deviation(vertices, faces, target_vertices, triangles, samples=samples)
    result.update(cad_tessellation_tolerance=float(tolerance), cad_angular_tolerance=0.5,
                  deflection_is_relative=False)
    return result


def assess_sampled_mesh_deviation(vertices: np.ndarray, faces: np.ndarray,
                                  target_vertices: np.ndarray, triangles: np.ndarray,
                                  samples: int = 2000) -> dict:
    """Compare mesh geometry independently of closure, genus and total volume."""
    import trimesh

    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if len(triangles) == 0 or len(target_vertices) == 0:
        raise RuntimeError("Cannot assess source deviation against an empty target mesh")
    if not np.isfinite(diagonal) or diagonal <= 0 or samples < 1:
        raise ValueError("Source deviation requires a finite nonzero source extent and positive samples")
    target = trimesh.Trimesh(target_vertices, np.asarray(triangles, dtype=np.int64).reshape(-1, 3), process=False)
    source = trimesh.Trimesh(vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3), process=False)
    distances, feature_distances, sample_counts = [], [], []
    for sampled, reference in ((source, target), (target, source)):
        sampled_points, _ = trimesh.sample.sample_surface(sampled, samples, seed=2026)
        # Area sampling rarely visits narrow ridges, small members, or corners.
        # Include deterministic vertices, all extrema, and sharp-edge sites;
        # these are still finite samples, not a continuous Hausdorff bound.
        vertex_indices = np.unique(np.r_[np.linspace(0, len(sampled.vertices) - 1,
                                                    min(samples, len(sampled.vertices)), dtype=int),
                                         np.argmin(sampled.vertices, axis=0),
                                         np.argmax(sampled.vertices, axis=0)])
        sharp_edges = sampled.face_adjacency_edges[
            sampled.face_adjacency_angles >= np.deg2rad(30.0)]
        feature_points = np.empty((0, 3), dtype=float)
        if len(sharp_edges):
            edge_indices = np.linspace(0, len(sharp_edges) - 1, min(samples, len(sharp_edges)), dtype=int)
            edges = sharp_edges[edge_indices]
            feature_points = np.unique(np.vstack((sampled.vertices[np.unique(edges)],
                                                  sampled.vertices[edges].mean(axis=1))), axis=0)
        extra_points = np.unique(np.vstack((sampled.vertices[vertex_indices], feature_points)), axis=0)
        points = np.vstack((sampled_points, extra_points))
        chunks = []
        for start in range(0, len(points), 128):
            _, values, _ = trimesh.proximity.closest_point(reference, points[start:start + 128])
            chunks.append(values)
        direction_distances = np.concatenate(chunks)
        distances.append(direction_distances)
        feature_distances.append(direction_distances[samples:])
        sample_counts.append({"surface": samples, "vertex_and_feature": len(extra_points),
                              "sharp_edges_in_mesh": len(sharp_edges)})
    values = np.concatenate(distances)
    edge_values = np.concatenate(feature_distances)
    return {"method": "Two-way deterministic surface, vertex, extrema and sharp-edge samples",
            "samples_per_direction": samples, "maximum_distance": float(values.max()),
            "sample_counts_per_direction": sample_counts,
            "maximum_vertex_and_feature_distance": float(edge_values.max()),
            "percentile_95_distance": float(np.percentile(values, 95)),
            "rms_distance": float(np.sqrt(np.mean(values ** 2))),
            "source_bounding_box_diagonal": diagonal,
            "maximum_relative_distance": float(values.max()) / diagonal,
            "certified_hausdorff_bound": False}

