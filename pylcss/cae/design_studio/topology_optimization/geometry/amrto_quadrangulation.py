# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Step 2: Quadrangulation using Instant-Meshes and 4-RoSy guidance.

Reference:
Section 2.4: Quadrangulation (step 2)
In AMRTO (Ren et al., 2025):
"To generate a larger and less spline surface in the reconstructed CAD model,
the triangular mesh of the topology optimization result is initially transformed
into a pure quad-mesh model comprising a substantial number of patches, which is
referred to as quadrangulation... Instant-Meshes has been used in this study
due to its capacity to perform efficient quadrangulation."
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from pylcss.process_utils import headless_subprocess_kwargs

from .amrto_common import get_instant_meshes_executable

logger = logging.getLogger(__name__)


class QuadrangulationValidationError(ValueError):
    """Rejected candidates with machine-readable diagnostic evidence."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _validate_triangle_surface(
    validator: Optional[Callable[[np.ndarray, np.ndarray], dict]],
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[Optional[dict], Optional[str]]:
    """Run an optional independent geometric guard without losing rejection evidence.

    A callback can raise, return ``passed=False``, or provide
    ``maximum_relative_distance`` and ``requested_maximum_relative_deviation``.
    ``requested_limit`` / ``maximum_relative_deviation`` are accepted aliases.
    A successful callback without an explicit limit owns its acceptance rule.
    """
    if validator is None:
        return None, None
    report = {}
    try:
        result = validator(np.asarray(vertices), np.asarray(triangles, dtype=np.int64))
        if not isinstance(result, dict):
            raise TypeError("Triangle surface validator must return a diagnostic dict")
        report = dict(result)
        limit = next((report[key] for key in (
            "requested_maximum_relative_deviation", "requested_limit",
            "maximum_relative_deviation", "requestedlimit",
        ) if key in report), None)
        maximum = report.get("maximum_relative_distance")
        error = None
        if maximum is not None and (not np.isfinite(float(maximum)) or float(maximum) < 0):
            error = "Triangle surface validator returned an invalid maximum relative distance"
        elif limit is not None:
            limit = float(limit)
            if not np.isfinite(limit) or limit < 0 or maximum is None:
                error = "Triangle surface validator returned an invalid or incomplete distance limit"
            elif float(maximum) > limit:
                error = f"Sampled source deviation {float(maximum):.6g} exceeds requested limit {limit:.6g}"
        if error is None and "maximum_allowed_vertex_and_feature_distance" in report:
            feature_limit = float(report["maximum_allowed_vertex_and_feature_distance"])
            feature_maximum = report.get("maximum_vertex_and_feature_distance")
            if (not np.isfinite(feature_limit) or feature_limit < 0 or feature_maximum is None
                    or not np.isfinite(float(feature_maximum)) or float(feature_maximum) < 0):
                error = "Triangle surface validator returned an invalid or incomplete feature-distance limit"
            elif float(feature_maximum) > feature_limit:
                error = f"Source vertex/crease deviation {float(feature_maximum):.6g} exceeds edge tolerance {feature_limit:.6g}"
        if error is None and "passed" in report:
            passed = report["passed"]
            if not isinstance(passed, (bool, np.bool_)):
                error = "Triangle surface validator returned a non-boolean acceptance flag"
            elif not bool(passed):
                error = str(report.get("error", "Triangle surface validator rejected candidate"))
        report["passed"] = error is None
        return report, error
    except Exception as exc:
        diagnostic = getattr(exc, "report", None)
        if isinstance(diagnostic, dict):
            report.update(diagnostic)
        error = f"{type(exc).__name__}: {exc}"
        report.update(passed=False, validator_exception=error)
        return report, error


def export_tri_mesh_to_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str | Path) -> Path:
    """Save triangular mesh to OBJ file for Instant-Meshes."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="ascii") as f:
        f.write("# AMRTO Triangular Mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in faces:
            # 1-indexed OBJ
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return p


def load_quad_mesh_from_obj(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse pure quad OBJ file generated by Instant-Meshes."""
    p = Path(filepath)
    vertices = []
    quads = []

    with open(p, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Quad face indices: f v1/vt1 v2/vt2 v3/vt3 v4/vt4
                face_idx = []
                for pt in parts[1:]:
                    idx_str = pt.split("/")[0]
                    if idx_str:
                        index = int(idx_str)
                        face_idx.append(index - 1 if index > 0 else len(vertices) + index)
                if len(face_idx) == 4:
                    if len(set(face_idx)) != 4:
                        raise ValueError("Instant-Meshes emitted a degenerate quad")
                    quads.append(face_idx)
                else:
                    raise ValueError("Instant-Meshes output must contain pure quadrilaterals")

    if not vertices or not quads:
        raise ValueError("Instant-Meshes produced an empty mesh")
    vertices_arr = np.asarray(vertices, dtype=float)
    quads_arr = np.asarray(quads, dtype=np.int64)
    if not np.isfinite(vertices_arr).all() or quads_arr.min() < 0 or quads_arr.max() >= len(vertices_arr):
        raise ValueError("Invalid vertex coordinates or face indices in quad mesh")
    return vertices_arr, quads_arr


def fallback_quadrangulate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python quadrangulation fallback via 1-to-3 quad midpoint division."""
    # Catmull-Clark style 1-to-3 triangle subdivision into 3 quads:
    # Each triangle (v0, v1, v2) has face centroid c, and edge midpoints m0, m1, m2.
    # Yields 3 quads: (v0, m0, c, m2), (v1, m1, c, m0), (v2, m2, c, m1).
    v_list = list(vertices)
    quads = []

    # Map undirected edges to midpoint index
    edge_to_mid = {}
    for tri in faces:
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0])),
        ]
        m_indices = []
        for e in edges:
            if e not in edge_to_mid:
                mid_pt = 0.5 * (vertices[e[0]] + vertices[e[1]])
                edge_to_mid[e] = len(v_list)
                v_list.append(mid_pt)
            m_indices.append(edge_to_mid[e])

        c_pt = (vertices[tri[0]] + vertices[tri[1]] + vertices[tri[2]]) / 3.0
        c_idx = len(v_list)
        v_list.append(c_pt)

        m0, m1, m2 = m_indices[0], m_indices[1], m_indices[2]
        quads.append([tri[0], m0, c_idx, m2])
        quads.append([tri[1], m1, c_idx, m0])
        quads.append([tri[2], m2, c_idx, m1])

    return np.asarray(v_list, dtype=float), np.asarray(quads, dtype=np.int64)



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


def _quad_topology_error(source: dict, output: dict) -> Optional[str]:
    if output["nonmanifold_edge_count"] or output["winding_conflict_count"] or output["degenerate_face_count"]:
        return "Quadrangulation produced nonmanifold, inconsistently oriented or degenerate faces"
    if source["is_closed"]:
        if not output["is_closed"]:
            return f"Quadrangulation opened {output['boundary_edge_count']} boundary edges in a closed input"
        if output["connected_components"] != source["connected_components"]:
            return "Quadrangulation changed the number of connected components"
        if output["euler_characteristic"] != source["euler_characteristic"]:
            return "Quadrangulation changed the Euler characteristic (surface genus)"
        if (source.get("component_euler_characteristics") is not None and
                output.get("component_euler_characteristics") != source["component_euler_characteristics"]):
            return "Quadrangulation changed the genus of individual closed components"
    return None


def _topology_preserving_decimation(vertices: np.ndarray, faces: np.ndarray,
                                    target_triangles: int, quality_threshold: Optional[float] = None):
    """Use MeshLab's link-condition checks, then independently validate output."""
    import pymeshlab
    import trimesh

    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(vertices, dtype=np.float64),
                                  face_matrix=np.asarray(faces, dtype=np.int32)))
    options = {} if quality_threshold is None else {"qualitythr": float(quality_threshold)}
    meshes.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_triangles), preservetopology=True,
        preservenormal=True, preserveboundary=True, optimalplacement=True,
        # Cleanup can remove an entire tiny closed component; independently
        # checked topology must refer to every component of the original mesh.
        autoclean=False, **options,
    )
    mesh = meshes.current_mesh()
    return trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix(), process=False)


def _bounded_fallback_triangles(vertices: np.ndarray, faces: np.ndarray, source_topology: dict,
                               quad_budget: int, maximum_quad_budget: int = 60000,
                               triangle_surface_validator: Optional[Callable[[np.ndarray, np.ndarray], dict]] = None,
                               ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Bound the CAD workload without accepting a changed closed-surface topology."""
    initial_budget = min(360000, max(3, int(quad_budget)))
    maximum_budget = min(360000, max(initial_budget, int(maximum_quad_budget)))
    budgets = [initial_budget]
    while budgets[-1] < maximum_budget:
        next_budget = 2 * budgets[-1] if budgets[-1] < 60000 or budgets[-1] >= 120000 else int(1.5 * budgets[-1])
        if budgets[-1] >= 240000:
            next_budget = 360000
        if budgets[-1] < 60000 < next_budget:
            next_budget = 60000
        budgets.append(min(maximum_budget, next_budget))
    import trimesh

    source = trimesh.Trimesh(vertices, faces, process=False)
    source_volume = abs(float(source.volume))
    failures = []
    for budget in budgets:
        # Complex lattices need enough faces to retain every cell handle. Raise
        # the workload limit before accepting any topology or volume change.
        if 3 * len(faces) <= budget:
            deviation, validation_error = _validate_triangle_surface(triangle_surface_validator, vertices, faces)
            if validation_error:
                failures.append({"quad_budget": budget, "simplifier": "original triangles",
                                 "triangle_count": len(faces), "error": validation_error,
                                 "sampled_source_deviation": deviation})
                continue
            return vertices, faces, {
                "method": "original triangles", "quad_budget": budget,
                "initial_quad_budget": initial_budget, "maximum_quad_budget": maximum_budget,
                "source_triangle_count": len(faces), "triangle_count": len(faces),
                "quad_face_count": 3 * len(faces), "source_volume": source_volume,
                "volume": source_volume, "relative_volume_delta": 0.0,
                "topology": source_topology, "rejected_simplifications": failures,
                **({"sampled_source_deviation": deviation} if deviation is not None else {}),
            }
        candidates = (("fast_simplification", 4, None), ("fast_simplification", 7, None),
                      ("PyMeshLab topology-preserving edge collapse", None, None),
                      ("PyMeshLab topology-preserving edge collapse", None, 0.1))
        for simplifier, aggression, quality_threshold in candidates:
            try:
                reduced = (
                    source.simplify_quadric_decimation(face_count=budget // 3, aggression=aggression)
                    if aggression is not None else
                    _topology_preserving_decimation(vertices, faces, budget // 3, quality_threshold)
                )
                topology = mesh_topology_report(reduced.vertices, reduced.faces)
                error = _quad_topology_error(source_topology, topology)
                errors = [error] if error is not None else []
                volume = abs(float(reduced.volume))
                delta = abs(volume - source_volume) / source_volume if source_volume > 0 else None
                if len(reduced.faces) * 3 > budget:
                    errors.append("Simplification did not reach the bounded quad budget")
                if source_topology["is_closed"] and (delta is None or delta > 0.03):
                    errors.append("Simplification changed closed-mesh volume by more than 3%")
                deviation, validation_error = _validate_triangle_surface(
                    triangle_surface_validator, reduced.vertices, reduced.faces)
                if validation_error:
                    errors.append(validation_error)
                if errors:
                    failures.append({"quad_budget": budget, "simplifier": simplifier, "aggression": aggression,
                                     "quality_threshold": quality_threshold,
                                     "triangle_count": len(reduced.faces), "error": "; ".join(errors),
                                     "topology": topology, "relative_volume_delta": delta,
                                     **({"sampled_source_deviation": deviation} if deviation is not None else {})})
                    continue
                return np.asarray(reduced.vertices), np.asarray(reduced.faces), {
                    "method": "validated quadric edge-collapse simplification",
                    "quad_budget": budget, "initial_quad_budget": initial_budget,
                    "maximum_quad_budget": maximum_budget, "source_triangle_count": len(faces),
                    "triangle_count": len(reduced.faces), "quad_face_count": 3 * len(reduced.faces),
                    "aggression": aggression, "source_volume": source_volume, "volume": volume,
                    "simplifier": simplifier,
                    "quality_threshold": quality_threshold,
                    "relative_volume_delta": delta, "topology": topology,
                    "rejected_simplifications": failures,
                    **({"sampled_source_deviation": deviation} if deviation is not None else {}),
                }
            except Exception as exc:
                failures.append({"quad_budget": budget, "simplifier": simplifier,
                                 "aggression": aggression, "quality_threshold": quality_threshold, "error": str(exc)})
        if budget < maximum_budget:
            logger.warning("Fallback could not preserve geometry within %d quads; increasing the checked budget", budget)
    raise QuadrangulationValidationError(
        f"No topology-preserving fallback within {maximum_budget} quads: {failures}",
        {"initial_quad_budget": initial_budget, "maximum_quad_budget": maximum_budget,
         "rejected_simplifications": failures})

def quadrangulate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 5000,
    crease_deg: float = 30.0,
    im_executable: Optional[Path] = None,
    triangle_surface_validator: Optional[Callable[[np.ndarray, np.ndarray], dict]] = None,
    require_native: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Step 2 entry point: Execute quadrangulation via Instant-Meshes."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    source_topology = mesh_topology_report(vertices, faces)
    im_exe = im_executable or get_instant_meshes_executable()
    fallback_reason = "Instant-Meshes executable is unavailable"
    attempts = []
    # Instant-Meshes interprets its requested count as a density target and
    # can emit several times that many quads. Keep the CAD patch workload
    # bounded even when later accuracy retries ask for a finer target.
    maximum_quad_faces = max(60_000, 8 * int(target_faces))
    if im_exe and im_exe.is_file():
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_obj = Path(tmp_dir) / "input_tri.obj"
            export_tri_mesh_to_obj(vertices, faces, in_obj)
            # Thin high-genus lattices may be opened by coarse quadrangulation.
            # Retry at a finer density with the requested creases retained.
            # Silently disabling them accepts a different geometric problem.
            configurations = [(max(1, int(target_faces * factor)), float(crease_deg)) for factor in (1, 2, 4, 8)]
            for number, (requested, crease) in enumerate(configurations):
                out_obj = Path(tmp_dir) / f"output_quad_{number}.obj"
                cmd = [str(im_exe), "-o", str(out_obj), "-r", "4", "-p", "4",
                       "-f", str(requested), "-d"]
                if crease > 0:
                    cmd.extend(["-c", str(crease)])
                cmd.append(str(in_obj))
                logger.info("Running Instant-Meshes: %s", " ".join(cmd))
                attempt = {"requested_faces": requested, "crease_deg": crease}
                try:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=60,
                                   **headless_subprocess_kwargs())
                    q_verts, q_faces = load_quad_mesh_from_obj(out_obj)
                    attempt["quad_face_count"] = int(len(q_faces))
                    if len(q_faces) > maximum_quad_faces:
                        error = (f"Instant-Meshes produced {len(q_faces)} quads, exceeding the "
                                 f"bounded CAD budget of {maximum_quad_faces}")
                        attempt["error"] = error
                        fallback_reason = error
                        attempts.append(attempt)
                        logger.warning("Instant-Meshes candidate rejected: %s", error)
                        continue
                    topology = mesh_topology_report(q_verts, q_faces)
                    attempt["topology"] = topology
                    error = _quad_topology_error(source_topology, topology)
                    output_triangles = np.vstack((q_faces[:, [0, 1, 2]], q_faces[:, [0, 2, 3]]))
                    if source_topology["is_closed"]:
                        import trimesh
                        source_signed_volume = float(trimesh.Trimesh(vertices, faces, process=False).volume)
                        output_signed_volume = float(trimesh.Trimesh(q_verts, output_triangles, process=False).volume)
                        if (source_topology['connected_components'] == 1 and
                                source_signed_volume * output_signed_volume < 0):
                            # Instant Meshes may reverse a consistently wound
                            # closed surface. Restore its source orientation
                            # before parameterization; the geometry is unchanged.
                            q_faces = q_faces[:, ::-1].copy()
                            output_triangles = np.vstack((q_faces[:, [0,1,2]], q_faces[:, [0,2,3]]))
                            attempt['global_winding_corrected'] = True
                        source_volume = abs(source_signed_volume)
                        output_volume = abs(output_signed_volume)
                        delta = abs(output_volume - source_volume) / source_volume if source_volume > 0 else None
                        attempt["relative_volume_delta"] = delta
                        if error is None and (delta is None or delta > 0.03):
                            error = "Instant-Meshes changed closed-mesh volume by more than 3%"
                    deviation, validation_error = _validate_triangle_surface(
                        triangle_surface_validator, q_verts, output_triangles)
                    if deviation is not None:
                        attempt["sampled_source_deviation"] = deviation
                    if validation_error:
                        error = "; ".join(value for value in (error, validation_error) if value)
                    if error:
                        attempt["error"] = error
                        fallback_reason = error
                        logger.warning("Instant-Meshes candidate check failed: %s; retrying", error)
                        attempts.append(attempt)
                        continue
                    attempts.append(attempt)
                    return q_verts, q_faces, {
                        "step": 2, "quadrangulator": "Instant-Meshes", "vertex_count": len(q_verts),
                        "quad_face_count": len(q_faces), "paper_stage_2": True,
                        "maximum_quad_faces": maximum_quad_faces,
                        "source_topology": source_topology, "topology": topology,
                        "attempts": attempts,
                        **({"sampled_source_deviation": deviation} if deviation is not None else {}),
                    }
                except Exception as exc:
                    fallback_reason = str(exc)
                    attempt["error"] = fallback_reason
                    attempts.append(attempt)
                    logger.warning("Instant-Meshes failed (%s); retrying", exc)

    if require_native:
        raise QuadrangulationValidationError(
            'Instant Meshes could not produce a topology-preserving quad mesh: ' + fallback_reason,
            {'step': 2, 'paper_stage_2': False, 'source_topology': source_topology,
             'attempts': attempts, 'fallback_reason': fallback_reason})

    quad_budget = min(60000, max(12000, 3 * int(target_faces)))
    # A lattice with hundreds of handles cannot be represented by the same
    # bounded polygon count as a beam. Retain the strict topology/3% volume
    # checks while allowing an explicitly reported larger CAD workload.
    handle_count = max(0, (2 * source_topology["connected_components"] -
                          source_topology["euler_characteristic"]) // 2)
    maximum_quad_budget = 60000
    if source_topology["is_closed"] and handle_count >= 100:
        maximum_quad_budget = 360000 if handle_count >= 2000 else 120000
    requested_initial_budget = quad_budget
    # For a closed triangular manifold F=2(V-chi), and every component
    # requires at least four vertices. Subdivision into three quads per face
    # cannot possibly meet a budget below this topology-derived lower bound.
    minimum_subdivision_quads = 0
    if source_topology["is_closed"]:
        minimum_subdivision_quads = 3 * max(0, 8 * source_topology["connected_components"] -
                                            2 * source_topology["euler_characteristic"])
        if minimum_subdivision_quads > quad_budget:
            quad_budget = min(maximum_quad_budget, 60000 * int(np.ceil(minimum_subdivision_quads / 60000)))
    try:
        fallback_vertices, fallback_faces, simplification = _bounded_fallback_triangles(
            vertices, faces, source_topology, quad_budget, maximum_quad_budget, triangle_surface_validator)
    except QuadrangulationValidationError as exc:
        exc.report = {"step": 2, "source_topology": source_topology, "attempts": attempts,
                      "fallback_reason": fallback_reason, "fallback_simplification": exc.report}
        raise
    simplification["requested_initial_quad_budget"] = requested_initial_budget
    simplification["topological_minimum_subdivision_quads"] = minimum_subdivision_quads
    q_verts, q_faces = fallback_quadrangulate_mesh(fallback_vertices, fallback_faces)
    topology = mesh_topology_report(q_verts, q_faces)
    error = _quad_topology_error(source_topology, topology)
    if error:
        raise ValueError(f"Fallback quadrangulation also failed: {error}")
    return q_verts, q_faces, {
        "step": 2, "quadrangulator": "Python Fallback", "vertex_count": len(q_verts),
        "quad_face_count": len(q_faces), "paper_stage_2": False,
        "maximum_quad_faces": maximum_quad_faces,
        "fallback_reason": fallback_reason,
        "method": "Triangle midpoint subdivision; does not optimize a 4-RoSy/4-PoSy field",
        "source_topology": source_topology, "topology": topology, "attempts": attempts,
        "fallback_simplification": simplification,
        **({"sampled_source_deviation": simplification["sampled_source_deviation"]}
           if "sampled_source_deviation" in simplification else {}),
    }
