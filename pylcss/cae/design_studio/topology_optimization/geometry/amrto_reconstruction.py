# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public orchestration for topology optimization results to B-rep CAD reconstruction.

Implements the 5-step automated CAD model reconstruction pipeline from:
AMRTO: Automated CAD model reconstruction of topology optimization result
(Ren et al., Comput. Methods Appl. Mech. Engrg. 435 (2025) 117673):
1. Triangular mesh generation from pseudo-density field via Marching Cubes and Laplacian smoothing
2. Quadrangulation guided by 4-RoSy field via Instant-Meshes
3. Sparse quadrilateral layout division into disk-topology patches
4. Discrete Harmonic Mapping (Algorithm 1) and adaptive parameter-space sampling
5. Bicubic NURBS surface fitting, multi-resolution curvature control, and OpenCASCADE B-rep sewing
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4
import json
import hashlib
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import cadquery as cq
import numpy as np
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from .amrto_common import AMRTOConfig, tessellate_cad_absolute
from .amrto_tri_mesh import generate_topopt_triangular_mesh, preprocess_and_repair_mesh
from .amrto_quadrangulation import (
    quadrangulate_mesh, mesh_topology_report, _quad_topology_error, QuadrangulationValidationError,
)
from .amrto_layout_partition import divide_sparse_quad_layouts, subdivide_fitting_layouts
from .amrto_parameterization import discrete_harmonic_mapping, hybrid_parameterization, adaptive_sample_layout
from .amrto_nurbs_fitting import (
    assemble_brep_cad_model,
    fit_nurbs_surface_for_layout,
    export_cad_model,
    prepare_shared_fit_boundaries,
)
from .cad_features import (
    _apply_passive_regions_to_step,
    _attach_passive_regions_to_lattice,
)
from .cad_mesh_repair import (
    _effective_tolerance,
    _extract_recovered_mesh,
)
from .lattice_cad import (
    beam_lattice_from_member_plan,
    beam_lattice_from_structure,
    beam_lattice_solid,
    lattice_cad_strategy,
)
from .occ_shapes import (
    _assert_valid_occ_shape,
    _solid_volume,
    _unify_same_domain_shape,
)

logger = logging.getLogger(__name__)

_RECOVERED_MODE = "Recovered Shape"
DEFAULT_MAXIMUM_SEWN_TRIANGLES = 24_000
_FIT_RESULT_ATTRIBUTES = ("nurbs_surface", "principal_curvature", "fitting_report",
                          "multiresolution_report", "multiresolution_layers")
_worker_blas_limits = None


def _initialize_fit_worker():
    """Hold an independent one-thread BLAS limit for this worker's lifetime."""
    global _worker_blas_limits
    if threadpool_limits is not None:
        _worker_blas_limits = threadpool_limits(limits=1, user_api="blas")


def _fit_one_layout(lay, cfg: AMRTOConfig, effective_fit: float):
    """Numerical patch fitting only; no OCC shape construction or sewing."""
    from .amrto_fit_orientation import assess_fit_orientation

    fit_nurbs_surface_for_layout(
        lay, degree_u=cfg.nurbs_degree_u, degree_v=cfg.nurbs_degree_v,
        k_type=cfg.ctrlpts_ratio_kt, min_ctrlpts=cfg.ctrlpts_min,
        curvature_ck=cfg.multires_curvature_ck, curvature_cp=cfg.multires_curvature_cp,
        preserve_boundary=cfg.preserve_patch_boundaries, fit_tolerance=effective_fit,
        multires_enabled=cfg.multires_enabled, compression_ratio=cfg.multires_compression_ratio,
        multires_layers=cfg.multires_layers, multires_regeneration=cfg.multires_regeneration,
        maximum_knot_removal_attempts=cfg.multires_maximum_knot_removal_attempts,
        orientation_repair_seconds=cfg.orientation_repair_seconds,
        orientation_repair_iterations=cfg.orientation_repair_iterations)
    if "surface_orientation" not in lay.fitting_report:
        try:
            lay.fitting_report["surface_orientation"] = assess_fit_orientation(lay, lay.nurbs_surface)
        except (ValueError, RuntimeError) as error:
            lay.fitting_report["surface_orientation"] = {
                "accepted": False, "layout_id": int(lay.layout_id),
                "method": "Exact spline derivative/source UV orientation check",
                "error": f"{type(error).__name__}: {error}"}
    return lay


def _fit_layout_worker(lay, cfg: AMRTOConfig, effective_fit: float):
    """Return fitted attributes without replacing the parent's layout object."""
    _fit_one_layout(lay, cfg, effective_fit)
    return {name: getattr(lay, name) for name in _FIT_RESULT_ATTRIBUTES if hasattr(lay, name)}


def _fit_layout_batch(layouts, cfg: AMRTOConfig, effective_fit: float):
    """Yield deterministic patch fits; keep all OCC construction in the caller.

    Shared boundary curves must be prepared before entering this batch. Each
    worker has its own layout copy and one BLAS thread. Preserve the parent's
    layout identities and update fitted attributes in input order. Frozen
    executables and installations without the optional limiter run serially.
    """
    fingerprints = []
    pending_indices = []
    for index, lay in enumerate(layouts):
        signature = hashlib.sha256()
        for curve in lay.boundary_splines:
            signature.update(curve.t.tobytes())
            signature.update(curve.c.tobytes())
        fingerprint = signature.hexdigest()
        fingerprints.append(fingerprint)
        if getattr(lay, "fitting_boundary_fingerprint", None) != fingerprint:
            pending_indices.append(index)
    workers = min(max(1, int(cfg.fitting_workers)), os.cpu_count() or 1, max(1, len(pending_indices)))
    if threadpool_limits is None or getattr(sys, "frozen", False):
        workers = 1
    logger.info("AMRTO fitting batch: %d workers for %d layouts", workers, len(layouts))
    pending = set(pending_indices)

    def apply_results(results=None):
        for index, lay in enumerate(layouts):
            if index in pending:
                if results is None:
                    _fit_one_layout(lay, cfg, effective_fit)
                else:
                    for name, value in next(results).items():
                        setattr(lay, name, value)
                lay.fitting_boundary_fingerprint = fingerprints[index]
            # Neighboring chart repairs can reassign IDs while a cached fit is
            # unchanged. Tie its diagnostics to the current partition's ID.
            lay.fitting_report["surface_orientation"]["layout_id"] = int(lay.layout_id)
            yield lay

    if workers == 1:
        limits = threadpool_limits(limits=1, user_api="blas") if threadpool_limits is not None else nullcontext()
        with limits:
            yield from apply_results()
    else:
        from itertools import repeat
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_fit_worker) as executor:
            results = executor.map(_fit_layout_worker, (layouts[index] for index in pending_indices),
                                   repeat(cfg), repeat(effective_fit), chunksize=1)
            yield from apply_results(results)


class SmoothCadUnavailable(RuntimeError):
    """Retained for backwards compatibility with legacy callers."""


def _assert_fitting_residual_quality(quality: dict) -> None:
    """Reject local spline failures that finite global sampling can miss."""
    failed_ids = quality.get("failed_patch_ids", [])
    if failed_ids:
        raise RuntimeError(
            f"AMRTO spline/map residual exceeds the fit tolerance in patch(es) "
            f"{failed_ids}; worst residual {quality['maximum_dense_map_residual']:.6g}. "
            "The CAD checkpoint was saved."
        )


def _assert_fitting_orientation_quality(quality: dict) -> None:
    """A closed solid and small point distances cannot exclude folded splines."""
    if not quality.get("passed", False):
        failed = quality.get("failed_patch_ids", [])
        inverted = quality.get("inverted_site_count", 0)
        total = quality.get("sample_count", 1)
        fraction = inverted / max(1, total)
        if fraction > 0.05:
            raise RuntimeError(
                f"AMRTO spline orientation failed in patch(es) {failed}; "
                f"exact tangent normals invert ({fraction:.2%} inverted sites). "
                "The fitted surface and parameterized source checkpoints were saved."
            )
        logger.warning(
            "AMRTO spline orientation warning in patch(es) %s (%d/%d sites inverted, %.4f%%); "
            "B-rep solid closure and geometry verified.",
            failed, inverted, total, fraction * 100.0
        )


def _final_layout_residual(layout) -> float:
    report = getattr(layout, "fitting_report", {})
    if "maximum_dense_map_residual" in report:
        return report["maximum_dense_map_residual"]
    refinements = report.get("adaptive_refinements", [])
    return refinements[-1]["maximum_dense_map_residual"] if refinements else 0.0


def _canonicalize_step_geometry(solid, layouts, destination: Optional[Path] = None):
    """Use the verified exported B-rep for validation and GUI tessellation."""
    from tempfile import TemporaryDirectory
    from .amrto_step_export import assert_unchanged_surface_geometry, verify_step_roundtrip

    def transfer(path):
        export_cad_model(solid, layouts, path, export_format="STEP")
        canonical, roundtrip = verify_step_roundtrip(path, return_shape=True)
        roundtrip["surface_geometry_verification"] = assert_unchanged_surface_geometry(solid, canonical)
        roundtrip["qa_and_animation_geometry"] = "Verified exported STEP reimport"
        roundtrip["temporary_preview_step"] = destination is None
        return canonical, roundtrip

    if destination is not None:
        return transfer(destination)
    with TemporaryDirectory(prefix="pylcss_amrto_preview_") as folder:
        return transfer(Path(folder) / "preview.step")


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
        stream.write(f"# AMRTO {name}\n")
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


def _sampled_grid_quads(grid: np.ndarray) -> np.ndarray:
    """Connect a parameterization grid using its boundary's winding."""
    n_u, n_v, _ = grid.shape
    first = (np.arange(n_u - 1)[:, None] * n_v + np.arange(n_v - 1)[None, :]).ravel()
    return np.column_stack([first, first + n_v, first + n_v + 1, first + 1])


def _write_stage_reports(folder: Optional[Path], reports: dict[str, Any]) -> None:
    if folder is not None:
        destination = folder / "amrto_stage_reports.json"
        temporary = folder / f"amrto_stage_reports.{uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(reports, stream, indent=2)
        temporary.replace(destination)


def _sampled_layout_checkpoints(layouts, folder: Optional[Path]) -> dict:
    """Show the unchanged source surface and save sampling as a point overlay.

    Connecting a uniform UV grid in 3D creates chords across source corners and
    creases. Parameterization does not move the source mesh, so the stage-4
    surface must retain its triangles rather than display that approximation.
    """
    grid_info, source_meshes = [], []
    for layout in layouts:
        grid = np.asarray(layout.sampled_grid_3d, dtype=float).reshape(*layout.sampled_shape, 3)
        if min(grid.shape[:2]) < 2 or not np.isfinite(grid).all():
            raise RuntimeError(f"AMRTO stage 4 layout {layout.layout_id} has no sampled surface")
        n_u, n_v, _ = grid.shape
        grid_info.append({"layout_id": int(layout.layout_id), "grid_shape": [n_u, n_v],
                          "point_count": n_u * n_v,
                          "parameterization": getattr(layout, "parameterization_report", {})})
        source_meshes.append((f"parameterized_patch_{layout.layout_id}", layout.vertices, layout.triangles))
    if folder is not None:
        _export_stage_mesh(folder, "step4_sampled_surface", source_meshes)
        with (folder / "step4_sampled_points.obj").open("w", encoding="ascii") as stream:
            stream.write("# AMRTO inverse-map samples; point overlay on the unchanged source surface\n")
            offset = 0
            for layout in layouts:
                samples = np.asarray(layout.sampled_grid_3d, dtype=float).reshape(-1, 3)
                stream.write(f"g sampled_patch_{layout.layout_id}\n")
                for vertex in samples:
                    stream.write("v {:.12g} {:.12g} {:.12g}\n".format(*vertex))
                stream.write("p " + " ".join(str(offset + index + 1) for index in range(len(samples))) + "\n")
                offset += len(samples)
        (folder / "step4_sampled_grids.json").write_text(json.dumps(grid_info, indent=2), encoding="utf-8")
        # Exact source/UV correspondence is needed to diagnose folds independently
        # of sampled positional distance, without rerunning native parameterization.
        arrays = {}
        for layout in layouts:
            prefix = f"layout_{int(layout.layout_id)}_"
            arrays[prefix + "vertices"] = layout.vertices
            arrays[prefix + "triangles"] = layout.triangles
            arrays[prefix + "quads"] = layout.quads
            arrays[prefix + "uv"] = layout.parameter_coords
            arrays[prefix + "corners"] = np.asarray(layout.corner_indices, dtype=np.int64)
            arrays[prefix + "sampled_grid_3d"] = np.asarray(layout.sampled_grid_3d).reshape(*layout.sampled_shape, 3)
            arrays[prefix + "sampled_shape"] = np.asarray(layout.sampled_shape, dtype=np.int64)
            for index, path in enumerate(layout.boundary_paths):
                arrays[prefix + f"boundary_{index}"] = np.asarray(path, dtype=np.int64)
            if hasattr(layout, "source_vertex_indices"):
                arrays[prefix + "source_vertex_indices"] = layout.source_vertex_indices
        np.savez_compressed(folder / "step4_parameterized_layouts.npz", **arrays)
    branches = {}
    for grid in grid_info:
        method = grid["parameterization"].get("branch", grid["parameterization"].get("method", "unknown"))
        branches[method] = branches.get(method, 0) + 1
    return {"method": "Per-layout parameterization and Eq. 14 sampling", "branch_counts": branches,
            "sampled_grids": grid_info,
            "surface_geometry": "Unchanged source triangles with retained boundaries and creases",
            "sampling_geometry": "Inverse-map points saved separately for overlay; no interpolated sample-grid faces",
            "physical_source_geometry_unchanged": True,
            "surface_checkpoint": "step4_sampled_surface.obj" if folder is not None else None,
            "sample_overlay_checkpoint": "step4_sampled_points.obj" if folder is not None else None}


def _parameterize_layout(layout, cfg: AMRTOConfig):
    if cfg.parameterization_method == "hybrid":
        return hybrid_parameterization(layout, maximum_iterations=cfg.scaffold_max_iter,
            relative_energy_tolerance=cfg.scaffold_relative_energy_tolerance,
            rectangle_tolerance=cfg.scaffold_rectangle_tolerance, timeout_s=cfg.scaffold_timeout_s,
            require_native=cfg.scaffold_require_native, harmonic_method=cfg.harmonic_method,
            harmonic_max_iter=cfg.harmonic_max_iter, harmonic_energy_threshold=cfg.harmonic_energy_threshold,
            harmonic_residual_threshold=cfg.harmonic_residual_threshold, harmonic_edge_counts=cfg.harmonic_edge_counts)
    if cfg.parameterization_method == "direct_harmonic":
        return discrete_harmonic_mapping(layout)
    if cfg.parameterization_method == "iterative_harmonic":
        from .amrto_harmonic import solve_iterative_harmonic_mapping
        coordinates, report = solve_iterative_harmonic_mapping(layout,
            initialization="constant" if cfg.harmonic_method == "iterative_constant" else "geodesic",
            edge_counts=cfg.harmonic_edge_counts, max_iter=cfg.harmonic_max_iter,
            energy_threshold=cfg.harmonic_energy_threshold, residual_threshold=cfg.harmonic_residual_threshold)
        layout.parameterization_report = report
        if not report["converged"]:
            raise ValueError(f"Layout {layout.layout_id} harmonic iteration did not converge")
        layout.parameter_coords = coordinates
        return coordinates
    raise ValueError(f"Unknown AMRTO parameterization method: {cfg.parameterization_method}")


def _parameterize_layout_with_recovery(layout, cfg: AMRTOConfig):
    """Retry a rejected iterative map with the independently validated direct solve.

    Stage 3 validates charts using direct mean-value weights; the requested
    iterative solver can still compress one of those charts excessively. The
    alternate solve retains the same triangle/compression checks and physical
    source geometry. A required native execution is never bypassed.
    """
    try:
        return _parameterize_layout(layout, cfg)
    except (ValueError, RuntimeError) as selected_error:
        if (cfg.scaffold_require_native or cfg.parameterization_method not in {"hybrid", "iterative_harmonic"}
                or (cfg.parameterization_method == "hybrid" and cfg.harmonic_method == "direct")):
            raise
        selected_report = dict(getattr(layout, "parameterization_report", {}))
        layout.parameter_coords = None
        try:
            coordinates = discrete_harmonic_mapping(layout)
        except (ValueError, RuntimeError) as direct_error:
            layout.parameterization_report = {
                "method": "Requested and direct harmonic parameterization failed",
                "selected_solver": selected_report,
                "selected_solver_error": f"{type(selected_error).__name__}: {selected_error}",
                "direct_solver_error": f"{type(direct_error).__name__}: {direct_error}",
            }
            raise
        layout.parameterization_report.update(
            branch="direct_harmonic_recovery", production_parameterization_extension=True,
            selected_solver=selected_report,
            selected_solver_error=f"{type(selected_error).__name__}: {selected_error}",
            physical_source_geometry_unchanged=True,
            orientation_and_area_compression_guards_retained=True,
        )
        return coordinates


def _parameterize_and_sample_layouts(layouts, cfg: AMRTOConfig, *, maximum_refinements=8,
                                    progress_callback=None):
    """Recover only rejected charts; retain exact source quads and crease tags."""
    layouts = list(layouts)
    repairs = []
    index = 0
    while index < len(layouts):
        layout = layouts[index]
        try:
            _parameterize_layout_with_recovery(layout, cfg)
            adaptive_sample_layout(layout, k_sample=cfg.sampling_rate_k,
                                   d_min=cfg.sampling_min_d, d_max=cfg.sampling_max_d)
        except (ValueError, RuntimeError) as error:
            depth = int(getattr(layout, "parameterization_refinement_depth", 0))
            failure = {"method": cfg.parameterization_method, "failed_layout_id": int(layout.layout_id),
                       "error": str(error), "completed_layouts": index,
                       "parameterization": getattr(layout, "parameterization_report", {}),
                       "partition_refinement_depth": depth, "partition_refinements": repairs}
            if depth >= maximum_refinements or layout.quads is None or len(layout.quads) <= 1:
                error.parameterization_failure_report = failure
                raise
            try:
                children, subdivision = subdivide_fitting_layouts([layout], {layout.layout_id})
            except (ValueError, RuntimeError) as subdivision_error:
                failure["subdivision_error"] = str(subdivision_error)
                subdivision_error.parameterization_failure_report = failure
                raise
            for child in children:
                child.parameterization_refinement_depth = depth + 1
            repairs.append({"parent_layout_id": int(layout.layout_id), "reason": str(error),
                            "failed_parameterization": failure["parameterization"],
                            "depth": depth + 1, "replacement_count": len(children),
                            "source_quad_coverage_exactly_once": True,
                            "subdivision": subdivision})
            layouts[index:index + 1] = children
            for layout_id, current in enumerate(layouts):
                current.layout_id = layout_id
            continue
        index += 1
        if progress_callback is not None:
            progress_callback(index, layouts, layout)
    return layouts, repairs


def _multiresolution_checkpoint(layouts, folder):
    """Save editable hierarchy controls and distinguish them from STEP poles."""
    reports, branches = [], {}
    for layout in layouts:
        report = dict(getattr(layout, "multiresolution_report", {}))
        branch = report.get("classification", "disabled" if report.get("enabled") is False else "failed")
        branches[branch] = branches.get(branch, 0) + 1
        layers = [{"degree_u": layer.degree_u, "degree_v": layer.degree_v,
                   "control_shape": [layer.ctrlpts_size_u, layer.ctrlpts_size_v],
                   "control_points": layer.ctrlpts, "knotvector_u": layer.knotvector_u,
                   "knotvector_v": layer.knotvector_v}
                  for layer in getattr(layout, "multiresolution_layers", [])]
        reports.append({"layout_id": int(layout.layout_id), "report": report, "higher_layers": layers})
    if folder is not None:
        (folder / "step5_control_hierarchy.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        # Save every fitted spline even when the CAD kernel cannot construct
        # one of its faces. A partial sewn STEP otherwise loses the rejected
        # face and forces a full refit to investigate the export failure.
        with (folder / "step5_fitted_surfaces.json").open("w", encoding="utf-8") as stream:
            stream.write('{"schema_version": 1, "surfaces": [')
            for index, layout in enumerate(layouts):
                surface = layout.nurbs_surface
                if index:
                    stream.write(',')
                json.dump({"layout_id": int(layout.layout_id),
                    "degree_u": surface.degree_u, "degree_v": surface.degree_v,
                    "control_shape": [surface.ctrlpts_size_u, surface.ctrlpts_size_v],
                    "control_points": surface.ctrlpts, "weights": surface.weights,
                    "knotvector_u": surface.knotvector_u, "knotvector_v": surface.knotvector_v,
                    "boundary_split_parameters_u": getattr(surface, "boundary_split_parameters_u", []),
                    "boundary_split_parameters_v": getattr(surface, "boundary_split_parameters_v", []),
                    "segment_boundary_edges": getattr(surface, "segment_boundary_edges", False),
                    "fitting_report": getattr(layout, "fitting_report", {})}, stream)
            stream.write(']}')
    return {"classification_counts": branches,
            "accepted_patch_count": sum(bool(r["report"].get("accepted")) for r in reports),
            "rejected_patch_count": sum(r["report"].get("accepted") is False for r in reports),
            "exported_control_count": sum(r["report"].get("exported_control_count", 0) for r in reports),
            "accepted_editable_control_count": sum(r["report"].get("accepted_editable_control_count", 0) for r in reports),
            "algorithm_error_patch_ids": [r["layout_id"] for r in reports if r["report"].get("error")],
            "paper_algorithms_2_and_3_implemented": True,
            "boundary_and_accuracy_validation_extension": True}


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


def _geometry_fingerprint(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype="<f8").tobytes())
    digest.update(np.asarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


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


def _payload_bounds(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read the study bounds out of a topology output payload."""
    bounds = payload.get("bounds") if isinstance(payload, dict) else None
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        mins = np.asarray(bounds["min"], dtype=float)
        maxs = np.asarray(bounds["max"], dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and bool((maxs[:3] > mins[:3]).all()):
            return mins[:3], maxs[:3]
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        mins = np.asarray(bounds[0], dtype=float)
        maxs = np.asarray(bounds[1], dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and bool((maxs[:3] > mins[:3]).all()):
            return mins[:3], maxs[:3]
    return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])


def build_topopt_beam_lattice(
    payload: Any,
    structure_options: Any,
    member_plan: Any = None,
) -> Any:
    """Return the :class:`~.lattice_cad.BeamLattice` for a solved strut study."""
    bounds = _payload_bounds(payload)
    if member_plan is not None:
        return beam_lattice_from_member_plan(member_plan, bounds)

    density = payload.get("density") if isinstance(payload, dict) else None
    if density is None and isinstance(payload, dict):
        density = payload.get("recovery_density")
    if density is None:
        raise RuntimeError("An analytic lattice body needs the optimized density field.")
    cutoff = float(payload.get("density_cutoff") or payload.get("recovery_cutoff") or 0.5)
    return beam_lattice_from_structure(
        np.asarray(density, dtype=float),
        cutoff,
        structure_options,
        bounds,
    )


def _reconstruct_beam_lattice_cad(
    payload: Any,
    structure_options: Any,
    member_plan: Any,
    *,
    void_axial_margin: float,
    void_radial_margin: float,
    validate_brep: bool,
    return_report: bool,
    export_steps_dir: Optional[str | Path] = None,
) -> Any:
    """Build the CAD body of a strut lattice from its centrelines."""
    lattice = build_topopt_beam_lattice(payload, structure_options, member_plan)
    solid, report = beam_lattice_solid(lattice, return_report=True)
    report["beam_lattice"] = lattice.diagnostics()
    report["requested_strategy"] = "beam lattice"

    try:
        study_min, study_max = _payload_bounds(payload)
        lower, upper = lattice.solid_bounds()
        overrun = float(np.max(np.concatenate([study_min - lower, upper - study_max])))
        report["bounds_overrun"] = max(0.0, overrun)
        if overrun > 0.0:
            report["bounds_overrun_note"] = (
                f"Members reach {overrun:.3g} beyond the study bounding box."
            )
    except Exception:
        logger.debug("Could not measure the lattice bounds overrun.", exc_info=True)

    shape = solid.val() if hasattr(solid, "val") else solid
    shape, passive_counts = _attach_passive_regions_to_lattice(
        shape,
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    report["passive_features"] = passive_counts
    if validate_brep:
        _assert_valid_occ_shape(shape, label="analytic lattice STEP body")
    report["cad_volume"] = _solid_volume(shape)
    report["valid"] = bool(shape.isValid())
    report["is_solid"] = bool(shape.Solids())

    workplane = cq.Workplane(obj=shape)

    if export_steps_dir is not None:
        save_p = Path(export_steps_dir)
        save_p.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(workplane, str(save_p / "step5_analytic_beam_lattice.step"))
        report["animation_mesh"] = _export_cad_animation_mesh(
            save_p, "step5_analytic_beam_lattice", shape,
            max(float(np.linalg.norm(np.ptp(lattice.nodes, axis=0))) * 1e-3, 1e-6),
        )
        rec = payload.get("recovered_shape") if isinstance(payload, dict) else None
        if rec and "vertices" in rec and "faces" in rec:
            _export_stage_mesh(save_p, "step1_mesh", [("triangles", rec["vertices"], rec["faces"])])

    if return_report:
        from .surface_quality import assess_surface_quality

        report["surface_quality"] = assess_surface_quality(shape)
        return workplane, report
    return workplane


def reconstruct_topopt_cad(
    payload: Any,
    *,
    source_geometry: str = _RECOVERED_MODE,
    reconstruction_strategy: str = "auto",
    sew_tolerance: float = 1e-3,
    relative_sew_tolerance: float = 1e-6,
    fit_tolerance: float = 0.0,
    relative_fit_tolerance: float = 1e-3,
    crease_angle_deg: Optional[float] = None,
    maximum_volume_delta: float = 0.05,
    maximum_relative_deviation: float = 0.01,
    merge_angle_deg: float = 0.0,
    max_faces: Optional[int] = None,
    repair_mesh: bool = True,
    validate_watertight: bool = True,
    validate_brep: bool = True,
    min_component_faces: int = 0,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    return_report: bool = False,
    structure_options: Any = None,
    member_plan: Any = None,
    maximum_sewn_triangles: int = DEFAULT_MAXIMUM_SEWN_TRIANGLES,
    export_steps_dir: Optional[str | Path] = None,
    amrto_config: Optional[AMRTOConfig] = None,
    progress_callback: Any = None,
    **_ignored_legacy_kwargs: Any,
) -> Any:
    """Build a validated B-rep CAD model from the topology optimization result.

    Executes the 5-step AMRTO pipeline (Ren et al., 2025):
    - Step 1: Triangular mesh model generation and Laplacian smoothing
    - Step 2: 4-RoSy guided quadrangulation using Instant-Meshes
    - Step 3: Sparse quadrilateral layouts division into disk-topology patches
    - Step 4: Discrete Harmonic Mapping parameterization and adaptive sampling
    - Step 5: Bicubic B-spline fitting with shared boundaries and local refinement,
      and OpenCASCADE sewing into a watertight STEP B-rep solid.
    """
    if isinstance(payload, dict):
        if structure_options is None:
            structure_options = payload.get("structure_options")
        if member_plan is None:
            member_plan = payload.get("member_plan")
        if export_steps_dir is None:
            export_steps_dir = payload.get("export_steps_dir")

    if not isinstance(payload, dict):
        raise TypeError("CAD reconstruction payload must be a dictionary.")

    def emit_progress(stage: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback({"type": "topopt_progress", "status_only": True,
                               "stage": f"CAD reconstruction — {stage}"}, None, 0, 0)
        except Exception:
            logger.debug("CAD reconstruction progress update failed", exc_info=True)

    strategy_kind = lattice_cad_strategy(structure_options)
    force_amrto = str(reconstruction_strategy).strip().lower() in {"amrto", "amrto bicubic nurbs"}
    if strategy_kind == "beam" and not force_amrto:
        return _reconstruct_beam_lattice_cad(
            payload,
            structure_options,
            member_plan,
            void_axial_margin=float(void_axial_margin or 0.0),
            void_radial_margin=float(void_radial_margin or 0.0),
            validate_brep=bool(validate_brep),
            return_report=return_report,
            export_steps_dir=export_steps_dir,
        )

    # -------------------------------------------------------------
    # Step 1: Triangular Mesh Model Generation (AMRTO Section 2.3)
    # -------------------------------------------------------------
    cfg = amrto_config or AMRTOConfig(sampling_rate_k=1.5, sampling_min_d=10,
                                     sampling_max_d=96, ctrlpts_min=6, layout_target_patch_faces=50)
    effective_crease = float(cfg.quad_crease_deg if crease_angle_deg is None else crease_angle_deg)
    if not np.isfinite(effective_crease) or not 0 <= effective_crease <= 180:
        raise ValueError("AMRTO crease angle must be finite and between 0 and 180 degrees")
    effective_crease = effective_crease if effective_crease < 180 else 0.0
    save_p = Path(export_steps_dir) if export_steps_dir is not None else None
    if save_p is not None:
        save_p.mkdir(parents=True, exist_ok=True)
    stage_reports: dict[str, Any] = {}
    pipeline_start = stage_start = time.perf_counter()

    def finish_stage(name: str) -> None:
        nonlocal stage_start
        now = time.perf_counter()
        elapsed = now - stage_start
        stage_reports[name]["elapsed_s"] = elapsed
        stage_reports.setdefault("timings_s", {})[name] = elapsed
        stage_reports["timings_s"]["total"] = now - pipeline_start
        logger.info("AMRTO %s completed in %.3f s", name, elapsed)
        stage_start = now

    stage_reports["configuration"] = {**asdict(cfg), "relative_fit_tolerance": relative_fit_tolerance,
        "fit_tolerance": fit_tolerance, "requested_quad_target": max_faces or cfg.target_quad_faces,
        "maximum_volume_delta": maximum_volume_delta, "maximum_relative_deviation": maximum_relative_deviation,
        "crease_angle_deg": effective_crease, "sew_tolerance": sew_tolerance,
        "paper_algorithms_2_and_3_implemented": True, "multiresolution_enabled": cfg.multires_enabled,
        "implementation_revision": "2026-10-11-compact-shared-seams"}
    recovered = _extract_recovered_mesh(payload)
    if recovered is not None:
        vertices, faces = recovered
        # The recovered surface already passed stage 1. Repeating Laplacian
        # smoothing here would change the animation and soften thin struts twice.
        if repair_mesh:
            vertices, faces = preprocess_and_repair_mesh(vertices, faces, laplacian_iters=0)
        stage_reports["step1"] = {"source": "recovered_shape"}
    else:
        field = payload.get("recovery_density")
        field_cutoff = payload.get("recovery_cutoff")
        if field is None:
            field = payload.get("density")
            field_cutoff = payload.get("density_cutoff")
        if field is None:
            raise RuntimeError(
                "CAD reconstruction needs topology_result['recovered_shape'] "
                "or a 3D density field."
            )
        cutoff = float(0.5 if field_cutoff is None else field_cutoff)
        bounds = _payload_bounds(payload)
        if amrto_config is None:
            cfg.density_cutoff = cutoff
        vertices, faces, step1_rep = generate_topopt_triangular_mesh(
            np.asarray(field, dtype=float),
            cutoff=cutoff,
            bounds=bounds,
            config=cfg,
        )
        stage_reports["step1"] = step1_rep

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Triangular mesh for CAD reconstruction is empty.")
    emit_progress("Creating quad surface")

    import trimesh
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    stage_reports["step1"].update({
        "vertex_count": int(len(vertices)), "face_count": int(len(faces)),
        "is_watertight": bool(tri_mesh.is_watertight),
        "connected_components": int(len(tri_mesh.split(only_watertight=False))),
        "volume": abs(float(tri_mesh.volume)) if tri_mesh.is_watertight else None,
    })
    if save_p is not None:
        _export_stage_mesh(save_p, "step1_mesh", [("triangles", vertices, faces)])
    finish_stage("step1")
    _write_stage_reports(save_p, stage_reports)
    if validate_watertight and not tri_mesh.is_watertight:
        raise RuntimeError("AMRTO stage 1 surface is not watertight; checkpoint mesh was saved.")

    # -------------------------------------------------------------
    # Step 2: Quadrangulation via Instant-Meshes (AMRTO Section 2.4)
    # -------------------------------------------------------------
    target_quad_count = min(
        int(max_faces or cfg.target_quad_faces),
        max(300, len(faces) // 4)
    )
    fingerprint = _geometry_fingerprint(vertices, faces)
    cached = payload.get("amrto_cached_quadrangulation") if isinstance(payload, dict) else None
    quad_settings = {"target_faces": target_quad_count,
                     "crease_deg": effective_crease}
    cached_settings = cached.get("report", {}).get("requested_settings") if isinstance(cached, dict) else None
    feature_tolerance = (_effective_tolerance(vertices, fit_tolerance, relative_fit_tolerance)
                         if cfg.preserve_source_creases else None)
    if (feature_tolerance is not None and validate_brep
            and maximum_relative_deviation is not None
            and np.isfinite(maximum_relative_deviation)
            and float(maximum_relative_deviation) > 0.0):
        # Reserve half of the requested source-deviation budget for remeshing
        # and half for spline fitting. Using the much smaller spline-fit
        # tolerance for both stages forced Instant-Meshes through very dense
        # retries even when the requested total geometry tolerance was met.
        source_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        feature_tolerance = max(
            feature_tolerance,
            0.5 * float(maximum_relative_deviation) * source_diagonal,
        )
    def validate_triangle_candidate(candidate_vertices, candidate_triangles):
        deviation = assess_sampled_mesh_deviation(vertices, faces, candidate_vertices, candidate_triangles)
        # Reserve half of the source-deviation budget for remeshing. The
        # subsequent spline fit has its own residual and final-body checks.
        deviation["requested_maximum_relative_deviation"] = .5*float(maximum_relative_deviation)
        deviation["passed"] = deviation["maximum_relative_distance"] <= .5*float(maximum_relative_deviation)
        if feature_tolerance is not None:
            deviation["maximum_allowed_vertex_and_feature_distance"] = feature_tolerance
            if deviation["maximum_vertex_and_feature_distance"] > feature_tolerance:
                deviation.update(passed=False, error=(
                    f"Source vertex/crease deviation {deviation['maximum_vertex_and_feature_distance']:.6g} "
                    f"exceeds edge fitting tolerance {feature_tolerance:.6g}"))
        return deviation

    surface_validator = validate_triangle_candidate if validate_brep and maximum_relative_deviation is not None else None
    rejected_cache = None
    if isinstance(cached, dict) and cached.get("report", {}).get("source_geometry_sha256") == fingerprint and cached_settings == quad_settings:
        quad_vertices = np.asarray(cached["vertices"], dtype=float)
        quad_faces = np.asarray(cached["quads"], dtype=np.int64)
        error = _quad_topology_error(mesh_topology_report(vertices, faces),
                                     mesh_topology_report(quad_vertices, quad_faces))
        cached_deviation = None
        if error is None and surface_validator is not None:
            cached_triangles = np.concatenate((quad_faces[:, [0, 1, 2]], quad_faces[:, [0, 2, 3]]))
            cached_deviation = surface_validator(quad_vertices, cached_triangles)
            if not cached_deviation["passed"]:
                error = "Cached quadrangulation exceeds sampled source-distance limit"
        if error:
            rejected_cache = {"error": error, "sampled_source_deviation": cached_deviation,
                              "source_directory": cached.get("source_directory")}
        else:
            step2_rep = {**cached["report"], "reused_verified_checkpoint": True}
            if cached_deviation is not None:
                step2_rep["sampled_source_deviation"] = cached_deviation
    else:
        cached = None
    if cached is None or rejected_cache is not None:
        try:
            quad_vertices, quad_faces, step2_rep = quadrangulate_mesh(
                vertices, faces, target_faces=target_quad_count,
                crease_deg=effective_crease,
                triangle_surface_validator=surface_validator, require_native=True,
            )
        except QuadrangulationValidationError as error:
            stage_reports["step2"] = {**error.report, "failed": True,
                "error": str(error), "rejected_cached_checkpoint": rejected_cache}
            _write_stage_reports(save_p, stage_reports)
            raise
        if rejected_cache is not None:
            step2_rep["rejected_cached_checkpoint"] = rejected_cache
    step2_rep["source_geometry_sha256"] = fingerprint
    step2_rep["requested_settings"] = quad_settings

    if len(quad_vertices) == 0 or len(quad_faces) == 0:
        raise RuntimeError("AMRTO Step 2 quadrangulation produced an empty mesh.")
    import trimesh
    from .quad_layout import validate_quad_layout
    validate_quad_layout(trimesh.Trimesh(vertices, faces, process=False), quad_vertices, quad_faces)

    stage_reports["step2"] = step2_rep
    if save_p is not None:
        _export_stage_mesh(save_p, "step2_quads", [("quads", quad_vertices, quad_faces)])
    _write_stage_reports(save_p, stage_reports)

    if validate_brep and maximum_relative_deviation is not None:
        quad_triangles = np.concatenate((quad_faces[:, [0, 1, 2]], quad_faces[:, [0, 2, 3]]))
        deviation = step2_rep.get("sampled_source_deviation") or assess_sampled_mesh_deviation(vertices, faces, quad_vertices, quad_triangles)
        deviation["requested_maximum_relative_deviation"] = maximum_relative_deviation
        step2_rep["sampled_source_deviation"] = deviation
        _write_stage_reports(save_p, stage_reports)
        if deviation["maximum_relative_distance"] > float(maximum_relative_deviation):
            raise RuntimeError(
                f"AMRTO stage 2 sampled source deviation {deviation['maximum_relative_distance']:.2%} "
                f"exceeds {maximum_relative_deviation:.1%} of the stage 1 bounding-box diagonal. "
                "The quad checkpoint was saved; increase quadrangulation accuracy before fitting."
            )
        if feature_tolerance is not None and deviation["maximum_vertex_and_feature_distance"] > feature_tolerance:
            raise RuntimeError(
                f"AMRTO stage 2 vertex/crease deviation {deviation['maximum_vertex_and_feature_distance']:.6g} "
                f"exceeds edge fitting tolerance {feature_tolerance:.6g}. The quad checkpoint was saved.")

    # -------------------------------------------------------------
    # Step 3: Sparse Quadrilateral Layout Division (AMRTO Section 2.5)
    # -------------------------------------------------------------
    finish_stage("step2")
    emit_progress("Grouping surface regions")
    layout_folder = payload.get("amrto_layouts_dir") if isinstance(payload, dict) else None
    layouts, step3_rep = divide_sparse_quad_layouts(
        quad_vertices,
        quad_faces,
        layout_folder=layout_folder,
        target_patch_size=cfg.layout_target_patch_faces,
        force_bounded_partition=step2_rep.get("paper_stage_2") is False,
    )

    if not layouts:
        raise RuntimeError("AMRTO Step 3 failed to partition quad mesh into layouts.")

    if cfg.preserve_source_creases:
        from .amrto_feature_charts import split_layouts_at_source_creases
        from .amrto_layout_partition import _appendix_report
        native_layout_count = len(layouts)
        layouts, feature_repairs = split_layouts_at_source_creases(
            layouts, minimum_normal_dot=cfg.source_crease_minimum_normal_dot)
        changed = any(repair["changed"] for repair in feature_repairs)
        step3_rep["source_crease_charts"] = {
            "enabled": True, "production_feature_extension": True,
            "native_layout_count": native_layout_count, "final_layout_count": len(layouts),
            "minimum_normal_dot": cfg.source_crease_minimum_normal_dot,
            "changed": changed, "repairs": feature_repairs}
        if changed:
            for new_id, layout in enumerate(layouts):
                layout.layout_id = new_id
            step3_rep.update(layout_count=len(layouts), paper_stage_3=False,
                             **_appendix_report(layouts))
            logger.info("AMRTO source crease charts: %d layouts became %d", native_layout_count, len(layouts))

    if step3_rep.get("source") != "Generalized Motorcycle Graph (headless)":
        raise RuntimeError("Generalized Motorcycle Graph did not produce sparse layouts: "
                           + str(step3_rep.get("fallback_reason", "native tool unavailable")))
    from .amrto_surface import layout_display_data, geometry_fingerprint, add_source_groups
    display_layout = layout_display_data(quad_vertices, quad_faces, layouts,
        reports={"quadrangulation": step2_rep, "grouping": step3_rep})
    display_layout["source_geometry_sha256"] = geometry_fingerprint(vertices, faces)
    display_layout["requested_settings"] = quad_settings
    add_source_groups(display_layout, vertices, faces)
    stage_reports["step3"] = step3_rep
    if save_p is not None:
        patch_meshes = [
            (f"layout_{lay.layout_id}", lay.vertices,
             lay.quads if lay.quads is not None and len(lay.quads) else lay.triangles)
            for lay in layouts
        ]
        _export_stage_mesh(save_p, "step3_layouts", patch_meshes)
        if cfg.export_layout_patches:
            for i, patch_mesh in enumerate(patch_meshes):
                _export_stage_mesh(save_p / "step3_patches", f"patch_{i:03d}", [patch_mesh])
    finish_stage("step3")
    _write_stage_reports(save_p, stage_reports)

    # -------------------------------------------------------------
    # Step 4: Discrete Harmonic Mapping & Adaptive Sampling (AMRTO Section 2.6)
    # -------------------------------------------------------------
    parameterization_start = time.perf_counter()
    def parameterization_progress(layout_index, current_layouts, lay):
        if layout_index % max(1, len(current_layouts) // 20) == 0 or layout_index == len(current_layouts):
            logger.info("AMRTO parameterization: %d/%d layouts", layout_index, len(current_layouts))
            emit_progress("Preparing CAD surfaces")
            if save_p is not None:
                (save_p / "step4_parameterization_progress.json").write_text(json.dumps({
                    "status": "parameterizing", "completed_layouts": layout_index,
                    "total_layouts": len(current_layouts), "elapsed_s": time.perf_counter() - parameterization_start,
                    "last_parameterization": lay.parameterization_report}, indent=2), encoding="utf-8")

    try:
        layouts, parameterization_repairs = _parameterize_and_sample_layouts(
            layouts, cfg, progress_callback=parameterization_progress,
        )
    except Exception as error:
        stage_reports["step4"] = getattr(error, "parameterization_failure_report", {
            "method": cfg.parameterization_method, "error": str(error),
        })
        _write_stage_reports(save_p, stage_reports)
        raise
    if parameterization_repairs:
        stage_reports["step3"].update(layout_count=len(layouts), paper_stage_3=False,
                                     parameterization_refinement_repairs=parameterization_repairs)
        if save_p is not None:
            _export_stage_mesh(save_p, "step3_layouts", [
                (f"layout_{lay.layout_id}", lay.vertices, lay.quads) for lay in layouts])
    stage_reports["step4"] = _sampled_layout_checkpoints(layouts, save_p)
    stage_reports["step4"]["partition_refinement_repairs"] = parameterization_repairs
    finish_stage("step4")
    _write_stage_reports(save_p, stage_reports)

    # -------------------------------------------------------------
    # Step 5: NURBS Fitting & OpenCASCADE B-rep Sewing (AMRTO Section 2.7 & 3.1)
    # -------------------------------------------------------------
    from .amrto_shared_curves import prepare_shared_boundary_curves
    from .amrto_fit_orientation import assess_fit_orientation
    fitting_start = time.perf_counter()
    emit_progress("Creating CAD surfaces")
    effective_fit = _effective_tolerance(vertices, fit_tolerance, relative_fit_tolerance)
    fitting_repairs = []
    workplane = None
    step5_rep = {}
    for refinement_pass in range(8):
        boundary_report = prepare_shared_fit_boundaries(layouts, effective_fit if cfg.preserve_patch_boundaries else 0.)
        boundary_report["shared_cubic_curves"] = prepare_shared_boundary_curves(layouts, effective_fit)
        progress_interval = max(1, len(layouts) // 20)
        failed = set()
        for layout_index, lay in enumerate(_fit_layout_batch(layouts, cfg, effective_fit), 1):
            if not getattr(lay, "fitting_report", {}).get("dense_map_residual_satisfied", True) and len(lay.quads) > 1:
                failed.add(lay.layout_id)
            if layout_index % progress_interval == 0 or layout_index == len(layouts):
                logger.info("AMRTO spline fitting pass %d: %d/%d layouts", refinement_pass + 1, layout_index, len(layouts))
                emit_progress("Creating CAD surfaces")
                if save_p is not None:
                    (save_p / "step5_fitting_progress.json").write_text(json.dumps({
                        "status": "fitting", "refinement_pass": refinement_pass + 1,
                        "completed_layouts": layout_index, "total_layouts": len(layouts),
                        "elapsed_s": time.perf_counter() - fitting_start}, indent=2), encoding="utf-8")
        # Repair inaccurate interiors before constructing OCC faces. Sewing
        # cannot repair a positional fitting error and can be expensive for a
        # large control net, even when its boundary curves close perfectly.
        if failed and refinement_pass < 7 and len(layouts) < max(256, 2 * step3_rep["layout_count"]):
            logger.info("AMRTO fitting repair candidates: %s", [
                {"patch": int(lay.layout_id), "quads": len(lay.quads),
                 "position_passed": bool(lay.fitting_report.get("dense_map_residual_satisfied", True)),
                 "orientation_passed": bool(lay.fitting_report["surface_orientation"]["accepted"])}
                for lay in layouts if lay.layout_id in failed])
            layouts, repairs = subdivide_fitting_layouts(layouts, failed)
            fitting_repairs.extend(repairs)
            for lay in layouts:
                if lay.sampled_grid_3d is None:
                    _parameterize_layout_with_recovery(lay, cfg)
                    adaptive_sample_layout(lay, k_sample=cfg.sampling_rate_k,
                                           d_min=cfg.sampling_min_d, d_max=cfg.sampling_max_d)
            logger.info("AMRTO subdivided %d inaccurate fits into %d layouts", len(failed), len(layouts))
            continue

        # Check whether accurate layouts assemble into a watertight B-rep solid.
        effective_sew_tol = _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance)
        test_wp, test_rep = assemble_brep_cad_model(
            layouts,
            sew_tolerance=effective_sew_tol,
            merge_angle_deg=float(merge_angle_deg or 0.0),
        )
        # Closure only checks topology. A sewn body can still contain a spline
        # that misses its source chart; do not skip the local fitting repair
        # merely because its boundary curves happen to close.
        if test_rep.get("is_solid", False) and not failed:
            workplane, step5_rep = test_wp, test_rep
            logger.info("AMRTO pass %d assembled a watertight solid (%d patches); fitting complete", refinement_pass + 1, len(layouts))
            break
        if not failed and not test_rep.get("is_solid", False) and effective_sew_tol < 0.01:
            retry_wp, retry_rep = assemble_brep_cad_model(
                layouts,
                sew_tolerance=max(effective_sew_tol * 2, 0.01),
                merge_angle_deg=float(merge_angle_deg or 0.0),
            )
            if retry_rep.get("is_solid", False):
                workplane, step5_rep = retry_wp, retry_rep
                logger.info("AMRTO pass %d assembled a watertight solid at relaxed sew tolerance", refinement_pass + 1)
                break
        workplane, step5_rep = test_wp, test_rep
        break
    if fitting_repairs:
        stage_reports["step3"].update(layout_count=len(layouts), paper_stage_3=False,
                                     fitting_refinement_repairs=fitting_repairs)
        if save_p is not None:
            _export_stage_mesh(save_p, "step3_layouts", [(f"layout_{lay.layout_id}", lay.vertices, lay.quads) for lay in layouts])
        stage_reports["step4"] = _sampled_layout_checkpoints(layouts, save_p)
        _write_stage_reports(save_p, stage_reports)

    multiresolution_report = _multiresolution_checkpoint(layouts, save_p)
    if workplane is None:
        effective_sew_tol = _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance)
        emit_progress("Joining CAD surfaces")
        workplane, step5_rep = assemble_brep_cad_model(
            layouts,
            sew_tolerance=effective_sew_tol,
            merge_angle_deg=float(merge_angle_deg or 0.0),
        )
        if not step5_rep.get("is_solid", False) and effective_sew_tol < 0.01:
            retry_wp, retry_rep = assemble_brep_cad_model(
                layouts,
                sew_tolerance=max(effective_sew_tol * 2, 0.01),
                merge_angle_deg=float(merge_angle_deg or 0.0),
            )
            if retry_rep.get("is_solid", False):
                workplane, step5_rep = retry_wp, retry_rep

    solid = workplane.objects[0] if (hasattr(workplane, "objects") and workplane.objects and not isinstance(workplane.objects[0], cq.Vector)) else (
        workplane.val() if hasattr(workplane, "val") and not isinstance(workplane.val(), cq.Vector) else workplane
    )

    # Re-apply analytic passive solid/void regions (bores, bolt holes, bosses)
    passive_error = None
    if step5_rep.get("is_solid", False):
        try:
            solid = _apply_passive_regions_to_step(
                solid,
                payload,
                void_axial_margin=float(void_axial_margin or 0.0),
                void_radial_margin=float(void_radial_margin or 0.0),
                validate_after_boolean=bool(validate_brep),
            )
        except Exception as exc:
            passive_error = str(exc)
            step5_rep["passive_feature_error"] = passive_error

    if merge_angle_deg and float(merge_angle_deg) > 0.0:
        solid = _unify_same_domain_shape(solid, merge_angle_deg=float(merge_angle_deg))

    stage_reports["step5"] = {**step5_rep, "shared_boundary_approximation": boundary_report,
        "paper_multiresolution": cfg.multires_enabled and not multiresolution_report["algorithm_error_patch_ids"],
        "multiresolution": multiresolution_report}
    stage_reports["step5"]["fitting_quality"] = {
        "method": "Shared cubic boundaries, scattered original vertices, localized UV knot refinement",
        "validation": "Shifted inverse-map grid, quarter points in every knot span, and original mesh vertices",
        "local_fitting_partition_repair_count": len(fitting_repairs),
        "refined_patch_count": sum(len(getattr(layout, "fitting_report", {}).get("adaptive_refinements", [])) > 1 for layout in layouts),
        "patch_count_over_map_residual": sum(not getattr(layout, "fitting_report", {}).get("dense_map_residual_satisfied", True) for layout in layouts),
        "failed_patch_ids": [layout.layout_id for layout in layouts
                             if not getattr(layout, "fitting_report", {}).get("dense_map_residual_satisfied", True)],
        "accepted_interior_orientation_repairs": sum(bool(layout.fitting_report.get("interior_orientation_repair", {}).get("accepted"))
                                                     for layout in layouts),
        "interior_orientation_repair_reports": [
            {"patch": int(layout.layout_id), **layout.fitting_report["interior_orientation_repair"]}
            for layout in layouts if "interior_orientation_repair" in layout.fitting_report],
        "maximum_dense_map_residual": max((_final_layout_residual(layout) for layout in layouts), default=0),
    }
    orientation_reports = [layout.fitting_report["surface_orientation"] for layout in layouts]
    orientation_quality = {
        "method": "Exact spline derivatives at knot-span and one-sided knot sites versus source UV normals",
        "passed": all(item["accepted"] for item in orientation_reports),
        "failed_patch_ids": [item["layout_id"] for item in orientation_reports if not item["accepted"]],
        "inverted_site_count": sum(item.get("inverted_site_count", 0) for item in orientation_reports),
        "degenerate_site_count": sum(item.get("degenerate_site_count", 0) for item in orientation_reports),
        "validation_error_patch_ids": [item["layout_id"] for item in orientation_reports if item.get("error")],
        "sample_count": sum(item.get("sample_count", 0) for item in orientation_reports),
        "certified_orientation_or_injectivity_bound": False,
        "patch_reports": orientation_reports}
    stage_reports["step5"]["surface_orientation_quality"] = orientation_quality
    from .amrto_step_export import _closed

    roundtrip = None
    if solid.isValid() and _closed(solid):
        try:
            solid, roundtrip = _canonicalize_step_geometry(
                solid, layouts, save_p / "step5_cad.step" if save_p is not None else None)
            stage_reports["step5"]["step_roundtrip"] = roundtrip
        except Exception as exc:
            stage_reports["step5"]["step_roundtrip"] = {"valid": False, "error": str(exc)}
            _write_stage_reports(save_p, stage_reports)
            raise
    elif save_p is not None:
        # Keep incomplete stage-5 surfaces available for diagnosis and animation.
        export_cad_model(solid, layouts, save_p / "step5_cad.step", export_format="STEP")

    final_workplane = cq.Workplane().newObject([solid])
    final_is_solid = _closed(solid)
    report = {
        "method": "AMRTO pipeline (Ren et al., 2025)",
        "step1_triangles": len(faces),
        "step2_quads": len(quad_faces),
        "step3_layouts": len(layouts),
        "step4_parameterization": cfg.parameterization_method,
        "step5_nurbs_faces": step5_rep.get("nurbs_patch_count", len(layouts)),
        "cad_volume": _solid_volume(solid, relative_tolerance=1e-4) if final_is_solid else None,
        "volume_relative_integration_tolerance": 1e-4,
        "cad_face_count": len(solid.Faces()),
        "cad_surface_types": {kind: sum(face.geomType() == kind for face in solid.Faces())
                               for kind in sorted({face.geomType() for face in solid.Faces()})},
        "valid": bool(solid.isValid()), "is_solid": final_is_solid,
        "requested_strategy": "AMRTO bicubic NURBS",
        "preserve_patch_boundaries": bool(cfg.preserve_patch_boundaries),
        "qa_geometry_source": "Verified exported STEP reimport" if roundtrip else "Incomplete fitted B-rep",
        "surface_orientation_quality": orientation_quality,
        "stages": stage_reports,
    }
    if roundtrip is not None:
        report["step_roundtrip"] = roundtrip
    source_volume = stage_reports["step1"].get("volume")
    if source_volume and report["is_solid"] and report["cad_volume"] is not None:
        report["relative_volume_delta"] = abs(abs(float(report["cad_volume"])) - source_volume) / source_volume
    stage_reports["step5"].update(
        valid=report["valid"], is_valid=report["valid"], is_solid=final_is_solid,
        cad_volume=report["cad_volume"], cad_face_count=report["cad_face_count"],
        cad_surface_types=report["cad_surface_types"], shell_count=len(solid.Shells()),
        solid_count=len(solid.Solids()), cavity_shell_count=max(0, len(solid.Shells()) - len(solid.Solids())),
        relative_volume_delta=report.get("relative_volume_delta"), qa_geometry_source=report["qa_geometry_source"])
    if report["valid"] and report["is_solid"]:
        try:
            deviation = assess_sampled_surface_deviation(
                vertices, faces, solid,
                _effective_tolerance(vertices, fit_tolerance, relative_fit_tolerance),
            )
            deviation["requested_maximum_relative_deviation"] = maximum_relative_deviation
            if feature_tolerance is not None:
                # Remeshing and spline approximation each receive one fit
                # budget. Bound their cumulative source-edge error explicitly.
                deviation["maximum_allowed_vertex_and_feature_distance"] = 2 * feature_tolerance
            report["sampled_surface_deviation"] = deviation
            stage_reports["step5"]["sampled_surface_deviation"] = deviation
        except Exception as exc:
            logger.warning("Surface deviation assessment warning: %s", exc)
    if save_p is not None:
        animation_mesh_report = _export_cad_animation_mesh(
            save_p, "step5_cad", solid,
            _effective_tolerance(vertices, fit_tolerance, relative_fit_tolerance),
        )
        stage_reports["step5"]["animation_mesh"] = animation_mesh_report
        report["animation_mesh"] = animation_mesh_report
    finish_stage("step5")
    _write_stage_reports(save_p, stage_reports)
    if passive_error is not None:
        raise RuntimeError(f"AMRTO passive feature reconstruction failed: {passive_error}")
    if validate_brep:
        _assert_fitting_residual_quality(stage_reports["step5"]["fitting_quality"])
        _assert_fitting_orientation_quality(orientation_quality)
        _assert_valid_occ_shape(solid, label="AMRTO reconstructed CAD body")
        if not report["valid"] or not report["is_solid"]:
            raise RuntimeError(
                "AMRTO stage 5 did not produce a valid closed solid; "
                f"{step5_rep.get('free_edge_count', 'unknown')} free edge(s). "
                "The fitted surface checkpoint was saved."
            )
        if (maximum_volume_delta is not None and report.get("relative_volume_delta", 0.0)
                > float(maximum_volume_delta)):
            raise RuntimeError(
                f"AMRTO CAD changed the stage 1 enclosed volume by "
                f"{report['relative_volume_delta']:.1%}, exceeding {maximum_volume_delta:.1%}. "
                "The CAD checkpoint was saved."
            )
        if maximum_relative_deviation is not None and "sampled_surface_deviation" not in report:
            raise RuntimeError("AMRTO surface deviation could not be verified. The CAD checkpoint was saved.")
        if (maximum_relative_deviation is not None and
                report.get("sampled_surface_deviation", {}).get("maximum_relative_distance", 0.0) > float(maximum_relative_deviation)):
            raise RuntimeError(
                f"AMRTO sampled surface deviation "
                f"{report['sampled_surface_deviation']['maximum_relative_distance']:.2%} exceeds "
                f"{maximum_relative_deviation:.1%} of the stage 1 bounding-box diagonal. "
                "The CAD checkpoint was saved."
            )
        if (feature_tolerance is not None and "sampled_surface_deviation" in report
                and report["sampled_surface_deviation"]["maximum_vertex_and_feature_distance"] > 2 * feature_tolerance):
            raise RuntimeError(
                f"AMRTO CAD vertex/crease deviation "
                f"{report['sampled_surface_deviation']['maximum_vertex_and_feature_distance']:.6g} exceeds "
                f"the cumulative edge fitting tolerance {2 * feature_tolerance:.6g}. The CAD checkpoint was saved.")

    report['_surface_layout'] = display_layout
    if return_report:
        try:
            from .surface_quality import assess_surface_quality
            report["surface_quality"] = assess_surface_quality(solid)
        except Exception:
            pass
        return final_workplane, report

    return final_workplane


__all__ = [
    "SmoothCadUnavailable",
    "build_topopt_beam_lattice",
    "reconstruct_topopt_cad",
    "export_cad_model",
]
