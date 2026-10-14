# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Step 4: Mesh parameterization and adaptive sampling.

Reference:
Section 2.6: Mesh parameterization and adaptive sampling (step 4)
Section 2.6.1: Hybrid mesh parameterization strategy & Algorithm 1 (Discrete harmonic mapping)
Section 2.6.2: Adaptive sampling & Eq. (14), (15)
Section 3.2: Parameter domain initialization method using geodesic distance
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from pylcss.process_utils import headless_subprocess_kwargs

from .amrto_common import QuadLayout, _WORKSPACE_ROOT

logger = logging.getLogger(__name__)


class ScaffoldMapExecutionError(RuntimeError):
    """Native process failure with an honest record of what actually ran."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the signed area of a 2D planar triangle."""
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_in_triangle_2d(
    pt: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-5,
) -> Tuple[bool, float, float, float]:
    """Check if 2D point is inside triangle (a, b, c) and return barycentric coordinates."""
    # Barycentric coordinates via areas
    area_total = triangle_area_2d(a, b, c)
    if abs(area_total) < 1e-12:
        return False, 0.0, 0.0, 0.0

    area_a = triangle_area_2d(pt, b, c)
    area_b = triangle_area_2d(a, pt, c)
    area_c = triangle_area_2d(a, b, pt)

    # Check signs match total area
    sign = 1.0 if area_total > 0 else -1.0
    u = (area_a * sign) / (abs(area_total))
    v = (area_b * sign) / (abs(area_total))
    w = (area_c * sign) / (abs(area_total))

    if u >= -tol and v >= -tol and w >= -tol:
        return True, float(u), float(v), float(w)
    return False, 0.0, 0.0, 0.0


def parameterize_layout_boundaries(layout: QuadLayout) -> Dict[int, np.ndarray]:
    """Map boundary vertices of the layout onto the perimeter of [0, 1]^2 (Algorithm 1)."""
    verts = layout.vertices
    paths = layout.boundary_paths
    if len(paths) != 4:
        raise ValueError(f"Layout {layout.layout_id} must have 4 boundary paths, found {len(paths)}")

    bnd_params: Dict[int, np.ndarray] = {}
    if any(len(path) < 2 for path in paths) or len({path[0] for path in paths}) != 4:
        raise ValueError("A layout requires four nondegenerate boundary paths with distinct corners")
    if any(paths[i][-1] != paths[(i + 1) % 4][0] for i in range(4)):
        raise ValueError("Boundary paths do not form one closed loop")

    def path_chord_lengths(path: list[int]) -> list[float]:
        dists = [0.0]
        for i in range(len(path) - 1):
            d = float(np.linalg.norm(verts[path[i + 1]] - verts[path[i]]))
            if not np.isfinite(d) or d <= 0:
                raise ValueError("A layout boundary contains a zero-length or nonfinite edge")
            dists.append(dists[-1] + d)
        return dists

    # Path 0: (0, 0) -> (1, 0)
    p0 = paths[0]
    d0 = path_chord_lengths(p0)
    L0 = d0[-1]
    for idx, v in enumerate(p0):
        t = d0[idx] / L0
        bnd_params[v] = np.array([t, 0.0])

    # Path 1: (1, 0) -> (1, 1)
    p1 = paths[1]
    d1 = path_chord_lengths(p1)
    L1 = d1[-1]
    for idx, v in enumerate(p1):
        t = d1[idx] / L1
        bnd_params[v] = np.array([1.0, t])

    # Path 2: (1, 1) -> (0, 1)
    p2 = paths[2]
    d2 = path_chord_lengths(p2)
    L2 = d2[-1]
    for idx, v in enumerate(p2):
        t = 1.0 - (d2[idx] / L2)
        bnd_params[v] = np.array([t, 1.0])

    # Path 3: (0, 1) -> (0, 0)
    p3 = paths[3]
    d3 = path_chord_lengths(p3)
    L3 = d3[-1]
    for idx, v in enumerate(p3):
        t = 1.0 - (d3[idx] / L3)
        bnd_params[v] = np.array([0.0, t])

    return bnd_params


def compute_floater_mean_value_weights(
    verts: np.ndarray,
    triangles: np.ndarray,
) -> Dict[int, Dict[int, float]]:
    """Floater weights from actual incident triangle angles, independent of indices.

    Summing half-angle contributions from the two triangles incident to each
    interior edge gives the cyclic one-ring formula without guessing its order.
    """
    verts = np.asarray(verts, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    vertex_count = len(verts)
    if vertex_count == 0:
        return {}
    rows, columns, values = [], [], []
    for offset in range(3):
        i = triangles[:, offset]
        j = triangles[:, (offset + 1) % 3]
        k = triangles[:, (offset + 2) % 3]
        first = verts[j] - verts[i]
        second = verts[k] - verts[i]
        length_first = np.linalg.norm(first, axis=1)
        length_second = np.linalg.norm(second, axis=1)
        if np.any(np.minimum(length_first, length_second) <= 0):
            raise ValueError("Mean-value coordinates require nondegenerate triangle edges")
        denominator = length_first * length_second
        cosine = np.clip(np.einsum("ij,ij->i", first, second) / denominator, -1.0, 1.0)
        sine = np.linalg.norm(np.cross(first, second), axis=1) / denominator
        if np.any(sine <= np.finfo(float).eps):
            raise ValueError("Mean-value coordinates require nondegenerate triangles")
        # tan(angle/2) expressed stably for either acute or obtuse angles.
        tangent = np.empty_like(sine)
        acute = cosine >= 0
        tangent[acute] = sine[acute] / (1 + cosine[acute])
        tangent[~acute] = (1 - cosine[~acute]) / sine[~acute]
        rows.extend((i, i))
        columns.extend((j, k))
        values.extend((tangent / length_first, tangent / length_second))

    rows = np.concatenate(rows)
    columns = np.concatenate(columns)
    values = np.concatenate(values)
    encoded = rows * vertex_count + columns
    unique, inverse = np.unique(encoded, return_inverse=True)
    accumulated = np.bincount(inverse, weights=values)
    unique_rows = unique // vertex_count
    unique_columns = unique % vertex_count
    totals = np.bincount(unique_rows, weights=accumulated, minlength=vertex_count)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        bad_vertex = int(np.flatnonzero(~np.isfinite(totals) | (totals <= 0))[0])
        raise ValueError(f"Vertex {bad_vertex} has no valid incident triangles")
    normalized = accumulated / totals[unique_rows]
    weights: Dict[int, Dict[int, float]] = {i: {} for i in range(vertex_count)}
    for row, column, weight in zip(unique_rows, unique_columns, normalized):
        weights[int(row)][int(column)] = float(weight)
    return weights


def discrete_harmonic_mapping(layout: QuadLayout, *, validate_area_compression: bool = True,
                              minimum_area_ratio: float = 1e-3) -> np.ndarray:
    """Solve the paper's mean-value equations directly on the unit square.

    This replaces Algorithm 1's iteration and therefore has no initialization
    step. ``hybrid_parameterization`` implements the separate ScaffoldMap choice.
    """
    verts = layout.vertices
    tris = layout.triangles
    n_v = len(verts)

    bnd_params = parameterize_layout_boundaries(layout)
    bnd_indices = set(bnd_params.keys())
    int_indices = [i for i in range(n_v) if i not in bnd_indices]

    param_coords = np.zeros((n_v, 2), dtype=float)
    for v_idx, pt in bnd_params.items():
        param_coords[v_idx] = pt

    if not int_indices:
        _validate_parameter_triangles(layout, param_coords, check_compression=validate_area_compression,
                                      minimum_area_ratio=minimum_area_ratio)
        layout.parameter_coords = param_coords
        layout.parameterization_report = {
            "method": "Direct sparse mean-value harmonic solve", "residual": 0.0,
            "initialization": "None; direct solution of Eq. (13)", "interior_vertex_count": 0,
        }
        return param_coords

    # Compute Floater mean-value weights (Eq. 12)
    weights = compute_floater_mean_value_weights(verts, tris)

    # Set up sparse linear system: L * V_int = B
    int_map = {old: new for new, old in enumerate(int_indices)}
    n_int = len(int_indices)

    rows, cols, data = [], [], []
    rhs_u = np.zeros(n_int, dtype=float)
    rhs_v = np.zeros(n_int, dtype=float)

    for old_i in int_indices:
        new_i = int_map[old_i]
        rows.append(new_i)
        cols.append(new_i)
        data.append(1.0)  # Diagonal

        w_nbrs = weights.get(old_i, {})
        for nbr, w in w_nbrs.items():
            if nbr in int_map:
                new_j = int_map[nbr]
                rows.append(new_i)
                cols.append(new_j)
                data.append(-w)
            elif nbr in bnd_params:
                rhs_u[new_i] += w * bnd_params[nbr][0]
                rhs_v[new_i] += w * bnd_params[nbr][1]

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_int, n_int))

    # The normalized mean-value matrix is generally nonsymmetric; CG is not valid.
    with warnings.catch_warnings():
        warnings.simplefilter("error", sla.MatrixRankWarning)
        solution = sla.spsolve(A, np.column_stack((rhs_u, rhs_v)))
    if not np.isfinite(solution).all():
        raise ValueError(f"Layout {layout.layout_id} has a singular harmonic mapping")
    residual = float(np.max(np.abs(A @ solution - np.column_stack((rhs_u, rhs_v)))))
    if residual > 1e-8 or solution.min() < -1e-8 or solution.max() > 1 + 1e-8:
        raise ValueError(f"Layout {layout.layout_id} failed the harmonic mapping residual/bounds check")
    param_coords[int_indices] = np.clip(solution, 0, 1)
    _validate_parameter_triangles(layout, param_coords, check_compression=validate_area_compression,
                                  minimum_area_ratio=minimum_area_ratio)
    layout.parameter_coords = param_coords
    layout.parameterization_report = {
        "method": "Direct sparse mean-value harmonic solve", "residual": residual,
        "initialization": "None; direct solution of Eq. (13)", "interior_vertex_count": len(int_indices),
    }
    return param_coords


def _validate_parameter_triangles(layout: QuadLayout, coordinates: np.ndarray, *, check_compression: bool = True,
                                  minimum_area_ratio: float = 1e-3) -> None:
    if not np.isfinite(minimum_area_ratio) or minimum_area_ratio < 0:
        raise ValueError("Minimum parameter area ratio must be finite and nonnegative")
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(layout.vertices), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Parameter coordinates must be finite and cover every source vertex")
    points = coordinates[layout.triangles]
    areas = ((points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
             - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0]))
    tol = 1e-12
    if np.any(np.abs(areas) <= 1e-14) or (np.any(areas > tol) and np.any(areas < -tol)):
        raise ValueError(f"Layout {layout.layout_id} has collapsed or flipped parameter triangles")
    physical = layout.vertices[layout.triangles]
    physical_areas = np.linalg.norm(np.cross(physical[:, 1] - physical[:, 0],
                                            physical[:, 2] - physical[:, 0]), axis=1)
    if not np.isfinite(physical_areas).all() or np.any(physical_areas <= 0):
        raise ValueError(f"Layout {layout.layout_id} has degenerate source triangles")
    relative_area = np.abs(areas) / (physical_areas / physical_areas.sum())
    # This optional workload heuristic is not a paper validity criterion.
    # Production fits independently check their full source correspondence.
    if check_compression and relative_area.min() < minimum_area_ratio:
        raise ValueError(f"Layout {layout.layout_id} has excessive UV area compression "
                         f"({relative_area.min():.3g} of its physical area fraction)")
    layout.parameter_area_ratio_minimum = float(relative_area.min())


def get_scaffold_map_executable() -> Path | None:
    """Find the noninteractive native ScaffoldMap adapter."""
    from .amrto_common import resolve_tool as resolve_executable

    result = resolve_executable(
        env_vars=("PYLCSS_AMRTO_SCAFFOLD_MAP",),
        candidates=[
            _WORKSPACE_ROOT / "external_solvers" / "amrto" / "scaffold_map_headless.exe",
            "scaffold_map_headless.exe", "scaffold_map_headless",
        ],
    )
    return Path(result) if result else None


def _validate_scaffold_disk(layout: QuadLayout) -> None:
    """Reject holes, disconnected charts and inconsistent boundary declarations."""
    vertices, triangles = np.asarray(layout.vertices), np.asarray(layout.triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2
            or triangles.shape[1] != 3 or len(triangles) == 0 or not np.isfinite(vertices).all()
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("ScaffoldMap requires finite vertices and indexed triangles")
    if len(np.unique(triangles)) != len(vertices):
        raise ValueError("ScaffoldMap disk has unused vertices")
    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    counts = Counter(tuple(sorted(map(int, edge))) for edge in directed)
    if any(count > 2 for count in counts.values()) or any(a == b for a, b in counts):
        raise ValueError("ScaffoldMap input is nonmanifold")
    if len(vertices) - len(counts) + len(triangles) != 1:
        raise ValueError("ScaffoldMap requires one topological disk")
    directed_counts = Counter(map(tuple, directed.tolist()))
    if any(counts[tuple(sorted(edge))] == 2 and directed_counts[edge] != 1 for edge in directed_counts):
        raise ValueError("ScaffoldMap disk has inconsistent triangle orientation")
    parameterize_layout_boundaries(layout)  # Checks the ordered four-side cycle.
    declared = [(int(a), int(b)) for path in layout.boundary_paths for a, b in zip(path[:-1], path[1:])]
    loop_vertices = [int(vertex) for path in layout.boundary_paths for vertex in path[:-1]]
    if len(loop_vertices) != len(set(loop_vertices)):
        raise ValueError("ScaffoldMap disk boundary visits a vertex more than once")
    boundary = {edge for edge, count in counts.items() if count == 1}
    declared_unordered = [tuple(sorted(edge)) for edge in declared]
    if len(declared_unordered) != len(set(declared_unordered)) or set(declared_unordered) != boundary:
        raise ValueError("ScaffoldMap four sides do not cover the disk boundary exactly once")
    neighbors = defaultdict(set)
    for a, b in counts:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen, queue = {0}, [0]
    while queue:
        for neighbor in neighbors[queue.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(seen) != len(vertices):
        raise ValueError("ScaffoldMap input contains disconnected charts")


def native_scaffold_mapping(
    layout: QuadLayout, *, executable: Path | str | None = None,
    maximum_iterations: int = 50, relative_energy_tolerance: float = 1e-4,
    timeout_s: float = 60.0,
) -> tuple[np.ndarray, dict]:
    """Run the actual native optimizer with a free original mesh boundary.

    Mean-value square coordinates seed the optimizer, scaled to the source area.
    No source vertex remains constrained. This seed is an explicit implementation
    choice, not the paper's geodesic initialization for its harmonic fallback.
    """
    if (maximum_iterations < 1 or maximum_iterations > 1000
            or not np.isfinite(relative_energy_tolerance) or relative_energy_tolerance <= 0
            or not np.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("ScaffoldMap requires positive bounded iterations, energy tolerance and timeout")
    _validate_scaffold_disk(layout)
    executable = Path(executable).resolve() if executable is not None else get_scaffold_map_executable()
    if executable is None or not executable.is_file():
        raise FileNotFoundError("Native ScaffoldMap adapter is unavailable")
    seed_layout = replace(layout, parameter_coords=None)
    initial = discrete_harmonic_mapping(seed_layout, validate_area_compression=False).copy()
    seed_report = dict(seed_layout.parameterization_report)
    points = layout.vertices[layout.triangles]
    double_area = np.linalg.norm(np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]), axis=1)
    area = float(double_area.sum() * 0.5)
    if not np.isfinite(area) or area <= 0 or np.any(double_area <= 0):
        raise ValueError("ScaffoldMap requires positive nondegenerate source triangle area")
    initial = (initial - 0.5) * np.sqrt(area)
    triangles = np.asarray(layout.triangles, dtype=np.int64)
    uv_points = initial[triangles]
    signed_area = ((uv_points[:, 1, 0]-uv_points[:, 0, 0]) * (uv_points[:, 2, 1]-uv_points[:, 0, 1])
                   - (uv_points[:, 1, 1]-uv_points[:, 0, 1]) * (uv_points[:, 2, 0]-uv_points[:, 0, 0]))
    reverse = bool(np.all(signed_area < 0))
    if reverse:
        triangles = triangles[:, [0, 2, 1]]
    with tempfile.TemporaryDirectory(prefix="pylcss_scaffold_") as folder:
        input_path, output_path = Path(folder)/"disk.txt", Path(folder)/"scaffold.json"
        with input_path.open("w", encoding="ascii") as stream:
            stream.write(f"{len(layout.vertices)} {len(triangles)}\n")
            np.savetxt(stream, np.column_stack((layout.vertices, initial)), fmt="%.17g")
            np.savetxt(stream, triangles, fmt="%d")
        try:
            subprocess.run(
                [str(executable), str(input_path), str(output_path), str(maximum_iterations),
                 str(relative_energy_tolerance)],
                cwd=folder, check=True, capture_output=True, text=True, timeout=timeout_s,
                **headless_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as error:
            stderr = getattr(error, "stderr", None)
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise ScaffoldMapExecutionError(str(error), {
                "executed": None if isinstance(error, subprocess.SubprocessError) else False,
                "process_started": isinstance(error, subprocess.SubprocessError),
                "process_completed": False, "native_solver_completed": False,
                "executable": str(executable), "native_error": stderr[-2000:] if stderr else None,
                "timeout_s": timeout_s,
            }) from error
        try:
            with output_path.open(encoding="utf-8") as stream:
                report = json.load(stream)
        except (OSError, ValueError) as error:
            raise ScaffoldMapExecutionError(f"Native output unavailable: {error}", {
                "executed": None, "process_started": True, "process_completed": True,
                "native_solver_completed": False, "executable": str(executable),
            }) from error
    coordinates = np.asarray(report.pop("uv"), dtype=float)
    energies = np.asarray(report.get("mesh_energy_history", []), dtype=float)
    if (coordinates.shape != (len(layout.vertices), 2) or not np.isfinite(coordinates).all()
            or report.get("vertex_count") != len(layout.vertices)
            or report.get("face_count") != len(layout.triangles)
            or report.get("source_vertex_constraints") != 0
            or report.get("iterations", 0) < 1
            or report.get("backend") != "Native igl::scaf_precompute/scaf_solve"
            or energies.shape != (report.get("iterations", 0)+1,) or not np.isfinite(energies).all()):
        raise ValueError("Native ScaffoldMap output has invalid coordinates, coverage or constraints")
    # Uniform scaling is irrelevant to positivity; this check also records area
    # distribution but leaves compression acceptance to the unit-domain decision.
    span = float(np.ptp(coordinates, axis=0).max())
    if span <= 0:
        raise ValueError("Native ScaffoldMap returned an empty parameter domain")
    _validate_parameter_triangles(layout, (coordinates-coordinates.min(axis=0))/span, check_compression=False)
    report.update({
        "executable": str(executable), "initialization": "Area-scaled direct mean-value harmonic seed",
        "initialization_report": seed_report, "native_triangle_orientation_reversed": reverse,
        "source_area": area, "original_boundary_free": True,
        "process_started": True, "process_completed": True, "native_solver_completed": True,
    })
    return coordinates, report


def normalize_rectangular_scaffold_domain(
    layout: QuadLayout, coordinates: np.ndarray, *, tolerance: float = 1e-6,
) -> tuple[np.ndarray | None, dict]:
    """Retain only four straight, ordered sides forming an orthogonal rectangle.

    Linear translation, rotation and axis scaling preserve the native map.
    No bent boundary is projected onto the square and no shear is accepted.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Rectangle tolerance must be positive")
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(layout.vertices), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid ScaffoldMap coordinates")
    parameterize_layout_boundaries(layout)
    corners = coordinates[[path[0] for path in layout.boundary_paths]]
    basis = np.column_stack((corners[1]-corners[0], corners[3]-corners[0]))
    extent = float(np.ptp(coordinates, axis=0).max())
    rectangle = {"is_rectangle": False, "tolerance": tolerance}
    if extent <= 0 or abs(np.linalg.det(basis)) <= extent**2 * 1e-14:
        return None, {**rectangle, "reason": "Collapsed corner rectangle"}
    orthogonality = float(abs(np.dot(basis[:, 0], basis[:, 1]))
                          / np.prod(np.linalg.norm(basis, axis=0)))
    if orthogonality > tolerance:
        return None, {**rectangle, "reason": "Adjacent corner sides are not orthogonal",
                      "orthogonality_error": orthogonality}
    normalized = (coordinates-corners[0]) @ np.linalg.inv(basis).T
    expected_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    corner_error = float(np.max(np.abs(normalized[[p[0] for p in layout.boundary_paths]]-expected_corners)))
    straight_error = 0.0
    for side, path in enumerate(layout.boundary_paths):
        fixed_axis = 1 if side % 2 == 0 else 0
        target = 0.0 if side in (0, 3) else 1.0
        side_points = normalized[path]
        straight_error = max(straight_error, float(np.max(np.abs(side_points[:, fixed_axis]-target))))
        progress = side_points[:, 1-fixed_axis]
        if side >= 2:
            progress = -progress
        if np.any(np.diff(progress) <= 0):
            return None, {**rectangle, "reason": "A rectangular side backtracks or collapses"}
    report = {**rectangle, "corner_error": corner_error, "maximum_side_straightness_error": straight_error,
              "orthogonality_error": orthogonality,
              "normalization": "Linear corner translation, rotation and axis scaling"}
    if max(corner_error, straight_error) > tolerance or normalized.min() < -tolerance or normalized.max() > 1+tolerance:
        return None, {**report, "reason": "Native boundary is not a rectangle"}
    normalized = np.clip(normalized, 0, 1)
    # The rectangle decision is geometric. Small positive triangle areas remain
    # a valid map; production fitting must independently verify source accuracy.
    # Rejecting a compression heuristic here would mislabel a completed native
    # optimizer as a runtime failure and change the paper's branch decision.
    _validate_parameter_triangles(layout, normalized, check_compression=False)
    return normalized, {**report, "is_rectangle": True,
                        "minimum_relative_parameter_triangle_area": layout.parameter_area_ratio_minimum,
                        "area_compression_admission_guard_applied": False}


def hybrid_parameterization(
    layout: QuadLayout, *, scaffold_executable: Path | str | None = None,
    maximum_iterations: int = 50, relative_energy_tolerance: float = 1e-4,
    rectangle_tolerance: float = 1e-6, timeout_s: float = 60.0,
    harmonic_mapper: Callable[[QuadLayout], np.ndarray] | None = None,
    harmonic_method: str = "iterative_geodesic", harmonic_max_iter: int = 1000,
    harmonic_energy_threshold: float = 1e-10, harmonic_residual_threshold: float = 1e-8,
    harmonic_edge_counts: bool = False, require_native: bool = False,
    minimum_area_ratio: float = 1e-3,
) -> np.ndarray:
    """AMRTO Section 2.6.1: native rectangle, else unit-square harmonic map.

    If native execution is unavailable or fails, its actual failure is recorded;
    ``require_native=True`` raises instead of accepting that runtime fallback.
    A successfully optimized nonrectangle always takes the paper's harmonic branch.
    """
    if harmonic_method not in {"iterative_geodesic", "iterative_constant", "direct"}:
        raise ValueError("Harmonic method must be iterative_geodesic, iterative_constant or direct")
    native_report: dict = {"executed": False, "process_started": False, "native_solver_completed": False}
    try:
        coordinates, report = native_scaffold_mapping(
            layout, executable=scaffold_executable, maximum_iterations=maximum_iterations,
            relative_energy_tolerance=relative_energy_tolerance, timeout_s=timeout_s,
        )
        native_report = {**report, "executed": True}
        normalized, rectangle_report = normalize_rectangular_scaffold_domain(
            layout, coordinates, tolerance=rectangle_tolerance,
        )
        native_report["rectangle"] = rectangle_report
        if normalized is not None:
            layout.parameter_coords = normalized
            layout.parameterization_report = {
                "method": "AMRTO hybrid: rectangular native ScaffoldMap retained",
                "branch": "scaffold_rectangle", "paper_section": "2.6.1",
                "scaffold": native_report,
            }
            return normalized
    except (OSError, ValueError, subprocess.SubprocessError, ScaffoldMapExecutionError, KeyError, TypeError) as error:
        if isinstance(error, ScaffoldMapExecutionError):
            native_report.update(error.report)
        native_report["failure"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            native_report["native_error"] = error.stderr[-2000:]
        if require_native:
            layout.parameterization_report = {"method": "AMRTO hybrid failed", "scaffold": native_report}
            raise
    try:
        if harmonic_mapper is not None:
            coordinates = harmonic_mapper(layout)
            harmonic_report = dict(getattr(layout, "parameterization_report", {}))
        elif harmonic_method == "direct":
            coordinates = discrete_harmonic_mapping(layout, minimum_area_ratio=minimum_area_ratio)
            harmonic_report = dict(layout.parameterization_report)
        else:
            from .amrto_harmonic import solve_iterative_harmonic_mapping

            coordinates, harmonic_report = solve_iterative_harmonic_mapping(
                layout, initialization="geodesic" if harmonic_method == "iterative_geodesic" else "constant",
                edge_counts=harmonic_edge_counts, max_iter=harmonic_max_iter,
                energy_threshold=harmonic_energy_threshold, residual_threshold=harmonic_residual_threshold,
                minimum_area_ratio=minimum_area_ratio,
            )
    except (ValueError, RuntimeError) as error:
        layout.parameter_coords = None
        layout.parameterization_report = {
            "method": "AMRTO hybrid failed", "branch": "harmonic_failed", "scaffold": native_report,
            "harmonic": {"method": harmonic_method, "failure": f"{type(error).__name__}: {error}"},
        }
        raise
    layout.parameterization_report = {
        "method": "AMRTO hybrid: mean-value harmonic unit square",
        "branch": "harmonic_nonrectangle" if "failure" not in native_report else "harmonic_native_failure",
        "paper_section": "2.6.1", "scaffold": native_report, "harmonic": harmonic_report,
    }
    if harmonic_report.get("converged") is False:
        layout.parameter_coords = None
        raise ValueError(f"Layout {layout.layout_id} harmonic iteration did not converge within "
                         f"{harmonic_report.get('maximum_iterations')} iterations")
    layout.parameter_coords = coordinates
    return coordinates


def compute_boundary_edge_lengths(layout: QuadLayout) -> Tuple[float, float, float, float]:
    """Calculate the lengths of the 4 boundary edges d1, d2, d3, d4 (Section 2.6.2)."""
    verts = layout.vertices
    paths = layout.boundary_paths
    lengths = []
    for p in paths[:4]:
        length = 0.0
        for i in range(len(p) - 1):
            length += float(np.linalg.norm(verts[p[i + 1]] - verts[p[i]]))
        lengths.append(max(1e-3, length))
    while len(lengths) < 4:
        lengths.append(1.0)
    return lengths[0], lengths[1], lengths[2], lengths[3]


def adaptive_sample_layout(
    layout: QuadLayout,
    k_sample: float = 1.5,
    d_min: int = 6,
    d_max: int = 2000,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Adaptive uniform sampling on the parameter domain (Eq. 14, 15).

    Returns:
        sampled_grid_3d: (n_u * n_v, 3) 3D coordinate array
        (n_u, n_v): grid dimensions
    """
    if layout.parameter_coords is None:
        discrete_harmonic_mapping(layout)

    if not np.isfinite(k_sample) or k_sample <= 0 or d_min < 2 or d_max < d_min:
        raise ValueError("Sampling requires k > 0 and 2 <= d_min <= d_max")

    d1, d2, d3, d4 = compute_boundary_edge_lengths(layout)

    # Formula (14):
    # n_u = round(k * (d1 + d3) / 2)
    # n_v = round(k * (d2 + d4) / 2)
    n_u = int(round(k_sample * (d1 + d3) * 0.5))
    n_v = int(round(k_sample * (d2 + d4) * 0.5))

    n_u = max(d_min, min(d_max, n_u))
    n_v = max(d_min, min(d_max, n_v))

    # Generate uniform (n_u x n_v) parameter grid
    grid_u = np.linspace(0.0, 1.0, n_u)
    grid_v = np.linspace(0.0, 1.0, n_v)

    sampled_3d = sample_parameter_grid(layout, grid_u, grid_v)
    layout.sampled_grid_3d = sampled_3d.reshape(-1, 3)
    layout.sampled_shape = (n_u, n_v)
    return layout.sampled_grid_3d, (n_u, n_v)


def sample_parameter_grid(layout: QuadLayout, grid_u: np.ndarray, grid_v: np.ndarray) -> np.ndarray:
    """Inverse barycentric mapping at arbitrary tensor-grid parameters.

    Vectorized per triangle and bounded to its grid footprint. A missing inverse
    mapping is an error, because nearest-vertex substitution conceals bad layouts.
    """
    if layout.parameter_coords is None:
        discrete_harmonic_mapping(layout)
    param_points = layout.parameter_coords
    sampled = np.zeros((len(grid_u), len(grid_v), 3), dtype=float)
    assigned = np.zeros((len(grid_u), len(grid_v)), dtype=bool)
    tolerance = 1e-5
    for triangle in layout.triangles:
        a, b, c = param_points[triangle]
        denominator = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(denominator) <= 1e-14:
            raise ValueError(f"Layout {layout.layout_id} has a collapsed parameter triangle")
        lower = np.minimum(np.minimum(a, b), c) - tolerance
        upper = np.maximum(np.maximum(a, b), c) + tolerance
        u_indices = np.flatnonzero((grid_u >= lower[0]) & (grid_u <= upper[0]))
        v_indices = np.flatnonzero((grid_v >= lower[1]) & (grid_v <= upper[1]))
        if not len(u_indices) or not len(v_indices):
            continue
        uu, vv = np.meshgrid(grid_u[u_indices], grid_v[v_indices], indexing="ij")
        beta = ((uu - a[0]) * (c[1] - a[1]) - (vv - a[1]) * (c[0] - a[0])) / denominator
        gamma = ((b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])) / denominator
        alpha = 1 - beta - gamma
        inside = (alpha >= -tolerance) & (beta >= -tolerance) & (gamma >= -tolerance)
        i, j = np.nonzero(inside)
        i_global, j_global = u_indices[i], v_indices[j]
        barycentric = np.maximum(np.column_stack((alpha[i, j], beta[i, j], gamma[i, j])), 0)
        barycentric /= barycentric.sum(axis=1, keepdims=True)
        sampled[i_global, j_global] = barycentric @ layout.vertices[triangle]
        assigned[i_global, j_global] = True
    if not assigned.all():
        unassigned_i, unassigned_j = np.nonzero(~assigned)
        unassigned_uv = np.column_stack((grid_u[unassigned_i], grid_v[unassigned_j]))
        for idx in range(len(unassigned_i)):
            pt = unassigned_uv[idx]
            best_dist = float("inf")
            best_bary = None
            best_tri = None
            for triangle in layout.triangles:
                a, b, c = param_points[triangle]
                denom = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if abs(denom) <= 1e-14:
                    continue
                b_val = ((pt[0] - a[0]) * (c[1] - a[1]) - (pt[1] - a[1]) * (c[0] - a[0])) / denom
                g_val = ((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])) / denom
                a_val = 1.0 - b_val - g_val
                violation = max(0.0, -a_val) + max(0.0, -b_val) + max(0.0, -g_val)
                if violation < best_dist:
                    best_dist = violation
                    best_tri = triangle
                    best_bary = np.maximum([a_val, b_val, g_val], 0.0)
            if best_dist < 1e-2 and best_tri is not None and best_bary.sum() > 0:
                best_bary /= best_bary.sum()
                sampled[unassigned_i[idx], unassigned_j[idx]] = best_bary @ layout.vertices[best_tri]
                assigned[unassigned_i[idx], unassigned_j[idx]] = True
        missing = int(np.count_nonzero(~assigned))
        if missing > 0:
            raise ValueError(f"Layout {layout.layout_id} inverse mapping misses {missing} sample points")
    return sampled
