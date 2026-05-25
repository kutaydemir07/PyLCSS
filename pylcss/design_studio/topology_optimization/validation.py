# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Phase 4 — validation re-analysis for topology-optimised geometry.

The voxel topology optimiser reports a SIMP-relaxed *proxy* stress field
during the search — useful to steer the optimiser, but not a trustworthy
production analysis.  This node closes the loop:

    optimised density field  ─►  conforming C3D10 (or C3D4) tet mesh
                             ─►  CalculiX `ccx` static solve
                             ─►  FRD → von Mises / displacement payload

By default the mesh uses **quadratic C3D10** tetrahedra: linear C3D4 tets are
markedly over-stiff and smear the very stress gradients this node exists to
check, so they make a "passing" validation optimistic.  An optional
**mesh-convergence study** then re-solves at successively refined meshes (voxel
supersampling keeps the geometry fixed while shrinking the elements) and reports
whether the global response has converged — the difference between a single
number and a trustworthy one.

We mesh the *density field* directly (not the smoothed marching-cubes
surface): every solid voxel becomes 6 tetrahedra via the Kuhn/Freudenthal
decomposition, with corner nodes welded across voxels (and, for C3D10, midside
nodes welded across shared edges).  This is:

  * **robust** — pure NumPy, no external mesher to segfault on the noisy,
    near-degenerate triangles that marching-cubes + smoothing produce;
  * **honest** — it analyses the topology the optimiser actually chose,
    not a smoothed approximation of it;
  * **conforming** — the Kuhn decomposition uses a consistent main diagonal
    so adjacent voxels share matching face triangulations (no cracks).

CalculiX was chosen deliberately as the project's single trusted FEA backend
(see [solver.py](solver.py)); the smoothed STL remains the *manufacturing*
deliverable, while this validates the *engineering* result.

The output dict matches `SolverNode`'s format so it plugs into the existing
FEA viewer unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight scikit-fem-compatible tet mesh wrapper
# ---------------------------------------------------------------------------

class _TetVolumeMesh:
    """Duck-typed scikit-fem mesh for the existing CalculiX deck writer.

    `run_calculix_static` only touches `.p` (3, N_nodes) and `.t` (4, N_elem),
    so a full skfem.MeshTet (and its dependency chain) is unnecessary.
    """

    def __init__(self, points_3d: np.ndarray, tets: np.ndarray):
        self.p = np.ascontiguousarray(np.asarray(points_3d, dtype=float).T)  # (3, N)
        # (4, M) for linear C3D4 or (10, M) for quadratic C3D10.  The deck
        # writer selects the element type from the row count.
        self.t = np.ascontiguousarray(np.asarray(tets, dtype=int).T)


# Kuhn / Freudenthal 6-tet decomposition of a unit cube.  All six tets share
# the main diagonal c000–c111; corners are addressed by (dx, dy, dz) ∈ {0,1}³.
# Applying the same pattern to every voxel yields a globally CONFORMING mesh.
_KUHN_TETS = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
)


# C3D10 midside edges in CalculiX/Abaqus order: nodes 5..10 sit on the
# corner-pairs (1-2), (2-3), (3-1), (1-4), (2-4), (3-4) → 0-indexed below.
_C3D10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _supersample_density(density: np.ndarray, factor: int) -> np.ndarray:
    """Block-replicate a voxel field by an integer factor on every axis.

    Nearest-neighbour replication keeps the thresholded boundary *identical*
    while shrinking the element size — exactly what an h-refinement mesh-
    convergence study needs (geometry fixed, mesh refined).  Trilinear
    upsampling would instead move the boundary between levels and so could not
    isolate discretisation error.
    """
    factor = int(factor)
    if factor <= 1:
        return density
    return np.repeat(np.repeat(np.repeat(density, factor, 0), factor, 1), factor, 2)


def _solid_voxel_count(density: np.ndarray, cutoff: float) -> int:
    """Number of voxels that survive the cutoff (with the same fallback rule)."""
    density = np.asarray(density, dtype=float)
    solid = density >= float(cutoff)
    if not np.any(solid):
        nz = density[density > 0.0]
        if nz.size == 0:
            return 0
        solid = density >= float(np.percentile(nz, 50.0))
    return int(np.count_nonzero(solid))


def _voxel_density_to_tet_mesh(
    density: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    cutoff: float,
    *,
    quadratic: bool = True,
    refine: int = 1,
) -> _TetVolumeMesh:
    """Mesh a thresholded voxel density field into a conforming tet mesh.

    Every voxel with ``density >= cutoff`` becomes 6 tets via the Kuhn
    decomposition; corner nodes are welded across voxels so the mesh is
    watertight and conforming.  ``refine`` block-replicates the field first
    (finer mesh, same geometry).  When ``quadratic`` is set, shared midside
    nodes are added in CalculiX C3D10 order, producing a quadratic mesh whose
    midside nodes are also welded across element edges (so the quadratic mesh
    stays watertight).

    Corner tets are reoriented to a positive signed volume *before* midside
    nodes are generated, so the C3D10 edge ordering is correct by construction
    and the deck writer never has to relabel midsides.
    """
    density = np.asarray(density, dtype=float)
    if density.ndim != 3:
        raise RuntimeError("Validation expects a 3-D voxel density field.")
    density = _supersample_density(density, int(refine))
    nelx, nely, nelz = density.shape

    solid = density >= float(cutoff)
    if not np.any(solid):
        nz = density[density > 0.0]
        if nz.size == 0:
            raise RuntimeError("Density field is empty — nothing to validate.")
        # Fall back to the median of the non-zero densities so we still build
        # *something* rather than erroring on an over-aggressive cutoff.
        solid = density >= float(np.percentile(nz, 50.0))

    mins = np.asarray(bounds[0], dtype=float)
    maxs = np.asarray(bounds[1], dtype=float)
    cell = (maxs - mins) / np.array([nelx, nely, nelz], dtype=float)

    node_index: Dict[Tuple[int, int, int], int] = {}
    points: List[np.ndarray] = []

    def _node(gi: int, gj: int, gk: int) -> int:
        key = (gi, gj, gk)
        idx = node_index.get(key)
        if idx is None:
            idx = len(points)
            node_index[key] = idx
            points.append(mins + np.array([gi, gj, gk], dtype=float) * cell)
        return idx

    corner_tets: List[List[int]] = []
    for (i, j, k) in np.argwhere(solid):
        i, j, k = int(i), int(j), int(k)
        for tet in _KUHN_TETS:
            corner_tets.append([_node(i + dx, j + dy, k + dz) for (dx, dy, dz) in tet])

    if not corner_tets:
        raise RuntimeError("No solid voxels survived the density cutoff.")

    pts = np.asarray(points, dtype=float)
    corners = np.asarray(corner_tets, dtype=int)  # (M, 4)

    # Reorient negative tets (swap corners 0/1) so all signed volumes are
    # positive before midside generation.
    v0, v1, v2, v3 = (pts[corners[:, c]] for c in range(4))
    signed_vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)
    neg = signed_vol < 0.0
    if np.any(neg):
        corners[neg, 0], corners[neg, 1] = corners[neg, 1], corners[neg, 0].copy()

    if not quadratic:
        return _TetVolumeMesh(pts, corners)

    # Welded midside nodes.  Build the (sorted) endpoint key for every tet edge,
    # deduplicate globally, and place one midside node at each unique edge.
    n_elem = corners.shape[0]
    edge_pairs = np.stack([corners[:, [a, b]] for (a, b) in _C3D10_EDGES], axis=0)
    edge_pairs = edge_pairs.reshape(6 * n_elem, 2)
    edge_keys = np.sort(edge_pairs, axis=1)
    unique_edges, inverse = np.unique(edge_keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    base = pts.shape[0]
    mid_coords = 0.5 * (pts[unique_edges[:, 0]] + pts[unique_edges[:, 1]])
    mid_ids = (base + inverse).reshape(6, n_elem).T  # (M, 6) in edge order

    tets10 = np.hstack([corners, mid_ids]).astype(int)
    all_points = np.vstack([pts, mid_coords])
    return _TetVolumeMesh(all_points, tets10)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _extract_density_field(
    payload: Any,
) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]]:
    """Pull (density_3d, (mins, maxs), cutoff) from a topology-opt result dict.

    Returns None if the payload carries no usable voxel density field.  Bounds
    default to one-unit voxels at the origin when the optimiser ran without a
    physical mesh attached.
    """
    if not isinstance(payload, dict):
        return None
    density = payload.get('density')
    if density is None:
        return None
    density = np.asarray(density, dtype=float)
    if density.ndim != 3 or min(density.shape) < 1:
        return None

    bounds_payload = payload.get('bounds')
    if (isinstance(bounds_payload, dict)
            and 'min' in bounds_payload and 'max' in bounds_payload):
        mins = np.asarray(bounds_payload['min'], dtype=float)
        maxs = np.asarray(bounds_payload['max'], dtype=float)
        if mins.size < 3 or maxs.size < 3 or not np.all(maxs[:3] > mins[:3]):
            mins = np.zeros(3)
            maxs = np.asarray(density.shape, dtype=float)
    else:
        mins = np.zeros(3)
        maxs = np.asarray(density.shape, dtype=float)

    cutoff = float(payload.get('density_cutoff', 0.5) or 0.5)
    return density, (mins, maxs), cutoff


def _flatten(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for v in values:
            out.extend(_flatten(v))
        return out
    return [values]


def run_topopt_validation(
    topo_payload: Dict[str, Any],
    material: Dict[str, Any],
    constraints: List[Any],
    loads: List[Any],
    *,
    validation_cutoff: float = 0.0,
    element_order: str = 'Quadratic (C3D10)',
    convergence_levels: int = 1,
    max_validation_elements: int = 500000,
    external_solver_path: str = '',
    external_work_dir: str = '',
    deck_only: bool = False,
    run_external_solver: bool = True,
    external_timeout_s: float = 3600.0,
    analysis_type: str = 'Linear',
    visualization: str = 'Von Mises Stress',
    deformation_scale: str = 'Auto',
) -> Dict[str, Any]:
    """Validate a topology result internally without exposing a graph block."""
    from pylcss.solver_backends import (
        ExternalRunConfig,
        SolverBackendError,
        run_calculix_static,
    )
    from pylcss.solver_backends.common import as_bool

    field = _extract_density_field(topo_payload)
    missing = []
    if field is None:
        missing.append("a topology-opt result with a density field")
    if material is None:
        missing.append("material")
    if not constraints:
        missing.append("at least one constraint")
    if not loads:
        missing.append("at least one load")
    if missing:
        raise RuntimeError("Topology validation requires " + ", ".join(missing) + ".")

    density, bounds, reported_cutoff = field
    cutoff = float(validation_cutoff or 0.0)
    if cutoff <= 0.0:
        cutoff = reported_cutoff

    order_label = str(element_order or '').upper()
    quadratic = ('C3D10' in order_label) or ('QUADRAT' in order_label)
    elem_label = 'C3D10' if quadratic else 'C3D4'
    n_levels = max(1, int(convergence_levels or 1))
    elem_budget = max(1, int(max_validation_elements or 500000))
    run_solver = as_bool(run_external_solver) and not as_bool(deck_only)

    def _solve_at(refine: int):
        mesh = _voxel_density_to_tet_mesh(
            density, bounds, cutoff, quadratic=quadratic, refine=refine,
        )
        config = ExternalRunConfig(
            executable=(external_solver_path or None),
            work_dir=(external_work_dir or None),
            run_solver=run_solver,
            timeout_s=float(external_timeout_s or 3600.0),
            job_name=f"pylcss_topopt_validation_L{refine}",
        )
        out = run_calculix_static(
            mesh,
            material if isinstance(material, dict) else {},
            _flatten(constraints),
            _flatten(loads),
            config,
            visualization_mode=str(visualization or 'Von Mises Stress'),
            analysis_type=str(analysis_type or 'Linear'),
        )
        return mesh, out

    try:
        base_solid = _solid_voxel_count(density, cutoff)
    except Exception:
        base_solid = 0
    factors = [1]
    if run_solver:
        for f in range(2, n_levels + 1):
            if base_solid * 6 * (f ** 3) > elem_budget:
                break
            factors.append(f)
    budget_capped = run_solver and (len(factors) < n_levels)

    levels: List[Dict[str, Any]] = []
    finest_output: Optional[Dict[str, Any]] = None
    try:
        for f in factors:
            mesh, out = _solve_at(f)
            finest_output = out
            peak_vm = out.get('max_stress_gauss')
            if peak_vm is None and out.get('stress') is not None:
                arr = np.asarray(out['stress'], dtype=float)
                peak_vm = float(np.max(arr)) if arr.size else None
            compliance = out.get('compliance')
            levels.append({
                'refine': int(f),
                'n_nodes': int(np.asarray(mesh.p).shape[1]),
                'n_elements': int(np.asarray(mesh.t).shape[1]),
                'peak_von_mises': (float(peak_vm) if peak_vm is not None else None),
                'compliance': (float(compliance) if compliance is not None else None),
                'solved': compliance is not None,
            })
    except RuntimeError:
        if finest_output is None:
            raise
    except SolverBackendError as exc:
        if finest_output is None:
            raise RuntimeError(f"CalculiX validation failed: {exc}") from exc
        levels.append({'refine': None, 'error': str(exc)})

    if finest_output is None:
        raise RuntimeError("Topology validation produced no CalculiX result.")

    output = finest_output
    tol = 0.05
    solved = [L for L in levels if L.get('solved') and L.get('compliance')]
    comp_rel = vm_rel = None
    converged: Optional[bool] = None
    if len(solved) >= 2:
        c_prev, c_last = solved[-2]['compliance'], solved[-1]['compliance']
        if c_last is not None and abs(c_last) > 1e-30:
            comp_rel = abs(c_last - c_prev) / abs(c_last)
            converged = comp_rel <= tol
        v_prev, v_last = solved[-2].get('peak_von_mises'), solved[-1].get('peak_von_mises')
        if v_prev is not None and v_last is not None and abs(v_last) > 1e-30:
            vm_rel = abs(v_last - v_prev) / abs(v_last)

    if converged is None:
        note = "Single mesh level; convergence not assessed."
    elif converged:
        note = f"Compliance changed {comp_rel*100:.1f}% between the two finest meshes."
    else:
        note = f"Compliance changed {comp_rel*100:.1f}% between the two finest meshes; not converged."
    if budget_capped:
        note += " Refinement was capped by max_validation_elements."

    output['element_order'] = elem_label
    output['convergence_study'] = {
        'element_order': elem_label,
        'tolerance': tol,
        'levels': levels,
        'compliance_rel_change': comp_rel,
        'peak_von_mises_rel_change': vm_rel,
        'converged': converged,
        'budget_capped': budget_capped,
        'note': note,
    }
    scale_prop = str(deformation_scale or 'Auto').strip()
    if scale_prop.lower() != 'auto':
        try:
            output['deformation_scale'] = float(scale_prop.lower().rstrip('x'))
        except Exception:
            pass
    output['source'] = 'topology_opt_validation'
    output['validation_cutoff'] = cutoff
    if isinstance(topo_payload, dict) and topo_payload.get('stress_pnorm') is not None:
        output['topopt_proxy_stress_pnorm'] = float(topo_payload['stress_pnorm'])
    return output
