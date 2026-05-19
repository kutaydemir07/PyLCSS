# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""3-D SIMP topology optimisation node on a structured voxel domain.

The UI-facing block is :class:`TopologyOptVoxelNode`.  The solver class below is
an internal backend used by that node and by headless tests/runtime calls, so the
topology optimisation logic is not tied to Qt or the node editor.

Boundary conditions are expressed through individual node properties:

  Per-face support (each of 6 faces:
    None | Fix X | Fix Y | Fix Z | Fix XY | Fix YZ | Fix XZ | Fix XYZ)
    left_support, right_support, top_support, bottom_support,
    front_support, back_support

  Force definition
    force_type      : Point | Distributed Face
    force_face      : Left | Right | Top | Bottom | Front | Back  (Distributed only)
    force_ix_frac   : 0.0–1.0  (fractional X location of point force)
    force_iy_frac   : 0.0–1.0  (fractional Y location of point force)
    force_iz_frac   : 0.0–1.0  (fractional Z location of point force)
    force_dir_x, force_dir_y, force_dir_z, force_magnitude

  Convenience presets (Cantilever Tip | Cantilever Distributed | Bridge | Custom)
  populate the above properties with typical values; selecting 'Custom' keeps
  whatever the user has set manually.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BC data structures
# ---------------------------------------------------------------------------

_SUPPORT_TO_DOFS: Dict[str, List[int]] = {
    'None':    [],
    'Fix X':   [0],
    'Fix Y':   [1],
    'Fix Z':   [2],
    'Fix XY':  [0, 1],
    'Fix YZ':  [1, 2],
    'Fix XZ':  [0, 2],
    'Fix XYZ': [0, 1, 2],
}


@dataclass
class VoxelBC:
    """Fully configurable boundary conditions for a 3-D voxel domain.

    ix/iy/iz coordinates are INTEGER node indices in the domain grid:
        ix : 0 … nelx   (left  → right,  X)
        iy : 0 … nely   (bottom → top,   Y)
        iz : 0 … nelz   (front → back,   Z)
    """
    fixed_left_face_dofs:   List[int] = field(default_factory=list)  # ix=0 face
    fixed_right_face_dofs:  List[int] = field(default_factory=list)  # ix=nelx face
    fixed_top_face_dofs:    List[int] = field(default_factory=list)  # iy=nely face
    fixed_bottom_face_dofs: List[int] = field(default_factory=list)  # iy=0 face
    fixed_front_face_dofs:  List[int] = field(default_factory=list)  # iz=0 face
    fixed_back_face_dofs:   List[int] = field(default_factory=list)  # iz=nelz face
    # Localized support boxes in fractional node coordinates:
    # (x_min, x_max, y_min, y_max, z_min, z_max, dofs)
    fixed_boxes: List[Tuple[float, float, float, float, float, float, List[int]]] = field(default_factory=list)
    # Point forces: list of (ix_frac, iy_frac, iz_frac, fx, fy, fz)
    #   fracs ∈ [0,1] scaled to nelx/nely/nelz
    point_forces: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    # Distributed force on a face: list of (face, fx_per, fy_per, fz_per)
    #   face ∈ {'left','right','top','bottom','front','back'}
    distributed_forces: List[Tuple[str, float, float, float]] = field(default_factory=list)


@dataclass
class TopologyOptVoxelProblem:
    """All parameters needed to solve a 3-D voxel topology optimisation."""
    nelx:     int   = 30
    nely:     int   = 20
    nelz:     int   = 10
    E0:       float = 1.0
    Emin:     float = 1e-9
    nu:       float = 0.3
    penal:    float = 3.0
    volfrac:  float = 0.5
    rmin:     float = 1.5
    unitx:    float = 1.0
    unity:    float = 1.0
    unitz:    float = 1.0
    optimizer: str  = 'OC'   # 'OC' | 'MMA'
    max_iter: int   = 80
    tol:      float = 0.01
    bc:       VoxelBC = field(default_factory=VoxelBC)


# ---------------------------------------------------------------------------
# OC update (Sigmund 2001 / pyMOTO 69-line)
# ---------------------------------------------------------------------------

def _oc_update(x: np.ndarray, dc: np.ndarray, volfrac: float) -> np.ndarray:
    maxvol = volfrac * len(x)
    move   = 0.2
    l1, l2 = 0.0, 1e5
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        be   = np.maximum(-dc / lmid, 0.0)
        xnew = np.clip(
            x * np.sqrt(be),
            np.maximum(1e-3, x - move),
            np.minimum(1.0,  x + move),
        )
        if np.sum(xnew) > maxvol:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TopologyOptVoxelResult:
    density:            np.ndarray
    compliance_history: List[float] = field(default_factory=list)
    change_history:     List[float] = field(default_factory=list)
    n_iter:             int   = 0
    converged:          bool  = False
    message:            str   = ""


def _density_grid_from_state(x: np.ndarray, domain: Any) -> np.ndarray:
    """Map pyMOTO's flat element numbering to density[ix, iy, iz]."""
    return np.asarray(x, dtype=float)[domain.elements]


def _voxel_origin_cell(
    shape: Tuple[int, int, int],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return physical origin and cell size for a structured voxel grid."""
    shape_arr = np.maximum(np.asarray(shape, dtype=float), 1.0)
    if bounds is not None:
        mins, maxs = bounds
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
            return mins[:3], (maxs[:3] - mins[:3]) / shape_arr
    return -0.5 * shape_arr, np.ones(3, dtype=float)


def _taubin_smooth_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 6,
) -> np.ndarray:
    """Light volume-preserving smoothing for marching-cubes output."""
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts

    verts = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=int)
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    for _ in range(max(0, int(iterations))):
        for factor in (0.5, -0.53):
            neighbor_sum = np.zeros_like(verts)
            neighbor_count = np.zeros(len(verts), dtype=float)
            np.add.at(neighbor_sum, edges[:, 0], verts[edges[:, 1]])
            np.add.at(neighbor_count, edges[:, 0], 1.0)
            np.add.at(neighbor_sum, edges[:, 1], verts[edges[:, 0]])
            np.add.at(neighbor_count, edges[:, 1], 1.0)
            mask = neighbor_count > 0
            avg = neighbor_sum[mask] / neighbor_count[mask, None]
            verts[mask] += factor * (avg - verts[mask])
    return verts


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    cutoff: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract a recovered surface from a structured voxel density field."""
    try:
        from skimage import measure
        import scipy.ndimage as ndi
    except ImportError:
        return None

    try:
        grid = np.asarray(density, dtype=float)
        if grid.ndim != 3 or min(grid.shape) < 1:
            return None

        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(grid)) <= 0.0:
            return None

        min_dim = max(1, min(grid.shape))
        upsample = int(np.clip(np.ceil(48.0 / min_dim), 1, 12))
        while upsample > 1 and np.prod(np.asarray(grid.shape) * upsample) > 2_500_000:
            upsample -= 1

        if upsample > 1:
            field = ndi.zoom(grid, zoom=upsample, order=3, mode='nearest')
        else:
            field = grid.copy()

        field = np.clip(field, 0.0, 1.0)
        sigma = 0.35 if upsample <= 1 else 0.65
        field = ndi.gaussian_filter(field, sigma=sigma)
        pad = max(3, min(10, upsample))
        field = np.pad(field, pad_width=pad, mode='constant', constant_values=0.0)

        origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        spacing = cell / float(upsample)
        level = float(np.clip(cutoff, 1e-6, 0.999999))
        mask = field >= level
        if not np.any(mask):
            nonzero = field[field > 0.0]
            if nonzero.size == 0:
                return None
            mask = field >= float(np.percentile(nonzero, 75.0))
        if np.all(mask):
            return None

        outside = ndi.distance_transform_edt(~mask, sampling=tuple(float(v) for v in spacing))
        inside = ndi.distance_transform_edt(mask, sampling=tuple(float(v) for v in spacing))
        signed_distance = outside - inside
        signed_distance = ndi.gaussian_filter(signed_distance, sigma=0.85)
        if not (float(np.min(signed_distance)) < 0.0 < float(np.max(signed_distance))):
            return None

        verts, faces, _, _ = measure.marching_cubes(
            signed_distance,
            level=0.0,
            spacing=tuple(float(v) for v in spacing),
            gradient_direction='descent',
        )
        if len(verts) == 0 or len(faces) == 0:
            return None

        surface_origin = origin + 0.5 * cell - float(pad) * spacing
        verts = verts + surface_origin
        verts = _taubin_smooth_surface(verts, faces, iterations=12)
        return {
            'vertices': np.asarray(verts, dtype=float),
            'faces': np.asarray(faces, dtype=int),
        }
    except Exception:
        logger.exception("Voxel shape recovery failed")
        return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TopologyOptVoxelSolver:
    """3-D SIMP topology optimiser backed by pyMOTO."""

    def __init__(self, problem: TopologyOptVoxelProblem):
        self.problem        = problem
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(
        self,
        callback: Optional[Callable[[int, float, float, np.ndarray], None]] = None,
    ) -> TopologyOptVoxelResult:
        """
        Run the optimisation loop.

        callback(iteration, compliance, change, density_3d) is called after
        every iteration so the UI can update live.  density_3d has shape
        (nelx, nely, nelz).
        """
        import pymoto as pym

        p = self.problem

        # ── domain ────────────────────────────────────────────────────────
        domain = pym.VoxelDomain(
            p.nelx, p.nely, p.nelz,
            unitx=p.unitx,
            unity=p.unity,
            unitz=p.unitz,
        )
        ndof   = domain.nnodes * domain.dim  # 3 DOFs per node

        # ── boundary conditions ───────────────────────────────────────────
        f = np.zeros(ndof)
        boundary_dofs = self._assemble_bcs(domain, p.bc, f)

        # ── pyMOTO network ────────────────────────────────────────────────
        sx = pym.Signal("x", state=np.ones(domain.nel) * p.volfrac)

        with pym.Network() as net:
            sxfilt = pym.DensityFilter(domain=domain, radius=p.rmin)(sx)
            sSIMP  = pym.MathExpression(
                expression=f"{p.Emin} + {p.E0 - p.Emin}*inp0^{p.penal}"
            )(sxfilt)
            sK  = pym.AssembleStiffness(
                domain=domain, bc=boundary_dofs, poisson_ratio=p.nu
            )(sSIMP)
            su  = pym.LinSolve(symmetric=True, positive_definite=True)(sK, f)
            sg0 = pym.EinSum(expression="i,i->")(su, f)   # compliance fᵀ u
            svol = pym.EinSum(expression="i->")(sxfilt)   # volume sum
            sg0.tag = "compliance"
            sg0_scaled = pym.Scaling(scaling=100.0)(sg0)
            sg0_scaled.tag = "objective"
            svol.tag = "volume"
            sg1 = pym.MathExpression(
                expression=f"10*(inp0/{domain.nel} - {p.volfrac})"
            )(svol)
            sg1.tag = "volume constraint"

        net.response()

        # ── iteration loop ────────────────────────────────────────────────
        result = TopologyOptVoxelResult(
            density=_density_grid_from_state(sx.state, domain)
        )
        comp_hist:   List[float] = []
        change_hist: List[float] = []

        if p.optimizer.upper() == 'MMA':
            mma = pym.MMA(
                sx, [sg0_scaled, sg1], net,
                xmin=np.full(domain.nel, 1e-3),
                xmax=np.ones(domain.nel),
                move=0.2,
                verbosity=0,
            )

        it, change = 0, 1.0
        while change > p.tol and it < p.max_iter:
            if self.stop_requested:
                result.message = "Stopped by user"
                break

            it     += 1
            x_old   = sx.state.copy()

            if p.optimizer.upper() == 'OC':
                net.reset()
                sg0.sensitivity = 1.0
                net.sensitivity()
                dc = sx.sensitivity.copy()
                sx.state = _oc_update(sx.state, dc, p.volfrac)
            else:
                x_new, _, _ = mma.step(x=sx.state)
                sx.state = np.asarray(x_new, dtype=float)
                mma.iter += 1

            net.response()

            comp_val = float(sg0.state)
            change = float(np.max(np.abs(sx.state - x_old)))
            comp_hist.append(comp_val)
            change_hist.append(change)
            result.n_iter = it

            density_3d = _density_grid_from_state(sx.state, domain)
            if callback is not None:
                callback(it, comp_val, change, density_3d.copy())

        if not result.message:
            if change <= p.tol:
                result.converged = True
                result.message = f"Converged in {it} iterations (Δx = {change:.2e})"
            else:
                result.message = f"Maximum iterations ({p.max_iter}) reached"

        result.density          = _density_grid_from_state(sx.state, domain)
        result.compliance_history = comp_hist
        result.change_history   = change_hist
        return result

    # ── BC assembly ───────────────────────────────────────────────────────

    def _assemble_bcs(
        self,
        domain: Any,
        bc: VoxelBC,
        f: np.ndarray,
    ) -> np.ndarray:
        """Build global DOF index array and populate force vector from VoxelBC.

        VoxelDomain node grid (3-D):
            domain.nodes[ix, iy, iz]  →  scalar node number
            ix = 0 … nelx  (left  → right,  X)
            iy = 0 … nely  (bottom → top,   Y)
            iz = 0 … nelz  (front → back,   Z)
        DOF layout:  x-DOF = 3*n,  y-DOF = 3*n+1,  z-DOF = 3*n+2
        """
        p   = self.problem
        dim = domain.dim   # 3
        fixed: List[int] = []

        def _add_face_dofs(nodes_2d: np.ndarray, dofs: List[int]) -> None:
            fixed.extend(
                domain.get_dofnumber(nodes_2d.flatten(), dofs, ndof=dim)
                .flatten()
                .astype(int)
                .tolist()
            )

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i + 1)

        # ── face supports ─────────────────────────────────────────────────
        if bc.fixed_left_face_dofs:
            _add_face_dofs(domain.nodes[0,      :,      :], bc.fixed_left_face_dofs)
        if bc.fixed_right_face_dofs:
            _add_face_dofs(domain.nodes[p.nelx, :,      :], bc.fixed_right_face_dofs)
        if bc.fixed_top_face_dofs:
            _add_face_dofs(domain.nodes[:,      p.nely, :], bc.fixed_top_face_dofs)
        if bc.fixed_bottom_face_dofs:
            _add_face_dofs(domain.nodes[:,      0,      :], bc.fixed_bottom_face_dofs)
        if bc.fixed_front_face_dofs:
            _add_face_dofs(domain.nodes[:,      :,      0], bc.fixed_front_face_dofs)
        if bc.fixed_back_face_dofs:
            _add_face_dofs(domain.nodes[:,      :,      p.nelz], bc.fixed_back_face_dofs)

        for xmin, xmax, ymin, ymax, zmin, zmax, dofs in bc.fixed_boxes:
            if not dofs:
                continue
            nodes = domain.nodes[
                _node_slice(xmin, xmax, p.nelx),
                _node_slice(ymin, ymax, p.nely),
                _node_slice(zmin, zmax, p.nelz),
            ]
            _add_face_dofs(nodes, dofs)

        # ── point forces ──────────────────────────────────────────────────
        for (ix_frac, iy_frac, iz_frac, fx, fy, fz) in bc.point_forces:
            ix = int(round(ix_frac * p.nelx))
            iy = int(round(iy_frac * p.nely))
            iz = int(round(iz_frac * p.nelz))
            n  = int(domain.nodes[ix, iy, iz])
            f[domain.get_dofnumber(n, 0, ndof=dim)] += fx
            f[domain.get_dofnumber(n, 1, ndof=dim)] += fy
            f[domain.get_dofnumber(n, 2, ndof=dim)] += fz

        # ── distributed face forces ───────────────────────────────────────
        _face_nodes = {
            'left':   domain.nodes[0,      :,      :].flatten(),
            'right':  domain.nodes[p.nelx, :,      :].flatten(),
            'top':    domain.nodes[:,      p.nely, :].flatten(),
            'bottom': domain.nodes[:,      0,      :].flatten(),
            'front':  domain.nodes[:,      :,      0].flatten(),
            'back':   domain.nodes[:,      :,      p.nelz].flatten(),
        }
        for (face, fx_per, fy_per, fz_per) in bc.distributed_forces:
            nodes   = _face_nodes.get(face, np.array([], dtype=int))
            n_nodes = len(nodes)
            if n_nodes == 0:
                continue
            for n in nodes:
                f[domain.get_dofnumber(int(n), 0, ndof=dim)] += fx_per / n_nodes
                f[domain.get_dofnumber(int(n), 1, ndof=dim)] += fy_per / n_nodes
                f[domain.get_dofnumber(int(n), 2, ndof=dim)] += fz_per / n_nodes

        return np.unique(fixed).astype(int)


# ---------------------------------------------------------------------------
# Helper: build VoxelBC from node-property strings
# ---------------------------------------------------------------------------

def _parse_support(label: str) -> List[int]:
    return _SUPPORT_TO_DOFS.get(label, [])


def _parse_support_region_dofs(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value if int(v) in (0, 1, 2)]

    text = str(value or "").strip()
    if not text:
        return []
    if text in _SUPPORT_TO_DOFS:
        return _SUPPORT_TO_DOFS[text]

    axes = text.upper().replace("FIX", "").replace(" ", "")
    return [idx for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)) if axis in axes]


def _flatten(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _mesh_bounds(mesh: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if mesh is None or not hasattr(mesh, 'p'):
        return None
    pts = np.asarray(mesh.p, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] == 0:
        return None
    return pts[:3].min(axis=1), pts[:3].max(axis=1)


def _bbox_tuple(bb: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    if bb is None:
        return None
    if isinstance(bb, dict):
        try:
            return (
                float(bb['xmin']), float(bb['xmax']),
                float(bb['ymin']), float(bb['ymax']),
                float(bb['zmin']), float(bb['zmax']),
            )
        except Exception:
            return None
    try:
        return (
            float(bb.xmin), float(bb.xmax),
            float(bb.ymin), float(bb.ymax),
            float(bb.zmin), float(bb.zmax),
        )
    except Exception:
        return None


def _entry_bboxes(entry: Dict[str, Any]) -> List[Tuple[float, float, float, float, float, float]]:
    bboxes: List[Tuple[float, float, float, float, float, float]] = []
    for face in entry.get('geometries') or []:
        try:
            bb = _bbox_tuple(face.BoundingBox())
        except Exception:
            bb = None
        if bb is not None:
            bboxes.append(bb)

    viz = entry.get('viz') if isinstance(entry.get('viz'), dict) else {}
    for face in viz.get('faces') or []:
        bb = _bbox_tuple(face.get('bbox') if isinstance(face, dict) else None)
        if bb is not None:
            bboxes.append(bb)
    bb = _bbox_tuple(viz.get('bbox'))
    if bb is not None:
        bboxes.append(bb)
    return bboxes


def _fraction(value: float, lo: float, hi: float, *, invert: bool = False) -> float:
    span = max(float(hi) - float(lo), 1e-12)
    frac = (float(value) - float(lo)) / span
    if invert:
        frac = 1.0 - frac
    return float(np.clip(frac, 0.0, 1.0))


def _fraction_box(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pad: float = 0.02,
) -> Tuple[float, float, float, float, float, float]:
    mins, maxs = bounds
    vals = [
        _fraction(bbox[0], mins[0], maxs[0]),
        _fraction(bbox[1], mins[0], maxs[0]),
        _fraction(bbox[2], mins[1], maxs[1]),
        _fraction(bbox[3], mins[1], maxs[1]),
        _fraction(bbox[4], mins[2], maxs[2]),
        _fraction(bbox[5], mins[2], maxs[2]),
    ]
    for i in (0, 2, 4):
        if abs(vals[i + 1] - vals[i]) < pad:
            center = 0.5 * (vals[i] + vals[i + 1])
            vals[i] = max(0.0, center - pad)
            vals[i + 1] = min(1.0, center + pad)
    return tuple(vals)  # type: ignore[return-value]


def _fraction_center(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float, float]:
    mins, maxs = bounds
    return (
        _fraction(0.5 * (bbox[0] + bbox[1]), mins[0], maxs[0]),
        _fraction(0.5 * (bbox[2] + bbox[3]), mins[1], maxs[1]),
        _fraction(0.5 * (bbox[4] + bbox[5]), mins[2], maxs[2]),
    )


def _bounds_payload(bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[str, List[float]]]:
    if bounds is None:
        return None
    mins, maxs = bounds
    return {
        'min': [float(v) for v in mins[:3]],
        'max': [float(v) for v in maxs[:3]],
    }


# ---------------------------------------------------------------------------
# CadQueryNode
# ---------------------------------------------------------------------------

class TopologyOptVoxelNode(CadQueryNode):
    """3-D SIMP topology optimisation using pyMOTO on a structured voxel grid.

    The graph-facing inputs match the normal FEM solver flow: mesh, material,
    constraints, and loads. Solver resolution and optimiser controls live as
    node properties instead of cluttering the graph with hyperparameter ports.
    """
    __identifier__ = 'com.cad.sim.topopt_voxel'
    NODE_NAME = 'Topology Opt (3D Voxel)'

    def __init__(self):
        super().__init__()
        self.add_output('result', color=(180, 255, 180))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)
        # ── Domain ────────────────────────────────────────────────────────
        self.create_property('nelx',    30,    widget_type='int')
        self.create_property('nely',    20,    widget_type='int')
        self.create_property('nelz',    10,    widget_type='int')
        self.create_property('volfrac', 0.5,   widget_type='float')
        self.create_property('rmin',    1.5,   widget_type='float')
        self.create_property('penal',   3.0,   widget_type='float')
        self.create_property('density_cutoff', 0.35, widget_type='float')
        self.create_property('visualization', 'Density', widget_type='combo',
                             items=['Density', 'Recovered Shape'])
        self.create_property('E0',      1.0,   widget_type='float')
        self.create_property('Emin',    1e-9,  widget_type='float')
        self.create_property('nu',      0.3,   widget_type='float')

        # ── Solver ────────────────────────────────────────────────────────
        self.create_property('optimizer', 'OC', widget_type='combo',
                             items=['OC', 'MMA'])
        self.create_property('max_iter', 80,   widget_type='int')
        self.create_property('tol',      0.01, widget_type='float')

        # ── BC Preset (convenience only) ─────────────────────────────────
        self.create_property('bc_preset', 'Cantilever Tip', widget_type='combo',
                             items=['Custom',
                                    'Cantilever Tip',
                                    'Cantilever Distributed',
                                    'MBB Beam',
                                    'Bridge'])

        # ── Face supports ─────────────────────────────────────────────────
        _support_items = ['None', 'Fix X', 'Fix Y', 'Fix Z',
                          'Fix XY', 'Fix YZ', 'Fix XZ', 'Fix XYZ']
        self.create_property('left_support',   'Fix XYZ', widget_type='combo', items=_support_items)
        self.create_property('right_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('top_support',    'None',    widget_type='combo', items=_support_items)
        self.create_property('bottom_support', 'None',    widget_type='combo', items=_support_items)
        self.create_property('front_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('back_support',   'None',    widget_type='combo', items=_support_items)
        self.create_property('support_regions', '[]', widget_type='text')

        # ── Force ─────────────────────────────────────────────────────────
        self.create_property('force_type', 'Point', widget_type='combo',
                             items=['Point', 'Distributed Face'])
        self.create_property('force_face', 'Right', widget_type='combo',
                             items=['Left', 'Right', 'Top', 'Bottom', 'Front', 'Back'])
        # Point force location as fractions of domain size [0, 1]
        self.create_property('force_ix_frac',   1.0,  widget_type='float')
        self.create_property('force_iy_frac',   0.5,  widget_type='float')
        self.create_property('force_iz_frac',   0.5,  widget_type='float')
        self.create_property('force_dir_x',     0.0,  widget_type='float')
        self.create_property('force_dir_y',    -1.0,  widget_type='float')
        self.create_property('force_dir_z',     0.0,  widget_type='float')
        self.create_property('force_magnitude', 1.0,  widget_type='float')

    # ── Property helpers ───────────────────────────────────────────────────

    def _apply_preset(self, preset: str) -> None:
        """Pre-fill support and force properties from a named preset."""
        _presets = {
            'Cantilever Tip': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                force_type='Point',
                force_ix_frac=1.0, force_iy_frac=0.5, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
            ),
            'Cantilever Distributed': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                force_type='Distributed Face', force_face='Right',
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
            ),
            'MBB Beam': dict(
                left_support='None',  right_support='None',
                top_support='None',   bottom_support='None',
                front_support='None', back_support='None',
                support_regions=json.dumps([
                    {"x": [0.00, 0.04], "y": [0.00, 0.04], "z": [0.0, 1.0], "dofs": "Fix XYZ"},
                    {"x": [0.96, 1.00], "y": [0.00, 0.04], "z": [0.0, 1.0], "dofs": "Fix YZ"},
                ]),
                force_type='Point',
                force_ix_frac=0.5, force_iy_frac=1.0, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
            ),
            'Bridge': dict(
                left_support='None',  right_support='None',
                top_support='None',   bottom_support='Fix Y',
                front_support='None', back_support='None',
                support_regions='[]',
                force_type='Point',
                force_ix_frac=0.5, force_iy_frac=1.0, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
            ),
        }
        if preset in _presets:
            for k, v in _presets[preset].items():
                self.set_property(k, v)

    def _build_bc(self) -> VoxelBC:
        """Construct a VoxelBC from the current node properties."""
        bc = VoxelBC(
            fixed_left_face_dofs   = _parse_support(self.get_property('left_support')),
            fixed_right_face_dofs  = _parse_support(self.get_property('right_support')),
            fixed_top_face_dofs    = _parse_support(self.get_property('top_support')),
            fixed_bottom_face_dofs = _parse_support(self.get_property('bottom_support')),
            fixed_front_face_dofs  = _parse_support(self.get_property('front_support')),
            fixed_back_face_dofs   = _parse_support(self.get_property('back_support')),
        )

        regions_text = str(self.get_property('support_regions') or '').strip()
        if regions_text:
            try:
                regions = json.loads(regions_text)
                if not isinstance(regions, list):
                    raise ValueError("support_regions must be a JSON list")
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    x0, x1 = region.get('x', [0.0, 0.0])
                    y0, y1 = region.get('y', [0.0, 0.0])
                    z0, z1 = region.get('z', [0.0, 1.0])
                    dofs = _parse_support_region_dofs(region.get('dofs', ''))
                    bc.fixed_boxes.append((
                        float(x0), float(x1),
                        float(y0), float(y1),
                        float(z0), float(z1),
                        dofs,
                    ))
            except Exception as exc:
                raise ValueError(f"Invalid support_regions JSON: {exc}") from exc

        mag  = float(self.get_property('force_magnitude') or 1.0)
        fdx  = float(self.get_property('force_dir_x') or 0.0)
        fdy  = float(self.get_property('force_dir_y') or -1.0)
        fdz  = float(self.get_property('force_dir_z') or 0.0)

        norm = (fdx ** 2 + fdy ** 2 + fdz ** 2) ** 0.5 or 1.0
        fdx, fdy, fdz = (fdx / norm) * mag, (fdy / norm) * mag, (fdz / norm) * mag

        ftype = self.get_property('force_type')
        if ftype == 'Point':
            ix_f = float(self.get_property('force_ix_frac') or 1.0)
            iy_f = float(self.get_property('force_iy_frac') or 0.5)
            iz_f = float(self.get_property('force_iz_frac') or 0.5)
            bc.point_forces.append((ix_f, iy_f, iz_f, fdx, fdy, fdz))
        else:  # Distributed Face
            face = (self.get_property('force_face') or 'Right').lower()
            bc.distributed_forces.append((face, fdx, fdy, fdz))

        return bc

    def _merge_graph_bcs(
        self,
        bc: VoxelBC,
        mesh: Any,
        constraints: List[Any],
        loads: List[Any],
    ) -> None:
        """Add ordinary PyLCSS face constraints and loads to the voxel BCs."""
        bounds = _mesh_bounds(mesh)
        if bounds is None:
            if constraints or loads:
                logger.warning(
                    "TopologyOptVoxelNode: cannot map graph BCs without mesh bounds."
                )
            return

        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            dofs = constraint.get('fixed_dofs') or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                continue
            for bbox in _entry_bboxes(constraint):
                bc.fixed_boxes.append((*_fraction_box(bbox, bounds), dofs))

        for load in loads:
            if not isinstance(load, dict) or load.get('type') != 'force':
                continue
            vector = load.get('vector')
            try:
                fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
            except Exception:
                continue
            bboxes = _entry_bboxes(load)
            if not bboxes:
                continue
            scale = 1.0 / max(1, len(bboxes))
            for bbox in bboxes:
                ix_f, iy_f, iz_f = _fraction_center(bbox, bounds)
                bc.point_forces.append((ix_f, iy_f, iz_f, fx * scale, fy * scale, fz * scale))

    # ── Node run ───────────────────────────────────────────────────────────

    def run(self, progress_callback=None) -> Optional[Dict[str, Any]]:
        preset = self.get_property('bc_preset')
        if preset != 'Custom':
            self._apply_preset(preset)

        mesh = self.get_input_value('mesh', None)
        bounds = _mesh_bounds(mesh)
        material = self.get_input_value('material', None)
        material = material if isinstance(material, dict) else {}
        constraint_list = _flatten(self.get_input_list('constraints'))
        load_list = _flatten(self.get_input_list('loads'))
        bc = self._build_bc()
        if constraint_list:
            bc.fixed_left_face_dofs = []
            bc.fixed_right_face_dofs = []
            bc.fixed_top_face_dofs = []
            bc.fixed_bottom_face_dofs = []
            bc.fixed_front_face_dofs = []
            bc.fixed_back_face_dofs = []
            bc.fixed_boxes = []
        if load_list:
            bc.point_forces = []
            bc.distributed_forces = []
        self._merge_graph_bcs(bc, mesh, constraint_list, load_list)

        nelx = int(self.get_property('nelx') or 30)
        nely = int(self.get_property('nely') or 20)
        nelz = int(self.get_property('nelz') or 10)
        unitx = unity = unitz = 1.0
        if bounds is not None:
            mins, maxs = bounds
            span = np.maximum(maxs[:3] - mins[:3], 1e-12)
            unitx = float(span[0] / max(nelx, 1))
            unity = float(span[1] / max(nely, 1))
            unitz = float(span[2] / max(nelz, 1))

        problem = TopologyOptVoxelProblem(
            nelx     = nelx,
            nely     = nely,
            nelz     = nelz,
            E0       = float(material.get('E', self.get_property('E0') or 1.0)),
            Emin     = float(self.get_property('Emin')  or 1e-9),
            nu       = float(material.get('nu', self.get_property('nu') or 0.3)),
            penal    = float(self.get_property('penal') or 3.0),
            volfrac  = float(self.get_property('volfrac') or 0.5),
            rmin     = float(self.get_property('rmin') or 1.5),
            unitx    = unitx,
            unity    = unity,
            unitz    = unitz,
            optimizer = self.get_property('optimizer') or 'OC',
            max_iter = int(self.get_property('max_iter') or 80),
            tol      = float(self.get_property('tol')   or 0.01),
            bc       = bc,
        )

        solver = TopologyOptVoxelSolver(problem)

        def _cb(it: int, comp: float, change: float, density: np.ndarray) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            'type': 'topopt_voxel',
                            'density': density,
                            'grid_shape': density.shape,
                            'bounds': _bounds_payload(bounds),
                            'density_cutoff': float(self.get_property('density_cutoff') or 0.35),
                            '_preview': True,
                        },
                        density,
                        max(0, it - 1),
                        problem.max_iter,
                    )
                except Exception:
                    pass

        try:
            result = solver.run(callback=_cb)
        except Exception as exc:
            logger.exception("TopologyOptVoxelNode: solver error")
            self.set_error(str(exc))
            return None

        logger.info("TopologyOptVoxelNode: %s", result.message)
        density = np.asarray(result.density, dtype=float)
        density_cutoff = float(self.get_property('density_cutoff') or 0.35)
        recovered = _recover_voxel_shape(density, bounds, density_cutoff)
        return {
            'type': 'topopt_voxel',
            'density': density,
            'grid_shape': density.shape,
            'bounds': _bounds_payload(bounds),
            'density_cutoff': density_cutoff,
            'recovered_shape': recovered,
            'visualization_mode': self.get_property('visualization') or 'Density',
            'target_vol_frac': problem.volfrac,
            'final_vol_frac': float(np.mean(density)),
            'compliance': (
                float(result.compliance_history[-1])
                if result.compliance_history else None
            ),
            'iterations': result.n_iter,
            'converged': result.converged,
            'message': result.message,
            'compliance_history': result.compliance_history,
            'change_history': result.change_history,
        }
