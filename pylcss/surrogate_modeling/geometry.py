# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD -> mesh -> SDF infrastructure for geometry-aware surrogates.

The tabular surrogates (MLP, GP, RF, PyTorch DNN) train on
``(parameter_vector, scalar_qoi)`` pairs and don't care about geometry.  The
geometric-aware surrogates (Geom-DeepONet, GINO) need *the actual geometry*
that each parameter vector produces, because their input is the design space's
**signed distance function** (SDF), not the parameter vector.

This module bridges the two:

1. ``cad_evaluate_geometry(cad_path, kind, params)`` runs PyLCSS's existing
   CAD graph (via :mod:`pylcss.cad.runtime`) at the given parameters and
   returns the resulting mesh + nodal fields.
2. ``compute_sdf(points, cells, query_points)`` evaluates the signed distance
   from ``query_points`` to the surface defined by ``(points, cells)``.
3. ``GeometryCache`` memoizes CAD evaluations so an optimizer that probes the
   same parameter set twice doesn't re-run FEA twice.

Both helpers degrade gracefully if their optional deps (``trimesh``) are
missing -- callers should check ``TRIMESH_AVAILABLE`` before assuming SDF
support is online.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.info("trimesh not available; SDF computation disabled.")


# ----------------------------------------------------------------------------
# Data class for a CAD evaluation result.  Mirrors what cad.runtime returns,
# but standardises the field/mesh access so downstream code can be agnostic
# about whether the CAD graph terminated in FEA, crash, or topopt.
# ----------------------------------------------------------------------------
@dataclass
class CadGeometry:
    """Mesh + nodal fields extracted from one CAD-graph evaluation."""

    points: np.ndarray                          # (n_nodes, 3)
    cells: np.ndarray                           # (n_cells, n_corners) -- triangles or tetra
    fields: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_nodes, n_components)
    scalars: Dict[str, float] = field(default_factory=dict)      # standardized scalars (max_stress, mass, ...)
    params: Dict[str, float] = field(default_factory=dict)       # the input parameter values

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (xyz min, xyz max)."""
        return np.min(self.points, axis=0), np.max(self.points, axis=0)


# ----------------------------------------------------------------------------
# CAD evaluation
# ----------------------------------------------------------------------------
def cad_evaluate_geometry(
    cad_path: str,
    kind: str,
    params: Mapping[str, float],
    field_name: Optional[str] = None,
) -> CadGeometry:
    """Run the CAD graph at the given parameters and extract its geometry +
    fields.

    Parameters
    ----------
    cad_path : str
        Path to the .json CAD graph file.
    kind : {"fea", "crash", "topopt"}
        Which terminal solver to evaluate.
    params : mapping
        Input variable names -> float values, matching the CAD graph's
        exposed inputs.
    field_name : str, optional
        If given, only this nodal field is extracted from ``raw``.  Useful to
        avoid copying the whole results dict when only one quantity matters.

    Returns
    -------
    :class:`CadGeometry`
        ``points`` and ``cells`` are required to be present in ``raw``; nodal
        fields are populated best-effort from common key names ("stress",
        "displacement", "von_mises", "node_stress", ...).
    """
    from pylcss.cad import runtime as cad_runtime

    if kind == "fea":
        result = cad_runtime.fea(cad_path, **params)
    elif kind == "crash":
        result = cad_runtime.crash(cad_path, **params)
    elif kind == "topopt":
        result = cad_runtime.topopt(cad_path, **params)
    else:
        raise ValueError(f"Unknown solver kind {kind!r}; expected fea/crash/topopt.")

    # CadResult.raw and CadResult.standard are methods, not properties --
    # they have to be CALLED.  ``result.raw`` returns the bound method object,
    # which has no .get() and produces "'function' object has no attribute
    # 'get'" deep inside _coerce_points.
    raw = result.raw()
    standard = result.standard()

    # Mesh: points + cells under several common names depending on solver.
    points = _coerce_points(raw)
    cells = _coerce_cells(raw)
    if points is None or cells is None:
        raise RuntimeError(
            f"CAD result has no mesh data; cannot build geometric surrogate. "
            f"Raw keys: {sorted(raw.keys())}"
        )

    # Nodal fields: vary heavily by solver. Pick a few canonical names.
    fields = _extract_nodal_fields(raw, points.shape[0], only=field_name)

    return CadGeometry(
        points=np.asarray(points, dtype=np.float64),
        cells=np.asarray(cells, dtype=np.int64),
        fields=fields,
        scalars=dict(standard),
        params=dict(params),
    )


# PyLCSS's CalculiX backend stuffs the FEA mesh into raw["mesh"] as a
# skfem.Mesh -- which uses ``.p`` (3, n_nodes) and ``.t`` (n_corners, n_cells)
# instead of the (n_nodes, 3) / (n_cells, n_corners) most numpy-style libraries
# use.  We unwrap that here so the rest of the SDF/training code can stay
# library-agnostic.
def _coerce_points(raw: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Pull mesh node coordinates out of a CAD raw dict."""
    # Preferred: raw["mesh"] is a skfem.Mesh with .p of shape (3, N).
    mesh_obj = raw.get("mesh")
    if mesh_obj is not None and hasattr(mesh_obj, "p"):
        p = np.asarray(mesh_obj.p, dtype=np.float64)
        if p.ndim == 2:
            if p.shape[0] == 3:        # skfem (3, N) -> (N, 3)
                return p.T
            if p.shape[1] == 3:        # already (N, 3)
                return p
            if p.shape[0] == 2:        # 2-D skfem -> lift to xyz
                return np.column_stack([p.T, np.zeros(p.shape[1])])

    # Fallback: top-level numpy arrays under common names.
    for key in ("points", "nodes", "node_coords", "vertices", "coords"):
        val = raw.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        if arr.ndim == 2 and arr.shape[1] == 2:
            return np.column_stack([arr, np.zeros(arr.shape[0])])
    return None


def _coerce_cells(raw: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Pull element connectivity out of a CAD raw dict."""
    # Preferred: raw["mesh"].t with shape (n_corners, n_cells).
    mesh_obj = raw.get("mesh")
    if mesh_obj is not None and hasattr(mesh_obj, "t"):
        t = np.asarray(mesh_obj.t, dtype=np.int64)
        if t.ndim == 2:
            # skfem stores connectivity as (n_corners, n_cells).  Tet4 has
            # 4 corners; quadratic tets (C3D10) come in as 10.  We only need
            # the first 4 corners for SDF surface extraction.
            if t.shape[0] in (3, 4, 6, 8, 10):
                conn = t.T  # (n_cells, n_corners)
            elif t.shape[1] in (3, 4, 6, 8, 10):
                conn = t   # already (n_cells, n_corners)
            else:
                conn = t.T
            return conn[:, :4] if conn.shape[1] >= 4 else conn

    for key in ("cells", "elements", "tris", "triangles", "tets", "tetrahedra", "faces"):
        val = raw.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.int64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr
    return None


# User-facing field name -> list of raw-dict keys to try (first hit wins).
# Lets the UI default to "von_mises" while CalculiX raw actually carries the
# scalar in "stress".
_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "von_mises": ("von_mises", "vonmises", "stress_vm", "node_stress",
                  "nodal_stress", "stress"),
    "stress":    ("stress", "von_mises", "vonmises", "stress_vm",
                  "node_stress", "nodal_stress"),
    "displacement": ("displacement", "node_disp", "u"),
    "energy":    ("ener_nodal", "energy", "node_energy"),
}


def _extract_nodal_fields(
    raw: Mapping[str, Any], n_nodes: int, only: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """Best-effort extraction of per-node arrays from the raw result.

    When ``only`` is given, the alias table is consulted so a user-facing
    "von_mises" pull works against a CalculiX backend that stores the same
    quantity under "stress".
    """
    out: Dict[str, np.ndarray] = {}

    if only is not None:
        # Try aliases for the requested name; expose the result under the
        # user-facing key so downstream code doesn't have to know aliases.
        keys_to_try = _FIELD_ALIASES.get(only, (only,))
        for key in keys_to_try:
            val = raw.get(key)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float64)
            arr = _normalize_nodal_array(arr, n_nodes)
            if arr is not None:
                out[only] = arr
                return out
        return out

    # No filter: scan all known field names.
    candidates = [
        "von_mises", "vonmises", "stress_vm", "node_stress", "nodal_stress",
        "stress", "displacement", "node_disp", "u", "u_x", "u_y", "u_z",
        "density", "strain", "temperature", "ener_nodal", "energy",
    ]
    for key in candidates:
        val = raw.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float64)
        arr = _normalize_nodal_array(arr, n_nodes)
        if arr is not None:
            out[key] = arr
    return out


def _normalize_nodal_array(arr: np.ndarray, n_nodes: int) -> Optional[np.ndarray]:
    """Coerce a per-node array into shape (n_nodes, n_components) or return
    None if the shape doesn't fit (cell-centred, wrong node count, ...).

    Handles two layouts seen in practice:
      - (n_nodes,)        -> (n_nodes, 1)
      - (n_nodes, k)      -> kept
      - (k, n_nodes)      -> transposed (rare; some FRD ingests)
      - (3 * n_nodes,)    -> reshaped to (n_nodes, 3) -- displacement flat layout
    """
    if arr.ndim == 1:
        if arr.shape[0] == n_nodes:
            return arr.reshape(-1, 1)
        if arr.shape[0] == 3 * n_nodes:
            return arr.reshape(n_nodes, 3)
        return None
    if arr.ndim == 2:
        if arr.shape[0] == n_nodes:
            return arr
        if arr.shape[1] == n_nodes and arr.shape[0] in (1, 3, 6, 9):
            return arr.T
    return None


# ----------------------------------------------------------------------------
# SDF computation
# ----------------------------------------------------------------------------
def compute_sdf(
    points: np.ndarray,
    cells: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Signed distance from each query point to the surface defined by the
    mesh.

    Inside the surface -> negative.  Outside -> positive.  The mesh must be
    closed (watertight) for the sign to be reliable; for open meshes we fall
    back to *unsigned* distance and emit a debug log.

    Parameters
    ----------
    points : (n_nodes, 3) float
    cells : (n_cells, n_corners) int
        Triangles (n_corners == 3) or tets (4) or other.  Surface extracted
        automatically.
    query_points : (n_query, 3) float

    Returns
    -------
    (n_query,) float : signed distances (or absolute if the mesh isn't closed)
    """
    if not TRIMESH_AVAILABLE:
        raise RuntimeError(
            "trimesh is required for SDF computation. "
            "Install with: pip install trimesh"
        )

    points = np.asarray(points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    # If the mesh is volumetric (tets), extract its surface; trimesh works on
    # triangular surfaces only.
    if cells.shape[1] == 4:
        faces = _tetra_to_surface(cells)
    elif cells.shape[1] == 3:
        faces = np.asarray(cells, dtype=np.int64)
    else:
        # n_corners > 4 (hex etc.): take the first 3 nodes of each cell as a
        # rough triangulation. This is a fallback -- accuracy degrades but
        # avoids a hard crash.
        faces = np.asarray(cells[:, :3], dtype=np.int64)
        logger.debug("Mesh has %d-corner cells; using fallback triangulation.", cells.shape[1])

    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    if mesh.is_watertight:
        # signed_distance is positive inside, negative outside in trimesh -- we
        # invert so the convention matches the literature (negative inside).
        sd = -mesh.nearest.signed_distance(query_points)
    else:
        logger.debug("Mesh isn't watertight; falling back to unsigned distance.")
        closest, dists, _ = mesh.nearest.on_surface(query_points)
        sd = dists
    return np.asarray(sd, dtype=np.float64)


def _tetra_to_surface(tets: np.ndarray) -> np.ndarray:
    """Extract the outer boundary triangles of a tetrahedral mesh.

    Each tet has 4 triangular faces; a face is on the boundary iff it appears
    in exactly one tet.  We dedupe sorted-triplet keys and keep singletons.
    """
    tris = np.vstack([
        tets[:, [0, 1, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 2, 3]],
    ])
    sorted_tris = np.sort(tris, axis=1)
    # Count occurrences via unique (axis=0)
    _, inv, counts = np.unique(sorted_tris, axis=0, return_inverse=True, return_counts=True)
    # Boundary faces: appear exactly once
    boundary_mask = counts[inv] == 1
    return tris[boundary_mask]


# ----------------------------------------------------------------------------
# Geometry cache
# ----------------------------------------------------------------------------
class GeometryCache:
    """Thread-safe LRU-ish cache for (cad_path, kind, params) -> CadGeometry.

    Sized small by default because each entry can hold a moderately large mesh
    (10k-100k nodes); used by training loops + optimizer probes that revisit
    the same parameters.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._store: Dict[str, CadGeometry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(cad_path: str, kind: str, params: Mapping[str, float]) -> str:
        # Hash the sorted params + path + kind. Round to 12 sig figs so trivial
        # float jitter doesn't cause cache misses.
        items = sorted((k, float(v)) for k, v in params.items())
        h = hashlib.sha1()
        h.update(cad_path.encode("utf-8"))
        h.update(b"|")
        h.update(kind.encode("utf-8"))
        h.update(b"|")
        for k, v in items:
            h.update(f"{k}={v:.12g};".encode("utf-8"))
        return h.hexdigest()

    def get(
        self, cad_path: str, kind: str, params: Mapping[str, float],
    ) -> Optional[CadGeometry]:
        key = self._key(cad_path, kind, params)
        with self._lock:
            return self._store.get(key)

    def put(
        self, cad_path: str, kind: str, params: Mapping[str, float], geom: CadGeometry,
    ) -> None:
        key = self._key(cad_path, kind, params)
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Drop an arbitrary old entry (dict preserves insertion order in 3.7+)
                self._store.pop(next(iter(self._store)))
            self._store[key] = geom

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# A module-level singleton -- callers can clear it from a UI button if memory
# matters.  Not used by default; pass an explicit cache to evaluate_with_cache.
_DEFAULT_CACHE = GeometryCache(max_entries=128)


def evaluate_with_cache(
    cad_path: str,
    kind: str,
    params: Mapping[str, float],
    field_name: Optional[str] = None,
    cache: Optional[GeometryCache] = None,
) -> CadGeometry:
    """Cached version of :func:`cad_evaluate_geometry` for repeated probes."""
    cache = cache if cache is not None else _DEFAULT_CACHE
    hit = cache.get(cad_path, kind, params)
    if hit is not None:
        return hit
    geom = cad_evaluate_geometry(cad_path, kind, params, field_name=field_name)
    cache.put(cad_path, kind, params, geom)
    return geom


# ----------------------------------------------------------------------------
# Helpers for backbones that need SDF on a fixed background grid (GINO)
# or on a per-design point cloud (Geom-DeepONet).
# ----------------------------------------------------------------------------
def make_background_grid(
    bbox_min: np.ndarray, bbox_max: np.ndarray, resolution: int = 32,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Uniform 3-D grid covering ``[bbox_min, bbox_max]`` at ``resolution``
    samples per axis. Padding is added so the grid extends ~10 % beyond the
    bbox, which makes interpolation back to surface nodes well-conditioned.

    Returns
    -------
    points : (resolution**3, 3) float
    shape  : (resolution, resolution, resolution)
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    extent = bbox_max - bbox_min
    pad = extent * 0.1
    lo, hi = bbox_min - pad, bbox_max + pad
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    zs = np.linspace(lo[2], hi[2], resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)  # (r, r, r, 3)
    return grid.reshape(-1, 3), (resolution, resolution, resolution)
