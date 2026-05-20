# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Mesh → fitted trimmed-NURBS B-rep reconstruction for topology results.

The smooth recovered mesh (marching cubes + SDF smoothing, see ``recovery``) is
clean and watertight but dense.  Sewing every triangle into its own planar OCC
face produces a huge, non-editable STEP.  This module instead reconstructs a
compact B-rep by *autosurfacing*:

    1. segment the mesh into smooth, single-valued regions,
    2. fit one trimmed B-spline surface per region,
    3. sew the patches into a solid.

The result is a handful-to-hundreds of smooth NURBS faces rather than ~10^5
planar triangles.

## Honest scope / limitations

* Watertightness comes from **sew-tolerance healing** of independently fitted
  patches, not from topologically shared edge curves.  Adjacent patches can
  deviate by up to the fit tolerance along a shared border; the sewing
  tolerance (tied to ``fit_tol``) bridges that gap.  Pathological thin / high
  curvature regions fall back to their original triangles, and if the sewn
  shell still will not close the *caller* falls back to the faceted / voxel
  path — so a valid solid is always produced somewhere in the chain.
* This is automatic autosurfacing, **not** a user-guided quad control cage: it
  reproduces the geometry to ``fit_tol``; it does not give an editable quad
  net (true PolyNURBS-style remeshing would need QuadriFlow / Instant-Meshes,
  which are not available here).
* Sharp edges are preserved only where they coincide with region borders
  (normal discontinuities) — which is where they physically sit on a topology
  optimised part (design-domain cut planes, hole walls).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _local_frame(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """PCA frame for a region: returns (centroid, U, V, N).

    ``N`` is the smallest-variance axis (the projection/normal direction); ``U``
    and ``V`` span the best-fit plane.  Returns None for degenerate regions.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vh.shape[0] < 3:
        return None
    u_axis = vh[0]
    v_axis = vh[1]
    normal = vh[2]
    n_norm = np.linalg.norm(normal)
    if not np.isfinite(n_norm) or n_norm < 1e-12:
        return None
    return centroid, u_axis, v_axis, normal / n_norm


def _trace_boundary_loops(region_faces: np.ndarray) -> List[List[int]]:
    """Return ordered boundary vertex loops (global indices) for a face set.

    A directed edge is on the region boundary when its reverse is not also a
    directed edge of the region.  Boundary edges are chained into closed loops;
    a disk-like region yields one loop, an annular region yields two, etc.
    """
    region_faces = np.asarray(region_faces, dtype=np.int64)
    if len(region_faces) == 0:
        return []

    directed = set()
    for tri in region_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        directed.add((a, b))
        directed.add((b, c))
        directed.add((c, a))

    # Outgoing boundary edges keyed by start vertex (lists handle the rare
    # non-manifold boundary vertex with >1 outgoing edge).
    out_edges: dict[int, list[int]] = {}
    for (a, b) in directed:
        if (b, a) not in directed:
            out_edges.setdefault(a, []).append(b)

    loops: List[List[int]] = []
    for start in list(out_edges.keys()):
        while out_edges.get(start):
            loop = [start]
            cur = start
            nxt = out_edges[cur].pop()
            guard = 0
            limit = len(directed) + 4
            while nxt != start and guard < limit:
                loop.append(nxt)
                successors = out_edges.get(nxt)
                if not successors:
                    break
                cur, nxt = nxt, successors.pop()
                guard += 1
            if nxt == start and len(loop) >= 3:
                loops.append(loop)
    return loops


def _polygon_area_2d(uv: np.ndarray) -> float:
    """Signed area (shoelace) of a 2-D polygon."""
    if len(uv) < 3:
        return 0.0
    x = uv[:, 0]
    y = uv[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# Per-region surface fitting
# ---------------------------------------------------------------------------

def _fit_bspline_surface(region_verts: np.ndarray, frame, fit_tol: float):
    """Fit a Geom_BSplineSurface over a region as a height field on its frame.

    Builds a structured grid on the PCA plane, samples the region height with a
    linear interpolant (nearest-fill outside the hull), and fits a C2 B-spline
    surface through the grid.  Returns the OCC surface or None.
    """
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.GeomAbs import GeomAbs_C2

    centroid, u_axis, v_axis, normal = frame
    rel = region_verts - centroid
    u = rel @ u_axis
    v = rel @ v_axis
    h = rel @ normal

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    du = max(u_max - u_min, 1e-9)
    dv = max(v_max - v_min, 1e-9)

    # Grid resolution from vertex count, biased by aspect ratio, clamped.
    n_base = int(np.clip(round(np.sqrt(len(region_verts))), 6, 40))
    aspect = np.sqrt(du / dv)
    nu = int(np.clip(round(n_base * aspect), 4, 40))
    nv = int(np.clip(round(n_base / aspect), 4, 40))

    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    except ImportError:
        return None

    uv = np.column_stack([u, v])
    try:
        lin = LinearNDInterpolator(uv, h)
        nearest = NearestNDInterpolator(uv, h)
    except Exception:
        return None

    gu = np.linspace(u_min, u_max, nu)
    gv = np.linspace(v_min, v_max, nv)
    grid_u, grid_v = np.meshgrid(gu, gv, indexing='ij')
    flat = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    gh = lin(flat)
    missing = ~np.isfinite(gh)
    if np.any(missing):
        gh[missing] = nearest(flat[missing])
    gh = gh.reshape(nu, nv)
    if not np.all(np.isfinite(gh)):
        return None

    arr = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        ui = gu[i]
        for j in range(nv):
            p = centroid + ui * u_axis + gv[j] * v_axis + gh[i, j] * normal
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    try:
        fitter = GeomAPI_PointsToBSplineSurface(arr, 3, 8, GeomAbs_C2, float(fit_tol))
        if not fitter.IsDone():
            return None
        return fitter.Surface()
    except Exception:
        return None


def _project_loop_uv(surf, loop_pts: np.ndarray) -> Optional[np.ndarray]:
    """Project 3-D loop points onto the fitted surface, returning (u,v) params.

    The fitted surface's parameterisation is not the PCA plane, so trim curves
    must be built from true surface parameters obtained by projection.
    """
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt

    u0, u1, v0, v1 = surf.Bounds()
    out = np.empty((len(loop_pts), 2), dtype=float)
    for k, p in enumerate(loop_pts):
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)
        if proj.NbPoints() < 1:
            return None
        pu, pv = proj.LowerDistanceParameters()
        out[k, 0] = float(np.clip(pu, u0, u1))
        out[k, 1] = float(np.clip(pv, v0, v1))
    return out


def _wire_from_uv(surf, uv: np.ndarray):
    """Build a wire on ``surf`` from a closed (u,v) polygon. Returns wire or None."""
    from OCP.gp import gp_Pnt2d
    from OCP.GCE2d import GCE2d_MakeSegment
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    if len(uv) < 3:
        return None
    mw = BRepBuilderAPI_MakeWire()
    n = len(uv)
    added = 0
    for i in range(n):
        a = uv[i]
        b = uv[(i + 1) % n]
        if np.allclose(a, b, atol=1e-12):
            continue
        try:
            seg = GCE2d_MakeSegment(
                gp_Pnt2d(float(a[0]), float(a[1])),
                gp_Pnt2d(float(b[0]), float(b[1])),
            ).Value()
            edge = BRepBuilderAPI_MakeEdge(seg, surf)
            if not edge.IsDone():
                continue
            mw.Add(edge.Edge())
            added += 1
        except Exception:
            continue
    if added < 3 or not mw.IsDone():
        return None
    return mw.Wire()


def _fit_region_face(region_verts: np.ndarray, region_faces: np.ndarray, fit_tol: float):
    """Fit a single trimmed NURBS face for one region. Returns OCC face or None."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepLib import BRepLib
    from OCP.BRepCheck import BRepCheck_Analyzer

    frame = _local_frame(region_verts)
    if frame is None:
        return None

    surf = _fit_bspline_surface(region_verts, frame, fit_tol)
    if surf is None:
        return None

    loops = _trace_boundary_loops(region_faces)
    if not loops:
        return None

    # Map global vertex ids to coordinates via the region's own vertices: the
    # loops use global ids, so look them up in the full vertex array passed in
    # through region_verts' companion map.  Here region_faces already index the
    # global vertex array, so we resolve against it via the closure provided by
    # the caller (see recovered_mesh_to_nurbs_brep_solid).
    raise RuntimeError("internal: _fit_region_face requires global vertex lookup")


# The fitting needs global vertex coordinates for boundary loops; the public
# routine wires the pieces together with the full vertex array in scope.


def _make_filling_face(loop_pts_list: List[np.ndarray], interior_pts: Optional[np.ndarray]):
    """Fallback: energy-minimising fill surface from boundary loops. Face or None."""
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0

    try:
        fill = BRepOffsetAPI_MakeFilling()
        added = 0
        for loop in loop_pts_list:
            n = len(loop)
            if n < 3:
                continue
            for i in range(n):
                a = loop[i]
                b = loop[(i + 1) % n]
                if np.allclose(a, b, atol=1e-12):
                    continue
                edge = BRepBuilderAPI_MakeEdge(
                    gp_Pnt(float(a[0]), float(a[1]), float(a[2])),
                    gp_Pnt(float(b[0]), float(b[1]), float(b[2])),
                )
                if edge.IsDone():
                    fill.Add(edge.Edge(), GeomAbs_C0)
                    added += 1
        if added < 3:
            return None
        if interior_pts is not None:
            for p in interior_pts:
                fill.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        fill.Build()
        if not fill.IsDone():
            return None
        from OCP.TopoDS import TopoDS
        return TopoDS.Face_s(fill.Shape())
    except Exception:
        return None


def _triangle_faces(verts: np.ndarray, region_faces: np.ndarray) -> list:
    """Fallback: emit a region's triangles as planar OCC faces."""
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace

    faces_out = []
    for tri in region_faces:
        try:
            pts = verts[np.asarray(tri[:3], dtype=int)]
        except Exception:
            continue
        if not np.all(np.isfinite(pts)):
            continue
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if area <= 1e-12:
            continue
        try:
            poly = BRepBuilderAPI_MakePolygon(
                gp_Pnt(*[float(x) for x in pts[0]]),
                gp_Pnt(*[float(x) for x in pts[1]]),
                gp_Pnt(*[float(x) for x in pts[2]]),
                True,
            )
            if not poly.IsDone():
                continue
            fb = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if fb.IsDone():
                faces_out.append(fb.Face())
        except Exception:
            continue
    return faces_out


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _segment_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_angle_deg: float,
    max_patches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label faces into smooth, single-valued regions.

    Returns (labels, face_normals).  Uses dihedral-angle connected components,
    splits regions whose normal cone is too wide (so each region stays
    projectable), then merges sliver regions into their best neighbour.
    """
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.face_normals, dtype=float)
    n_faces = len(faces)

    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)        # (M, 2)
    ang = np.asarray(mesh.face_adjacency_angles, dtype=float)    # (M,)
    angle_thresh = np.radians(float(region_angle_deg))

    if len(adj) == 0:
        return np.zeros(n_faces, dtype=np.int64), normals

    keep = ang < angle_thresh
    rows = adj[keep, 0]
    cols = adj[keep, 1]
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n_faces, n_faces),
    ).tocsr()
    n_comp, labels = connected_components(graph, directed=False)

    # Split components whose normal cone is too wide to be single-valued.
    cone_limit = np.radians(72.0)
    next_label = int(n_comp)
    try:
        from sklearn.cluster import KMeans
        have_kmeans = True
    except ImportError:
        have_kmeans = False

    if have_kmeans:
        for lab in range(n_comp):
            fidx = np.flatnonzero(labels == lab)
            if len(fidx) < 8:
                continue
            nrm = normals[fidx]
            mean = nrm.mean(axis=0)
            mn = np.linalg.norm(mean)
            if mn < 1e-9:
                k = 3
            else:
                mean /= mn
                cos = np.clip(nrm @ mean, -1.0, 1.0)
                if float(np.max(np.arccos(cos))) <= cone_limit:
                    continue
                k = int(np.clip(np.ceil(float(np.max(np.arccos(cos))) / cone_limit) + 1, 2, 6))
            try:
                km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(nrm)
            except Exception:
                continue
            for c in range(k):
                sub = fidx[km.labels_ == c]
                if len(sub) == 0:
                    continue
                labels[sub] = next_label
                next_label += 1

    # Relabel to a dense 0..K-1 range.
    _, labels = np.unique(labels, return_inverse=True)
    n_lab = int(labels.max()) + 1

    # Merge sliver regions into the neighbour with the most shared edges and the
    # closest mean normal.  Cheap, a couple of passes.
    mean_normals = np.zeros((n_lab, 3), dtype=float)
    for lab in range(n_lab):
        fidx = np.flatnonzero(labels == lab)
        m = normals[fidx].mean(axis=0)
        nm = np.linalg.norm(m)
        mean_normals[lab] = m / nm if nm > 1e-9 else m

    min_region_faces = max(4, n_faces // max(1, int(max_patches)) // 4)
    for _ in range(3):
        counts = np.bincount(labels, minlength=n_lab)
        small = set(int(l) for l in np.flatnonzero(counts < min_region_faces))
        if not small:
            break
        # neighbour shared-edge tallies across the full adjacency
        la = labels[adj[:, 0]]
        lb = labels[adj[:, 1]]
        cross = la != lb
        moved = False
        for lab in small:
            mask = cross & ((la == lab) | (lb == lab))
            if not np.any(mask):
                continue
            neigh = np.where(la[mask] == lab, lb[mask], la[mask])
            cand, tally = np.unique(neigh, return_counts=True)
            # prefer the neighbour with most shared edges, break ties by normal
            best = cand[np.argmax(tally)]
            labels[labels == lab] = best
            moved = True
        if not moved:
            break
        _, labels = np.unique(labels, return_inverse=True)
        n_lab = int(labels.max()) + 1
        mean_normals = np.zeros((n_lab, 3), dtype=float)
        for lab in range(n_lab):
            fidx = np.flatnonzero(labels == lab)
            m = normals[fidx].mean(axis=0)
            nm = np.linalg.norm(m)
            mean_normals[lab] = m / nm if nm > 1e-9 else m

    return labels.astype(np.int64), normals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recovered_mesh_to_nurbs_brep_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sew_tolerance: float = 1e-3,
    fit_tol: float = 0.0,
    region_angle_deg: float = 30.0,
    max_patches: int = 600,
    merge_angle_deg: float = 1.0,
):
    """Reconstruct a CAD solid from a recovered triangle mesh by NURBS fitting.

    Returns a CadQuery ``Solid`` (post sew + ShapeFix + UnifySameDomain).  Raises
    on failure so the caller can fall back to the faceted / voxel paths.
    """
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

    # Reuse the proven sew→shell→solid + unify helpers from the sibling module.
    from pylcss.design_studio.topology_optimization.cad_reconstruction import (
        _drop_degenerate_faces,
        _shape_to_shells,
        _shell_to_solid,
        _unify_same_domain_shape,
    )

    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    vertices, faces = _drop_degenerate_faces(vertices, faces)
    if len(faces) < 4:
        raise RuntimeError("recovered mesh has too few faces for NURBS fitting")

    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if fit_tol <= 0.0:
        fit_tol = max(1e-4, 0.005 * diag)
    sew_tol = max(float(sew_tolerance), 1.5 * fit_tol)

    labels, _normals = _segment_regions(vertices, faces, region_angle_deg, max_patches)
    n_lab = int(labels.max()) + 1

    sew = BRepBuilderAPI_Sewing(float(sew_tol))
    n_nurbs = 0
    n_fill = 0
    n_tri = 0
    devs: List[float] = []

    for lab in range(n_lab):
        fidx = np.flatnonzero(labels == lab)
        region_faces = faces[fidx]
        used = np.unique(region_faces)
        region_verts = vertices[used]
        if len(region_verts) < 3 or len(region_faces) < 1:
            continue

        face = None
        loop_pts_list: List[np.ndarray] = []
        if len(region_faces) >= 4 and len(region_verts) >= 6:
            frame = _local_frame(region_verts)
            if frame is not None:
                surf = _fit_bspline_surface(region_verts, frame, fit_tol)
                if surf is not None:
                    loops = _trace_boundary_loops(region_faces)
                    loop_pts_list = [vertices[np.asarray(lp, dtype=int)] for lp in loops]
                    face = _trim_surface(surf, loop_pts_list, region_verts, devs)

        if face is None and loop_pts_list:
            # Energy-minimising fallback over the same boundary loops.
            centroid = region_verts.mean(axis=0)
            interior = region_verts[:: max(1, len(region_verts) // 12)]
            ff = _make_filling_face(loop_pts_list, interior)
            if ff is not None:
                face = ff
                n_fill += 1
        elif face is not None:
            n_nurbs += 1

        if face is not None:
            sew.Add(face)
        else:
            for tri_face in _triangle_faces(vertices, region_faces):
                sew.Add(tri_face)
                n_tri += 1

    sew.Perform()
    sewed = sew.SewedShape()
    shells = _shape_to_shells(sewed)
    if not shells:
        raise RuntimeError("NURBS patch sewing produced no shell")

    occ_solid = _shell_to_solid(shells[0])
    solid = cq.Solid(occ_solid)
    if not solid.isValid():
        raise RuntimeError("NURBS reconstruction produced an invalid B-rep solid")
    try:
        if abs(float(solid.Volume())) <= 1e-12:
            raise RuntimeError("NURBS reconstruction produced a zero-volume solid")
    except AttributeError:
        pass

    solid = _unify_same_domain_shape(solid, merge_angle_deg=merge_angle_deg)

    max_dev = max(devs) if devs else 0.0
    rms_dev = float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0
    logger.info(
        "NURBS CAD reconstruction: %d regions -> %d B-spline + %d filled + %d "
        "triangle faces (fit_tol=%.3g, max dev=%.3g, rms dev=%.3g).",
        n_lab, n_nurbs, n_fill, n_tri, fit_tol, max_dev, rms_dev,
    )
    return solid


def _trim_surface(surf, loop_pts_list, region_verts, devs):
    """Trim a fitted surface to its region boundary; return a valid face or None."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepLib import BRepLib
    from OCP.BRepCheck import BRepCheck_Analyzer

    if not loop_pts_list:
        return None

    # Outer loop = largest |area| in (projected) space; remaining loops = holes.
    uv_loops = []
    for pts in loop_pts_list:
        uv = _project_loop_uv(surf, pts)
        if uv is None or len(uv) < 3:
            continue
        uv_loops.append(uv)
    if not uv_loops:
        return None
    uv_loops.sort(key=lambda a: abs(_polygon_area_2d(a)), reverse=True)

    outer = uv_loops[0]
    if _polygon_area_2d(outer) < 0:
        outer = outer[::-1]
    outer_wire = _wire_from_uv(surf, outer)
    if outer_wire is None:
        return None

    try:
        mf = BRepBuilderAPI_MakeFace(surf, outer_wire, True)
        for hole in uv_loops[1:]:
            if _polygon_area_2d(hole) > 0:
                hole = hole[::-1]
            hw = _wire_from_uv(surf, hole)
            if hw is not None:
                mf.Add(hw)
        if not mf.IsDone():
            return None
        face = mf.Face()
        BRepLib.BuildCurves3d_s(face)
        if not BRepCheck_Analyzer(face).IsValid():
            return None
    except Exception:
        return None

    # Record fit deviation: max distance of region vertices to the surface.
    try:
        from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCP.gp import gp_Pnt
        sample = region_verts[:: max(1, len(region_verts) // 40)]
        for p in sample:
            proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)
            if proj.NbPoints() >= 1:
                devs.append(float(proj.LowerDistance()))
    except Exception:
        pass
    return face
