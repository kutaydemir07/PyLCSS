# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recovered triangle-mesh cleanup and smoothing."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from .analytic_shapes import AnalyticShape

logger = logging.getLogger(__name__)

#: Above this face count the handoff screening stops being free.
#:
#: Every check below was sized against a recovered *solid* envelope, which this
#: pipeline caps at a few hundred thousand triangles. An explicit lattice is a
#: different object: a 25% gyroid filling a 240,000 mm^3 block comes back at
#: 2.9 million triangles and genus ~16,800, because its surface area is the
#: point of it. Measured on that mesh, the screening cost 115 s against 15 s to
#: generate the geometry — the run looks hung, and nothing about the extra time
#: was informative.
#:
#: The checks are not silently dropped: the report says which ones ran over
#: what sample, so a thin screen never reads as a clean bill of health.
FULL_SCREENING_FACE_LIMIT = 750_000

def _screening_budget(face_count: int) -> dict[str, Any]:
    """Decide how much handoff screening this mesh size can afford.

    Returns the thickness sample count, whether the self-intersection scan and
    the coplanar cleanup run, and a plain-language note for the report.
    """
    faces = int(max(face_count, 0))
    if faces <= FULL_SCREENING_FACE_LIMIT:
        return {
            "thickness_samples": 500,
            "self_intersection": True,
            "coplanar_cleanup": True,
            "note": "full",
        }
    return {
        # Both scans reach the mesh through trimesh's r-tree over every face,
        # and building it costs 13-14 s at 2.9M triangles before a single ray
        # is cast or a single pair is tested. That cost is fixed, so a smaller
        # sample buys almost nothing and neither scan is worth it at this size.
        "thickness_samples": 0,
        "self_intersection": False,
        # The lossless coplanar collapse and the Delaunay flip after it both
        # target the uniform tessellation of *flat* regions. A periodic lattice
        # isosurface has none: measured on a 2.9M-triangle gyroid the pair spent
        # 21 s to remove 2 triangles and flip 17 edges.
        "coplanar_cleanup": False,
        "note": (
            f"reduced for a {faces:,}-triangle mesh, above the "
            f"{FULL_SCREENING_FACE_LIMIT:,} budgeted for full screening: wall "
            "thickness and self-intersection not screened, coplanar cleanup "
            "not run"
        ),
    }


def _topology_signature(mesh) -> tuple[int, int, int]:
    """Return component, Euler, and boundary-edge counts for a triangle mesh."""
    import trimesh

    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
    )
    edge_use = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=int),
        minlength=len(mesh.edges_unique),
    )
    return (
        len(components),
        int(mesh.euler_number),
        int(np.count_nonzero(edge_use == 1)),
    )


def _triangle_aspect_ratios(mesh) -> np.ndarray:
    """Longest edge over shortest altitude, per face.

    One is an equilateral triangle and large values are slivers. This is the
    quantity a downstream tetrahedral mesher chokes on, so it belongs in the
    handoff report even though it says nothing about the design itself.
    """
    triangles = np.asarray(mesh.triangles, dtype=float)
    if len(triangles) == 0:
        return np.empty(0, dtype=float)
    lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    longest = np.max(lengths, axis=1)
    minimum_altitude = (
        2.0 * np.asarray(mesh.area_faces, dtype=float) / np.maximum(longest, 1e-30)
    )
    return longest / np.maximum(minimum_altitude, 1e-30)


def _sampled_wall_thickness(
    mesh,
    sample_count: int,
) -> tuple[Optional[float], Optional[float]]:
    """Ray-sampled wall thickness at face centres.

    Reported as the minimum and 5th percentile so a single sliver cannot be
    mistaken for a systematic thin wall. This is a screening measurement on a
    finite sample of faces, not a medial-axis thickness field, and it needs a
    ray backend; both limits are stated in the returned warnings.
    """
    if sample_count <= 0 or len(mesh.faces) == 0 or not mesh.is_watertight:
        return None, None
    try:
        import trimesh

        count = min(int(sample_count), len(mesh.faces))
        indices = np.linspace(0, len(mesh.faces) - 1, count, dtype=np.int64)
        normals = np.asarray(mesh.face_normals, dtype=float)[indices]
        # Step inside the surface so the ray does not immediately re-hit the
        # face it started from.
        epsilon = max(float(mesh.scale), 1.0) * 1e-8
        points = np.asarray(mesh.triangles_center, dtype=float)[indices]
        points = points - epsilon * normals
        values = np.asarray(
            trimesh.proximity.thickness(
                mesh, points, normals=-normals, method="ray"
            ),
            dtype=float,
        )
    except Exception:
        logger.debug("Ray thickness sampling unavailable; skipping")
        return None, None
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None, None
    return float(np.min(values)), float(np.quantile(values, 0.05))


def _segments_cross_triangles(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Möller-Trumbore, restricted to the segment and the triangle interior.

    Returns a boolean per (segment, triangle) pair. Hits that land exactly on a
    shared vertex or edge are excluded by the strictly-inside margins, so two
    triangles that merely touch along their common edge do not register.
    """
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(directions, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    # A parallel segment either misses or is coplanar; coplanar overlap is left
    # to the other five edge tests of the pair.
    scale = np.maximum(
        np.linalg.norm(edge1, axis=1) * np.linalg.norm(pvec, axis=1),
        1e-30,
    )
    usable = np.abs(determinant) > 1e-9 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_det = np.where(usable, 1.0 / np.where(usable, determinant, 1.0), 0.0)
        tvec = origins - triangles[:, 0]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1)
        v = np.einsum("ij,ij->i", directions, qvec) * inv_det
        t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    margin = 1e-9
    return (
        usable
        & (u > margin)
        & (v > margin)
        & (u + v < 1.0 - margin)
        & (t > margin)
        & (t < 1.0 - margin)
    )


def _self_intersecting_faces(mesh, *, sample_limit: int = 40_000) -> Optional[int]:
    """Count faces that pass through another, non-adjacent face.

    Watertight, winding-consistent and two-manifold together still permit a
    surface that passes through itself, and that is the one defect this
    pipeline can actively create: Taubin fairing, the analytic plane and
    cylinder snapping and the final clip to the design bounds all move vertices
    without consulting the rest of the mesh, and a thin web is where they fold.
    Slicers and tetrahedral meshers reject the result, so it is worth naming
    before the user discovers it downstream.

    Broad phase is trimesh's r-tree of face bounding boxes; narrow phase is a
    segment-triangle test on the six edges of each candidate pair. Adjacent
    faces are excluded by their shared vertices rather than by geometry.
    Returns ``None`` when the r-tree backend is unavailable, and screens a
    uniform sample of faces on meshes above ``sample_limit`` rather than
    refusing to answer -- the same screening contract as the thickness sample.

    Two limitations are deliberate. Faces that merely touch are not counted, so
    disjoint bodies in a multibody result and bodies meeting along a shared
    face both read clean. The cost is that a pair whose crossing segment ends
    exactly on a shared boundary -- two identical boxes offset in-plane, say --
    is read as touching rather than crossing; that is an exactly-coplanar
    configuration a recovered isosurface does not produce, and separating it
    from genuine contact would cost far more false positives than it is worth.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) < 2:
        return 0
    try:
        tree = mesh.triangles_tree
    except Exception:
        logger.debug("Self-intersection broad phase unavailable.", exc_info=True)
        return None

    triangles = np.asarray(mesh.triangles, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    epsilon = float(np.ptp(bounds, axis=0).max()) * 1e-9
    if len(faces) > int(sample_limit):
        probe = np.linspace(0, len(faces) - 1, int(sample_limit), dtype=np.int64)
    else:
        probe = np.arange(len(faces), dtype=np.int64)

    offending: set[int] = set()
    for index in probe:
        triangle = triangles[index]
        box = np.concatenate(
            (triangle.min(axis=0) - epsilon, triangle.max(axis=0) + epsilon)
        )
        candidates = np.fromiter(tree.intersection(box), dtype=np.int64)
        if candidates.size == 0:
            continue
        # Drop the face itself and anything sharing a vertex with it: those
        # touch by construction and are not defects.
        shared = np.isin(faces[candidates], faces[index]).any(axis=1)
        candidates = candidates[~shared]
        if candidates.size == 0:
            continue
        others = triangles[candidates]
        this = np.broadcast_to(triangle, (len(candidates), 3, 3))
        hit = np.zeros(len(candidates), dtype=bool)
        for corner in range(3):
            nxt = (corner + 1) % 3
            # This face's edges against the candidates, then theirs against it.
            hit |= _segments_cross_triangles(
                this[:, corner],
                this[:, nxt] - this[:, corner],
                others,
            )
            hit |= _segments_cross_triangles(
                others[:, corner],
                others[:, nxt] - others[:, corner],
                this,
            )
        if np.any(hit):
            offending.add(int(index))
            offending.update(int(value) for value in candidates[hit])
    return int(len(offending))


def _surface_quality_report(
    mesh,
    *,
    thickness_samples: int = 500,
    screen_self_intersection: bool = True,
    screening_note: str = "full",
) -> dict[str, Any]:
    """Describe a recovered surface for CAD/FEA handoff.

    Watertightness alone does not tell a user whether the mesh can be remeshed,
    printed or trusted. Winding, non-manifold seams, sliver triangles, edge-length
    spread and sampled wall thickness each decide a different downstream step, so
    all of them are reported together with a plain-language warning list.

    The four keys the result panel and the study gate already read --
    ``connected_components``, ``euler_number``, ``open_boundary_edges`` and
    ``watertight`` -- keep their names and meaning.

    ``thickness_samples`` and ``screen_self_intersection`` come from
    :func:`_screening_budget` on large meshes. What was actually screened is
    reported in ``screening_scope`` and in the warnings, because a check that
    did not run must never be read as a check that passed.
    """
    components, euler_number, boundary_edges = _topology_signature(mesh)

    edge_use = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=int),
        minlength=len(mesh.edges_unique),
    )
    nonmanifold_edges = int(np.count_nonzero(edge_use > 2))
    degenerate_faces = int(
        len(mesh.faces) - np.count_nonzero(mesh.nondegenerate_faces())
    )
    aspect = _triangle_aspect_ratios(mesh)
    edge_lengths = np.asarray(mesh.edges_unique_length, dtype=float)
    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    thickness_min, thickness_p05 = _sampled_wall_thickness(mesh, thickness_samples)
    self_intersecting_faces = (
        _self_intersecting_faces(mesh) if screen_self_intersection else None
    )

    warnings: list[str] = []
    if not watertight:
        warnings.append(
            f"Surface is not watertight ({boundary_edges} open boundary edges); "
            "close it before tetrahedral remeshing or printing."
        )
    if not winding_consistent:
        warnings.append(
            "Face winding is inconsistent, so the inside/outside test is "
            "unreliable and the enclosed volume may be wrong."
        )
    if nonmanifold_edges:
        warnings.append(
            f"{nonmanifold_edges} non-manifold edges: more than two faces share "
            "an edge, which most meshers and slicers reject."
        )
    if degenerate_faces:
        warnings.append(f"{degenerate_faces} zero-area faces remain.")
    if components > 1:
        warnings.append(
            f"{components} disconnected bodies. This can be intentional in a "
            "multibody result; confirm none of them is floating material."
        )
    if aspect.size and float(np.quantile(aspect, 0.95)) > 20.0:
        warnings.append(
            f"Sliver triangles present (95th percentile aspect ratio "
            f"{float(np.quantile(aspect, 0.95)):.1f}); remesh before analysis."
        )
    if thickness_min is None:
        warnings.append(
            "Wall thickness was not sampled "
            + (
                f"(this mesh has {len(mesh.faces):,} triangles, above the "
                f"{FULL_SCREENING_FACE_LIMIT:,} the ray sample is budgeted for)."
                if int(thickness_samples) <= 0
                else "; it needs a watertight mesh and a ray backend."
            )
            + " It is a screening sample rather than a medial-axis thickness "
            "field either way."
        )
    if self_intersecting_faces is None:
        reason = (
            f"this mesh has {len(mesh.faces):,} triangles, above the "
            f"{FULL_SCREENING_FACE_LIMIT:,} the scan is budgeted for"
            if not screen_self_intersection
            else "the spatial-index backend is unavailable"
        )
        warnings.append(
            f"Self-intersection was not screened ({reason}). Watertightness "
            "alone does not rule out a surface that passes through itself; "
            "check it in the slicer or mesher before release."
        )
    elif self_intersecting_faces:
        warnings.append(
            f"{self_intersecting_faces} self-intersecting faces: the surface "
            "passes through itself, which slicers and tetrahedral meshers "
            "reject even though the mesh is closed."
        )

    return {
        # Keys consumed by the study gate and the result panel.
        "connected_components": components,
        "euler_number": euler_number,
        "open_boundary_edges": boundary_edges,
        "watertight": watertight,
        "intentional_openings_preserved": True,
        # Handoff detail.
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": degenerate_faces,
        # None means "not screened", which is not the same as zero.
        "self_intersecting_faces": self_intersecting_faces,
        "winding_consistent": winding_consistent,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "enclosed_volume": float(abs(mesh.volume)) if watertight else None,
        "estimated_genus": (
            float((2 * components - euler_number) / 2.0) if watertight else None
        ),
        "minimum_edge_length": (
            float(edge_lengths.min()) if edge_lengths.size else 0.0
        ),
        "maximum_edge_length": (
            float(edge_lengths.max()) if edge_lengths.size else 0.0
        ),
        "median_triangle_aspect_ratio": (
            float(np.median(aspect)) if aspect.size else 0.0
        ),
        "p95_triangle_aspect_ratio": (
            float(np.quantile(aspect, 0.95)) if aspect.size else 0.0
        ),
        "sampled_minimum_thickness": thickness_min,
        "sampled_p05_thickness": thickness_p05,
        # How thorough the screening above was. `full` means every check ran at
        # its normal sample; anything else names what was reduced or skipped.
        "screening_scope": str(screening_note),
        "thickness_sample_count": int(thickness_samples),
        "warnings": tuple(warnings),
    }


def _collapse_degenerate_slivers(mesh, *, area_tolerance: float = 1e-10) -> Any:
    """Collapse zero-area triangles instead of deleting them.

    ``_project_extruded_planes`` and the clip-to-bounds step both pull vertices
    onto an exact plane. Where a swept wall meets an end cap that leaves a few
    triangles exactly *collinear* -- not coincident, so no weld tolerance can
    merge them. Deleting a collinear sliver from a closed shell punches a hole:
    on the bundled 96x48x8 cantilever, 8 such slivers were being removed and
    took watertightness with them, which left the enclosed volume unreportable
    and failed STEP export on its own 2-manifold check.

    The standard repair is an edge collapse rather than a deletion -- collapse
    the shortest edge of the sliver, so the triangle disappears and its two
    neighbours become adjacent. Applied here with a union-find so several
    collapses sharing a vertex compose correctly, then validated: the surface
    must keep its topology signature and its enclosed volume, or the original
    mesh is returned untouched.
    """
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh

    # Taken lazily, below, once a sliver is known to exist. A topology signature
    # walks the whole face-adjacency graph -- 2.5 s on a 2.9M-triangle lattice --
    # and the common case is that there is nothing to collapse and nothing to
    # validate against.
    signature_before: Optional[tuple[int, int, int]] = None
    volume_before: Optional[float] = None
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    collapses = 0
    working = faces
    # A collapse can flatten a neighbour, so iterate; in practice this settles
    # after one or two passes.
    for _ in range(4):
        triangles = vertices[working]
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        distinct = (
            (working[:, 0] != working[:, 1])
            & (working[:, 1] != working[:, 2])
            & (working[:, 0] != working[:, 2])
        )
        flat = np.flatnonzero((areas <= float(area_tolerance)) & distinct)
        if flat.size == 0:
            break
        if signature_before is None:
            signature_before = _topology_signature(mesh)
            volume_before = (
                float(abs(mesh.volume)) if mesh.is_watertight else None
            )
        for face_index in flat:
            corners = working[face_index]
            pairs = ((corners[0], corners[1]), (corners[1], corners[2]),
                     (corners[2], corners[0]))
            lengths = [
                float(np.linalg.norm(vertices[u] - vertices[v])) for u, v in pairs
            ]
            u, v = pairs[int(np.argmin(lengths))]
            root_u, root_v = find(int(u)), find(int(v))
            if root_u != root_v:
                parent[max(root_u, root_v)] = min(root_u, root_v)
                collapses += 1
        # Resolve the forest to roots without a Python loop over every vertex.
        roots = parent.copy()
        while True:
            lifted = roots[roots]
            if np.array_equal(lifted, roots):
                break
            roots = lifted
        working = roots[faces]

    if collapses == 0:
        return mesh

    keep = (
        (working[:, 0] != working[:, 1])
        & (working[:, 1] != working[:, 2])
        & (working[:, 0] != working[:, 2])
    )
    working = working[keep]
    if len(working) < 4:
        return mesh
    used = np.unique(working)
    remap = np.zeros(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    collapsed = trimesh.Trimesh(
        vertices=vertices[used],
        faces=remap[working],
        process=False,
    )

    if _topology_signature(collapsed) != signature_before:
        logger.info("Sliver collapse rejected: it changed the surface topology.")
        return mesh
    if volume_before is not None:
        volume_after = float(abs(collapsed.volume)) if collapsed.is_watertight else None
        # A collapse moves one endpoint of a sliver's shortest edge, so the
        # volume it can shift is bounded by that sub-voxel length. Measured
        # across the quality presets and both extraction backends the worst case
        # is 4.9e-7 relative, from 3481 collapses on a Standard-preset grid --
        # a tighter bound than this rejects a correct repair, while anything
        # that actually damaged a load path would move the volume far more.
        if volume_after is None or not np.isclose(
            volume_after, volume_before, rtol=1e-6, atol=0.0
        ):
            logger.info("Sliver collapse rejected: it changed the enclosed volume.")
            return mesh

    logger.info(
        "Collapsed %d zero-area sliver(s) at plane-snapped edges (%d to %d faces).",
        collapses,
        len(faces),
        len(collapsed.faces),
    )
    return collapsed


def _flip_coplanar_edges_to_delaunay(
    mesh,
    *,
    rounds: int = 24,
    # How far two normals may disagree and still be treated as one plane, as
    # ``1 - cos(angle)``. Deliberately near the numerical floor. The benefit on
    # flat regions saturates here -- the cap reaches 9.18/18.92 at 1e-12 and
    # does not improve at 1e-4 -- while looser gates only add flips across real
    # dihedrals, which move the surface: 1e-4 shifts the enclosed volume by
    # 1.4e-5 relative for no gain the flat regions did not already have.
    coplanar_tolerance: float = 1e-10,
) -> tuple[Any, int]:
    """Restore triangle shape on flat regions by Delaunay edge flipping.

    :func:`_collapse_redundant_coplanar_faces` removes the triangles that
    describe nothing, but it chooses collapses by quadric error alone and has no
    reason to leave a well-shaped triangulation behind. On the reference
    cantilever the flat end caps come out at a median aspect ratio of 11.3 with
    a 95th percentile of 28 -- geometrically exact, since every one of those
    triangles lies in the cap plane, but poor input for anything that later
    remeshes the surface.

    Flipping the shared edge of two *coplanar* triangles is the one repair that
    costs nothing: it changes connectivity only, so the enclosed volume and the
    surface area are preserved to the bit, and the Delaunay in-circle criterion
    maximizes the minimum angle over the triangulation it converges to. The
    coplanarity gate is what makes it safe -- flipping across a real dihedral
    would move the surface.

    Measured on the cantilever's flat regions: median 11.12 -> 9.18 and p95
    29.73 -> 18.92 over 3678 flips, with the surface area unchanged to 1.5e-15
    relative and the enclosed volume to 3.5e-11.

    The remaining gap to a well-shaped mesh is point distribution, not
    connectivity: only 9.2% of the cap's vertices are interior, and no amount of
    flipping fixes that. Closing it needs Delaunay refinement with Steiner
    points, which among the available libraries means packages whose licensing
    does not fit this distribution. The permissively licensed option, poly2tri,
    is constrained Delaunay without refinement -- it converges to the same
    triangulation this function already reaches for free.
    """
    original = mesh
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(faces) < 4:
        return mesh, 0

    signature_before = _topology_signature(mesh)
    area_before = float(mesh.area)
    volume_before = float(abs(mesh.volume)) if mesh.is_watertight else None

    total = 0
    for _ in range(max(1, int(rounds))):
        corners = np.stack(
            [
                np.stack([faces[:, k], faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]])
                for k in range(3)
            ],
            axis=0,
        )
        starts = np.concatenate([corners[k][0] for k in range(3)])
        ends = np.concatenate([corners[k][1] for k in range(3)])
        opposite = np.concatenate([corners[k][2] for k in range(3)])
        owner = np.tile(np.arange(len(faces), dtype=np.int64), 3)

        low = np.minimum(starts, ends)
        high = np.maximum(starts, ends)
        order = np.lexsort((high, low))
        low, high = low[order], high[order]
        owner, opposite = owner[order], opposite[order]
        paired = np.flatnonzero((low[:-1] == low[1:]) & (high[:-1] == high[1:]))
        if paired.size == 0:
            break
        # A non-manifold edge appears three or more times; skip the whole run.
        manifold = np.ones(paired.size, dtype=bool)
        manifold[:-1] &= paired[1:] != paired[:-1] + 1
        manifold[1:] &= paired[:-1] != paired[1:] - 1
        paired = paired[manifold]
        if paired.size == 0:
            break

        face_a, face_b = owner[paired], owner[paired + 1]
        far_a, far_b = opposite[paired], opposite[paired + 1]
        shared_lo, shared_hi = low[paired], high[paired]

        normals = np.asarray(mesh.face_normals, dtype=float)
        normal_a, normal_b = normals[face_a], normals[face_b]
        coplanar = (
            np.sum(normal_a * normal_b, axis=1) >= 1.0 - float(coplanar_tolerance)
        )
        if not np.any(coplanar):
            break
        idx = np.flatnonzero(coplanar)

        # Project the quad onto its shared plane and test in 2-D.
        normal = normal_a[idx]
        helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(idx), 1))
        alt = np.abs(np.sum(normal * helper, axis=1)) > 0.9
        helper[alt] = np.array([1.0, 0.0, 0.0])
        axis_u = np.cross(normal, helper)
        axis_u /= np.maximum(np.linalg.norm(axis_u, axis=1)[:, None], 1e-30)
        axis_v = np.cross(normal, axis_u)

        def flatten(point_ids: np.ndarray) -> np.ndarray:
            points = vertices[point_ids]
            return np.stack(
                (np.sum(points * axis_u, axis=1), np.sum(points * axis_v, axis=1)),
                axis=1,
            )

        p_a = flatten(shared_lo[idx])
        p_b = flatten(far_a[idx])
        p_c = flatten(shared_hi[idx])
        p_d = flatten(far_b[idx])

        def side(p, q, r):
            return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (
                q[:, 1] - p[:, 1]
            ) * (r[:, 0] - p[:, 0])

        convex = (side(p_a, p_b, p_c) * side(p_a, p_d, p_c) < 0.0) & (
            side(p_b, p_c, p_d) * side(p_b, p_a, p_d) < 0.0
        )

        # In-circle needs a counter-clockwise reference triangle.
        turn = side(p_a, p_b, p_c)
        first = np.where((turn < 0.0)[:, None], p_c, p_a)
        third = np.where((turn < 0.0)[:, None], p_a, p_c)
        ax_, ay_ = first[:, 0] - p_d[:, 0], first[:, 1] - p_d[:, 1]
        bx_, by_ = p_b[:, 0] - p_d[:, 0], p_b[:, 1] - p_d[:, 1]
        cx_, cy_ = third[:, 0] - p_d[:, 0], third[:, 1] - p_d[:, 1]
        in_circle = (
            (ax_ * ax_ + ay_ * ay_) * (bx_ * cy_ - cx_ * by_)
            - (bx_ * bx_ + by_ * by_) * (ax_ * cy_ - cx_ * ay_)
            + (cx_ * cx_ + cy_ * cy_) * (ax_ * by_ - bx_ * ay_)
        ) > 1e-12

        candidates = idx[convex & in_circle]
        if candidates.size == 0:
            break

        # One flip per face per round; a face touched twice would be rebuilt
        # from a neighbour that no longer exists.
        touched = np.zeros(len(faces), dtype=bool)
        chosen: list[int] = []
        for candidate in candidates:
            fa, fb = int(face_a[candidate]), int(face_b[candidate])
            if touched[fa] or touched[fb]:
                continue
            touched[fa] = touched[fb] = True
            chosen.append(int(candidate))
        if not chosen:
            break

        picked = np.asarray(chosen, dtype=np.int64)
        patch_normal = normal_a[picked]
        replacements = (
            np.stack((far_a[picked], far_b[picked], shared_hi[picked]), axis=1),
            np.stack((far_b[picked], far_a[picked], shared_lo[picked]), axis=1),
        )
        # Both halves of a planar quad must keep facing the way the patch did.
        # Deriving that from the original corner order instead would need the
        # incident faces' winding, which this edge table does not carry.
        for replacement in replacements:
            corners_xyz = vertices[replacement]
            candidate_normal = np.cross(
                corners_xyz[:, 1] - corners_xyz[:, 0],
                corners_xyz[:, 2] - corners_xyz[:, 0],
            )
            inverted = np.sum(candidate_normal * patch_normal, axis=1) < 0.0
            replacement[inverted] = replacement[inverted][:, ::-1]
        faces[face_a[picked]] = replacements[0]
        faces[face_b[picked]] = replacements[1]
        total += len(picked)

        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if total == 0:
        return mesh, 0

    # Flips are exact in-plane operations, so anything that moved is a bug.
    signature_after = _topology_signature(mesh)
    area_after = float(mesh.area)
    closed_after = bool(mesh.is_watertight)
    volume_after = float(abs(mesh.volume)) if closed_after else None
    if (
        signature_after != signature_before
        or not np.isclose(area_after, area_before, rtol=1e-8, atol=0.0)
        or (
            volume_before is not None
            and (
                volume_after is None
                or not np.isclose(volume_after, volume_before, rtol=1e-8, atol=0.0)
            )
        )
    ):
        logger.info(
            "Coplanar edge flipping rejected after %d flip(s): topology %s -> %s, "
            "area %.10g -> %.10g, volume %s -> %s.",
            total,
            signature_before,
            signature_after,
            area_before,
            area_after,
            f"{volume_before:.10g}" if volume_before is not None else "open",
            f"{volume_after:.10g}" if volume_after is not None else "open",
        )
        return original, 0

    logger.info(
        "Delaunay-flipped %d coplanar edge(s) to improve triangle shape.", total
    )
    return mesh, total


def _collapse_redundant_coplanar_faces(mesh) -> tuple[Any, int]:
    """Drop isosurface triangles that carry no shape information.

    Flying Edges tessellates at a uniform grid density, so a flat region is
    subdivided as finely as a sharply curved one. On an extrusion-constrained
    result the two end caps are exactly planar, and at the in-plane recovery
    resolution this pipeline uses they dominate the triangle budget while
    describing two polygons: on the reference 80x40x8 cantilever they are
    259,813 of 284,967 triangles -- 91% of the mesh for 40% of the surface area.
    That redundancy costs viewer memory, and it costs far more downstream, where
    CAD reconstruction runs per-triangle edge-adjacency graphs and either sews
    one B-rep face per triangle or intersects the whole soup for a section.

    ``fast_simplification``'s lossless mode collapses only edges whose quadric
    error sits at the numerical floor, so it removes that redundancy without
    moving the surface. On the same cantilever it returns 28,301 triangles in
    0.33 s with an unchanged enclosed volume and a peak deviation of 1e-6 model
    units -- roughly three millionths of a recovery voxel, six orders of
    magnitude inside any tolerance the downstream stages apply.

    This is deliberately not the user-facing ``decimate_ratio`` control and does
    not replace it. That one trades geometric detail for file size and stays
    opt-in; this one only deletes triangles that describe nothing, so gating it
    behind a preference would just mean shipping a slower result for no gain.

    Returns the reduced mesh and the number of triangles removed. The caller's
    topology signature guard still applies to the result.
    """
    try:
        import fast_simplification
        import trimesh
    except ImportError:
        logger.info(
            "Coplanar triangle reduction skipped — install `fast_simplification` "
            "to shrink flat isosurface regions before CAD reconstruction."
        )
        return mesh, 0

    original_count = len(mesh.faces)
    if original_count < 64:
        return mesh, 0

    try:
        topology_before = _topology_signature(mesh)
        vertices, faces = fast_simplification.simplify(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            lossless=True,
        )
        if len(faces) == 0 or len(faces) >= original_count:
            return mesh, 0
        reduced = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        if _topology_signature(reduced) != topology_before:
            # Should not happen for a zero-error collapse, but a silent topology
            # change here would propagate into the CAD body and the study gate.
            logger.info(
                "Coplanar triangle reduction rejected because it changed "
                "connected components, genus, or open boundaries."
            )
            return mesh, 0
    except Exception:
        logger.debug("Coplanar triangle reduction failed; keeping full mesh")
        return mesh, 0

    removed = original_count - len(reduced.faces)
    logger.info(
        "Recovered surface: removed %d redundant coplanar triangles "
        "(%d to %d, %.1f%%).",
        removed,
        original_count,
        len(reduced.faces),
        100.0 * removed / max(original_count, 1),
    )
    return reduced, removed


def _enhanced_mesh_postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    decimate_ratio: float = 1.0,
    smoothing_iterations: int = 2,
    preserve_components: bool = False,
) -> Optional[dict[str, Any]]:
    """Run a topology-preserving surface post-processing pipeline.

    Pipeline:
        1. Remove duplicate/degenerate faces and unreferenced vertices.
        2. Light Humphrey smoothing (volume-preserving alternative to Laplacian).
        3. Lossless collapse of redundant coplanar triangles.
        4. Delaunay edge flips on flat regions to restore triangle shape.
        5. Optional quadric decimation (requires `fast_simplification`).

    Returns None if trimesh is unavailable so the caller can fall back to the
    legacy Taubin path.

    Deliberately do not fill holes or discard resolved connected components
    here. Both operations can erase intentional passive voids, powder escape
    holes, thin members, or legitimate bodies in a multibody result. The sole
    exception, unless ``preserve_components`` is requested by cutoff-preserving
    solid recovery, is conservative cleanup of sub-resolution numerical residue:
    closed islands no larger than 64 triangles and less than 0.001% of total
    area, plus a single triangle/quad contouring hole (at most eight boundary
    edges in the whole mesh). Neither represents a manufacturable feature.
    """
    try:
        import trimesh
        import trimesh.smoothing
    except ImportError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    # 1. Safe cleanup only; preserve the topology produced by recovery.
    try:
        mesh.update_faces(mesh.unique_faces())
        # Plane snapping leaves collinear slivers; collapse them rather than
        # letting the removal below delete them and open the shell.
        mesh = _collapse_degenerate_slivers(mesh)
        # Whatever the collapse could not resolve is still only safe to delete
        # if the surface survives it. A zero-area triangle inside a closed shell
        # costs nothing downstream -- it encloses no volume, OCC sews across it,
        # and the quality report counts it under `degenerate_faces`. Losing
        # watertightness costs the user the CAD body outright.
        closed_before = bool(mesh.is_watertight)
        # Flying Edges can emit coincident coordinates under distinct point
        # IDs where contour blocks meet. They appear as paired open edges even
        # though the geometric surface is closed. Weld exact/quantized
        # duplicates before judging watertightness.
        candidate = mesh.copy()
        candidate.update_faces(candidate.nondegenerate_faces())
        candidate.merge_vertices()
        candidate.remove_unreferenced_vertices()
        if closed_before and not candidate.is_watertight:
            logger.info(
                "Kept %d zero-area triangle(s): removing them would open the "
                "closed recovered surface.",
                len(mesh.faces) - len(candidate.faces),
            )
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
        else:
            mesh = candidate

    except Exception:
        logger.debug("Topology-preserving mesh cleanup failed; continuing")

    # Remove only sub-resolution contouring specks. A resolved engineering
    # component remains in the mesh and is judged by the downstream gate.
    removed_numerical_fragments = 0
    if not preserve_components:
        try:
            components = trimesh.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces)),
            )
            total_area = max(float(mesh.area), 1e-30)
            keep = np.ones(len(mesh.faces), dtype=bool)
            for component in components:
                face_ids = np.asarray(component, dtype=int)
                component_area = float(np.sum(mesh.area_faces[face_ids]))
                if (
                    len(face_ids) <= 64
                    and component_area / total_area < 1e-5
                ):
                    keep[face_ids] = False
                    removed_numerical_fragments += 1
            if removed_numerical_fragments and np.any(keep):
                mesh.update_faces(keep)
                mesh.remove_unreferenced_vertices()
        except Exception:
            logger.debug("Numerical-fragment cleanup failed; continuing")

    filled_numerical_holes = 0
    try:
        edge_use = np.bincount(
            np.asarray(mesh.edges_unique_inverse, dtype=int),
            minlength=len(mesh.edges_unique),
        )
        open_edges = int(np.count_nonzero(edge_use == 1))
        if 0 < open_edges <= 8 and trimesh.repair.fill_holes(mesh):
            filled_numerical_holes = 1
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
    except Exception:
        logger.debug("Numerical-hole cleanup failed; continuing")

    # 2. Volume-preserving Humphrey smoothing.
    if int(smoothing_iterations) > 0:
        try:
            trimesh.smoothing.filter_humphrey(
                mesh,
                alpha=0.1,
                beta=0.5,
                iterations=int(smoothing_iterations),
            )
        except Exception:
            logger.debug("Humphrey smoothing failed; falling back to Taubin")
            try:
                trimesh.smoothing.filter_taubin(
                    mesh, iterations=int(smoothing_iterations)
                )
            except Exception:
                pass

    # 3. Remove the uniform-density tessellation of flat regions, then recover
    # triangle shape on those regions where a flip is free. Both run before any
    # lossy decimation so that a requested ratio is spent on the triangles that
    # actually describe curvature.
    #
    # Neither is free on a surface with no flat regions to fix: each costs full
    # topology signatures over the whole mesh, and on a 2.9M-triangle gyroid the
    # pair spent 21 s to remove 2 triangles and flip 17 edges. Budget them by
    # size, like the screening below.
    removed_coplanar_faces = 0
    flipped_coplanar_edges = 0
    if _screening_budget(len(mesh.faces))["coplanar_cleanup"]:
        mesh, removed_coplanar_faces = _collapse_redundant_coplanar_faces(mesh)
        mesh, flipped_coplanar_edges = _flip_coplanar_edges_to_delaunay(mesh)
    else:
        logger.info(
            "Coplanar triangle cleanup skipped on a %d-triangle surface: both "
            "passes only act on flat regions, which a periodic lattice "
            "isosurface does not have.",
            len(mesh.faces),
        )

    # 5. Optional decimation (requires `fast_simplification`).
    if 0.0 < float(decimate_ratio) < 1.0 and len(mesh.faces) > 0:
        target = max(64, int(len(mesh.faces) * float(decimate_ratio)))
        try:
            topology_before = _topology_signature(mesh)
            decimated = mesh.simplify_quadric_decimation(face_count=target)
            if (
                decimated is not None
                and len(decimated.faces) > 0
                and _topology_signature(decimated) == topology_before
            ):
                mesh = decimated
            elif decimated is not None:
                logger.info(
                    "STL decimation rejected because it changed connected "
                    "components, holes, or open boundaries."
                )
        except ImportError:
            logger.info(
                "STL decimation skipped — install `fast_simplification` for "
                "quadric decimation support."
            )
        except Exception:
            logger.debug("Decimation failed; keeping un-decimated mesh")

    # Decimation can move the mesh across a budget boundary, so re-ask.
    budget = _screening_budget(len(mesh.faces))
    surface_quality = _surface_quality_report(
        mesh,
        thickness_samples=budget["thickness_samples"],
        screen_self_intersection=budget["self_intersection"],
        screening_note=budget["note"],
    )
    surface_quality["removed_numerical_fragments"] = (
        removed_numerical_fragments
    )
    surface_quality["filled_numerical_holes"] = filled_numerical_holes
    surface_quality["removed_coplanar_faces"] = removed_coplanar_faces
    surface_quality["flipped_coplanar_edges"] = flipped_coplanar_edges
    return {
        "vertices": np.asarray(mesh.vertices, dtype=float),
        "faces": np.asarray(mesh.faces, dtype=int),
        "surface_quality": surface_quality,
    }


def _undirected_edges(
    verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate the triangle-soup edges and count how many faces use each.

    Encoding an edge as a single ``lo * n + hi`` integer keeps this on the fast
    one-dimensional ``np.unique`` path and avoids the axis-dependent
    ``return_inverse`` shape that changed between NumPy 1.x and 2.x.
    """
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    count = np.int64(max(len(verts), 1))
    low = np.minimum(directed[:, 0], directed[:, 1])
    high = np.maximum(directed[:, 0], directed[:, 1])
    unique_keys, counts = np.unique(low * count + high, return_counts=True)
    edges = np.stack((unique_keys // count, unique_keys % count), axis=1)
    return edges, counts


def _locked_surface_vertices(
    verts: np.ndarray,
    edges: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Mark vertices on an open boundary or a non-manifold seam.

    An edge used by anything other than exactly two faces bounds a hole or a
    seam.  Under an unconstrained umbrella operator those loops migrate inwards,
    because a boundary vertex has neighbours on one side only, so a surface cut
    by the design-domain box curls away from the cut plane and loses material.
    Holding the ring fixed keeps the cut flat; measured on a 111k-vertex recovery
    mesh, it turns a 0.3-0.8% volume loss into 0.0-0.1%.

    Deliberately no dihedral-angle feature locking.  It is the standard companion
    to this operator, but it does no work on this pipeline's input: the recovery
    grid is upsampled and Gaussian-faired before extraction, which caps the
    dihedral angle at roughly 41 degrees, so any threshold high enough to skip
    voxel terracing (which sits at exactly 45 degrees) selects nothing at all.
    Sharp engineering features are restored afterwards and to their exact
    prescribed size by the analytic-shape and extruded-plane projections, which
    is a stronger guarantee than freezing whatever the isosurface produced.
    """
    locked = np.zeros(len(verts), dtype=bool)
    abnormal = counts != 2
    if np.any(abnormal):
        locked[np.unique(edges[abnormal])] = True
    return locked


def _taubin_smooth_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 6,
    shapes: Sequence[AnalyticShape] = (),
    tolerance: float = 0.0,
    *,
    lock_open_boundaries: bool = True,
    maximum_step: Optional[float] = None,
) -> np.ndarray:
    """Volume-preserving smoothing for isosurface output.

    Taubin's alternating positive/negative umbrella passes remove voxel
    terracing with far less shrinkage than repeated Laplacian smoothing, but on
    their own they still migrate open boundaries and can thin a narrow load path.
    Three constraints bound that damage:

    * Open-boundary and non-manifold vertices are held at their extracted
      positions (see :func:`_locked_surface_vertices`).
    * ``maximum_step`` clamps the per-pass umbrella vector in model units, so no
      single vertex can be dragged across a thin member however many iterations
      run.  On a well-resolved recovery grid the raw vectors are already well
      inside a sub-voxel clamp, so this is insurance against a coarse or badly
      conditioned mesh rather than an everyday control.
    * Near an analytic passive shape the motion is projected onto the shape's
      tangent plane, so bores and interface pads keep their prescribed size.

    Connectivity is never modified; only vertex positions move.
    """
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts

    from scipy.sparse import coo_matrix

    points = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=np.int64)
    original = points.copy()
    n_vertices = len(points)

    edges, counts = _undirected_edges(points, faces)
    locked = (
        _locked_surface_vertices(points, edges, counts)
        if lock_open_boundaries
        else np.zeros(n_vertices, dtype=bool)
    )

    # Normalized uniform umbrella operator built once from the deduplicated
    # edges. The previous `np.add.at` formulation ran per pass and double
    # weighted every interior edge, because a shared edge appears once per
    # incident face in the raw triangle soup.
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(rows.size, dtype=float), (rows, cols)),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()
    inverse_degree = np.divide(
        1.0, degree, out=np.zeros_like(degree), where=degree > 0.0
    )
    has_neighbours = degree > 0.0
    movable = has_neighbours & ~locked
    if not np.any(movable):
        return points

    limit = None if maximum_step is None else float(maximum_step)
    if limit is not None and limit <= 0.0:
        limit = None

    for _ in range(max(0, int(iterations))):
        for factor in (0.5, -0.53):
            displacement = np.zeros_like(points)
            neighbour_mean = inverse_degree[:, None] * (adjacency @ points)
            displacement[has_neighbours] = (
                neighbour_mean[has_neighbours] - points[has_neighbours]
            )

            if limit is not None:
                magnitude = np.linalg.norm(displacement, axis=1)
                scale = np.minimum(1.0, limit / np.maximum(magnitude, 1e-30))
                displacement *= scale[:, None]

            if shapes and tolerance > 0.0:
                for shape in shapes:
                    distance = shape.sdf(points)
                    near = np.abs(distance) <= tolerance
                    if np.any(near):
                        normal = shape.get_normal(points[near])
                        tangential = displacement[near]
                        along = np.sum(tangential * normal, axis=-1, keepdims=True)
                        displacement[near] = tangential - along * normal

            points[movable] += factor * displacement[movable]

    # Numerical drift cannot be allowed to accumulate on a locked vertex.
    points[locked] = original[locked]
    return points
