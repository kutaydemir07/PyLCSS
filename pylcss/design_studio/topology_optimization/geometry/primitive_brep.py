# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Turn recognized planar regions into real OpenCASCADE planar faces.

:mod:`.surface_segmentation` decides *which* triangles form a flat face and
what plane they lie on. This module is the step that makes that a CAD face
rather than an annotation: it walks the region's free edges into ordered
loops, tells the outer boundary from the holes by signed area in the plane,
and builds a bounded planar face.

The face is genuinely a ``Plane`` in the B-rep, not a spline that happens to
be flat. That is the whole point -- a column pad has to arrive downstream as a
plane so it can be machined, mated and dimensioned as one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .surface_segmentation import PlanarRegion, snap_surface_to_planar_regions

logger = logging.getLogger(__name__)

__all__ = [
    "PlanarFaceBuild",
    "build_hybrid_solid",
    "build_planar_region_face",
    "region_boundary_loops",
]


@dataclass(frozen=True)
class PlanarFaceBuild:
    """One planar face plus the measurements that show it is faithful."""

    face: Any
    outer_loop: np.ndarray
    hole_loops: list[np.ndarray]
    face_area: float
    triangle_area: float

    @property
    def area_error(self) -> float:
        """Relative area difference against the triangles the face replaces."""
        if self.triangle_area <= 0.0:
            return 0.0
        return abs(self.face_area - self.triangle_area) / self.triangle_area


def region_boundary_loops(
    faces: Sequence[Sequence[int]],
    face_indices: Sequence[int],
    *,
    plane_points: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return each closed boundary loop of a triangle region, as vertex ids.

    An edge interior to the region is used by two of its triangles; an edge on
    its rim is used by one. Chaining those rim edges gives the region outline,
    and a region with a hole in it simply yields more than one chain.

    A region's rim can pass through the same vertex twice -- a waist, or two
    lobes joined at a point. Picking an arbitrary continuation there produces a
    figure of eight whose lobes have opposite signed area and cancel: on the
    press crown that silently emptied 21 of 57 faces. ``plane_points`` supplies
    the in-plane coordinates so the walk can instead take the next edge in
    angular order around the junction, which is the standard planar-embedding
    traversal and can only close simple loops.
    """
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    member = np.asarray(face_indices, dtype=np.int64)
    if len(member) == 0:
        return []

    triangle = faces[member]
    directed = np.concatenate(
        [triangle[:, [0, 1]], triangle[:, [1, 2]], triangle[:, [2, 0]]],
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _unique, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    rim = counts[inverse] == 1
    rim_edges = directed[rim]
    if len(rim_edges) == 0:
        return []

    # Keep the winding the triangles already carry: following it means the
    # loop comes out oriented consistently with the region's outward normal.
    successor: dict[int, list[int]] = {}
    for start, end in rim_edges:
        successor.setdefault(int(start), []).append(int(end))

    def _choose(current: int, previous: int | None, options: list[int]) -> int:
        if len(options) == 1 or plane_points is None or previous is None:
            return options[-1]
        here = plane_points[current]
        back = plane_points[previous] - here
        reference = math.atan2(float(back[1]), float(back[0]))
        best, best_turn = options[-1], math.inf
        for candidate in options:
            direction = plane_points[candidate] - here
            turn = (
                math.atan2(float(direction[1]), float(direction[0])) - reference
            ) % (2.0 * math.pi)
            if turn <= 1.0e-12:
                turn = 2.0 * math.pi
            if turn < best_turn:
                best, best_turn = candidate, turn
        return best

    loops: list[np.ndarray] = []
    for origin in list(successor):
        while successor.get(origin):
            loop = [origin]
            current = origin
            previous: int | None = None
            while True:
                options = successor.get(current)
                if not options:
                    loop = []
                    break
                nxt = _choose(current, previous, options)
                options.remove(nxt)
                if not options:
                    successor.pop(current, None)
                if nxt == origin:
                    break
                if nxt in loop:
                    # Still possible on a non-manifold rim: cut the closed part
                    # free and carry on with the remainder.
                    cut = loop.index(nxt)
                    loops.append(np.asarray(loop[cut:], dtype=np.int64))
                    loop = loop[:cut]
                    if not loop:
                        break
                    previous = loop[-2] if len(loop) > 1 else None
                    current = loop[-1]
                    continue
                loop.append(nxt)
                previous, current = current, nxt
            if len(loop) >= 3:
                loops.append(np.asarray(loop, dtype=np.int64))
    return [loop for loop in loops if len(loop) >= 3]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(normal[2])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    axis_u = np.cross(normal, helper)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _signed_area(points_2d: np.ndarray) -> float:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _make_wire(points: np.ndarray):
    """Closed polygonal wire through the given 3-D points."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt

    builder = BRepBuilderAPI_MakePolygon()
    for point in points:
        builder.Add(gp_Pnt(float(point[0]), float(point[1]), float(point[2])))
    builder.Close()
    if not builder.IsDone():
        raise RuntimeError("Region boundary did not close into a wire.")
    return builder.Wire()


def build_planar_region_face(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    region: PlanarRegion,
    *,
    minimum_hole_area_ratio: float = 1.0e-4,
) -> PlanarFaceBuild | None:
    """Build one bounded planar face for a recognized region.

    Loop points are projected onto the fitted plane first, so the wire is
    exactly planar and OpenCASCADE accepts it without a tolerance fight.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces_array = np.asarray(faces, dtype=np.int64)[:, :3]
    normal = np.asarray(region.normal, dtype=float)
    origin = np.asarray(region.origin, dtype=float)
    axis_u, axis_v = _plane_basis(normal)

    local_all = vertices - origin
    plane_points = np.column_stack([local_all @ axis_u, local_all @ axis_v])
    loops = region_boundary_loops(
        faces_array,
        region.face_indices,
        plane_points=plane_points,
    )
    if not loops:
        return None

    projected: list[np.ndarray] = []
    areas: list[float] = []
    for loop in loops:
        points = vertices[loop]
        points = points - ((points - origin) @ normal)[:, None] * normal
        local = points - origin
        planar = np.column_stack([local @ axis_u, local @ axis_v])
        projected.append(points)
        areas.append(_signed_area(planar))

    outer_index = int(np.argmax(np.abs(areas)))
    outer_points = projected[outer_index]
    outer_area = abs(areas[outer_index])
    if outer_area <= 0.0:
        return None

    plane = gp_Pln(
        gp_Ax3(
            gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
            gp_Dir(float(normal[0]), float(normal[1]), float(normal[2])),
        )
    )
    try:
        maker = BRepBuilderAPI_MakeFace(plane, _make_wire(outer_points))
    except RuntimeError:
        return None
    if not maker.IsDone():
        return None

    hole_loops: list[np.ndarray] = []
    for index, points in enumerate(projected):
        if index == outer_index:
            continue
        if abs(areas[index]) < minimum_hole_area_ratio * outer_area:
            continue
        try:
            wire = _make_wire(points)
        except RuntimeError:
            continue
        # A hole runs opposite to the outer boundary; OCC needs that winding to
        # know it removes material rather than adding a second face. Reversed()
        # hands back a TopoDS_Shape, which MakeFace.Add will not take.
        if np.sign(areas[index]) == np.sign(areas[outer_index]):
            wire = TopoDS.Wire_s(wire.Reversed())
        maker.Add(wire)
        hole_loops.append(loops[index])
    if not maker.IsDone():
        return None
    face = maker.Face()

    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, properties)
    face_area = float(properties.Mass())

    triangle = vertices[faces_array[np.asarray(region.face_indices, dtype=np.int64)]]
    cross = np.cross(
        triangle[:, 1, :] - triangle[:, 0, :],
        triangle[:, 2, :] - triangle[:, 0, :],
    )
    triangle_area = float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))

    _surface = BRep_Tool.Surface_s(face)
    return PlanarFaceBuild(
        face=face,
        outer_loop=loops[outer_index],
        hole_loops=hole_loops,
        face_area=face_area,
        triangle_area=triangle_area,
    )


@dataclass(frozen=True)
class HybridSolidBuild:
    """A solid whose recognized faces are analytic and the rest tessellated."""

    solid: Any
    planar_face_count: int
    facet_face_count: int
    surface_type_counts: dict[str, int]
    solid_volume: float
    mesh_volume: float
    closed: bool

    @property
    def volume_error(self) -> float:
        """Relative volume difference against the surface it was built from."""
        if self.mesh_volume == 0.0:
            return 0.0
        return abs(self.solid_volume - self.mesh_volume) / abs(self.mesh_volume)


def mesh_volume(vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> float:
    """Signed volume enclosed by a closed triangle surface."""
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    triangle = vertices[np.asarray(faces, dtype=np.int64)[:, :3]]
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangle[:, 0, :],
                np.cross(triangle[:, 1, :], triangle[:, 2, :]),
            )
        )
        / 6.0
    )


def build_hybrid_solid(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    segmentation: Any,
    *,
    maximum_facet_faces: int = 24_000,
    sew_tolerance: float | None = None,
) -> HybridSolidBuild:
    """Sew recognized planar faces together with the tessellated remainder.

    The vertices are snapped onto their fitted planes first, and both sides are
    then built from that same snapped array. Without it the planar face is
    projected onto its plane while the neighbouring facet keeps the original
    vertex, the shared edge moves apart by the region's flatness error, and the
    sew leaves a gap: on a truncated sphere that dropped 1333 faces to 287 and
    the shell never closed. Snapping is not a cleanup pass here, it is what
    makes the two sides share a boundary at all.

    The remainder is tessellated only because a boundary-constrained patch
    fitter does not exist yet. It is capped, because OpenCASCADE's sewing cost
    makes a six-figure face count take hours rather than minutes.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces_array = np.asarray(faces, dtype=np.int64)[:, :3]

    assigned = np.asarray(segmentation.face_region, dtype=np.int64) >= 0
    remainder = np.flatnonzero(~assigned)
    if len(remainder) > int(maximum_facet_faces):
        raise RuntimeError(
            f"{len(remainder):,} triangles are left unrecognized, above the "
            f"{int(maximum_facet_faces):,} that can be sewn in reasonable time. "
            "A boundary-constrained patch fitter has to cover the freeform "
            "remainder before a body this size can be assembled."
        )

    vertices, _snap_report = snap_surface_to_planar_regions(
        vertices, faces_array, segmentation
    )

    if sew_tolerance is None:
        span = float(np.max(np.ptp(vertices, axis=0)))
        sew_tolerance = max(span * 1.0e-7, 1.0e-7)

    built_planar = []
    for region in segmentation.regions:
        build = build_planar_region_face(vertices, faces_array, region)
        if build is not None:
            built_planar.append(build)

    sewing = BRepBuilderAPI_Sewing(float(sew_tolerance))
    for build in built_planar:
        sewing.Add(build.face)
    for index in remainder:
        polygon = BRepBuilderAPI_MakePolygon()
        for vertex_id in faces_array[index]:
            point = vertices[vertex_id]
            polygon.Add(gp_Pnt(float(point[0]), float(point[1]), float(point[2])))
        polygon.Close()
        if not polygon.IsDone():
            continue
        facet = BRepBuilderAPI_MakeFace(polygon.Wire())
        if facet.IsDone():
            sewing.Add(facet.Face())
    sewing.Perform()
    sewn = sewing.SewedShape()

    explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    if not explorer.More():
        raise RuntimeError("Sewing did not produce a shell.")
    shell = TopoDS.Shell_s(explorer.Current())
    closed = bool(BRep_Tool.IsClosed_s(shell))
    solid_maker = BRepBuilderAPI_MakeSolid(shell)
    solid = solid_maker.Solid()

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, properties)
    volume = float(properties.Mass())

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_FACE

    type_counts: dict[str, int] = {}
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while face_explorer.More():
        adaptor = BRepAdaptor_Surface(TopoDS.Face_s(face_explorer.Current()))
        name = str(adaptor.GetType()).rsplit(".", 1)[-1].replace("GeomAbs_", "")
        type_counts[name] = type_counts.get(name, 0) + 1
        face_explorer.Next()

    return HybridSolidBuild(
        solid=solid,
        planar_face_count=len(built_planar),
        facet_face_count=int(len(remainder)),
        surface_type_counts=type_counts,
        solid_volume=volume,
        mesh_volume=mesh_volume(vertices, faces_array),
        closed=closed,
    )
