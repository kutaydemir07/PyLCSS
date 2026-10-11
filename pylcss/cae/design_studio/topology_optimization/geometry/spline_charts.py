# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Join compatible Bezier patches into exact tensor-product B-spline charts.

The join inserts degree-multiplicity knots rather than fitting another surface.
It works on arbitrary curved patches, not only recognized analytic primitives.
Creases, incompatible boundaries, and periodic/extraordinary topology delimit
the charts. Internal parameter continuity is C0; existing geometric G1
continuity is retained because each polynomial piece remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SplineChart:
    poles: np.ndarray
    # Patch id and orientation at each cell; rotations preserve surface normal.
    cells: np.ndarray
    rotations: np.ndarray


def _oriented_quad(quads, face, rotation):
    return np.roll(quads[face], rotation)


def consolidate_patches(poles, quads, *, crease_edges=(), maximum_chart_patches=64,
                        maximum_chart_spans=16, boundary_tolerance=1e-9):
    """Partition a closed quad layout into bounded rectangular surface charts.

    Quad vertices are topological ids, not points clustered by proximity. A
    proposed row/column must have all its neighbors, matching oriented control
    curves, and an injective vertex grid. Thus a chart cannot close a hole,
    cross a branch point, or join unrelated sheets that happen to be nearby.
    """
    poles = np.asarray(poles, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    if (poles.ndim != 4 or poles.shape[1:] != (6, 6, 3)
            or quads.shape != (len(poles), 4) or not np.isfinite(poles).all()):
        raise ValueError('Expected one finite biquintic control grid per quad')
    if maximum_chart_patches < 1 or maximum_chart_spans < 1 or boundary_tolerance < 0:
        raise ValueError('Chart budgets must be positive and tolerance nonnegative')
    creases = {tuple(sorted(map(int, edge))) for edge in crease_edges}
    owners = {}
    for face, quad in enumerate(quads):
        for a, b in zip(quad, np.roll(quad, -1)):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    used = set()
    charts = []
    maximum_gap = 0.

    def neighbor(cell, side):
        face, rotation = cell
        quad = _oriented_quad(quads, face, rotation)
        # +u, +v, -u, -v: wanted two corners in the neighbor's chart frame.
        edge, corners, values = (
            ((quad[1], quad[2]), (0, 3), (quad[1], quad[2])),
            ((quad[3], quad[2]), (0, 1), (quad[3], quad[2])),
            ((quad[0], quad[3]), (1, 2), (quad[0], quad[3])),
            ((quad[0], quad[1]), (3, 2), (quad[0], quad[1])),
        )[side]
        key = tuple(sorted(map(int, edge)))
        adjacent = owners[key]
        if key in creases or len(adjacent) != 2:
            return None
        other = adjacent[0] if adjacent[1] == face else adjacent[1]
        if other in used:
            return None
        for turn in range(4):
            candidate = _oriented_quad(quads, other, turn)
            if all(candidate[corner] == value for corner, value in zip(corners, values)):
                return other, turn
        return None

    def coherent(grid):
        """Check all shared vertices and polynomial boundaries, not endpoints."""
        vertices = np.full((len(grid) + 1, len(grid[0]) + 1), -1, dtype=np.int64)
        controls = np.full((5 * len(grid) + 1, 5 * len(grid[0]) + 1, 3), np.nan)
        ids = set()
        gap = 0.
        for i, row in enumerate(grid):
            for j, (face, rotation) in enumerate(row):
                if face in ids:
                    return None
                ids.add(face)
                quad = _oriented_quad(quads, face, rotation)
                for a, b, vertex in ((i, j, quad[0]), (i + 1, j, quad[1]),
                                      (i + 1, j + 1, quad[2]), (i, j + 1, quad[3])):
                    if vertices[a, b] != -1 and vertices[a, b] != vertex:
                        return None
                    vertices[a, b] = vertex
                patch = np.rot90(poles[face], rotation, axes=(0, 1))
                destination = controls[5*i:5*i+6, 5*j:5*j+6]
                shared = np.isfinite(destination[..., 0])
                if shared.any():
                    gap = max(gap, float(np.max(np.abs(destination[shared] - patch[shared]))))
                    if gap > boundary_tolerance:
                        return None
                # Retain the existing boundary rather than averaging/refitting.
                destination[~shared] = patch[~shared]
        if len(np.unique(vertices)) != vertices.size:
            return None
        # Reject an internal edge explicitly classified as a physical crease.
        for i in range(1, len(grid)):
            for j in range(len(grid[0])):
                if tuple(sorted((int(vertices[i, j]), int(vertices[i, j+1])))) in creases:
                    return None
        for i in range(len(grid)):
            for j in range(1, len(grid[0])):
                if tuple(sorted((int(vertices[i, j]), int(vertices[i+1, j])))) in creases:
                    return None
        return controls, gap

    for seed in range(len(quads)):
        if seed in used:
            continue
        grid = [[(seed, 0)]]
        while True:
            options = []
            for side in range(4):
                nu, nv = len(grid), len(grid[0])
                new_u, new_v = nu + (side % 2 == 0), nv + (side % 2 == 1)
                if (new_u * new_v > maximum_chart_patches
                        or max(new_u, new_v) > maximum_chart_spans):
                    continue
                boundary = (grid[-1] if side == 0 else [row[-1] for row in grid] if side == 1
                            else grid[0] if side == 2 else [row[0] for row in grid])
                extension = [neighbor(cell, side) for cell in boundary]
                if any(cell is None for cell in extension):
                    continue
                if side == 0:
                    candidate = grid + [extension]
                elif side == 2:
                    candidate = [extension] + grid
                elif side == 1:
                    candidate = [row + [cell] for row, cell in zip(grid, extension)]
                else:
                    candidate = [[cell] + row for row, cell in zip(grid, extension)]
                result = coherent(candidate)
                if result is not None:
                    options.append(((new_u * new_v, -max(new_u, new_v), -side), candidate))
            if not options:
                break
            grid = max(options, key=lambda option: option[0])[1]
        controls, gap = coherent(grid)
        maximum_gap = max(maximum_gap, gap)
        cells = np.array([[cell[0] for cell in row] for row in grid], dtype=np.int64)
        rotations = np.array([[cell[1] for cell in row] for row in grid], dtype=np.int64)
        used.update(map(int, cells.ravel()))
        charts.append(SplineChart(controls, cells, rotations))
    count = len(charts)
    return charts, {
        'method': 'Exact piecewise Bezier surface consolidation',
        'input_patch_count': len(poles), 'output_surface_count': count,
        'merged_patch_count': len(poles) - count,
        'removed_patch_boundaries': sum((chart.cells.shape[0]-1)*chart.cells.shape[1]
                                        + (chart.cells.shape[1]-1)*chart.cells.shape[0]
                                        for chart in charts),
        'maximum_patches_per_surface': max((chart.cells.size for chart in charts), default=0),
        'maximum_shared_control_gap': maximum_gap,
        'boundary_tolerance': float(boundary_tolerance),
        'geometric_pieces_retained': True, 'refitted': False,
        'parameter_continuity': 'C0 at retained internal knots; original geometric continuity retained',
    }


def chart_surface(chart):
    """Make the native degree-five B-spline without approximation/knot removal."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.gp import gp_Pnt
    nu, nv = chart.poles.shape[:2]
    points = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            points.SetValue(i + 1, j + 1, gp_Pnt(*map(float, chart.poles[i, j])))

    def axis(spans):
        knots = TColStd_Array1OfReal(1, spans + 1)
        multiplicities = TColStd_Array1OfInteger(1, spans + 1)
        for index in range(spans + 1):
            knots.SetValue(index + 1, float(index))
            multiplicities.SetValue(index + 1, 6 if index in (0, spans) else 5)
        return knots, multiplicities

    uknots, umults = axis(chart.cells.shape[0])
    vknots, vmults = axis(chart.cells.shape[1])
    return Geom_BSplineSurface(points, uknots, vknots, umults, vmults, 5, 5)


def chart_faces(charts, quads, tolerance):
    """Create shared boundary edges with exact pcurves on each chart.

    A long natural surface edge can span many neighboring short face edges.
    OCC's geometric sewing may miss these matches at spline knots. Preserve
    the source topological edge partition and reuse the actual TopoDS edges
    instead, with one exact parameter-space segment on each incident face.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex,
                                  BRepBuilderAPI_MakeWire)
    from OCP.Geom import Geom_BSplineCurve
    from OCP.Geom2d import Geom2d_BezierCurve
    from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array1OfPnt2d
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.TopoDS import TopoDS, TopoDS_Face
    from OCP.gp import gp_Pnt, gp_Pnt2d

    builder = BRep_Builder()
    vertices, edges, faces = {}, {}, []
    knots, multiplicities = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
    for index, value in ((1, 0.), (2, 1.)):
        knots.SetValue(index, value)
        multiplicities.SetValue(index, 6)

    for chart in charts:
        surface = chart_surface(chart)
        face = TopoDS_Face()
        builder.MakeFace(face, surface, float(tolerance))
        nu, nv = chart.cells.shape
        boundary = ([(i, 0, 0, (i, 0), (i+1, 0)) for i in range(nu)]
                    + [(nu-1, j, 1, (nu, j), (nu, j+1)) for j in range(nv)]
                    + [(i, nv-1, 2, (i+1, nv), (i, nv)) for i in range(nu-1, -1, -1)]
                    + [(0, j, 3, (0, j+1), (0, j)) for j in range(nv-1, -1, -1)])
        wire = BRepBuilderAPI_MakeWire()
        for i, j, side, start_uv, end_uv in boundary:
            quad = _oriented_quad(quads, chart.cells[i, j], chart.rotations[i, j])
            a, b = int(quad[side]), int(quad[(side+1) % 4])
            patch = chart.poles[5*i:5*i+6, 5*j:5*j+6]
            curve_poles = (patch[:, 0], patch[-1, :], patch[::-1, -1], patch[0, ::-1])[side]
            key = tuple(sorted((a, b)))
            forward = a < b
            if not forward:
                curve_poles = curve_poles[::-1]
            if key not in edges:
                points = TColgp_Array1OfPnt(1, 6)
                for k, value in enumerate(curve_poles, 1):
                    points.SetValue(k, gp_Pnt(*map(float, value)))
                curve = Geom_BSplineCurve(points, knots, multiplicities, 5)
                for vertex, value in zip(key, (curve_poles[0], curve_poles[-1])):
                    if vertex not in vertices:
                        vertices[vertex] = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, value))).Vertex()
                edges[key] = BRepBuilderAPI_MakeEdge(curve, vertices[key[0]], vertices[key[1]], 0., 1.).Edge()
            edge = edges[key]
            parameters = (start_uv, end_uv) if forward else (end_uv, start_uv)
            points_2d = TColgp_Array1OfPnt2d(1, 2)
            for k, value in enumerate(parameters, 1):
                points_2d.SetValue(k, gp_Pnt2d(*map(float, value)))
            builder.UpdateEdge(edge, Geom2d_BezierCurve(points_2d), face, float(tolerance))
            builder.Range(edge, face, 0., 1.)
            builder.SameParameter(edge, True)
            builder.SameRange(edge, True)
            wire.Add(edge if forward else TopoDS.Edge_s(edge.Reversed()))
        if not wire.IsDone():
            raise RuntimeError('A consolidated surface boundary could not form a wire')
        builder.Add(face, wire.Wire())
        faces.append(face)
    return faces
