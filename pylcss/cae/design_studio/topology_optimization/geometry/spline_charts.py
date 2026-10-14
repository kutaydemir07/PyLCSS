# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Join compatible Bezier patches into exact tensor-product B-spline charts.

The join inserts degree-multiplicity knots rather than fitting another surface.
It works on arbitrary curved patches, not only recognized analytic primitives.
Creases, incompatible boundaries, and periodic/extraordinary topology delimit
the charts. Every polynomial piece remains unchanged with C0 continuity
at retained internal knots.
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

    @property
    def degree(self):
        return (self.poles.shape[0] - 1) // self.cells.shape[0]


def _oriented_quad(quads, face, rotation):
    return np.roll(quads[face], rotation)


def consolidate_patches(poles, quads, *, crease_edges=(), boundary_tolerance=1e-9):
    """Grow compatible rectangular charts without size limits or global solves.

    Edges are checked once. Growth validates only the newly added strip and
    retains a vertex-to-UV map, so no candidate rescans or copies a whole chart.
    Topological seams and physical creases, rather than face budgets, delimit
    the resulting surfaces. Every polynomial piece is retained exactly.
    """
    poles = np.asarray(poles, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    if (poles.ndim != 4 or poles.shape[1] != poles.shape[2]
            or poles.shape[1] < 3 or poles.shape[-1] != 3
            or quads.shape != (len(poles), 4) or not np.isfinite(poles).all()):
        raise ValueError('Expected one finite square Bezier control grid per quad')
    if boundary_tolerance < 0:
        raise ValueError('Boundary tolerance must be nonnegative')
    degree = poles.shape[1]-1
    creases = {tuple(sorted(map(int, edge))) for edge in crease_edges}
    owners, curves = {}, {}
    maximum_gap = 0.
    incompatible = set()
    for face, quad in enumerate(quads):
        patch = poles[face]
        for side, (a, b) in enumerate(zip(quad, np.roll(quad, -1))):
            key = tuple(sorted((int(a), int(b))))
            curve = (patch[:, 0], patch[-1, :], patch[::-1, -1], patch[0, ::-1])[side]
            if a > b:
                curve = curve[::-1]
            owners.setdefault(key, []).append(face)
            if key in curves:
                gap = float(np.max(np.abs(curve-curves[key])))
                maximum_gap = max(maximum_gap, gap)
                if gap > boundary_tolerance:
                    incompatible.add(key)
            else:
                curves[key] = curve
    used, charts = set(), []

    def neighbor(cell, side):
        face, rotation = cell
        quad = _oriented_quad(quads, face, rotation)
        edge, corners, values = (
            ((quad[1], quad[2]), (0, 3), (quad[1], quad[2])),
            ((quad[3], quad[2]), (0, 1), (quad[3], quad[2])),
            ((quad[0], quad[3]), (1, 2), (quad[0], quad[3])),
            ((quad[0], quad[1]), (3, 2), (quad[0], quad[1])),
        )[side]
        key = tuple(sorted(map(int, edge)))
        adjacent = owners[key]
        if key in creases or key in incompatible or len(adjacent) != 2:
            return None
        other = adjacent[0] if adjacent[1] == face else adjacent[1]
        if other in used:
            return None
        for turn in range(4):
            candidate = _oriented_quad(quads, other, turn)
            if all(candidate[corner] == value for corner, value in zip(corners, values)):
                return other, turn
        return None

    for seed in range(len(quads)):
        if seed in used:
            continue
        grid = {(0, 0): (seed, 0)}
        vertices, coordinates = {}, {}
        chart_ids = {seed}
        u0 = u1 = v0 = v1 = 0

        def strip_vertices(strip):
            additions, inverse, ids = {}, {}, set()
            for (i, j), (face, rotation) in strip:
                if face in chart_ids or face in ids:
                    return None
                ids.add(face)
                quad = _oriented_quad(quads, face, rotation)
                for coord, vertex in zip(((i,j), (i+1,j), (i+1,j+1), (i,j+1)), quad):
                    vertex = int(vertex)
                    if (coord in vertices and vertices[coord] != vertex
                            or coord in additions and additions[coord] != vertex
                            or vertex in coordinates and coordinates[vertex] != coord
                            or vertex in inverse and inverse[vertex] != coord):
                        return None
                    additions[coord], inverse[vertex] = vertex, coord
                # A new strip can contain a crease between adjacent new cells.
                for a, b in zip(quad, np.roll(quad, -1)):
                    key = tuple(sorted((int(a), int(b))))
                    if key in creases or key in incompatible:
                        if any(other in ids for other in owners[key] if other != face):
                            return None
            return additions, inverse

        quad = quads[seed]
        for coord, vertex in zip(((0,0), (1,0), (1,1), (0,1)), quad):
            vertices[coord], coordinates[int(vertex)] = int(vertex), coord
        blocked = set()
        while len(blocked) < 4:
            for side in range(4):
                if side in blocked:
                    continue
                boundary = ([(u1,j) for j in range(v0,v1+1)] if side == 0 else
                            [(i,v1) for i in range(u0,u1+1)] if side == 1 else
                            [(u0,j) for j in range(v0,v1+1)] if side == 2 else
                            [(i,v0) for i in range(u0,u1+1)])
                delta = ((1,0),(0,1),(-1,0),(0,-1))[side]
                strip = [((i+delta[0], j+delta[1]), neighbor(grid[i,j], side)) for i,j in boundary]
                if any(cell is None for _, cell in strip):
                    blocked.add(side)
                    continue
                additions = strip_vertices(strip)
                if additions is None:
                    blocked.add(side)
                    continue
                vertices.update(additions[0])
                coordinates.update(additions[1])
                grid.update(strip)
                chart_ids.update(cell[0] for _, cell in strip)
                if side == 0: u1 += 1
                elif side == 1: v1 += 1
                elif side == 2: u0 -= 1
                else: v0 -= 1
        cells = np.empty((u1-u0+1,v1-v0+1), dtype=np.int64)
        rotations = np.empty_like(cells)
        controls = np.empty((degree*cells.shape[0]+1,degree*cells.shape[1]+1,3))
        for (u, v), (face, rotation) in grid.items():
            i, j = u-u0, v-v0
            cells[i,j], rotations[i,j] = face, rotation
            controls[degree*i:degree*(i+1)+1,degree*j:degree*(j+1)+1] = np.rot90(poles[face],rotation,axes=(0,1))
        used.update(chart_ids)
        charts.append(SplineChart(controls, cells, rotations))
    return charts, {
        'method': 'Incremental shared-curve chart consolidation',
        'input_patch_count': len(poles), 'output_surface_count': len(charts),
        'merged_patch_count': len(poles)-len(charts),
        'removed_patch_boundaries': sum((c.cells.shape[0]-1)*c.cells.shape[1]
                                        +(c.cells.shape[1]-1)*c.cells.shape[0] for c in charts),
        'maximum_shared_control_gap': maximum_gap, 'boundary_tolerance': float(boundary_tolerance),
        'geometric_pieces_retained': True, 'refitted': False,
        'parameter_continuity': 'C0 at retained internal knots',
    }


def chart_surface(chart):
    """Make a native B-spline with the degree of its Bezier pieces."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.gp import gp_Pnt
    degree = chart.degree
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
            multiplicities.SetValue(index + 1, degree+1 if index in (0, spans) else degree)
        return knots, multiplicities

    uknots, umults = axis(chart.cells.shape[0])
    vknots, vmults = axis(chart.cells.shape[1])
    return Geom_BSplineSurface(points, uknots, vknots, umults, vmults, degree, degree)


def chart_faces(charts, quads, tolerance, *, surface_recovery_report=None):
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
    from OCP.GeomProjLib import GeomProjLib
    from OCP.ShapeCustom import ShapeCustom_Surface

    builder = BRep_Builder()
    vertices, edges, faces = {}, {}, []
    knots, multiplicities = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
    for index, value in ((1, 0.), (2, 1.)):
        knots.SetValue(index, value)

    for chart in charts:
        degree = chart.degree
        for index in (1, 2):
            multiplicities.SetValue(index, degree+1)
        surface = chart_surface(chart)
        converter = ShapeCustom_Surface(surface)
        analytic = converter.ConvertToAnalytical(max(1e-10, float(tolerance)*.25), False)
        converted = analytic is not None
        if converted:
            surface = analytic
        if surface_recovery_report is not None:
            kind = surface.DynamicType().Name().removeprefix('Geom_')
            types = surface_recovery_report.setdefault('surface_types', {})
            types[kind] = types.get(kind, 0)+1
            if converted:
                surface_recovery_report['analytic_converted_chart_count'] = surface_recovery_report.get('analytic_converted_chart_count', 0)+1
                surface_recovery_report['maximum_analytic_conversion_gap'] = max(
                    surface_recovery_report.get('maximum_analytic_conversion_gap', 0.), converter.Gap())
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
            patch = chart.poles[degree*i:degree*(i+1)+1, degree*j:degree*(j+1)+1]
            curve_poles = (patch[:, 0], patch[-1, :], patch[::-1, -1], patch[0, ::-1])[side]
            key = tuple(sorted((a, b)))
            forward = a < b
            if not forward:
                curve_poles = curve_poles[::-1]
            if key not in edges:
                points = TColgp_Array1OfPnt(1, degree+1)
                for k, value in enumerate(curve_poles, 1):
                    points.SetValue(k, gp_Pnt(*map(float, value)))
                curve = Geom_BSplineCurve(points, knots, multiplicities, degree)
                for vertex, value in zip(key, (curve_poles[0], curve_poles[-1])):
                    if vertex not in vertices:
                        vertices[vertex] = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, value))).Vertex()
                edges[key] = BRepBuilderAPI_MakeEdge(curve, vertices[key[0]], vertices[key[1]], 0., 1.).Edge()
            edge = edges[key]
            parameters = (start_uv, end_uv) if forward else (end_uv, start_uv)
            points_2d = TColgp_Array1OfPnt2d(1, 2)
            for k, value in enumerate(parameters, 1):
                points_2d.SetValue(k, gp_Pnt2d(*map(float, value)))
            if converted:
                from OCP.BRep import BRep_Tool
                curve = BRep_Tool.Curve_s(edge, 0., 1.)
                pcurve = GeomProjLib.Curve2d_s(curve, 0., 1., surface, max(1e-10, float(tolerance)*.1))
                if pcurve is None:
                    raise RuntimeError('An analytic surface boundary could not be parameterized')
            else:
                pcurve = Geom2d_BezierCurve(points_2d)
            builder.UpdateEdge(edge, pcurve, face, float(tolerance))
            builder.Range(edge, face, 0., 1.)
            builder.SameParameter(edge, True)
            builder.SameRange(edge, True)
            wire.Add(edge if forward else TopoDS.Edge_s(edge.Reversed()))
        if not wire.IsDone():
            raise RuntimeError('A consolidated surface boundary could not form a wire')
        builder.Add(face, wire.Wire())
        faces.append(face)
    return faces
