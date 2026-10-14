# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Compact cubic surface recovery with shared curves and independent local fits.

No global tangent equations or harmonic parameter solves are required. Closure
comes from fitting each boundary curve once, and reusing it on both faces.
Compatible patches are consolidated into larger native B-spline surfaces.
Continuity is positional (C0); this is not an isogeometric G1 surface solver.
"""
from __future__ import annotations

from math import comb
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .cad_config import CadReconstructionConfig
from .mesh_topology import mesh_topology_report, topology_error


class SurfaceFitError(RuntimeError):
    """Reconstruction failed a geometry acceptance check."""


def bernstein(degree, parameters):
    t = np.asarray(parameters, dtype=float)
    return np.stack([comb(degree, i) * t**i * (1 - t)**(degree - i)
                     for i in range(degree + 1)], axis=-1)


def _shell_signature(mesh):
    return sorted((int(part.euler_number), 1 if part.volume > 0 else -1)
                  for part in mesh.split(only_watertight=False))


def _quality_decimation(source, target):
    """MeshLab's link-condition and normal-preserving quadric collapse."""
    import pymeshlab
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(source.vertices, dtype=np.float64),
                                  face_matrix=np.asarray(source.faces, dtype=np.int32)))
    meshes.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target), preservetopology=True, preservenormal=True,
        preserveboundary=True, qualitythr=.5, planarquadric=True, autoclean=False)
    mesh = meshes.current_mesh()
    return trimesh.Trimesh(mesh.vertex_matrix(), mesh.face_matrix(), process=False)


def _pair_triangles(mesh, crease_angle_deg=55.):
    from .quad_layout import pair_triangles
    return pair_triangles(mesh, crease_angle_deg)


def build_quad_cage(source, target_triangles, crease_angle_deg=55., *, implicit_surface=False, quad_layout=None):
    """Decimate only when every closed boundary shell and its genus survive."""
    original = mesh_topology_report(source.vertices, source.faces)
    if not original['is_closed']:
        raise SurfaceFitError('The recovered surface must be closed, consistently wound and nondegenerate')
    from .quad_layout import layout_quality, validate_quad_layout
    signature = _shell_signature(source)
    if quad_layout is not None:
        try:
            vertices, quads = validate_quad_layout(source, quad_layout['vertices'],
                                                    quad_layout['quads'])
        except (KeyError, ValueError) as error:
            raise SurfaceFitError(f'Invalid supplied quad layout: {error}') from error
        crease_edges = _layout_creases(source, vertices, quads, crease_angle_deg, implicit_surface)
        return vertices, quads, {'source_triangles': len(source.faces), 'cage_triangles': None,
                                 'patch_count': len(quads), 'topology': original,
                                 'paired_quad_count': len(quads), 'crease_edges': crease_edges,
                                 'layout_method': 'Validated supplied quad layout',
                                 'parameter_initialization': 'Supplied cage positions',
                                 'rejected_decimations': [], **layout_quality(vertices, quads)}
    reduced = source
    rejected = []
    polygons = None
    # A usable quad network in the original triangulation can be smaller than
    # subdivision of a decimated mesh. Keep it when the fitting budget allows.
    if len(source.faces) <= min(2500, 4*target_triangles):
        candidate_polygons = _pair_triangles(source, crease_angle_deg)
        if all(len(p) == 4 for p in candidate_polygons) and len(candidate_polygons) <= 2*target_triangles:
            polygons = candidate_polygons
    if polygons is None and len(source.faces) > target_triangles:
        for method in ('quality', 4, 7):
            try:
                candidate = (_quality_decimation(source, target_triangles) if method == 'quality' else
                             source.simplify_quadric_decimation(face_count=int(target_triangles),
                                                                aggression=method))
            except (ImportError, RuntimeError) as error:
                rejected.append(str(error))
                continue
            error = topology_error(original, mesh_topology_report(candidate.vertices, candidate.faces))
            if not error and _shell_signature(candidate) != signature:
                error = 'Decimation changed cavity orientation or individual shell genus'
            if not error:
                delta = abs(candidate.volume - source.volume) / max(abs(source.volume), 1e-15)
                if delta > 0.05:
                    error = f'Decimation changed signed material volume by {delta:.1%}'
            if error:
                rejected.append(error)
            else:
                reduced = candidate
                break
    if polygons is None:
        from .quad_layout import compact_quad_network
        # MeshLab smart pairing uses a global matching solve. Use it only
        # for small local cages; large cages use sorted local pairing.
        compact = compact_quad_network(reduced) if len(reduced.faces) < 500 else None
        if compact is not None:
            vertices, quads, removed = compact
            crease_edges = _layout_creases(source, vertices, quads, crease_angle_deg, implicit_surface)
            return vertices, quads, {'source_triangles': len(source.faces), 'cage_triangles': len(reduced.faces),
                                     'patch_count': len(quads), 'topology': original,
                                     'paired_quad_count': len(quads), 'crease_edges': crease_edges,
                                                                          'layout_method': 'Convex smart quad pairing',
                                     'parameter_initialization': 'Topology-preserving source cage vertices',
                                     'rejected_decimations': rejected, **layout_quality(vertices, quads)}
        polygons = _pair_triangles(reduced, crease_angle_deg)
    direct_quads = all(len(polygon) == 4 for polygon in polygons)
    if direct_quads:
        vertices, quads = np.asarray(reduced.vertices), np.asarray(polygons, dtype=np.int64)
    else:
        points = list(np.asarray(reduced.vertices))
        edge_midpoints, quads = {}, []
        for polygon in polygons:
            mids = []
            for start, end in zip(polygon, np.roll(polygon, -1)):
                key = tuple(sorted((int(start), int(end))))
                if key not in edge_midpoints:
                    edge_midpoints[key] = len(points)
                    points.append((reduced.vertices[start] + reduced.vertices[end]) * .5)
                mids.append(edge_midpoints[key])
            centre = len(points)
            points.append(reduced.vertices[list(polygon)].mean(axis=0))
            for i, vertex in enumerate(polygon):
                quads.append((vertex, mids[i], centre, mids[(i - 1) % len(mids)]))
        vertices, quads = np.asarray(points), np.asarray(quads, dtype=np.int64)
    error = topology_error(original, mesh_topology_report(vertices, quads))
    if error:
        raise SurfaceFitError(error)
    crease_edges = _layout_creases(source, vertices, quads, crease_angle_deg, implicit_surface)
    return vertices, quads, {'source_triangles': len(source.faces), 'cage_triangles': len(reduced.faces),
                             'patch_count': len(quads), 'topology': original,
                             'paired_quad_count': sum(len(polygon) == 4 for polygon in polygons),
                             'crease_edges': crease_edges,
                             'parameter_initialization': 'Topology-preserving source cage vertices',
                             'rejected_decimations': rejected,
                             'layout_method': ('Feature/curvature-aware matching; retained quad network' if direct_quads
                                               else 'Feature/curvature-aware matching; conforming polygon subdivision'),
                             **layout_quality(vertices, quads)}


def _layout_creases(source, vertices, quads, crease_angle_deg, implicit_surface):
    """A coarse bend is a crease only when supported by source features."""
    corners = vertices[quads]
    normals = np.cross(corners, np.roll(corners, -1, axis=1)).sum(axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    owners = {}
    caps = []
    diagonal = np.linalg.norm(source.extents)
    for face, quad in enumerate(quads):
        caps.append({(axis, side) for axis in range(3) for side in range(2)
                     if np.max(np.abs(corners[face, :, axis]-source.bounds[side, axis])) < diagonal*1e-7})
        for a, b in zip(quad, np.roll(quad, -1)):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    feature_edges = source.face_adjacency_edges[source.face_adjacency_angles >= np.deg2rad(crease_angle_deg)]
    feature_points = source.vertices[feature_edges].mean(axis=1) if len(feature_edges) else np.empty((0, 3))
    tree = cKDTree(feature_points) if len(feature_points) else None
    creases = []
    for edge, faces in owners.items():
        if len(faces) != 2:
            raise SurfaceFitError('A quad cage edge has no manifold pair')
        if implicit_surface:
            sharp = caps[faces[0]] != caps[faces[1]] and bool(caps[faces[0]] or caps[faces[1]])
        else:
            angle = np.rad2deg(np.arccos(np.clip(normals[faces[0]] @ normals[faces[1]], -1, 1)))
            sharp = False
            if angle >= crease_angle_deg and tree is not None:
                centre = vertices[list(edge)].mean(axis=0)
                _, nearest = tree.query(centre, k=min(16, len(feature_edges)))
                segments = source.vertices[feature_edges[np.atleast_1d(nearest)]]
                delta = segments[:, 1]-segments[:, 0]
                parameter = np.clip(np.sum((centre-segments[:, 0])*delta, axis=1)/np.sum(delta**2, axis=1), 0, 1)
                distance = np.linalg.norm(centre-segments[:, 0]-parameter[:, None]*delta, axis=1).min()
                sharp = distance <= max(diagonal*1e-7, .15*np.linalg.norm(vertices[edge[1]]-vertices[edge[0]]))
        if sharp:
            creases.append(edge)
    return creases



class _SourceProjector:
    """Exact oriented surface projection using triangle bounds.

    A centroid-neighbour query provides an upper bound only. Large triangles
    can have distant centroids even when the query lies on their surface.
    The bounds query includes every oriented triangle that can beat that bound.
    """

    def __init__(self, source):
        self.source = source
        self.triangles = source.triangles
        self.normals = source.face_normals
        self.tree = cKDTree(source.triangles_center)
        self.bounds_tree = source.triangles_tree

    def project(self, points, normals):
        result = np.empty_like(points)
        normals = np.broadcast_to(normals, points.shape)
        for first in range(0, len(points), 1024):
            query = points[first:first+1024]
            direction = normals[first:first+len(query)]
            _, ids = self.tree.query(query, k=min(32, len(self.triangles)))
            ids = np.asarray(ids).reshape(len(query), -1)
            repeated = np.repeat(query, ids.shape[1], axis=0)
            sites = trimesh.triangles.closest_point(self.triangles[ids.ravel()], repeated)
            distances = np.sum((sites-repeated)**2, axis=1).reshape(ids.shape)
            distances[np.einsum('nki,ni->nk', self.normals[ids], direction) <= 0] = np.inf
            upper = np.sqrt(distances.min(axis=1))
            for index in np.flatnonzero(~np.isfinite(upper)):
                # This is an orientation search, never a geometry replacement.
                compatible = np.flatnonzero(self.normals @ direction[index] > 0)
                if not len(compatible):
                    raise SurfaceFitError('No oriented source surface near the control cage')
                projected = trimesh.triangles.closest_point(
                    self.triangles[compatible], np.broadcast_to(query[index], (len(compatible), 3)))
                upper[index] = np.linalg.norm(projected-query[index], axis=1).min()
            radius = upper[:, None]+1e-10
            bounds = np.column_stack((query-radius, query+radius))
            candidates = [np.fromiter(self.bounds_tree.intersection(bound), dtype=np.int64)
                          for bound in bounds]
            counts = np.array([len(candidate) for candidate in candidates])
            indices = np.concatenate(candidates)
            owners = np.repeat(np.arange(len(query)), counts)
            projected = trimesh.triangles.closest_point(self.triangles[indices], query[owners])
            distance = np.sum((projected-query[owners])**2, axis=1)
            distance[np.einsum('ni,ni->n', self.normals[indices], direction[owners]) <= 0] = np.inf
            offsets = np.r_[0, np.cumsum(counts)]
            chosen = np.array([offsets[i]+np.argmin(distance[offsets[i]:offsets[i+1]])
                               for i in range(len(query))])
            result[first:first+len(query)] = projected[chosen]
        return result


def evaluate_patches(poles, parameters):
    basis = bernstein(poles.shape[1]-1, parameters)
    return np.einsum('ui,fijc,vj->fuvc', basis, poles, basis)


def _project_regular_cage(vertices, projected, quads, normals):
    """Move shared corners toward the source without inverting their fans."""
    weights = np.ones(len(vertices))
    while True:
        candidate = vertices+(projected-vertices)*weights[:, None]
        corners = candidate[quads]
        sides = np.roll(corners, -1, axis=1)-corners
        turns = np.einsum('fkc,fc->fk', np.cross(sides, np.roll(sides, -1, axis=1)), normals)
        area = np.linalg.norm(sides, axis=2)*np.linalg.norm(np.roll(sides, -1, axis=1), axis=2)
        bad = np.any(turns <= np.maximum(area*1e-10, 1e-20), axis=1)
        if not bad.any():
            return candidate
        affected = np.unique(quads[bad])
        weights[affected] *= .5
        if np.any(weights[affected] < np.finfo(float).eps):
            raise SurfaceFitError('A control-cage corner fan is not locally injective')


def _boundary_patches(vertices, quads, edges):
    """Initialize one cubic control grid from four shared boundary curves."""
    poles = np.zeros((len(quads), 4, 4, 3))
    for face, quad in enumerate(quads):
        for side in range(4):
            a, b = int(quad[side]), int(quad[(side+1) % 4])
            curve = edges[tuple(sorted((a, b)))]
            if a > b:
                curve = curve[::-1]
            if side == 0: poles[face, :, 0] = curve
            elif side == 1: poles[face, -1, :] = curve
            elif side == 2: poles[face, ::-1, -1] = curve
            else: poles[face, 0, ::-1] = curve
    u, v = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing='ij')
    weights = np.stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v), axis=-1)
    bilinear = np.einsum('ijk,fkc->fijc', weights, vertices[quads])
    coons = (poles[:, :, :1]*(1-v)[None, ..., None]+poles[:, :, -1:]*v[None, ..., None]
             +poles[:, :1, :]*(1-u)[None, ..., None]+poles[:, -1:, :]*u[None, ..., None]-bilinear)
    poles[:, 1:3, 1:3] = coons[:, 1:3, 1:3]
    return poles


def _regular_patches(poles, normals, projector):
    t = np.linspace(1e-4, 1-1e-4, 7)
    degree = poles.shape[1]-1
    basis, derivative = bernstein(degree, t), bernstein(degree-1, t)
    du = np.einsum('ui,fijc,vj->fuvc', derivative, degree*np.diff(poles, axis=1), basis)
    dv = np.einsum('ui,fijc,vj->fuvc', basis, degree*np.diff(poles, axis=2), derivative)
    cross = np.cross(du, dv)
    orientation = np.einsum('fuvc,fc->fuv', cross, normals)
    area = np.linalg.norm(du, axis=-1)*np.linalg.norm(dv, axis=-1)
    regular = orientation > np.maximum(area*1e-7, 1e-15)
    # A toroidal or strongly curved patch can turn beyond its cage normal.
    # Compare those sites to the actual source rather than forcing every
    # patch to be a height field over its coarse quadrilateral.
    if not regular.all():
        positions = evaluate_patches(poles, t)
        indices = np.nonzero(~regular)
        sites = positions[indices]
        _, distance, ids = trimesh.proximity.closest_point(projector.source, sites)
        reference = projector.normals[ids]
        oriented = np.einsum('nc,nc->n', cross[indices], reference)
        regular[indices] = oriented > np.maximum(area[indices]*1e-7, 1e-15)
    regular &= np.linalg.norm(cross, axis=-1) > np.maximum(area*1e-7, 1e-15)
    return np.all(regular, axis=(1, 2))


def fit_patches(source, vertices, quads, config: CadReconstructionConfig, *, progress_callback=None, **legacy_options):
    """Fit shared cubic edges once, then solve four interior poles per patch."""
    from ._compat import warn_retired_options
    warn_retired_options(legacy_options, operation='Compact spline fitting')
    origin = source.bounds.mean(axis=0)
    scale = float(np.linalg.norm(source.extents))
    local_source = trimesh.Trimesh((source.vertices-origin)/scale, source.faces, process=False)
    projector = _SourceProjector(local_source)
    vertices = (vertices-origin)/scale
    corners = vertices[quads]
    normals = np.cross(corners, np.roll(corners, -1, axis=1)).sum(axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    vertex_normals = np.zeros_like(vertices)
    for corner in range(4):
        np.add.at(vertex_normals, quads[:, corner], normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-30)
    vertices = _project_regular_cage(vertices, projector.project(vertices, vertex_normals), quads, normals)
    owners = {}
    for face, quad in enumerate(quads):
        for a, b in zip(quad, np.roll(quad, -1)):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    if any(len(faces) != 2 for faces in owners.values()):
        raise SurfaceFitError('The fitting cage must have two faces at every edge')
    edge_ids = list(owners)
    t = np.linspace(0, 1, config.samples_per_axis)
    basis = bernstein(3, t)
    edge_fit = basis[:, 1:3]
    ridge = config.fairness
    solve_edge = np.linalg.inv(edge_fit.T@edge_fit+ridge*np.eye(2))
    edge_poles = np.empty((len(edge_ids),4,3))
    controls = np.linspace(0,1,4)[None,:,None]
    for first in range(0,len(edge_ids),2048):
        ids = edge_ids[first:first+2048]
        ends = vertices[np.asarray(ids)]
        edge_normals = np.array([normals[owners[key]].sum(axis=0) for key in ids])
        sites = ends[:, :1]*(1-t[None,:,None])+ends[:, 1:]*t[None,:,None]
        targets = projector.project(sites.reshape(-1,3),
                                    np.repeat(edge_normals,len(t),axis=0)).reshape(sites.shape)
        batch = ends[:, :1]*(1-controls)+ends[:, 1:]*controls
        fixed = basis[:, :1]*ends[:,0,None,:]+basis[:, -1:]*ends[:,1,None,:]
        rhs = np.einsum('si,esc->eic',edge_fit,targets-fixed)+ridge*batch[:,1:3]
        batch[:,1:3] = np.einsum('ij,ejc->eic',solve_edge,rhs)
        edge_poles[first:first+len(ids)] = batch
    ends = vertices[np.asarray(edge_ids)]
    initial_edges = ends[:, :1]*(1-controls)+ends[:, 1:]*controls
    edge_steps = np.ones(len(edge_ids))
    edge_indices = {edge: index for index, edge in enumerate(edge_ids)}
    face_edges = np.array([[edge_indices[tuple(sorted((int(a), int(b))))]
                           for a,b in zip(quad,np.roll(quad,-1))] for quad in quads])
    # Feasible line search for shared curves. A bad boundary used to cause an
    # identical failure at every global refinement. Every affected edge is
    # updated once for both owners; acceptance still uses the source tolerance.
    boundary_iterations = 0
    while True:
        boundary_iterations += 1
        current_edges = initial_edges+(edge_poles-initial_edges)*edge_steps[:, None, None]
        poles = _boundary_patches(vertices, quads, dict(zip(edge_ids, current_edges)))
        regular = _regular_patches(poles, normals, projector)
        if regular.all(): break
        affected = np.unique(face_edges[~regular])
        edge_steps[affected] *= .5
        if np.any(edge_steps[affected] < np.finfo(float).eps):
            raise SurfaceFitError('The fitted shared curves do not admit a regular patch layout')
    if progress_callback:
        progress_callback(f'Fitting {len(quads)} cubic patches with shared boundaries')
    tensor = np.einsum('ui,vj->uvij', basis, basis).reshape(len(t)**2, 16)
    interior_ids = np.array((5, 6, 9, 10))
    fit = tensor[:, interior_ids]
    solve = np.linalg.inv(fit.T@fit+ridge*np.eye(4))
    patch_errors = np.empty(len(quads))
    squared_error, sample_count, interior_iterations = 0., 0, 0
    # Batch size bounds temporary storage, not the number of patches a model
    # may contain. Every local solve has just four unknown interior points.
    for first in range(0, len(poles), 128):
        batch = poles[first:first+128]
        batch_normals = normals[first:first+len(batch)]
        seed = evaluate_patches(batch, t)
        targets = projector.project(seed.reshape(-1, 3),
                                    np.repeat(batch_normals, len(t)**2, axis=0)).reshape(len(batch), -1, 3)
        flat = batch.reshape(len(batch), 16, 3)
        fixed = (np.einsum('si,fic->fsc', tensor, flat)
                 -np.einsum('si,fic->fsc', fit, flat[:, interior_ids]))
        rhs = np.einsum('si,fsc->fic', fit, targets-fixed)+ridge*flat[:, interior_ids]
        initial = flat[:, interior_ids].copy()
        fitted_interior = np.einsum('ij,fjc->fic', solve, rhs)
        step = np.ones(len(batch))
        while True:
            interior_iterations += 1
            flat[:, interior_ids] = initial+(fitted_interior-initial)*step[:, None, None]
            regular = _regular_patches(batch, batch_normals, projector)
            if regular.all(): break
            step[~regular] *= .5
            if np.any(step[~regular] < np.finfo(float).eps):
                raise SurfaceFitError('The cubic interior solve cannot preserve a regular parameter map')
        fitted = evaluate_patches(batch, t).reshape(targets.shape)
        errors = np.linalg.norm(fitted-targets, axis=2)*scale
        patch_errors[first:first+len(batch)] = errors.max(axis=1)
        squared_error += float(np.sum(errors**2))
        sample_count += errors.size
    return poles*scale+origin, {
        'solver': 'Batched local least squares with orientation-preserving line search; shared boundary curves',
        'boundary_line_search_iterations': boundary_iterations,
        'interior_line_search_iterations': interior_iterations,
        'degree_u': 3, 'degree_v': 3, 'rational': False,
        'continuity': 'C0 with exact shared curves; tangent continuity is not imposed',
        'boundary_controls_shared': True,
        'fitting_rms_distance': float(np.sqrt(squared_error/sample_count)),
        'patch_maximum_source_distance': patch_errors.tolist(),
        'parameter_mapping': {'method': 'Coons initialization and oriented local projection',
                              'sample_count': len(quads)*len(t)**2, 'samples_per_axis': len(t)},
    }


def patch_mesh(poles, samples=9):
    grids = evaluate_patches(poles, np.linspace(0, 1, samples))
    i, j = np.meshgrid(np.arange(samples-1), np.arange(samples-1), indexing='ij')
    a = (i*samples+j).ravel()
    face = np.concatenate((np.stack((a, a+samples, a+samples+1), axis=1),
                           np.stack((a, a+samples+1, a+1), axis=1)))
    faces = face[None, :, :]+(np.arange(len(poles))*samples**2)[:, None, None]
    return grids.reshape(-1, 3), faces.reshape(-1, 3)


def assemble_solid(poles, tolerance, *, quads=None, crease_edges=()):
    """Build consolidated OCC B-spline faces and preserve cavity shells."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt
    from .spline_charts import SplineChart, chart_surface, chart_faces, consolidate_patches

    if quads is None:
        charts = [SplineChart(patch, np.array([[i]]), np.array([[0]]))
                  for i, patch in enumerate(poles)]
        consolidation = {'input_patch_count': len(poles), 'output_surface_count': len(poles),
                         'geometric_pieces_retained': True, 'refitted': False}
    else:
        charts, consolidation = consolidate_patches(
            poles, quads, crease_edges=crease_edges,
            boundary_tolerance=max(1e-10, float(tolerance) * 1e-3))
    sewer = BRepBuilderAPI_Sewing(float(tolerance), True, True, True, False)
    if quads is not None:
        for face in chart_faces(charts, quads, tolerance, surface_recovery_report=consolidation):
            sewer.Add(face)
        consolidation['shared_topological_boundaries'] = True
    else:
        for chart in charts:
            face = BRepBuilderAPI_MakeFace(chart_surface(chart), float(tolerance))
            if not face.IsDone():
                raise SurfaceFitError('OpenCASCADE could not build a fitted face')
            sewer.Add(face.Face())
    sewer.Perform()
    sewn = cq.Shape.cast(sewer.SewedShape())
    shells = sewn.Shells()
    if not shells or any(not shell.wrapped.Closed() for shell in shells):
        raise SurfaceFitError('Fitted faces did not sew into closed boundary shells')
    signed = [(shell, cq.Shape.cast(BRepBuilderAPI_MakeSolid(shell.wrapped).Solid())) for shell in shells]
    outers = [(shell, solid) for shell, solid in signed if solid.Volume(tol=1e-6) > 0]
    inners = [(shell, solid) for shell, solid in signed if solid.Volume(tol=1e-6) < 0]
    if not outers:
        raise SurfaceFitError('Fitted boundary orientation does not enclose material')
    cavities = [[] for _ in outers]
    for shell, _ in inners:
        point = shell.Vertices()[0].Center()
        owners = [i for i, (_, solid) in enumerate(outers)
                  if BRepClass3d_SolidClassifier(solid.wrapped, gp_Pnt(point.x, point.y, point.z),
                                               float(tolerance)).State() == TopAbs_IN]
        if not owners:
            raise SurfaceFitError('An inward cavity shell has no containing material body')
        owner = min(owners, key=lambda i: outers[i][1].Volume(tol=1e-6))
        cavities[owner].append(shell)
    bodies = []
    for (outer, _), holes in zip(outers, cavities):
        builder = BRepBuilderAPI_MakeSolid(outer.wrapped)
        for hole in holes:
            builder.Add(hole.wrapped)
        solid = builder.Solid()
        BRepLib.OrientClosedSolid_s(solid)
        body = cq.Shape.cast(solid)
        if not body.isValid():
            raise SurfaceFitError('The fitted solid failed OpenCASCADE B-rep validation')
        bodies.append(body)
    result = bodies[0] if len(bodies) == 1 else cq.Compound.makeCompound(bodies)
    consolidation['face_count_before_same_domain_unification'] = len(result.Faces())
    result = result.clean()
    if not result.isValid():
        raise SurfaceFitError('Recovered analytic and spline surfaces failed solid validation')
    consolidation['output_surface_count'] = len(result.Faces())
    return result, {'solid_count': len(bodies), 'boundary_shell_count': len(shells),
                    'cavity_shell_count': len(inners), 'cad_face_count': len(result.Faces()),
                    'surface_consolidation': consolidation}
