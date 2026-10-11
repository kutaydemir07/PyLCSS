# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Sparse biquintic fitting with shared boundaries and geometric G1 constraints.

The quadratic gluing equation is described in Marsala et al., *From CAD to
representations suitable for isogeometric analysis: a complete pipeline*,
Engineering with Computers (2024), doi:10.1007/s00366-024-02065-0.
This implementation uses triangle-to-quad subdivision, not CADfix, and solves
the equations in Bernstein coefficients. No third-party paper code is used.
G1 applies across uncreased patch edges; it does not imply G2 continuity.
"""
from __future__ import annotations

from math import comb
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .cad_config import CadReconstructionConfig
from .mesh_topology import mesh_topology_report, topology_error


class SurfaceFitError(RuntimeError):
    """A bounded reconstruction failed a geometry acceptance check."""


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


def _pair_triangles(mesh):
    """Prefer convex quad pairs and retain valence >= 3 at smooth vertices.

    Pairing before conforming subdivision avoids introducing a valence-three
    extraordinary vertex in every triangle of a recovered marching-cubes mesh.
    All remaining triangles are subdivided alongside quads, so no T-junctions
    or incomplete boundaries are introduced.
    """
    candidates = []
    valence = np.bincount(mesh.faces.ravel(), minlength=len(mesh.vertices))
    for faces, edge, angle in zip(mesh.face_adjacency, mesh.face_adjacency_edges, mesh.face_adjacency_angles):
        if angle > np.deg2rad(45):
            continue
        sides = []
        for face in faces:
            triangle = mesh.faces[face]
            sides.extend((int(a), int(b)) for a, b in zip(triangle, np.roll(triangle, -1))
                         if set((int(a), int(b))) != set(edge))
        successor = dict(sides)
        quad = [sides[0][0]]
        for _ in range(3):
            quad.append(successor[quad[-1]])
        points = mesh.vertices[quad]
        normal = mesh.face_normals[faces].sum(axis=0)
        turns = np.cross(np.roll(points, -1, axis=0) - points,
                         np.roll(points, -2, axis=0) - np.roll(points, -1, axis=0)) @ normal
        if np.any(turns <= 1e-12):
            continue
        vectors = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(vectors, axis=1)
        cosines = np.sum(vectors * np.roll(vectors, 1, axis=0), axis=1) / (lengths * np.roll(lengths, 1))
        score = float(np.max(np.abs(cosines)) + .2 * (lengths.max() / lengths.min() - 1) + angle)
        candidates.append((score, int(faces[0]), int(faces[1]), tuple(quad), tuple(edge)))
    paired, polygons = set(), []
    for _, left, right, quad, edge in sorted(candidates):
        if left in paired or right in paired or any(valence[v] <= 3 for v in edge):
            continue
        paired.update((left, right))
        polygons.append(quad)
        valence[list(edge)] -= 1
    polygons.extend(tuple(face) for index, face in enumerate(mesh.faces) if index not in paired)
    return polygons


def build_quad_cage(source, target_triangles, maximum_patches, crease_angle_deg=55., *, implicit_surface=False):
    """Decimate only when every closed boundary shell and its genus survive."""
    original = mesh_topology_report(source.vertices, source.faces)
    if not original['is_closed']:
        raise SurfaceFitError('The recovered surface must be closed, consistently wound and nondegenerate')
    signature = _shell_signature(source)
    reduced = source
    rejected = []
    if len(source.faces) > target_triangles:
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
    if len(reduced.faces) * 2 > maximum_patches:
        raise SurfaceFitError(f'Topology-preserving cage exceeds the {maximum_patches} patch budget; '
                              f'at least {len(reduced.faces) * 2} patches are needed. '
                              'Increase the patch budget or simplify the recovered boundary')
    points = list(np.asarray(reduced.vertices))
    edge_midpoints = {}
    polygon_edge_faces = {}
    quads = []
    polygons = _pair_triangles(reduced)
    polygon_normals = []
    for face_index, polygon in enumerate(polygons):
        polygon_points = reduced.vertices[list(polygon)]
        normal = np.cross(polygon_points, np.roll(polygon_points, -1, axis=0)).sum(axis=0)
        polygon_normals.append(normal / np.linalg.norm(normal))
        mids = []
        for start, end in zip(polygon, np.roll(polygon, -1)):
            key = tuple(sorted((int(start), int(end))))
            polygon_edge_faces.setdefault(key, []).append(face_index)
            if key not in edge_midpoints:
                edge_midpoints[key] = len(points)
                points.append((reduced.vertices[start] + reduced.vertices[end]) * 0.5)
            mids.append(edge_midpoints[key])
        centre = len(points)
        points.append(reduced.vertices[list(polygon)].mean(axis=0))
        for i, vertex in enumerate(polygon):
            quads.append((vertex, mids[i], centre, mids[(i - 1) % len(mids)]))
    vertices, quads = np.asarray(points), np.asarray(quads, dtype=np.int64)
    if len(quads) > maximum_patches:
        raise SurfaceFitError(f'Conforming quad cage requires {len(quads)} patches, '
                              f'exceeding the {maximum_patches} patch budget')
    error = topology_error(original, mesh_topology_report(vertices, quads))
    if error:
        raise SurfaceFitError(error)
    crease_edges = []
    cap_planes = []
    for polygon in polygons:
        points_on_polygon = reduced.vertices[list(polygon)]
        planes = set()
        for axis in range(3):
            for side in range(2):
                if np.max(np.abs(points_on_polygon[:, axis] - source.bounds[side, axis])) < np.linalg.norm(source.extents) * 1e-7:
                    planes.add((axis, side))
        cap_planes.append(planes)
    for (a, b), owners in polygon_edge_faces.items():
        left, right = (polygon_normals[i] for i in owners)
        angle = np.rad2deg(np.arccos(np.clip(left @ right, -1, 1)))
        sharp = angle >= crease_angle_deg
        if implicit_surface:
            # Fourier sheet/network walls are smooth. A coarse polygon's angle
            # describes curvature, not a physical crease. Retain explicit
            # clipping planes as feature boundaries, then validate the actual
            # recovered boundary geometry after fitting.
            sharp = bool(cap_planes[owners[0]] != cap_planes[owners[1]]
                         and (cap_planes[owners[0]] or cap_planes[owners[1]]))
        if sharp:
            midpoint = edge_midpoints[a, b]
            crease_edges.extend((tuple(sorted((a, midpoint))), tuple(sorted((b, midpoint)))))
    vertices = _limit_vertex_positions(vertices, quads, crease_edges)
    return vertices, quads, {'source_triangles': len(source.faces), 'cage_triangles': len(reduced.faces),
                             'patch_count': len(quads), 'topology': original,
                             'paired_quad_count': sum(len(polygon) == 4 for polygon in polygons),
                             'crease_edges': crease_edges,
                             'parameter_initialization': 'Catmull-Clark limit vertex masks with sharp feature curves',
                             'rejected_decimations': rejected}


def _limit_vertex_positions(vertices, quads, crease_edges):
    """Regularize the parameter sites without imposing a fixed source map."""
    neighbors = [set() for _ in vertices]
    opposite = [[] for _ in vertices]
    sharp = [set() for _ in vertices]
    for quad in quads:
        for i, vertex in enumerate(quad):
            neighbors[vertex].update((int(quad[(i - 1) % 4]), int(quad[(i + 1) % 4])))
            opposite[vertex].append(int(quad[(i + 2) % 4]))
    for a, b in crease_edges:
        sharp[a].add(b)
        sharp[b].add(a)
    result = vertices.copy()
    for vertex, ring in enumerate(neighbors):
        if len(sharp[vertex]) == 2:
            result[vertex] = (4 * vertices[vertex] + vertices[list(sharp[vertex])].sum(axis=0)) / 6
        elif sharp[vertex]:
            continue
        elif len(ring) >= 3:
            n = len(ring)
            result[vertex] = (n*n * vertices[vertex] + 4 * vertices[list(ring)].sum(axis=0)
                              + vertices[opposite[vertex]].sum(axis=0)) / (n * (n + 5))
    return result


def _edge_indices(edge, inward=False):
    k = np.arange(6)
    if edge == 0:
        return k, np.full(6, 1 if inward else 0)
    if edge == 1:
        return np.full(6, 4 if inward else 5), k
    if edge == 2:
        return 5 - k, np.full(6, 4 if inward else 5)
    return np.full(6, 1 if inward else 0), 5 - k


def shared_control_indices(vertices, quads):
    """Eliminate G0 constraints by sharing every boundary control point."""
    grids = np.full((len(quads), 6, 6), -1, dtype=np.int64)
    next_id = len(vertices)
    edges = {}
    for face, quad in enumerate(quads):
        for corner, site in zip(quad, ((0, 0), (5, 0), (5, 5), (0, 5))):
            grids[face, site[0], site[1]] = corner
        for edge in range(4):
            start, end = int(quad[edge]), int(quad[(edge + 1) % 4])
            key = tuple(sorted((start, end)))
            if key not in edges:
                edges[key] = {'controls': np.r_[key[0], np.arange(next_id, next_id + 4), key[1]],
                              'faces': []}
                next_id += 4
            ids = edges[key]['controls']
            grids[face][_edge_indices(edge)] = ids if start < end else ids[::-1]
            edges[key]['faces'].append((face, edge, start < end))
        grids[face, 1:5, 1:5] = np.arange(next_id, next_id + 16).reshape(4, 4)
        next_id += 16
    return grids, edges, next_id


def gluing_constraints(vertices, quads, grids, edges, count, crease_angle_deg, crease_edges=None):
    """D_left(t) + D_right(t) = a(t) C'(t), coefficient by coefficient.

    In canonical edge direction, a has quadratic Bernstein coefficients
    [2 cos(2 pi/n_start), 0, -2 cos(2 pi/n_end)]. Derivatives of degree five
    are elevated to degree six to match a*C'. Seven equations enforce the
    identity over the entire edge, not only at sample points.
    """
    corners = vertices[quads]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    crease_keys = set(map(tuple, crease_edges)) if crease_edges is not None else set()
    incident = [[] for _ in vertices]
    for face, quad in enumerate(quads):
        for vertex in quad:
            incident[vertex].append(face)
    for key, info in edges.items():
        if len(info['faces']) != 2:
            raise SurfaceFitError('A quad cage edge has no manifold pair')
        left, right = (site[0] for site in info['faces'])
        angle = np.rad2deg(np.arccos(np.clip(normals[left] @ normals[right], -1, 1)))
        if crease_edges is None and angle >= crease_angle_deg:
            crease_keys.add(key)
    # A sharp vertex has separate smooth sectors. Applying full cyclic valence
    # masks inside a sector forces its tangent fan to wrap through 360 degrees
    # and folds clipped sheets. Use the sector's measured opening angle instead.
    sector_alpha = {}
    for vertex, faces in enumerate(incident):
        crease_degree = sum(vertex in edge for edge in crease_keys)
        adjacency = {face: set() for face in faces}
        for face in faces:
            corner = int(np.flatnonzero(quads[face] == vertex)[0])
            for neighbor in (quads[face, (corner - 1) % 4], quads[face, (corner + 1) % 4]):
                key = tuple(sorted((vertex, int(neighbor))))
                if key not in crease_keys:
                    for other, _, _ in edges[key]['faces']:
                        adjacency[face].add(other)
        remaining = set(faces)
        while remaining:
            group, pending = set(), [next(iter(remaining))]
            while pending:
                face = pending.pop()
                if face not in group:
                    group.add(face)
                    pending.extend(adjacency[face] - group)
            remaining -= group
            if len(group) == len(faces):
                angle = 2 * np.pi
            elif crease_degree == 2:
                # Two crease edges belong to one smooth feature curve. Their
                # endpoint derivatives are opposite, even when the discrete
                # cage polyline bends. Each smooth sector is a half tangent fan.
                # Using the cage's measured angle here wrongly forces the two
                # sides into the same tangent plane and folds curved end caps.
                angle = np.pi
            else:
                angle = 0.
                for face in group:
                    corner = int(np.flatnonzero(quads[face] == vertex)[0])
                    a = vertices[quads[face, (corner - 1) % 4]] - vertices[vertex]
                    b = vertices[quads[face, (corner + 1) % 4]] - vertices[vertex]
                    angle += np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            value = 2 * np.cos(angle / len(group))
            if abs(value - round(value)) < 1e-14:
                value = float(round(value))
            for face in group:
                sector_alpha[vertex, face] = value
    rows, cols, data = [], [], []
    row = 0
    creases = 0
    for (start, end), info in edges.items():
        if len(info['faces']) != 2:
            raise SurfaceFitError('A quad cage edge has no manifold pair')
        (left, _, _), (right, _, _) = info['faces']
        if (start, end) in crease_keys:
            creases += 1
            continue
        curve = info['controls']
        inward_rows = []
        for face, edge, forward in info['faces']:
            inward = grids[face][_edge_indices(edge, True)]
            inward_rows.append(inward if forward else inward[::-1])
        alpha = (sector_alpha[start, left], 0., -sector_alpha[end, left])
        for k in range(7):
            coefficients = {}
            def add(index, value):
                coefficients[int(index)] = coefficients.get(int(index), 0.) + value
            for j, weight in ((k - 1, k / 6), (k, 1 - k / 6)):
                if 0 <= j <= 5:
                    for inward in inward_rows:
                        add(inward[j], 5 * weight)
                        add(curve[j], -5 * weight)
            for i in range(3):
                j = k - i
                if 0 <= j <= 4:
                    weight = alpha[i] * comb(2, i) * comb(4, j) / comb(6, k)
                    add(curve[j + 1], -5 * weight)
                    add(curve[j], 5 * weight)
            norm = np.linalg.norm(list(coefficients.values()))
            if norm > 1e-12:
                for index, value in coefficients.items():
                    rows.append(row)
                    cols.append(index)
                    data.append(value / norm)
                row += 1
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(row, count)).tocsr()
    matrix.eliminate_zeros()
    return matrix, {'g1_edge_count': len(edges) - creases, 'crease_edge_count': creases,
                    'constraint_count': row, 'boundary_controls_shared': True}


def _fairness_matrix(grids, count):
    rows, cols, values = [], [], []
    row = 0
    for grid in grids:
        for axis in (0, 1):
            for i in range(1, 5):
                for j in range(6):
                    ids = grid[i - 1:i + 2, j] if axis == 0 else grid[j, i - 1:i + 2]
                    rows.extend((row,) * 3)
                    cols.extend(ids)
                    values.extend((1., -2., 1.))
                    row += 1
    return sparse.coo_matrix((values, (rows, cols)), shape=(row, count)).tocsr()


def fit_patches(source, vertices, quads, config: CadReconstructionConfig, *, crease_edges=None):
    """Constrained least squares in normalized model coordinates."""
    grids, edges, count = shared_control_indices(vertices, quads)
    constraints, report = gluing_constraints(vertices, quads, grids, edges, count,
                                             config.crease_angle_deg, crease_edges)
    t = np.linspace(0, 1, config.samples_per_axis)
    u, v = np.meshgrid(t, t, indexing='ij')
    u, v = u.ravel(), v.ravel()
    basis = (bernstein(5, u)[:, :, None] * bernstein(5, v)[:, None, :]).reshape(-1, 36)
    bilinear = np.stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v), axis=1)
    sample_count = len(u)
    points = np.einsum('sk,fkc->fsc', bilinear, vertices[quads]).reshape(-1, 3)
    corners = vertices[quads]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    sample_normals = np.repeat(normals, sample_count, axis=0)
    targets = []
    centre_tree = None
    for first in range(0, len(points), 256):
        query = points[first:first + 256]
        projected, _, triangles = trimesh.proximity.closest_point(source, query)
        directions = sample_normals[first:first + len(query)]
        flipped = np.einsum('ij,ij->i', source.face_normals[triangles], directions) < 0.
        if flipped.any():
            # A closest-point query on a thin sheet can jump to its opposite
            # wall. Restrict these correspondences to consistently oriented
            # nearby source triangles so thickness does not silently collapse.
            if centre_tree is None:
                centre_tree = cKDTree(source.triangles_center)
            _, candidates = centre_tree.query(query[flipped], k=min(64, len(source.faces)))
            samples = np.repeat(query[flipped], candidates.shape[1], axis=0)
            sites = trimesh.triangles.closest_point(source.triangles[candidates.ravel()], samples)
            distance = np.linalg.norm(sites - samples, axis=1).reshape(candidates.shape)
            compatible = np.einsum('ikc,ic->ik', source.face_normals[candidates], directions[flipped]) > 0.
            distance[~compatible] = np.inf
            choice = np.argmin(distance, axis=1)
            if not np.isfinite(distance[np.arange(len(choice)), choice]).all():
                raise SurfaceFitError('The cage has no consistently oriented source correspondence')
            projected[flipped] = sites.reshape(*candidates.shape, 3)[np.arange(len(choice)), choice]
        targets.append(projected)
    targets = np.concatenate(targets)
    origin = source.bounds.mean(axis=0)
    scale = float(np.linalg.norm(source.extents))
    targets = (targets - origin) / scale
    row_ids = np.repeat(np.arange(len(points)), 36)
    col_ids = np.repeat(grids.reshape(-1, 1, 36), sample_count, axis=1).ravel()
    fit = sparse.coo_matrix((np.tile(basis.ravel(), len(quads)), (row_ids, col_ids)),
                            shape=(len(points), count)).tocsr()
    fit.eliminate_zeros()
    fair = _fairness_matrix(grids, count)
    hessian = fit.T @ fit + config.fairness * (fair.T @ fair) + sparse.eye(count) * 1e-9
    rhs = fit.T @ targets
    # Endpoint derivatives control the orientation of extraordinary-vertex
    # tangent fans. Penalize oscillatory derivative reversal independently of
    # positional fitting, while the exact G1 equations project these targets
    # into the compatible tangent space (including sharp vertex sectors).
    tangent_rows, tangent_cols, tangent_values, tangent_targets = [], [], [], []
    for (start, end), edge in edges.items():
        for vertex, neighbor, pole in ((start, end, edge['controls'][1]),
                                      (end, start, edge['controls'][-2])):
            row = len(tangent_targets)
            tangent_rows.extend((row, row))
            tangent_cols.extend((int(pole), vertex))
            tangent_values.extend((1., -1.))
            tangent_targets.append((vertices[neighbor] - vertices[vertex]) / (5 * scale))
    tangent = sparse.coo_matrix((tangent_values, (tangent_rows, tangent_cols)),
                                shape=(len(tangent_targets), count)).tocsr()
    hessian += config.tangent_stiffness * (tangent.T @ tangent)
    rhs += config.tangent_stiffness * (tangent.T @ np.asarray(tangent_targets))
    multipliers = np.zeros((constraints.shape[0], 3))
    repairs = []
    for repair_iteration in range(4):
        controls, multipliers, residual, iteration = _constrained_fit(hessian, rhs, constraints, config, multipliers)
        poles = controls[grids] * scale + origin
        try:
            _assert_regular_patches(poles, vertices, quads, source)
            break
        except SurfaceFitError as error:
            failed = getattr(error, 'failed_patches', [])
            if repair_iteration == 3 or not len(failed) or len(failed) > 8:
                raise
            # Local oscillation repair adds a convex target on the affected
            # control grids and resolves the entire coupled G1 system. Accuracy,
            # orientation and volume checks still apply to the repaired shape.
            face_targets = targets.reshape(len(quads), len(t), len(t), 3)[failed]
            corner_targets = face_targets[:, (0, -1, -1, 0), (0, 0, -1, -1)]
            parameter = np.linspace(0, 1, 6)
            a, b = np.meshgrid(parameter, parameter, indexing='ij')
            weights = np.stack(((1-a)*(1-b), a*(1-b), a*b, (1-a)*b), axis=-1)
            template = np.einsum('ijk,fkc->fijc', weights, corner_targets).reshape(-1, 3)
            columns = grids[failed].ravel()
            rows = np.arange(len(columns))
            anchor = sparse.coo_matrix((np.ones(len(columns)), (rows, columns)),
                                       shape=(len(columns), count)).tocsr()
            strength = 10. ** repair_iteration
            hessian += strength * (anchor.T @ anchor)
            rhs += strength * (anchor.T @ template)
            repairs.append({'patch_count': len(failed), 'strength': strength})
    report.update(solver='Sparse augmented Lagrangian', solver_iterations=iteration + 1,
                  normalized_constraint_residual=residual, control_point_count=count,
                  fitting_rms_distance=float(np.sqrt(np.mean((fit @ controls - targets)**2))) * scale,
                  degree_u=5, degree_v=5, rational=False)
    report['orientation_repairs'] = repairs
    report['sampled_jacobians_valid'] = True
    return poles, report


def _constrained_fit(hessian, rhs, constraints, config, multipliers):
    penalty = 1e5
    # Symmetric ordering avoids general LU fill-in on coupled TPMS patch graphs.
    solve = splu((hessian + penalty * (constraints.T @ constraints)).tocsc(),
                  permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.,
                  options={'SymmetricMode': True})
    residual = float('inf')
    for iteration in range(config.constraint_iterations):
        controls = solve.solve(rhs - constraints.T @ multipliers)
        error = constraints @ controls
        residual = float(np.max(np.abs(error))) if error.size else 0.
        if residual <= config.constraint_tolerance:
            break
        multipliers += penalty * error
    if residual > config.constraint_tolerance:
        raise SurfaceFitError(f'G1 constraint solver did not converge ({residual:.3g})')
    return controls, multipliers, residual, iteration


def evaluate_patches(poles, parameters):
    basis = bernstein(5, parameters)
    return np.einsum('ui,fijc,vj->fuvc', basis, poles, basis)


def _assert_regular_patches(poles, vertices, quads, source=None):
    t = np.linspace(1e-5, 1 - 1e-5, 11)
    basis, derivative = bernstein(5, t), bernstein(4, t)
    du = np.einsum('ui,fijc,vj->fuvc', derivative, 5 * np.diff(poles, axis=1), basis)
    dv = np.einsum('ui,fijc,vj->fuvc', basis, 5 * np.diff(poles, axis=2), derivative)
    cross = np.cross(du, dv)
    corners = vertices[quads]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    direction = np.einsum('fuvc,fc->fuv', cross, normal)
    threshold = np.linalg.norm(normal, axis=1)**2 * 1e-8
    bad = direction <= threshold[:, None, None]
    sine = np.linalg.norm(cross, axis=-1) / np.maximum(np.linalg.norm(du, axis=-1) * np.linalg.norm(dv, axis=-1), 1e-30)
    if source is not None and bad.any():
        # A curved patch may turn beyond its coarse cage normal while remaining
        # regular and correctly oriented on the actual source. Resolve those
        # sites against the recovered boundary, rather than imposing a planar
        # height-field restriction on an arbitrary curved sheet.
        positions = evaluate_patches(poles, t)
        indices = np.argwhere(bad)
        for first in range(0, len(indices), 128):
            sites = indices[first:first + 128]
            key = tuple(sites.T)
            _, distance, triangle_ids = trimesh.proximity.closest_point(source, positions[key])
            orientation = np.einsum('ij,ij->i', cross[key], source.face_normals[triangle_ids])
            bad[key] = orientation <= 0.
            ambiguous = orientation <= 0.
            if ambiguous.any():
                # At a sharp source edge, several faces can have the exact same
                # closest point. A single arbitrarily selected triangle normal
                # is not an orientation reference for both sides of that edge.
                # Accept only equally near faces, with a small numeric tie bound.
                query = positions[key][ambiguous]
                _, candidates = cKDTree(source.triangles_center).query(query, k=min(64, len(source.faces)))
                repeated = np.repeat(query, candidates.shape[1], axis=0)
                closest = trimesh.triangles.closest_point(source.triangles[candidates.ravel()], repeated)
                distances = np.linalg.norm(closest - repeated, axis=1).reshape(candidates.shape)
                tied = distances <= distance[ambiguous, None] + np.linalg.norm(source.extents) * 1e-7
                aligned = np.einsum('ikc,ic->ik', source.face_normals[candidates], cross[key][ambiguous]) > 0.
                resolved = np.any(tied & aligned, axis=1)
                bad[tuple(sites[ambiguous].T)] = ~resolved
    bad |= sine <= 1e-8
    failed = np.flatnonzero(np.any(bad, axis=(1, 2)))
    if len(failed):
        unit = cross / np.maximum(np.linalg.norm(cross, axis=-1)[..., None], 1e-30)
        adjacent = min(float(np.min(np.sum(unit[:, :-1] * unit[:, 1:], axis=-1))),
                       float(np.min(np.sum(unit[:, :, :-1] * unit[:, :, 1:], axis=-1))))
        axis_aligned = np.any(np.ptp(corners[failed], axis=1) <= np.linalg.norm(np.ptp(vertices, axis=0)) * 1e-7, axis=1)
        error = SurfaceFitError(f'Fitting folded or collapsed {len(failed)} patch(es) '
                                f'({int(axis_aligned.sum())} on clipping planes; minimum Jacobian sine {sine.min():.3g}, adjacent normal dot {adjacent:.3g})')
        error.failed_patches = failed
        raise error


def patch_mesh(poles, samples=13):
    grids = evaluate_patches(poles, np.linspace(0, 1, samples))
    faces = []
    for face in range(len(poles)):
        offset = face * samples**2
        for i in range(samples - 1):
            for j in range(samples - 1):
                a = offset + i * samples + j
                faces.extend(((a, a + samples, a + samples + 1), (a, a + samples + 1, a + 1)))
    return grids.reshape(-1, 3), np.asarray(faces, dtype=np.int64)


def assemble_solid(poles, tolerance, *, quads=None, crease_edges=(),
                   maximum_chart_patches=64, maximum_chart_spans=16):
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
            maximum_chart_patches=maximum_chart_patches, maximum_chart_spans=maximum_chart_spans,
            boundary_tolerance=max(1e-10, float(tolerance) * 1e-3))
    sewer = BRepBuilderAPI_Sewing(float(tolerance), True, True, True, False)
    if quads is not None:
        for face in chart_faces(charts, quads, tolerance):
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
    return result, {'solid_count': len(bodies), 'boundary_shell_count': len(shells),
                    'cavity_shell_count': len(inners), 'cad_face_count': len(result.Faces()),
                    'surface_consolidation': consolidation}
