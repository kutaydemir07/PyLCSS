# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Recover compact analytic bodies before constructing a spline control cage.

Recognition uses the source geometry, not a failed spline fit. Candidates must
preserve shell topology, material volume and two-way geometric deviation. A
coaxial stack supports pipes, annular flanges and stepped shafts as native
cylinders and planes. Freeform geometry remains on the local cubic surface route.
"""
from __future__ import annotations

import numpy as np
import cadquery as cq
import trimesh

from .cad_diagnostics import assess_sampled_surface_deviation
from .cad_tessellation import tessellate_cad_absolute
from .mesh_topology import mesh_topology_report, topology_error


def _frame(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first), axis))


def _sphere(points, tolerance):
    matrix = np.column_stack((2*points, np.ones(len(points))))
    coefficients = np.linalg.lstsq(matrix, np.sum(points**2, axis=1), rcond=None)[0]
    centre = coefficients[:3]
    radius = np.sqrt(max(0., coefficients[3]+centre @ centre))
    if radius <= 0 or np.max(np.abs(np.linalg.norm(points-centre, axis=1)-radius)) > tolerance:
        return None
    return centre, radius


def _oriented_box(source, origin, tolerance):
    # Surface normals identify axes even when a cube's PCA eigenvalues repeat.
    _, ids = np.unique(np.round(source.face_normals, 6), axis=0, return_index=True)
    if len(ids) != 6:
        return None
    directions = []
    for normal in source.face_normals[ids]:
        if all(abs(normal @ previous) < 1-1e-6 for previous in directions):
            directions.append(normal)
    if len(directions) != 3 or np.max(np.abs(np.array(directions) @ np.array(directions).T-np.eye(3))) > 1e-6:
        return None
    first, second = directions[:2]
    frame = np.column_stack((first, second, np.cross(first, second)))
    local = (source.vertices-origin) @ frame
    lower, upper = local.min(axis=0), local.max(axis=0)
    distance = np.min(np.minimum(np.abs(local-lower), np.abs(local-upper)), axis=1)
    if np.max(distance) > tolerance:
        return None
    centre = origin+frame @ ((lower+upper)*.5)
    plane = cq.Plane(origin=tuple(centre), xDir=tuple(first), normal=tuple(frame[:, 2]))
    return cq.Workplane(plane).box(*(upper-lower)).val(), {'family': 'Six recovered planes'}


def _extruded_profile(source, axis, origin, tolerance):
    """Recover an invariant cross-section as few native extruded surfaces.

    This is a geometry candidate, not a substitute for the mesh: the caller
    checks its complete boundary, volume, and topology before accepting it.
    """
    frame = _frame(axis)
    local = (source.vertices - origin) @ frame
    lower, upper = local[:, 2].min(), local[:, 2].max()
    thickness = upper - lower
    if thickness <= tolerance:
        return None
    # Only attempt a sweep when both end planes exist. The full two-way check
    # below rejects tapered, bent, or otherwise changing cross-sections.
    cap = np.abs(source.face_normals @ axis) > .999
    if np.count_nonzero(cap) < 2:
        return None
    section = source.section(plane_origin=origin, plane_normal=axis)
    if section is None:
        return None
    contours = []
    for contour in section.discrete:
        points = (np.asarray(contour) - origin) @ frame
        if len(points) < 4 or np.linalg.norm(points[0] - points[-1]) > max(tolerance, 1e-6):
            return None
        points = points[:, :2]
        area = .5 * np.sum(points[:-1, 0]*points[1:, 1] - points[1:, 0]*points[:-1, 1])
        if abs(area) > tolerance**2:
            contours.append((abs(area), points))
    if not contours:
        return None
    contours.sort(key=lambda item: item[0], reverse=True)
    wires = []
    for index, (_, points) in enumerate(contours):
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.], np.cumsum(lengths)))
        perimeter = cumulative[-1]
        if perimeter <= 0:
            return None
        count = int(np.clip(np.ceil(perimeter / max(2*tolerance, perimeter/256)), 32, 256))
        sample = np.linspace(0., perimeter, count, endpoint=False)
        profile = np.column_stack([np.interp(sample, cumulative, points[:, i]) for i in range(2)])
        signed_area = np.sum(profile[:, 0]*np.roll(profile[:, 1], -1) - profile[:, 1]*np.roll(profile[:, 0], -1))
        if (signed_area > 0) != (index == 0):
            profile = profile[::-1]
        world = origin + profile @ frame[:, :2].T + lower*axis
        edge = cq.Edge.makeSpline([cq.Vector(*point) for point in world], periodic=True)
        wires.append(cq.Wire.assembleEdges([edge]))
    try:
        shape = cq.Solid.extrudeLinear(wires[0], wires[1:], tuple(thickness*axis))
        return shape, {'family': 'Recovered extruded profile', 'profile_count': len(wires)}
    except (ValueError, RuntimeError):
        return None


def _coaxial_stack(source, frame, origin, scale, tolerance):
    points = (source.vertices-origin) @ frame / scale
    centres = (source.triangles_center-origin) @ frame / scale
    normals = source.face_normals @ frame
    side = np.abs(normals[:, 2]) < .05
    if np.sum(side) < 8:
        return None
    # On a faceted cylinder, its face normal passes through the true axis.
    transverse = np.column_stack((-normals[side, 1], normals[side, 0]))
    centre = np.linalg.lstsq(transverse, np.sum(transverse*centres[side, :2], axis=1), rcond=None)[0]
    vertices = points[source.faces[side]]
    radii = np.linalg.norm(vertices[:, :, :2]-centre, axis=2)
    mean_radius = radii.mean(axis=1)
    if np.max(np.ptp(radii, axis=1)) > tolerance/scale:
        return None
    alignment = np.sum((centres[side, :2]-centre)*normals[side, :2], axis=1)
    sorted_ids = np.argsort(mean_radius)
    groups = np.split(sorted_ids, np.flatnonzero(np.diff(mean_radius[sorted_ids]) > tolerance/scale)+1)
    outer, inner = [], []
    for group in groups:
        radius = float(np.median(radii[group]))*scale
        lower, upper = float(vertices[group, :, 2].min())*scale, float(vertices[group, :, 2].max())*scale
        if radius <= tolerance or upper-lower <= tolerance:
            return None
        if np.ptp(radii[group]) > 2*tolerance/scale:
            return None
        base = origin+frame @ np.array((centre[0]*scale, centre[1]*scale, lower))
        cylinder = cq.Solid.makeCylinder(radius, upper-lower, cq.Vector(*base), cq.Vector(*frame[:, 2]))
        target = outer if np.median(alignment[group]) > 0 else inner
        target.append(cylinder)
    if not outer:
        return None
    shape = outer[0]
    if len(outer) > 1:
        shape = shape.fuse(*outer[1:])
    if inner:
        shape = shape.cut(*inner)
    return shape.clean(), {'family': 'Coaxial pipes, flanges and stepped cylinders',
                           'outer_cylinder_count': len(outer), 'inner_cylinder_count': len(inner)}


def recover_analytic_solid(source, *, fit_tolerance, deviation_limit, maximum_volume_delta,
                           validation_samples=1000):
    """Return a recognized, independently verified body or no classification."""
    origin = source.bounds.mean(axis=0)
    scale = float(np.linalg.norm(source.extents))
    tolerance = max(float(fit_tolerance), scale*1e-7)
    normalized = (source.vertices-origin)/scale
    original = mesh_topology_report(source.vertices, source.faces)
    if original['connected_components'] != 1:
        return None
    _, axes = np.linalg.eigh(np.cov(normalized.T))
    candidates = []
    box = _oriented_box(source, origin, tolerance) if original['euler_characteristic'] == 2 else None
    if box is not None:
        candidates.append(box)
    sphere = _sphere(normalized, tolerance/scale)
    if sphere is not None:
        centre, radius = sphere
        candidates.append((cq.Workplane('XY').sphere(radius*scale).translate(tuple(origin+centre*scale)).val(),
                           {'family': 'Sphere'}))
    for axis in range(3):
        frame = _frame(axes[:, axis])
        local = normalized @ frame
        extent = np.ptp(local, axis=0)
        # Six planes determine an oriented box; every source vertex must lie
        # on its boundary. Subsequent two-way checks reject missing features.
        plane_distance = np.min(np.minimum(np.abs(local-local.min(axis=0)),
                                            np.abs(local-local.max(axis=0))), axis=1)
        if np.max(plane_distance) <= tolerance/scale:
            centre = origin+frame @ (local.min(axis=0)+local.max(axis=0))*scale*.5
            plane = cq.Plane(origin=tuple(centre), xDir=tuple(frame[:, 0]), normal=tuple(frame[:, 2]))
            candidates.append((cq.Workplane(plane).box(*(extent*scale)).val(), {'family': 'Six recovered planes'}))
        stack = _coaxial_stack(source, frame, origin, scale, tolerance)
        if stack is not None:
            candidates.append(stack)
        rho = np.linalg.norm(local[:, :2], axis=1)
        reduced = np.column_stack((rho, local[:, 2], np.zeros(len(rho))))
        # Circle fitting in (radial distance, axial position) identifies a
        # torus without introducing a dense rectangular patch network.
        matrix = np.column_stack((2*reduced[:, :2], np.ones(len(reduced))))
        c = np.linalg.lstsq(matrix, np.sum(reduced[:, :2]**2, axis=1), rcond=None)[0]
        major, axial = c[:2]
        minor = np.sqrt(max(0., c[2]+major**2+axial**2))
        if major > minor > 0 and np.max(np.abs(np.hypot(rho-major, local[:, 2]-axial)-minor)) <= tolerance/scale:
            centre = origin+frame[:, 2]*axial*scale
            candidates.append((cq.Solid.makeTorus(major*scale, minor*scale, cq.Vector(*centre),
                                                 cq.Vector(*frame[:, 2])), {'family': 'Torus'}))
    # Extruded topology boundaries can contain several through-holes. Recover
    # their closed profiles instead of creating a CAD face for every mesh patch.
    for axis in range(3):
        try:
            extrusion = _extruded_profile(source, axes[:, axis], origin, tolerance)
        except Exception:
            # Degenerate section wires are a rejected recognition candidate;
            # they must not interrupt the remaining surface-recovery route.
            extrusion = None
        if extrusion is not None:
            candidates.append(extrusion)
    for shape, recognition in candidates:
        if not shape.isValid() or not shape.Solids():
            continue
        volume_delta = abs(shape.Volume()-source.volume)/source.volume
        if maximum_volume_delta and volume_delta > maximum_volume_delta:
            continue
        vertices, faces = tessellate_cad_absolute(shape, max(scale*1e-4, tolerance*.1), .2)
        welded = trimesh.Trimesh(vertices, faces, process=True, validate=True)
        if topology_error(original, mesh_topology_report(welded.vertices, welded.faces)):
            continue
        deviation = assess_sampled_surface_deviation(source.vertices, source.faces, shape,
                                                     max(scale*1e-4, tolerance*.1), samples=validation_samples)
        if deviation_limit and deviation['maximum_distance'] > deviation_limit:
            continue
        if fit_tolerance and deviation['rms_distance'] > fit_tolerance:
            continue
        recognition.update(input_triangle_count=len(source.faces), output_face_count=len(shape.Faces()),
                           relative_volume_delta=float(volume_delta), deviation=deviation,
                           source_geometry_verified=True)
        recognition['surface_types'] = {}
        for face in shape.Faces():
            kind = face.geomType()
            recognition['surface_types'][kind] = recognition['surface_types'].get(kind, 0)+1
        return shape, recognition
    return None
