"""Bounded lattice display data; the manufacturing/export mesh stays intact."""
from __future__ import annotations

import numpy as np

LATTICE_DISPLAY_REVISION = 2
LATTICE_DISPLAY_FACE_BUDGETS = {
    'Standard': 300_000,
    'High': 1_500_000,
    'Professional': 4_000_000,
}
MAX_LATTICE_DISPLAY_FACES = max(LATTICE_DISPLAY_FACE_BUDGETS.values())
LATTICE_DISPLAY_CREASE_ANGLE = 55.


def lattice_display_quality(payload):
    """Resolve saved settings, including projects predating display quality."""
    settings = (payload.get('generated_lattice') or {}).get('settings') or {}
    value = settings.get('post_lattice_quality') or payload.get('post_lattice_quality') or 'High'
    return value if value in LATTICE_DISPLAY_FACE_BUDGETS else 'High'


def _surface_polydata(vertices, faces):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1, dtype=np.int64)*3, deep=True),
                  numpy_to_vtkIdTypeArray(np.asarray(faces, dtype=np.int64).ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _sampled_surface_deviation(source, candidate, samples=2048):
    """Check both boundaries at deterministic vertices and triangle centres."""
    import vtk

    maximum = 0.
    for probes, target in ((source, candidate), (candidate, source)):
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(_surface_polydata(target.vertices, target.faces))
        vertex_indices = np.linspace(0, len(probes.vertices)-1,
                                     min(samples, len(probes.vertices)), dtype=np.int64)
        face_indices = np.linspace(0, len(probes.faces)-1,
                                   min(samples, len(probes.faces)), dtype=np.int64)
        sites = np.concatenate((probes.vertices[vertex_indices],
                                probes.vertices[probes.faces[face_indices]].mean(axis=1)))
        maximum = max(maximum, max(abs(float(distance.EvaluateFunction(point))) for point in sites))
    return maximum


def prepare_lattice_display(recovered, *, quality='High', max_faces=None):
    """Prepare checked detail and crease-aware lighting in the CAD process."""
    import trimesh
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vertices = np.asarray(recovered['vertices'], dtype=float)
    faces = np.asarray(recovered['faces'])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('The lattice display needs finite, indexed surface triangles')
    if quality not in LATTICE_DISPLAY_FACE_BUDGETS:
        raise ValueError('Choose Standard, High, or Professional lattice display quality')
    max_faces = LATTICE_DISPLAY_FACE_BUDGETS[quality] if max_faces is None else max_faces
    if (isinstance(max_faces, bool) or not isinstance(max_faces, (int, np.integer))
            or not 4 <= max_faces <= MAX_LATTICE_DISPLAY_FACES):
        raise ValueError('The lattice display face limit is invalid')
    source = trimesh.Trimesh(vertices, faces, process=False)
    source.remove_unreferenced_vertices()
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    if not source.is_watertight or not source.is_winding_consistent:
        raise ValueError('The lattice display requires a closed oriented material surface')

    report = {'source_faces': len(faces), 'maximum_display_faces': max_faces,
              'manufacturing_mesh_changed': False, 'display_only': True,
              'revision': LATTICE_DISPLAY_REVISION, 'quality': quality,
              'absolute_maximum_display_faces': MAX_LATTICE_DISPLAY_FACES,
              'normal_crease_angle_degrees': LATTICE_DISPLAY_CREASE_ANGLE,
              'method': 'Quality-aware detail with checked topology, volume and sampled surface deviation'}
    selected = source
    if len(faces) > max_faces:
        from .amrto_quadrangulation import _topology_preserving_decimation
        # This existing MeshLab backend checks the edge-collapse link condition.
        # VTK's reducers can create nonmanifold edges on high-genus TPMS input.
        def shell_signature(mesh):
            return sorted((int(part.euler_number), int(np.sign(part.volume)))
                          for part in mesh.split(only_watertight=False))
        source_shells = shell_signature(source)
        diagonal = float(np.linalg.norm(np.ptp(source.vertices, axis=0)))
        allowed_deviation = diagonal * {'Standard': .0015, 'High': .00075, 'Professional': .0004}[quality]
        allowed_volume_error = {'Standard': .05, 'High': .02, 'Professional': .01}[quality]
        report.update(allowed_sampled_surface_deviation=allowed_deviation,
                      allowed_relative_volume_error=allowed_volume_error)
        attempts = []
        budgets = [int(max_faces)]
        if len(source.faces) > MAX_LATTICE_DISPLAY_FACES and max_faces < MAX_LATTICE_DISPLAY_FACES:
            budgets.append(MAX_LATTICE_DISPLAY_FACES)
        selected = None
        for budget in budgets:
            try:
                candidate = _topology_preserving_decimation(vertices, faces, budget)
                candidate.remove_unreferenced_vertices()
                if len(candidate.faces) > budget:
                    raise ValueError('The reducer could not meet the triangle budget')
                if (not candidate.is_watertight or not candidate.is_winding_consistent
                        or candidate.euler_number != source.euler_number
                        or candidate.body_count != source.body_count
                        or shell_signature(candidate) != source_shells):
                    raise ValueError('The reduced display changed a closed shell or its topology')
                volume_error = abs(float(candidate.volume)-float(source.volume)) / max(abs(float(source.volume)), 1e-30)
                if volume_error > allowed_volume_error:
                    raise ValueError('The reduced display exceeded its volume error allowance')
                deviation = _sampled_surface_deviation(source, candidate)
                if deviation > allowed_deviation:
                    raise ValueError('The reduced display exceeded its sampled surface deviation allowance')
                selected = candidate
                report.update(source_euler=int(source.euler_number), display_euler=int(candidate.euler_number),
                              relative_display_volume_error=volume_error,
                              sampled_surface_deviation=deviation)
                break
            except (ValueError, RuntimeError) as exc:
                attempts.append({'face_budget': budget, 'reason': str(exc)})
        if selected is None:
            if len(source.faces) > MAX_LATTICE_DISPLAY_FACES:
                raise ValueError('The lattice cannot be reduced within the safe display limit while preserving '
                                 'surface detail. The complete export mesh is retained. ' + attempts[-1]['reason'])
            # Prefer the complete surface over a smaller display which changes
            # thin struts. Its size is still bounded by the absolute limit.
            selected = source
        report.update(reduced=selected is not source, rejected_reductions=attempts,
                      quality_budget_exceeded=len(selected.faces) > max_faces)
    else:
        report['reduced'] = False

    poly = _surface_polydata(selected.vertices, selected.faces)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    # Duplicate shading vertices only at real creases; retain the welded
    # oriented surface separately for section capping and pore winding.
    normals.SplittingOn()
    normals.SetFeatureAngle(LATTICE_DISPLAY_CREASE_ANGLE)
    normals.ComputePointNormalsOn()
    normals.Update()
    prepared = normals.GetOutput()
    display_vertices = np.asarray(vtk_to_numpy(prepared.GetPoints().GetData())).copy()
    display_faces = np.asarray(vtk_to_numpy(prepared.GetPolys().GetConnectivityArray()), dtype=np.int64).reshape(-1, 3).copy()
    point_normals = np.asarray(vtk_to_numpy(prepared.GetPointData().GetNormals()), dtype=np.float32).copy()
    if not np.isfinite(point_normals).all() or len(display_faces) > MAX_LATTICE_DISPLAY_FACES:
        raise ValueError('The prepared lattice display has invalid normals or excessive triangles')
    report.update(display_faces=len(display_faces), display_vertices=len(display_vertices),
                  effective_face_budget=max(int(max_faces), len(display_faces)))
    return {'vertices': display_vertices, 'faces': display_faces,
            'point_normals': point_normals, 'report': report,
            'section_mesh': {'vertices': np.asarray(selected.vertices).copy(),
                             'faces': np.asarray(selected.faces, dtype=np.int64).copy()}}
