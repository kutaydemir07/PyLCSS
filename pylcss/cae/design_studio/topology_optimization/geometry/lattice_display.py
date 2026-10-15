"""Bounded lattice display data; the manufacturing/export mesh stays intact."""
from __future__ import annotations

import numpy as np

MAX_LATTICE_DISPLAY_FACES = 150_000


def prepare_lattice_display(recovered, *, max_faces=MAX_LATTICE_DISPLAY_FACES):
    """Prepare a closed topology-preserving LOD and normals outside Qt."""
    import trimesh
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    vertices = np.asarray(recovered['vertices'], dtype=float)
    faces = np.asarray(recovered['faces'])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('The lattice display needs finite, indexed surface triangles')
    if not 4 <= max_faces <= MAX_LATTICE_DISPLAY_FACES:
        raise ValueError('The lattice display face limit is invalid')
    source = trimesh.Trimesh(vertices, faces, process=False)
    source.remove_unreferenced_vertices()
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    if not source.is_watertight or not source.is_winding_consistent:
        raise ValueError('The lattice display requires a closed oriented material surface')

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1, dtype=np.int64)*3, deep=True),
                  numpy_to_vtkIdTypeArray(np.asarray(faces, dtype=np.int64).ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    report = {'source_faces': len(faces), 'maximum_display_faces': max_faces,
              'manufacturing_mesh_changed': False, 'display_only': True,
              'method': 'Link-condition quadric display with independent topology checks'}
    if len(faces) > max_faces:
        from .amrto_quadrangulation import _topology_preserving_decimation
        # This existing MeshLab backend checks the edge-collapse link condition.
        # VTK's reducers can create nonmanifold edges on high-genus TPMS input.
        candidate = _topology_preserving_decimation(vertices, faces, max_faces)
        candidate.remove_unreferenced_vertices()
        if len(candidate.faces) > max_faces:
            raise ValueError('The lattice cannot be displayed within the safe triangle limit '
                             'while preserving its topology. The complete export mesh is retained.')
        if (not candidate.is_watertight or not candidate.is_winding_consistent
                or candidate.euler_number != source.euler_number
                or candidate.body_count != source.body_count):
            raise ValueError('The reduced lattice display changed its closed surface topology')
        def shell_signature(mesh):
            return sorted((int(part.euler_number), int(np.sign(part.volume)))
                          for part in mesh.split(only_watertight=False))
        if shell_signature(candidate) != shell_signature(source):
            raise ValueError('The reduced lattice display changed a shell genus or cavity orientation')
        volume_error = abs(float(candidate.volume)-float(source.volume)) / max(abs(float(source.volume)), 1e-30)
        if volume_error > .05:
            raise ValueError('The reduced lattice display changed material volume by more than 5%')
        report.update(reduced=True, source_euler=int(source.euler_number),
                      display_euler=int(candidate.euler_number),
                      relative_display_volume_error=volume_error)
        # Some native reducers retain unused input points. Publish only the
        # compact, independently checked surface for Qt and section capping.
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(candidate.vertices), deep=True))
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(candidate.faces)+1, dtype=np.int64)*3, deep=True),
                      numpy_to_vtkIdTypeArray(np.asarray(candidate.faces, dtype=np.int64).ravel(), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
    else:
        report['reduced'] = False

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    # Keep the welded surface for section capping and material cavity winding.
    normals.SplittingOff()
    normals.SetFeatureAngle(50.)
    normals.ComputePointNormalsOn()
    normals.Update()
    prepared = normals.GetOutput()
    display_vertices = np.asarray(vtk_to_numpy(prepared.GetPoints().GetData())).copy()
    display_faces = np.asarray(vtk_to_numpy(prepared.GetPolys().GetConnectivityArray()), dtype=np.int64).reshape(-1, 3).copy()
    point_normals = np.asarray(vtk_to_numpy(prepared.GetPointData().GetNormals()), dtype=np.float32).copy()
    if not np.isfinite(point_normals).all() or len(display_faces) > max_faces:
        raise ValueError('The prepared lattice display has invalid normals or excessive triangles')
    report.update(display_faces=len(display_faces), display_vertices=len(display_vertices))
    return {'vertices': display_vertices, 'faces': display_faces,
            'point_normals': point_normals, 'report': report}
