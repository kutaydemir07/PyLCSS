# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Absolute OpenCASCADE tessellation in model units."""
from typing import Any
import numpy as np

def private_cad_mesh_copy(shape: Any):
    """Isolate CAD topology and geometry before meshing can populate pcurves."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy

    shape = shape.val() if hasattr(shape, "val") else shape
    if not hasattr(shape, "wrapped"):
        return shape  # Non-CAD display providers retain their existing protocol.
    return cq.Shape.cast(BRepBuilderAPI_Copy(shape.wrapped, True, False).Shape())


def tessellate_cad_absolute(shape: Any, tolerance: float, angular_tolerance: float = 0.5,
                            *, return_face_ranges: bool = False,
                            max_triangles: int | None = None, max_vertices: int | None = None,
                            missing_face_mesh_parameters: tuple[tuple[float, float], ...] = ()):
    """Mesh a CAD shape with absolute model-unit deflection, preserving seams.

    CadQuery's Shape.tessellate uses relative OCC deflection. Extract the OCC
    triangulation directly so it cannot replace this absolute mesh implicitly.
    Display callers may recover missing faces in isolation; this can change
    their boundary sampling and is therefore disabled for export by default.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    retry_parameters = tuple(missing_face_mesh_parameters)
    if (len(retry_parameters) > 2
            or any(len(pair) != 2 or not np.all(np.isfinite(pair)) or min(pair) <= 0
                   for pair in retry_parameters)):
        raise ValueError("Missing-face recovery permits at most two finite positive meshing parameter pairs")

    # OCC meshing updates pcurves as well as triangulation caches. Copy the
    # geometry too; a topology-only copy can still mutate shared CAD surfaces.
    source = shape.val() if hasattr(shape, "val") else shape
    source_faces = source.Faces()
    shape = private_cad_mesh_copy(source)
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, float(tolerance), False,
                                    float(angular_tolerance), True)
    if not mesher.IsDone():
        raise RuntimeError("Absolute CAD tessellation failed")
    triangulated_faces = []
    vertex_count = triangle_count = 0
    faces = shape.Faces()
    if len(faces) != len(source_faces):
        raise RuntimeError("Private CAD tessellation copy changed the native face count")
    for face_index, face in enumerate(faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if retry_parameters and (triangulation is None or triangulation.NbNodes() == 0
                                 or triangulation.NbTriangles() == 0):
            # Whole-body meshing can leave a small trim without a mesh.
            # Its edges/pcurves may already carry shared meshing state, so
            # retry from the untouched original face at the caller's fixed,
            # bounded parameters. Preserve the first success and keep its
            # private face alive through extraction.
            for retry_tolerance, retry_angle in retry_parameters:
                face = private_cad_mesh_copy(source_faces[face_index])
                BRepTools.Clean_s(face.wrapped)
                retry = BRepMesh_IncrementalMesh(face.wrapped, float(retry_tolerance), False,
                                                 float(retry_angle), True)
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
                if (retry.IsDone() and triangulation is not None
                        and triangulation.NbNodes() > 0 and triangulation.NbTriangles() > 0):
                    break
                triangulation = None
        elif triangulation is None:
            BRepMesh_IncrementalMesh(face.wrapped, float(tolerance), False,
                                     float(angular_tolerance), True)
            triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if triangulation is not None:
            vertex_count += triangulation.NbNodes()
            triangle_count += triangulation.NbTriangles()
        triangulated_faces.append((face, location, triangulation))
    # Check before allocating Python coordinate/triangle buffers. This limits
    # export data, not native OCCT meshing memory or its execution time.
    if ((max_triangles is not None and triangle_count > max_triangles)
            or (max_vertices is not None and vertex_count > max_vertices)):
        raise ValueError(f'CAD tessellation exceeds the export mesh limit '
                         f'({vertex_count} vertices, {triangle_count} triangles).')
    vertices, triangles, face_ranges = [], [], []
    for face, location, triangulation in triangulated_faces:
        if triangulation is None:
            face_ranges.append((len(vertices), len(vertices), len(triangles), len(triangles)))
            continue
        transform = location.Transformation()
        offset = len(vertices)
        triangle_offset = len(triangles)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))
        reversed_face = face.wrapped.Orientation() == TopAbs_REVERSED
        for triangle in triangulation.Triangles():
            a, b, c = triangle.Get()
            triangles.append((offset + a - 1, offset + (c if reversed_face else b) - 1,
                              offset + (b if reversed_face else c) - 1))
        face_ranges.append((offset, len(vertices), triangle_offset, len(triangles)))
    result = (np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))
    return (*result, face_ranges) if return_face_ranges else result

