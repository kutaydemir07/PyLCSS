# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Absolute OpenCASCADE tessellation in model units."""
from typing import Any
import numpy as np

def tessellate_cad_absolute(shape: Any, tolerance: float, angular_tolerance: float = 0.5,
                            *, return_face_ranges: bool = False):
    """Mesh a CAD shape with absolute model-unit deflection, preserving seams.

    CadQuery's Shape.tessellate uses relative OCC deflection. Extract the OCC
    triangulation directly so it cannot replace this absolute mesh implicitly.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    shape = shape.val() if hasattr(shape, "val") else shape
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, float(tolerance), False,
                                    float(angular_tolerance), True)
    if not mesher.IsDone():
        raise RuntimeError("Absolute CAD tessellation failed")
    vertices, triangles, face_ranges = [], [], []
    for face in shape.Faces():
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if triangulation is None:
            BRepMesh_IncrementalMesh(face.wrapped, float(tolerance), False,
                                     float(angular_tolerance), True)
            triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
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

