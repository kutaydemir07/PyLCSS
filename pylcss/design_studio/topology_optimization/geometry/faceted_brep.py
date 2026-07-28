# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Convert a validated recovered mesh into a faceted STEP B-rep."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .cad_mesh_repair import (
    _drop_degenerate_faces,
    _effective_tolerance,
    _mesh_repair_and_validate,
    _simplify_mesh_for_step,
)
from .occ_shapes import (
    _assert_valid_occ_shape,
    _shape_to_shells,
    _shells_to_cq_shape,
    _solid_volume,
    _unify_same_domain_shape,
)

logger = logging.getLogger(__name__)


def _recovered_mesh_to_faceted_brep_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sew_tolerance: float = 1e-4,
    relative_sew_tolerance: float = 1e-6,
    merge_angle_deg: float = 0.0,
    max_faces: Optional[int] = 1500,
    repair_mesh: bool = True,
    validate_watertight: bool = True,
    validate_brep: bool = True,
    min_component_faces: int = 8,
) -> Any:
    """Sew the recovered triangle mesh into an OCC B-rep shape.

    A validated quadric simplification bounds the one-OCC-face-per-triangle
    sewing cost. Analytic passive cylinders/boxes are applied afterwards.
    """
    import cadquery as cq
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt

    effective_tol = _effective_tolerance(
        vertices, sew_tolerance, relative_sew_tolerance
    )
    if repair_mesh:
        vertices, faces = _mesh_repair_and_validate(
            vertices,
            faces,
            weld_tolerance=effective_tol,
            min_component_faces=min_component_faces,
            validate_watertight=validate_watertight,
        )
    else:
        vertices, faces = _drop_degenerate_faces(vertices, faces)
    vertices, faces = _simplify_mesh_for_step(vertices, faces, max_faces=max_faces)
    if repair_mesh:
        vertices, faces = _mesh_repair_and_validate(
            vertices,
            faces,
            weld_tolerance=effective_tol,
            min_component_faces=min_component_faces,
            validate_watertight=validate_watertight,
        )

    if len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError(
            "Recovered Shape has too few valid triangles for STEP sewing."
        )

    sew = BRepBuilderAPI_Sewing(float(effective_tol))
    n_added = 0
    n_skipped = 0
    for tri in faces:
        poly = BRepBuilderAPI_MakePolygon()
        for idx in tri[:3]:
            p = vertices[int(idx)]
            poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        poly.Close()
        try:
            face_builder = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if hasattr(face_builder, "IsDone") and not face_builder.IsDone():
                n_skipped += 1
                continue
            sew.Add(face_builder.Face())
            n_added += 1
        except Exception:
            n_skipped += 1

    if n_added < 4:
        raise RuntimeError(
            "Recovered Shape produced too few sewable triangles for STEP."
        )

    sew.Perform()
    shells = _shape_to_shells(sew.SewedShape())
    if not shells:
        raise RuntimeError("Recovered Shape sewing produced no shell.")

    shape = _shells_to_cq_shape(shells)
    if merge_angle_deg and float(merge_angle_deg) > 0.0:
        shape = _unify_same_domain_shape(shape, merge_angle_deg=merge_angle_deg)
    if not isinstance(shape, cq.Shape):
        shape = cq.Shape.cast(shape)
    if shape is None or not shape.isValid():
        raise RuntimeError("Recovered Shape STEP body is invalid after sewing.")
    if validate_brep:
        _assert_valid_occ_shape(shape, label="STEP body after mesh sewing")

    volume = _solid_volume(shape)
    logger.info(
        "Recovered Shape CAD reconstruction: %d triangles sewn%s, sew_tol=%g%s.",
        n_added,
        f", {n_skipped} skipped" if n_skipped else "",
        effective_tol,
        f", volume={volume:.3f}" if volume is not None else "",
    )
    return shape
