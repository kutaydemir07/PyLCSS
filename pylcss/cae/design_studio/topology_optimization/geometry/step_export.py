# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Verify exported STEP topology and repair only transfer-induced trim defects."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer


def export_cad_model(workplane_or_shape, layouts=(), export_path=None, export_format="STEP", *, verify=True):
    """Export native CAD geometry; check STEP by reimporting the result.

    ``layouts`` is retained for callers of the previous public signature.
    Passing the destination as the second argument is also supported.
    """
    if export_path is None:
        export_path = layouts
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = workplane_or_shape.val() if hasattr(workplane_or_shape, 'val') else workplane_or_shape
    kind = str(export_format).upper()
    if kind in {'STEP', 'STP'}:
        writer = STEPControl_Writer()
        if (writer.Transfer(shape.wrapped, STEPControl_AsIs) != IFSelect_RetDone
                or writer.Write(str(path)) != IFSelect_RetDone):
            raise RuntimeError(f'STEP export failed: {path}')
        if verify:
            verify_step_roundtrip(path)
    elif kind in {'IGES', 'IGS'}:
        from OCP.IGESControl import IGESControl_Writer
        writer = IGESControl_Writer()
        if not writer.AddShape(shape.wrapped) or not writer.Write(str(path)):
            raise RuntimeError(f'IGES export failed: {path}')
    else:
        raise ValueError(f'Unsupported CAD export format: {export_format}; use STEP or IGES')
    return path


def _closed(shape):
    solids = shape.Solids()
    return (bool(solids) and len(shape.Faces()) == sum(len(solid.Faces()) for solid in solids)
            and all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in shape.Shells()))


def _surface_signature(face):
    surface = BRep_Tool.Surface_s(face.wrapped)
    if face.geomType() != "BSPLINE":
        return face.geomType(), None
    values = [surface.UDegree(), surface.VDegree(), surface.NbUPoles(), surface.NbVPoles(),
              surface.NbUKnots(), surface.NbVKnots()]
    for i in range(1, surface.NbUPoles() + 1):
        for j in range(1, surface.NbVPoles() + 1):
            point = surface.Pole(i, j)
            values.extend((point.X(), point.Y(), point.Z(), surface.Weight(i, j)))
    for direction in ("U", "V"):
        for i in range(1, getattr(surface, f"Nb{direction}Knots")() + 1):
            values.extend((getattr(surface, f"{direction}Knot")(i), getattr(surface, f"{direction}Multiplicity")(i)))
    return "BSPLINE", np.asarray(values)


def _same_spline_surfaces(before, after):
    first, second = before.Faces(), after.Faces()
    if len(first) != len(second):
        return False
    for left, right in zip(first, second):
        kind_left, signature_left = _surface_signature(left)
        kind_right, signature_right = _surface_signature(right)
        if kind_left != kind_right:
            return False
        if kind_left != "BSPLINE":
            # Repairing analytic or mixed STEP shapes needs a separate contract.
            return False
        if signature_left.shape != signature_right.shape or not np.allclose(
                signature_left, signature_right, rtol=1e-10, atol=1e-9):
            return False
    return True


def _geometry_signature(face, linear_tolerance):
    """Compare surface geometry independently of face order and axis direction."""
    kind, values = _surface_signature(face)
    if kind == "BSPLINE":
        tolerances = np.full(len(values), 1e-10)
        pole_count = int(values[2] * values[3])
        for coordinate in range(3):
            tolerances[6 + coordinate:6 + 4 * pole_count:4] = linear_tolerance
        return kind, values, tolerances

    adaptor = BRepAdaptor_Surface(face.wrapped)

    def point(p):
        return np.array((p.X(), p.Y(), p.Z()))

    def direction(d):
        axis = np.array((d.X(), d.Y(), d.Z()))
        significant = np.flatnonzero(np.abs(axis) > 1e-12)
        return axis if axis[significant[0]] > 0 else -axis

    if kind == "PLANE":
        plane = adaptor.Plane()
        normal = direction(plane.Axis().Direction())
        values = np.r_[normal, normal @ point(plane.Location())]
        tolerances = np.r_[np.full(3, 1e-10), linear_tolerance]
    elif kind == "CYLINDER":
        cylinder = adaptor.Cylinder()
        axis = direction(cylinder.Axis().Direction())
        location = point(cylinder.Location())
        origin = location - (location @ axis) * axis
        values = np.r_[axis, origin, cylinder.Radius()]
        tolerances = np.r_[np.full(3, 1e-10), np.full(4, linear_tolerance)]
    elif kind == "SPHERE":
        sphere = adaptor.Sphere()
        values = np.r_[point(sphere.Location()), sphere.Radius()]
        tolerances = np.full(4, linear_tolerance)
    elif kind == "CONE":
        cone = adaptor.Cone()
        values = np.r_[direction(cone.Axis().Direction()), point(cone.Apex()), abs(cone.SemiAngle())]
        tolerances = np.r_[np.full(3, 1e-10), np.full(3, linear_tolerance), 1e-10]
    elif kind == "TORUS":
        torus = adaptor.Torus()
        values = np.r_[direction(torus.Axis().Direction()), point(torus.Location()),
                       torus.MajorRadius(), torus.MinorRadius()]
        tolerances = np.r_[np.full(3, 1e-10), np.full(5, linear_tolerance)]
    else:
        raise RuntimeError(f"STEP surface geometry verification does not support {kind}")
    return kind, values, tolerances


def assert_unchanged_surface_geometry(before, after):
    """Reject transferred geometry changes while allowing trim/pcurve repairs."""
    bounds = before.BoundingBox()
    scale = np.linalg.norm((bounds.xlen, bounds.ylen, bounds.zlen))
    tolerance = max(1e-8, float(scale) * 1e-9)
    unmatched = [_geometry_signature(face, tolerance) for face in after.Faces()]
    if len(unmatched) != len(before.Faces()):
        raise RuntimeError("STEP canonicalization changed the number of CAD surfaces")
    for face in before.Faces():
        kind, values, tolerances = _geometry_signature(face, tolerance)
        for index, (candidate_kind, candidate, _) in enumerate(unmatched):
            if (candidate_kind == kind and candidate.shape == values.shape
                    and np.all(np.abs(values - candidate) <= tolerances + 1e-11 * np.abs(values))):
                unmatched.pop(index)
                break
        else:
            raise RuntimeError(f"STEP canonicalization changed {kind} surface geometry")
    volume_before = abs(before.Volume(tol=1e-5))
    volume_after = abs(after.Volume(tol=1e-5))
    volume_delta = abs(volume_after - volume_before) / max(volume_before, 1e-15)
    if volume_delta > 1e-3:
        raise RuntimeError("STEP canonicalization changed the enclosed CAD volume")
    return {"unchanged_surface_geometry": True, "surface_count": len(before.Faces()),
            "geometry_linear_tolerance": tolerance, "relative_volume_delta": volume_delta}


def verify_step_roundtrip(path, *, allow_trim_repair=True, return_shape=False):
    """Reimport a STEP file, keeping any failed candidate available for diagnosis.

    A guarded repair preserves every spline's poles, weights and knots. It may
    adjust wire/pcurve topology damaged during transfer; it cannot refit a face.
    When requested, return the verified imported shape alongside its report.
    """
    path = Path(path)
    source = cq.importers.importStep(str(path)).val()
    report = {"valid": bool(source.isValid()), "is_solid": _closed(source),
              "face_count": len(source.Faces()), "trim_repair": False}
    if report["valid"] and report["is_solid"]:
        return (source, report) if return_shape else report
    if not allow_trim_repair or not report["is_solid"]:
        raise RuntimeError(f"STEP round-trip did not produce a valid closed solid: {path.name}")
    original_volume = abs(source.Volume(tol=1e-5))
    for precision in (1e-9, 1e-7):
        fixer = ShapeFix_Shape(source.copy().wrapped)
        fixer.SetPrecision(precision)
        fixer.SetMinTolerance(precision)
        fixer.SetMaxTolerance(precision)
        fixer.Perform()
        candidate = cq.Shape.cast(fixer.Shape())
        if not _same_spline_surfaces(source, candidate):
            continue
        descriptor, candidate_path = tempfile.mkstemp(prefix=".step_trim_", suffix=".step", dir=path.parent)
        os.close(descriptor)
        previous_mode = Interface_Static.IVal_s("write.precision.mode")
        previous_precision = Interface_Static.RVal_s("write.precision.val")
        try:
            Interface_Static.SetIVal_s("write.precision.mode", 1)
            Interface_Static.SetRVal_s("write.precision.val", precision)
            writer = STEPControl_Writer()
            if writer.Transfer(candidate.wrapped, STEPControl_AsIs) != IFSelect_RetDone:
                continue
            if writer.Write(candidate_path) != IFSelect_RetDone:
                continue
            restored = cq.importers.importStep(candidate_path).val()
            if not restored.isValid() or not _closed(restored) or not _same_spline_surfaces(source, restored):
                continue
            volume_delta = abs(abs(restored.Volume(tol=1e-5)) - original_volume) / original_volume
            if volume_delta > 1e-3:
                continue
            raw = path.with_name(path.stem + "_before_trim_repair.step")
            shutil.copyfile(path, raw)
            os.replace(candidate_path, path)
            report.update(valid=True, is_solid=True, trim_repair=True,
                          repair_precision=precision, unchanged_spline_surfaces=True,
                          repair_relative_volume_delta=volume_delta, original_checkpoint=raw.name)
            return (restored, report) if return_shape else report
        finally:
            Interface_Static.SetIVal_s("write.precision.mode", previous_mode)
            Interface_Static.SetRVal_s("write.precision.val", previous_precision)
            if os.path.isfile(candidate_path):
                os.remove(candidate_path)
    raise RuntimeError(f"STEP round-trip remains invalid after guarded trim repair: {path.name}")
