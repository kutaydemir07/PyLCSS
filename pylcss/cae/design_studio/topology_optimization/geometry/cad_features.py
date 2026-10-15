# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Reapply analytic passive and non-design regions to recovered CAD."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .occ_shapes import (
    _assess_brep_self_intersections,
    _assert_valid_occ_shape,
    _compound_or_single,
    _single_solid_if_possible,
    _unify_same_domain_shape,
)

logger = logging.getLogger(__name__)

CylinderRegion = tuple[str, float, float, float, float, float, float]
BoxRegion = tuple[float, float, float, float, float, float]


def _payload_bounds(payload: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return physical bounds stored by TopologyOptVoxelNode, if present."""
    if not isinstance(payload, dict):
        return None
    bounds = payload.get("bounds")
    mins = maxs = None
    if isinstance(bounds, dict):
        mins = bounds.get("min")
        if mins is None:
            mins = bounds.get("mins")
        if mins is None:
            mins = bounds.get("minimum")
        maxs = bounds.get("max")
        if maxs is None:
            maxs = bounds.get("maxs")
        if maxs is None:
            maxs = bounds.get("maximum")
    elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        mins, maxs = bounds[0], bounds[1]
    if mins is None or maxs is None:
        return None
    try:
        mins_arr = np.asarray(mins, dtype=float)[:3]
        maxs_arr = np.asarray(maxs, dtype=float)[:3]
    except Exception:
        return None
    if mins_arr.size < 3 or maxs_arr.size < 3:
        return None
    if not np.all(np.isfinite(mins_arr)) or not np.all(np.isfinite(maxs_arr)):
        return None
    if not np.all(maxs_arr > mins_arr):
        return None
    return mins_arr, maxs_arr


def _payload_region_list(payload: Any, key: str) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    regions = payload.get("passive_regions")
    if not isinstance(regions, dict):
        return []
    value = regions.get(key) or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _payload_region_boxes(payload: Any, key: str) -> list[BoxRegion]:
    out: list[BoxRegion] = []
    for item in _payload_region_list(payload, key):
        if isinstance(item, dict):
            vals = item.get("bounds") or item.get("box") or item.get("values")
            item = vals if vals is not None else item
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            x0, x1, y0, y1, z0, z1 = [float(v) for v in item[:6]]
            out.append((x0, x1, y0, y1, z0, z1))
        except Exception:
            continue
    return out


def _payload_region_cylinders(payload: Any, key: str) -> list[CylinderRegion]:
    """Return passive cylinders, accepting both 6- and 7-value region tuples."""
    out: list[CylinderRegion] = []
    for item in _payload_region_list(payload, key):
        if isinstance(item, dict):
            vals = item.get("fractional") or item.get("region") or item.get("values")
            item = vals if vals is not None else item
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            axis, c0, c1, lo, hi, r0 = item[:6]
            r1 = item[6] if len(item) > 6 else r0
            r0 = float(r0)
            r1 = float(r1)
            if r0 <= 0.0 or r1 <= 0.0:
                continue
            out.append(
                (
                    str(axis or "z").strip().lower(),
                    float(c0),
                    float(c1),
                    float(lo),
                    float(hi),
                    r0,
                    r1,
                )
            )
        except Exception:
            continue
    return out


def _region_box_to_solid(
    box: BoxRegion,
    bounds: tuple[np.ndarray, np.ndarray],
) -> Any:
    import cadquery as cq

    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-12)
    x0, x1, y0, y1, z0, z1 = [float(v) for v in box]
    lo = mins + np.asarray([min(x0, x1), min(y0, y1), min(z0, z1)], dtype=float) * span
    hi = mins + np.asarray([max(x0, x1), max(y0, y1), max(z0, z1)], dtype=float) * span
    size = hi - lo
    if not np.all(size > 1e-9):
        return None
    return cq.Solid.makeBox(
        float(size[0]),
        float(size[1]),
        float(size[2]),
        cq.Vector(float(lo[0]), float(lo[1]), float(lo[2])),
    )


def _region_cylinder_to_solid(
    cylinder: CylinderRegion,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    axial_margin: float = 0.0,
    radial_margin: float = 0.0,
) -> Any:
    """Convert a fractional passive cylinder to an analytic CadQuery cylinder."""
    import cadquery as cq

    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-12)
    axis, c0, c1, lo, hi, r0, r1 = cylinder
    axis = str(axis or "z").strip().lower()
    lo, hi = sorted((float(lo), float(hi)))
    lo -= float(axial_margin)
    hi += float(axial_margin)

    if axis == "x":
        radius = 0.5 * (float(r0) * span[1] + float(r1) * span[2])
        height = (hi - lo) * span[0]
        base = (
            mins[0] + lo * span[0],
            mins[1] + float(c0) * span[1],
            mins[2] + float(c1) * span[2],
        )
        direction = (1.0, 0.0, 0.0)
    elif axis == "y":
        radius = 0.5 * (float(r0) * span[0] + float(r1) * span[2])
        height = (hi - lo) * span[1]
        base = (
            mins[0] + float(c0) * span[0],
            mins[1] + lo * span[1],
            mins[2] + float(c1) * span[2],
        )
        direction = (0.0, 1.0, 0.0)
    else:
        radius = 0.5 * (float(r0) * span[0] + float(r1) * span[1])
        height = (hi - lo) * span[2]
        base = (
            mins[0] + float(c0) * span[0],
            mins[1] + float(c1) * span[1],
            mins[2] + lo * span[2],
        )
        direction = (0.0, 0.0, 1.0)

    radius += float(radial_margin or 0.0)
    if radius <= 1e-9 or height <= 1e-9:
        return None
    return cq.Solid.makeCylinder(
        float(radius),
        float(height),
        cq.Vector(float(base[0]), float(base[1]), float(base[2])),
        cq.Vector(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def _as_float_vec(value: Any, *, length: int = 3) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)[:length]
    except Exception:
        return None
    if arr.size < length or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float)


def _payload_physical_cad_features(
    payload: Any, role: str, feature_type: str
) -> list[dict[str, Any]]:
    """Return optional exact CAD feature metadata in physical model units.

    Supported payload locations:
    ``payload['cad_features']`` or ``payload['passive_cad_features']``.

    Example cylinder entry::

        {
            "type": "cylinder",
            "role": "void",  # or "solid"
            "center": [x, y, z],
            "axis": [0, 0, 1],
            "radius": r,
            "height": h,
            "name": "front_left_bearing_bore"
        }

    Start/end form is also accepted using ``start`` and ``end`` instead of
    ``center``/``height``.
    """
    if not isinstance(payload, dict):
        return []
    raw = []
    for key in ("cad_features", "passive_cad_features"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            raw.extend(value)
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() != feature_type:
            continue
        if str(item.get("role", "")).strip().lower() != role:
            continue
        out.append(item)
    return out


def _physical_box_to_solid(feature: dict[str, Any]) -> Any:
    import cadquery as cq

    lo = _as_float_vec(feature.get("min") or feature.get("mins") or feature.get("lo"))
    hi = _as_float_vec(feature.get("max") or feature.get("maxs") or feature.get("hi"))
    if lo is None or hi is None:
        center = _as_float_vec(feature.get("center"))
        size = _as_float_vec(feature.get("size") or feature.get("dimensions"))
        if center is None or size is None:
            return None
        lo = center - 0.5 * size
        hi = center + 0.5 * size
    size = hi - lo
    if not np.all(size > 1e-9):
        return None
    return cq.Solid.makeBox(
        float(size[0]),
        float(size[1]),
        float(size[2]),
        cq.Vector(float(lo[0]), float(lo[1]), float(lo[2])),
    )


def _physical_cylinder_to_solid(
    feature: dict[str, Any],
    *,
    axial_margin: float = 0.0,
    radial_margin: float = 0.0,
) -> Any:
    import cadquery as cq

    radius = feature.get("radius")
    try:
        radius = float(radius) + float(radial_margin or 0.0)
    except Exception:
        return None
    if radius <= 1e-9:
        return None

    start = _as_float_vec(feature.get("start"))
    end = _as_float_vec(feature.get("end"))
    if start is not None and end is not None:
        vec = end - start
        height = float(np.linalg.norm(vec))
        if height <= 1e-9:
            return None
        axis = vec / height
        base = start - axis * float(axial_margin or 0.0)
        height += 2.0 * float(axial_margin or 0.0)
    else:
        center = _as_float_vec(feature.get("center"))
        axis = _as_float_vec(feature.get("axis"))
        if center is None or axis is None:
            return None
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return None
        axis = axis / axis_norm
        try:
            height = float(feature.get("height") or feature.get("length"))
        except Exception:
            return None
        if height <= 1e-9:
            return None
        base = center - axis * (0.5 * height + float(axial_margin or 0.0))
        height += 2.0 * float(axial_margin or 0.0)

    return cq.Solid.makeCylinder(
        float(radius),
        float(height),
        cq.Vector(float(base[0]), float(base[1]), float(base[2])),
        cq.Vector(float(axis[0]), float(axis[1]), float(axis[2])),
    )


#: Fuzzy tolerances retried, in order, when an exact boolean comes back
#: invalid. An automatic load/support pad is generated *on* the face its
#: boundary condition was selected from, so the tool and the body meet
#: face-to-face — the configuration OCC's exact boolean handles worst. A fuzzy
#: value lets it merge coincident geometry instead of producing a shell with a
#: sliver in it. Kept small: this is for degenerate contact, not for repairing
#: a body that is genuinely wrong.
_BOOLEAN_FUZZY_RETRIES = (1e-5, 1e-4, 1e-3)


def _material_bounds(shape: Any) -> np.ndarray:
    """Bound CAD geometry independently of cached display triangulation.

    Trimmed spline faces can retain a conservative UV-rectangle envelope.
    A difference between these boxes needs the trim-aware distance check below.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    # CadQuery BoundingBox() uses triangulation when available. STEP transfer
    # drops that cache, so its deflection must never enter a material check.
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    bounds = np.asarray(box.Get(), dtype=float)[[0, 3, 1, 4, 2, 5]]
    if not np.isfinite(bounds).all():
        raise RuntimeError("Material bounds require finite CAD geometry")
    return bounds


def _material_bounds_delta(before: Any, after: Any, tolerance: float) -> np.ndarray:
    """Resolve disputed box extrema against the actual trimmed material."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.Precision import Precision
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    delta = _material_bounds(before) - _material_bounds(after)
    disputed = np.flatnonzero(np.abs(delta) > tolerance)
    if not len(disputed):
        return delta

    # AddOptimal can keep a UV envelope when its trimmed-face extrema solver
    # fails. Bound the full geometry conservatively, then place common support
    # planes outside both operands. DistanceSS classifies extrema on the real
    # face trims and also checks their boundary edges and vertices.
    envelope = Bnd_Box()
    for shape in (before, after):
        BRepBndLib.Add_s(shape.wrapped, envelope, False)
    limits = np.asarray(envelope.Get(), dtype=float)
    if not np.isfinite(limits).all():
        raise RuntimeError("Could not bound the trimmed CAD material")
    lower, upper = limits[:3], limits[3:]
    center = (lower + upper) * 0.5
    margin = max(float(np.linalg.norm(upper - lower)), 1.0)
    for index in disputed:
        axis, end = divmod(int(index), 2)
        point = center.copy()
        point[axis] = upper[axis] + margin if end else lower[axis] - margin
        normal = np.eye(3)[axis]
        plane = BRepBuilderAPI_MakeFace(
            gp_Pln(gp_Pnt(*point), gp_Dir(*normal)),
            -margin, margin, -margin, margin,
        ).Face()
        supports = []
        for shape in (before, after):
            distance = BRepExtrema_DistShapeShape()
            distance.LoadS1(shape.wrapped)
            distance.LoadS2(plane)
            distance.SetDeflection(Precision.Confusion_s() * 0.1)
            distance.SetMultiThread(True)
            distance.Perform()
            if not distance.IsDone() or distance.NbSolution() < 1:
                raise RuntimeError("Could not verify the trimmed CAD material bounds")
            value = float(distance.Value())
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError("Invalid distance to the CAD material support plane")
            supports.append(point[axis] - value if end else point[axis] + value)
        delta[index] = supports[0] - supports[1]
    return delta


def _prepare_boolean_continuity(shape: Any, *, return_parent_face_indices: bool = False) -> Any:
    """Split internal C0 surface spans without changing their geometry.

    Valid fitted faces can contain cubic knots of multiplicity three. OCC's
    Boolean intersector can build unusable microscopic intersection edges on
    such faces. Exact C1 subdivision gives it regular surface spans; bounded
    topology healing reconnects their shared edges without refitting surfaces.
    Optional parent indices preserve source-patch provenance for strict checks
    of the normalized faces before a reconstruction is accepted.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.GeomAbs import GeomAbs_C0, GeomAbs_C1
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity

    faces = shape.Faces()
    needs_split = False
    for face in faces:
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.UContinuity() == GeomAbs_C0 or surface.VContinuity() == GeomAbs_C0:
            needs_split = True
            break
    if not needs_split:
        return (shape, list(range(len(faces)))) if return_parent_face_indices else shape

    solids = shape.Solids()
    if (not solids or not shape.isValid()
            or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in shape.Shells())):
        raise RuntimeError("Analytic Boolean continuity preparation requires a valid closed body")
    # Copy the complete topology graph while sharing its exact geometry.
    # Divider/healer edge-tolerance changes must not mutate the fitted input.
    copied = BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape()
    divider = ShapeUpgrade_ShapeDivideContinuity(copied)
    divider.SetSurfaceCriterion(GeomAbs_C1)
    divider.SetBoundaryCriterion(GeomAbs_C1)
    divider.SetPCurveCriterion(GeomAbs_C1)
    divider.SetPrecision(1e-7)
    if not divider.Perform():
        return (shape, list(range(len(faces)))) if return_parent_face_indices else shape
    fixer = ShapeFix_Shape(divider.Result())
    fixer.SetPrecision(1e-7)
    fixer.SetMaxTolerance(1e-5)
    fixer.Perform()
    candidate = cq.Shape.cast(fixer.Shape())
    if (not candidate.isValid() or len(candidate.Solids()) != len(solids)
            or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in candidate.Shells())):
        raise RuntimeError("Exact C1 subdivision did not retain a valid closed CAD body")

    bounds = _material_bounds(shape)
    scale = float(np.linalg.norm(bounds[1::2] - bounds[::2]))
    geometry_tolerance = max(1e-8, scale * 1e-10)
    bounds_delta = _material_bounds_delta(shape, candidate, geometry_tolerance)
    if (np.any(bounds_delta[::2] > geometry_tolerance)
            or np.any(bounds_delta[1::2] < -geometry_tolerance)):
        raise RuntimeError("Exact C1 subdivision changed the CAD bounds")
    from .amrto_volume import assert_unchanged_cad_volume
    assert_unchanged_cad_volume(shape, candidate, relative_tolerance=1e-6)

    # The divider uses exact spline segmentation, not approximation. Check
    # that healing retained each child on an original surface at identical UVs.
    parents = [(BRep_Tool.Surface_s(face.wrapped), face._uvBounds()) for face in faces]
    parent_face_indices = []
    for face in candidate.Faces():
        surface = BRep_Tool.Surface_s(face.wrapped)
        u0, u1, v0, v1 = face._uvBounds()
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        center = surface.Value(um, vm)
        matches = [(center.Distance(parent.Value(um, vm)), index, parent)
                   for index, (parent, (pu0, pu1, pv0, pv1)) in enumerate(parents)
                   if pu0 - 1e-10 <= um <= pu1 + 1e-10
                   and pv0 - 1e-10 <= vm <= pv1 + 1e-10]
        if not matches:
            raise RuntimeError("Exact C1 subdivision lost source surface correspondence")
        distance, parent_index, parent = min(matches, key=lambda pair: pair[0])
        if distance > geometry_tolerance or any(
                surface.Value(u0 + a * (u1 - u0), v0 + b * (v1 - v0)).Distance(
                    parent.Value(u0 + a * (u1 - u0), v0 + b * (v1 - v0))) > geometry_tolerance
                for a in (.125, .5, .875) for b in (.125, .5, .875)):
            raise RuntimeError("C1 topology healing changed the fitted surface geometry")
        parent_face_indices.append(parent_index)
    from .amrto_volume import assert_rectangular_spline_partition
    assert_rectangular_spline_partition(shape, candidate, parent_face_indices)
    logger.info("Prepared %d fitted CAD faces as %d exact C1 faces for analytic booleans",
                len(faces), len(candidate.Faces()))
    return (candidate, parent_face_indices) if return_parent_face_indices else candidate


def _native_boolean(body: Any, tool: Any, operation: str, fuzzy: float = 0.0) -> Any:
    """Preserve shared fitted geometry while protecting the input topology."""
    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCP.TopTools import TopTools_ListOfShape
    arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(body.wrapped)
    tools.Append(tool.wrapped)
    builder = {'fuse': BRepAlgoAPI_Fuse, 'cut': BRepAlgoAPI_Cut,
               'common': BRepAlgoAPI_Common}[operation]()
    builder.SetArguments(arguments)
    builder.SetTools(tools)
    builder.SetNonDestructive(True)
    builder.SetRunParallel(True)
    if fuzzy:
        builder.SetFuzzyValue(fuzzy)
    builder.Build()
    if not builder.IsDone():
        raise RuntimeError(f"OpenCASCADE {operation} failed")
    return cq.Shape.cast(builder.Shape())


def _normalize_boolean_trims(shape: Any) -> Any:
    """Normalize Boolean trim bookkeeping without changing its material surfaces."""
    import cadquery as cq
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Precision import Precision
    from .step_export import (
        _closed, _assert_matching_surface_geometry, export_cad_model, verify_step_roundtrip,
    )

    # An unorientable trim can belong to an otherwise closed Boolean result.
    # STEP transfer can reconstruct its wire/pcurve relationships. Open results
    # and changes to the surfaces or enclosed material are never eligible.
    if not _closed(shape):
        raise RuntimeError("Boolean trim normalization requires closed solid geometry")
    before = abs(float(shape.Volume(tol=1e-7)))
    if not np.isfinite(before) or before <= 0.0:
        raise RuntimeError("Boolean trim normalization requires positive finite material volume")
    with TemporaryDirectory(prefix="pylcss_boolean_trims_") as directory:
        isolated = cq.Shape.cast(BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape())
        path = Path(directory) / "material.step"
        export_cad_model(isolated, export_path=path, verify=False)
        canonical, _ = verify_step_roundtrip(path, return_shape=True)
    # Reuse the two strict 1e-7 integrations below instead of also performing
    # the generic STEP helper's weaker, duplicate 1e-5 integrations.
    _assert_matching_surface_geometry(shape, canonical)
    if (not canonical.isValid() or not _closed(canonical)
            or len(canonical.Solids()) != len(shape.Solids())):
        raise RuntimeError("Boolean trim normalization did not preserve valid closed solids")
    after = abs(float(canonical.Volume(tol=1e-7)))
    if not np.isfinite(after) or abs(after - before) > max(1e-9, before * 1e-6):
        raise RuntimeError("Boolean trim normalization changed the enclosed material")
    bounds = _material_bounds(shape)
    bound_tolerance = max(2.0 * Precision.Confusion_s(),
                          float(np.linalg.norm(bounds[1::2] - bounds[::2])) * 1e-9)
    bounds_delta = _material_bounds_delta(shape, canonical, bound_tolerance)
    if not np.isfinite(bounds_delta).all() or np.max(np.abs(bounds_delta)) > bound_tolerance:
        raise RuntimeError("Boolean trim normalization changed the material bounds "
                           f"(maximum change {np.max(np.abs(bounds_delta)):.9g}; "
                           f"tolerance {bound_tolerance:.9g})")
    return canonical


def _boolean_or_raise(result: Any, tool: Any, *, operation: str, allow_empty_cut: bool = False,
                      check_normalized_intersections: bool = True) -> Any:
    """Protect operands; defer intermediate scans only to a mandatory body check."""
    solid_input = hasattr(result, "Solids") and bool(result.Solids())

    last_error: Exception | None = None

    def attempts():
        nonlocal last_error
        # Native success pays for no transfer. If it fails, first retry the
        # unchanged material in canonical STEP trim representation: increasing
        # the fuzzy tolerance cannot repair inconsistent pcurve bookkeeping.
        yield result, 0.0, False
        body = result
        try:
            body = _normalize_boolean_trims(result)
        except Exception as exc:
            last_error = exc
        else:
            yield body, 0.0, True
        for fuzzy in _BOOLEAN_FUZZY_RETRIES:
            yield body, fuzzy, body is not result

    for body, fuzzy, normalized_operand in attempts():
        try:
            # Non-destructive mode protects the owner without copying shared
            # fitted geometry before the native operation.
            candidate = _native_boolean(body, tool, operation, fuzzy)
        except Exception as exc:  # noqa: PERF203 - each tolerance is a retry
            last_error = exc
            continue
        if candidate is None:
            continue
        intersections_checked = False
        if hasattr(candidate, "isValid") and not candidate.isValid():
            try:
                candidate = _normalize_boolean_trims(candidate)
                if (check_normalized_intersections
                        and not _assess_brep_self_intersections(candidate).get("passed", False)):
                    raise RuntimeError("Normalized Boolean trims retain CAD intersections")
                intersections_checked = check_normalized_intersections
                logger.info("Recovered Shape STEP: analytic %s trim normalization preserved exact surfaces and material.",
                            operation)
            except Exception as exc:
                last_error = exc
                continue
        # OCC can return a valid empty Compound after a failed boolean. A
        # manufactured solid must retain solid geometry before accepting it.
        if solid_input and (not candidate.Solids() or not candidate.Faces()):
            if operation == "cut" and allow_empty_cut:
                # A clearance may legitimately remove an entire lattice member
                # or joint. Confirm complete containment independently before
                # treating an empty Boolean result as a successful removal.
                overlap = _native_boolean(result, tool, 'common')
                volume = abs(float(result.Volume()))
                if overlap.isValid() and volume > 0 and abs(abs(overlap.Volume()) - volume) <= volume * 1e-7:
                    return candidate
            last_error = RuntimeError(f"Analytic {operation} erased the solid body")
            continue
        if check_normalized_intersections and normalized_operand and not intersections_checked:
            if not _assess_brep_self_intersections(candidate).get("passed", False):
                last_error = RuntimeError("Boolean on normalized trims retains CAD intersections")
                continue
        if solid_input and operation == "fuse":
            original_volume = abs(float(result.Volume(tol=1e-4)))
            candidate_volume = abs(float(candidate.Volume(tol=1e-4)))
            if candidate_volume < original_volume * (1.0 - 1e-3):
                last_error = RuntimeError("Analytic fuse removed material from the solid body")
                continue
        if fuzzy:
            logger.info(
                "Recovered Shape STEP: analytic %s needed a %g fuzzy "
                "tolerance to merge face-coincident geometry.",
                operation,
                fuzzy,
            )
        return _single_solid_if_possible(candidate)

    if last_error is not None:
        raise RuntimeError(
            f"Recovered Shape STEP failed during analytic {operation}."
        ) from last_error
    raise RuntimeError(
        f"Recovered Shape STEP analytic {operation} produced an invalid body."
    )


def _intersecting_tools(shape: Any, tools: list[Any]) -> list[Any]:
    """Keep analytic solids that overlap the recovered manufactured body.

    Blindly fusing every requested passive feature can create detached sleeves
    when thresholding removed the surrounding load path. Such a STEP file is
    technically valid but physically misleading. A feature is retained only
    when OCC finds a non-negligible common volume with the recovered body.
    """
    retained: list[Any] = []
    for tool in tools:
        try:
            # A common-volume pre-check is read-only from the caller's point
            # of view. OCC may still update its operands before rejecting a
            # detached feature, so isolate both geometries here as well.
            common = _native_boolean(shape, tool, 'common')
            common_volume = abs(float(common.Volume(tol=1e-4)))
            tool_volume = max(abs(float(tool.Volume(tol=1e-4))), 1.0)
            if common_volume <= max(1e-9, tool_volume * 1e-8):
                logger.warning(
                    "Recovered Shape STEP: skipped a detached passive solid feature."
                )
                continue
        except Exception:
            # An OCC intersection diagnostic should not make an otherwise
            # usable export impossible. The subsequent validated boolean is
            # still authoritative when the pre-check is unavailable.
            logger.debug(
                "Could not pre-check passive feature intersection",
                exc_info=True,
            )
        retained.append(tool)
    return retained


def _cut_tools_sequentially(shape: Any, tools: list[Any], *, allow_empty_cut: bool = False,
                            check_normalized_intersections: bool = True) -> Any:
    """Apply overlapping bore/cleanup tools one at a time.

    OCC compounds containing overlapping cylinders are not a boolean union;
    cutting by such a compound can leave a near-complete duplicate pin and
    microscopic sliver solids. Sequential validated cuts are deterministic for
    the small number of preserved engineering features used here.
    """
    result = shape
    for tool in tools:
        result = _boolean_or_raise(result, tool, operation="cut", allow_empty_cut=allow_empty_cut,
                                  check_normalized_intersections=check_normalized_intersections)
        if allow_empty_cut and not result.Solids():
            break
    return result


def _drop_boolean_sliver_solids(
    shape: Any,
    *,
    relative_volume: float = 1e-3,
) -> Any:
    """Remove tiny disconnected solids created only by OCC cut tolerances."""
    import cadquery as cq

    try:
        solids = list(shape.Solids())
    except Exception:
        return shape
    if len(solids) <= 1:
        return shape
    volumes = np.asarray(
        [max(0.0, abs(float(solid.Volume(tol=1e-4)))) for solid in solids],
        dtype=float,
    )
    largest = float(np.max(volumes)) if volumes.size else 0.0
    if largest <= 0.0:
        return shape
    threshold = max(1e-9, largest * max(float(relative_volume), 0.0))
    kept = [
        solid
        for solid, volume in zip(solids, volumes, strict=True)
        if float(volume) >= threshold
    ]
    removed = len(solids) - len(kept)
    if removed <= 0 or not kept:
        return shape
    logger.info(
        "Recovered Shape STEP: removed %d boolean sliver solid(s) below "
        "%.6g model-volume units.",
        removed,
        threshold,
    )
    return kept[0] if len(kept) == 1 else cq.Compound.makeCompound(kept)


def _coerce_to_solid(geom: Any) -> Any:
    """Coerce arbitrary CadQuery/OCC geometry into a cq.Shape solid/compound."""
    if geom is None:
        return None
    import cadquery as cq

    if isinstance(geom, cq.Shape):
        return geom
    if hasattr(geom, "val") and callable(geom.val):
        try:
            val = geom.val()
            if isinstance(val, cq.Shape):
                return val
        except Exception:
            pass
    if hasattr(geom, "toCompound") and callable(geom.toCompound):
        try:
            val = geom.toCompound()
            if isinstance(val, cq.Shape):
                return val
        except Exception:
            pass
    if hasattr(geom, "wrapped"):
        try:
            return cq.Shape.cast(geom.wrapped)
        except Exception:
            pass
    try:
        from OCP.TopoDS import TopoDS_Shape

        if isinstance(geom, TopoDS_Shape):
            return cq.Shape.cast(geom)
    except Exception:
        pass
    return None


def _preserved_bore_records(shape: Any, *, minimum_angle: float = 2*np.pi-1e-6) -> list[tuple[Any, ...]]:
    """Identify inward cylinder walls, excluding short concave fillet arcs."""
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    records = []
    for face in shape.Faces():
        if face.geomType() != 'CYLINDER':
            continue
        adaptor = BRepAdaptor_Surface(face.wrapped)
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        if u1-u0 < minimum_angle:
            continue
        cylinder = adaptor.Cylinder()
        axis = np.asarray(cq.Vector(cylinder.Axis().Direction()).toTuple())
        location = np.asarray(cq.Vector(cylinder.Location()).toTuple())
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        point = cq.Vector(adaptor.Value((u0+u1)/2, (v0+v1)/2))
        radial = np.asarray(point.toTuple())-location
        radial -= (radial @ axis)*axis
        if np.asarray(face.normalAt(point).toTuple()) @ radial >= 0:
            continue
        significant = np.flatnonzero(np.abs(axis)>1e-12)
        if axis[significant[0]] < 0:
            axis = -axis
        origin = location-(location @ axis)*axis
        ends = [np.asarray(cq.Vector(adaptor.Value((u0+u1)/2,v)).toTuple()) @ axis for v in (v0,v1)]
        records.append((axis, origin, cylinder.Radius(), min(ends), max(ends)))
    return records


def source_bore_cylinders(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Carry source-CAD holes into implicit lattice surface recovery.

    A counterbore can open through the edge of a mounting foot, so its wall
    need not describe a complete circle. Half-circle or larger inward walls
    are included; small concave blend arcs are not treated as drilled holes.
    """
    source = _coerce_to_solid(payload.get('source_cad_shape'))
    bounds = _payload_bounds(payload)
    if source is None or bounds is None:
        return []
    lower, upper = bounds
    span = np.asarray(upper)-np.asarray(lower)
    cylinders = []
    for axis, origin, radius, start, end in _preserved_bore_records(source, minimum_angle=np.pi-1e-6):
        index = int(np.argmax(np.abs(axis)))
        if not np.allclose(axis, np.eye(3)[index], atol=1e-8):
            continue
        radial = [i for i in range(3) if i != index]
        cylinders.append(('xyz'[index],
            float((origin[radial[0]]-lower[radial[0]])/span[radial[0]]),
            float((origin[radial[1]]-lower[radial[1]])/span[radial[1]]),
            float((start-.02-lower[index])/span[index]),
            float((end+.02-lower[index])/span[index]),
            float(radius/span[radial[0]]), float(radius/span[radial[1]])))
    return cylinders


def _preserved_bore_tools(shapes: list[Any], axial_margin: float) -> list[Any]:
    import cadquery as cq
    groups = {}
    for shape in shapes:
        for axis, origin, radius, lower, upper in _preserved_bore_records(shape):
            key = tuple(np.round(np.r_[axis, origin, radius],6))
            group = groups.setdefault(key, (axis, origin, radius, []))
            group[3].append((lower,upper))
    tools = []
    for axis, origin, radius, intervals in groups.values():
        merged = []
        for lower, upper in sorted(intervals):
            if merged and lower <= merged[-1][1]+2*axial_margin:
                merged[-1][1] = max(merged[-1][1],upper)
            else:
                merged.append([lower,upper])
        for lower,upper in merged:
            if upper-lower > 1e-9:
                tools.append(cq.Solid.makeCylinder(radius, upper-lower+2*axial_margin,
                    cq.Vector(*(origin+axis*(lower-axial_margin))), cq.Vector(*axis)))
    return tools


def _canonical_annular_layer(shape: Any) -> Any:
    """Rebuild an exact cylindrical sleeve with canonical, checked face trims.

    Offset-generated annuli can carry valid but Boolean-incompatible seam
    parameterizations. Only a complete two-cylinder, two-plane annulus is
    eligible. Its analytic dimensions and two-sided native material equality
    must agree; all other protected shapes retain their original geometry.
    """
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from .step_export import _closed

    faces = shape.Faces()
    cylinders = [face for face in faces if face.geomType() == 'CYLINDER']
    planes = [face for face in faces if face.geomType() == 'PLANE']
    if (len(faces) != 4 or len(cylinders) != 2 or len(planes) != 2
            or len(shape.Solids()) != 1 or not shape.isValid() or not _closed(shape)):
        return shape
    try:
        records = []
        for face in cylinders:
            surface = BRepAdaptor_Surface(face.wrapped)
            if abs(surface.LastUParameter()-surface.FirstUParameter()-2*np.pi) > 1e-8:
                return shape
            cylinder = surface.Cylinder()
            axis = np.asarray(cq.Vector(cylinder.Axis().Direction()).toTuple())
            nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
            if axis[nonzero[0]] < 0:
                axis = -axis
            location = np.asarray(cq.Vector(cylinder.Location()).toTuple())
            origin = location-(location @ axis)*axis
            midpoint = (surface.FirstUParameter()+surface.LastUParameter())*.5
            ends = sorted(np.asarray(cq.Vector(surface.Value(midpoint, value)).toTuple()) @ axis
                          for value in (surface.FirstVParameter(), surface.LastVParameter()))
            records.append((axis, origin, cylinder.Radius(), *ends))
        inner, outer = sorted(records, key=lambda record: record[2])
        axis, origin, radius, lower, upper = outer
        scale = max(radius, upper-lower, 1.0)
        linear = scale*1e-10
        if (inner[2] <= 0 or radius <= inner[2]
                or not np.allclose(inner[0], axis, rtol=0, atol=1e-12)
                or np.linalg.norm(inner[1]-origin) > linear
                or max(abs(inner[3]-lower), abs(inner[4]-upper)) > linear):
            return shape
        height = upper-lower
        if height <= linear:
            return shape
        area = np.pi*(radius**2-inner[2]**2)
        for face in planes:
            surface = BRepAdaptor_Surface(face.wrapped)
            plane = surface.Plane()
            normal = np.asarray(cq.Vector(plane.Axis().Direction()).toTuple())
            level = np.asarray(cq.Vector(plane.Location()).toTuple()) @ axis
            if (abs(abs(normal @ axis)-1) > 1e-12
                    or min(abs(level-lower), abs(level-upper)) > linear
                    or abs(face.Area()-area) > max(1e-8, area*1e-10)):
                return shape
        for face, record in zip(cylinders, records):
            expected_area = 2*np.pi*record[2]*height
            if abs(face.Area()-expected_area) > max(1e-8, expected_area*1e-10):
                return shape
        base = cq.Vector(*(origin+axis*lower))
        canonical = cq.Solid.makeCylinder(radius, height, base, cq.Vector(*axis)).cut(
            cq.Solid.makeCylinder(inner[2], height, base, cq.Vector(*axis)))
        expected_volume = area*height
        volume_tolerance = max(1e-8, expected_volume*1e-10)
        if (not canonical.isValid() or not _closed(canonical)
                or abs(shape.Volume(tol=1e-7)-expected_volume) > volume_tolerance
                or np.max(np.abs(_material_bounds_delta(shape, canonical, linear))) > max(linear, 2e-7)):
            return shape
        for first, second in ((shape, canonical), (canonical, shape)):
            outside = _native_boolean(first, second, 'cut')
            if (not outside.isValid()
                    or sum(abs(part.Volume(tol=1e-7)) for part in outside.Solids()) > volume_tolerance):
                return shape
        common = _native_boolean(shape, canonical, 'common')
        if (not common.isValid() or not _closed(common)
                or abs(common.Volume(tol=1e-7)-expected_volume) > volume_tolerance):
            return shape
        return canonical
    except Exception:
        return shape


def _collect_passive_tools(
    payload: Any,
    *,
    void_axial_margin: float,
    void_radial_margin: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Return the (solid, cut, hardware) analytic tools a payload asks for.

    Split out of :func:`_apply_passive_regions_to_step` so the analytic lattice
    path can reuse exactly the same feature set without inheriting the general
    boolean strategy, which is wrong for a compound of overlapping members.
    """
    solid_tools: list[Any] = []
    cut_tools: list[Any] = []
    hardware_tools: list[Any] = []

    bounds = _payload_bounds(payload)
    if bounds is not None:
        # Fractional passive regions were imposed on the signed-distance field,
        # but the extraction/smoothing/subdivision stages turn their exact
        # planes and cylinders into approximations.  Reapply intersecting keep
        # regions as canonical OCC solids.  `_intersecting_tools` below rejects
        # detached features before any fuse, which retains the earlier safety
        # rule without sacrificing analytic support/load interfaces.
        for box in _payload_region_boxes(payload, "solid_boxes"):
            tool = _region_box_to_solid(box, bounds)
            if tool is not None:
                solid_tools.append(tool)
        for cylinder in _payload_region_cylinders(payload, "solid_cylinders"):
            tool = _region_cylinder_to_solid(cylinder, bounds)
            if tool is not None:
                solid_tools.append(tool)
        for box in _payload_region_boxes(payload, "void_boxes"):
            tool = _region_box_to_solid(box, bounds)
            if tool is not None:
                cut_tools.append(tool)
        for cylinder in _payload_region_cylinders(payload, "void_cylinders"):
            tool = _region_cylinder_to_solid(
                cylinder,
                bounds,
                axial_margin=void_axial_margin,
                radial_margin=void_radial_margin,
            )
            if tool is not None:
                cut_tools.append(tool)
        for cylinder in _payload_region_cylinders(payload, "joint_pin_cylinders"):
            tool = _region_cylinder_to_solid(cylinder, bounds)
            if tool is not None:
                hardware_tools.append(tool)
            # Remove the recovered-mesh copy across the *entire* pin span,
            # including the clearance gap between bodies. Bore cutters cover
            # only each lug thickness and otherwise leave small pin fragments
            # that become misleading extra STEP solids.
            cleanup_tool = _region_cylinder_to_solid(
                cylinder,
                bounds,
                axial_margin=max(float(void_axial_margin), 0.02),
                radial_margin=max(float(void_radial_margin), 0.01),
            )
            if cleanup_tool is not None:
                cut_tools.append(cleanup_tool)

    # Optional newer exact-geometry metadata in physical CAD units.
    for feature in _payload_physical_cad_features(payload, "solid", "box"):
        tool = _physical_box_to_solid(feature)
        if tool is not None:
            solid_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "solid", "cylinder"):
        tool = _physical_cylinder_to_solid(feature)
        if tool is not None:
            solid_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "void", "box"):
        tool = _physical_box_to_solid(feature)
        if tool is not None:
            cut_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "void", "cylinder"):
        tool = _physical_cylinder_to_solid(
            feature,
            axial_margin=void_axial_margin,
            radial_margin=void_radial_margin,
        )
        if tool is not None:
            cut_tools.append(tool)

    passive_info = (
        payload.get("passive_regions") if isinstance(payload, dict) else None
    )
    if isinstance(passive_info, dict):
        preserved_shapes = []
        for raw_shape in passive_info.get("explicit_solid_shapes", []):
            tool = _coerce_to_solid(raw_shape)
            if tool is not None:
                solid_tools.append(tool)
                preserved_shapes.append(tool)
        # Reinstating an annular non-design layer must also reinstate its
        # original bore; adding the ring alone cannot remove fitted material
        # that intruded into the hole.
        cut_tools.extend(_preserved_bore_tools(preserved_shapes, void_axial_margin))
        for raw_shape in passive_info.get("explicit_void_shapes", []):
            tool = _coerce_to_solid(raw_shape)
            if tool is not None:
                cut_tools.append(tool)

    return solid_tools, cut_tools, hardware_tools


def _box_domain_contains_shape(shape: Any, domain: Any) -> bool:
    """Prove containment using conservative CAD bounds for a solid box only."""
    if len(domain.Solids()) != 1 or any(face.geomType() != "PLANE" for face in domain.Faces()):
        return False
    domain_bounds = _material_bounds(domain).reshape(3, 2)
    body_bounds = _material_bounds(shape).reshape(3, 2)
    box_lengths = domain_bounds[:, 1] - domain_bounds[:, 0]
    box_volume = float(np.prod(box_lengths))
    if abs(abs(domain.Volume(tol=1e-7)) - box_volume) > max(1e-7, box_volume * 1e-8):
        return False
    tolerance = max(1e-7, float(np.max(box_lengths)) * 1e-8)
    return bool(np.all(body_bounds[:, 0] >= domain_bounds[:, 0] - tolerance)
                and np.all(body_bounds[:, 1] <= domain_bounds[:, 1] + tolerance))


def _repair_boolean_vertex_contacts(shape: Any, quality: Optional[dict[str, Any]] = None) -> Any:
    """Merge only existing short contacts in overlapping vertex tolerances.

    Boolean intersections can split a curve twice inside its endpoint
    uncertainty. Shrinking the resulting vertex tolerances is invalid: those
    tolerances may be required by the incident curves. OCC's wireframe repair
    instead merges adjacent short edges while preserving the exact surfaces.
    A genuine surface/edge intersection is never eligible for this repair.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Precision import Precision
    from OCP.ShapeFix import ShapeFix_Wireframe
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from .step_export import _assert_matching_surface_geometry

    quality = quality if quality is not None else _assess_brep_self_intersections(shape)
    if quality.get("passed", False):
        return shape

    transfer = _normalize_boolean_trims

    def vertex_contacts(report):
        faults = report.get("self_intersection_faults", [])
        return (bool(faults) and not report.get("unmapped_fault_count", 0)
                and not report.get("other_fault_count", 0)
                and all(fault.get("shape_types") == ["VERTEX"]
                        and len(fault.get("vertex_indices", [])) == 2 for fault in faults))

    # A failed raw Boolean can acquire consistent pcurve bookkeeping during
    # STEP transfer. Require a clean result to survive the next transfer too:
    # a second transfer can expose redundant vertices masked by the first.
    # Already mapped vertex contacts are repaired directly, without another
    # transfer changing their graph before the merge.
    if not vertex_contacts(quality):
        for _ in range(2):
            shape = transfer(shape)
            quality = _assess_brep_self_intersections(shape)
            if vertex_contacts(quality):
                break
            if not quality.get("passed", False):
                raise RuntimeError("Post-Boolean CAD has intersections beyond repairable vertex contacts")
        else:
            return shape
    faults = quality["self_intersection_faults"]

    vertices = shape.Vertices()
    points = np.asarray([vertex.Center().toTuple() for vertex in vertices])
    tolerances = np.asarray([BRep_Tool.Tolerance_s(vertex.wrapped) for vertex in vertices])
    eligible: set[int] = set()
    adjacent: set[int] = set()
    precision = 0.0
    for fault in faults:
        first, second = fault["vertex_indices"]
        if min(first, second) < 0 or max(first, second) >= len(vertices):
            raise RuntimeError("Post-Boolean CAD contact vertices could not be mapped")
        tolerance_sum = float(tolerances[first] + tolerances[second])
        distance = float(np.linalg.norm(points[first] - points[second]))
        connecting_edges = [edge for edge in shape.Edges() if all(
            any(vertex.wrapped.IsSame(vertices[index].wrapped) for vertex in edge.Vertices())
            for index in (first, second))]
        if (not 0.0 < distance <= tolerance_sum or not connecting_edges
                or any(edge.Length() > tolerance_sum for edge in connecting_edges)):
            raise RuntimeError("Post-Boolean CAD vertex contact has no bounded short-edge repair")
        eligible.update((first, second))
        precision = max(precision, tolerance_sum)
    vertex_map = TopTools_IndexedMapOfShape()
    for vertex in vertices:
        vertex_map.Add(vertex.wrapped)
    for edge in shape.Edges():
        indices = {vertex_map.FindIndex(part.wrapped) - 1 for part in edge.Vertices()}
        if indices & eligible:
            adjacent.update(indices)

    # Isolate all topology and tolerances. Sharing the original exact geometry
    # retains the Boolean surface representation without mutating its owner.
    isolated = BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape()
    fixer = ShapeFix_Wireframe(isolated)
    fixer.SetPrecision(precision)
    fixer.SetMaxTolerance(precision)
    fixer.ModeDropSmallEdges = False
    if not fixer.FixSmallEdges():
        raise RuntimeError("Post-Boolean CAD vertex contact could not be merged")
    repaired = cq.Shape.cast(fixer.Shape())
    if (not repaired.isValid() or not repaired.Solids()
            or len(repaired.Solids()) != len(shape.Solids())
            or len(repaired.Faces()) != sum(len(solid.Faces()) for solid in repaired.Solids())
            or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in repaired.Shells())):
        raise RuntimeError("Post-Boolean vertex repair did not preserve a valid closed solid")
    # The stricter material-volume comparison immediately below supersedes
    # the generic STEP helper's looser comparison; integrate each solid once.
    _assert_matching_surface_geometry(shape, repaired)
    volume_before, volume_after = (abs(float(body.Volume(tol=1e-7))) for body in (shape, repaired))
    if abs(volume_after - volume_before) > max(1e-9, volume_before * 1e-6):
        raise RuntimeError("Post-Boolean vertex repair changed the enclosed material")
    # OCC's optimal bounds use Confusion padding after curve consolidation;
    # unchanged planes can consequently gain 1e-7 numerical bounds padding.
    linear_tolerance = max(2.0 * Precision.Confusion_s(),
                           float(np.linalg.norm(np.ptp(points, axis=0))) * 1e-9)
    if np.max(np.abs(_material_bounds_delta(shape, repaired, linear_tolerance))) > linear_tolerance:
        raise RuntimeError("Post-Boolean vertex repair changed the material bounds")

    # The high-level OCC operation may inspect other short edges. Accept only
    # a local merge: every retained vertex remains at an original point and
    # only fault vertices may disappear. Merged trim curves can carry a new
    # approximation tolerance; only their existing adjacent endpoints may
    # inherit that edge tolerance, within the original contact uncertainty.
    retained: set[int] = set()
    for vertex in repaired.Vertices():
        distances = np.linalg.norm(points - np.asarray(vertex.Center().toTuple()), axis=1)
        index = int(np.argmin(distances))
        allowed_tolerance = float(tolerances[index])
        if index in adjacent:
            incident_tolerances = [BRep_Tool.Tolerance_s(edge.wrapped) for edge in repaired.Edges()
                if any(part.wrapped.IsSame(vertex.wrapped) for part in edge.Vertices())]
            edge_tolerance = max(incident_tolerances, default=0.0)
            if edge_tolerance > precision:
                raise RuntimeError("Post-Boolean trim repair exceeded the existing contact uncertainty")
            allowed_tolerance = max(allowed_tolerance, edge_tolerance)
        if (distances[index] > 1e-10
                or BRep_Tool.Tolerance_s(vertex.wrapped) > allowed_tolerance * (1.0 + 1e-6) + 1e-12):
            raise RuntimeError("Post-Boolean vertex repair changed unrelated vertex geometry")
        retained.add(index)
    if not (set(range(len(vertices))) - retained).issubset(eligible):
        raise RuntimeError("Post-Boolean vertex repair removed unrelated vertices")
    if not _assess_brep_self_intersections(repaired).get("passed", False):
        raise RuntimeError("Post-Boolean vertex repair left unresolved intersections")
    logger.info("Recovered Shape STEP: merged %d overlapping Boolean contact vertex/vertices",
                len(vertices) - len(retained))
    verified = transfer(repaired)
    if not _assess_brep_self_intersections(verified).get("passed", False):
        raise RuntimeError("Post-Boolean vertex repair did not survive verified STEP transfer")
    return verified


def _source_overlap_witness(material: Any, domain: Any) -> bool:
    """Establish nonempty overlap independently of a failed Boolean classifier."""
    from cadquery import Vector
    from OCP.BRep import BRep_Tool
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.Precision import Precision
    from OCP.TopAbs import TopAbs_IN

    tolerance = Precision.Confusion_s()
    classifiers = [BRepClass3d_SolidClassifier(solid.wrapped) for solid in domain.Solids()]
    points = [BRep_Tool.Pnt_s(vertex.wrapped) for vertex in material.Vertices()]
    for solid in material.Solids():
        center = solid.Center().toPnt()
        own_classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        own_classifier.Perform(center, tolerance)
        if own_classifier.State() == TopAbs_IN:
            points.append(center)
        # An annulus has its mass center inside the bore, and every seam
        # vertex may lie exactly ON a containing body's analytic interfaces.
        # Seek actual interior points beside its face interiors instead of
        # mistaking that absence of vertex/centroid witnesses for detachment.
        # Every candidate must first classify IN the requested material.
        box = solid.BoundingBox()
        lengths = [length for length in (box.xlen, box.ylen, box.zlen) if length > tolerance]
        step = min(lengths, default=1.0) * 1e-3
        for face in solid.Faces():
            try:
                u0, u1, v0, v1 = face._uvBounds()
                surface = BRep_Tool.Surface_s(face.wrapped)
                point = surface.Value((u0+u1)*.5, (v0+v1)*.5)
                vector = face.normalAt(Vector(point)).multiply(-step)
                candidate = point.Translated(vector.wrapped)
                own_classifier.Perform(candidate, tolerance)
                if own_classifier.State() == TopAbs_IN:
                    points.append(candidate)
            except Exception:
                # This is positive evidence only. Unsupported faces retain
                # the existing vertex/centroid test and paired Boolean guards.
                continue
    for point in points:
        for classifier in classifiers:
            classifier.Perform(point, tolerance)
            if classifier.State() == TopAbs_IN:
                return True
    return False


def _assert_source_contains_material(material: Any, domain: Any) -> None:
    """Verify both sides of a contained solid's Boolean partition."""
    from OCP.Precision import Precision
    from .step_export import _closed

    volume = abs(float(material.Volume(tol=1e-7)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise RuntimeError("Source containment requires positive finite material volume")
    volume_tolerance = max(1e-7, volume * 1e-6)
    bounds = _material_bounds(material)
    linear_tolerance = max(2.0 * Precision.Confusion_s(),
                           float(np.linalg.norm(bounds[1::2] - bounds[::2])) * 1e-9)
    last_error = None
    retry_operand = material
    # Coincident trimmed faces can be misclassified even in a valid Boolean
    # result. Retry only the smallest existing contact tolerance, and require
    # both an empty Cut and a complete Common before accepting containment.
    for fuzzy in (0.0, _BOOLEAN_FUZZY_RETRIES[0]):
        if fuzzy and not _source_overlap_witness(retry_operand, domain):
            last_error = RuntimeError("no independent source-interior overlap witness")
            break
        try:
            outside = _native_boolean(material, domain, 'cut', fuzzy)
            outside_volume = sum(abs(float(solid.Volume(tol=1e-7)))
                                 for solid in outside.Solids())
            if not fuzzy and outside.Solids():
                # Retry a claimed whole-body exterior only when that actual
                # exterior contradicts an independent source classification.
                retry_operand = outside
            if (outside.isValid() and np.isfinite(outside_volume)
                    and volume_tolerance < outside_volume < volume - volume_tolerance):
                # A real partial protrusion is not a whole-body classifier
                # failure. Never erase it by retrying at a fuzzy tolerance.
                last_error = RuntimeError(f"outside partition contains {outside_volume:.9g} model-volume units")
                break
            if (not outside.isValid() or not np.isfinite(outside_volume)
                    or outside_volume > volume_tolerance):
                raise RuntimeError(f"outside partition contains {outside_volume:.9g} model-volume units")
            inside = _native_boolean(material, domain, 'common', fuzzy)
            if not inside.isValid() or not _closed(inside):
                raise RuntimeError("inside partition is not a valid closed solid")
            inside_volume = abs(float(inside.Volume(tol=1e-7)))
            if (not np.isfinite(inside_volume)
                    or abs(inside_volume - volume) > volume_tolerance):
                raise RuntimeError("inside partition did not retain the complete material volume")
            if np.max(np.abs(_material_bounds_delta(material, inside, linear_tolerance))) > linear_tolerance:
                raise RuntimeError("inside partition changed the material bounds")
        except Exception as error:
            last_error = error
            continue
        if fuzzy:
            logger.info("Verified source containment with paired Cut/Common at %g contact tolerance", fuzzy)
        return
    raise RuntimeError("Could not verify reconstructed material stays within the source CAD domain: "
                       f"{last_error}") from last_error


def _apply_passive_regions_to_step(
    shape: Any,
    payload: Any,
    *,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    validate_after_boolean: bool = True,
    repair_boolean_contacts: bool = False,
    prepare_continuity: bool = True,
    require_all_solid_tools: bool = False,
    source_clip_after_passive: bool = False,
) -> Any:
    """Re-apply passive boxes/cylinders as analytic STEP features."""
    import cadquery as cq

    result = shape
    solid_tools, cut_tools, hardware_tools = _collect_passive_tools(
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    requested_solid_tools = list(solid_tools)
    if require_all_solid_tools:
        solid_tools = [_canonical_annular_layer(tool) for tool in solid_tools]
        # Canonical annuli have already proved two-sided material equality
        # with the original selections. Keep that same verified trim form in
        # the downstream containment proof against the reconstructed body.
        requested_solid_tools = list(solid_tools)
    source_domain = _coerce_to_solid(payload.get('source_cad_shape')) if isinstance(payload,dict) else None
    validation_cut_tools = list(cut_tools)
    # The repair path always checks the complete material body before any
    # material/source/void guard or return. Its whole-body scan also covers
    # normalized intermediate Booleans, avoiding another scan per operation.
    check_intermediate_intersections = not repair_boolean_contacts
    if prepare_continuity and (source_domain is not None or solid_tools or cut_tools):
        result = _prepare_boolean_continuity(result)
    if source_domain is not None and source_clip_after_passive:
        # Reordering (F intersect S) union A to (F union A) intersect S
        # requires every mandatory added material A to lie within S. Keep
        # the actual source CAD and its final Boolean/containment guards.
        for tool in solid_tools:
            _assert_source_contains_material(tool, source_domain)
    if source_domain is not None and not source_clip_after_passive:
        # Match source bores and outer surfaces before adding face-coincident
        # protected layers, so both operands share the analytic boundaries.
        # A contained body already satisfies a box domain. Avoid intersecting
        # coincident fitted faces when conservative bounds prove this exactly.
        # Domains with bores or concavities still require the actual Boolean.
        if not _box_domain_contains_shape(result, source_domain):
            result = _boolean_or_raise(result, source_domain, operation='common',
                check_normalized_intersections=check_intermediate_intersections)
        # A bore already excluded by the source CAD needs no further cut.
        # Re-cutting coincident cylinder walls can create microscopic strips
        # that pass solid checks but cannot be tessellated reliably.
        needed_cuts = []
        for tool in cut_tools:
            overlap = _native_boolean(source_domain, tool, 'common')
            overlap_volume = sum(abs(float(solid.Volume(tol=1e-7))) for solid in overlap.Solids())
            if not overlap.isValid() or overlap_volume > max(1e-8,abs(float(tool.Volume(tol=1e-7)))*1e-8):
                needed_cuts.append(tool)
        cut_tools = needed_cuts

    # First remove the bores from the reconstructed body. This also removes the
    # recovered-mesh copy of a joint pin, allowing the exact analytic pin to
    # be added once at the end.
    defer_bore_cuts = source_clip_after_passive and require_all_solid_tools and bool(solid_tools)
    if cut_tools and not defer_bore_cuts:
        result = _cut_tools_sequentially(result, cut_tools,
            check_normalized_intersections=check_intermediate_intersections)

    if not require_all_solid_tools:
        solid_tools = _intersecting_tools(result, solid_tools)
    if solid_tools:
        # Overlapping preserved regions are material solids, not an already
        # fused compound. Add them independently to retain the complete body.
        for tool in solid_tools:
            for solid in tool.Solids():
                previous_solid_count = len(result.Solids())
                fused = _boolean_or_raise(result, solid, operation="fuse",
                    check_normalized_intersections=check_intermediate_intersections)
                if require_all_solid_tools and len(fused.Solids()) > previous_solid_count:
                    raise RuntimeError('Reconstructed CAD did not retain the complete non-design material '
                                       'as connected material')
                result = fused
        # A sleeve is an outer keep-material cylinder, not a plug. Re-cut the
        # exact bore after the fuse so its running surface stays analytic.
        if cut_tools:
            result = _cut_tools_sequentially(result, cut_tools,
                check_normalized_intersections=check_intermediate_intersections)

    if source_domain is not None and source_clip_after_passive:
        # Analytic collars and bores are present before this intersection,
        # avoiding intersections of approximated hole walls with their exact
        # source counterparts. This is the same mandatory material set.
        result = _boolean_or_raise(result, source_domain, operation='common',
            check_normalized_intersections=check_intermediate_intersections)

    result = _drop_boolean_sliver_solids(result)
    # Validate the geometry actually returned after face consolidation.
    # Material checks must not use the pre-consolidation B-rep.
    if solid_tools or cut_tools:
        result = _unify_same_domain_shape(result, merge_angle_deg=0.25)
    if repair_boolean_contacts:
        result = _repair_boolean_vertex_contacts(result)
    material_result = result

    # Pins/shafts are assembly hardware. Add exact analytic cylinders after
    # bore cuts and keep them as separate solids; fusing would incorrectly
    # claim bonded contact and would erase the intended running clearance.
    if hardware_tools:
        existing_parts = []
        try:
            existing_parts.extend(result.Solids())
        except Exception:
            existing_parts.append(result)
        result = _compound_or_single(existing_parts + hardware_tools)

    n_fused = len(solid_tools)
    n_cut = len(cut_tools)
    n_hardware = len(hardware_tools)
    if n_fused or n_cut or n_hardware:
        if validate_after_boolean:
            _assert_valid_occ_shape(result, label="STEP body after passive booleans")
        logger.info(
            "Recovered Shape STEP: re-applied %d analytic passive solid(s), "
            "cut %d analytic passive void(s), added %d separate hardware "
            "solid(s).",
            n_fused,
            n_cut,
            n_hardware,
        )
    result = _single_solid_if_possible(result)
    if not isinstance(result, cq.Shape):
        result = cq.Shape.cast(result)
    # A valid solid alone cannot establish preservation: failed booleans can
    # return a different, still-valid body. Check the requested material.
    for tool in requested_solid_tools:
        expected = _cut_tools_sequentially(tool, cut_tools, allow_empty_cut=True) if cut_tools else tool
        if not expected.Solids():
            continue
        try:
            # A coincident cylindrical layer can produce an invalid whole-
            # layer Cut even after a successful union. Empty Cut alone also
            # cannot prove retention. Require the paired empty exterior and
            # complete, closed Common, with independent volume/bounds checks.
            _assert_source_contains_material(expected, material_result)
        except RuntimeError as error:
            raise RuntimeError('Reconstructed CAD did not retain the complete non-design material') from error
    for tool in validation_cut_tools:
        intrusion = _native_boolean(material_result, tool, 'common')
        intrusion_volume = sum(abs(float(solid.Volume(tol=1e-7))) for solid in intrusion.Solids())
        if not intrusion.isValid() or intrusion_volume > max(1e-7, abs(float(tool.Volume(tol=1e-4)))*1e-6):
            raise RuntimeError('Reconstructed CAD intruded into a protected bore or void')
    if source_domain is not None and not _box_domain_contains_shape(material_result, source_domain):
        _assert_source_contains_material(material_result, source_domain)
    return result


def restore_paper_non_design_regions(
    shape: Any,
    payload: Any,
    *,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Restore protected CAD interfaces after the single paper fitting stage.

    The fitted surface covers the whole recovered boundary, including sampled
    copies of the original holes and non-design layers. Those samples are not
    exact CAD interfaces. Intersecting with the actual source solid removes
    bore intrusions and exterior bumps; adding the original protected layers
    restores their complete analytic material. The existing paired material,
    source and bore checks remain authoritative.

    A shared-boundary tensor surface can retain C0 knots at chart junctions.
    Regularize those exact spline spans once before the native Booleans; this
    preserves their surfaces and checks the enclosed material independently.
    This operation never refits or repartitions the source charts.
    """
    from .step_export import _closed

    solids, cuts, hardware = _collect_passive_tools(
        payload, void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    source = (_coerce_to_solid(payload.get('source_cad_shape'))
              if isinstance(payload, dict) else None)
    report = {
        'method': 'Exact source-domain clipping and original non-design CAD layers',
        'source_cad_applied': source is not None,
        'preserved_solid_count': len(solids),
        'preserved_void_count': len(cuts),
        'hardware_count': len(hardware),
        'fitting_attempts': 0,
    }
    if source is None and not (solids or cuts or hardware):
        return shape, {**report, 'applied': False}
    if not shape.isValid() or not _closed(shape):
        raise RuntimeError('Exact non-design restoration requires a valid closed reconstructed body')
    boolean_ready = _prepare_boolean_continuity(shape)
    report.update(input_face_count=len(shape.Faces()),
                  boolean_ready_face_count=len(boolean_ready.Faces()),
                  exact_continuity_subdivision=len(boolean_ready.Faces()) != len(shape.Faces()))
    from .amrto_parameter_domains import normalize_paper_parameter_domains
    boolean_ready, parameter_domains = normalize_paper_parameter_domains(boolean_ready)
    report['parameter_domains'] = parameter_domains
    result = _apply_passive_regions_to_step(
        boolean_ready, payload, void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin, prepare_continuity=False,
        require_all_solid_tools=True,
    )
    if not result.isValid() or not _closed(result):
        raise RuntimeError('Exact non-design restoration did not retain valid closed material')
    return result, {**report, 'applied': True,
                    'solid_count': len(result.Solids()),
                    'face_count': len(result.Faces())}


def _attach_passive_regions_to_lattice(
    shape: Any,
    payload: Any,
    *,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
) -> tuple[Any, dict[str, int]]:
    """Add passive features to an analytic lattice without a general fuse.

    A strut lattice is deliberately a compound of overlapping exact solids --
    that is what keeps its face count proportional to the member count -- and
    OCC's general fuse is measured to come back *invalid* on exactly that
    argument set: fusing three interface pads into a 208-solid BCC block here
    produced an invalid body and lost the whole reconstruction.

    So the preserved pads join the compound as solids of their own, which is
    the same relationship the members already have with one another. Both forms
    are valid STEP and both are exact; what changes is that this one survives.

    Member radii can cross a protected clearance even when their centrelines
    pass envelope clipping. Cut only members whose bounds reach a void tool.
    """
    import cadquery as cq

    solid_tools, cut_tools, hardware_tools = _collect_passive_tools(
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    counts = {
        "attached_solids": 0,
        "hardware_solids": len(hardware_tools),
        "member_solids": 0,
        "cut_member_solids": 0,
    }

    parts: list[Any] = []
    try:
        parts.extend(shape.Solids())
    except Exception:
        parts.append(shape)
    counts["member_solids"] = len(parts)
    if cut_tools:
        trimmed = []
        for part in parts:
            nearby = [tool for tool in cut_tools
                      if _bounds_reach_any(_shape_bounds(part), [_shape_bounds(tool)])]
            if nearby:
                part = _cut_tools_sequentially(part, nearby, allow_empty_cut=True)
                counts["cut_member_solids"] += 1
                trimmed.extend(part.Solids())
            else:
                trimmed.append(part)
        parts = trimmed

    if solid_tools or hardware_tools:
        # `_intersecting_tools` runs an OCC common per tool. Against a compound
        # of a few thousand overlapping members that boolean does not finish in
        # any useful time -- measured here at over ten minutes for two pads on a
        # 1775-member block, versus the 1.2 s the whole lattice took to build.
        # Bounding boxes answer the only question being asked (does this pad
        # reach the lattice at all, or is it a detached sleeve left behind by
        # thresholding) at a cost of one comparison per member.
        member_boxes = [_shape_bounds(part) for part in parts]
        member_boxes = [box for box in member_boxes if box is not None]
        retained = [
            tool
            for tool in solid_tools
            if _bounds_reach_any(_shape_bounds(tool), member_boxes)
        ]
        skipped = len(solid_tools) - len(retained)
        if skipped:
            logger.warning(
                "Analytic lattice STEP: skipped %d detached passive solid "
                "feature(s) that no member reaches.",
                skipped,
            )
        counts["attached_solids"] = len(retained)
        retained = [_cut_tools_sequentially(tool, cut_tools, allow_empty_cut=True) if cut_tools else tool
                    for tool in retained]
        retained = [tool for tool in retained if tool.Solids()]
        parts = parts + retained + list(hardware_tools)

    result = _compound_or_single(parts)
    if result is None:
        raise RuntimeError("Protected clearances removed the entire beam lattice")
    if not isinstance(result, cq.Shape):
        result = cq.Shape.cast(result)
    logger.info(
        "Analytic lattice STEP: %d member solid(s), %d passive solid(s) "
        "attached, %d hardware solid(s) added.",
        counts["member_solids"],
        counts["attached_solids"],
        counts["hardware_solids"],
    )
    return result, counts


def _shape_bounds(shape: Any) -> Optional[tuple[float, ...]]:
    """Axis-aligned bounds of a shape as ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
    try:
        box = shape.BoundingBox()
        return (box.xmin, box.xmax, box.ymin, box.ymax, box.zmin, box.zmax)
    except Exception:
        return None


def _bounds_reach_any(
    box: Optional[tuple[float, ...]],
    others: list[tuple[float, ...]],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """True when ``box`` overlaps at least one of ``others``."""
    if box is None:
        return False
    for other in others:
        if (
            box[0] <= other[1] + tolerance
            and other[0] <= box[1] + tolerance
            and box[2] <= other[3] + tolerance
            and other[2] <= box[3] + tolerance
            and box[4] <= other[5] + tolerance
            and other[4] <= box[5] + tolerance
        ):
            return True
    return False
