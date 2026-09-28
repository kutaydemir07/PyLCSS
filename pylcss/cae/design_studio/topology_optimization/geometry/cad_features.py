# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Reapply analytic passive and non-design regions to recovered CAD."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .occ_shapes import (
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


def _boolean_or_raise(result: Any, tool: Any, *, operation: str) -> Any:
    """Run a CadQuery boolean and give a targeted failure message."""

    def _run(fuzzy: float) -> Any:
        if operation == "fuse":
            return result.fuse(tool, tol=fuzzy) if fuzzy else result.fuse(tool)
        return result.cut(tool, tol=fuzzy) if fuzzy else result.cut(tool)

    last_error: Exception | None = None
    for fuzzy in (0.0, *_BOOLEAN_FUZZY_RETRIES):
        try:
            candidate = _run(fuzzy)
        except Exception as exc:  # noqa: PERF203 - each tolerance is a retry
            last_error = exc
            continue
        if candidate is None:
            continue
        if hasattr(candidate, "isValid") and not candidate.isValid():
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
            common = shape.intersect(tool)
            common_volume = abs(float(common.Volume()))
            tool_volume = max(abs(float(tool.Volume())), 1.0)
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


def _cut_tools_sequentially(shape: Any, tools: list[Any]) -> Any:
    """Apply overlapping bore/cleanup tools one at a time.

    OCC compounds containing overlapping cylinders are not a boolean union;
    cutting by such a compound can leave a near-complete duplicate pin and
    microscopic sliver solids. Sequential validated cuts are deterministic for
    the small number of preserved engineering features used here.
    """
    result = shape
    for tool in tools:
        result = _boolean_or_raise(result, tool, operation="cut")
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
        [max(0.0, abs(float(solid.Volume()))) for solid in solids],
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

    return solid_tools, cut_tools, hardware_tools


def _apply_passive_regions_to_step(
    shape: Any,
    payload: Any,
    *,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    validate_after_boolean: bool = True,
) -> Any:
    """Re-apply passive boxes/cylinders as analytic STEP features."""
    import cadquery as cq

    result = shape
    solid_tools, cut_tools, hardware_tools = _collect_passive_tools(
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )

    # First remove the bores from the reconstructed body. This also removes the
    # recovered-mesh copy of a joint pin, allowing the exact analytic pin to
    # be added once at the end.
    if cut_tools:
        result = _cut_tools_sequentially(result, cut_tools)

    solid_tools = _intersecting_tools(result, solid_tools)
    solid_compound = _compound_or_single(solid_tools)
    if solid_compound is not None:
        result = _boolean_or_raise(result, solid_compound, operation="fuse")
        # A sleeve is an outer keep-material cylinder, not a plug. Re-cut the
        # exact bore after the fuse so its running surface stays analytic.
        if cut_tools:
            result = _cut_tools_sequentially(result, cut_tools)

    result = _drop_boolean_sliver_solids(result)

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
        result = _unify_same_domain_shape(result, merge_angle_deg=0.25)
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
    return result


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

    Keep-void regions are deliberately not cut here. A member only exists where
    the optimizer's density passed the cutoff, and a keep-void region holds that
    density at zero, so the lattice was never generated inside one. Cutting it
    again costs an OCC boolean per member -- thousands of them -- to remove
    nothing.
    """
    import cadquery as cq

    solid_tools, _cut_tools, hardware_tools = _collect_passive_tools(
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    counts = {
        "attached_solids": 0,
        "hardware_solids": len(hardware_tools),
        "member_solids": 0,
    }

    parts: list[Any] = []
    try:
        parts.extend(shape.Solids())
    except Exception:
        parts.append(shape)
    counts["member_solids"] = len(parts)

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
        parts = parts + retained + list(hardware_tools)

    result = _compound_or_single(parts)
    if result is None:
        result = shape
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
