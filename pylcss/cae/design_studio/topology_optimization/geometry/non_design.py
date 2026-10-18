# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Closed, conformal layers of preserved material selected on CAD faces."""

from __future__ import annotations

import json
import math
from typing import Any

import cadquery as cq
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid

_NON_DESIGN_GEOMETRY_REVISION = 2


def design_domain_shape(value: Any) -> cq.Shape:
    """Resolve the same CAD face order used by the viewer, including assemblies."""
    if isinstance(value, dict):
        value = value.get("shape", value.get("components", value.get("geometry")))
    if hasattr(value, "toCompound"):
        value = value.toCompound()
    if isinstance(value, (list, tuple)):
        shapes = [design_domain_shape(item) for item in value if item is not None]
        if not shapes:
            raise ValueError("Connect a CAD design domain before selecting faces.")
        return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
    if hasattr(value, "val"):
        value = value.val()
    if not isinstance(value, cq.Shape) or not value.Solids():
        raise ValueError("Non-design face layers require a closed CAD design domain.")
    return value


def read_non_design_regions(value: Any) -> list[dict[str, Any]]:
    """Validate the serializable face selections and their physical thicknesses."""
    try:
        regions = json.loads(value or "[]") if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid saved non-design face selection.") from exc
    if not isinstance(regions, list):
        raise ValueError("Non-design regions must be a list of face selections.")
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Invalid non-design region.")
        indices = region.get("face_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("Select at least one CAD face for each non-design region.")
        try:
            thickness = float(region["thickness_mm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Enter a positive non-design thickness in mm.") from exc
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Non-design thickness must be positive and finite.")
    return regions


def face_signature(face: cq.Face) -> dict[str, Any]:
    return {
        "type": face.geomType(),
        "area": face.Area(),
        "center": list(face.Center().toTuple()),
    }


def _matches_signature(face: cq.Face, signature: dict[str, Any]) -> bool:
    current = face_signature(face)
    return (
        current["type"] == signature.get("type")
        and math.isclose(
            current["area"],
            float(signature.get("area", -1)),
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        and all(
            math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-6)
            for a, b in zip(current["center"], signature.get("center", []))
        )
        and len(signature.get("center", [])) == 3
    )


def _clip_cylindrical_layer(layer: cq.Shape, domain: cq.Shape) -> cq.Shape:
    """Intersect an offset cylinder without introducing subtractive trim seams.

    Some coincident cylinders are misclassified by Common. Its independent
    exterior partition must account for all of the input material, and the
    result must pass both existing source and offset-layer containment checks.
    Only an exact trim normalization may precede a retry. The older two-Cut
    construction remains a checked fallback for the coincident-face case.
    """
    from .amrto_volume import measure_cad_volume
    from .cad_features import (
        _assert_source_contains_material, _native_boolean, _normalize_boolean_trims,
    )
    from .step_export import _closed

    # An offset can already lie completely inside the source. Prove this with
    # both Boolean partitions before avoiding a redundant clipping operation:
    # even Common can introduce tiny self-crossing cap trims at a shoulder.
    try:
        _assert_source_contains_material(layer, domain)
        return layer
    except (ValueError, RuntimeError):
        pass
    normalized_layer = None
    try:
        # A simple bore can have ambiguous offset pcurves too. Normalize the
        # unchanged layer before trusting an invalid exterior classification.
        normalized_layer = _normalize_boolean_trims(layer)
        _assert_source_contains_material(normalized_layer, domain)
        return normalized_layer
    except (ValueError, RuntimeError):
        pass

    reference, _ = measure_cad_volume(layer)
    tolerance = reference * 1e-6
    exterior = _native_boolean(layer, domain, "cut")
    if not exterior.isValid() or (exterior.Faces() and not _closed(exterior)):
        raise ValueError("The cylindrical layer exterior is not a valid closed partition.")
    outside = sum(measure_cad_volume(solid)[0] for solid in exterior.Solids())
    expected = reference - outside
    if expected <= 0.0:
        raise ValueError("The cylindrical face layer has no material inside its source.")

    def checked(candidate):
        if not candidate.isValid() or not _closed(candidate):
            raise ValueError("The cylindrical face intersection is not a valid closed solid.")
        volume, _ = measure_cad_volume(candidate)
        if abs(volume - expected) > tolerance:
            raise ValueError("The cylindrical face intersection did not retain its complete material partition.")
        _assert_source_contains_material(candidate, domain)
        _assert_source_contains_material(candidate, layer)
        return candidate

    for normalize in (False, True):
        try:
            operand = (normalized_layer if normalized_layer is not None
                       else _normalize_boolean_trims(layer)) if normalize else layer
            return checked(_native_boolean(operand, domain, "common"))
        except (ValueError, RuntimeError):
            continue
    return checked(_native_boolean(
        domain, _native_boolean(domain, layer, "cut", 1e-6), "cut", 1e-6,
    ))


def face_material_layer(face: cq.Face, domain: cq.Shape, thickness: float) -> cq.Shape:
    """Thicken the actual trimmed surface and retain only its interior material.

    The negative offset follows the oriented solid face inward, including
    cavity faces. Clipping retains only material inside the source body.
    """
    candidates = []
    if face.geomType() == "PLANE":
        vector = face.normalAt().multiply(-thickness)
        layer = cq.Solid.extrudeLinear(face.outerWire(), face.innerWires(), vector)
        candidates.append(layer.intersect(domain))
    else:
        for solid in domain.Solids():
            solid_faces = solid.Faces()
            if len(solid_faces) == 1 and solid_faces[0].isSame(face):
                try:
                    candidates.append(solid.shell([], -thickness))
                except Exception:
                    pass
        for offset in (-thickness,):
            try:
                builder = BRepOffsetAPI_MakeThickSolid()
                builder.MakeThickSolidBySimple(face.wrapped, offset)
                if builder.IsDone():
                    layer = cq.Shape.cast(builder.Shape())
                    if layer.Volume() < 0.0:
                        layer = cq.Shape.cast(layer.wrapped.Reversed())
                    if layer.isValid() and layer.Solids():
                        if face.geomType() == "CYLINDER":
                            # Subtracting twice can introduce malformed caps
                            # where a counterbore meets a curved shoulder.
                            clipped = _clip_cylindrical_layer(layer, domain)
                        else:
                            clipped = domain.cut(domain.cut(layer, tol=1e-6), tol=1e-6)
                        if (
                            not clipped.Solids() or clipped.Volume() <= 1e-9
                        ) and face.geomType() in {"BSPLINE", "BEZIER", "OFFSET"}:
                            # OCC may classify a valid offset spline as empty
                            # during intersection. A sewn faceted copy removes
                            # that ambiguous offset-surface representation;
                            # clip it against the original CAD boundary.
                            vertices, triangles = layer.tessellate(
                                max(0.001, thickness * 0.2), 0.3
                            )
                            facets = [
                                cq.Face.makeFromWires(
                                    cq.Wire.makePolygon(
                                        [vertices[index] for index in triangle],
                                        close=True,
                                    )
                                )
                                for triangle in triangles
                            ]
                            faceted = cq.Solid.makeSolid(
                                cq.Shell.makeShell(facets)
                            ).fix()
                            if faceted.Volume() < 0.0:
                                faceted = cq.Shape.cast(faceted.wrapped.Reversed())
                            if faceted.isValid():
                                clipped = faceted.intersect(domain)
                        candidates.append(clipped)
            except Exception:
                # Report an invalid offset below instead of preserving a
                # different face or thickness silently.
                continue
    candidates = [
        layer
        for layer in candidates
        if layer.Solids() and layer.isValid() and layer.Volume() > 1e-9
    ]
    if not candidates:
        raise ValueError(
            "This face cannot form a valid inward layer at the requested thickness. Reduce the thickness or select another face."
        )
    return max(candidates, key=lambda layer: layer.Volume())


def cached_non_design_payloads(
    node: Any, design_domain: Any,
) -> list[dict[str, Any]] | None:
    """Read preserved layers for display without doing CAD boolean operations."""
    if not node.get_property("non_design_enabled"):
        return []
    raw = node.get_property("non_design_regions") or "[]"
    if not read_non_design_regions(raw):
        return []
    cached = getattr(node, "_non_design_cache", None)
    if (cached is not None and len(cached) == 4
            and cached[3] == _NON_DESIGN_GEOMETRY_REVISION and cached[0] == raw):
        domain = design_domain_shape(design_domain)
        if domain.isSame(cached[1]):
            return cached[2]
    return None


def non_design_payloads(node: Any, design_domain: Any) -> list[dict[str, Any]]:
    """Build (and cache) exactly the volumes used by preview and optimization."""
    if not node.get_property("non_design_enabled"):
        return []
    raw = node.get_property("non_design_regions") or "[]"
    regions = read_non_design_regions(raw)
    if not regions:
        return []
    domain = design_domain_shape(design_domain)
    cached = getattr(node, "_non_design_cache", None)
    if (cached is not None and len(cached) == 4
            and cached[3] == _NON_DESIGN_GEOMETRY_REVISION
            and cached[0] == raw and domain.isSame(cached[1])):
        return cached[2]
    faces = domain.Faces()
    payloads = []
    for region in regions:
        layers = []
        signatures = region.get("face_signatures", [])
        for position, index in enumerate(region["face_indices"]):
            if index >= len(faces) or (
                signatures
                and (
                    position >= len(signatures)
                    or not _matches_signature(faces[index], signatures[position])
                )
            ):
                raise ValueError(
                    "The selected non-design face has changed. Select its faces again in the inspector."
                )
            layers.append(
                face_material_layer(faces[index], domain, float(region["thickness_mm"]))
            )
        geometry = layers[0]
        for layer in layers[1:]:
            geometry = geometry.fuse(layer)
        payloads.append(
            {
                "type": "topology_non_design_region",
                "region_type": "solid",
                "geometry": geometry,
            }
        )
    node._non_design_cache = (raw, domain, payloads, _NON_DESIGN_GEOMETRY_REVISION)
    return payloads
