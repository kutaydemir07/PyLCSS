# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Measure continuity on the final CAD boundary without certifying Class A.

Spline degree, sewing success and smooth display normals do not establish
continuity between faces. This module samples the actual OCCT face pcurves,
first derivatives and second derivatives. G2 compares the full curvature
tensor in world coordinates, rather than just the two principal curvatures
(which would miss a change of their directions).

These are bounded, sampled boundary diagnostics. They do not replace interior
curvature/highlight inspection, manufacturing tolerances, or structural
reanalysis after reconstruction. See Autodesk's "Understanding Class A
Modeling" and OCCT's BRepAdaptor_Surface / BRepAdaptor_Curve2d documentation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _xyz(value: Any) -> np.ndarray:
    return np.array((value.X(), value.Y(), value.Z()), dtype=float)


def _surface_differential(
    surface: Any, u: float, v: float, orientation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position, oriented unit normal and ambient curvature tensor."""
    from OCP.gp import gp_Pnt, gp_Vec

    point = gp_Pnt()
    du, dv, duu, dvv, duv = (gp_Vec() for _ in range(5))
    surface.D2(float(u), float(v), point, du, dv, duu, dvv, duv)
    first = np.column_stack((_xyz(du), _xyz(dv)))
    normal = np.cross(first[:, 0], first[:, 1])
    scale = float(np.linalg.norm(first[:, 0]) * np.linalg.norm(first[:, 1]))
    magnitude = float(np.linalg.norm(normal))
    if not np.isfinite(scale) or scale == 0.0 or magnitude <= 1e-12 * scale:
        raise ValueError("Surface derivatives are singular at the boundary sample.")
    normal *= float(orientation) / magnitude
    second = np.array(
        ((normal @ _xyz(duu), normal @ _xyz(duv)),
         (normal @ _xyz(duv), normal @ _xyz(dvv))),
        dtype=float,
    )
    # The pseudoinverse maps world tangent vectors to (du, dv). Pulling the
    # second fundamental form back to world coordinates makes comparisons
    # independent of the faces' parameter directions and parameter speeds.
    inverse = np.linalg.pinv(first, rcond=1e-12)
    curvature = inverse.T @ second @ inverse
    position = _xyz(point)
    if not all(np.all(np.isfinite(item)) for item in (position, normal, curvature)):
        raise ValueError("Non-finite CAD surface derivative.")
    return position, normal, curvature


def assess_surface_quality(
    candidate: Any,
    *,
    position_tolerance: float = 1e-5,
    tangent_tolerance_deg: float = 0.1,
    curvature_relative_tolerance: float = 0.05,
    curvature_absolute_tolerance: float = 1e-6,
    samples_per_edge: int = 7,
    max_edges: int = 2000,
) -> dict[str, Any]:
    """Return JSON-safe sampled G0/G1/G2 measurements for a CAD shape.

    Accepts a CadQuery shape or a TopoDS shape. Position tolerance uses model
    length units; absolute curvature tolerance uses their inverse. Defaults
    are diagnostic settings, not an industry Class A specification. Pass
    counts are cumulative: each G2 edge is also counted in G1 and G0.

    Sharp edges are reported, never silently labelled intentional or excluded
    from the result. Periodic seams use both pcurves. Singular/degenerate
    edges and unmeasurable pcurves remain explicitly unevaluated. Sampling
    excludes exact vertices, and does not prove continuity between samples.
    """
    tolerances = {
        "position_tolerance": float(position_tolerance),
        "tangent_tolerance_deg": float(tangent_tolerance_deg),
        "curvature_relative_tolerance": float(curvature_relative_tolerance),
        "curvature_absolute_tolerance": float(curvature_absolute_tolerance),
    }
    if any(not np.isfinite(v) or v < 0.0 for v in tolerances.values()):
        raise ValueError("Surface-quality tolerances must be finite and non-negative.")
    if tangent_tolerance_deg > 180.0:
        raise ValueError("Tangent tolerance must not exceed 180 degrees.")
    if int(samples_per_edge) != samples_per_edge or samples_per_edge < 3:
        raise ValueError("Surface-quality sampling needs at least three samples per edge.")
    if int(max_edges) != max_edges or max_edges < 1:
        raise ValueError("Surface-quality edge budget must be a positive integer.")
    report: dict[str, Any] = {
        "status": "unavailable",
        "class_a_certified": False,
        "scope": "sampled CAD face boundaries",
        "edge_count": 0,
        "evaluated_edges": 0,
        "g0_edges": 0,
        "g1_edges": 0,
        "g2_edges": 0,
        "sharp_edges": 0,
        "curvature_discontinuous_edges": 0,
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "degenerate_edges": 0,
        "seam_edges": 0,
        "unevaluated_edges": 0,
        "sample_count": 0,
        "max_gap": None,
        "max_normal_angle_deg": None,
        "max_curvature_relative_error": None,
        "max_curvature_absolute_error": None,
        "samples_per_edge": int(samples_per_edge),
        "edge_budget": int(max_edges),
        "limitations": [
            "Finite boundary samples exclude exact vertices and do not prove continuity between samples.",
            "Face interiors, curvature variation (G3), highlight flow and reflection quality are not assessed.",
            "Class A requires project-specific tolerances and additional visual and engineering review.",
        ],
        **tolerances,
    }
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve2d, BRepAdaptor_Surface
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCP.TopoDS import TopoDS

        wrapped = getattr(candidate, "wrapped", candidate)
        if wrapped is None or wrapped.IsNull():
            raise ValueError("No CAD shape is available for surface evaluation.")
        adjacency = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(wrapped, TopAbs_EDGE, TopAbs_FACE, adjacency)
    except Exception as exc:
        report["reason"] = str(exc)
        return report

    report["edge_count"] = int(adjacency.Extent())
    # Near-endpoint samples see more of the boundary while avoiding coordinate
    # singularities at poles and patch vertices, which need a separate audit.
    fractions = np.linspace(1e-6, 1.0 - 1e-6, int(samples_per_edge))
    attempted = 0
    failure_reasons: set[str] = set()

    def maximum(key: str, value: float) -> None:
        old = report[key]
        report[key] = float(value if old is None else max(old, value))

    for index in range(1, int(adjacency.Extent()) + 1):
        edge = TopoDS.Edge_s(adjacency.FindKey(index))
        faces = []
        for ancestor in adjacency.FindFromIndex(index):
            if not any(ancestor.IsSame(existing) for existing in faces):
                faces.append(TopoDS.Face_s(ancestor))
        if BRep_Tool.Degenerated_s(edge):
            report["degenerate_edges"] += 1
            report["unevaluated_edges"] += 1
            continue
        if len(faces) == 1 and BRep_Tool.IsClosed_s(edge, faces[0]):
            report["seam_edges"] += 1
            faces.append(faces[0])
            edges = (edge, TopoDS.Edge_s(edge.Reversed()))
        elif len(faces) == 2:
            edges = (edge, edge)
        else:
            report["boundary_edges" if len(faces) < 2 else "nonmanifold_edges"] += 1
            continue
        if attempted >= int(max_edges):
            report["unevaluated_edges"] += 1
            failure_reasons.add("The edge sampling budget was reached.")
            continue
        attempted += 1
        try:
            # Identical t must denote the same physical point on each face.
            # Proportional pcurve sampling without SameParameter can invent
            # either gaps or continuity for differently parameterized trims.
            if not BRep_Tool.SameParameter_s(edge) or not BRep_Tool.SameRange_s(edge):
                raise ValueError("An edge has no verified common pcurve parameterization.")
            curves = [BRepAdaptor_Curve2d(e, f) for e, f in zip(edges, faces)]
            surfaces = [BRepAdaptor_Surface(face) for face in faces]
            orientations = [
                -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
                for face in faces
            ]
            first, last = curves[0].FirstParameter(), curves[0].LastParameter()
            if not np.isfinite(first) or not np.isfinite(last) or last <= first:
                raise ValueError("An edge has an unusable pcurve range.")
            if not np.allclose(
                (first, last),
                (curves[1].FirstParameter(), curves[1].LastParameter()),
                rtol=1e-12, atol=1e-12,
            ):
                raise ValueError("Adjacent pcurves have different parameter ranges.")
            gaps, angles, relative_errors, absolute_errors = [], [], [], []
            for fraction in fractions:
                t = first + float(fraction) * (last - first)
                values = []
                for curve, surface, orientation in zip(curves, surfaces, orientations):
                    uv = curve.Value(t)
                    values.append(_surface_differential(surface, uv.X(), uv.Y(), orientation))
                (p1, n1, k1), (p2, n2, k2) = values
                gap = float(np.linalg.norm(p1 - p2))
                angle = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n1, n2)), np.dot(n1, n2))))
                gaps.append(gap)
                angles.append(angle)
                # At a crease there is no common tangent plane for G2.
                if gap <= position_tolerance and angle <= tangent_tolerance_deg:
                    error = float(np.linalg.norm(k1 - k2, ord=2))
                    scale = max(float(np.linalg.norm(k1, ord=2)), float(np.linalg.norm(k2, ord=2)))
                    absolute_errors.append(error)
                    relative_errors.append(error / max(scale, curvature_absolute_tolerance, 1e-30))
            report["evaluated_edges"] += 1
            report["sample_count"] += len(gaps)
            maximum("max_gap", max(gaps))
            maximum("max_normal_angle_deg", max(angles))
            if absolute_errors:
                maximum("max_curvature_absolute_error", max(absolute_errors))
                maximum("max_curvature_relative_error", max(relative_errors))
            g0 = max(gaps) <= position_tolerance
            g1 = g0 and max(angles) <= tangent_tolerance_deg
            g2 = g1 and all(
                absolute <= curvature_absolute_tolerance
                or relative <= curvature_relative_tolerance
                for absolute, relative in zip(absolute_errors, relative_errors)
            )
            report["g0_edges"] += int(g0)
            report["g1_edges"] += int(g1)
            report["g2_edges"] += int(g2)
            report["sharp_edges"] += int(g0 and not g1)
            report["curvature_discontinuous_edges"] += int(g1 and not g2)
        except Exception as exc:
            report["unevaluated_edges"] += 1
            failure_reasons.add(str(exc))

    incomplete = report["unevaluated_edges"] + report["nonmanifold_edges"]
    if incomplete:
        report["status"] = "partial"
    elif report["evaluated_edges"]:
        report["status"] = "complete"
    else:
        report["status"] = "no_adjacent_faces"
    if report["boundary_edges"]:
        report["limitations"].append("Open boundary edges have no adjacent face to compare.")
    if report["degenerate_edges"]:
        report["limitations"].append("Degenerate edges need separate singularity evaluation.")
    report["limitations"].extend(sorted(failure_reasons))
    return report


__all__ = ["assess_surface_quality"]
