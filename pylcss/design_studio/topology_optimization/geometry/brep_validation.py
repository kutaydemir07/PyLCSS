# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Acceptance gates shared by every topology-to-CAD reconstruction route.

A reconstruction is only useful if it still describes the structure the
optimizer analyzed, so each route is held to the same measurements: the
above-cutoff cell centres must survive, the enclosed volume must be preserved,
and the surface must stay within an admitted deviation. Keeping the gates in
one module stops the routes from drifting into differently-strict versions of
the same check.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# The containment band is a fraction of the admitted envelope. Every point the
# band misclassifies is by construction inside that band of the exact surface,
# so it is an order of magnitude below the deletion the gate is looking for.
_CONTAINMENT_BAND_FRACTION = 0.1


def protected_point_pitch(points: np.ndarray) -> float:
    """Infer the source cell pitch from a grid of protected cell centres."""
    try:
        pitches = []
        for axis in range(int(np.asarray(points).shape[1])):
            values = np.unique(np.round(np.asarray(points)[:, axis], 9))
            if len(values) < 2:
                continue
            steps = np.diff(values)
            steps = steps[steps > 1.0e-9]
            if steps.size:
                pitches.append(float(np.min(steps)))
        return float(min(pitches)) if pitches else 0.0
    except Exception:
        return 0.0


def distance_to_solids(point: np.ndarray, solids: list) -> float:
    """Shortest distance from a point to the nearest of several solids."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(*(float(value) for value in point))
    ).Vertex()
    best = float("inf")
    for solid in solids:
        measure = BRepExtrema_DistShapeShape(vertex, solid.wrapped)
        if measure.IsDone():
            best = min(best, float(measure.Value()))
    return best


def _contained_by_bounded_tessellation(
    candidate: Any,
    points: np.ndarray,
    *,
    band: float,
) -> Optional[np.ndarray]:
    """Classify points against a triangulation bounded to ``band`` of the B-rep.

    ``Shape.tessellate(d)`` keeps every point of the exact boundary within ``d``
    of the triangulation, so the two surfaces are within ``d`` of each other and
    the only points the triangulation can misclassify lie in that band.  With
    ``d`` a tenth of the admitted envelope, such a point is retained material
    under either surface, which is exactly what this gate is asked to decide.

    This replaces ``BRepClass3d_SolidClassifier`` as the containment test.  OCC's
    classifier does not cast a ray at the nearest boundary; it searches for a
    segment to a sampled point on some face, and on a body whose draw surface is
    a single large trimmed B-spline that search fails outright -- on the ribbed
    housing benchmark it reported cell centres 9 mm deep inside the solid as
    OUT, at every tolerance and from every jittered position, while the merged
    tessellation and the exact distance both put them well inside.  Returns
    ``None`` when the triangulation is not closed and the caller must fall back.
    """
    try:
        import trimesh

        vertices, triangles = candidate.tessellate(max(float(band), 1.0e-9))
        # Face triangulations are emitted independently, so the shared edges
        # have to be welded before the mesh is a closed volume.
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [[point.x, point.y, point.z] for point in vertices],
                dtype=float,
            ),
            faces=np.asarray(triangles, dtype=np.int64),
            process=True,
        )
        if not mesh.is_watertight:
            logger.debug("Bounded containment tessellation is not closed.")
            return None
        contained = np.asarray(mesh.contains(np.asarray(points, dtype=float)), dtype=bool)
    except Exception:
        logger.debug("Bounded containment tessellation failed.", exc_info=True)
        return None
    if contained.shape != (len(points),):
        return None
    return contained


def _contained_by_occ_classifier(
    candidate_solids: list,
    points: np.ndarray,
) -> np.ndarray:
    """Fall back to OCC's exact solid classifier, one query point at a time."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    classifiers = [
        BRepClass3d_SolidClassifier(solid.wrapped) for solid in candidate_solids
    ]
    contained = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        query = gp_Pnt(*(float(value) for value in point))
        for classifier in classifiers:
            classifier.Perform(query, 1.0e-7)
            if classifier.State() in (TopAbs_IN, TopAbs_ON):
                contained[index] = True
                break
    return contained


def protected_point_coverage(
    candidate: Any,
    protected_points: Optional[np.ndarray],
    *,
    fit_tolerance: float,
    description: str = "Reconstruction",
) -> dict[str, Any]:
    """Verify a candidate solid still contains every above-cutoff cell centre.

    Raises ``RuntimeError`` when reconstruction deleted material the optimizer
    kept, and otherwise returns the measured coverage for the report.
    """
    empty = {
        "protected_profile_points_inside": 0,
        "protected_profile_points_total": 0,
        "protected_profile_point_coverage": None,
        "protected_profile_point_worst_excursion": 0.0,
        "protected_profile_point_admitted_excursion": 0.0,
    }
    if protected_points is None:
        return empty
    points_to_check = np.asarray(protected_points, dtype=float)
    if points_to_check.ndim != 2 or points_to_check.shape[1] != 3:
        return empty

    protected_total = int(len(points_to_check))
    candidate_solids = list(candidate.Solids())
    # Half a source cell, not a fraction of the model diagonal. What is being
    # tested is whether reconstruction deleted material the optimizer kept, and
    # the unit of that material is the cell -- a centre less than half a cell
    # from the boundary still has most of its cell on the solid side. Tying it
    # to the diagonal made the gate scale with the wrong thing: on the 3.6 m
    # press crown it admitted 41 mm of deletion, while on the cantilever it
    # rejected a point 0.33 cells out because the diagonal-based figure had no
    # relation to the cell at all.
    #
    # The distance below is unsigned, so this envelope has to stay tight: a
    # point just inside a hole the profile opened is the same measurement as
    # one just outside the outer wall, and only the first is real material loss.
    # The two terms are independent error sources, so they add rather than
    # compete. A protected point is a cell-centre sample, so the boundary it
    # describes is only located to within half a pitch; the profile is then
    # fitted to that boundary to within `fit_tolerance`. Taking the max granted
    # the sampling term and spent nothing on the fitting, which put the gate
    # inside its own tolerance chain: on the cantilever benchmark it admitted
    # 0.50 cells while RDP simplification and curvature fairing together are
    # allowed 0.16 cells more than that, and reconstructions failed by 0.02 mm
    # on bodies whose volume error was 0.6% and whose sampled deviation was 40%
    # of its own allowance.
    cell_pitch = protected_point_pitch(points_to_check)
    inside_tolerance = max(
        0.5 * float(cell_pitch) + float(fit_tolerance), 1.0e-7
    )

    band = max(_CONTAINMENT_BAND_FRACTION * inside_tolerance, 1.0e-9)
    contained = _contained_by_bounded_tessellation(
        candidate,
        points_to_check,
        band=band,
    )
    containment_method = "bounded tessellation"
    if contained is None:
        contained = _contained_by_occ_classifier(candidate_solids, points_to_check)
        containment_method = "OCC solid classifier"
        band = 0.0

    # The envelope is applied by measuring, never by relaxing a classifier
    # tolerance. BRepClass3d_SolidClassifier's tolerance is compared against the
    # parameter of the segment it casts to a sampled point on a face, not
    # against the distance to the boundary, so it is not a geometric dilation:
    # on the cantilever benchmark a centre 0.2719 from the wall still
    # classified OUT at a tolerance of 2.0. Passing the envelope in there
    # produced the self-contradicting failure "worst point lies 0.2719 outside;
    # admitted envelope is 0.4167". Whatever comes back outside is measured
    # against the exact B-rep with BRepExtrema, which is the same unsigned
    # distance the diagnostic reports.
    outside = np.flatnonzero(~contained)
    measured = [
        float(distance_to_solids(points_to_check[index], candidate_solids))
        for index in outside
    ]
    deleted = [
        distance
        for distance in measured
        if not np.isfinite(distance) or distance > inside_tolerance
    ]
    protected_inside = protected_total - len(deleted)
    worst_excursion = max(measured, default=0.0)
    if deleted:
        if cell_pitch > 0.0:
            detail = (
                f" Worst point lies {worst_excursion:.4g} outside"
                f" ({worst_excursion / cell_pitch:.2f} cells);"
                f" admitted envelope is {inside_tolerance:.4g}"
                f" ({inside_tolerance / cell_pitch:.2f} cells)."
            )
        else:
            detail = (
                f" Worst point lies {worst_excursion:.4g} outside;"
                f" admitted envelope is {inside_tolerance:.4g}."
            )
        raise RuntimeError(
            f"{description} excluded {len(deleted)} of {protected_total} "
            "above-cutoff profile points." + detail
        )

    return {
        "protected_profile_points_inside": int(protected_inside),
        "protected_profile_points_total": int(protected_total),
        "protected_profile_point_coverage": (
            float(protected_inside) / float(protected_total)
            if protected_total
            else None
        ),
        "protected_profile_point_worst_excursion": float(worst_excursion),
        "protected_profile_point_admitted_excursion": float(inside_tolerance),
        "protected_profile_point_containment": containment_method,
        "protected_profile_point_containment_band": float(band),
    }


__all__ = [
    "distance_to_solids",
    "protected_point_coverage",
    "protected_point_pitch",
]
