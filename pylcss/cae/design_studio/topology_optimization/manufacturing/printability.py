# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Process checks on the explicit manufactured structure.

The optimizer's manufacturing constraints act on the macro density field: the
overhang filter, the pull-out direction and the member-size bounds all shape
the envelope. The explicit cells are built afterwards, from the family's own
geometry, and are not covered by any of them. These checks close that gap by
measuring what was actually built — its wall or strut thickness in model units,
and the inclination of its members against the build direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lattice_families import strut_edges

__all__ = [
    "SELF_SUPPORTING_ANGLE_DEG",
    "lattice_feature_size_report",
    "lattice_overhang_report",
    "overhang_advice",
]

#: Inclination from the build plate below which a feature is generally taken to
#: need support on a powder-bed process. Used only when a study states no angle
#: of its own.
SELF_SUPPORTING_ANGLE_DEG = 45.0

_BUILD_AXIS_VECTORS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def lattice_feature_size_report(
    options: Any,
    voxel_size: Any,
    minimum_feature_size: Any,
) -> dict[str, Any] | None:
    """Compare the built wall/strut and skin against a stated printable floor.

    Returns ``None`` when there is nothing to check — a solid result, an
    unresolved grid, or a study that states no process capability. Reporting a
    thickness in model units is not the same as checking it: without this the
    physical sizes were computed, published, and never compared to anything.
    """
    mode = getattr(options, "mode", "solid")
    if options is None or mode == "solid":
        return None
    try:
        edge = float(voxel_size or 0.0)
        floor = float(minimum_feature_size or 0.0)
    except (TypeError, ValueError):
        return None
    if edge <= 0.0 or floor <= 0.0:
        return None

    member = float(options.member_thickness_voxels) * edge
    skin = float(options.skin_thickness_voxels) * edge
    violations: list[str] = []
    if member < floor * (1.0 - 1e-9):
        violations.append(
            f"the {options.display_name.lower()} wall/member is {member:.3g} "
            f"against a {floor:.3g} minimum"
        )
    # A zero skin is a deliberate choice — it is what lets powder out — so only
    # a skin that exists and is too thin to build is a violation.
    if 0.0 < skin < floor * (1.0 - 1e-9):
        violations.append(
            f"the solid skin is {skin:.3g} against a {floor:.3g} minimum"
        )
    return {
        "minimum_feature_size": floor,
        "member_thickness": member,
        "skin_thickness": skin,
        "violations": tuple(violations),
    }


def lattice_overhang_report(
    options: Any,
    build_axis: Any,
    overhang_angle_deg: Any = SELF_SUPPORTING_ANGLE_DEG,
    *,
    voxel_scale: Any = None,
) -> dict[str, Any] | None:
    """Measure the cell's own members against the build direction.

    ``None`` when no build direction is set or the result is solid. A strut
    family is measured exactly from its unit-cell edge list; a minimal surface
    or a prismatic cell has a continuum of orientations and is reported as
    unchecked rather than guessed at.

    ``voxel_scale`` is the per-axis voxel edge length. A cell is a fixed number
    of voxels on every axis, so on a non-cubic grid the built cell is stretched
    and its members do not point where the unit cell says they do.
    """
    mode = getattr(options, "mode", "solid")
    if options is None or mode == "solid":
        return None
    axis_key = str(build_axis or "none").strip().lower()
    direction = _BUILD_AXIS_VECTORS.get(axis_key)
    if direction is None:
        return None
    try:
        threshold = float(overhang_angle_deg or SELF_SUPPORTING_ANGLE_DEG)
    except (TypeError, ValueError):
        threshold = SELF_SUPPORTING_ANGLE_DEG
    threshold = float(np.clip(threshold, 0.0, 90.0))

    family = options.family
    report: dict[str, Any] = {
        "build_axis": axis_key,
        "self_supporting_angle_deg": threshold,
        "cell_family": options.display_name,
    }
    if family is None or not family.is_strut:
        report["checked"] = False
        report["reason"] = (
            "A minimal-surface or prismatic cell presents a continuum of "
            "surface orientations, so its overhangs are not resolved from the "
            "cell definition alone."
        )
        return report

    try:
        edges = np.asarray(strut_edges(family.key), dtype=float)
    except (ValueError, KeyError):
        report["checked"] = False
        report["reason"] = "This family has no unit-cell edge list to measure."
        return report
    if edges.ndim != 3 or edges.shape[0] == 0:
        report["checked"] = False
        report["reason"] = "This family has no unit-cell edge list to measure."
        return report

    vectors = edges[:, 1, :] - edges[:, 0, :]
    scale = np.ones(3, dtype=float)
    if voxel_scale is not None:
        candidate = np.asarray(voxel_scale, dtype=float).reshape(-1)
        if candidate.size == 3 and np.all(np.isfinite(candidate)) and np.all(candidate > 0.0):
            scale = candidate
    vectors = vectors * scale
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-12
    vectors, lengths = vectors[keep], lengths[keep]
    if not len(vectors):
        report["checked"] = False
        report["reason"] = "This family has no unit-cell edge list to measure."
        return report

    # Inclination from the build plate: 90 degrees is a member along the build
    # direction, 0 degrees is one lying flat across it.
    cosine = np.abs(vectors @ np.asarray(direction, dtype=float)) / lengths
    inclination = np.degrees(np.arcsin(np.clip(cosine, 0.0, 1.0)))
    unsupported = inclination < threshold - 1e-9
    report.update(
        {
            "checked": True,
            "member_count": int(len(inclination)),
            "unsupported_member_count": int(np.count_nonzero(unsupported)),
            "minimum_inclination_deg": float(inclination.min()),
            "horizontal_member_count": int(
                np.count_nonzero(inclination < 1e-6)
            ),
        }
    )
    return report


def overhang_advice(report: dict[str, Any] | None) -> str | None:
    """Render an overhang report as one engineering sentence, or ``None``."""
    if not report:
        return None
    family = str(report.get("cell_family") or "This lattice")
    axis = str(report.get("build_axis") or "").upper()
    if not report.get("checked"):
        return (
            f"{family} was not overhang-checked against the {axis} build "
            f"direction. {report.get('reason', '')} Verify the manufactured "
            "geometry against the process before release."
        ).strip()
    unsupported = int(report.get("unsupported_member_count") or 0)
    if not unsupported:
        return None
    total = int(report.get("member_count") or 0)
    threshold = float(report.get("self_supporting_angle_deg") or 0.0)
    minimum = float(report.get("minimum_inclination_deg") or 0.0)
    horizontal = int(report.get("horizontal_member_count") or 0)
    horizontal_text = (
        f" {horizontal} of them lie flat across it."
        if horizontal
        else ""
    )
    return (
        f"{unsupported} of the {total} members in a {family.lower()} cell are "
        f"inclined below {threshold:g} degrees to the {axis} build direction "
        f"(shallowest {minimum:.1f} degrees).{horizontal_text} The overhang "
        "constraint shapes the optimized envelope but not the cells inside it, "
        "so these members are unsupported as built. Choose a cell with no "
        "shallow members, change the build direction, or accept them as "
        "self-bridging at this diameter."
    )
