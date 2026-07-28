"""Pure helpers for consistent engineering-condition visualization.

The VTK viewer intentionally keeps these calculations free of VTK so scaling,
sampling, labels, and palettes can be tested without an OpenGL context.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


BC_PALETTE = {
    "selection": (0.72, 0.76, 0.84),
    "support": (0.00, 0.84, 1.00),
    "prescribed": (1.00, 0.55, 0.00),
    "force": (1.00, 0.78, 0.20),
    "pressure": (1.00, 0.24, 0.73),
    "gravity": (0.72, 0.46, 1.00),
    "velocity": (0.00, 0.90, 1.00),
    "impact": (1.00, 0.46, 0.12),
    "joint": (1.00, 0.62, 0.11),
    "keep_material": (0.41, 0.94, 0.68),
    "keep_void": (1.00, 0.32, 0.32),
    "thermal_sink": (0.25, 0.77, 1.00),
    "heat": (1.00, 0.36, 0.12),
}


def bounds_diagonal(bounds: Sequence[float] | None, fallback: float = 1.0) -> float:
    """Return a finite positive diagonal for a VTK-style six-value bounds tuple."""
    try:
        values = np.asarray(bounds, dtype=float).reshape(-1)
        if values.size != 6 or not np.all(np.isfinite(values)):
            raise ValueError
        diagonal = float(
            np.linalg.norm(
                [
                    values[1] - values[0],
                    values[3] - values[2],
                    values[5] - values[4],
                ]
            )
        )
        if diagonal <= 1e-12:
            raise ValueError
        return diagonal
    except (TypeError, ValueError):
        return max(float(fallback), 1e-6)


def overlay_scale(bounds: Sequence[float] | None) -> float:
    """World-space reference size tied only to the displayed engineering model.

    There is deliberately no absolute-unit floor: a 2 mm part and a 20 m
    structure must receive proportionally equivalent glyphs.
    """
    return max(bounds_diagonal(bounds) * 0.035, 1e-6)


def dof_caption(fixed_dofs: Iterable[int] | None) -> str:
    """Return a compact, explicit translational-DOF caption."""
    names = ("UX", "UY", "UZ")
    indices = sorted(
        {
            int(index)
            for index in (fixed_dofs or ())
            if isinstance(index, (int, np.integer)) and 0 <= int(index) < 3
        }
    )
    return " ".join(names[index] for index in indices) or "NO TRANSLATIONAL DOF"


def constraint_color(metadata: dict | None = None) -> tuple[float, float, float]:
    """Return the viewer-owned color for a constraint's engineering meaning.

    Backend/debug payloads historically carried their own colors, so the same
    fixed support could change from blue to cyan when a result replaced the CAD
    preview. The renderer is now the sole color authority.
    """
    metadata = metadata or {}
    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("visual_kind", "constraint_type", "legend_label", "region_type")
    ).lower()
    if "joint" in text:
        return BC_PALETTE["joint"]
    if "keep void" in text or "non-design void" in text:
        return BC_PALETTE["keep_void"]
    if "keep material" in text or "non-design material" in text:
        return BC_PALETTE["keep_material"]
    if "thermal" in text or "sink" in text:
        return BC_PALETTE["thermal_sink"]
    if "displacement" in text or "prescribed" in text:
        return BC_PALETTE["prescribed"]
    return BC_PALETTE["support"]


def compact_constraint_label(metadata: dict | None = None) -> str:
    """Build a short 3D annotation; detailed values remain in the inspector."""
    metadata = metadata or {}
    constraint_type = str(metadata.get("constraint_type") or "Support").strip()
    normalized = constraint_type.lower()
    fixed_dofs = list(metadata.get("fixed_dofs") or [])
    displacement = metadata.get("displacement")

    if displacement is not None or "displacement" in normalized:
        components = []
        for index, name in enumerate(("UX", "UY", "UZ")):
            if index not in fixed_dofs:
                continue
            try:
                value = float(displacement[index])
            except (TypeError, ValueError, IndexError):
                value = 0.0
            components.append(f"{name} {value:.3g} mm")
        if len(components) == 1:
            return components[0]
        return "PRESCRIBED U"

    if normalized == "fixed":
        return "FIXED"
    if normalized.startswith("pinned"):
        return "PINNED"
    if normalized.startswith("roller "):
        axis = normalized.rsplit(" ", 1)[-1].upper()
        return f"ROLLER {axis}"
    if normalized.startswith("block ") and fixed_dofs:
        return f"{dof_caption(fixed_dofs)} = 0"
    if normalized.startswith("symmetry "):
        axis = normalized.rsplit(" ", 1)[-1].upper()
        return f"SYM {axis}"
    if "joint" in normalized:
        return normalized.replace(" joint", "").upper()
    if "non-design material" in normalized:
        return "KEEP"
    if "non-design void" in normalized:
        return "VOID"
    if "thermal sink" in normalized:
        return "SINK"
    return constraint_type.upper()[:24]


def farthest_spatial_samples(
    points: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
    max_points: int = 6,
    min_spacing: float = 0.0,
) -> list[np.ndarray]:
    """Select deterministic, well-spaced samples from an uneven tessellation.

    Starting with the largest weighted candidate and then maximizing distance
    avoids the clustering produced by simply taking the largest triangles.
    """
    candidates = np.asarray(points, dtype=float)
    if (
        candidates.ndim != 2
        or candidates.shape[1] < 3
        or len(candidates) == 0
        or max_points <= 0
    ):
        return []
    candidates = candidates[:, :3]
    finite = np.all(np.isfinite(candidates), axis=1)
    candidates = candidates[finite]
    if len(candidates) == 0:
        return []

    if weights is None:
        candidate_weights = np.ones(len(candidates), dtype=float)
    else:
        raw_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(raw_weights) != len(finite):
            candidate_weights = np.ones(len(candidates), dtype=float)
        else:
            candidate_weights = raw_weights[finite]
            candidate_weights = np.where(
                np.isfinite(candidate_weights) & (candidate_weights > 0.0),
                candidate_weights,
                0.0,
            )
            if not np.any(candidate_weights):
                candidate_weights = np.ones(len(candidates), dtype=float)

    first = int(np.argmax(candidate_weights))
    selected_indices = [first]
    remaining = np.ones(len(candidates), dtype=bool)
    remaining[first] = False
    spacing = max(float(min_spacing), 0.0)
    normalized_weight = candidate_weights / max(float(np.max(candidate_weights)), 1e-12)

    while len(selected_indices) < min(int(max_points), len(candidates)):
        available = np.flatnonzero(remaining)
        if len(available) == 0:
            break
        selected = candidates[selected_indices]
        distances = np.min(
            np.linalg.norm(
                candidates[available, None, :] - selected[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        eligible = distances >= spacing
        if not np.any(eligible):
            break
        available = available[eligible]
        distances = distances[eligible]
        # Geometry dominates; area only resolves similarly spaced candidates.
        score = distances * (0.85 + 0.15 * normalized_weight[available])
        chosen = int(available[int(np.argmax(score))])
        selected_indices.append(chosen)
        remaining[chosen] = False

    return [candidates[index].copy() for index in selected_indices]


def direction_from_label(label: str | None) -> np.ndarray:
    """Resolve the six global-axis direction labels used by body forces."""
    mapping = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0),
        "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    return np.asarray(mapping.get(str(label or "").strip().upper(), (0.0, -1.0, 0.0)))
