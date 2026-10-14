# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Geometric voxel resolution, cubic-grid plans, and resource estimates.

Preset resolutions are PyLCSS policy, not standardized accuracy guarantees.
The optional global continuation follows the early, bounded mesh updates of
Wang, de Sturler and Paulino (2010), https://arxiv.org/abs/1009.4975.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, prod
from typing import Sequence

import numpy as np

# (cells across the longest CAD extent, cells across a selected small feature).
# These determine the geometric target for one cubic pitch.
VOXEL_MESH_RESOLUTIONS = {
    "Coarse": (24, 3),
    "Medium": (40, 4),
    "Fine": (64, 6),
    "Very Fine": (96, 8),
}
DEFAULT_VOXEL_MESH_QUALITY = "Fine"
DEFAULT_MAX_VOXEL_ELEMENTS = 1_000_000
REFINEMENT_MIN_ITERATIONS = 5
REFINEMENT_MAX_ITERATIONS = 10
REFINEMENT_STABILITY_TOLERANCE = 0.01
INITIAL_REFINEMENT_ELEMENTS = 5_000


def available_voxel_memory_bytes() -> int | None:
    """Read available RAM without requiring an optional monitoring package."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        import os
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length',ctypes.c_ulong), ('load',ctypes.c_ulong),
                            *[(name,ctypes.c_ulonglong) for name in
                              ('total_physical','available_physical','total_page','available_page',
                               'total_virtual','available_virtual','extended')]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        elif hasattr(os,'sysconf'):
            try:
                return int(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
            except (ValueError,OSError):
                pass
    return None


def estimated_voxel_memory_bytes(shape: Sequence[int]) -> int:
    """Conservative planning allowance, not a measured allocation guarantee."""
    # Assembly keeps several sparsity maps for the robust SIMP fields and
    # creates large temporary maps. Include that peak, rather than only the
    # final sparse matrix and solution vectors.
    return 65_536 * prod(shape) + 512 * prod(count + 1 for count in shape)


def voxel_mesh_quality(value: object) -> str:
    """Normalize saved/API choices; reject unrecognized mesh requests."""
    normalized = str(value or DEFAULT_VOXEL_MESH_QUALITY).strip().lower()
    normalized = " ".join(normalized.replace("_", " ").replace("-", " ").split())
    for choice in VOXEL_MESH_RESOLUTIONS:
        if normalized == choice.lower():
            return choice
    raise ValueError("Voxel mesh quality must be Coarse, Medium, Fine, or Very Fine.")


def voxel_element_limit(value: object) -> int:
    """Return the separate configurable safeguard; zero disables it."""
    number = float(DEFAULT_MAX_VOXEL_ELEMENTS if value is None else value)
    if not isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Maximum voxel elements must be a non-negative integer (0 is unlimited).")
    return int(number)


def voxel_grid_from_size(span: Sequence[float], size: float) -> tuple[int, int, int]:
    """Round grid counts up; the caller pads bounds to keep the exact pitch."""
    if not isfinite(size) or size <= 0:
        raise ValueError("Voxel size must be finite and positive.")
    counts = []
    for extent in span:
        ratio = float(extent) / size
        if not isfinite(ratio) or ratio > np.iinfo(np.int32).max:
            raise ValueError("Voxel size is too small for the supported grid index range.")
        if ratio <= 0:
            raise ValueError("Voxel meshing needs three finite, positive domain extents.")
        count = max(2, ceil(ratio - 1e-12 * max(1.0, ratio)))
        alignment = 4 if count >= 4 else 2
        counts.append(alignment * ceil(count / alignment))
    if len(counts) != 3:
        raise ValueError("Voxel meshing needs three domain extents.")
    return tuple(counts)


def geometric_voxel_size(
    span: Sequence[float], quality: object,
    *, feature_bboxes: Sequence[Sequence[float]] = (), feature_lengths: Sequence[float] = (),
) -> float:
    """Resolve a geometry-based scalar pitch without a resource-budget clamp."""
    extents = np.asarray(span, dtype=float)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError("Voxel meshing needs three finite, positive domain extents.")
    longest_cells, feature_cells = VOXEL_MESH_RESOLUTIONS[voxel_mesh_quality(quality)]
    pitch = min(float(np.max(extents)) / longest_cells, float(np.min(extents)) / 2)
    lengths = list(feature_lengths)
    boundary_tolerance = max(1e-6, float(np.max(extents)) * 1e-8)
    for bbox in feature_bboxes:
        # CAD face boxes carry geometric tolerance in their nominally zero
        # normal direction. It is not a physical thin feature to resolve.
        lengths.extend(length for i in (0, 2, 4)
                       if (length := abs(float(bbox[i + 1]) - float(bbox[i]))) > boundary_tolerance)
    positive = [float(length) for length in lengths if isfinite(float(length)) and float(length) > 1e-9]
    if positive:
        pitch = min(pitch, min(positive) / feature_cells)
    return pitch


@dataclass(frozen=True)
class VoxelMeshPlan:
    shape: tuple[int, int, int]
    bounds: tuple[np.ndarray, np.ndarray]
    source_bounds: tuple[np.ndarray, np.ndarray]
    pitch: float | None
    quality: str
    mode: str
    element_limit: int
    geometric_target_pitch: float | None = None
    automatic_element_budget: int | None = None
    resource_limited: bool = False

    @property
    def elements(self) -> int:
        return prod(self.shape)

    @property
    def voxel_edges(self) -> list[float]:
        return ((self.bounds[1] - self.bounds[0]) / np.asarray(self.shape)).tolist()

    @property
    def estimated_memory_bytes(self) -> int:
        # Planning allowance for assembled 3-D stiffness/filter operators,
        # assembly buffers, solver vectors and sensitivities. Not a measured
        # allocation or a guarantee for sparse direct-factorization fill-in.
        return estimated_voxel_memory_bytes(self.shape)

    def resource_error(self) -> str | None:
        if self.element_limit and self.elements > self.element_limit:
            return (
                f"Requested voxel mesh needs {self.elements:,} elements; the configured "
                f"maximum is {self.element_limit:,}. Increase Maximum Elements, choose "
                "a coarser preset, or increase voxel size. The requested resolution was preserved."
            )
        return None


def voxel_mesh_plan(
    bounds: tuple[Sequence[float], Sequence[float]], quality: object = None,
    *, voxel_size: float = 0, automatic: bool = True, manual_shape: Sequence[int] = (30, 20, 10),
    max_elements: object = None, feature_bboxes: Sequence[Sequence[float]] = (),
    feature_lengths: Sequence[float] = (),
    available_memory_bytes: int | None = None,
) -> VoxelMeshPlan:
    """Plan counts, padding and resources without allocating or voxelizing."""
    lower, upper = (np.asarray(value, dtype=float)[:3].copy() for value in bounds)
    span = upper - lower
    quality = voxel_mesh_quality(quality)
    limit = voxel_element_limit(max_elements)
    size = float(voxel_size)
    if not isfinite(size) or size < 0:
        raise ValueError("Voxel size must be finite and non-negative (0 uses the preset).")
    if not np.all(np.isfinite(span)) or np.any(span <= 0) or span.shape != (3,):
        raise ValueError("Voxel meshing needs three finite, positive domain extents.")
    if size > 0 or automatic:
        pitch = size if size > 0 else geometric_voxel_size(
            span, quality, feature_bboxes=feature_bboxes, feature_lengths=feature_lengths,
        )
        shape = voxel_grid_from_size(span, pitch)
        target_pitch = pitch
        automatic_budget = None
        if size == 0 and automatic:
            budgets = [limit] if limit else []
            if available_memory_bytes is not None:
                # Keep half the available RAM for CAD, the desktop and
                # differences between the estimate and the actual backend.
                budgets.append(max(8, int(available_memory_bytes) // (2 * 66_560)))
            automatic_budget = min(budgets) if budgets else None
            if automatic_budget is not None and prod(shape) > automatic_budget:
                if automatic_budget < 8:
                    raise ValueError('Automatic voxel meshing needs a budget of at least 8 elements.')
                lower_pitch = pitch
                upper_pitch = max(pitch, (float(np.prod(span))/automatic_budget)**(1/3))
                while prod(voxel_grid_from_size(span, upper_pitch)) > automatic_budget:
                    upper_pitch *= 1.2
                for _ in range(60):
                    candidate = (lower_pitch + upper_pitch) / 2
                    if prod(voxel_grid_from_size(span, candidate)) > automatic_budget:
                        lower_pitch = candidate
                    else:
                        upper_pitch = candidate
                pitch = upper_pitch
                shape = voxel_grid_from_size(span, pitch)
                pitch = float(np.nextafter(max(pitch, float(np.max(span / np.asarray(shape)))), np.inf))
        # Anchor at the CAD minimum; only extend the far sides. Source
        # geometry defines solids, and every cell has the exact same pitch.
        grid_bounds = lower, lower + np.asarray(shape) * pitch
        mode = "Voxel Size" if size > 0 else "Preset"
    else:
        shape = tuple(int(count) for count in manual_shape)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("Topology grid dimensions must be positive integers.")
        grid_bounds, pitch, mode = (lower, upper), None, "Element Counts"
        target_pitch, automatic_budget = None, None
    return VoxelMeshPlan(shape, grid_bounds, (lower, upper), pitch, quality, mode, limit,
                         target_pitch, automatic_budget,
                         bool(pitch is not None and target_pitch is not None and pitch > target_pitch * (1+1e-9)))


def progressive_voxel_shapes(target: Sequence[int], *, isotropic: bool = False) -> list[tuple[int, int, int]]:
    """Return short warm starts and the exact target; cubic grids stay cubic."""
    final = tuple(int(count) for count in target)
    if len(final) != 3 or min(final) < 1:
        raise ValueError("Voxel grid dimensions must be three positive integers.")
    levels = [final]
    while prod(levels[0]) > INITIAL_REFINEMENT_ELEMENTS and len(levels) < 4:
        current = levels[0]
        if isotropic:
            # Exact common halving over fixed bounds preserves cubic pitch.
            # Stop before a thin/odd axis would become anisotropic.
            if any(count < 2 or count % 2 for count in current):
                break
            coarse = tuple(count // 2 for count in current)
        else:
            coarse = tuple(1 if count == 1 else min(count, max(2, 2 * round(count / 4))) for count in current)
        if coarse == current:
            break
        levels.insert(0, coarse)
    return levels
