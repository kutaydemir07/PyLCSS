# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Element-budget presets, cubic-grid plans, and resource estimates.

Preset budgets are PyLCSS policy, not standardized accuracy guarantees.
The optional global continuation borrows the early, bounded update timing from
Wang, de Sturler and Paulino (2010), https://arxiv.org/abs/1009.4975. It does
not implement that paper's local adaptive refinement or void derefinement.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import ceil, isfinite, prod
import os
from typing import Sequence

import numpy as np

# Whole bounding-grid element budgets; every voxel has the same cubic pitch.
VOXEL_MESH_ELEMENT_BUDGETS = {
    "Coarse": 10_000,
    "Medium": 40_000,
    "Fine": 160_000,
    "Very Fine": 640_000,
}
DEFAULT_VOXEL_MESH_QUALITY = "Fine"
DEFAULT_MAX_VOXEL_ELEMENTS = 1_000_000
# With at least two cells per axis, nodes/elements is at most (3/2)**3.
# Bound the existing memory estimate even for thin, highly padded grids.
AUTOMATIC_VOXEL_BYTES_PER_ELEMENT = 65_536 + 512 * 27 // 8
# Small robust-SIMP RSS probes include about 100 MiB of one-time runtime/import
# overhead beyond the element estimate. Reserve 128 MiB for cold startup.
# This remains a planning allowance, not a factorization-memory guarantee.
VOXEL_SOLVER_STARTUP_BYTES = 128 * 1024**2
REFINEMENT_MIN_ITERATIONS = 5
REFINEMENT_MAX_ITERATIONS = 10
REFINEMENT_STABILITY_TOLERANCE = 0.01
INITIAL_REFINEMENT_ELEMENTS = 5_000


if os.name == "nt":
    class _WindowsMemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            *(
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_physical",
                    "available_physical",
                    "total_page",
                    "available_page",
                    "total_virtual",
                    "available_virtual",
                    "extended",
                )
            ),
        ]


def available_voxel_memory_bytes() -> int | None:
    """Read available RAM without requiring an optional monitoring package."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        if os.name == "nt":
            try:
                status = _WindowsMemoryStatus()
                status.length = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.available_physical)
            except Exception:
                pass
        elif hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                if pages > 0 and page_size > 0:
                    return int(pages * page_size)
            except (ValueError, OSError):
                pass
    return None


def estimated_voxel_memory_bytes(shape: Sequence[int]) -> int:
    """Conservative planning allowance, not a measured allocation guarantee."""
    # Assembly keeps several sparsity maps for the robust SIMP fields and
    # creates large temporary maps. Include that peak, rather than only the
    # final sparse matrix and solution vectors.
    return VOXEL_SOLVER_STARTUP_BYTES + 65_536 * prod(shape) + 512 * prod(count + 1 for count in shape)


def voxel_mesh_quality(value: object) -> str:
    """Normalize saved/API choices; reject unrecognized mesh requests."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_VOXEL_MESH_QUALITY
    if not isinstance(value, str):
        raise ValueError("Voxel mesh quality must be Coarse, Medium, Fine, or Very Fine.")
    normalized = " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
    for choice in VOXEL_MESH_ELEMENT_BUDGETS:
        if normalized == choice.lower():
            return choice
    raise ValueError("Voxel mesh quality must be Coarse, Medium, Fine, or Very Fine.")


def voxel_element_limit(value: object) -> int:
    """Return the separate configurable safeguard; zero disables it."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Maximum voxel elements must be a non-negative integer (0 is unlimited).")
    try:
        number = float(DEFAULT_MAX_VOXEL_ELEMENTS if value is None else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Maximum voxel elements must be a non-negative integer (0 is unlimited).") from exc
    if not isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Maximum voxel elements must be a non-negative integer (0 is unlimited).")
    return int(number)


def _positive_grid_shape(shape: Sequence[int], error_message: str) -> tuple[int, int, int]:
    try:
        if isinstance(shape, (str, bytes)):
            raise ValueError
        items = tuple(shape)
    except TypeError as exc:
        raise ValueError(error_message) from exc
    if len(items) != 3:
        raise ValueError(error_message)
    counts: list[int] = []
    for raw in items:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(error_message)
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error_message) from exc
        if not isfinite(number) or number < 1 or not number.is_integer():
            raise ValueError(error_message)
        counts.append(int(number))
    return counts[0], counts[1], counts[2]


def voxel_grid_from_size(span: Sequence[float], size: float) -> tuple[int, int, int]:
    """Round grid counts up; the caller pads bounds to keep the exact pitch."""
    if isinstance(size, (bool, np.bool_)) or not isfinite(size) or size <= 0:
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


def budget_voxel_size(span: Sequence[float], budget: int) -> float:
    """Find the smallest cubic pitch whose rounded grid fits the budget."""
    if isinstance(budget, (bool, np.bool_)) or budget < 8:
        raise ValueError("Automatic voxel meshing needs a budget of at least 8 elements.")
    longest = float(max(span))
    # The two shorter axes need at least two cells each. These bounds cover
    # both nearly flat CAD and compact bodies without a volume-based guess.
    lower_pitch = longest / (budget // 4)
    upper_pitch = longest / 2
    for _ in range(60):
        candidate = (lower_pitch + upper_pitch) / 2
        if prod(voxel_grid_from_size(span, candidate)) > budget:
            lower_pitch = candidate
        else:
            upper_pitch = candidate
    shape = voxel_grid_from_size(span, upper_pitch)
    # Undo rounding tolerance at the boundary only when IEEE division rounded
    # below the domain extent, so exact pitches remain exact.
    pitch = max(float(extent) / count for extent, count in zip(span, shape))
    if any(count * pitch < float(extent) for extent, count in zip(span, shape)):
        pitch = float(np.nextafter(pitch, np.inf))
    return float(pitch)


@dataclass(frozen=True)
class VoxelMeshPlan:
    shape: tuple[int, int, int]
    bounds: tuple[np.ndarray, np.ndarray]
    source_bounds: tuple[np.ndarray, np.ndarray]
    pitch: float | None
    quality: str
    mode: str
    element_limit: int
    preset_element_budget: int | None = None
    automatic_element_budget: int | None = None
    resource_limited: bool = False
    memory_fraction: float = 0.5

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
    max_elements: object = None,
    available_memory_bytes: int | None = None,
    memory_fraction: float | None = 0.5,
) -> VoxelMeshPlan:
    """Plan counts, padding and resources without allocating or voxelizing."""
    lower, upper = (np.asarray(value, dtype=float)[:3].copy() for value in bounds)
    span = upper - lower
    quality = voxel_mesh_quality(quality)
    limit = voxel_element_limit(max_elements)
    if isinstance(memory_fraction, (bool, np.bool_)):
        raise ValueError("Available RAM allocation must be greater than 0 and at most 1.")
    fraction = float(0.5 if memory_fraction is None else memory_fraction)
    if not isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Available RAM allocation must be greater than 0 and at most 1.")
    if isinstance(voxel_size, (bool, np.bool_)):
        raise ValueError("Voxel size must be finite and non-negative (0 uses the preset).")
    size = float(voxel_size)
    if not isfinite(size) or size < 0:
        raise ValueError("Voxel size must be finite and non-negative (0 uses the preset).")
    if not np.all(np.isfinite(span)) or np.any(span <= 0) or span.shape != (3,):
        raise ValueError("Voxel meshing needs three finite, positive domain extents.")
    preset_budget = automatic_budget = None
    if size > 0 or automatic:
        if size > 0:
            pitch = size
        else:
            preset_budget = VOXEL_MESH_ELEMENT_BUDGETS[quality]
            budgets = [preset_budget]
            if limit:
                budgets.append(limit)
            if available_memory_bytes is not None:
                # Default to half the available RAM; the user can change the
                # allowance without weakening the configurable element cap.
                allocation = int(int(available_memory_bytes) * fraction) - VOXEL_SOLVER_STARTUP_BYTES
                budgets.append(max(8, max(0, allocation) // AUTOMATIC_VOXEL_BYTES_PER_ELEMENT))
            automatic_budget = min(budgets)
            pitch = budget_voxel_size(span, max(8, automatic_budget))
        shape = voxel_grid_from_size(span, pitch)
        # Anchor at the CAD minimum; only extend the far sides. Source
        # geometry defines solids, and every cell has the exact same pitch.
        grid_bounds = lower, lower + np.asarray(shape) * pitch
        mode = "Voxel Size" if size > 0 else "Preset"
    else:
        shape = _positive_grid_shape(
            manual_shape,
            "Topology grid dimensions must be positive integers.",
        )
        grid_bounds, pitch, mode = (lower, upper), None, "Element Counts"
    return VoxelMeshPlan(shape, grid_bounds, (lower, upper), pitch, quality, mode, limit,
                         preset_budget, automatic_budget,
                         bool(preset_budget is not None and automatic_budget < preset_budget), fraction)


def progressive_voxel_shapes(target: Sequence[int], *, isotropic: bool = False) -> list[tuple[int, int, int]]:
    """Return short warm starts and the exact target; cubic grids stay cubic."""
    final = _positive_grid_shape(
        target,
        "Voxel grid dimensions must be three positive integers.",
    )
    levels = [final]
    while prod(levels[0]) > INITIAL_REFINEMENT_ELEMENTS and len(levels) < 4:
        current = levels[0]
        if isotropic:
            # Exact common halving over fixed bounds preserves cubic pitch and
            # keeps coarse dimensions even so geometric multigrid stays active.
            if any(count < 4 or count % 4 for count in current):
                break
            coarse = tuple(count // 2 for count in current)
        else:
            coarse = tuple(1 if count == 1 else min(count, max(2, 2 * round(count / 4))) for count in current)
        if coarse == current:
            break
        levels.insert(0, coarse)
    return levels
