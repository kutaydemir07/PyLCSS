# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""AMRTO common definitions, tool discovery, and mesh utilities.

Reference:
Hongyuan Ren, Bo Xia, Wenrui Wang, Xueqian Chen, Yang Liu, Pingzhang Zhou, Jianbin Du.
"AMRTO: Automated CAD model reconstruction of topology optimization result",
Computer Methods in Applied Mechanics and Engineering 435 (2025) 117673.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .cad_tessellation import tessellate_cad_absolute as tessellate_cad_absolute

logger = logging.getLogger(__name__)



# Search paths for AMRTO binaries and data
_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]


def resolve_tool(*, env_vars=(), candidates=()):
    """Find geometry tools without importing unrelated solver backends."""
    for value in [*(os.environ.get(name) for name in env_vars), *candidates]:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(str(value))
        if found:
            return found
    return None


def get_instant_meshes_executable() -> Optional[Path]:
    """Find the Instant-Meshes executable."""
    resolve_executable = resolve_tool

    candidates = [
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "Instant Meshes.exe",
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "Instant-Meshes and GMCG_revision" / "Instant Meshes.exe",
        Path.home() / "Desktop" / "topopt" / "AMRTO" / "CODE_AMRTO" / "Instant-Meshes and GMCG_revision" / "Instant Meshes.exe",
        Path.home() / "Desktop" / "topopt" / "Instant-Meshes and GMCG_revision" / "Instant Meshes.exe",
        Path.home() / "Desktop" / "topopt" / "Instant Meshes.exe",
        "Instant Meshes.exe",
        "Instant Meshes",
        "instant_meshes",
    ]
    res = resolve_executable(
        env_vars=("PYLCSS_AMRTO_INSTANT_MESHES",),
        candidates=candidates,
    )
    return Path(res) if res else None


def get_json2on_executable() -> Optional[Path]:
    """Find the json2on.exe executable for Rhino 3DM export."""
    resolve_executable = resolve_tool

    candidates = [
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "json2on.exe",
        Path.home() / "Desktop" / "topopt" / "AMRTO" / "CODE_AMRTO" / "json2on.exe",
        Path.home() / "Desktop" / "topopt" / "json2on.exe",
        "json2on.exe",
        "json2on",
    ]
    res = resolve_executable(
        env_vars=("PYLCSS_AMRTO_JSON2ON",),
        candidates=candidates,
    )
    return Path(res) if res else None


def get_gmcg_executable() -> Optional[Path]:
    """Find the GMCG_revision executable."""
    resolve_executable = resolve_tool

    candidates = [
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "gmcg_headless.exe",
        Path.home() / "Desktop" / "topopt" / "reconstruction_work" / "gmcg_headless" / "build" / "gmcg_headless.exe",
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "GMCG_revision.exe",
        _WORKSPACE_ROOT / "external_solvers" / "amrto" / "Instant-Meshes and GMCG_revision" / "GMCG_revision.exe",
        Path.home() / "Desktop" / "topopt" / "AMRTO" / "CODE_AMRTO" / "Instant-Meshes and GMCG_revision" / "GMCG_revision.exe",
        "GMCG_revision.exe",
        "GMCG_revision",
    ]
    res = resolve_executable(
        env_vars=("PYLCSS_AMRTO_GMCG",),
        candidates=candidates,
    )
    return Path(res) if res else None


def get_gmcg_headless_executable() -> Optional[Path]:
    """Find the noninteractive GMCG CLI without selecting the GUI override."""
    resolve_executable = resolve_tool

    executable = resolve_executable(
        env_vars=("PYLCSS_AMRTO_GMCG_HEADLESS",),
        candidates=[
            _WORKSPACE_ROOT / "external_solvers" / "amrto" / "gmcg_headless.exe",
            Path.home() / "Desktop" / "topopt" / "reconstruction_work" / "gmcg_headless" / "build" / "gmcg_headless.exe",
            "gmcg_headless.exe", "gmcg_headless",
        ],
    )
    if executable is not None:
        return Path(executable)
    legacy = get_gmcg_executable()
    return legacy if legacy is not None and "headless" in legacy.stem.lower() else None


@dataclass
class AMRTOConfig:
    """Configuration parameters for the 5-step AMRTO reconstruction pipeline."""

    # Step 1: Triangular mesh generation & smoothing
    density_cutoff: float = 0.5
    density_smooth_sigma: float = 0.5
    mesh_laplacian_smoothing_iters: int = 2
    mesh_laplacian_lambda: float = 0.3

    # Step 2: Quadrangulation
    target_quad_faces: int = 10000
    quad_crease_deg: float = 0.0  # Topopt isosurfaces are smooth continuum geometry without artificial facet creases
    quadrangulation_require_native: bool = True

    # Step 3: Sparse layouts division
    layout_target_patch_faces: int = 200  # Raised from 50: larger patches on smooth topology-opt surfaces
    export_layout_patches: bool = False  # Grouped stage 3 OBJ retains every patch
    # Feature charting every sharp input edge is a production extension, not
    # part of the paper's GMCG layouts. Optimization meshes are faceted, so
    # enabling it by default can multiply a benchmark-sized patch set severalfold.
    preserve_source_creases: bool = False
    source_crease_minimum_normal_dot: float = float(np.cos(np.deg2rad(30.0)))

    # Step 4: Parameterization and adaptive sampling
    sampling_rate_k: float = 1.5  # k in Eq. (14)
    sampling_min_d: int = 6       # d_min
    sampling_max_d: int = 2000    # d_max
    parameterization_method: str = "hybrid"
    # Optional harmonic distortion heuristic; native rectangles follow the paper.
    # Positivity and source-fit checks always run.
    parameterization_minimum_area_ratio: float = 0.0
    scaffold_max_iter: int = 50
    scaffold_relative_energy_tolerance: float = 1e-4
    scaffold_rectangle_tolerance: float = 1e-6
    scaffold_timeout_s: float = 60.0
    scaffold_require_native: bool = True
    harmonic_method: str = "iterative_geodesic"
    harmonic_max_iter: int = 10000
    harmonic_energy_threshold: float = 1e-10
    harmonic_residual_threshold: float = 1e-8
    harmonic_edge_counts: bool = False

    # Step 5: NURBS fitting and Algorithms 2/3 control-net processing.
    nurbs_degree_u: int = 3
    nurbs_degree_v: int = 3
    ctrlpts_ratio_kt: float = 0.5 # k_type_geomdl: ratio of control points vs sampling points
    ctrlpts_min: int = 6          # Minimum number of control points per direction
    multires_enabled: bool = True
    multires_curvature_ck: float = 0.1
    multires_curvature_cp: float = 10.0
    # K * reference length is unit-independent. None uses the source diagonal.
    multires_reference_length: Optional[float] = None
    multires_compression_ratio: float = 0.5  # Highest-layer total controls / original total
    multires_layers: int = 1
    multires_regeneration: str = "author_uniform"
    multires_maximum_knot_removal_attempts: int = 64
    preserve_patch_boundaries: bool = True  # Common boundary curves for sewing
    source_box_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    fitting_workers: int = 4  # Bound patch processes; each uses one BLAS thread
    # Repair is a bounded quality extension run once per patch. Keep its total
    # cost predictable on large layouts; unresolved patches are handled by the
    # existing subdivision and validation path.
    orientation_repair_seconds: float = 2.0
    orientation_repair_iterations: int = 2



@dataclass
class QuadLayout:
    """Represents a sparse quadrilateral layout isomorphic to a disk."""

    layout_id: int
    vertices: np.ndarray             # (N, 3) 3D coordinates
    triangles: np.ndarray            # (M, 3) triangular face indices
    quads: Optional[np.ndarray] = None  # (Q, 4) quad face indices if available
    corner_indices: list[int] = field(default_factory=list)  # [c0, c1, c2, c3]
    boundary_paths: list[list[int]] = field(default_factory=list) # 4 boundary vertex paths
    parameter_coords: Optional[np.ndarray] = None  # (N, 2) mapped to [0, 1]^2
    sampled_grid_3d: Optional[np.ndarray] = None   # (n_u * n_v, 3), u-major grid
    sampled_shape: Tuple[int, int] = (0, 0)        # (n_u, n_v)
    nurbs_surface: Any = None
    principal_curvature: float = 0.0
