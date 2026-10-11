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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def tessellate_cad_absolute(shape: Any, tolerance: float, angular_tolerance: float = 0.5,
                            *, return_face_ranges: bool = False):
    """Mesh a CAD shape with absolute model-unit deflection, preserving seams.

    CadQuery's Shape.tessellate uses relative OCC deflection. Extract the OCC
    triangulation directly so it cannot replace this absolute mesh implicitly.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    shape = shape.val() if hasattr(shape, "val") else shape
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, float(tolerance), False,
                                    float(angular_tolerance), True)
    if not mesher.IsDone():
        raise RuntimeError("Absolute CAD tessellation failed")
    vertices, triangles, face_ranges = [], [], []
    for face in shape.Faces():
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if triangulation is None:
            BRepMesh_IncrementalMesh(face.wrapped, float(tolerance), False,
                                     float(angular_tolerance), True)
            triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
        if triangulation is None:
            face_ranges.append((len(vertices), len(vertices), len(triangles), len(triangles)))
            continue
        transform = location.Transformation()
        offset = len(vertices)
        triangle_offset = len(triangles)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))
        reversed_face = face.wrapped.Orientation() == TopAbs_REVERSED
        for triangle in triangulation.Triangles():
            a, b, c = triangle.Get()
            triangles.append((offset + a - 1, offset + (c if reversed_face else b) - 1,
                              offset + (b if reversed_face else c) - 1))
        face_ranges.append((offset, len(vertices), triangle_offset, len(triangles)))
    result = (np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))
    return (*result, face_ranges) if return_face_ranges else result

# Search paths for AMRTO binaries and data
_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]


def get_instant_meshes_executable() -> Optional[Path]:
    """Find the Instant-Meshes executable."""
    from pylcss.cae.solver_backends.execution import resolve_executable

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
    from pylcss.cae.solver_backends.execution import resolve_executable

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
    from pylcss.cae.solver_backends.execution import resolve_executable

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
    from pylcss.cae.solver_backends.execution import resolve_executable

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
    density_smooth_window: int = 3
    density_smooth_sigma: float = 0.5
    mesh_laplacian_smoothing_iters: int = 2
    mesh_laplacian_lambda: float = 0.3

    # Step 2: Quadrangulation
    target_quad_faces: int = 10000
    quad_crease_deg: float = 0.0  # Topopt isosurfaces are smooth continuum geometry without artificial facet creases

    # Step 3: Sparse layouts division
    layout_min_faces: int = 4
    layout_max_faces: int = 1500
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
    scaffold_max_iter: int = 50
    scaffold_relative_energy_tolerance: float = 1e-4
    scaffold_rectangle_tolerance: float = 1e-6
    scaffold_timeout_s: float = 60.0
    scaffold_require_native: bool = False
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
    multires_compression_ratio: float = 0.5  # Highest-layer total controls / original total
    multires_layers: int = 1
    multires_regeneration: str = "author_uniform"
    multires_maximum_knot_removal_attempts: int = 64
    preserve_patch_boundaries: bool = True  # Common boundary curves for sewing
    fitting_workers: int = 4  # Bound patch processes; each uses one BLAS thread
    # Repair is a bounded quality extension run once per patch. Keep its total
    # cost predictable on large layouts; unresolved patches are handled by the
    # existing subdivision and validation path.
    orientation_repair_seconds: float = 2.0
    orientation_repair_iterations: int = 2

    # OCC B-rep assembly
    sew_tolerance: float = 1e-3
    merge_angle_deg: float = 0.0
    export_step: bool = True
    export_iges: bool = False
    export_3dm: bool = False


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
