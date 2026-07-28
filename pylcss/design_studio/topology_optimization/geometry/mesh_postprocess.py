# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recovered triangle-mesh cleanup and smoothing."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .analytic_shapes import AnalyticShape

logger = logging.getLogger(__name__)


def _enhanced_mesh_postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    decimate_ratio: float = 1.0,
    smoothing_iterations: int = 2,
) -> Optional[dict[str, np.ndarray]]:
    """Run a print-ready post-processing pipeline using trimesh.

    Pipeline:
        1. Split into connected components, keep ones above 1% of max volume.
        2. Fill small holes.
        3. Light Humphrey smoothing (volume-preserving alternative to Laplacian).
        4. Optional quadric decimation (requires `fast_simplification`).

    Returns None if trimesh is unavailable so the caller can fall back to the
    legacy Taubin path.
    """
    try:
        import trimesh
        import trimesh.smoothing
    except ImportError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    # 1. Keep only meaningful connected components.
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = [mesh]
    if components:
        volumes = [abs(float(c.volume)) for c in components]
        if volumes:
            max_vol = max(volumes)
            kept = [
                c
                for c, v in zip(components, volumes, strict=True)
                if v >= 0.01 * max_vol
            ]
            if kept:
                mesh = trimesh.util.concatenate(kept)

    # 2. Close small holes.
    try:
        mesh.fill_holes()
    except Exception:
        logger.debug("trimesh.fill_holes failed; continuing")

    # 3. Volume-preserving Humphrey smoothing.
    if int(smoothing_iterations) > 0:
        try:
            trimesh.smoothing.filter_humphrey(
                mesh,
                alpha=0.1,
                beta=0.5,
                iterations=int(smoothing_iterations),
            )
        except Exception:
            logger.debug("Humphrey smoothing failed; falling back to Taubin")
            try:
                trimesh.smoothing.filter_taubin(
                    mesh, iterations=int(smoothing_iterations)
                )
            except Exception:
                pass

    # 4. Optional decimation (requires `fast_simplification`).
    if 0.0 < float(decimate_ratio) < 1.0 and len(mesh.faces) > 0:
        target = max(64, int(len(mesh.faces) * float(decimate_ratio)))
        try:
            decimated = mesh.simplify_quadric_decimation(face_count=target)
            if decimated is not None and len(decimated.faces) > 0:
                mesh = decimated
        except ImportError:
            logger.info(
                "STL decimation skipped — install `fast_simplification` for "
                "quadric decimation support."
            )
        except Exception:
            logger.debug("Decimation failed; keeping un-decimated mesh")

    return {
        "vertices": np.asarray(mesh.vertices, dtype=float),
        "faces": np.asarray(mesh.faces, dtype=int),
    }


def _taubin_smooth_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 6,
    shapes: Sequence[AnalyticShape] = (),
    tolerance: float = 0.0,
) -> np.ndarray:
    """Light volume-preserving smoothing for marching-cubes output."""
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts

    verts = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=int)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )

    for _ in range(max(0, int(iterations))):
        for factor in (0.5, -0.53):
            neighbor_sum = np.zeros_like(verts)
            neighbor_count = np.zeros(len(verts), dtype=float)
            np.add.at(neighbor_sum, edges[:, 0], verts[edges[:, 1]])
            np.add.at(neighbor_count, edges[:, 0], 1.0)
            np.add.at(neighbor_sum, edges[:, 1], verts[edges[:, 0]])
            np.add.at(neighbor_count, edges[:, 1], 1.0)
            mask = neighbor_count > 0
            avg = neighbor_sum[mask] / neighbor_count[mask, None]

            displacement = avg - verts[mask]

            if shapes and tolerance > 0.0:
                for shape in shapes:
                    dist = shape.sdf(verts[mask])
                    near_mask = np.abs(dist) <= tolerance
                    if np.any(near_mask):
                        n = shape.get_normal(verts[mask][near_mask])
                        disp_sub = displacement[near_mask]
                        proj_dot = np.sum(disp_sub * n, axis=-1, keepdims=True)
                        displacement[near_mask] = disp_sub - proj_dot * n

            verts[mask] += factor * displacement
    return verts
