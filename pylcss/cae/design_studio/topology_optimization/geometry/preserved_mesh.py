# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Sample original protected geometry without reconstructing lattice CAD."""

from __future__ import annotations

from collections import OrderedDict
import logging
from threading import RLock
from time import perf_counter

import numpy as np

from .cad_fingerprint import cad_shape_fingerprint
from .passive_regions import _resample_source_mask


logger = logging.getLogger(__name__)

# Keep reusable occupancy only, not CAD objects or their tessellations. A few
# refined lattice masks can be large, so bound both their count and total size.
_PRESERVED_MASK_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_PRESERVED_MASK_CACHE_BYTES = 64 * 1024 * 1024
_PRESERVED_MASK_CACHE_ENTRIES = 32
_PRESERVED_MASK_CACHE_LOCK = RLock()


def _cached_mask(key):
    with _PRESERVED_MASK_CACHE_LOCK:
        mask = _PRESERVED_MASK_CACHE.get(key)
        if mask is not None:
            _PRESERVED_MASK_CACHE.move_to_end(key)
    # Callers combine and modify masks. Never expose the cache's own storage.
    return None if mask is None else mask.copy()


def _remember_mask(key, mask):
    if (_PRESERVED_MASK_CACHE_ENTRIES <= 0
            or mask.nbytes > _PRESERVED_MASK_CACHE_BYTES):
        return
    retained = mask.copy()
    retained.flags.writeable = False
    with _PRESERVED_MASK_CACHE_LOCK:
        _PRESERVED_MASK_CACHE[key] = retained
        _PRESERVED_MASK_CACHE.move_to_end(key)
        while (len(_PRESERVED_MASK_CACHE) > _PRESERVED_MASK_CACHE_ENTRIES
               or sum(value.nbytes for value in _PRESERVED_MASK_CACHE.values())
               > _PRESERVED_MASK_CACHE_BYTES):
            _PRESERVED_MASK_CACHE.popitem(last=False)


def refined_preserved_masks(
    bounds, analysis_shape, build_shape, solid_mask, void_mask,
    *, solid_shapes=(), void_shapes=(), source_shape=None,
):
    """Replace coarse copies of explicit face layers with finer source samples.

    Mask-only constraints remain intact. The CAD inputs are only tessellated;
    their surfaces and topology are retained in the saved result unchanged.
    """
    from ..integration.voxelization import _surface_design_domain_grid, _cad_design_domain_grid
    from .cad_features import _coerce_to_solid

    started = perf_counter()
    lower, upper = (np.asarray(value, dtype=float) for value in bounds)
    analysis_shape, build_shape = tuple(analysis_shape), tuple(build_shape)
    bounds_key = tuple(np.r_[lower, upper])
    spacing = (upper - lower) / np.asarray(build_shape)
    tolerance = max(1e-7, float(np.min(spacing)) * 0.1)
    statistics = {"mask_cache_hits": 0, "mask_cache_misses": 0,
                  "same_grid_reuses": 0, "cad_classifier_passes": 0}

    def sampler(raw):
        shape = _coerce_to_solid(raw)
        if shape is None or not shape.isValid() or not shape.Solids():
            raise ValueError("A saved non-design or source body is not a valid solid.")
        # A TShape/hashCode identity does not change after all in-place edits.
        # Fingerprint actual geometry, topology and placement, excluding the
        # triangulation generated below so a cache hit needs no new meshing.
        fingerprint = cad_shape_fingerprint(shape)
        mesh = None

        def raster(dims):
            nonlocal mesh
            dims = tuple(dims)
            key = (fingerprint, bounds_key, dims, tolerance)
            mask = _cached_mask(key)
            if mask is not None:
                statistics["mask_cache_hits"] += 1
                return mask
            statistics["mask_cache_misses"] += 1
            if mesh is None:
                from .cad_tessellation import private_cad_mesh_copy
                vertices, faces = private_cad_mesh_copy(shape).tessellate(tolerance, 0.2)
                mesh = {
                    "vertices": np.asarray([point.toTuple() for point in vertices]),
                    "faces": np.asarray(faces, dtype=np.int64),
                    "source_shape": shape,
                }
            mask = _surface_design_domain_grid(mesh, bounds, *dims)
            if mask is None:
                statistics["cad_classifier_passes"] += 1
                mask = _cad_design_domain_grid(shape, bounds, *dims)
            if mask is None:
                raise ValueError("The original non-design geometry could not be sampled for the lattice mesh.")
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != dims:
                raise ValueError("The original geometry mask does not match the requested grid.")
            _remember_mask(key, mask)
            return mask

        return raster

    masks = []
    coarse_solids = np.zeros(tuple(build_shape), dtype=bool)
    for raw_shapes, saved, is_solid in (
        (solid_shapes, solid_mask, True), (void_shapes, void_mask, False),
    ):
        refined = np.zeros(tuple(build_shape), dtype=bool)
        coarse = np.zeros(tuple(analysis_shape), dtype=bool)
        for raw in raw_shapes:
            raster = sampler(raw)
            fine = raster(build_shape)
            if not np.any(fine):
                raise ValueError("A non-design region is too thin for the lattice build grid.")
            refined |= fine
            if analysis_shape == build_shape:
                coarse |= fine
                statistics["same_grid_reuses"] += 1
            else:
                coarse |= raster(analysis_shape)
        coarse = _resample_source_mask(coarse, tuple(build_shape))
        retained = np.zeros(tuple(build_shape), dtype=bool) if saved is None else np.asarray(saved, dtype=bool).copy()
        retained = (retained & ~coarse) | refined
        masks.append(retained)
        if is_solid:
            coarse_solids = coarse

    domain = sampler(source_shape)(build_shape) if source_shape is not None else None
    if domain is not None:
        masks[0] &= domain
    masks[0] &= ~masks[1]
    report = {
        "solid_body_count": len(solid_shapes), "void_body_count": len(void_shapes),
        "source_boundary_sampled": domain is not None,
        "source_tessellation_tolerance": tolerance,
        "build_spacing": spacing.tolist(),
        "solid_sample_count": int(np.count_nonzero(masks[0])),
        "method": "Original geometry sampled on the manufacturing grid; no lattice CAD reconstruction",
        **statistics,
        "mask_generation_seconds": perf_counter() - started,
    }
    logger.info("Preserved geometry masks: %d computed, %d cached, %d identical-grid reuses (%.3f s)",
                statistics["mask_cache_misses"], statistics["mask_cache_hits"],
                statistics["same_grid_reuses"], report["mask_generation_seconds"])
    return masks[0], masks[1], domain, coarse_solids, report
