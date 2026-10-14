# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Sample original protected geometry without reconstructing lattice CAD."""

from __future__ import annotations

import numpy as np

from .passive_regions import _resample_source_mask


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

    lower, upper = (np.asarray(value, dtype=float) for value in bounds)
    spacing = (upper - lower) / np.asarray(build_shape)
    tolerance = max(1e-7, float(np.min(spacing)) * 0.1)

    def surface(raw):
        shape = _coerce_to_solid(raw)
        if shape is None or not shape.isValid() or not shape.Solids():
            raise ValueError("A saved non-design or source body is not a valid solid.")
        vertices, faces = shape.tessellate(tolerance, 0.2)
        return {
            "vertices": np.asarray([point.toTuple() for point in vertices]),
            "faces": np.asarray(faces, dtype=np.int64),
            "source_shape": shape,
        }

    def raster(mesh, dims):
        mask = _surface_design_domain_grid(mesh, bounds, *dims)
        if mask is None:
            mask = _cad_design_domain_grid(mesh["source_shape"], bounds, *dims)
        if mask is None:
            raise ValueError("The original non-design geometry could not be sampled for the lattice mesh.")
        return mask

    masks = []
    coarse_solids = np.zeros(tuple(build_shape), dtype=bool)
    for raw_shapes, saved, is_solid in (
        (solid_shapes, solid_mask, True), (void_shapes, void_mask, False),
    ):
        refined = np.zeros(tuple(build_shape), dtype=bool)
        coarse = np.zeros(tuple(analysis_shape), dtype=bool)
        for raw in raw_shapes:
            mesh = surface(raw)
            fine = raster(mesh, build_shape)
            if not np.any(fine):
                raise ValueError("A non-design region is too thin for the lattice build grid.")
            refined |= fine
            coarse |= raster(mesh, analysis_shape)
        coarse = _resample_source_mask(coarse, tuple(build_shape))
        retained = np.zeros(tuple(build_shape), dtype=bool) if saved is None else np.asarray(saved, dtype=bool).copy()
        retained = (retained & ~coarse) | refined
        masks.append(retained)
        if is_solid:
            coarse_solids = coarse

    domain = raster(surface(source_shape), build_shape) if source_shape is not None else None
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
    }
    return masks[0], masks[1], domain, coarse_solids, report
