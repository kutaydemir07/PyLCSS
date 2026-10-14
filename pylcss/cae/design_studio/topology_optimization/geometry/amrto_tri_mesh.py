# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""AMRTO stage 1, using the shared feature-preserving mesh recovery engine."""
from .amrto_common import AMRTOConfig
from .mesh_surface import (
    smooth_pseudo_density_field as smooth_pseudo_density_field,
    extract_isosurface_mesh as extract_isosurface_mesh,
    preprocess_and_repair_mesh as preprocess_and_repair_mesh,
    _smooth_mesh_preserving_features as _smooth_mesh_preserving_features,
    _collapse_zero_area_faces_preserving_topology as _collapse_zero_area_faces_preserving_topology,
    generate_topopt_triangular_mesh as _generate_mesh,
)


def generate_topopt_triangular_mesh(field, cutoff=0.5, bounds=None, config=None):
    vertices, faces, report = _generate_mesh(field, cutoff=cutoff, bounds=bounds,
                                            config=config or AMRTOConfig(density_cutoff=cutoff))
    report['step'] = 1
    return vertices, faces, report
