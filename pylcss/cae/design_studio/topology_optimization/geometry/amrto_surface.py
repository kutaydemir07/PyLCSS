# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Verified AMRTO quadrangulation and sparse layouts, reusable without a solve."""
from __future__ import annotations

import hashlib
import numpy as np

from .amrto_common import AMRTOConfig
from .amrto_quadrangulation import quadrangulate_mesh
from .amrto_layout_partition import divide_sparse_quad_layouts
from .cad_mesh_repair import _extract_recovered_mesh
from .cad_diagnostics import assess_sampled_mesh_deviation
from .mesh_topology import triangulate_quad_faces
from .quad_layout import validate_quad_layout

SURFACE_LAYOUT_REVISION = 2


def geometry_fingerprint(vertices, faces):
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype='<f8').tobytes())
    digest.update(np.asarray(faces, dtype='<i8').tobytes())
    return digest.hexdigest()


def layout_display_data(vertices, quads, layouts, *, reports=None):
    """Give each quad exactly one selectable layout and retain real graph seams."""
    lookup = {tuple(sorted(map(int, quad))): i for i, quad in enumerate(quads)}
    if len(lookup) != len(quads):
        raise ValueError('Quad mesh contains duplicate faces')
    owners = np.full(len(quads), -1, dtype=np.int64)
    boundaries = set()
    for group, layout in enumerate(layouts):
        ids = np.asarray(layout.source_vertex_indices, dtype=np.int64)
        for quad in ids[layout.quads]:
            key = tuple(sorted(map(int, quad)))
            if key not in lookup or owners[lookup[key]] != -1:
                raise ValueError('Sparse layouts must cover every quad exactly once')
            owners[lookup[key]] = group
        for path in layout.boundary_paths:
            for a, b in zip(ids[path[:-1]], ids[path[1:]]):
                boundaries.add(tuple(sorted((int(a), int(b)))))
    if np.any(owners < 0):
        raise ValueError('Sparse layouts leave uncovered quads')
    return {'vertices': np.asarray(vertices), 'quads': np.asarray(quads),
            'group_ids': owners, 'boundary_edges': np.asarray(sorted(boundaries), dtype=np.int64),
            'group_count': len(layouts), 'reports': dict(reports or {}),
            'revision': SURFACE_LAYOUT_REVISION}


def cached_surface_layout(payload, *, requested_settings=None):
    if not isinstance(payload, dict):
        return None
    recovered = _extract_recovered_mesh(payload)
    cache = payload.get('surface_layout')
    if cache is None and isinstance(payload.get('recovered_shape'), dict):
        cache = payload['recovered_shape'].get('surface_layout')
    if recovered is None or not isinstance(cache, dict):
        return None
    if (cache.get('revision') != SURFACE_LAYOUT_REVISION
            or cache.get('source_geometry_sha256') != geometry_fingerprint(*recovered)):
        return None
    if requested_settings is not None and cache.get('requested_settings') != requested_settings:
        return None
    try:
        vertices = np.asarray(cache.get('vertices'), dtype=float)
        quads = np.asarray(cache.get('quads'))
        groups = np.asarray(cache.get('group_ids'))
        source_groups = np.asarray(cache.get('source_group_ids'))
        count = int(cache.get('group_count', 0))
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or quads.ndim != 2 or quads.shape[1] != 4 or not len(quads)
                or not np.issubdtype(quads.dtype, np.integer)
                or quads.min() < 0 or quads.max() >= len(vertices)
                or groups.shape != (len(quads),) or source_groups.shape != (len(recovered[1]),)
                or count < 1):
            return None
        for owners in (groups, source_groups):
            if not np.issubdtype(owners.dtype, np.integer) or owners.min() < 0 or owners.max() >= count:
                return None
    except (TypeError, ValueError, OverflowError):
        return None
    return cache


def add_source_groups(result, vertices, faces):
    """Map display layouts to original triangles for saved graph selections."""
    import trimesh
    quads = result['quads']
    triangles = triangulate_quad_faces(result['vertices'], quads)
    mesh = trimesh.Trimesh(result['vertices'], triangles, process=False)
    centers = np.asarray(vertices)[faces].mean(axis=1)
    groups = np.tile(result['group_ids'], 2)
    result['source_group_ids'] = np.empty(len(faces), dtype=np.int64)
    for start in range(0, len(centers), 512):
        _, _, nearest = trimesh.proximity.closest_point(mesh, centers[start:start+512])
        result['source_group_ids'][start:start+512] = groups[nearest]
    return result


def prepare_amrto_surface(payload, *, progress_callback=None, target_faces=None,
                          crease_deg=0., maximum_relative_deviation=.01,
                          maximum_volume_delta=.05):
    """Prepare native quads and validated layouts, recording grouping provenance."""
    import trimesh
    recovered = _extract_recovered_mesh(payload)
    if recovered is None:
        raise ValueError('A recovered topology surface is required')
    vertices, faces = recovered
    source = trimesh.Trimesh(vertices, faces, process=False)
    if not source.is_watertight or not source.is_winding_consistent or source.volume <= 0:
        raise ValueError('The recovered topology surface must be closed and outward oriented')
    requested = int(target_faces or min(AMRTOConfig().target_quad_faces, max(300, len(faces)//4)))
    if requested < 1 or not np.isfinite(crease_deg) or not 0 <= crease_deg <= 180:
        raise ValueError('Quad target must be positive and crease angle must be between 0 and 180')
    if any(not np.isfinite(value) or value < 0
           for value in (maximum_relative_deviation, maximum_volume_delta)):
        raise ValueError('Surface deviation and volume limits must be finite and nonnegative')
    settings = {'target_faces': requested, 'crease_deg': float(crease_deg),
                'maximum_relative_deviation': float(maximum_relative_deviation),
                'maximum_volume_delta': float(maximum_volume_delta)}
    cache = cached_surface_layout(payload, requested_settings=settings)
    if cache is not None:
        return cache
    def progress(stage):
        if progress_callback:
            progress_callback({'type': 'topopt_progress', 'status_only': True,
                               'stage': stage}, None, 0, 0)
    def validate(candidate_vertices, candidate_faces):
        report = assess_sampled_mesh_deviation(vertices, faces, candidate_vertices, candidate_faces,
                                              samples=1000)
        report['requested_maximum_relative_deviation'] = .5*maximum_relative_deviation
        report['passed'] = report['maximum_relative_distance'] <= .5*maximum_relative_deviation
        return report
    progress('Creating quad surface')
    quad_vertices, quads, quad_report = quadrangulate_mesh(
        vertices, faces, target_faces=requested, crease_deg=crease_deg,
        triangle_surface_validator=validate, require_native=True,
        maximum_volume_delta=maximum_volume_delta)
    validate_quad_layout(source, quad_vertices, quads)
    progress('Grouping surface regions')
    layouts, group_report = divide_sparse_quad_layouts(quad_vertices, quads, target_patch_size=50)
    result = layout_display_data(quad_vertices, quads, layouts,
                                 reports={'quadrangulation': quad_report, 'grouping': group_report})
    result['source_geometry_sha256'] = geometry_fingerprint(vertices, faces)
    result['requested_settings'] = settings
    return add_source_groups(result, vertices, faces)
