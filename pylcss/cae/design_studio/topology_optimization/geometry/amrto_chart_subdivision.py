# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Local source-quad subdivision proposals for rejected fitted charts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from .amrto_layout_partition import subdivide_fitting_layouts


def _point(point):
    return tuple(map(float, point))


def _quad_coverage(layouts):
    """Compare exact directed source polygons, permitting cyclic reindexing."""
    coverage = Counter()
    for layout in layouts:
        points, quads = np.asarray(layout.vertices), np.asarray(layout.quads)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or quads.ndim != 2 or quads.shape[1] != 4 or not len(quads)
                or not np.issubdtype(quads.dtype, np.integer)
                or quads.min() < 0 or quads.max() >= len(points)):
            raise ValueError('Source-chart subdivision requires finite indexed original quads')
        for quad in quads:
            vertices = tuple(_point(point) for point in points[quad])
            if len(set(map(int, quad))) != 4:
                raise ValueError('Source-chart subdivision cannot replace degenerate source quads')
            coverage[min(vertices[offset:]+vertices[:offset] for offset in range(4))] += 1
    return coverage


def _cancel_internal_edges(edges):
    remainder = edges.copy()
    for first, second in edges:
        if first == second:
            raise ValueError('Source-chart boundary has a zero-length geometric edge')
        count = min(remainder[first, second], remainder[second, first])
        remainder[first, second] -= count
        remainder[second, first] -= count
    return +remainder


def _directed_boundary(layout):
    """Validate the complete declared boundary against the original polygons."""
    quad_edges = Counter()
    for quad in np.asarray(layout.quads):
        for first, second in zip(quad, np.roll(quad, -1)):
            quad_edges[_point(layout.vertices[first]), _point(layout.vertices[second])] += 1
    expected = _cancel_internal_edges(quad_edges)
    paths = layout.boundary_paths
    if (len(paths) != 4 or any(len(path) < 2 for path in paths)
            or any(paths[index][-1] != paths[(index+1) % 4][0] for index in range(4))):
        raise ValueError('Source-chart subdivision requires four connected boundary paths')
    actual = Counter((_point(layout.vertices[first]), _point(layout.vertices[second]))
                     for path in paths for first, second in zip(path, path[1:]))
    # A declared outer loop cannot borrow cancelling interior traversals.
    # Compare its complete directed coverage, or net boundary if interior cut edges cancel.
    if actual != expected and _cancel_internal_edges(actual) != expected:
        raise ValueError('Source-chart boundary paths changed directed original quad coverage')
    return actual


def _check_source_correspondence(parent, pieces):
    source = np.asarray(getattr(parent, 'source_vertex_indices', np.arange(len(parent.vertices))))
    if source.shape != (len(parent.vertices),):
        raise ValueError('Source-chart subdivision requires complete parent source correspondence')
    original = {(int(index), _point(point)) for index, point in zip(source, parent.vertices) if index >= 0}
    for piece in pieces:
        indices = np.asarray(piece.source_vertex_indices)
        if (indices.shape != (len(piece.vertices),) or np.any(indices < -1)
                or np.any(indices[piece.quads] < 0)
                or any((int(index), _point(point)) not in original
                       for index, point in zip(indices, piece.vertices) if index >= 0)):
            raise ValueError('Source-chart subdivision changed original source vertex correspondence')
        if any(indices[index] < 0 for path in piece.boundary_paths for index in path):
            raise ValueError('Source-chart subdivision added a synthetic shared-boundary vertex')


def subdivide_rejected_source_charts(layouts, rejected_layout_ids):
    """Propose one exact-quad subdivision pass only for named rejected charts.

    This engineering extension neither fits nor accepts CAD. The caller owns
    the two-round retry limit and must reparameterize replacement charts,
    rebuild shared boundaries and refit affected neighbors before all normal
    source, orientation, assembly and non-design checks. Untargeted layouts
    retain their objects, IDs, geometry, mappings and completed fits.
    """
    layouts = list(layouts)
    ids = [int(layout.layout_id) for layout in layouts]
    if len(ids) != len(set(ids)):
        raise ValueError('Source-chart subdivision requires unique layout IDs')
    rejected = set(map(int, rejected_layout_ids))
    if rejected-set(ids):
        raise ValueError('Source-chart subdivision names an unknown rejected layout')
    report = {
        'method': 'Subdivide only rejected charts using their unchanged original quads',
        'engineering_extension': True, 'paper_algorithm': False,
        'requested_layout_ids': sorted(rejected), 'replacements': [], 'unrefined': [],
        'untargeted_layouts_unchanged': True, 'surface_fits': 0,
        'source_quad_coordinates_unchanged': True, 'source_quad_winding_unchanged': True,
        'source_vertex_correspondence_unchanged': True,
        'source_quad_coverage_exactly_once': True, 'directed_outer_boundary_unchanged': True,
        'caller_must_rebuild_shared_boundaries': True,
    }
    if not rejected:
        return layouts, {**report, 'applied': False, 'replacement_layout_ids': [],
                         'incident_layout_ids': []}
    parents = {int(layout.layout_id): layout for layout in layouts}
    replacements = {}
    next_id = max(ids, default=-1)+1
    changed_boundary_points = set()
    for parent_id in sorted(rejected):
        parent = parents[parent_id]
        before = _quad_coverage([parent])
        outer = _directed_boundary(parent)
        if len(parent.quads) <= 1:
            report['unrefined'].append({'parent_layout_id': parent_id,
                'reason': 'The rejected chart already contains one original quad'})
            continue
        # The older helper renumbers its whole argument. Keep that operation
        # private to this one parent, including any parameterization caches.
        pieces, details = subdivide_fitting_layouts([deepcopy(parent)], {parent_id})
        if len(pieces) <= 1 or any(len(piece.quads) >= len(parent.quads) for piece in pieces):
            raise ValueError('Source-chart subdivision failed to reduce the rejected chart')
        if _quad_coverage(pieces) != before:
            raise ValueError('Source-chart subdivision changed directed source quads or coordinates')
        _check_source_correspondence(parent, pieces)
        replacement_boundary = Counter()
        for piece in pieces:
            replacement_boundary.update(_directed_boundary(piece))
        if _cancel_internal_edges(replacement_boundary) != _cancel_internal_edges(outer):
            raise ValueError('Source-chart subdivision changed the directed outer boundary')
        origin = int(getattr(parent, 'source_chart_parent_id', parent_id))
        generation = int(getattr(parent, 'source_chart_subdivision_generation', 0))+1
        new_ids = []
        for piece in pieces:
            piece.layout_id = next_id
            next_id += 1
            piece.source_chart_parent_id = origin
            piece.source_chart_subdivision_generation = generation
            # A proposal needs a fresh full Stage4 map and Stage5 fit. The
            # partitioner's trial harmonic coordinates are not an acceptance.
            piece.parameter_coords = None
            piece.sampled_grid_3d = None
            piece.sampled_shape = (0, 0)
            piece.nurbs_surface = None
            new_ids.append(piece.layout_id)
        replacements[parent_id] = pieces
        changed_boundary_points.update(point for edge in outer for point in edge)
        report['replacements'].append({
            'parent_layout_id': parent_id, 'original_source_parent_id': origin,
            'subdivision_generation': generation, 'source_quad_count': len(parent.quads),
            'replacement_layout_ids': new_ids,
            'replacement_quad_counts': [len(piece.quads) for piece in pieces],
            'covered_original_quads': sum(before.values()),
            'directed_outer_boundary_edge_count': sum(_cancel_internal_edges(outer).values()),
            'partition_details': details,
        })
    result = [piece for layout in layouts
              for piece in replacements.get(int(layout.layout_id), [layout])]
    # Point-adjacent neighbors are conservative candidates for shared-chain
    # refitting. Actual changed-curve comparison can reduce this set further.
    incident = sorted(int(layout.layout_id) for layout in layouts
                      if int(layout.layout_id) not in replacements and any(
                          _point(layout.vertices[index]) in changed_boundary_points
                          for path in layout.boundary_paths for index in path))
    report.update(applied=bool(replacements), incident_layout_ids=incident,
                  replacement_layout_ids=[piece.layout_id for pieces in replacements.values() for piece in pieces])
    return result, report
