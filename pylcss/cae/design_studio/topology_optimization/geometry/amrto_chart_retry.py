# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""One local chart proposal after failed solid integration.

This is an engineering extension to the paper, not a different source surface.
It changes square corners and shared endpoint tangents only on rejected charts.
All source vertices, triangles, quads and directed boundary edges are retained.
The caller must repeat the existing fitting and solid acceptance checks.
"""
from collections import Counter
from copy import deepcopy

import numpy as np

from .amrto_layout_partition import _required_square_corners, repair_reflex_square_corners
from .amrto_parameterization import parameterize_paper_layout, adaptive_sample_layout


def _balanced_required_corners(loop, valid, required, distance):
    """Minimax four-gap search that cannot skip a mandatory source ear.

    Used only when the original greedy selector has no admissible quartet.
    Four dynamic-programming stages bound runtime to O(4*n**2); candidate
    count is explicitly bounded. All choices remain existing convex vertices.
    """
    if not 1 <= len(required) <= 4 or not required.issubset({loop[i] for i in valid}):
        raise ValueError('No convex quartet can retain every required source triangle ear')
    if len(valid) > 4096:
        raise ValueError('Required-ear corner search exceeded its bounded candidate count')
    anchor = next(i for i in valid if loop[i] in required)
    indices = sorted(valid, key=lambda i: (i-anchor) % len(loop))
    positions = np.r_[(distance[indices]-distance[anchor]) % distance[-1], distance[-1]]
    mandatory = np.array([j for j,i in enumerate(indices) if loop[i] in required] + [len(indices)])
    previous = np.full(len(positions), np.inf)
    previous[0] = 0.
    back = np.full((5, len(positions)), -1, dtype=int)
    for count in range(1, 5):
        current = np.full(len(positions), np.inf)
        for j in (range(1, len(indices)) if count < 4 else [len(indices)]):
            lower = int(mandatory[mandatory < j][-1])
            candidates = np.arange(lower, j)
            values = np.maximum(previous[candidates],
                abs((positions[j]-positions[candidates])/distance[-1]-.25))
            winner = int(np.argmin(values))
            current[j], back[count, j] = values[winner], candidates[winner]
        previous = current
    if not np.isfinite(previous[-1]):
        raise ValueError('No four convex corners contain every mandatory source ear')
    selected, j = [], len(indices)
    for count in range(4, 0, -1):
        j = int(back[count, j])
        selected.append(indices[j])
    selected.sort()
    if len(set(selected)) != 4 or not required.issubset({loop[i] for i in selected}):
        raise ValueError('Required-ear corner search lost a mandatory source vertex')
    gaps = np.diff(np.r_[distance[selected], distance[selected[0]]+distance[-1]])
    return float(previous[-1]), selected, gaps


def balance_convex_corners(layout):
    """Select four unambiguous convex corners with balanced boundary lengths."""
    old_paths = layout.boundary_paths
    if len(old_paths) != 4 or any(len(path) < 2 for path in old_paths):
        raise ValueError('Balanced corners require four connected source boundary paths')
    if any(old_paths[i][-1] != old_paths[(i + 1) % 4][0] for i in range(4)):
        raise ValueError('Source boundary paths are not connected')
    loop = [int(vertex) for path in old_paths for vertex in path[:-1]]
    if len(set(loop)) != len(loop):
        raise ValueError('Balanced corners require one simple source boundary')
    xyz, triangles = np.asarray(layout.vertices), np.asarray(layout.triangles)
    normals = np.cross(xyz[triangles[:, 1]] - xyz[triangles[:, 0]],
                       xyz[triangles[:, 2]] - xyz[triangles[:, 0]])
    first = (loop[0], loop[1])
    for triangle in triangles:
        edges = set(zip(map(int, triangle), map(int, np.roll(triangle, -1))))
        if first in edges:
            break
        if first[::-1] in edges:
            normals = -normals
            break
    else:
        raise ValueError('Declared source boundary edge is absent from source triangles')
    valid = []
    for index, vertex in enumerate(loop):
        cross = np.cross(xyz[loop[(index + 1) % len(loop)]] - xyz[vertex],
                         xyz[loop[index - 1]] - xyz[vertex])
        incident = np.flatnonzero(np.any(triangles == vertex, axis=1))
        scales = np.linalg.norm(cross) * np.linalg.norm(normals[incident], axis=1)
        if len(incident) and np.all(scales > 0) and np.all(normals[incident] @ cross / scales > 1e-8):
            valid.append(index)
    if len(valid) < 4:
        raise ValueError('Fewer than four unambiguous convex source boundary vertices')
    distance = np.r_[0., np.cumsum(np.linalg.norm(xyz[np.roll(loop, -1)] - xyz[loop], axis=1))]
    if not np.isfinite(distance[-1]) or distance[-1] <= 0:
        raise ValueError('Source boundary has invalid arclength')
    required = _required_square_corners(triangles, loop)
    candidates = []
    for anchor in valid:
        relative = (distance[np.asarray(valid)] - distance[anchor]) % distance[-1]
        selected = [anchor]
        for fraction in (.25, .5, .75):
            choice = min((j for j in range(len(valid)) if valid[j] not in selected),
                         key=lambda j: abs(relative[j] - fraction * distance[-1]))
            selected.append(valid[choice])
        if not required.issubset({loop[i] for i in selected}):
            continue
        selected.sort()
        gaps = np.diff(np.r_[distance[selected], distance[selected[0]] + distance[-1]])
        score = float(np.max(np.abs(gaps / distance[-1] - .25)))
        candidates.append((score, selected, gaps))
    used_required_search = not bool(candidates)
    if candidates:
        score, indices, gaps = min(candidates, key=lambda row: (row[0], row[1]))
    else:
        score, indices, gaps = _balanced_required_corners(loop, valid, required, distance)
    ordered = loop[indices[0]:] + loop[:indices[0]]
    corners = [loop[i] for i in indices]
    positions = [ordered.index(vertex) for vertex in corners]
    paths = [ordered[positions[i]:positions[i + 1] + 1] if i < 3
             else ordered[positions[i]:] + [ordered[0]] for i in range(4)]
    edges = lambda ps: Counter((a, b) for path in ps for a, b in zip(path, path[1:]))
    if edges(paths) != edges(old_paths):
        raise ValueError('Balanced corners changed directed source boundary coverage')
    report = {'method': 'Perimeter-balanced unambiguous convex square corners',
              'engineering_extension': True, 'paper_algorithm': False,
              'applied': corners != [path[0] for path in old_paths],
              'original_corners': [int(path[0]) for path in old_paths], 'final_corners': corners,
              'perimeter_fraction_max_error': score, 'boundary_arclengths': gaps.tolist(),
              'required_source_ear_vertices': sorted(map(int, required)),
              'required_ear_dynamic_search': used_required_search,
              'convex_candidate_count': len(valid),
              'source_vertices_triangles_quads_unchanged': True,
              'directed_source_boundary_coverage_unchanged': True}
    if report['applied']:
        layout.corner_indices, layout.boundary_paths = corners, paths
        layout.parameter_coords = None
        layout.sampled_grid_3d = None
    return report


def propose_failed_chart_retry(clean_layouts, rejected_layout_ids, cfg):
    """Return an isolated single proposal; never mutate the original attempt."""
    targets = set(map(int, rejected_layout_ids))
    layouts = deepcopy(clean_layouts)
    report = {'method': 'One bounded local chart and endpoint-tangent retry after failed solid integration',
              'engineering_extension': True, 'paper_algorithm': False, 'attempts': 1,
              'target_layout_ids': sorted(targets), 'patch_partition_unchanged': True,
              'source_geometry_unchanged': True, 'corner_proposals': []}
    tangent_edges = set()
    for layout in layouts:
        if int(layout.layout_id) not in targets:
            continue
        before = tuple(np.asarray(getattr(layout, name)).copy() for name in ('vertices', 'triangles', 'quads'))
        lengths = np.asarray([np.linalg.norm(np.diff(layout.vertices[path], axis=0), axis=1).sum()
                              for path in layout.boundary_paths])
        ratio = float(lengths.max() / lengths.min()) if lengths.min() > 0 else float('inf')
        # A severely unbalanced square undersamples a wide physical interior.
        # This dimensionless heuristic proposes corners; it does not admit CAD.
        proposal = None
        if ratio > 8.0:
            try:
                proposal = balance_convex_corners(layout)
            except ValueError as error:
                report['corner_proposals'].append({'layout_id': int(layout.layout_id),
                    'balanced_proposal_rejection': str(error), 'original_boundary_length_ratio': ratio})
        if proposal is None:
            proposal = repair_reflex_square_corners(layout)
            if not proposal.get('applied') and len(layout.quads) > 256:
                # A large chart can fold at an internal common-seam knot even
                # when none of its four corners is reflex. Try one balanced
                # square on the same source before changing only its tangents.
                try:
                    balanced = balance_convex_corners(layout)
                    balanced['initial_corner_orientation_assessment'] = proposal
                    proposal = balanced
                except ValueError as error:
                    proposal['balanced_proposal_rejection'] = str(error)
        proposal.update(layout_id=int(layout.layout_id), original_boundary_length_ratio=ratio)
        report['corner_proposals'].append(proposal)
        if proposal.get('applied'):
            parameterize_paper_layout(layout, cfg)
            adaptive_sample_layout(layout, k_sample=cfg.sampling_rate_k,
                                   d_min=cfg.sampling_min_d, d_max=cfg.sampling_max_d)
        for old, name in zip(before, ('vertices', 'triangles', 'quads')):
            if not np.array_equal(old, getattr(layout, name)):
                raise ValueError(f'Chart retry modified source {name}')
        tangent_edges.update(tuple(sorted((tuple(map(float, layout.vertices[a])),
                                            tuple(map(float, layout.vertices[b])))))
                             for path in layout.boundary_paths for a, b in zip(path, path[1:]))
    report['tangent_preserving_source_edge_count'] = len(tangent_edges)
    return layouts, tangent_edges, report


def refine_failed_chart_retry(clean_layouts, rejected_layout_ids, cfg, *, round_index):
    """One of at most two local subdivision rounds, with a bounded chart size.

    Large rejected charts get one alternative convex corner proposal instead
    of an uncontrolled explosion into hundreds of small fitted patches.
    """
    from .amrto_chart_subdivision import subdivide_rejected_source_charts
    if round_index not in (1, 2):
        raise ValueError('Source chart refinement permits at most two rounds')
    targets = set(map(int, rejected_layout_ids))
    layouts = deepcopy(clean_layouts)
    small = {int(layout.layout_id) for layout in layouts
             if int(layout.layout_id) in targets and 1 < len(layout.quads) <= 256}
    layouts, subdivision = subdivide_rejected_source_charts(layouts, small)
    replacements = set(subdivision['replacement_layout_ids'])
    report = {'round': round_index, 'subdivision': subdivision, 'corner_proposals': [],
              'maximum_subdivided_chart_quads': 256, 'maximum_subdivision_rounds': 2,
              'engineering_extension': True, 'paper_algorithm': False, 'applied': subdivision['applied']}
    tangent_edges = set()
    for layout in layouts:
        index = int(layout.layout_id)
        remap = index in replacements
        if remap:
            repair = repair_reflex_square_corners(layout)
            report['corner_proposals'].append({'layout_id': index, **repair})
        elif index in targets and len(layout.quads) > 256 and round_index == 1:
            try:
                repair = balance_convex_corners(layout)
                report['corner_proposals'].append({'layout_id': index, **repair})
                remap = bool(repair['applied'])
                report['applied'] |= remap
            except ValueError as error:
                report['corner_proposals'].append({'layout_id': index, 'rejection_reason': str(error)})
        if remap:
            parameterize_paper_layout(layout, cfg)
            adaptive_sample_layout(layout, k_sample=cfg.sampling_rate_k,
                                   d_min=cfg.sampling_min_d, d_max=cfg.sampling_max_d)
        if index in targets or index in replacements:
            tangent_edges.update(tuple(sorted((tuple(map(float, layout.vertices[a])),
                                                tuple(map(float, layout.vertices[b])))))
                                 for path in layout.boundary_paths for a,b in zip(path,path[1:]))
    return layouts, tangent_edges, report
