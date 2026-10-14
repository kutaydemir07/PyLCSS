# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Strut-network clipping and compatibility with saved member plans.

The current application sizes cells from density and geometry controls.
OptimizedMemberPlan retains the representation of historical sized results
so they can still be restored, rasterized, and exported. Shared clipping
and reconnection helpers support analytic beam-lattice reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimizedMemberPlan:
    """Geometry and diagnostics retained for historical sized-strut results."""

    mode: str
    grid_shape: tuple[int, int, int]
    nodes_fractional: np.ndarray
    members: np.ndarray
    radii_voxels: np.ndarray
    iterations: int
    converged: bool
    maximum_stress: float
    maximum_stress_utilization: float
    maximum_buckling_utilization: float
    maximum_displacement: float
    compliance: float
    member_volume: float
    allowable_stress: float
    stabilizing_member_count: int = 0
    warnings: tuple[str, ...] = ()

    def signature(self) -> str:
        """Return a stable cache signature for recovery at another resolution."""
        digest = hashlib.sha256()
        digest.update(self.mode.encode("utf-8"))
        digest.update(np.asarray(self.grid_shape, dtype=np.int64).tobytes())
        digest.update(
            np.ascontiguousarray(self.nodes_fractional, dtype=np.float64).tobytes()
        )
        digest.update(np.ascontiguousarray(self.members, dtype=np.int64).tobytes())
        digest.update(
            np.ascontiguousarray(self.radii_voxels, dtype=np.float64).tobytes()
        )
        return digest.hexdigest()

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-friendly engineering diagnostics."""
        return {
            "method": "axial 3-D truss fully-stressed sizing",
            "member_count": int(len(self.members)),
            "boundary_stabilizing_member_count": int(self.stabilizing_member_count),
            "node_count": int(len(self.nodes_fractional)),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "maximum_stress": float(self.maximum_stress),
            "allowable_stress": float(self.allowable_stress),
            "maximum_stress_utilization": float(self.maximum_stress_utilization),
            "maximum_buckling_utilization": float(self.maximum_buckling_utilization),
            "maximum_displacement": float(self.maximum_displacement),
            "compliance": float(self.compliance),
            "member_volume": float(self.member_volume),
            "warnings": list(self.warnings),
        }


def _sample_envelope(
    envelope: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    shape = np.asarray(envelope.shape, dtype=int)
    indices = np.rint(np.clip(points, 0.0, 1.0) * (shape - 1)).astype(int)
    return np.asarray(envelope[tuple(indices.T)], dtype=bool)


def _clip_network_to_envelope(
    envelope: np.ndarray,
    nodes: np.ndarray,
    members: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample members, trim exterior endpoints, and merge nearby nodes.

    Retain members with at least 35% sampled occupancy, falling back to 25%
    when none meet that threshold. Merge nodes within 2% of normalized domain
    coordinates, deduplicate members, and compact the node array.
    """
    if not len(members):
        return nodes[:0], members
    parameters = np.linspace(0.0, 1.0, 17)
    a = nodes[members[:, 0]]
    b = nodes[members[:, 1]]
    samples = (
        a[:, None, :] * (1.0 - parameters[None, :, None])
        + b[:, None, :] * parameters[None, :, None]
    )
    inside = _sample_envelope(envelope, samples.reshape((-1, 3))).reshape(
        (len(members), len(parameters))
    )
    inside_ratio = np.mean(inside, axis=1)
    keep_mask = inside_ratio >= 0.35
    if not np.any(keep_mask):
        keep_mask = inside_ratio >= 0.25
    if not np.any(keep_mask):
        return nodes[:0], members[:0]

    kept_members = members[keep_mask]
    kept_inside = inside[keep_mask]
    kept_a = a[keep_mask]
    kept_b = b[keep_mask]

    new_nodes_list = list(nodes)
    new_members = []

    for i, (m0, m1) in enumerate(kept_members):
        ins = kept_inside[i]
        pt_a = kept_a[i]
        pt_b = kept_b[i]

        idx_a = int(m0)
        if not ins[0]:
            first_idx = int(np.argmax(ins))
            t_first = float(parameters[first_idx])
            clipped_pt_a = pt_a * (1.0 - t_first) + pt_b * t_first
            idx_a = len(new_nodes_list)
            new_nodes_list.append(clipped_pt_a)

        idx_b = int(m1)
        if not ins[-1]:
            last_idx = len(ins) - 1 - int(np.argmax(ins[::-1]))
            t_last = float(parameters[last_idx])
            clipped_pt_b = pt_a * (1.0 - t_last) + pt_b * t_last
            idx_b = len(new_nodes_list)
            new_nodes_list.append(clipped_pt_b)

        if idx_a != idx_b:
            new_members.append((min(idx_a, idx_b), max(idx_a, idx_b)))

    if not new_members:
        return nodes[:0], members[:0]

    out_nodes = np.asarray(new_nodes_list, dtype=float)
    out_members = np.asarray(new_members, dtype=int)

    # Snap near-boundary nodes within 2% of the domain so boundary struts meet
    # at shared boundary anchor points
    from scipy.spatial import cKDTree

    tree = cKDTree(out_nodes)
    pairs = tree.query_pairs(0.02)
    if pairs:
        parent = np.arange(len(out_nodes))

        def find(u: int) -> int:
            path = []
            while parent[u] != u:
                path.append(u)
                u = parent[u]
            for p in path:
                parent[p] = u
            return u

        for u, v in pairs:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        roots = np.array([find(u) for u in range(len(out_nodes))])
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        merged_nodes = np.zeros((len(unique_roots), 3), dtype=float)
        for u in range(len(unique_roots)):
            merged_nodes[u] = np.mean(out_nodes[inverse == u], axis=0)
        merged_members = inverse[out_members]
        merged_members = merged_members[merged_members[:, 0] != merged_members[:, 1]]
        merged_members = np.sort(merged_members, axis=1)
        merged_members = np.unique(merged_members, axis=0)
        out_nodes, out_members = merged_nodes, merged_members

    used = np.unique(out_members)
    remap = np.full(len(out_nodes), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return out_nodes[used], remap[out_members]


def _connect_clipped_network(
    envelope: np.ndarray,
    nodes: np.ndarray,
    members: np.ndarray,
    cell_size_voxels: float,
    *,
    maximum_span_cells: float = 2.6,
) -> tuple[np.ndarray, int]:
    """Attempt to reconnect graph islands with short explicit members.

    Consider nearest node pairs first, up to maximum_span_cells. Accept each
    bridge only when at least 78% of its envelope samples are inside.
    Return the augmented member list and the number of added members.
    """
    if not len(nodes) or not len(members):
        return members, 0
    from scipy.spatial import cKDTree

    scaled = (
        nodes
        * np.asarray(envelope.shape, dtype=float)
        / max(float(cell_size_voxels), 1e-9)
    )

    parent = np.arange(len(nodes), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        parent[max(root_a, root_b)] = min(root_a, root_b)
        return True

    for node_a, node_b in members:
        union(int(node_a), int(node_b))

    tree = cKDTree(scaled)
    existing = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in members}
    added: list[tuple[int, int]] = []
    parameters = np.linspace(0.0, 1.0, 9)

    # Candidate bridges, shortest first: a lattice is repaired by its closest
    # loose ends, not by whichever island happens to be visited first.
    pairs = tree.query_pairs(float(maximum_span_cells), output_type="ndarray")
    if not len(pairs):
        return members, 0
    lengths = np.linalg.norm(scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1)
    for index in np.argsort(lengths):
        node_a, node_b = int(pairs[index, 0]), int(pairs[index, 1])
        if find(node_a) == find(node_b):
            continue
        edge = (min(node_a, node_b), max(node_a, node_b))
        if edge in existing:
            continue
        points = (
            nodes[node_a][None, :] * (1.0 - parameters[:, None])
            + nodes[node_b][None, :] * parameters[:, None]
        )
        # The same in-envelope test the bracing pass uses. A bridge that leaves
        # the optimized envelope is a bridge across a designed opening.
        if float(np.mean(_sample_envelope(envelope, points))) < 0.78:
            continue
        existing.add(edge)
        added.append(edge)
        union(node_a, node_b)

    if not added:
        return members, 0
    augmented = np.vstack((members, np.asarray(added, dtype=int)))
    return augmented, len(added)


def rasterize_member_plan(
    envelope: np.ndarray,
    plan: OptimizedMemberPlan,
) -> np.ndarray:
    """Rasterize sized circular capsules on the current recovery grid."""
    mask = np.asarray(envelope, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Member-plan rasterization requires a 3-D envelope.")
    result = np.zeros_like(mask, dtype=bool)
    current_shape = np.asarray(mask.shape, dtype=float)
    base_shape = np.asarray(plan.grid_shape, dtype=float)
    radius_scale = float(np.mean(current_shape / np.maximum(base_shape, 1.0)))
    endpoints = plan.nodes_fractional * np.maximum(current_shape - 1.0, 1.0)

    for member_index, (node_a, node_b) in enumerate(plan.members):
        a = endpoints[int(node_a)]
        b = endpoints[int(node_b)]
        radius = max(0.25, float(plan.radii_voxels[member_index]) * radius_scale)
        lower = np.maximum(0, np.floor(np.minimum(a, b) - radius - 1.0).astype(int))
        upper = np.minimum(
            np.asarray(mask.shape) - 1,
            np.ceil(np.maximum(a, b) + radius + 1.0).astype(int),
        )
        if np.any(upper < lower):
            continue
        grid = np.moveaxis(
            np.indices(tuple((upper - lower + 1).astype(int)), dtype=float),
            0,
            -1,
        )
        points = grid + lower + 0.5
        segment = b - a
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-15:
            continue
        projection = np.sum((points - a) * segment, axis=-1) / length_squared
        projection = np.clip(projection, 0.0, 1.0)
        delta = points - (a + projection[..., None] * segment)
        capsule = np.sum(delta * delta, axis=-1) <= radius**2
        slices = tuple(slice(int(lower[i]), int(upper[i]) + 1) for i in range(3))
        result[slices] |= capsule
    return result & mask
