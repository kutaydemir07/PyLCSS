# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Load-aware member sizing for explicit cubic and octet-truss lattices.

This module is the second phase of the lattice workflow.  The continuum
topology result defines the admissible envelope; an axial 3-D truss model then
sizes every retained circular member for the envelope of all structural load
cases.  Stress, Euler buckling, and an optional global displacement limit are
enforced by a damped fully-stressed-design iteration.

The model deliberately remains a beam surrogate.  Reanalysis of the recovered
solid is still required for joint bending, local stress concentrations,
contact, and manufacturing defects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import warnings
from typing import Any

import numpy as np

from ..models.study import LoadCase, VoxelBC


@dataclass(frozen=True)
class OptimizedMemberPlan:
    """Serializable geometry and diagnostics for a sized strut network."""

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
            "boundary_stabilizing_member_count": int(
                self.stabilizing_member_count
            ),
            "node_count": int(len(self.nodes_fractional)),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "maximum_stress": float(self.maximum_stress),
            "allowable_stress": float(self.allowable_stress),
            "maximum_stress_utilization": float(self.maximum_stress_utilization),
            "maximum_buckling_utilization": float(
                self.maximum_buckling_utilization
            ),
            "maximum_displacement": float(self.maximum_displacement),
            "compliance": float(self.compliance),
            "member_volume": float(self.member_volume),
            "warnings": list(self.warnings),
        }


def _node_key(point: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(round(float(value) * 1_000_000_000)) for value in point)


def _candidate_network(
    shape: tuple[int, int, int],
    mode: str,
    cell_size_voxels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a conforming unit-cell network in fractional coordinates.

    The graph comes from :mod:`.lattice_families`, which is the same edge list
    the voxel rasterizer measures its distance field against and the same one
    the analytic CAD path builds solids from. Three descriptions of one cell
    was three chances for the sized truss to stop being the geometry that gets
    manufactured.
    """
    from .lattice_families import strut_network

    dimensions = np.asarray(shape, dtype=float)
    cell_counts = np.maximum(
        1, np.ceil(dimensions / max(float(cell_size_voxels), 3.0)).astype(int)
    )
    return strut_network(mode, tuple(int(v) for v in cell_counts))


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
    if not len(members):
        return nodes[:0], members
    parameters = np.linspace(0.0, 1.0, 9)
    a = nodes[members[:, 0]]
    b = nodes[members[:, 1]]
    samples = (
        a[:, None, :] * (1.0 - parameters[None, :, None])
        + b[:, None, :] * parameters[None, :, None]
    )
    inside = _sample_envelope(envelope, samples.reshape((-1, 3))).reshape(
        (len(members), len(parameters))
    )
    # The majority rule follows a topology boundary while tolerating the
    # half-voxel error of a thresholded continuum envelope.
    members = members[np.mean(inside, axis=1) >= 0.67]
    if not len(members):
        return nodes[:0], members
    used = np.unique(members)
    remap = np.full(len(nodes), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[members]


def _stabilize_clipped_network(
    envelope: np.ndarray,
    nodes: np.ndarray,
    members: np.ndarray,
    cell_size_voxels: float,
) -> tuple[np.ndarray, int]:
    """Add short, explicit boundary braces where clipping weakens an Octet cell.

    Cutting a periodic lattice at a curved topology boundary removes some
    struts from its boundary nodes.  A node with too few independent
    connections can introduce a pin-jointed mechanism even though the
    untrimmed Octet is stretch dominated.  Add only short, in-envelope braces;
    these are returned as real members and therefore appear in the recovered
    geometry instead of acting as hidden numerical springs.
    """
    if not len(nodes) or not len(members):
        return members, 0
    from scipy.spatial import cKDTree

    scaled = (
        nodes
        * np.asarray(envelope.shape, dtype=float)
        / max(float(cell_size_voxels), 1e-9)
    )
    tree = cKDTree(scaled)
    existing = {tuple(int(value) for value in member) for member in members}
    adjacency: list[set[int]] = [set() for _ in range(len(nodes))]
    for node_a, node_b in members:
        adjacency[int(node_a)].add(int(node_b))
        adjacency[int(node_b)].add(int(node_a))
    added: list[tuple[int, int]] = []

    for node in range(len(nodes)):
        def local_rank() -> int:
            neighbours = sorted(adjacency[node])
            if not neighbours:
                return 0
            directions = scaled[neighbours] - scaled[node]
            return int(np.linalg.matrix_rank(directions, tol=1e-9))

        if len(adjacency[node]) >= 6 and local_rank() >= 3:
            continue
        distances, neighbours = tree.query(
            scaled[node],
            k=min(24, len(nodes)),
        )
        for distance, neighbour_raw in zip(
            np.atleast_1d(distances)[1:],
            np.atleast_1d(neighbours)[1:],
        ):
            neighbour = int(neighbour_raw)
            edge = (min(node, neighbour), max(node, neighbour))
            if edge in existing or float(distance) > 1.55:
                continue
            parameters = np.linspace(0.0, 1.0, 9)
            points = (
                nodes[node][None, :] * (1.0 - parameters[:, None])
                + nodes[neighbour][None, :] * parameters[:, None]
            )
            if float(np.mean(_sample_envelope(envelope, points))) < 0.78:
                continue
            existing.add(edge)
            adjacency[node].add(neighbour)
            adjacency[neighbour].add(node)
            added.append(edge)
            if len(adjacency[node]) >= 6 and local_rank() >= 3:
                break

    if not added:
        return members, 0
    augmented = np.vstack((members, np.asarray(added, dtype=int)))
    augmented = np.asarray(sorted(set(map(tuple, augmented.tolist()))), dtype=int)
    return augmented, int(len(augmented) - len(members))


def _connect_clipped_network(
    envelope: np.ndarray,
    nodes: np.ndarray,
    members: np.ndarray,
    cell_size_voxels: float,
    *,
    maximum_span_cells: float = 2.6,
) -> tuple[np.ndarray, int]:
    """Rejoin a clipped strut network that trimming broke into islands.

    Cutting a periodic network at an optimized boundary deletes whichever
    members straddle it, and where the envelope is thinner than the cell that
    leaves the survivors in separate islands. The Density view still shows one
    connected envelope, so the displayed lattice is then a strictly worse
    description of the same design: on the bundled rocker link a cell pitch of 8
    voxels left a BCC network in 12 pieces and an Octet one in 15, inside an
    envelope of 2 bodies.

    The repair is the one the conformal-lattice literature calls a net-skin:
    join the trimmed boundary nodes so the incomplete cells at the surface stop
    being loose ends. Here that is done member by member -- the shortest node
    pair bridging two islands, accepted only if the new member itself lies
    inside the envelope, so the connection never crosses a designed void and
    islands in genuinely separate envelope bodies stay separate.

    Returns the augmented member list and the number of members added.
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
    existing = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in members
    }
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


def _face_nodes(nodes: np.ndarray, face: str, tolerance: float) -> np.ndarray:
    face_key = str(face or "").strip().lower()
    axis_side = {
        "left": (0, 0.0),
        "right": (0, 1.0),
        "bottom": (1, 0.0),
        "top": (1, 1.0),
        "front": (2, 0.0),
        "back": (2, 1.0),
    }.get(face_key)
    if axis_side is None:
        return np.array([], dtype=int)
    axis, side = axis_side
    selected = np.flatnonzero(np.abs(nodes[:, axis] - side) <= tolerance)
    if selected.size:
        return selected
    distance = np.abs(nodes[:, axis] - side)
    return np.flatnonzero(distance <= float(np.min(distance)) + 1e-12)


def _box_nodes(
    nodes: np.ndarray,
    box: Any,
    tolerance: float,
) -> np.ndarray:
    values = np.asarray(box[:6], dtype=float)
    lower = np.asarray(
        (min(values[0], values[1]), min(values[2], values[3]), min(values[4], values[5]))
    )
    upper = np.asarray(
        (max(values[0], values[1]), max(values[2], values[3]), max(values[4], values[5]))
    )
    selected = np.flatnonzero(
        np.all(nodes >= lower - tolerance, axis=1)
        & np.all(nodes <= upper + tolerance, axis=1)
    )
    if selected.size:
        return selected
    center = 0.5 * (lower + upper)
    return np.asarray([int(np.argmin(np.sum((nodes - center) ** 2, axis=1)))])


def _base_fixed_dofs(
    nodes: np.ndarray,
    bc: VoxelBC,
    tolerance: float,
) -> np.ndarray:
    fixed: list[int] = []
    face_fields = (
        ("left", bc.fixed_left_face_dofs),
        ("right", bc.fixed_right_face_dofs),
        ("top", bc.fixed_top_face_dofs),
        ("bottom", bc.fixed_bottom_face_dofs),
        ("front", bc.fixed_front_face_dofs),
        ("back", bc.fixed_back_face_dofs),
    )
    for face, components in face_fields:
        for node in _face_nodes(nodes, face, tolerance):
            fixed.extend(3 * int(node) + int(dof) for dof in components)
    for box in bc.fixed_boxes:
        for node in _box_nodes(nodes, box, tolerance):
            fixed.extend(3 * int(node) + int(dof) for dof in box[6])
    return np.unique(fixed).astype(int)


def _case_force(
    nodes: np.ndarray,
    case: LoadCase,
    tolerance: float,
) -> np.ndarray:
    force = np.zeros(3 * len(nodes), dtype=float)
    for point in case.point_forces:
        location = np.asarray(point[:3], dtype=float)
        node = int(np.argmin(np.sum((nodes - location) ** 2, axis=1)))
        force[3 * node : 3 * node + 3] += np.asarray(point[3:6], dtype=float)
    for box in case.box_forces:
        selected = _box_nodes(nodes, box, tolerance)
        share = np.asarray(box[6:9], dtype=float) / max(1, len(selected))
        for node in selected:
            force[3 * int(node) : 3 * int(node) + 3] += share
    for face, fx, fy, fz in case.distributed_forces:
        selected = _face_nodes(nodes, face, tolerance)
        share = np.asarray((fx, fy, fz), dtype=float) / max(1, len(selected))
        for node in selected:
            force[3 * int(node) : 3 * int(node) + 3] += share
    return force


def _analysis_cases(
    nodes: np.ndarray,
    bc: VoxelBC,
    tolerance: float,
) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    base_fixed = _base_fixed_dofs(nodes, bc, tolerance)
    cases = list(bc.load_cases)
    if not cases:
        cases = [
            LoadCase(
                name="LC1",
                weight=1.0,
                point_forces=list(bc.point_forces),
                box_forces=list(bc.box_forces),
                distributed_forces=list(bc.distributed_forces),
            )
        ]
    result: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    for case in cases:
        force = _case_force(nodes, case, tolerance)
        if not np.any(np.abs(force) > 1e-15):
            continue
        if case.fixed_boxes:
            local_fixed: list[int] = []
            for box in case.fixed_boxes:
                for node in _box_nodes(nodes, box, tolerance):
                    local_fixed.extend(
                        3 * int(node) + int(dof) for dof in box[6]
                    )
            local = np.unique(local_fixed).astype(int)
            fixed = (
                local
                if case.replace_supports
                else np.unique(np.concatenate((base_fixed, local))).astype(int)
            )
        elif case.replace_supports:
            fixed = np.array([], dtype=int)
        else:
            fixed = base_fixed
        result.append((case.name, float(case.weight), force, fixed))
    return result


def _retain_load_path_components(
    nodes: np.ndarray,
    members: np.ndarray,
    bc: VoxelBC,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discard disconnected islands that cannot link a support to a load."""
    if not len(members):
        return nodes, members
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate((members[:, 0], members[:, 1]))
    cols = np.concatenate((members[:, 1], members[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes))
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    cases = _analysis_cases(nodes, bc, tolerance)
    keep_labels: set[int] = set()
    for _name, _weight, force, fixed in cases:
        load_nodes = np.unique(np.flatnonzero(np.abs(force) > 1e-15) // 3)
        fixed_nodes = np.unique(np.asarray(fixed, dtype=int) // 3)
        keep_labels.update(set(labels[load_nodes]).intersection(labels[fixed_nodes]))
    if not keep_labels:
        return nodes, members
    node_keep = np.isin(labels, np.asarray(sorted(keep_labels)))
    member_keep = node_keep[members[:, 0]] & node_keep[members[:, 1]]
    members = members[member_keep]
    used = np.unique(members)
    remap = np.full(len(nodes), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[members]


def _assemble_stiffness(
    nodes_physical: np.ndarray,
    members: np.ndarray,
    areas: np.ndarray,
    youngs_modulus: float,
):
    from scipy.sparse import coo_matrix

    a = nodes_physical[members[:, 0]]
    b = nodes_physical[members[:, 1]]
    vectors = b - a
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for member_index, (node_a, node_b) in enumerate(members):
        direction = directions[member_index]
        block = (
            youngs_modulus
            * areas[member_index]
            / max(lengths[member_index], 1e-12)
            * np.outer(direction, direction)
        )
        dofs_a = 3 * int(node_a) + np.arange(3)
        dofs_b = 3 * int(node_b) + np.arange(3)
        dofs = np.concatenate((dofs_a, dofs_b))
        local = np.block([[block, -block], [-block, block]])
        row_grid, col_grid = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(row_grid.ravel())
        cols.extend(col_grid.ravel())
        values.extend(local.ravel())
    stiffness = coo_matrix(
        (values, (rows, cols)),
        shape=(3 * len(nodes_physical), 3 * len(nodes_physical)),
    ).tocsr()
    return stiffness, lengths, directions


def _solve_cases(
    stiffness: Any,
    cases: list[tuple[str, float, np.ndarray, np.ndarray]],
    nodes_physical: np.ndarray,
    members: np.ndarray,
    areas: np.ndarray,
    lengths: np.ndarray,
    directions: np.ndarray,
    youngs_modulus: float,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    from scipy.sparse import eye
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    member_force = np.zeros((len(cases), len(members)), dtype=float)
    displacements = np.zeros((len(cases), 3 * len(nodes_physical)), dtype=float)
    max_displacement = 0.0
    compliance = 0.0
    stable = True
    diagonal = np.asarray(stiffness.diagonal(), dtype=float)
    regularization = max(float(np.max(np.abs(diagonal))), 1.0) * 1e-11

    for case_index, (_name, weight, force, fixed) in enumerate(cases):
        free = np.setdiff1d(
            np.arange(stiffness.shape[0], dtype=int),
            np.asarray(fixed, dtype=int),
            assume_unique=False,
        )
        if not len(free) or not len(fixed):
            stable = False
            continue
        reduced = stiffness[free][:, free] + regularization * eye(
            len(free), format="csr"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MatrixRankWarning)
            solved = spsolve(reduced, force[free])
        if any(issubclass(item.category, MatrixRankWarning) for item in caught):
            stable = False
        if not np.all(np.isfinite(solved)):
            stable = False
            solved = np.nan_to_num(solved, nan=0.0, posinf=0.0, neginf=0.0)
        displacement = displacements[case_index]
        displacement[free] = solved
        residual = stiffness[free][:, free] @ solved - force[free]
        relative_residual = float(
            np.linalg.norm(residual)
            / max(float(np.linalg.norm(force[free])), 1e-15)
        )
        if relative_residual > 1e-5:
            stable = False
        max_displacement = max(
            max_displacement,
            float(np.max(np.linalg.norm(displacement.reshape((-1, 3)), axis=1))),
        )
        compliance += float(weight) * float(np.dot(force, displacement))
        relative = (
            displacement.reshape((-1, 3))[members[:, 1]]
            - displacement.reshape((-1, 3))[members[:, 0]]
        )
        axial_extension = np.einsum("ij,ij->i", relative, directions)
        member_force[case_index] = (
            youngs_modulus * areas / np.maximum(lengths, 1e-12) * axial_extension
        )
    return member_force, displacements, max_displacement, compliance, stable


def optimize_lattice_members(
    envelope: np.ndarray,
    *,
    mode: str,
    cell_size_voxels: float,
    minimum_diameter_voxels: float,
    maximum_diameter_voxels: float,
    bc: VoxelBC,
    span: tuple[float, float, float],
    youngs_modulus: float,
    allowable_stress: float,
    displacement_limit: float = 0.0,
    buckling_length_factor: float = 1.0,
    maximum_iterations: int = 35,
) -> OptimizedMemberPlan:
    """Size a strut network inside ``envelope`` for all load cases."""
    from .lattice_families import FAMILIES, STRUT_FAMILIES, normalize_family_key

    mode_key = normalize_family_key(mode)
    if mode_key not in STRUT_FAMILIES:
        raise ValueError(
            "Member sizing is available for the strut lattices only: "
            + ", ".join(
                FAMILIES[key].display_name for key in STRUT_FAMILIES
            )
            + "."
        )
    field = np.asarray(envelope, dtype=bool)
    if field.ndim != 3 or not np.any(field):
        raise ValueError("Member sizing requires a non-empty 3-D topology envelope.")
    shape = tuple(int(value) for value in field.shape)
    span_array = np.asarray(span, dtype=float)
    if span_array.shape != (3,) or np.any(~np.isfinite(span_array)) or np.any(span_array <= 0):
        raise ValueError("Member sizing requires three positive physical spans.")
    youngs_modulus = float(youngs_modulus)
    allowable_stress = float(allowable_stress)
    if youngs_modulus <= 0.0 or allowable_stress <= 0.0:
        raise ValueError("Young's modulus and allowable stress must be positive.")
    min_diameter = float(minimum_diameter_voxels)
    max_diameter = float(maximum_diameter_voxels)
    if not (0.0 < min_diameter <= max_diameter):
        raise ValueError("Member diameters require 0 < minimum <= maximum.")
    buckling_length_factor = float(buckling_length_factor)
    if buckling_length_factor <= 0.0:
        raise ValueError("Buckling length factor must be positive.")
    maximum_iterations = max(1, int(maximum_iterations))

    nodes, members = _candidate_network(shape, mode_key, cell_size_voxels)
    nodes, members = _clip_network_to_envelope(field, nodes, members)
    if len(members) < 3:
        raise ValueError(
            "The selected topology envelope contains too few resolved lattice members. "
            "Reduce cell size or use a finer topology grid."
        )
    stabilizing_member_count = 0
    # Only a stretch-dominated network is worth bracing back to stretch
    # dominance after clipping. A bending-dominated family was never rigid
    # under pin joints in the first place, so adding boundary braces would
    # misrepresent it rather than repair it.
    if FAMILIES[mode_key].maxwell == "stretch":
        members, stabilizing_member_count = _stabilize_clipped_network(
            field,
            nodes,
            members,
            cell_size_voxels,
        )
    tolerance = max(
        0.01,
        0.75 * float(cell_size_voxels) / max(float(max(shape)), 1.0),
    )
    nodes, members = _retain_load_path_components(
        nodes, members, bc, tolerance
    )
    cases = _analysis_cases(nodes, bc, tolerance)
    if not cases:
        raise ValueError("Member sizing found no non-zero structural load case.")
    if not all(len(case[3]) for case in cases):
        raise ValueError("Every lattice sizing load case requires at least one support.")

    voxel_pitch = float(np.mean(span_array / np.asarray(shape, dtype=float)))
    minimum_area = math.pi * (0.5 * min_diameter * voxel_pitch) ** 2
    maximum_area = math.pi * (0.5 * max_diameter * voxel_pitch) ** 2
    areas = np.full(len(members), minimum_area, dtype=float)
    nodes_physical = nodes * span_array
    warning_messages: list[str] = []
    converged = False
    iteration = 0

    for iteration in range(1, maximum_iterations + 1):
        stiffness, lengths, directions = _assemble_stiffness(
            nodes_physical, members, areas, youngs_modulus
        )
        member_force, _u, maximum_displacement, _compliance, stable = _solve_cases(
            stiffness,
            cases,
            nodes_physical,
            members,
            areas,
            lengths,
            directions,
            youngs_modulus,
        )
        if not stable and "The axial truss contains a mechanism or near-mechanism." not in warning_messages:
            warning_messages.append(
                "The axial truss contains a mechanism or near-mechanism."
            )
        force_envelope = np.max(np.abs(member_force), axis=0)
        compression = np.max(np.maximum(-member_force, 0.0), axis=0)
        required_stress_area = force_envelope / allowable_stress
        required_buckling_area = np.sqrt(
            4.0
            * compression
            * (buckling_length_factor * lengths) ** 2
            / (math.pi * youngs_modulus)
        )
        required = np.maximum.reduce(
            (
                np.full_like(areas, minimum_area),
                required_stress_area,
                required_buckling_area,
            )
        )
        if displacement_limit > 0.0 and maximum_displacement > displacement_limit:
            required = np.maximum(
                required,
                areas * maximum_displacement / float(displacement_limit),
            )
        required = np.clip(required, minimum_area, maximum_area)
        damped = 0.55 * areas + 0.45 * required

        # Light nodal averaging prevents isolated diameter jumps that create
        # poor unions and unrealistic joint stiffness in the recovered solid.
        incident_sum = np.zeros(len(nodes), dtype=float)
        incident_count = np.zeros(len(nodes), dtype=float)
        for end in (0, 1):
            np.add.at(incident_sum, members[:, end], damped)
            np.add.at(incident_count, members[:, end], 1.0)
        nodal_mean = incident_sum / np.maximum(incident_count, 1.0)
        neighbour_mean = 0.5 * (
            nodal_mean[members[:, 0]] + nodal_mean[members[:, 1]]
        )
        smoothed = 0.90 * damped + 0.10 * neighbour_mean
        # Joint smoothing may increase a lightly loaded neighbour, but it must
        # never undo the local stress/buckling/displacement sizing update.
        updated = np.clip(
            np.maximum(smoothed, damped),
            minimum_area,
            maximum_area,
        )
        relative_change = float(
            np.max(np.abs(updated - areas) / np.maximum(areas, minimum_area))
        )
        areas = updated
        stress_ok = bool(
            np.max(required_stress_area / np.maximum(areas, 1e-15)) <= 1.005
        )
        buckling_ok = bool(
            np.max(required_buckling_area / np.maximum(areas, 1e-15)) <= 1.005
        )
        displacement_ok = bool(
            displacement_limit <= 0.0
            or maximum_displacement <= 1.005 * displacement_limit
        )
        if (
            relative_change < 2.5e-3
            and stress_ok
            and buckling_ok
            and displacement_ok
        ):
            converged = True
            break

    stiffness, lengths, directions = _assemble_stiffness(
        nodes_physical, members, areas, youngs_modulus
    )
    member_force, _u, maximum_displacement, compliance, stable = _solve_cases(
        stiffness,
        cases,
        nodes_physical,
        members,
        areas,
        lengths,
        directions,
        youngs_modulus,
    )
    if not stable and "The axial truss contains a mechanism or near-mechanism." not in warning_messages:
        warning_messages.append("The axial truss contains a mechanism or near-mechanism.")
    maximum_stress = float(
        np.max(np.abs(member_force) / np.maximum(areas[None, :], 1e-15))
    )
    stress_utilization = maximum_stress / allowable_stress
    compression = np.max(np.maximum(-member_force, 0.0), axis=0)
    second_moment = areas**2 / (4.0 * math.pi)
    critical_force = (
        math.pi**2
        * youngs_modulus
        * second_moment
        / np.maximum((buckling_length_factor * lengths) ** 2, 1e-15)
    )
    buckling_utilization = float(
        np.max(compression / np.maximum(critical_force, 1e-15))
    )
    if stress_utilization > 1.005:
        if bool(np.any(areas >= 0.999 * maximum_area)):
            warning_messages.append(
                "At least one member reaches the maximum diameter before "
                "satisfying the allowable-stress constraint."
            )
        else:
            warning_messages.append(
                "The member-sizing iteration did not satisfy the "
                "allowable-stress constraint."
            )
    if buckling_utilization > 1.005:
        if bool(np.any(areas >= 0.999 * maximum_area)):
            warning_messages.append(
                "At least one compression member reaches the maximum diameter "
                "before satisfying Euler buckling."
            )
        else:
            warning_messages.append(
                "The member-sizing iteration did not satisfy Euler buckling."
            )
    if displacement_limit > 0.0 and maximum_displacement > 1.005 * displacement_limit:
        warning_messages.append(
            "The maximum member diameter is insufficient for the displacement limit."
        )
    if mode_key == "cubic":
        warning_messages.append(
            "Cubic cells are weak in shear; use Octet Truss for load-bearing parts."
        )

    diameters_voxels = (
        2.0 * np.sqrt(np.maximum(areas, 0.0) / math.pi) / voxel_pitch
    )
    return OptimizedMemberPlan(
        mode=mode_key,
        grid_shape=shape,
        nodes_fractional=np.asarray(nodes, dtype=float),
        members=np.asarray(members, dtype=int),
        radii_voxels=0.5 * diameters_voxels,
        iterations=iteration,
        converged=converged,
        maximum_stress=maximum_stress,
        maximum_stress_utilization=float(stress_utilization),
        maximum_buckling_utilization=buckling_utilization,
        maximum_displacement=float(maximum_displacement),
        compliance=float(compliance),
        member_volume=float(np.sum(areas * lengths)),
        allowable_stress=allowable_stress,
        stabilizing_member_count=stabilizing_member_count,
        warnings=tuple(dict.fromkeys(warning_messages)),
    )


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
