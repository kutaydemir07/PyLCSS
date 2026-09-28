# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Boundary, load-case, joint, and passive-mask assembly."""

from __future__ import annotations

import logging

import numpy as np

from ..models.study import JointDefinition, LoadCase, ThermalBC, VoxelBC
from .pymoto_runtime import PyMotoDomain
from .results import _StructuralCase, _ThermalCase

logger = logging.getLogger(__name__)


class _SolverAssemblyMixin:
    """Assembly operations shared by the topology solver orchestration."""

    def _assemble_supports(
        self,
        domain: PyMotoDomain,
        bc: VoxelBC,
    ) -> np.ndarray:
        """Build the global fixed-DOF array from VoxelBC face/box supports.

        VoxelDomain node grid (3-D):
            domain.nodes[ix, iy, iz]  →  scalar node number
            ix = 0 … nelx  (left  → right,  X)
            iy = 0 … nely  (bottom → top,   Y)
            iz = 0 … nelz  (front → back,   Z)
        DOF layout:  x-DOF = 3*n,  y-DOF = 3*n+1,  z-DOF = 3*n+2
        """
        p = self.problem
        dim = domain.dim
        fixed: list[int] = []

        def _add_face_dofs(nodes_2d: np.ndarray, dofs: list[int]) -> None:
            fixed.extend(
                domain.get_dofnumber(nodes_2d.flatten(), dofs, ndof=dim)
                .flatten()
                .astype(int)
                .tolist()
            )

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i + 1)

        if bc.fixed_left_face_dofs:
            _add_face_dofs(domain.nodes[0, :, :], bc.fixed_left_face_dofs)
        if bc.fixed_right_face_dofs:
            _add_face_dofs(domain.nodes[p.nelx, :, :], bc.fixed_right_face_dofs)
        if bc.fixed_top_face_dofs:
            _add_face_dofs(domain.nodes[:, p.nely, :], bc.fixed_top_face_dofs)
        if bc.fixed_bottom_face_dofs:
            _add_face_dofs(domain.nodes[:, 0, :], bc.fixed_bottom_face_dofs)
        if bc.fixed_front_face_dofs:
            _add_face_dofs(domain.nodes[:, :, 0], bc.fixed_front_face_dofs)
        if bc.fixed_back_face_dofs:
            _add_face_dofs(domain.nodes[:, :, p.nelz], bc.fixed_back_face_dofs)

        for xmin, xmax, ymin, ymax, zmin, zmax, dofs in bc.fixed_boxes:
            if not dofs:
                continue
            nodes = domain.nodes[
                _node_slice(xmin, xmax, p.nelx),
                _node_slice(ymin, ymax, p.nely),
                _node_slice(zmin, zmax, p.nelz),
            ]
            _add_face_dofs(nodes, dofs)

        return np.unique(fixed).astype(int)

    def _apply_load_case_to_force(
        self,
        domain: PyMotoDomain,
        lc: LoadCase,
        f: np.ndarray,
    ) -> None:
        """Populate the force vector with point, patch, and face loads."""
        p = self.problem
        dim = domain.dim

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            if hi_i <= lo_i:
                hi_i = min(nmax, lo_i + 1)
            return slice(lo_i, hi_i + 1)

        for ix_frac, iy_frac, iz_frac, fx, fy, fz in lc.point_forces:
            ix = int(round(ix_frac * p.nelx))
            iy = int(round(iy_frac * p.nely))
            iz = int(round(iz_frac * p.nelz))
            n = int(domain.nodes[ix, iy, iz])
            f[domain.get_dofnumber(n, 0, ndof=dim)] += fx
            f[domain.get_dofnumber(n, 1, ndof=dim)] += fy
            f[domain.get_dofnumber(n, 2, ndof=dim)] += fz

        def _spread_total_force(
            nodes: np.ndarray,
            components: tuple[float, float, float],
        ) -> None:
            """Apply a total force over a node block with consistent shares.

            ``nodes`` keeps its structured shape, because the share a node takes
            is decided by how many loaded elements touch it. Integrating a
            uniform traction against the trilinear shape functions puts t*A/4 on
            each corner of every loaded element face, so after assembly an
            interior node carries twice an edge node and four times a corner
            node, per in-plane axis.

            Splitting the total equally over the selected nodes instead — which
            is what this did — overloads the perimeter of the loaded region: on
            a 40x20 element face a corner node took 3.7x and an edge node 1.9x
            its correct share, while every interior node was 7% light. SIMP
            answers a load it believes is concentrated at the rim by growing
            material there, so the artefact ends up in the topology.
            """
            block = np.asarray(nodes, dtype=int)
            if block.size == 0:
                return
            weights = np.ones(block.shape, dtype=float)
            for axis, count in enumerate(block.shape):
                axis_weights = np.ones(int(count), dtype=float)
                if count > 2:
                    axis_weights[1:-1] = 2.0
                shape = [1] * block.ndim
                shape[axis] = int(count)
                weights = weights * axis_weights.reshape(shape)
            total = float(np.sum(weights))
            if total <= 0.0:
                return
            weights = (weights / total).ravel()
            flat_nodes = block.ravel()
            for component, value in enumerate(components):
                if abs(float(value)) <= 0.0:
                    continue
                np.add.at(
                    f,
                    domain.get_dofnumber(flat_nodes, component, ndof=dim),
                    float(value) * weights,
                )

        for x0, x1, y0, y1, z0, z1, fx, fy, fz in lc.box_forces:
            _spread_total_force(
                domain.nodes[
                    _node_slice(x0, x1, p.nelx),
                    _node_slice(y0, y1, p.nely),
                    _node_slice(z0, z1, p.nelz),
                ],
                (fx, fy, fz),
            )

        # Kept as 2-D node blocks: the tributary weighting needs the face
        # topology, which a flattened list no longer carries.
        _face_nodes = {
            "left": domain.nodes[0, :, :],
            "right": domain.nodes[p.nelx, :, :],
            "top": domain.nodes[:, p.nely, :],
            "bottom": domain.nodes[:, 0, :],
            "front": domain.nodes[:, :, 0],
            "back": domain.nodes[:, :, p.nelz],
        }
        for face, fx_per, fy_per, fz_per in lc.distributed_forces:
            # Matched case-insensitively, because that is how the load case
            # validated it. A "Left" that passed validation and then missed
            # this lookup applied no force at all, and the study reported a
            # compliance for a structure nothing was pushing on.
            nodes = _face_nodes.get(str(face).strip().lower())
            if nodes is None or np.size(nodes) == 0:
                logger.warning(
                    "Load case %r references unknown face %r; no force applied.",
                    lc.name,
                    face,
                )
                continue
            _spread_total_force(nodes, (fx_per, fy_per, fz_per))

    def _assemble_load_cases(
        self,
        domain: PyMotoDomain,
        bc: VoxelBC,
        ndof: int,
        base_boundary_dofs: np.ndarray | None = None,
    ) -> list[_StructuralCase]:
        """Assemble structural cases, including pose-local supports and joints."""
        base_dofs = (
            self._assemble_supports(domain, bc)
            if base_boundary_dofs is None
            else np.asarray(base_boundary_dofs, dtype=int)
        )
        cases: list[_StructuralCase] = []

        for lc in bc.load_cases:
            f = np.zeros(ndof)
            self._apply_load_case_to_force(domain, lc, f)
            if np.any(f):
                case_dofs = self._assemble_case_supports(domain, lc, base_dofs)
                joint_defs = (
                    list(lc.joints) if lc.replace_joints else [*bc.joints, *lc.joints]
                )
                cases.append(
                    _StructuralCase(
                        name=str(lc.name or "LC"),
                        weight=float(lc.weight or 1.0),
                        force=f,
                        boundary_dofs=case_dofs,
                        joint_stiffness=self._assemble_joint_stiffness(
                            domain, joint_defs, case_dofs
                        ),
                    )
                )

        if not cases:
            legacy_lc = LoadCase(
                name="LC1",
                weight=1.0,
                point_forces=list(bc.point_forces),
                box_forces=list(bc.box_forces),
                distributed_forces=list(bc.distributed_forces),
            )
            f = np.zeros(ndof)
            self._apply_load_case_to_force(domain, legacy_lc, f)
            if np.any(f):
                cases.append(
                    _StructuralCase(
                        name="LC1",
                        weight=1.0,
                        force=f,
                        boundary_dofs=base_dofs,
                        joint_stiffness=self._assemble_joint_stiffness(
                            domain, bc.joints, base_dofs
                        ),
                    )
                )

        return cases

    def _assemble_case_supports(
        self,
        domain: PyMotoDomain,
        load_case: LoadCase,
        base_boundary_dofs: np.ndarray,
    ) -> np.ndarray:
        """Combine or replace global supports for a moving assembly pose."""
        if not load_case.fixed_boxes and not load_case.replace_supports:
            return np.asarray(base_boundary_dofs, dtype=int)
        if not load_case.fixed_boxes:
            return np.array([], dtype=int)
        local_bc = VoxelBC(fixed_boxes=list(load_case.fixed_boxes))
        local = self._assemble_supports(domain, local_bc)
        if load_case.replace_supports:
            return local
        return np.unique(
            np.concatenate([np.asarray(base_boundary_dofs, dtype=int), local])
        ).astype(int)

    def _assemble_joint_stiffness(
        self,
        domain: PyMotoDomain,
        joints: list[JointDefinition],
        boundary_dofs: np.ndarray | None = None,
    ) -> object | None:
        """Build sparse translational penalty couplings for multibody joints."""
        if not joints:
            return None
        from scipy.sparse import coo_matrix

        p = self.problem
        ndof = int(domain.nnodes * domain.dim)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        axis_dof = {"x": 0, "y": 1, "z": 2}
        characteristic = float(p.E0) * min(
            float(p.unitx), float(p.unity), float(p.unitz)
        )
        source_indices = None
        try:
            source_grid = np.asarray(p.design_domain, dtype=bool)
            if source_grid.shape == (p.nelx, p.nely, p.nelz):
                source_indices = np.argwhere(source_grid)
        except Exception:
            source_indices = None

        def _node_at(anchor: tuple[float, float, float]) -> int:
            ix = int(np.clip(round(anchor[0] * p.nelx), 0, p.nelx))
            iy = int(np.clip(round(anchor[1] * p.nely), 0, p.nely))
            iz = int(np.clip(round(anchor[2] * p.nelz), 0, p.nelz))
            return int(domain.nodes[ix, iy, iz])

        def _attachment_nodes(
            anchor: tuple[float, float, float],
            center_node: int,
            count: int = 6,
        ) -> list[int]:
            """Find source-body nodes around a bore-center joint anchor.

            A joint anchor is normally located in the intentionally void bore.
            Coupling only that central low-density voxel leaves the kinematic
            spring floating and lets an entire link disappear.  A small
            penalty-spider to nearby source-body nodes transfers the joint
            force into the preserved bearing sleeve.
            """
            if source_indices is None or not len(source_indices):
                return []
            anchor_arr = np.asarray(anchor, dtype=float)[:3]
            centers = (source_indices.astype(float) + 0.5) / np.asarray(
                [p.nelx, p.nely, p.nelz], dtype=float
            )
            distances = np.linalg.norm(centers - anchor_arr[None, :], axis=1)
            nearest_elements = source_indices[
                np.argsort(distances)[: min(24, len(source_indices))]
            ]
            candidates: set[int] = set()
            for ix, iy, iz in nearest_elements:
                block = domain.nodes[
                    int(ix) : int(ix) + 2,
                    int(iy) : int(iy) + 2,
                    int(iz) : int(iz) + 2,
                ]
                candidates.update(int(value) for value in np.asarray(block).ravel())
            candidates.discard(int(center_node))
            if not candidates:
                return []

            def _distance(node_number: int) -> float:
                location = np.argwhere(domain.nodes == int(node_number))
                if not len(location):
                    return float("inf")
                normalized = location[0].astype(float) / np.asarray(
                    [p.nelx, p.nely, p.nelz], dtype=float
                )
                return float(np.linalg.norm(normalized - anchor_arr))

            return sorted(candidates, key=_distance)[: max(1, int(count))]

        def _add_penalty_pair(
            node_a: int,
            node_b: int,
            components: list[int],
            stiffness: float,
        ) -> None:
            if int(node_a) == int(node_b):
                return
            for component in components:
                ia = int(domain.get_dofnumber(int(node_a), component, ndof=domain.dim))
                ib = int(domain.get_dofnumber(int(node_b), component, ndof=domain.dim))
                rows.extend([ia, ia, ib, ib])
                cols.extend([ia, ib, ia, ib])
                data.extend([stiffness, -stiffness, -stiffness, stiffness])

        def _add_directional_penalty_pair(
            node_a: int,
            node_b: int,
            direction: np.ndarray,
            stiffness: float,
        ) -> None:
            """Add k(n·(u_a-u_b))² without locking tangential motion."""
            vector = np.asarray(direction, dtype=float).reshape(3)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12 or int(node_a) == int(node_b):
                return
            vector /= norm
            block = float(stiffness) * np.outer(vector, vector)
            dofs_a = [
                int(domain.get_dofnumber(int(node_a), component, ndof=domain.dim))
                for component in range(3)
            ]
            dofs_b = [
                int(domain.get_dofnumber(int(node_b), component, ndof=domain.dim))
                for component in range(3)
            ]
            for i in range(3):
                for j in range(3):
                    value = float(block[i, j])
                    if abs(value) <= 1e-18:
                        continue
                    rows.extend([dofs_a[i], dofs_a[i], dofs_b[i], dofs_b[i]])
                    cols.extend([dofs_a[j], dofs_b[j], dofs_a[j], dofs_b[j]])
                    data.extend([value, -value, -value, value])

        domain_span = np.asarray(
            [
                float(p.nelx) * float(p.unitx),
                float(p.nely) * float(p.unity),
                float(p.nelz) * float(p.unitz),
            ],
            dtype=float,
        )

        def _node_offset_from_anchor(
            node_number: int,
            anchor: tuple[float, float, float],
        ) -> np.ndarray:
            location = np.argwhere(domain.nodes == int(node_number))
            if not len(location):
                return np.zeros(3, dtype=float)
            fractional = location[0].astype(float) / np.asarray(
                [p.nelx, p.nely, p.nelz], dtype=float
            )
            return (fractional - np.asarray(anchor, dtype=float)[:3]) * domain_span

        for joint in joints:
            node_a = _node_at(joint.anchor_a)
            node_b = _node_at(joint.anchor_b)
            if node_a == node_b:
                logger.warning(
                    "Multibody joint %r maps both anchors to the same voxel node; "
                    "increase grid resolution or separate the anchors.",
                    joint.name,
                )
                continue
            coupled_dofs = [0, 1, 2]
            if joint.kind == "prismatic":
                released = axis_dof[joint.axis]
                coupled_dofs = [d for d in coupled_dofs if d != released]
            stiffness = characteristic * float(joint.relative_stiffness)
            _add_penalty_pair(node_a, node_b, coupled_dofs, stiffness)
            for center, anchor in (
                (node_a, joint.anchor_a),
                (node_b, joint.anchor_b),
            ):
                attachments = _attachment_nodes(anchor, center)
                if not attachments:
                    logger.warning(
                        "Multibody joint %r has no source-body attachment "
                        "nodes near one anchor.",
                        joint.name,
                    )
                    continue
                share = stiffness / float(len(attachments))
                for attachment in attachments:
                    if joint.kind == "fixed":
                        _add_penalty_pair(
                            center,
                            attachment,
                            [0, 1, 2],
                            share,
                        )
                        continue

                    offset = _node_offset_from_anchor(attachment, anchor)
                    axis_vector = np.zeros(3, dtype=float)
                    axis_vector[axis_dof[joint.axis]] = 1.0
                    radial = offset - np.dot(offset, axis_vector) * axis_vector

                    if joint.kind == "spherical":
                        # Radial spokes transmit bearing force but perform no
                        # work under an infinitesimal rigid-body rotation.
                        _add_directional_penalty_pair(center, attachment, offset, share)
                    elif joint.kind == "revolute":
                        # A cylindrical bearing constrains radial and axial
                        # separation while leaving circumferential motion free.
                        _add_directional_penalty_pair(center, attachment, radial, share)
                        _add_directional_penalty_pair(
                            center, attachment, axis_vector, share
                        )
                    elif joint.kind == "prismatic":
                        # The guide constrains transverse separation and
                        # releases translation along its declared axis.
                        _add_directional_penalty_pair(center, attachment, radial, share)
        if not data:
            return None
        matrix = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tolil()
        fixed = np.asarray(
            [] if boundary_dofs is None else boundary_dofs,
            dtype=int,
        ).ravel()
        if fixed.size:
            matrix[fixed, :] = 0.0
            matrix[:, fixed] = 0.0
        result = matrix.tocsr()
        result.eliminate_zeros()
        return result

    def _assemble_thermal_supports(
        self,
        domain: PyMotoDomain,
        thermal_bc: ThermalBC,
    ) -> np.ndarray:
        """Return scalar node indices held at the reference temperature."""
        p = self.problem
        face_nodes = {
            "left": domain.nodes[0, :, :],
            "right": domain.nodes[p.nelx, :, :],
            "top": domain.nodes[:, p.nely, :],
            "bottom": domain.nodes[:, 0, :],
            "front": domain.nodes[:, :, 0],
            "back": domain.nodes[:, :, p.nelz],
        }
        nodes: list[int] = []
        for face in thermal_bc.fixed_faces:
            selected = face_nodes.get(str(face).lower())
            if selected is not None:
                nodes.extend(np.asarray(selected, dtype=int).ravel().tolist())

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i, hi_i = sorted(
                (
                    int(round(float(lo) * nmax)),
                    int(round(float(hi) * nmax)),
                )
            )
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i + 1)

        for x0, x1, y0, y1, z0, z1 in thermal_bc.fixed_boxes:
            selected = domain.nodes[
                _node_slice(x0, x1, p.nelx),
                _node_slice(y0, y1, p.nely),
                _node_slice(z0, z1, p.nelz),
            ]
            nodes.extend(np.asarray(selected, dtype=int).ravel().tolist())
        return np.unique(nodes).astype(int)

    def _assemble_thermal_load_cases(
        self,
        domain: PyMotoDomain,
        thermal_bc: ThermalBC,
        fixed_nodes: np.ndarray,
    ) -> list[_ThermalCase]:
        """Assemble total heat inputs on scalar nodal temperature DOFs."""
        p = self.problem
        face_nodes = {
            "left": domain.nodes[0, :, :],
            "right": domain.nodes[p.nelx, :, :],
            "top": domain.nodes[:, p.nely, :],
            "bottom": domain.nodes[:, 0, :],
            "front": domain.nodes[:, :, 0],
            "back": domain.nodes[:, :, p.nelz],
        }

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i, hi_i = sorted(
                (
                    int(round(float(lo) * nmax)),
                    int(round(float(hi) * nmax)),
                )
            )
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i + 1)

        cases: list[_ThermalCase] = []
        for load_case in thermal_bc.load_cases:
            heat = np.zeros(int(domain.nnodes), dtype=float)
            for x, y, z, total in load_case.point_sources:
                ix = int(np.clip(round(x * p.nelx), 0, p.nelx))
                iy = int(np.clip(round(y * p.nely), 0, p.nely))
                iz = int(np.clip(round(z * p.nelz), 0, p.nelz))
                heat[int(domain.nodes[ix, iy, iz])] += float(total)
            for x0, x1, y0, y1, z0, z1, total in load_case.box_sources:
                selected = np.unique(
                    np.asarray(
                        domain.nodes[
                            _node_slice(x0, x1, p.nelx),
                            _node_slice(y0, y1, p.nely),
                            _node_slice(z0, z1, p.nelz),
                        ],
                        dtype=int,
                    ).ravel()
                )
                if selected.size:
                    heat[selected] += float(total) / float(selected.size)
            for face, total in load_case.distributed_sources:
                selected = np.unique(
                    np.asarray(face_nodes.get(str(face).lower(), []), dtype=int).ravel()
                )
                if selected.size:
                    heat[selected] += float(total) / float(selected.size)
            # A prescribed-temperature node cannot simultaneously carry an
            # independent heat source in this zero-Dirichlet formulation.
            heat[np.asarray(fixed_nodes, dtype=int)] = 0.0
            if np.any(np.abs(heat) > 1e-15):
                cases.append(
                    _ThermalCase(
                        name=load_case.name,
                        weight=load_case.weight,
                        heat=heat,
                    )
                )
        return cases

    def _assemble_passive_masks(
        self,
        domain: PyMotoDomain,
        bc: VoxelBC,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (active_mask, passive_density) over the flat element vector.

        Voxels in `bc.solid_boxes` are clamped to ρ=1 (must-keep material).
        Voxels in `bc.void_boxes`  are clamped to ρ=1e-3 (must-be-empty).
        Explicit passive solids are not clipped by source-mask voxelisation;
        this preserves selected rings/sleeves even when the coarse source grid
        misses a thin wall. Void regions still run last and win in overlaps.

        The automatic load/support pads are a different case and *are* clipped
        to the design domain. They are generated by centring a symmetric box on
        the selected entity, so a BC on a domain boundary puts half the box
        outside the part — on the L-bracket benchmark that froze solid material
        above the re-entrant corner, in the quadrant the geometry cuts away.
        The user asked for a pad on their part, not for new material beside it.
        """
        p = self.problem
        n_total = int(domain.nel)
        active_mask = np.ones(n_total, dtype=bool)
        passive_density = np.zeros(n_total, dtype=float)

        source_active = None
        if p.design_domain is not None:
            try:
                source_grid = np.asarray(p.design_domain, dtype=bool)
                if source_grid.shape == (p.nelx, p.nely, p.nelz):
                    source_active = np.zeros(n_total, dtype=bool)
                    source_active[domain.elements] = source_grid
                    active_mask[~source_active] = False
                    passive_density[~source_active] = 1e-3
                else:
                    logger.warning(
                        "TopologyOptVoxelNode: ignoring design-domain mask "
                        "with shape %s; expected (%d, %d, %d).",
                        source_grid.shape,
                        p.nelx,
                        p.nely,
                        p.nelz,
                    )
            except Exception:
                logger.warning(
                    "TopologyOptVoxelNode: failed to apply design-domain mask.",
                    exc_info=True,
                )

        def _voxel_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_f, hi_f = sorted((float(lo), float(hi)))
            lo_f = max(0.0, min(1.0, lo_f))
            hi_f = max(0.0, min(1.0, hi_f))
            if abs(hi_f - lo_f) < 1e-12:
                if lo_f <= 0.0:
                    return slice(0, min(1, nmax))
                if hi_f >= 1.0:
                    return slice(max(0, nmax - 1), nmax)
                idx = max(0, min(nmax - 1, int(np.floor(lo_f * nmax))))
                return slice(idx, idx + 1)
            lo_i = int(np.floor(lo_f * nmax))
            hi_i = int(np.ceil(hi_f * nmax))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            if hi_i <= lo_i:
                hi_i = min(nmax, lo_i + 1)
            return slice(lo_i, hi_i)

        def _flat_indices(
            box: tuple[float, float, float, float, float, float],
        ) -> np.ndarray:
            x0, x1, y0, y1, z0, z1 = box
            sub = domain.elements[
                _voxel_slice(x0, x1, p.nelx),
                _voxel_slice(y0, y1, p.nely),
                _voxel_slice(z0, z1, p.nelz),
            ]
            return np.asarray(sub, dtype=int).ravel()

        cylinder_grid = None

        def _cylinder_indices(
            cylinder: tuple[object, ...],
        ) -> np.ndarray:
            nonlocal cylinder_grid
            if len(cylinder) < 6:
                return np.array([], dtype=int)
            axis, c0, c1, lo, hi, radius_a = cylinder[:6]
            radius_b = cylinder[6] if len(cylinder) > 6 else radius_a
            axis = str(axis or "z").lower()
            lo, hi = sorted((float(lo), float(hi)))
            radius_a = float(radius_a)
            radius_b = float(radius_b)
            if radius_a <= 0.0 or radius_b <= 0.0:
                return np.array([], dtype=int)

            if cylinder_grid is None:
                xs = (np.arange(p.nelx, dtype=float) + 0.5) / max(p.nelx, 1)
                ys = (np.arange(p.nely, dtype=float) + 0.5) / max(p.nely, 1)
                zs = (np.arange(p.nelz, dtype=float) + 0.5) / max(p.nelz, 1)
                cylinder_grid = np.meshgrid(xs, ys, zs, indexing="ij")
            xx, yy, zz = cylinder_grid

            if axis == "x":
                axial = xx
                dist2 = ((yy - float(c0)) / radius_a) ** 2 + (
                    (zz - float(c1)) / radius_b
                ) ** 2
            elif axis == "y":
                axial = yy
                dist2 = ((xx - float(c0)) / radius_a) ** 2 + (
                    (zz - float(c1)) / radius_b
                ) ** 2
            else:
                axial = zz
                dist2 = ((xx - float(c0)) / radius_a) ** 2 + (
                    (yy - float(c1)) / radius_b
                ) ** 2

            mask = (axial >= lo) & (axial <= hi) & (dist2 <= 1.0)
            return np.asarray(domain.elements[mask], dtype=int).ravel()

        def _inside_design_domain(idx: np.ndarray) -> np.ndarray:
            """Keep only the pad voxels that lie on the part itself.

            A voxel-exact test, so it also handles a pad that straddles a slot
            or bore, which the analytic trim applied at generation time cannot
            express as a box.
            """
            if source_active is None or idx.size == 0:
                return idx
            return idx[source_active[idx]]

        for box in bc.solid_boxes:
            idx = _inside_design_domain(_flat_indices(box))
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1.0

        for cylinder in getattr(bc, "solid_cylinders", []):
            idx = _inside_design_domain(_cylinder_indices(cylinder))
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1.0

        if p.passive_solid_mask is not None:
            keep = np.asarray(p.passive_solid_mask, dtype=bool)
            if keep.shape != (p.nelx, p.nely, p.nelz):
                raise ValueError(
                    "Non-design material mask does not match the topology grid."
                )
            idx = np.asarray(domain.elements[keep], dtype=int).ravel()
            active_mask[idx] = False
            passive_density[idx] = 1.0

        # Joint anchors need a small non-design attachment pad; otherwise a
        # mathematically stiff coupling can terminate at an almost-void voxel.
        all_joints = list(getattr(bc, "joints", []))
        for load_case in getattr(bc, "load_cases", []):
            all_joints.extend(getattr(load_case, "joints", []))
        for joint in all_joints:
            for anchor in (joint.anchor_a, joint.anchor_b):
                ix = int(np.clip(round(anchor[0] * p.nelx), 0, p.nelx))
                iy = int(np.clip(round(anchor[1] * p.nely), 0, p.nely))
                iz = int(np.clip(round(anchor[2] * p.nelz), 0, p.nelz))
                sub = domain.elements[
                    max(0, ix - 1) : min(p.nelx, ix + 1),
                    max(0, iy - 1) : min(p.nely, iy + 1),
                    max(0, iz - 1) : min(p.nelz, iz + 1),
                ]
                idx = _inside_design_domain(np.asarray(sub, dtype=int).ravel())
                if idx.size == 0:
                    continue
                active_mask[idx] = False
                passive_density[idx] = 1.0

        for box in bc.void_boxes:
            idx = _flat_indices(box)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1e-3

        for cylinder in getattr(bc, "void_cylinders", []):
            idx = _cylinder_indices(cylinder)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1e-3

        if p.passive_void_mask is not None:
            cut = np.asarray(p.passive_void_mask, dtype=bool)
            if cut.shape != (p.nelx, p.nely, p.nelz):
                raise ValueError(
                    "Non-design void mask does not match the topology grid."
                )
            idx = np.asarray(domain.elements[cut], dtype=int).ravel()
            active_mask[idx] = False
            passive_density[idx] = 1e-3

        return active_mask, passive_density
