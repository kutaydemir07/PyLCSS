# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Numerical homogenization of a periodic lattice unit cell.

A lattice cannot be meshed at part scale — a 100 mm part with 2 mm cells and
0.3 mm walls needs on the order of 1e9 elements. Every practical method
therefore replaces the cell by an effective continuum, optimizes that, and
generates explicit geometry only at the end.

This module computes that effective continuum. For a periodic voxel cell it
solves the six unit-macroscopic-strain boundary-value problems and assembles
the homogenized elasticity tensor

    C^H_ij = 1/|V| * sum_e (u0_i - chi_i)_e^T k_e (u0_j - chi_j)_e

following the standard energy-based formulation (Andreassen & Andreasen,
*How to determine composite material properties using numerical
homogenization*, Comput. Mater. Sci. 83, 2014).

The cells themselves come from :mod:`.structures`, so the properties describe
exactly the geometry PyLCSS goes on to generate rather than an idealized
textbook cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

__all__ = [
    "CubicElasticity",
    "homogenize_cell",
    "hex8_stiffness_matrix",
    "isotropic_elasticity_matrix",
]

# Void is given a small but finite stiffness. A true zero makes the periodic
# system singular whenever the solid phase is disconnected by rasterization,
# which happens at the low-density end of every cell family.
VOID_STIFFNESS_RATIO = 1e-6

_GAUSS_POINTS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

# Corner ordering of the trilinear hexahedron, as (di, dj, dk) offsets.
_HEX_CORNERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ],
    dtype=int,
)

# Natural coordinates of those corners, in the same order.
_HEX_NATURAL = 2.0 * _HEX_CORNERS.astype(float) - 1.0


@dataclass(frozen=True)
class CubicElasticity:
    """Effective constants of a cell with cubic symmetry.

    TPMS and strut families used here are cubic, so the 6x6 tensor collapses
    to three independent constants. ``young_isotropic`` and
    ``poisson_isotropic`` are the Voigt-Reuss-Hill average, used where an
    isotropic law is enough (a gyroid sheet is close to isotropic by design).
    """

    relative_density: float
    c11: float
    c12: float
    c44: float
    young_isotropic: float
    poisson_isotropic: float
    # The full tensor before the cubic reduction. Kept so callers can measure
    # how far a cell actually is from cubic symmetry instead of assuming it.
    tensor: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 6), dtype=float)
    )

    @property
    def zener_ratio(self) -> float:
        """Anisotropy measure; 1.0 is perfectly isotropic."""
        denominator = self.c11 - self.c12
        if abs(denominator) < 1e-30:
            return float("inf")
        return float(2.0 * self.c44 / denominator)

    @property
    def cubic_symmetry_residual(self) -> float:
        """Relative norm of everything the cubic reduction discards.

        A voxelized cell is only cubic to within its rasterization, and a
        clipped or badly resolved cell may not be cubic at all. This is the
        number that says whether the three-constant law is trustworthy: it is
        the Frobenius norm of ``tensor`` minus its best cubic fit, divided by
        the norm of ``tensor``.
        """
        tensor = np.asarray(self.tensor, dtype=float)
        norm = float(np.linalg.norm(tensor))
        if norm <= 0.0:
            return 0.0
        cubic = np.zeros((6, 6), dtype=float)
        cubic[:3, :3] = self.c12
        for index in range(3):
            cubic[index, index] = self.c11
            cubic[3 + index, 3 + index] = self.c44
        return float(np.linalg.norm(tensor - cubic) / norm)


def isotropic_elasticity_matrix(young: float, poisson: float) -> np.ndarray:
    """Return the 6x6 isotropic constitutive matrix in Voigt notation."""
    factor = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lame = factor * poisson
    shear = young / (2.0 * (1.0 + poisson))
    matrix = np.zeros((6, 6), dtype=float)
    matrix[:3, :3] = lame
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = factor * (1.0 - poisson)
    matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = shear
    return matrix


def _shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return dN/d(natural) for the eight corners, shape (8, 3)."""
    natural = _HEX_NATURAL
    signs_xi, signs_eta, signs_zeta = natural[:, 0], natural[:, 1], natural[:, 2]
    d_xi = 0.125 * signs_xi * (1.0 + signs_eta * eta) * (1.0 + signs_zeta * zeta)
    d_eta = 0.125 * (1.0 + signs_xi * xi) * signs_eta * (1.0 + signs_zeta * zeta)
    d_zeta = 0.125 * (1.0 + signs_xi * xi) * (1.0 + signs_eta * eta) * signs_zeta
    return np.column_stack([d_xi, d_eta, d_zeta])


def _strain_displacement(derivatives: np.ndarray) -> np.ndarray:
    """Assemble the 6x24 Voigt strain-displacement matrix."""
    b_matrix = np.zeros((6, 24), dtype=float)
    for corner in range(8):
        dx, dy, dz = derivatives[corner]
        column = 3 * corner
        b_matrix[0, column + 0] = dx
        b_matrix[1, column + 1] = dy
        b_matrix[2, column + 2] = dz
        b_matrix[3, column + 0] = dy
        b_matrix[3, column + 1] = dx
        b_matrix[4, column + 1] = dz
        b_matrix[4, column + 2] = dy
        b_matrix[5, column + 0] = dz
        b_matrix[5, column + 2] = dx
    return b_matrix


def hex8_stiffness_matrix(
    young: float = 1.0,
    poisson: float = 0.3,
    size: float = 1.0,
) -> np.ndarray:
    """Return the 24x24 stiffness of a cubic trilinear hexahedron.

    Every voxel in a structured grid is the same cube, so this is built once
    and reused for the whole cell.
    """
    constitutive = isotropic_elasticity_matrix(young, poisson)
    # A cube of edge `size` maps from natural coordinates with a constant
    # diagonal Jacobian, so the inverse and determinant are scalars.
    jacobian = 0.5 * float(size)
    det_jacobian = jacobian**3
    stiffness = np.zeros((24, 24), dtype=float)
    for xi in _GAUSS_POINTS:
        for eta in _GAUSS_POINTS:
            for zeta in _GAUSS_POINTS:
                derivatives = _shape_derivatives(xi, eta, zeta) / jacobian
                b_matrix = _strain_displacement(derivatives)
                stiffness += (
                    b_matrix.T @ constitutive @ b_matrix
                ) * det_jacobian
    return stiffness


def _periodic_node_ids(shape: tuple[int, int, int]) -> np.ndarray:
    """Return element connectivity for a fully periodic voxel grid.

    Wrapping the corner indices makes the opposite faces share nodes, which
    imposes periodicity directly instead of through constraint equations.
    """
    nx, ny, nz = shape
    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    # Element fields elsewhere in the homogenizer use Fortran ordering
    # (x-index fastest), matching the node-number formula below. Keeping the
    # coordinate enumeration in that same order is essential for asymmetric
    # or directionally graded custom cells.
    i, j, k = (
        i.ravel(order="F"),
        j.ravel(order="F"),
        k.ravel(order="F"),
    )
    connectivity = np.empty((i.size, 8), dtype=np.int64)
    for corner, (di, dj, dk) in enumerate(_HEX_CORNERS):
        connectivity[:, corner] = (
            ((i + di) % nx)
            + ((j + dj) % ny) * nx
            + ((k + dk) % nz) * nx * ny
        )
    return connectivity


def _macroscopic_displacements(
    shape: tuple[int, int, int],
    connectivity: np.ndarray,
    size: float,
) -> np.ndarray:
    """Nodal displacement fields u0 = eps0 . x for the six unit strains.

    Returned per element, shape (6, n_elements, 24), because the periodic
    wrap makes a node's position ambiguous — an element's own corner offsets
    give the unambiguous local coordinates.
    """
    nx, ny, nz = shape
    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    base = np.column_stack(
        [
            i.ravel(order="F"),
            j.ravel(order="F"),
            k.ravel(order="F"),
        ]
    ).astype(float)
    n_elements = base.shape[0]

    # Absolute corner coordinates, ignoring the wrap (local element frame).
    corner_coords = (
        base[:, None, :] + _HEX_CORNERS[None, :, :].astype(float)
    ) * float(size)

    strains = np.zeros((6, 3, 3), dtype=float)
    strains[0, 0, 0] = 1.0
    strains[1, 1, 1] = 1.0
    strains[2, 2, 2] = 1.0
    # Engineering shear: gamma = 1 split evenly over the symmetric pair.
    strains[3, 0, 1] = strains[3, 1, 0] = 0.5
    strains[4, 1, 2] = strains[4, 2, 1] = 0.5
    strains[5, 0, 2] = strains[5, 2, 0] = 0.5

    fields = np.zeros((6, n_elements, 24), dtype=float)
    for case in range(6):
        displacement = corner_coords @ strains[case].T  # (nel, 8, 3)
        fields[case] = displacement.reshape(n_elements, 24)
    _ = connectivity
    return fields


def homogenize_cell(
    occupancy: np.ndarray,
    *,
    young: float = 1.0,
    poisson: float = 0.3,
    void_ratio: float = VOID_STIFFNESS_RATIO,
) -> CubicElasticity:
    """Return the effective elasticity of one periodic voxel cell.

    ``occupancy`` is a 3-D array in [0, 1]; 1 is solid, 0 is void. It is used
    directly as a stiffness scale, so a partially filled voxel contributes
    proportionally.
    """
    field = np.clip(np.asarray(occupancy, dtype=float), 0.0, 1.0)
    if field.ndim != 3:
        raise ValueError("A unit cell must be a 3-D occupancy array.")
    shape = tuple(int(n) for n in field.shape)
    if min(shape) < 2:
        raise ValueError("A unit cell must span at least two voxels per axis.")

    size = 1.0 / float(max(shape))
    element_stiffness = hex8_stiffness_matrix(young, poisson, size)
    connectivity = _periodic_node_ids(shape)
    n_nodes = int(np.prod(shape))
    n_dofs = 3 * n_nodes

    dofs = np.repeat(connectivity, 3, axis=1) * 3
    dofs[:, 0::3] += 0
    dofs[:, 1::3] += 1
    dofs[:, 2::3] += 2

    scale = np.maximum(field.ravel(order="F"), float(void_ratio))
    # Assemble sum_e scale_e * k_e into one sparse operator.
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    values = (
        scale[:, None] * element_stiffness.ravel()[None, :]
    ).ravel()

    # Periodicity leaves a rigid-translation nullspace. A soft diagonal pin on
    # one node removes it without any sparse fancy-indexing, which dominated
    # the runtime. The pin cannot bias the result: the energy form below is
    # built from element stiffnesses, which annihilate rigid translation.
    penalty = float(np.abs(element_stiffness).max()) * 1e3
    rows = np.concatenate([rows, np.arange(3)])
    cols = np.concatenate([cols, np.arange(3)])
    values = np.concatenate([values, np.full(3, penalty)])
    stiffness = sps.coo_matrix(
        (values, (rows, cols)), shape=(n_dofs, n_dofs)
    ).tocsc()

    macro = _macroscopic_displacements(shape, connectivity, size)

    forces = np.zeros((n_dofs, 6), dtype=float)
    flat_dofs = dofs.ravel()
    for case in range(6):
        element_forces = (
            scale[:, None] * (macro[case] @ element_stiffness.T)
        )
        # bincount scatter-adds an order of magnitude faster than np.add.at.
        forces[:, case] = np.bincount(
            flat_dofs, weights=element_forces.ravel(), minlength=n_dofs
        )

    # Jacobi-preconditioned CG, not a factorization. Periodic wrap-around
    # connects opposite faces, which leaves the sparsity graph without a good
    # separator and makes LU fill-in pathological — measured at 20^3 it was
    # 145 s for LU against 3.8 s for CG, agreeing to 3e-13.
    inverse_diagonal = 1.0 / stiffness.diagonal()
    preconditioner = spla.LinearOperator(
        (n_dofs, n_dofs),
        matvec=lambda vector: vector * inverse_diagonal,
    )
    fluctuation = np.zeros((n_dofs, 6), dtype=float)
    for case in range(6):
        solution, info = spla.cg(
            stiffness,
            forces[:, case],
            rtol=1e-10,
            maxiter=20_000,
            M=preconditioner,
        )
        if info != 0:
            raise RuntimeError(
                f"Unit-cell homogenization did not converge (case {case}, "
                f"info={info}). The cell may be disconnected at this density."
            )
        fluctuation[:, case] = solution

    volume = float(np.prod(np.asarray(shape, dtype=float) * size))
    tensor = np.zeros((6, 6), dtype=float)
    difference = np.empty((6, dofs.shape[0], 24), dtype=float)
    for case in range(6):
        difference[case] = macro[case] - fluctuation[:, case][dofs]
    for a in range(6):
        weighted = (difference[a] @ element_stiffness) * scale[:, None]
        for b in range(a, 6):
            value = float(np.einsum("ij,ij->", weighted, difference[b]))
            tensor[a, b] = tensor[b, a] = value / volume

    return _cubic_constants(tensor, float(field.mean()))


def _cubic_constants(
    tensor: np.ndarray,
    relative_density: float,
) -> CubicElasticity:
    """Reduce a homogenized tensor to cubic constants and isotropic moduli."""
    c11 = float(np.mean([tensor[0, 0], tensor[1, 1], tensor[2, 2]]))
    c12 = float(np.mean([tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
    c44 = float(np.mean([tensor[3, 3], tensor[4, 4], tensor[5, 5]]))

    # Voigt-Reuss-Hill average of the cubic constants.
    bulk = (c11 + 2.0 * c12) / 3.0
    shear_voigt = (c11 - c12 + 3.0 * c44) / 5.0
    denominator = 4.0 * c44 + 3.0 * (c11 - c12)
    shear_reuss = (
        5.0 * c44 * (c11 - c12) / denominator if abs(denominator) > 1e-30 else 0.0
    )
    shear = 0.5 * (shear_voigt + max(shear_reuss, 0.0))
    if bulk <= 0.0 or shear <= 0.0:
        young, poisson = 0.0, 0.0
    else:
        young = 9.0 * bulk * shear / (3.0 * bulk + shear)
        poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return CubicElasticity(
        relative_density=float(relative_density),
        c11=c11,
        c12=c12,
        c44=c44,
        young_isotropic=float(young),
        poisson_isotropic=float(poisson),
        tensor=np.asarray(tensor, dtype=float),
    )
