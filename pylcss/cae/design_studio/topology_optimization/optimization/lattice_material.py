# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Couple homogenized lattice cell laws into the part-scale sensitivity loop.

This is the macro half of the two-scale problem. :mod:`..manufacturing.cell_material`
answers "what stiffness does this cell have at this relative density"; this
module puts that answer inside the finite-element operator the optimizer
differentiates, so the density field is driven by the lattice that will
actually be built rather than by an isotropic stand-in for it.

How the anisotropy survives assembly
------------------------------------
pyMOTO assembles :math:`\\mathbf{K} = \\sum_e \\sum_i x_{i,e}\\mathbf{K}_{i,e}`,
so a design-dependent *material* — not just a design-dependent scale factor —
only needs the constitutive matrix split into pieces with constant geometry.
For a cubic cell that split is exact and has three terms:

.. math::

    \\mathbf{D}(\\rho) = C_{11}(\\rho)\\mathbf{E}_1
                       + C_{12}(\\rho)\\mathbf{E}_2
                       + C_{44}(\\rho)\\mathbf{E}_3

with :math:`\\mathbf{E}_1 = \\mathrm{diag}(1,1,1,0,0,0)`, :math:`\\mathbf{E}_2`
the off-diagonal ones of the normal-stress block, and
:math:`\\mathbf{E}_3 = \\mathrm{diag}(0,0,0,1,1,1)`. Integrating each against
the same strain-displacement operator gives three constant element matrices,
and the three density-dependent constants become three scaling fields. The
result is an exact anisotropic assembly at the cost of three scalar fields
instead of one — no extra linear solves, and the adjoint is unchanged because
pyMOTO already returns one sensitivity per scaling field.

Because the shear block of a cubic tensor is :math:`C_{44}\\mathbf{I}`, the
split is invariant to whether the shear strains are ordered ``[yz, zx, xy]``
or ``[xy, yz, zx]``, so it cannot be desynchronized from pyMOTO's Voigt
convention.
"""
from __future__ import annotations

import logging

import numpy as np

from ..manufacturing.cell_material import CellMaterialLaw, solid_cubic_constants
from .pymoto_runtime import PyMotoDomain, import_pymoto

logger = logging.getLogger(__name__)

__all__ = [
    "cubic_basis_element_matrices",
    "cubic_voigt_basis",
    "make_cell_material_module",
]


def cubic_voigt_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three 6x6 Voigt tensors a cubic material decomposes onto."""
    normal = np.zeros((6, 6), dtype=float)
    normal[0, 0] = normal[1, 1] = normal[2, 2] = 1.0

    coupling = np.zeros((6, 6), dtype=float)
    coupling[:3, :3] = 1.0
    coupling[0, 0] = coupling[1, 1] = coupling[2, 2] = 0.0

    shear = np.zeros((6, 6), dtype=float)
    shear[3, 3] = shear[4, 4] = shear[5, 5] = 1.0
    return normal, coupling, shear


def cubic_basis_element_matrices(
    domain: PyMotoDomain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the three basis element stiffness matrices for ``domain``.

    Built through pyMOTO's own shape-function derivatives and Gauss points
    rather than a private copy, so the element size, node ordering and
    integration rule cannot drift from what the rest of the network assumes.
    """
    pym = import_pymoto()
    from pymoto.modules.assembly import get_B

    if int(domain.dim) != 3:
        raise ValueError("Homogenized cell laws are defined for 3-D studies only.")

    nodes_per_element = 2 ** int(domain.dim)
    dofs = nodes_per_element * int(domain.dim)
    size = domain.element_size
    weight = float(np.prod(size[: int(domain.dim)] / 2.0))

    basis = cubic_voigt_basis()
    matrices = [np.zeros((dofs, dofs), dtype=float) for _ in basis]
    for corner in domain.node_numbering:
        position = corner * (size / 2.0) / np.sqrt(3.0)
        strain_displacement = get_B(domain.eval_shape_fun_der(position))
        for index, tensor in enumerate(basis):
            matrices[index] += weight * (
                strain_displacement.T @ tensor @ strain_displacement
            )
    _ = pym
    return tuple(matrices)  # type: ignore[return-value]


def make_cell_material_module() -> type[object]:
    """Build the density-to-cubic-constants pyMOTO module class.

    Defined as a factory for the same reason as the other modules here: the
    file must import when pyMOTO is absent.
    """
    pym = import_pymoto()

    class _HomogenizedCellMaterial(pym.Module):
        """Map element density to the three cubic constants of the cell.

        Inputs:
            rho — physical element density, shape ``(nel,)``
        Outputs:
            c11, c12, c44 — element constants, shape ``(nel,)`` each

        The void floor mirrors SIMP's ``Emin + (E0 - Emin) rho^p``: a solid
        tensor scaled by the minimum stiffness is added to the homogenized
        law, so an empty element stays invertible and a saturated one recovers
        the base material to within ``Emin/E0``.
        """

        def __init__(
            self,
            law: CellMaterialLaw,
            *,
            young: float,
            minimum_young: float,
            poisson: float,
        ) -> None:
            super().__init__()
            self.law = law
            self.young = float(young)
            self.floor = np.asarray(
                solid_cubic_constants(poisson), dtype=float
            ) * float(minimum_young)

        def __call__(self, density: object) -> tuple[np.ndarray, ...]:
            rho = np.asarray(density, dtype=float)
            values, gradients = self.law.evaluate_with_gradient(
                rho, young=self.young
            )
            self._gradients = gradients
            return tuple(
                value + self.floor[index] for index, value in enumerate(values)
            )

        def _sensitivity(self, *derivatives: object) -> list[np.ndarray]:
            total: np.ndarray | None = None
            for index, derivative in enumerate(derivatives):
                if derivative is None:
                    continue
                contribution = np.asarray(derivative, dtype=float) * self._gradients[
                    index
                ]
                total = contribution if total is None else total + contribution
            if total is None:
                return [np.zeros_like(self._gradients[0])]
            return [total]

    return _HomogenizedCellMaterial
