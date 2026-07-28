# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Small pyMOTO modules used to assemble the topology network."""

from __future__ import annotations

import numpy as np

from .level_set import (
    LEVEL_SET_BETA,
    level_set_heaviside,
    level_set_heaviside_derivative,
)
from .pymoto_runtime import PyMotoDomain, import_pymoto


def _density_3d_to_flat(
    x_3d: np.ndarray,
    domain: PyMotoDomain,
) -> np.ndarray:
    """Inverse of `_density_grid_from_state` — write a (nelx,nely,nelz) grid back
    into a flat element vector using pyMOTO's `domain.elements` mapping."""
    flat = np.empty(domain.nel, dtype=float)
    flat[domain.elements] = np.asarray(x_3d, dtype=float)
    return flat


def _make_passive_clamp_module() -> type[object]:
    pym = import_pymoto()

    class _PassiveClamp(pym.Module):
        def __init__(
            self,
            active_mask: object,
            passive_density: object,
        ) -> None:
            super().__init__()
            self.active_mask = np.asarray(active_mask, dtype=bool)
            self.passive_density = np.asarray(passive_density, dtype=float)

        def __call__(self, x: object) -> np.ndarray:
            y = np.asarray(x, dtype=float).copy()
            y[~self.active_mask] = self.passive_density[~self.active_mask]
            return y

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            dx = np.asarray(dy, dtype=float).copy()
            dx[~self.active_mask] = 0.0
            return [dx]

    return _PassiveClamp


def _make_concat_module() -> type[object]:
    """Concatenate N vectors of length nel into one length-N·nel vector.

    Used to aggregate per-load-case vm² fields into a single PNorm.
    Concatenation + PNorm is mathematically equivalent to a single PNorm
    over the union of all elemental stresses.
    """
    pym = import_pymoto()

    class _Concat(pym.Module):
        def __call__(self, *inputs: object) -> np.ndarray:
            self._shapes = [np.asarray(x).shape for x in inputs]
            self._sizes = [int(np.asarray(x).size) for x in inputs]
            return np.concatenate([np.asarray(x, dtype=float).ravel() for x in inputs])

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            dy_flat = np.asarray(dy, dtype=float).ravel()
            out = []
            offset = 0
            for sz, shape in zip(
                self._sizes,
                self._shapes,
                strict=True,
            ):
                out.append(dy_flat[offset : offset + sz].copy().reshape(shape))
                offset += sz
            return out

    return _Concat


def _make_heaviside_module() -> type[object]:
    """Build the smooth-Heaviside projection pyMOTO Module class.

    Three-field SIMP (Sigmund/Wang/Lazarov 2011): physical density =
    H_β(filtered density), with β stepped from ~1 → ~32 by the iteration
    loop. β is a *mutable attribute* so the loop can update it between
    `net.response()` calls without rebuilding the network.
    """
    pym = import_pymoto()

    class _HeavisideProjection(pym.Module):
        def __init__(self, beta: float = 1.0, eta: float = 0.5) -> None:
            super().__init__()
            self.beta = float(beta)
            self.eta = float(eta)

        def __call__(self, x: object) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            self._x = x_arr
            beta = float(self.beta)
            eta = float(self.eta)
            if beta < 1e-6:
                return x_arr.copy()
            tanh_be = np.tanh(beta * eta)
            tanh_b1e = np.tanh(beta * (1.0 - eta))
            denom = tanh_be + tanh_b1e
            return (tanh_be + np.tanh(beta * (x_arr - eta))) / denom

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            dy_arr = np.asarray(dy, dtype=float)
            beta = float(self.beta)
            eta = float(self.eta)
            if beta < 1e-6:
                return [dy_arr.copy()]
            x = self._x
            tanh_be = np.tanh(beta * eta)
            tanh_b1e = np.tanh(beta * (1.0 - eta))
            denom = tanh_be + tanh_b1e
            dproj = beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / denom
            return [dy_arr * dproj]

    return _HeavisideProjection


def _make_level_set_heaviside_module() -> type[object]:
    """Build the differentiable ersatz-material map for a signed interface."""
    pym = import_pymoto()

    class _LevelSetHeaviside(pym.Module):
        def __init__(self, beta: float = LEVEL_SET_BETA) -> None:
            super().__init__()
            self.beta = float(beta)

        def __call__(self, phi: object) -> np.ndarray:
            self._phi = np.asarray(phi, dtype=float)
            return level_set_heaviside(self._phi, beta=self.beta)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            derivative = level_set_heaviside_derivative(
                self._phi,
                beta=self.beta,
            )
            return [np.asarray(dy, dtype=float) * derivative]

    return _LevelSetHeaviside


def _make_sparse_to_csc_module() -> type[object]:
    pym = import_pymoto()
    from scipy.sparse import csc_matrix, isspmatrix_csc

    class _SparseToCSC(pym.Module):
        """Convert sparse matrices to CSC before SciPy splu.

        This is mathematically an identity operation; it only changes sparse
        storage format so pyMOTO/SciPy does not warn during factorization.
        """

        def __call__(self, matrix: object) -> object:
            self._input_format = getattr(matrix, "format", None)

            if isspmatrix_csc(matrix):
                return matrix

            if hasattr(matrix, "tocsc"):
                return matrix.tocsc(copy=False)

            return csc_matrix(matrix)

        def _sensitivity(self, derivative: object) -> list[object]:
            # Format conversion is identity with respect to matrix entries.
            try:
                fmt = getattr(self, "_input_format", None)
                if fmt and hasattr(derivative, "asformat"):
                    return [derivative.asformat(fmt)]
            except Exception:
                pass
            return [derivative]

    return _SparseToCSC


# Quadratic-form matrix for von Mises stress in 3-D Voigt notation.
_VM_A = np.array(
    [
        [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
        [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
        [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    ],
    dtype=float,
)


def _make_vm_module() -> type[object]:
    """Build the relaxed-von-Mises pyMOTO Module class on first use.

    Defined as a factory so this file imports fine when pyMOTO is absent
    (the rest of the module stays usable for headless data manipulation).
    """
    pym = import_pymoto()

    class _VonMisesSquaredRelaxed(pym.Module):
        """SIMP-relaxed von Mises² per element.

        Inputs:
            s   — Voigt stress, shape (6, nel)
            rho — element density, shape (nel,)
        Output:
            vm_sq — σ_vm² per element, shape (nel,)
                    with σ_relaxed = ρ^q · σ_linear and vm_sq = σ_relaxedᵀ A σ_relaxed

        Sensitivities:
            ∂vm_sq/∂s_e = 2 · ρ_e^(2q) · A · s_e
            ∂vm_sq/∂ρ_e = 2q · ρ_e^(2q-1) · s_eᵀ A s_e
        """

        def __init__(self, stress_penalty: float = 1.0) -> None:
            super().__init__()
            self.q = float(stress_penalty)

        def __call__(
            self,
            stress: object,
            density: object,
        ) -> np.ndarray:
            s_arr = np.asarray(stress, dtype=float)
            rho_arr = np.asarray(density, dtype=float)
            self._s = s_arr
            self._rho = rho_arr
            rho_q = rho_arr**self.q  # (nel,)
            s_relaxed = s_arr * rho_q[np.newaxis, :]  # (6, nel)
            vm_sq = np.einsum("ij,ie,je->e", _VM_A, s_relaxed, s_relaxed)
            return vm_sq

        def _sensitivity(self, derivative: object) -> list[np.ndarray]:
            dvm = np.asarray(derivative, dtype=float)  # (nel,)
            s, rho = self._s, self._rho

            # ∂vm²/∂s = 2 · ρ^(2q) · A · s_lin
            A_s = _VM_A @ s  # (6, nel)
            rho_2q = rho ** (2.0 * self.q)
            ds = 2.0 * (rho_2q[np.newaxis, :] * A_s) * dvm[np.newaxis, :]

            # ∂vm²/∂ρ = 2q · ρ^(2q-1) · s_linᵀ A s_lin
            vm_sq_lin = np.einsum("ij,ie,je->e", _VM_A, s, s)
            with np.errstate(divide="ignore", invalid="ignore"):
                drho_raw = np.where(
                    rho > 1e-12,
                    2.0 * self.q * (rho ** (2.0 * self.q - 1.0)) * vm_sq_lin,
                    0.0,
                )
            drho = drho_raw * dvm
            return [ds, drho]

    return _VonMisesSquaredRelaxed
