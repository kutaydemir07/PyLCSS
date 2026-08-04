# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Small pyMOTO modules used to assemble the topology network."""

from __future__ import annotations

import numpy as np

from ..manufacturing.constraints import (
    _BUILD_AXIS_MAP,
    _PATTERN_AXIS_PLANE,
    _apply_extrusion,
    _apply_symmetry,
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


def _make_volume_constraint_module() -> type[object]:
    """Build a mutable mean-density constraint for robust volume adaptation."""
    pym = import_pymoto()

    class _VolumeConstraint(pym.Module):
        def __init__(self, target_fraction: float) -> None:
            super().__init__()
            self.target_fraction = float(target_fraction)

        def __call__(self, density: object) -> float:
            values = np.asarray(density, dtype=float)
            self._size = max(int(values.size), 1)
            return float(10.0 * (np.mean(values) - self.target_fraction))

        def _sensitivity(self, derivative: object) -> list[np.ndarray]:
            scalar = float(np.asarray(derivative, dtype=float).reshape(-1)[0])
            return [np.full(self._size, 10.0 * scalar / self._size, dtype=float)]

    return _VolumeConstraint


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


def _make_simp_module() -> type[object]:
    """Build the modified-SIMP interpolation with a mutable exponent.

    ``property = minimum + (full - minimum) * rho^p``. Written as a module
    rather than a `pym.MathExpression` with the exponent baked into its string
    so the iteration loop can run p-continuation: p enters at 1, where the
    compliance problem is convex and has a unique solution, and is raised
    toward the study's penalization in steps. Starting at the target exponent
    instead lets the very first descent direction commit the topology, and that
    commitment is what continuation exists to avoid.
    """
    pym = import_pymoto()

    class _SimpInterpolation(pym.Module):
        def __init__(
            self,
            minimum: float,
            full: float,
            penal: float = 3.0,
        ) -> None:
            super().__init__()
            self.minimum = float(minimum)
            self.full = float(full)
            self.penal = float(penal)

        def __call__(self, x: object) -> np.ndarray:
            # Densities are bounded away from zero by the optimizer's lower
            # bound, but a passive void clamp can sit at exactly 1e-3 and a
            # non-integer exponent has no real derivative at a negative base.
            x_arr = np.clip(np.asarray(x, dtype=float), 0.0, None)
            self._x = x_arr
            span = self.full - self.minimum
            return self.minimum + span * x_arr ** float(self.penal)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            penal = float(self.penal)
            span = self.full - self.minimum
            derivative = penal * span * self._x ** (penal - 1.0)
            return [np.asarray(dy, dtype=float) * derivative]

    return _SimpInterpolation


def _make_linear_grid_projection_module(
    domain: PyMotoDomain,
    *,
    symmetry: str = "none",
    extrusion: str = "none",
) -> type[object]:
    """Build the exact-adjoint symmetry/extrusion density mapping."""
    pym = import_pymoto()

    class _LinearGridProjection(pym.Module):
        def __call__(self, x: object) -> np.ndarray:
            grid = np.asarray(x, dtype=float)[domain.elements]
            grid = _apply_symmetry(grid, symmetry)
            grid = _apply_extrusion(grid, extrusion)
            return _density_3d_to_flat(grid, domain)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            # Mirror averaging and axis averaging are orthogonal projections:
            # their transpose is the same operation.
            grid = np.asarray(dy, dtype=float)[domain.elements]
            grid = _apply_extrusion(grid, extrusion)
            grid = _apply_symmetry(grid, symmetry)
            return [_density_3d_to_flat(grid, domain)]

    return _LinearGridProjection


def _make_pull_out_module(
    domain: PyMotoDomain,
    pull_axis: str,
) -> type[object]:
    """Build a running-minimum pull-out mapping with a valid subgradient."""
    pym = import_pymoto()
    key = str(pull_axis or "none").lower()
    axis_step = _BUILD_AXIS_MAP.get(key)

    class _PullOutProjection(pym.Module):
        def __call__(self, x: object) -> np.ndarray:
            grid = np.asarray(x, dtype=float)[domain.elements]
            if axis_step is None or grid.shape[axis_step[0]] < 2:
                self._source_indices = None
                return np.asarray(x, dtype=float).copy()

            axis, step = axis_step
            work = np.moveaxis(grid, axis, 0)
            if step < 0:
                work = work[::-1]
            flat = work.reshape(work.shape[0], -1)
            out = np.empty_like(flat)
            argmin = np.empty(flat.shape, dtype=np.int64)
            out[0] = flat[0]
            argmin[0] = 0
            for layer in range(1, flat.shape[0]):
                take_new = flat[layer] < out[layer - 1]
                out[layer] = np.where(take_new, flat[layer], out[layer - 1])
                argmin[layer] = np.where(take_new, layer, argmin[layer - 1])
            self._source_indices = argmin
            projected = out.reshape(work.shape)
            if step < 0:
                projected = projected[::-1]
            projected = np.moveaxis(projected, 0, axis)
            return _density_3d_to_flat(projected, domain)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            if self._source_indices is None or axis_step is None:
                return [np.asarray(dy, dtype=float).copy()]
            axis, step = axis_step
            grad = np.asarray(dy, dtype=float)[domain.elements]
            work = np.moveaxis(grad, axis, 0)
            if step < 0:
                work = work[::-1]
            flat_grad = work.reshape(work.shape[0], -1)
            flat_out = np.zeros_like(flat_grad)
            columns = np.broadcast_to(
                np.arange(flat_grad.shape[1], dtype=np.int64),
                flat_grad.shape,
            )
            np.add.at(
                flat_out,
                (self._source_indices.ravel(), columns.ravel()),
                flat_grad.ravel(),
            )
            output = flat_out.reshape(work.shape)
            if step < 0:
                output = output[::-1]
            output = np.moveaxis(output, 0, axis)
            return [_density_3d_to_flat(output, domain)]

    return _PullOutProjection


def _make_max_member_module(
    domain: PyMotoDomain,
    radius_voxels: float,
    threshold: float,
) -> type[object]:
    """Build the local-density cap and its analytic transpose sensitivity."""
    pym = import_pymoto()
    from scipy.ndimage import uniform_filter

    size = max(3, int(round(2.0 * float(radius_voxels) + 1.0)))
    limit = float(threshold)

    class _MaximumMemberProjection(pym.Module):
        def __call__(self, x: object) -> np.ndarray:
            grid = np.asarray(x, dtype=float)[domain.elements]
            local = uniform_filter(grid, size=size, mode="constant", cval=0.0)
            active = local > limit
            scale = np.ones_like(local)
            scale[active] = limit / local[active]
            self._grid = grid
            self._local = local
            self._scale = scale
            self._active = active
            return _density_3d_to_flat(grid * scale, domain)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            grad = np.asarray(dy, dtype=float)[domain.elements]
            dscale = np.zeros_like(self._local)
            dscale[self._active] = -limit / self._local[self._active] ** 2
            through_average = uniform_filter(
                grad * self._grid * dscale,
                size=size,
                mode="constant",
                cval=0.0,
            )
            dx = grad * self._scale + through_average
            return [_density_3d_to_flat(dx, domain)]

    return _MaximumMemberProjection


def _rotation_operator(
    shape: tuple[int, int, int],
    axes: tuple[int, int],
    angle_degrees: float,
) -> object:
    """Sparse bilinear rotation matching ``reshape=False`` about grid centre."""
    from scipy.sparse import coo_matrix

    coords = np.indices(shape, dtype=float)
    first, second = axes
    angle = np.deg2rad(float(angle_degrees))
    rotation = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    output_plane = np.vstack(
        [coords[first].ravel(), coords[second].ravel()]
    )
    centre = 0.5 * (np.asarray([shape[first], shape[second]], dtype=float) - 1.0)
    input_plane = rotation @ (output_plane - centre[:, None]) + centre[:, None]

    lower = np.floor(input_plane).astype(np.int64)
    fraction = input_plane - lower
    output_rows = np.arange(int(np.prod(shape)), dtype=np.int64)
    base_indices = [coords[i].ravel().astype(np.int64) for i in range(3)]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for da in (0, 1):
        for db in (0, 1):
            source = list(base_indices)
            source[first] = lower[0] + da
            source[second] = lower[1] + db
            valid = (
                (source[first] >= 0)
                & (source[first] < shape[first])
                & (source[second] >= 0)
                & (source[second] < shape[second])
            )
            weight = (
                (fraction[0] if da else 1.0 - fraction[0])
                * (fraction[1] if db else 1.0 - fraction[1])
            )
            if np.any(valid):
                source_tuple = tuple(index[valid] for index in source)
                rows.append(output_rows[valid])
                cols.append(np.ravel_multi_index(source_tuple, shape))
                data.append(weight[valid])
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(int(np.prod(shape)), int(np.prod(shape))),
    ).tocsr()


def _make_pattern_repeat_module(
    domain: PyMotoDomain,
    n_fold: int,
    axis: str,
) -> type[object]:
    """Build an N-fold rotation average with an exact sparse transpose."""
    pym = import_pymoto()
    from scipy.sparse import eye

    shape = (int(domain.nelx), int(domain.nely), int(domain.nelz))
    plane = _PATTERN_AXIS_PLANE[str(axis or "y").lower()]
    count = max(1, int(n_fold))
    operator = eye(int(np.prod(shape)), format="csr")
    for index in range(1, count):
        operator = operator + _rotation_operator(
            shape,
            plane,
            360.0 * float(index) / float(count),
        )
    operator = operator * (1.0 / float(count))

    class _PatternRepeatProjection(pym.Module):
        def __call__(self, x: object) -> np.ndarray:
            grid = np.asarray(x, dtype=float)[domain.elements]
            output = np.asarray(operator @ grid.ravel()).reshape(shape)
            return _density_3d_to_flat(output, domain)

        def _sensitivity(self, dy: object) -> list[np.ndarray]:
            grid = np.asarray(dy, dtype=float)[domain.elements]
            output = np.asarray(operator.T @ grid.ravel()).reshape(shape)
            return [_density_3d_to_flat(output, domain)]

    return _PatternRepeatProjection


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


def _make_matrix_sum_module() -> type[object]:
    """Add two assembled system matrices into one operator.

    Used to place a design-dependent convection term alongside the thermal
    conduction operator, giving ``(K + H) T = Q``. Both operands depend on the
    density, so the sum has to happen inside the network rather than as a
    constant offset.
    """
    pym = import_pymoto()

    class _MatrixSum(pym.Module):
        def __call__(self, first: object, second: object) -> object:
            if second is None:
                return first
            return first + second

        def _sensitivity(self, derivative: object) -> list[object]:
            # d(A + B) is the identity with respect to both operands.
            return [derivative, derivative]

    return _MatrixSum


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
