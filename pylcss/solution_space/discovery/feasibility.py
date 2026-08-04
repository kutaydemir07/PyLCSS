# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

import numpy as np


class FeasibilityProblem:
    """Feasibility objective and constraints for solution-space anchor search."""

    def __init__(
        self,
        problem,
        parameters,
        ind_parameters,
        reqL,
        reqU,
        dv_norm,
        dv_norm_l,
        include_objectives=False,
        objective_indices=None,
        objective_weights=None,
    ):
        self.problem = problem
        self.parameters = parameters
        self.ind_parameters = ind_parameters
        self.reqL = np.asarray(reqL, dtype=float)
        self.reqU = np.asarray(reqU, dtype=float)
        self.dv_norm = np.asarray(dv_norm, dtype=float)
        self.dv_norm_l = np.asarray(dv_norm_l, dtype=float)
        self.include_objectives = include_objectives
        self.objective_indices = (
            objective_indices if objective_indices is not None else []
        )
        self.objective_weights = (
            objective_weights if objective_weights is not None else []
        )
        # Per-objective reference magnitude (fixed lazily on first use) so the
        # constrained anchor search sees a well-scaled, O(1) objective.
        self._obj_ref_scale = None

    def denormalize(self, x_norm):
        """Convert normalized design variables to physical space."""
        x_norm = np.asarray(x_norm, dtype=float)
        if x_norm.ndim == 2 and self.dv_norm.ndim == 1:
            return x_norm * self.dv_norm[:, np.newaxis] + self.dv_norm_l[:, np.newaxis]
        return x_norm * self.dv_norm + self.dv_norm_l

    def construct_full_vector(self, x_phys):
        """Construct full input vector including fixed/uncertain parameters."""
        x_phys = np.asarray(x_phys, dtype=float)
        is_2d = x_phys.ndim == 2
        total_vars = self.parameters.shape[1]
        ind_dvs = np.setdiff1d(np.arange(total_vars), self.ind_parameters)

        if is_2d:
            n_samples = x_phys.shape[1]
            x_full = np.zeros((total_vars, n_samples))
            x_full[ind_dvs, :] = x_phys
            if len(self.ind_parameters) > 0:
                x_full[self.ind_parameters, :] = self.parameters[
                    0, self.ind_parameters
                ].reshape(-1, 1)
        else:
            x_full = np.zeros(total_vars)
            x_full[ind_dvs] = x_phys
            if len(self.ind_parameters) > 0:
                x_full[self.ind_parameters] = self.parameters[0, self.ind_parameters]
        return x_full

    def evaluate(self, x_norm):
        """Evaluate model for ``x_norm`` and return ``(y, input_was_row_batch)``."""
        x_norm = np.asarray(x_norm, dtype=float)
        is_scipy_2d = x_norm.ndim == 2 and x_norm.shape[1] == len(self.dv_norm)
        if is_scipy_2d:
            x_norm = x_norm.T

        x_phys = self.denormalize(x_norm)
        x_full = self.construct_full_vector(x_phys)

        if x_full.ndim == 2:
            y = self.problem.evaluate_matrix(x_full)
        else:
            y = self.problem.evaluate_matrix(x_full.reshape(-1, 1)).flatten()
        return y, is_scipy_2d

    def evaluate_batch(self, x_batch):
        """Evaluate normalized row-major candidates with shape ``(N, dim)``.

        Multi-Modal discovery works with optimizer-style row-major populations,
        while :meth:`DesignProblem.evaluate_matrix` expects ``(dim, N)``. Keep
        that conversion here so all discovery algorithms use the same parameter
        expansion and model-evaluation path as the single-space solver.
        """
        x_batch = np.asarray(x_batch, dtype=float)
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        if x_batch.ndim != 2 or x_batch.shape[1] != len(self.dv_norm):
            raise ValueError(
                f"x_batch must have shape (N, active_dimensions); got {x_batch.shape}"
            )

        x_phys = self.denormalize(x_batch.T)
        x_full = self.construct_full_vector(x_phys)
        return np.asarray(self.problem.evaluate_matrix(x_full), dtype=float)

    def compute_violation_and_margin(self, x_batch):
        """Return normalized violation and robustness margin for each candidate.

        A feasible candidate has zero violation.  Its margin is the smallest
        normalized distance to any finite requirement boundary, which gives
        basin discovery a useful ranking inside feasible regions.  Non-finite
        model outputs are treated as invalid candidates.
        """
        y = self.evaluate_batch(x_batch)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_qoi, n_samples = y.shape
        req_l = self.reqL.astype(float)
        req_u = self.reqU.astype(float)
        has_lower = np.isfinite(req_l)
        has_upper = np.isfinite(req_u)

        req_width = np.abs(req_u - req_l)
        scale_u = np.where(has_upper, np.abs(req_u), 1.0)
        scale_l = np.where(has_lower, np.abs(req_l), 1.0)
        scale_u = np.where(scale_u < 1e-12, 1.0, scale_u)
        scale_l = np.where(scale_l < 1e-12, 1.0, scale_l)
        width_u = np.where(np.isfinite(req_width), req_width, scale_u)
        width_l = np.where(np.isfinite(req_width), req_width, scale_l)
        width_u = np.where(width_u < 1e-12, 1.0, width_u)[:, None]
        width_l = np.where(width_l < 1e-12, 1.0, width_l)[:, None]

        upper_violation = np.zeros((n_qoi, n_samples), dtype=float)
        lower_violation = np.zeros((n_qoi, n_samples), dtype=float)
        if np.any(has_upper):
            upper_violation[has_upper] = np.maximum(
                0.0,
                (y[has_upper] - req_u[has_upper, None]) / width_u[has_upper],
            )
        if np.any(has_lower):
            lower_violation[has_lower] = np.maximum(
                0.0,
                (req_l[has_lower, None] - y[has_lower]) / width_l[has_lower],
            )

        invalid = ~np.isfinite(y)
        upper_violation[invalid] = 1e6
        lower_violation[invalid] = 1e6
        violation = np.sum(upper_violation + lower_violation, axis=0)
        violation = np.nan_to_num(violation, nan=1e12, posinf=1e12, neginf=1e12)

        margins = []
        if np.any(has_upper):
            margins.append((req_u[has_upper, None] - y[has_upper]) / width_u[has_upper])
        if np.any(has_lower):
            margins.append((y[has_lower] - req_l[has_lower, None]) / width_l[has_lower])
        margin = (
            np.min(np.vstack(margins), axis=0)
            if margins
            else np.zeros(n_samples, dtype=float)
        )
        margin = np.where(np.any(invalid, axis=0), -1e6, margin)
        margin = np.nan_to_num(margin, nan=-1e6, posinf=0.0, neginf=-1e6)
        return violation, margin

    def _constraint_scales(self):
        req_width = np.abs(self.reqU - self.reqL)
        scale_u = np.where(np.isfinite(self.reqU), np.abs(self.reqU), 1.0)
        scale_l = np.where(np.isfinite(self.reqL), np.abs(self.reqL), 1.0)
        scale_u = np.where(scale_u < 1e-12, 1.0, scale_u)
        scale_l = np.where(scale_l < 1e-12, 1.0, scale_l)
        w_u = np.where(np.isfinite(req_width), req_width, scale_u)
        w_l = np.where(np.isfinite(req_width), req_width, scale_l)
        w_u = np.where(w_u == 0, 1.0, w_u)
        w_l = np.where(w_l == 0, 1.0, w_l)
        return w_u, w_l

    def compute_objective(self, x_norm):
        """Compute the paper's aggregate violation ``V0(x)``.

        Each finite lower or upper requirement is represented as a normalized
        one-sided constraint ``g_j(x)``. The discovery objective is therefore
        ``V0(x) = sum_j max(0, g_j(x))`` and a good design has
        ``V0(x) < epsilon``.
        """
        y, _ = self.evaluate(x_norm)
        w_u, w_l = self._constraint_scales()

        c_upper = np.zeros_like(y, dtype=float)
        c_lower = np.zeros_like(y, dtype=float)

        mask_u = np.isfinite(self.reqU)
        if np.any(mask_u):
            req_u = self.reqU[mask_u, np.newaxis] if y.ndim == 2 else self.reqU[mask_u]
            denom = w_u[mask_u, np.newaxis] if y.ndim == 2 else w_u[mask_u]
            c_upper[mask_u] = np.maximum(0.0, (y[mask_u] - req_u) / denom)

        mask_l = np.isfinite(self.reqL)
        if np.any(mask_l):
            req_l = self.reqL[mask_l, np.newaxis] if y.ndim == 2 else self.reqL[mask_l]
            denom = w_l[mask_l, np.newaxis] if y.ndim == 2 else w_l[mask_l]
            c_lower[mask_l] = np.maximum(0.0, (req_l - y[mask_l]) / denom)

        violations = np.concatenate([c_upper, c_lower], axis=0)
        obj_sum = np.sum(np.maximum(0.0, violations), axis=0)

        if self.include_objectives and self.objective_indices:
            for idx, weight in zip(self.objective_indices, self.objective_weights):
                obj_sum += weight * y[idx]

        return np.where(np.isnan(obj_sum) | np.isinf(obj_sum), 1e10, obj_sum)

    def compute_design_objective(self, x_norm):
        """Weighted, scale-normalized design objective ONLY (no violation term).

        This is the objective for the constrained anchor search (SLSQP/Nevergrad),
        where the requirement constraints are supplied separately. Each objective
        is divided by a fixed reference magnitude so the problem is well-conditioned
        and weights are comparable across objectives of different scales — unlike
        the old path, which added the raw objective onto normalized violations and
        let a large objective (e.g. a GP cost ~17) swamp feasibility.
        """
        y, _ = self.evaluate(x_norm)
        batched = y.ndim == 2
        if not (self.include_objectives and self.objective_indices):
            return np.zeros(y.shape[1]) if batched else 0.0

        if self._obj_ref_scale is None:
            self._obj_ref_scale = {}
            for idx in self.objective_indices:
                ref = float(np.mean(np.abs(np.atleast_1d(y[idx]))))
                self._obj_ref_scale[idx] = ref if ref > 1e-12 else 1.0

        total = np.zeros(y.shape[1]) if batched else 0.0
        for idx, weight in zip(self.objective_indices, self.objective_weights):
            total = total + weight * (y[idx] / self._obj_ref_scale[idx])
        return np.where(np.isnan(total) | np.isinf(total), 1e10, total)

    def compute_constraints_normalized(self, x_norm):
        """Normalized constraints where values >= 0 indicate satisfaction."""
        y, _ = self.evaluate(x_norm)
        req_width = self.reqU - self.reqL
        req_width = np.where(np.isinf(req_width), 1.0, req_width)

        c_upper = np.zeros_like(y, dtype=float)
        c_lower = np.zeros_like(y, dtype=float)

        mask_u = np.isfinite(self.reqU)
        if np.any(mask_u):
            w_u = np.where(req_width[mask_u] == 0, 1.0, req_width[mask_u])
            req_u = self.reqU[mask_u, np.newaxis] if y.ndim == 2 else self.reqU[mask_u]
            denom = w_u[:, np.newaxis] if y.ndim == 2 else w_u
            c_upper[mask_u] = (req_u - y[mask_u]) / denom
        else:
            c_upper.fill(1e18)

        mask_l = np.isfinite(self.reqL)
        if np.any(mask_l):
            w_l = np.where(req_width[mask_l] == 0, 1.0, req_width[mask_l])
            req_l = self.reqL[mask_l, np.newaxis] if y.ndim == 2 else self.reqL[mask_l]
            denom = w_l[:, np.newaxis] if y.ndim == 2 else w_l
            c_lower[mask_l] = (y[mask_l] - req_l) / denom
        else:
            c_lower.fill(1e18)

        c_upper[~mask_u] = 1e18
        c_lower[~mask_l] = 1e18
        return np.concatenate((c_upper, c_lower))

    def compute_constraints_raw(self, x_norm):
        """Raw constraints where values >= 0 indicate satisfaction."""
        y, _ = self.evaluate(x_norm)
        c_upper = self.reqU.reshape(-1, 1) - y if y.ndim == 2 else self.reqU - y
        c_lower = y - self.reqL.reshape(-1, 1) if y.ndim == 2 else y - self.reqL
        c_upper = np.where(
            np.isinf(self.reqU.reshape(-1, 1) if y.ndim == 2 else self.reqU),
            1e6,
            c_upper,
        )
        c_lower = np.where(
            np.isinf(self.reqL.reshape(-1, 1) if y.ndim == 2 else self.reqL),
            1e6,
            c_lower,
        )
        return np.concatenate((c_upper, c_lower))

    def compute_constraints_finite_only(self, x_norm):
        """Only finite constraints, useful for solvers sensitive to infinities."""
        y, _ = self.evaluate(x_norm)
        constraints = []
        if np.any(np.isfinite(self.reqU)):
            constraints.append(
                self.reqU[np.isfinite(self.reqU)] - y[np.isfinite(self.reqU)]
            )
        if np.any(np.isfinite(self.reqL)):
            constraints.append(
                y[np.isfinite(self.reqL)] - self.reqL[np.isfinite(self.reqL)]
            )
        return np.concatenate(constraints) if constraints else np.array([0.0])
