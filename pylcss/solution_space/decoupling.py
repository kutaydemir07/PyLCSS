# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 - Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universitaet Bochum

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np

from .multimodal_models import BoxSolutionSpace, DecoupledMultiModalForm
from .bayesian import good_fraction_lower_bound
from .monte_carlo import classify_good_bad

logger = logging.getLogger(__name__)


class DecouplingResolutionMixin:
    """Stage-5 decoupling and extended-problem refinement for MMSS modes."""

    def _compute_decoupled_form(
        self,
        boxes: List[BoxSolutionSpace],
        callback: Optional[Any] = None,
        stop_callback: Optional[Any] = None,
    ) -> DecoupledMultiModalForm:
        """Compute the paper's decoupled form from the retained modal boxes."""
        from .extended import (
            build_extended_layout,
            extended_initial_bounds,
            project_extended_to_modes,
            run_extended_refinement,
        )

        if len(boxes) < 2:
            return DecoupledMultiModalForm(mode_boxes=list(boxes))

        params = self.params
        n_dims_orig = boxes[0].bounds.shape[0]
        sample_size = max(50, int(params.optimization_sample_size))
        growth_rate = float(getattr(params, "phase1_growth_rate", 0.05))
        target_a = float(params.target_good_fraction)
        confidence = float(params.good_fraction_confidence)
        phase1_max = int(params.max_iterations)
        phase2_max = int(getattr(params, "phase2_max_iterations", 0) or phase1_max)
        phase1_tol = float(getattr(params, "phase1_convergence_tol", 1e-4))

        spec = self._select_decoupling_spec(boxes)
        if spec is None:
            return DecoupledMultiModalForm(mode_boxes=list(boxes))

        selected_indices = spec["indices"]
        selected_boxes = [boxes[i] for i in selected_indices]
        shared_groups = self._shared_groups_from_decoupling_spec(spec)

        if callback:
            mode_labels = ", ".join(str(boxes[i].box_id + 1) for i in selected_indices)
            callback(
                None,
                None,
                "Stage 5 - Decoupling: selected mode(s) "
                f"{mode_labels} with {len(spec['common_dims'])} common and "
                f"{len(spec['separated_dims'])} separating variable(s).",
            )

        if not shared_groups:
            return self._make_decoupling_family(
                selected_boxes,
                tuple(range(len(selected_boxes))),
                spec,
                validation=None,
            )

        K = len(selected_boxes)
        layout = build_extended_layout(K, n_dims_orig, shared_groups)
        if layout.is_trivial():
            return self._make_decoupling_family(
                selected_boxes,
                tuple(range(len(selected_boxes))),
                spec,
                validation=None,
            )

        initial_extended_bounds = extended_initial_bounds(
            layout, selected_boxes, shared_groups
        )

        if callback:
            n_groups = sum(len(v) for v in shared_groups.values())
            callback(
                None,
                None,
                "Stage 5 - Decoupling: refining the extended problem "
                f"(n_z={layout.n_z}, K={K}, {n_groups} common group(s))...",
            )

        result = run_extended_refinement(
            base_problem=self.problem,
            layout=layout,
            initial_extended_bounds=initial_extended_bounds,
            dsl_orig=self.dsl,
            dsu_orig=self.dsu,
            reqL=self.reqL,
            reqU=self.reqU,
            parameters_orig=self.parameters,
            ind_parameters_orig=self.ind_parameters,
            sample_size=sample_size,
            growth_rate=growth_rate,
            target_good_fraction=target_a,
            confidence=confidence,
            phase1_max_iterations=phase1_max,
            phase2_max_iterations=phase2_max,
            phase1_convergence_tol=phase1_tol,
            callback=callback,
            stop_callback=stop_callback,
        )
        if result is None:
            if callback:
                callback(
                    None,
                    None,
                    "Stage 5 - Decoupling: extended-problem refinement failed; "
                    "keeping the pre-refinement decoupled form.",
                )
            return self._make_decoupling_family(
                selected_boxes,
                tuple(range(len(selected_boxes))),
                spec,
                validation=None,
            )

        projected = project_extended_to_modes(
            result.bounds, layout, selected_boxes
        )
        for k, box in enumerate(projected):
            box.compute_volume(self.dv_norm)
            box.label = box.label or f"Mode {k + 1} (decoupled)"

        refined_spec = self._decoupling_spec_for_subset(
            projected,
            tuple(range(len(projected))),
        )
        if refined_spec is None:
            return self._make_decoupling_family(
                selected_boxes,
                tuple(range(len(selected_boxes))),
                spec,
                validation=None,
            )

        validation = None
        if len(refined_spec["separated_dims"]) == 1:
            validation = self._validate_decoupled_interval_unions(
                refined_spec["rows"],
                sample_size=sample_size,
                confidence=confidence,
            )
            lower, good_frac, m, N = validation
            if lower < target_a:
                if callback:
                    callback(
                        None,
                        None,
                        "Stage 5 - Decoupling: the one-separating-variable "
                        "interval product did not reach "
                        f"target a*={target_a:.4f} "
                        f"(a={good_frac:.4f}, a_l={lower:.4f}, {m}/{N}); "
                        "keeping the modal boxes.",
                    )
                validation = None

        family = self._make_decoupling_family(
            projected,
            tuple(range(len(projected))),
            refined_spec,
            validation=validation,
        )

        if callback:
            callback(
                None,
                None,
                "Stage 5 - Decoupling: complete "
                f"(extended a_l={result.good_fraction_lower_bound:.4f}, "
                f"{family.message}, "
                f"P1 iters={result.phase1_iters}, P2 iters={result.phase2_iters}).",
            )
        return family

    def _select_decoupling_spec(
        self,
        boxes: List[BoxSolutionSpace],
    ) -> Optional[Dict[str, Any]]:
        """Try 1 separated dimension, then 2, and so on."""
        K = len(boxes)
        if K < 2:
            return None

        n_dims = boxes[0].bounds.shape[0]
        subsets = self._candidate_subsets(K)

        for n_separated in range(1, n_dims + 1):
            candidates: List[Dict[str, Any]] = []
            for indices in subsets:
                spec = self._decoupling_spec_for_subset(boxes, indices)
                if spec is None:
                    continue
                if len(spec["separated_dims"]) == n_separated:
                    candidates.append(spec)

            if candidates:
                candidates.sort(
                    key=lambda item: (
                        len(item["indices"]),
                        item["score"],
                    ),
                    reverse=True,
                )
                return candidates[0]

        return None

    @staticmethod
    def _candidate_subsets(K: int) -> List[tuple]:
        """Return branch subsets to inspect for decoupling."""
        if K < 2:
            return []
        if K <= 10:
            return [
                subset
                for r in range(2, K + 1)
                for subset in combinations(range(K), r)
            ]

        # Avoid combinatorial blow-up for unusually large K. The default Kmax
        # is small, but keep a bounded fallback for interactive use.
        order = list(range(K))
        subsets = [tuple(order)]
        for i in range(min(K, 12)):
            for j in range(i + 1, min(K, 12)):
                subsets.append((order[i], order[j]))
        return subsets

    def _decoupling_spec_for_subset(
        self,
        boxes: List[BoxSolutionSpace],
        indices: tuple,
    ) -> Optional[Dict[str, Any]]:
        """Classify one mode subset into common and separating dimensions."""
        if len(indices) < 2:
            return None

        n_dims = boxes[0].bounds.shape[0]
        common_dims: List[int] = []
        separated_dims: List[int] = []
        rows: List[List[tuple]] = []

        for j in range(n_dims):
            lows = [float(boxes[i].bounds[j, 0]) for i in indices]
            highs = [float(boxes[i].bounds[j, 1]) for i in indices]
            inter_lo = max(lows)
            inter_hi = min(highs)

            if inter_hi - inter_lo > 1e-12:
                common_dims.append(j)
                rows.append([(inter_lo, inter_hi)])
                continue

            separated_dims.append(j)
            rows.append(self._merge_intervals(list(zip(lows, highs))))

        if not separated_dims:
            return None

        score = self._decoupling_score(boxes, indices, rows, separated_dims)
        return {
            "indices": tuple(indices),
            "common_dims": common_dims,
            "separated_dims": separated_dims,
            "rows": rows,
            "score": score,
        }

    @staticmethod
    def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
        """Merge overlapping one-dimensional intervals."""
        merged: List[tuple] = []
        for lo, hi in sorted(intervals):
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                continue
            if merged and lo <= merged[-1][1] + 1e-12:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def _decoupling_score(
        self,
        boxes: List[BoxSolutionSpace],
        indices: tuple,
        rows: List[List[tuple]],
        separated_dims: List[int],
    ) -> float:
        """Score candidates within the same separated-dimension count."""
        if len(separated_dims) == 1:
            return self._interval_rows_volume(rows)
        return float(sum(max(0.0, boxes[i].volume) for i in indices))

    def _shared_groups_from_decoupling_spec(
        self,
        spec: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Build extended shared coordinates for dimensions common to the subset."""
        n_branches = len(spec["indices"])
        all_branches = list(range(n_branches))
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for j in spec["common_dims"]:
            lo, hi = spec["rows"][j][0]
            groups[j] = [{
                "branches": all_branches,
                "bounds": np.array([lo, hi], dtype=float),
            }]
        return groups

    def _make_decoupling_family(
        self,
        boxes: List[BoxSolutionSpace],
        indices: tuple,
        spec: Dict[str, Any],
        validation: Optional[tuple] = None,
    ) -> DecoupledMultiModalForm:
        """Create the user-facing decoupling result."""
        selected_boxes = [boxes[i] for i in indices]
        common_dims = list(spec["common_dims"])
        separated_dims = list(spec["separated_dims"])
        mode_labels = ", ".join(str(box.box_id + 1) for box in selected_boxes)

        if len(separated_dims) == 1 and validation is not None:
            lower, good_frac, m, N = validation
            message = (
                f"decoupled interval product from mode(s) {mode_labels}: "
                f"a={good_frac:.4f}, a_l={lower:.4f} ({m}/{N})"
            )
            interval_unions = spec["rows"]
            volume = self._interval_rows_volume(spec["rows"])
        else:
            message = (
                f"decoupled form from mode(s) {mode_labels}: "
                f"{len(common_dims)} common dimension(s), "
                f"{len(separated_dims)} separating dimension(s)"
            )
            interval_unions = []
            volume = float(sum(max(0.0, box.volume) for box in selected_boxes))

        return DecoupledMultiModalForm(
            common_variable_indices=common_dims,
            separating_variable_indices=separated_dims,
            mode_boxes=selected_boxes,
            label="Decoupled multi-modal form",
            message=message,
            volume=volume,
            common_groups_per_dim=self._shared_groups_from_decoupling_spec(spec),
            interval_unions=interval_unions,
        )

    # Compatibility for the initial port, which treated decoupling as a third
    # phase instead of Stage 5 of the paper's MMSS algorithm.

    def _interval_rows_volume(self, interval_rows: List[List[tuple]]) -> float:
        """Relative volume of a Cartesian product of per-dimension interval rows."""
        if not interval_rows:
            return 0.0

        active = np.asarray(self.dv_norm, dtype=float) > 1e-20
        rel_widths: List[float] = []
        for j, intervals in enumerate(interval_rows):
            if j >= active.size or not active[j]:
                continue
            width = sum(max(0.0, hi - lo) for lo, hi in intervals)
            rel_widths.append(width / max(float(self.dv_norm[j]), 1e-20))
        if not rel_widths:
            return 0.0
        return float(np.prod(np.maximum(rel_widths, 0.0)))

    @staticmethod
    def _sample_interval_rows(
        interval_rows: List[List[tuple]],
        n: int,
    ) -> np.ndarray:
        """Sample uniformly from a Cartesian product of interval rows."""
        dim = len(interval_rows)
        samples = np.zeros((dim, n), dtype=float)
        rng = np.random.default_rng()

        for j, intervals in enumerate(interval_rows):
            clean = [(float(lo), float(hi)) for lo, hi in intervals if float(hi) > float(lo)]
            if not clean:
                continue
            widths = np.array([hi - lo for lo, hi in clean], dtype=float)
            probs = widths / widths.sum()
            choices = rng.choice(len(clean), size=n, p=probs)
            unit = rng.random(n)
            for idx, (lo, hi) in enumerate(clean):
                mask = choices == idx
                if np.any(mask):
                    samples[j, mask] = lo + (hi - lo) * unit[mask]

        return samples

    def _validate_decoupled_interval_unions(
        self,
        interval_rows: List[List[tuple]],
        sample_size: int,
        confidence: float,
    ) -> tuple:
        """Validate a one-separated-variable decoupled interval product."""
        if not interval_rows:
            return 0.0, 0.0, 0, 0

        x_design = self._sample_interval_rows(interval_rows, int(sample_size))
        _y, good, _bad, _c_min, _viol = classify_good_bad(
            self.problem,
            x_design,
            self.parameters,
            self.ind_parameters,
            self.reqL,
            self.reqU,
        )
        N = int(good.size)
        m = int(np.sum(good))
        good_frac = (m / N) if N else 0.0
        lower = good_fraction_lower_bound(m, N, confidence) if N else 0.0
        return float(lower), float(good_frac), m, N
