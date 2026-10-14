# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 - Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universitaet Bochum

from __future__ import annotations

from typing import Optional

import numpy as np

from .contracts import FloatArray, ProgressCallback
from .models import BoxSolutionSpace


class RedundancyResolutionMixin:
    """Retain one representative per connected group of overlapping modal boxes.

    Requires the host solver to expose ``self.dv_norm`` (per-dim design-space
    widths). Positive overlap is tested in active design coordinates; fixed
    and uncertain coordinates must agree. The largest box represents each
    overlap group, following Stage 4 of the multi-modal algorithm.
    """

    dv_norm: FloatArray

    def _resolve_redundant_solution_spaces(
        self,
        boxes: list[BoxSolutionSpace],
        callback: Optional[ProgressCallback] = None,
    ) -> list[BoxSolutionSpace]:
        """Resolve overlap groups and assign IDs to the retained modes."""
        if len(boxes) <= 1:
            return list(boxes)

        for box in boxes:
            box.compute_volume(self.dv_norm)

        boxes = sorted(boxes, key=lambda box: box.volume, reverse=True)
        boxes = self._remove_redundant_boxes(boxes, callback=callback)

        for i, box in enumerate(boxes):
            box.box_id = i
            box.label = box.label or f"Mode {i + 1}"
        return boxes

    def _remove_redundant_boxes(
        self,
        boxes: list[BoxSolutionSpace],
        callback: Optional[ProgressCallback] = None,
    ) -> list[BoxSolutionSpace]:
        """Keep the largest box from each connected overlap component."""
        if len(boxes) <= 1:
            return list(boxes)
        lower = np.array([box.bounds[:, 0] for box in boxes])
        upper = np.array([box.bounds[:, 1] for box in boxes])
        active = np.asarray(self.dv_norm) > 0.0
        intersection_lower = np.maximum(lower[:, None, :], lower[None, :, :])
        intersection_upper = np.minimum(upper[:, None, :], upper[None, :, :])
        overlap = np.all(
            intersection_upper[:, :, active] > intersection_lower[:, :, active],
            axis=2,
        )
        inactive = ~active
        overlap &= np.all(
            lower[:, None, inactive] == lower[None, :, inactive], axis=2
        ) & np.all(upper[:, None, inactive] == upper[None, :, inactive], axis=2)
        parent = list(range(len(boxes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                if overlap[left, right]:
                    parent[find(left)] = find(right)
        components: dict[int, list[BoxSolutionSpace]] = {}
        for index, box in enumerate(boxes):
            components.setdefault(find(index), []).append(box)
        retained = []
        for component in components.values():
            winner = max(component, key=lambda box: box.volume)
            retained.append(winner)
            if callback and len(component) > 1:
                callback(
                    None,
                    None,
                    f"Stage 4 - Redundancy: retained the largest of "
                    f"{len(component)} overlapping boxes (volume={winner.volume:.3e}).",
                )
        return sorted(retained, key=lambda box: box.volume, reverse=True)
