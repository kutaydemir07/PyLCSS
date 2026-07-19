# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 - Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universitaet Bochum

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .multimodal_models import BoxSolutionSpace

logger = logging.getLogger(__name__)


class RedundancyResolutionMixin:
    """Remove redundant boxes before decoupling is attempted.

    Requires the host solver to expose ``self.dv_norm`` (per-dim design-space
    widths). Two boxes are redundant when they intersect in every design
    variable; each connected component of that intersection graph keeps only
    its largest-volume member.
    """

    def _resolve_redundant_solution_spaces(
        self,
        boxes: List[BoxSolutionSpace],
        callback: Optional[Any] = None,
    ) -> List[BoxSolutionSpace]:
        """Resolve overlap-induced redundancy between expanded boxes."""
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
        boxes: List[BoxSolutionSpace],
        callback: Optional[Any] = None,
    ) -> List[BoxSolutionSpace]:
        """Keep only the largest box per connected intersection component."""
        K = len(boxes)
        if K <= 1:
            return list(boxes)

        lower = np.array([box.bounds[:, 0] for box in boxes])
        upper = np.array([box.bounds[:, 1] for box in boxes])
        inter_min = np.maximum(lower[:, None, :], lower[None, :, :])
        inter_max = np.minimum(upper[:, None, :], upper[None, :, :])
        intersect = np.all(inter_max - inter_min > 0.0, axis=2)
        np.fill_diagonal(intersect, False)

        parent = list(range(K))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(K):
            for j in range(i + 1, K):
                if not intersect[i, j]:
                    continue
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

        components: Dict[int, List[int]] = {}
        for i in range(K):
            components.setdefault(find(i), []).append(i)

        kept: List[BoxSolutionSpace] = []
        for members in components.values():
            members.sort(key=lambda idx: boxes[idx].volume, reverse=True)
            winner = boxes[members[0]]
            kept.append(winner)
            for idx in members[1:]:
                msg = (
                    "Stage 4 - Redundancy: removed solution space "
                    f"(volume={boxes[idx].volume:.3e}); retained the largest "
                    f"in its overlap set (volume={winner.volume:.3e})."
                )
                logger.info(msg)
                if callback:
                    callback(None, None, msg)

        kept.sort(key=lambda box: box.volume, reverse=True)
        return kept

    # Compatibility for code written before the paper's five-stage naming was
    # adopted in PyLCSS.
    def _phase3_redundancy_resolution(self, boxes, callback=None):
        return self._resolve_redundant_solution_spaces(boxes, callback=callback)
