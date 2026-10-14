# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Construction and combination of evaluated sample batches."""

from collections.abc import Sequence

import numpy as np

from .contracts import SampleBatch


def empty_samples(n_dimensions: int, n_qoi: int) -> SampleBatch:
    return {
        "points": np.zeros((n_dimensions, 0)),
        "is_good": np.zeros(0, dtype=bool),
        "is_bad": np.zeros(0, dtype=bool),
        "qoi_values": np.zeros((n_qoi, 0)),
        "violation_idx": np.zeros(0, dtype=int),
    }


def merge_samples(
    batches: Sequence[SampleBatch], *, bad_first: bool = False
) -> SampleBatch:
    if not batches:
        raise ValueError("at least one sample batch is required")
    merged = {
        "points": np.hstack([batch["points"] for batch in batches]),
        "qoi_values": np.hstack([batch["qoi_values"] for batch in batches]),
        "is_good": np.concatenate([batch["is_good"] for batch in batches]),
        "is_bad": np.concatenate([batch["is_bad"] for batch in batches]),
        "violation_idx": np.concatenate([batch["violation_idx"] for batch in batches]),
    }
    if bad_first:
        order = np.argsort(merged["is_good"].astype(np.int8), kind="stable")
        merged = {
            key: values[:, order] if values.ndim == 2 else values[order]
            for key, values in merged.items()
        }
    return merged
