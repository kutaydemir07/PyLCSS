# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared type contracts for surrogate training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ProgressCallback: TypeAlias = Callable[[int, str], None]
StopFlag: TypeAlias = Callable[[], bool]
LossCallback: TypeAlias = Callable[[dict[str, float | int]], None]
Metrics: TypeAlias = dict[str, Any]


class SpyPort(TypedDict):
    """Named input or output exposed by generated spy-model code."""

    name: str
