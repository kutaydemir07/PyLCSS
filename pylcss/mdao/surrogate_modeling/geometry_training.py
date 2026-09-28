# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility exports for geometry-aware training."""

from .geometry_data import GeometrySample, GeometryScaling, select_torch_device
from .geometry_strategies import GeomDeepONetStrategy, GINOStrategy

__all__ = [
    "GINOStrategy",
    "GeomDeepONetStrategy",
    "GeometrySample",
    "GeometryScaling",
    "select_torch_device",
]
