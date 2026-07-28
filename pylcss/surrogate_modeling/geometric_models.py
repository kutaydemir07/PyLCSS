# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility exports for the renamed neural-operator module."""

from .operators import GeomDeepONet, GINONet, SIRENLayer, SIRENNet

__all__ = ["GINONet", "GeomDeepONet", "SIRENLayer", "SIRENNet"]
