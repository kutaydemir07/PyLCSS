# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared boundary constraints for recovering a saved topology result."""

from __future__ import annotations

from typing import Any


def surface_recovery_constraints(payload: dict[str, Any]) -> dict[str, Any]:
    """Forward original CAD and voxel constraints through every recovery path."""
    regions = payload.get("passive_regions")
    if not isinstance(regions, dict):
        regions = {}
    constraints = {
        name: regions.get(name, payload.get(name)) or ()
        for name in (
            "solid_boxes", "void_boxes", "solid_cylinders", "void_cylinders",
            "joint_pin_cylinders", "explicit_solid_shapes", "explicit_void_shapes",
        )
    }
    source_mask = payload.get("source_mask")
    if source_mask is None:
        source_mask = payload.get("design_domain")
    constraints.update(
        source_mask=source_mask,
        source_cad_shape=payload.get("source_cad_shape"),
        passive_solid_mask=payload.get("passive_solid_mask"),
        passive_void_mask=payload.get("passive_void_mask"),
    )
    return constraints
