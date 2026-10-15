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
    else:
        regions = dict(regions)
    recovered = payload.get("recovered_shape") or {}
    if isinstance(recovered, dict):
        regions.update(recovered.get("preserved_lattice_geometry") or {})
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
    lattice_geometry = payload.get("source_lattice_geometry")
    if lattice_geometry is None and payload.get("generated_lattice"):
        from ..integration.post_lattice import lattice_base_geometry, topology_source

        lattice_geometry, _ = lattice_base_geometry(topology_source(payload))
    if lattice_geometry is None and payload.get("study_kind") == "lattice_infill":
        lattice_geometry = payload.get("source_cad_shape")
        if lattice_geometry is None:
            raise ValueError("The infill result needs its original CAD/STL boundary to regenerate cells.")
    constraints.update(
        source_mask=source_mask,
        source_cad_shape=payload.get("source_cad_shape"),
        source_lattice_geometry=lattice_geometry,
        passive_solid_mask=payload.get("passive_solid_mask"),
        passive_void_mask=payload.get("passive_void_mask"),
    )
    return constraints
