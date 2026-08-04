# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Surface recovery and CAD reconstruction."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = ["_recover_voxel_shape", "reconstruct_topopt_cad"]

_LAZY_EXPORTS = {
    "_recover_voxel_shape": (
        "pylcss.design_studio.topology_optimization.geometry.surface_recovery",
        "_recover_voxel_shape",
    ),
    "reconstruct_topopt_cad": (
        "pylcss.design_studio.topology_optimization.geometry.cad_reconstruction",
        "reconstruct_topopt_cad",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
