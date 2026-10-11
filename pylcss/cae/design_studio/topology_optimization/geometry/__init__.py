# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Surface recovery and CAD reconstruction."""

from __future__ import annotations

from pylcss.cae.design_studio._lazy_imports import load_attribute, public_names

__all__ = ["_recover_voxel_shape", "reconstruct_topopt_cad", "export_cad_model",
           "CadReconstructionConfig", "SurfaceRecoveryConfig"]

_LAZY_EXPORTS = {
    "CadReconstructionConfig": (
        "pylcss.cae.design_studio.topology_optimization.geometry.cad_config",
        "CadReconstructionConfig",
    ),
    "SurfaceRecoveryConfig": (
        "pylcss.cae.design_studio.topology_optimization.geometry.cad_config",
        "SurfaceRecoveryConfig",
    ),
    "_recover_voxel_shape": (
        "pylcss.cae.design_studio.topology_optimization.geometry.surface_recovery",
        "_recover_voxel_shape",
    ),
    "reconstruct_topopt_cad": (
        "pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction",
        "reconstruct_topopt_cad",
    ),
    "export_cad_model": (
        "pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction",
        "export_cad_model",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
