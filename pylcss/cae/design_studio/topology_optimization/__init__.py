# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Topology study models, numerical solvers, recovery, and verification."""

from __future__ import annotations

from pylcss.cae.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "CellMaterialLaw",
    "JointDefinition",
    "LatticeInfillNode",
    "LoadCase",
    "ManufacturingConstraints",
    "ManufacturingStructureOptions",
    "TopologyLoadNode",
    "TopologyOptVoxelNode",
    "TopologyOptVoxelProblem",
    "TopologyOptVoxelSolver",
    "TopologySupportNode",
    "VoxelBC",
    "build_manufacturing_field",
    "cell_material_law",
    "reconstruct_topopt_cad",
    "run_topopt_fea",
]

_MODELS = "pylcss.cae.design_studio.topology_optimization.models"
_OPTIMIZATION = "pylcss.cae.design_studio.topology_optimization.optimization"
_INTEGRATION = "pylcss.cae.design_studio.topology_optimization.integration"
_MANUFACTURING = "pylcss.cae.design_studio.topology_optimization.manufacturing"

_LAZY_EXPORTS = {
    "CellMaterialLaw": (_MANUFACTURING, "CellMaterialLaw"),
    "JointDefinition": (_MODELS, "JointDefinition"),
    "LatticeInfillNode": (_INTEGRATION, "LatticeInfillNode"),
    "LoadCase": (_MODELS, "LoadCase"),
    "ManufacturingConstraints": (_MODELS, "ManufacturingConstraints"),
    "ManufacturingStructureOptions": (
        _MANUFACTURING,
        "ManufacturingStructureOptions",
    ),
    "TopologyLoadNode": (_INTEGRATION, "TopologyLoadNode"),
    "TopologyOptVoxelNode": (_INTEGRATION, "TopologyOptVoxelNode"),
    "TopologyOptVoxelProblem": (
        _OPTIMIZATION,
        "TopologyOptVoxelProblem",
    ),
    "TopologyOptVoxelSolver": (
        _OPTIMIZATION,
        "TopologyOptVoxelSolver",
    ),
    "TopologySupportNode": (_INTEGRATION, "TopologySupportNode"),
    "VoxelBC": (_MODELS, "VoxelBC"),
    "build_manufacturing_field": (
        _MANUFACTURING,
        "build_manufacturing_field",
    ),
    "reconstruct_topopt_cad": (
        "pylcss.cae.design_studio.topology_optimization.geometry",
        "reconstruct_topopt_cad",
    ),
    "run_topopt_fea": (
        "pylcss.cae.design_studio.topology_optimization.verification",
        "run_topopt_fea",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
