# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Topology study models, numerical solvers, recovery, and verification."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "CellMaterialLaw",
    "JointDefinition",
    "LatticeInfillNode",
    "LatticeOptVoxelNode",
    "LoadCase",
    "ManufacturingConstraints",
    "ManufacturingStructureOptions",
    "ThermalBC",
    "ThermalLoadCase",
    "TopologyLoadNode",
    "TopologyOptVoxelNode",
    "TopologyOptVoxelProblem",
    "TopologyOptVoxelSolver",
    "TopologySupportNode",
    "VoxelBC",
    "build_manufacturing_field",
    "cell_material_law",
    "reconstruct_topopt_cad",
    "run_topopt_validation",
]

_MODELS = "pylcss.design_studio.topology_optimization.models"
_OPTIMIZATION = "pylcss.design_studio.topology_optimization.optimization"
_INTEGRATION = "pylcss.design_studio.topology_optimization.integration"
_MANUFACTURING = "pylcss.design_studio.topology_optimization.manufacturing"

_LAZY_EXPORTS = {
    "CellMaterialLaw": (_MANUFACTURING, "CellMaterialLaw"),
    "JointDefinition": (_MODELS, "JointDefinition"),
    "LatticeInfillNode": (_INTEGRATION, "LatticeInfillNode"),
    "LatticeOptVoxelNode": (_INTEGRATION, "LatticeOptVoxelNode"),
    "LoadCase": (_MODELS, "LoadCase"),
    "ManufacturingConstraints": (_MODELS, "ManufacturingConstraints"),
    "ManufacturingStructureOptions": (
        _MANUFACTURING,
        "ManufacturingStructureOptions",
    ),
    "ThermalBC": (_MODELS, "ThermalBC"),
    "ThermalLoadCase": (_MODELS, "ThermalLoadCase"),
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
        "pylcss.design_studio.topology_optimization.geometry",
        "reconstruct_topopt_cad",
    ),
    "run_topopt_validation": (
        "pylcss.design_studio.topology_optimization.verification",
        "run_topopt_validation",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
