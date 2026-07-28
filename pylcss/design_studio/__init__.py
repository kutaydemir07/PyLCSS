# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Node-based CAD and engineering simulation components.

Package structure:
    ``core``                    shared node and graph contracts
    ``nodes``                   CAD authoring and selection nodes
    ``fem``                     meshing, loads, and CalculiX studies
    ``crash``                   impact setup and OpenRadioss studies
    ``topology_optimization``   topology studies and shape recovery
    ``freecad_bridge``          FreeCAD document synchronization
    ``engine``                  dependency-aware graph execution
    ``runtime``                 headless project execution

The node registries are loaded lazily so importing a focused module such as
``pylcss.design_studio.engine`` does not initialize Qt and every solver backend.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pylcss.design_studio._lazy_imports import load_attribute, public_names

try:
    __version__ = version("pylcss")
except PackageNotFoundError:  # Source tree without installed package metadata.
    __version__ = "0+unknown"

__author__ = "Kutay Demir"

__all__ = ["NODE_CLASS_MAPPING", "NODE_NAME_MAPPING"]

_LAZY_EXPORTS = {
    "NODE_CLASS_MAPPING": (
        "pylcss.design_studio.node_library",
        "NODE_CLASS_MAPPING",
    ),
    "NODE_NAME_MAPPING": (
        "pylcss.design_studio.node_library",
        "NODE_NAME_MAPPING",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
