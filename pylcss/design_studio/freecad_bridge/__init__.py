# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""FreeCAD subprocess integration and BREP document synchronization.

FreeCAD owns a PySide2 application while PyLCSS uses PySide6, so the bridge
keeps FreeCAD in a separate process and exchanges files instead of embedding
its Python runtime.
"""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "FCStdWatcher",
    "FreeCadImportedShape",
    "FreeCadLauncher",
    "collect_param_values_from_node",
    "find_freecad_cmd",
    "find_freecad_executable",
    "find_freecad_python",
    "freecad_data_dir",
    "is_freecad_installed",
    "read_brep_from_fcstd",
    "write_parameters_to_fcstd",
]

_PATHS = "pylcss.design_studio.freecad_bridge.paths"
_LAZY_EXPORTS = {
    "FCStdWatcher": (
        "pylcss.design_studio.freecad_bridge.watcher",
        "FCStdWatcher",
    ),
    "FreeCadImportedShape": (
        "pylcss.design_studio.freecad_bridge.brep_reader",
        "FreeCadImportedShape",
    ),
    "FreeCadLauncher": (
        "pylcss.design_studio.freecad_bridge.launcher",
        "FreeCadLauncher",
    ),
    "collect_param_values_from_node": (
        "pylcss.design_studio.freecad_bridge.param_writer",
        "collect_param_values_from_node",
    ),
    "find_freecad_cmd": (_PATHS, "find_freecad_cmd"),
    "find_freecad_executable": (_PATHS, "find_freecad_executable"),
    "find_freecad_python": (_PATHS, "find_freecad_python"),
    "freecad_data_dir": (_PATHS, "freecad_data_dir"),
    "is_freecad_installed": (_PATHS, "is_freecad_installed"),
    "read_brep_from_fcstd": (
        "pylcss.design_studio.freecad_bridge.brep_reader",
        "read_brep_from_fcstd",
    ),
    "write_parameters_to_fcstd": (
        "pylcss.design_studio.freecad_bridge.param_writer",
        "write_parameters_to_fcstd",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
