# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""FreeCAD subprocess integration and BREP document synchronization.

FreeCAD runs in a separate process so its GUI and Python dependencies remain
independent of PyLCSS. Geometry and parameters are exchanged through files.
The export hook uses the modern FreeCAD document and spreadsheet APIs.
"""

from __future__ import annotations

from pylcss.cae.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "FCStdWatcher",
    "FreeCadImportedShape",
    "FreeCadLauncher",
    "find_freecad_cmd",
    "find_freecad_executable",
    "freecad_data_dir",
    "read_brep_from_fcstd",
    "write_parameters_to_fcstd",
]

_PATHS = "pylcss.cae.design_studio.freecad_bridge.paths"
_LAZY_EXPORTS = {
    "FCStdWatcher": (
        "pylcss.cae.design_studio.freecad_bridge.watcher",
        "FCStdWatcher",
    ),
    "FreeCadImportedShape": (
        "pylcss.cae.design_studio.freecad_bridge.brep_reader",
        "FreeCadImportedShape",
    ),
    "FreeCadLauncher": (
        "pylcss.cae.design_studio.freecad_bridge.launcher",
        "FreeCadLauncher",
    ),
    "find_freecad_cmd": (_PATHS, "find_freecad_cmd"),
    "find_freecad_executable": (_PATHS, "find_freecad_executable"),
    "freecad_data_dir": (_PATHS, "freecad_data_dir"),
    "read_brep_from_fcstd": (
        "pylcss.cae.design_studio.freecad_bridge.brep_reader",
        "read_brep_from_fcstd",
    ),
    "write_parameters_to_fcstd": (
        "pylcss.cae.design_studio.freecad_bridge.param_writer",
        "write_parameters_to_fcstd",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
