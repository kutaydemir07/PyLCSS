# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
PyLCSS <-> FreeCAD bridge.

This package launches FreeCAD as a *subprocess* (NOT embedded) because
FreeCAD owns its own QApplication on PySide2 while PyLCSS runs PySide6 --
the two cannot coexist in one process.

The flow we support:

  1. A ``com.cad.freecad_part`` node owns a ``.FCStd`` file under
     ``<repo>/data_freecad/<node_id>.FCStd``.
  2. Double-clicking the node opens FreeCAD on that file via
     :class:`FreeCadLauncher`.  A start-up macro is injected so that
     every save automatically exports a sibling ``.brep`` + ``.json``
     describing the document's part body + named selections.
  3. A :class:`FCStdWatcher` (QFileSystemWatcher under the hood) notices
     the save, the ``.brep`` is re-read into PyLCSS's viewer, and any
     named faces / FEM constraint definitions are translated into the
     existing ``com.cad.sim.*`` node graph.

This POC focuses on steps 1 and 2; step 3's BREP reader and FEM
translator land in follow-up modules under the same package.
"""

from pylcss.cad.freecad_bridge.paths import (
    find_freecad_executable,
    find_freecad_cmd,
    find_freecad_python,
    freecad_data_dir,
    is_freecad_installed,
)
from pylcss.cad.freecad_bridge.launcher import FreeCadLauncher
from pylcss.cad.freecad_bridge.watcher import FCStdWatcher
from pylcss.cad.freecad_bridge.brep_reader import (
    FreeCadImportedShape,
    read_brep_from_fcstd,
)
from pylcss.cad.freecad_bridge.fem_translator import translate_fem_summary
from pylcss.cad.freecad_bridge.param_writer import (
    collect_param_values_from_node,
    write_parameters_to_fcstd,
)

__all__ = [
    "find_freecad_executable",
    "find_freecad_cmd",
    "find_freecad_python",
    "freecad_data_dir",
    "is_freecad_installed",
    "FreeCadLauncher",
    "FCStdWatcher",
    "FreeCadImportedShape",
    "read_brep_from_fcstd",
    "translate_fem_summary",
    "collect_param_values_from_node",
    "write_parameters_to_fcstd",
]
