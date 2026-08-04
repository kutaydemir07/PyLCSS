# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Shared file I/O for CAD, scientific data, reports, and project JSON.

The package deliberately exposes a small facade. Format-specific parsing and
atomic-write mechanics remain private so callers have one stable import point.
"""

from pylcss.io_manager._mesh import MeshData
from pylcss.io_manager.cad_io import CADImporter
from pylcss.io_manager.data_io import DataExporter
from pylcss.io_manager.project_io import atomic_json_dump, load_json_object

__all__ = [
    "CADImporter",
    "DataExporter",
    "MeshData",
    "atomic_json_dump",
    "load_json_object",
]
