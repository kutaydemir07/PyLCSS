# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
pylcss Import/Export Manager
============================
Professional file I/O system for CAD, simulation, optimization data.

Supported formats:
    CAD:   STEP, IGES, STL, OBJ, BREP, 3MF
    Mesh:  VTK, MSH (Gmsh), INP (Abaqus), BDF (Nastran)  
    Data:  CSV, JSON, HDF5, MAT (MATLAB), Excel
    Image: PNG, SVG, PDF (plots/reports)
    Project: .pylcss (full project archive)
"""

from pylcss.io_manager.cad_io import CADImporter, CADExporter
from pylcss.io_manager.mesh_io import MeshImporter, MeshExporter
from pylcss.io_manager.data_io import DataImporter, DataExporter
from pylcss.io_manager.project_io import ProjectManager

__all__ = [
    "CADImporter",
    "CADExporter",
    "MeshImporter",
    "MeshExporter",
    "DataImporter",
    "DataExporter",
    "ProjectManager",
]
