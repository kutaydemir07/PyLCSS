# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Mesh file import and export for FEA simulation workflows.
Supports: VTK, MSH (Gmsh), INP (Abaqus), BDF (Nastran), MED
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MESH_IMPORT_FORMATS = {
    ".vtk": "VTK Legacy",
    ".vtu": "VTK Unstructured",
    ".msh": "Gmsh",
    ".inp": "Abaqus Input",
    ".bdf": "Nastran Bulk Data",
    ".med": "MED (Salome)",
    ".mesh": "Medit",
    ".xdmf": "XDMF/HDF5",
}


class MeshImporter:
    """Import mesh from various FEA formats."""

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        return dict(MESH_IMPORT_FORMATS)

    @staticmethod
    def get_filter_string() -> str:
        all_exts = " ".join(f"*{ext}" for ext in MESH_IMPORT_FORMATS)
        parts = [f"All Mesh Files ({all_exts})"]
        for ext, name in MESH_IMPORT_FORMATS.items():
            parts.append(f"{name} (*{ext})")
        return ";;".join(parts)

    @staticmethod
    def import_file(filepath: str, **kwargs) -> Dict:
        """
        Import mesh file via meshio.
        
        Returns dict with:
            - points: (N, 3) node coordinates
            - cells: list of cell blocks
            - point_data: nodal field data
            - cell_data: element field data
            - field_data: named sets
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio required for mesh import: pip install meshio")

        mesh = meshio.read(filepath)
        logger.info(
            f"Imported mesh: {len(mesh.points)} nodes, "
            f"{sum(len(c.data) for c in mesh.cells)} elements from {filepath}"
        )

        return {
            "points": mesh.points,
            "cells": [{"type": c.type, "data": c.data} for c in mesh.cells],
            "point_data": dict(mesh.point_data),
            "cell_data": dict(mesh.cell_data),
            "field_data": dict(mesh.field_data) if mesh.field_data else {},
            "point_sets": dict(mesh.point_sets) if mesh.point_sets else {},
            "cell_sets": dict(mesh.cell_sets) if mesh.cell_sets else {},
        }

    @staticmethod
    def to_skfem_mesh(mesh_data: Dict):
        """Convert imported mesh to scikit-fem mesh object."""
        from skfem import MeshTet, MeshTri

        points = mesh_data["points"]
        for cell_block in mesh_data["cells"]:
            if cell_block["type"] == "tetra":
                return MeshTet(points.T, cell_block["data"].T)
            elif cell_block["type"] == "triangle":
                return MeshTri(points[:, :2].T, cell_block["data"].T)

        raise ValueError("No supported element types found in mesh")


class MeshExporter:
    """Export mesh to various FEA formats."""

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        return dict(MESH_IMPORT_FORMATS)  # meshio supports read/write

    @staticmethod
    def get_filter_string() -> str:
        parts = []
        for ext, name in MESH_IMPORT_FORMATS.items():
            parts.append(f"{name} (*{ext})")
        return ";;".join(parts)

    @staticmethod
    def export_file(
        filepath: str,
        points: np.ndarray,
        cells: list,
        point_data: Optional[Dict] = None,
        cell_data: Optional[Dict] = None,
    ) -> None:
        """
        Export mesh to file via meshio.
        
        Args:
            filepath: output path (format detected from extension)
            points: (N, 3) node coordinates
            cells: list of (cell_type, connectivity_array) tuples
            point_data: dict of nodal field arrays
            cell_data: dict of element field arrays
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio required for mesh export")

        mesh_cells = []
        for cell in cells:
            if isinstance(cell, dict):
                mesh_cells.append(meshio.CellBlock(cell["type"], cell["data"]))
            elif isinstance(cell, tuple):
                mesh_cells.append(meshio.CellBlock(cell[0], cell[1]))
            else:
                mesh_cells.append(cell)

        mesh = meshio.Mesh(
            points=points,
            cells=mesh_cells,
            point_data=point_data or {},
            cell_data=cell_data or {},
        )
        mesh.write(filepath)
        logger.info(f"Exported mesh: {filepath}")

    @staticmethod
    def from_skfem_mesh(mesh, filepath: str, **field_data) -> None:
        """Export scikit-fem mesh with optional field data."""
        points = mesh.p.T
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        if hasattr(mesh, "t"):
            cells = mesh.t.T
            if cells.shape[1] == 4:
                cell_type = "tetra"
            elif cells.shape[1] == 3:
                cell_type = "triangle"
            else:
                cell_type = "line"
        else:
            raise ValueError("Cannot determine element type")

        MeshExporter.export_file(
            filepath,
            points,
            [(cell_type, cells)],
            point_data=field_data if field_data else None,
        )

    @staticmethod
    def export_vtk_unstructured(
        filepath: str,
        points: np.ndarray,
        cells: np.ndarray,
        cell_type: str = "tetra",
        point_scalars: Optional[Dict[str, np.ndarray]] = None,
        cell_scalars: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Export VTK unstructured grid with multiple scalar fields.
        Useful for FEA result visualization in ParaView.
        """
        try:
            import vtk
            from vtkmodules.util.numpy_support import numpy_to_vtk

            # Create points
            vtk_points = vtk.vtkPoints()
            for p in points:
                vtk_points.InsertNextPoint(*p[:3])

            # Create cells
            vtk_cells = vtk.vtkCellArray()
            VTK_TYPE_MAP = {
                "tetra": vtk.VTK_TETRA,
                "triangle": vtk.VTK_TRIANGLE,
                "quad": vtk.VTK_QUAD,
                "hexahedron": vtk.VTK_HEXAHEDRON,
            }
            vtk_cell_type = VTK_TYPE_MAP.get(cell_type, vtk.VTK_TETRA)

            for cell in cells:
                vtk_cell = vtk.vtkIdList()
                for idx in cell:
                    vtk_cell.InsertNextId(int(idx))
                vtk_cells.InsertNextCell(vtk_cell)

            # Build grid
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(vtk_points)
            cell_types = np.full(len(cells), vtk_cell_type, dtype=np.uint8)
            grid.SetCells(vtk_cell_type, vtk_cells)

            # Add point data
            if point_scalars:
                for name, data in point_scalars.items():
                    arr = numpy_to_vtk(np.asarray(data, dtype=np.float64))
                    arr.SetName(name)
                    grid.GetPointData().AddArray(arr)

            # Add cell data
            if cell_scalars:
                for name, data in cell_scalars.items():
                    arr = numpy_to_vtk(np.asarray(data, dtype=np.float64))
                    arr.SetName(name)
                    grid.GetCellData().AddArray(arr)

            # Write
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(grid)
            writer.Write()
            logger.info(f"Exported VTK: {filepath}")

        except ImportError:
            # Fallback to meshio
            MeshExporter.export_file(
                filepath,
                points,
                [(cell_type, cells)],
                point_data=point_scalars,
                cell_data=cell_scalars,
            )
