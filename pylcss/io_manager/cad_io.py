# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Geometry file import and export with multi-format support.
Supports: STEP, IGES, STL, OBJ, BREP, 3MF
"""

import logging
import os
import struct
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------
CAD_IMPORT_FORMATS = {
    ".step": "STEP (ISO 10303)",
    ".stp": "STEP (ISO 10303)",
    ".iges": "IGES",
    ".igs": "IGES",
    ".stl": "STL (Stereolithography)",
    ".obj": "Wavefront OBJ",
    ".brep": "BREP (OpenCascade)",
    ".3mf": "3MF (3D Manufacturing)",
}

CAD_EXPORT_FORMATS = {
    ".step": "STEP (ISO 10303)",
    ".stp": "STEP (ISO 10303)",
    ".stl": "STL (Stereolithography)",
    ".obj": "Wavefront OBJ",
    ".brep": "BREP (OpenCascade)",
    ".svg": "SVG (2D projection)",
    ".dxf": "DXF (AutoCAD)",
}


class CADImporter:
    """Import geometry from various file formats."""

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        return dict(CAD_IMPORT_FORMATS)

    @staticmethod
    def get_filter_string() -> str:
        """File dialog filter string."""
        all_exts = " ".join(f"*{ext}" for ext in CAD_IMPORT_FORMATS)
        parts = [f"All Geometry Files ({all_exts})"]
        seen = set()
        for ext, name in CAD_IMPORT_FORMATS.items():
            if name not in seen:
                seen.add(name)
                exts = " ".join(
                    f"*{e}" for e, n in CAD_IMPORT_FORMATS.items() if n == name
                )
                parts.append(f"{name} ({exts})")
        return ";;".join(parts)

    @staticmethod
    def import_file(filepath: str, **kwargs) -> Any:
        """
        Import geometry file and return CadQuery shape or mesh data.
        
        Returns:
            CadQuery Workplane, shape, or dict with mesh data
        """
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()

        if ext in (".step", ".stp"):
            return CADImporter._import_step(filepath, **kwargs)
        elif ext in (".iges", ".igs"):
            return CADImporter._import_iges(filepath, **kwargs)
        elif ext == ".stl":
            return CADImporter._import_stl(filepath, **kwargs)
        elif ext == ".obj":
            return CADImporter._import_obj(filepath, **kwargs)
        elif ext == ".brep":
            return CADImporter._import_brep(filepath, **kwargs)
        elif ext == ".3mf":
            return CADImporter._import_3mf(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported import format: {ext}")

    @staticmethod
    def _import_step(filepath: str, **kwargs) -> Any:
        """Import STEP file via CadQuery / OCP."""
        try:
            import cadquery as cq
            result = cq.importers.importStep(filepath)
            logger.info(f"Imported STEP: {filepath}")
            return result
        except ImportError:
            logger.error("cadquery not available for STEP import")
            raise

    @staticmethod
    def _import_iges(filepath: str, **kwargs) -> Any:
        """Import IGES file via OCP."""
        try:
            from OCP.IGESControl import IGESControl_Reader
            from OCP.IFSelect import IFSelect_RetDone
            import cadquery as cq

            reader = IGESControl_Reader()
            status = reader.ReadFile(filepath)
            if status != IFSelect_RetDone:
                raise IOError(f"IGES read failed with status {status}")
            reader.TransferRoots()
            shape = reader.OneShape()
            result = cq.Workplane("XY").newObject([cq.Shape(shape)])
            logger.info(f"Imported IGES: {filepath}")
            return result
        except ImportError:
            logger.error("OCP not available for IGES import")
            raise

    @staticmethod
    def _import_stl(filepath: str, **kwargs) -> Dict:
        """Import STL file (binary or ASCII) as mesh data."""
        with open(filepath, "rb") as f:
            f.read(80)
            n_triangles = struct.unpack("<I", f.read(4))[0]

        # Check if binary
        file_size = os.path.getsize(filepath)
        expected_binary_size = 84 + n_triangles * 50

        if abs(file_size - expected_binary_size) < 10:
            return CADImporter._import_stl_binary(filepath, n_triangles)
        else:
            return CADImporter._import_stl_ascii(filepath)

    @staticmethod
    def _import_stl_binary(filepath: str, n_triangles: int) -> Dict:
        """Import binary STL."""
        vertices = []
        faces = []
        normals = []

        with open(filepath, "rb") as f:
            f.read(84)  # header + count
            for i in range(n_triangles):
                data = struct.unpack("<12fH", f.read(50))
                normal = data[0:3]
                v1, v2, v3 = data[3:6], data[6:9], data[9:12]
                idx = len(vertices)
                vertices.extend([v1, v2, v3])
                faces.append((idx, idx + 1, idx + 2))
                normals.append(normal)

        logger.info(f"Imported binary STL: {n_triangles} triangles from {filepath}")
        return {
            "vertices": np.array(vertices, dtype=np.float64),
            "faces": np.array(faces, dtype=np.int32),
            "normals": np.array(normals, dtype=np.float64),
            "format": "stl",
        }

    @staticmethod
    def _import_stl_ascii(filepath: str) -> Dict:
        """Import ASCII STL."""
        vertices = []
        faces = []
        normals = []

        with open(filepath, "r") as f:
            current_normal = None
            tri_verts = []
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "facet" and parts[1] == "normal":
                    current_normal = tuple(float(x) for x in parts[2:5])
                elif parts[0] == "vertex":
                    tri_verts.append(tuple(float(x) for x in parts[1:4]))
                elif parts[0] == "endfacet":
                    if len(tri_verts) == 3:
                        idx = len(vertices)
                        vertices.extend(tri_verts)
                        faces.append((idx, idx + 1, idx + 2))
                        if current_normal:
                            normals.append(current_normal)
                    tri_verts = []

        logger.info(f"Imported ASCII STL: {len(faces)} triangles from {filepath}")
        return {
            "vertices": np.array(vertices, dtype=np.float64),
            "faces": np.array(faces, dtype=np.int32),
            "normals": np.array(normals, dtype=np.float64) if normals else None,
            "format": "stl",
        }

    @staticmethod
    def _import_obj(filepath: str, **kwargs) -> Dict:
        """Import Wavefront OBJ file."""
        vertices = []
        faces = []
        normals = []
        texcoords = []

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].startswith("#"):
                    continue
                if parts[0] == "v":
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == "vn":
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == "vt":
                    texcoords.append([float(x) for x in parts[1:3]])
                elif parts[0] == "f":
                    face_verts = []
                    for vert in parts[1:]:
                        idx = int(vert.split("/")[0]) - 1  # OBJ is 1-indexed
                        face_verts.append(idx)
                    # Triangulate polygons
                    for i in range(1, len(face_verts) - 1):
                        faces.append((face_verts[0], face_verts[i], face_verts[i + 1]))

        logger.info(f"Imported OBJ: {len(vertices)} vertices, {len(faces)} faces")
        return {
            "vertices": np.array(vertices, dtype=np.float64),
            "faces": np.array(faces, dtype=np.int32),
            "normals": np.array(normals, dtype=np.float64) if normals else None,
            "format": "obj",
        }

    @staticmethod
    def _import_brep(filepath: str, **kwargs) -> Any:
        """Import OpenCascade BREP."""
        try:
            import cadquery as cq
            from OCP.BRep import BRep_Builder
            from OCP.BRepTools import BRepTools
            from OCP.TopoDS import TopoDS_Shape

            builder = BRep_Builder()
            shape = TopoDS_Shape()
            BRepTools.Read_s(shape, filepath, builder)
            result = cq.Workplane("XY").newObject([cq.Shape(shape)])
            logger.info(f"Imported BREP: {filepath}")
            return result
        except ImportError:
            raise ImportError("OCP/cadquery required for BREP import")

    @staticmethod
    def _import_3mf(filepath: str, **kwargs) -> Dict:
        """Import 3MF file (3D Manufacturing Format) via meshio."""
        try:
            import meshio
            mesh = meshio.read(filepath)
            logger.info(f"Imported 3MF: {filepath}")
            return {
                "vertices": mesh.points,
                "faces": mesh.cells[0].data if mesh.cells else np.array([]),
                "format": "3mf",
            }
        except ImportError:
            raise ImportError("meshio required for 3MF import")
