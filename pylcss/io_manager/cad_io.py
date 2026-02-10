# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
CAD file import and export with multi-format support.
Supports: STEP, IGES, STL, OBJ, BREP, 3MF
"""

import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Import CAD geometry from various file formats."""

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        return dict(CAD_IMPORT_FORMATS)

    @staticmethod
    def get_filter_string() -> str:
        """File dialog filter string."""
        all_exts = " ".join(f"*{ext}" for ext in CAD_IMPORT_FORMATS)
        parts = [f"All CAD Files ({all_exts})"]
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
        Import CAD file and return CadQuery shape or mesh data.
        
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
            header = f.read(80)
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


class CADExporter:
    """Export CAD geometry to various file formats."""

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        return dict(CAD_EXPORT_FORMATS)

    @staticmethod
    def get_filter_string() -> str:
        parts = []
        seen = set()
        for ext, name in CAD_EXPORT_FORMATS.items():
            if name not in seen:
                seen.add(name)
                exts = " ".join(
                    f"*{e}" for e, n in CAD_EXPORT_FORMATS.items() if n == name
                )
                parts.append(f"{name} ({exts})")
        return ";;".join(parts)

    @staticmethod
    def export_file(shape: Any, filepath: str, **kwargs) -> None:
        """Export CadQuery shape/mesh to file."""
        ext = Path(filepath).suffix.lower()

        if ext in (".step", ".stp"):
            CADExporter._export_step(shape, filepath, **kwargs)
        elif ext == ".stl":
            CADExporter._export_stl(shape, filepath, **kwargs)
        elif ext == ".obj":
            CADExporter._export_obj(shape, filepath, **kwargs)
        elif ext == ".brep":
            CADExporter._export_brep(shape, filepath, **kwargs)
        elif ext == ".svg":
            CADExporter._export_svg(shape, filepath, **kwargs)
        elif ext == ".dxf":
            CADExporter._export_dxf(shape, filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {ext}")

    @staticmethod
    def _export_step(shape, filepath: str, **kwargs) -> None:
        """Export to STEP."""
        import cadquery as cq
        if isinstance(shape, cq.Workplane):
            cq.exporters.export(shape, filepath, exportType="STEP")
        elif hasattr(shape, "val"):
            cq.exporters.export(shape, filepath, exportType="STEP")
        else:
            # Try OCP direct
            from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
            writer.Write(filepath)
        logger.info(f"Exported STEP: {filepath}")

    @staticmethod
    def _export_stl(
        shape,
        filepath: str,
        tolerance: float = 0.01,
        angular_tolerance: float = 0.1,
        binary: bool = True,
        **kwargs,
    ) -> None:
        """Export to STL with tessellation control."""
        import cadquery as cq

        if isinstance(shape, dict) and "vertices" in shape:
            # Mesh data
            verts = np.asarray(shape["vertices"])
            faces = np.asarray(shape["faces"])
            if binary:
                CADExporter._write_stl_binary(verts, faces, filepath)
            else:
                CADExporter._write_stl_ascii(verts, faces, filepath)
        elif isinstance(shape, (cq.Workplane, cq.Shape)):
            cq.exporters.export(
                shape,
                filepath,
                exportType="STL",
                tolerance=tolerance,
                angularTolerance=angular_tolerance,
            )
        else:
            raise TypeError(f"Cannot export {type(shape)} to STL")
        logger.info(f"Exported STL: {filepath}")

    @staticmethod
    def _write_stl_binary(
        vertices: np.ndarray, faces: np.ndarray, filepath: str
    ) -> None:
        """Write binary STL from mesh data."""
        with open(filepath, "wb") as f:
            f.write(b"\0" * 80)  # header
            f.write(struct.pack("<I", len(faces)))
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal /= norm_len
                f.write(struct.pack("<3f", *normal))
                f.write(struct.pack("<3f", *v0))
                f.write(struct.pack("<3f", *v1))
                f.write(struct.pack("<3f", *v2))
                f.write(struct.pack("<H", 0))

    @staticmethod
    def _write_stl_ascii(
        vertices: np.ndarray, faces: np.ndarray, filepath: str
    ) -> None:
        """Write ASCII STL from mesh data."""
        with open(filepath, "w") as f:
            f.write("solid pylcss_export\n")
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal /= norm_len
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in (v0, v1, v2):
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid pylcss_export\n")

    @staticmethod
    def _export_obj(shape, filepath: str, **kwargs) -> None:
        """Export to Wavefront OBJ."""
        import cadquery as cq

        if isinstance(shape, dict) and "vertices" in shape:
            verts, faces = shape["vertices"], shape["faces"]
        elif isinstance(shape, (cq.Workplane, cq.Shape)):
            # Tessellate CadQuery shape
            if isinstance(shape, cq.Workplane):
                shape_val = shape.val()
            else:
                shape_val = shape
            tess = shape_val.tessellate(0.01)
            verts = np.array([(v.x, v.y, v.z) for v in tess[0]])
            faces = np.array(tess[1])
        else:
            raise TypeError(f"Cannot export {type(shape)} to OBJ")

        with open(filepath, "w") as f:
            f.write("# Exported by pylcss\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed
                indices = " ".join(str(i + 1) for i in face)
                f.write(f"f {indices}\n")
        logger.info(f"Exported OBJ: {filepath}")

    @staticmethod
    def _export_brep(shape, filepath: str, **kwargs) -> None:
        """Export to OpenCascade BREP."""
        import cadquery as cq
        from OCP.BRepTools import BRepTools

        if isinstance(shape, cq.Workplane):
            occ_shape = shape.val().wrapped
        elif hasattr(shape, "wrapped"):
            occ_shape = shape.wrapped
        else:
            occ_shape = shape

        BRepTools.Write_s(occ_shape, filepath)
        logger.info(f"Exported BREP: {filepath}")

    @staticmethod
    def _export_svg(shape, filepath: str, **kwargs) -> None:
        """Export 2D projection as SVG."""
        import cadquery as cq
        if isinstance(shape, cq.Workplane):
            cq.exporters.export(shape, filepath, exportType="SVG")
        logger.info(f"Exported SVG: {filepath}")

    @staticmethod
    def _export_dxf(shape, filepath: str, **kwargs) -> None:
        """Export 2D projection as DXF."""
        import cadquery as cq
        if isinstance(shape, cq.Workplane):
            cq.exporters.export(shape, filepath, exportType="DXF")
        logger.info(f"Exported DXF: {filepath}")
