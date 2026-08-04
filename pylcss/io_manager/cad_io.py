# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Validated geometry import for CAD exchange and triangle-mesh formats."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from pylcss.io_manager._mesh import load_3mf, load_obj, load_stl

__all__ = ["CADImporter", "CAD_IMPORT_FORMATS"]

logger = logging.getLogger(__name__)

CAD_IMPORT_FORMATS: Final = MappingProxyType(
    {
        ".step": "STEP (ISO 10303)",
        ".stp": "STEP (ISO 10303)",
        ".iges": "IGES",
        ".igs": "IGES",
        ".stl": "STL (Stereolithography)",
        ".obj": "Wavefront OBJ",
        ".brep": "BREP (OpenCascade)",
        ".3mf": "3MF (3D Manufacturing)",
    }
)


class CADImporter:
    """Import exact CAD geometry or validated triangulated mesh data."""

    @staticmethod
    def get_supported_formats() -> dict[str, str]:
        """Return a copy of the extension-to-description registry."""
        return dict(CAD_IMPORT_FORMATS)

    @staticmethod
    def get_filter_string() -> str:
        """Return a Qt-compatible file-dialog filter string."""
        all_extensions = " ".join(f"*{ext}" for ext in CAD_IMPORT_FORMATS)
        parts = [f"All Geometry Files ({all_extensions})"]
        seen: set[str] = set()
        for description in CAD_IMPORT_FORMATS.values():
            if description in seen:
                continue
            seen.add(description)
            extensions = " ".join(
                f"*{ext}"
                for ext, name in CAD_IMPORT_FORMATS.items()
                if name == description
            )
            parts.append(f"{description} ({extensions})")
        return ";;".join(parts)

    @staticmethod
    def import_file(
        filepath: str | os.PathLike[str],
        **options: object,
    ) -> Any:
        """Import ``filepath`` as a CadQuery workplane or mesh-data dictionary."""
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"Unsupported CAD import option(s): {names}.")

        path = Path(filepath).expanduser()
        extension = path.suffix.lower()
        handlers: dict[str, Callable[[Path], Any]] = {
            ".step": CADImporter._import_step,
            ".stp": CADImporter._import_step,
            ".iges": CADImporter._import_iges,
            ".igs": CADImporter._import_iges,
            ".stl": load_stl,
            ".obj": load_obj,
            ".brep": CADImporter._import_brep,
            ".3mf": load_3mf,
        }
        try:
            handler = handlers[extension]
        except KeyError as exc:
            label = extension or "<none>"
            raise ValueError(f"Unsupported CAD import format: {label}.") from exc
        if not path.exists():
            raise FileNotFoundError(f"CAD input file does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"CAD input path is not a file: {path}")

        result = handler(path)
        logger.info(
            "Imported %s file: %s",
            extension.removeprefix(".").upper(),
            path,
        )
        return result

    @staticmethod
    def _import_step(filepath: Path) -> Any:
        """Import a STEP file through CadQuery."""
        try:
            import cadquery as cq
        except ImportError as exc:
            raise ImportError("cadquery is required for STEP import.") from exc
        return cq.importers.importStep(str(filepath))

    @staticmethod
    def _import_iges(filepath: Path) -> Any:
        """Import an IGES file and rebuild closed shells as CAD solids.

        IGES commonly stores a solid as a collection of trimmed surfaces.  A
        plain ``OneShape()`` therefore gives CadQuery loose faces: querying
        its ``Volume`` can even return the surface area, which looks numeric
        but is not a usable solid.  Sew the transferred roots first and turn
        every closed shell into a solid while preserving genuinely open
        surface models as shells.
        """
        try:
            import cadquery as cq
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
            from OCP.IFSelect import IFSelect_RetDone
            from OCP.IGESControl import IGESControl_Reader
            from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import TopoDS
        except ImportError as exc:
            raise ImportError("cadquery and OCP are required for IGES import.") from exc

        reader = IGESControl_Reader()
        status = reader.ReadFile(str(filepath))
        if status != IFSelect_RetDone:
            raise OSError(f"IGES reader rejected {filepath.name} with status {status}.")
        if reader.TransferRoots() <= 0:
            raise ValueError(f"{filepath.name} does not contain transferable IGES roots.")
        shape = reader.OneShape()
        if shape.IsNull():
            raise ValueError(f"{filepath.name} contains an empty IGES shape.")

        sewing = BRepBuilderAPI_Sewing(1.0e-6, True, True, True, False)
        sewing.Add(shape)
        sewing.Perform()
        sewed = sewing.SewedShape()
        if sewed.IsNull():
            sewed = shape

        transferred_solids: list[Any] = []
        solids = TopExp_Explorer(sewed, TopAbs_SOLID)
        while solids.More():
            transferred_solids.append(cq.Shape(TopoDS.Solid_s(solids.Current())))
            solids.Next()
        if transferred_solids:
            return cq.Workplane("XY").newObject(transferred_solids)

        rebuilt: list[Any] = []
        shells = TopExp_Explorer(sewed, TopAbs_SHELL)
        while shells.More():
            shell = TopoDS.Shell_s(shells.Current())
            if shell.Closed():
                solid_builder = BRepBuilderAPI_MakeSolid(shell)
                if solid_builder.IsDone() and not solid_builder.Solid().IsNull():
                    rebuilt.append(cq.Shape(solid_builder.Solid()))
                else:
                    rebuilt.append(cq.Shape(shell))
            else:
                rebuilt.append(cq.Shape(shell))
            shells.Next()

        if rebuilt:
            return cq.Workplane("XY").newObject(rebuilt)
        return cq.Workplane("XY").newObject([cq.Shape(sewed)])

    @staticmethod
    def _import_brep(filepath: Path) -> Any:
        """Import an OpenCascade BREP file."""
        try:
            import cadquery as cq
            from OCP.BRep import BRep_Builder
            from OCP.BRepTools import BRepTools
            from OCP.TopoDS import TopoDS_Shape
        except ImportError as exc:
            raise ImportError("cadquery and OCP are required for BREP import.") from exc

        builder = BRep_Builder()
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(filepath), builder) or shape.IsNull():
            raise ValueError(f"{filepath.name} is not a valid OpenCascade BREP file.")
        return cq.Workplane("XY").newObject([cq.Shape(shape)])
