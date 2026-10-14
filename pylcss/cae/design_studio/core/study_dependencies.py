# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Input-file fingerprints for saved studies, independent of Qt loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .study_schema import STUDY_NODE_SCHEMAS, study_node_identifier


_DEPENDENCY_SUFFIXES = {
    ".step",
    ".stp",
    ".iges",
    ".igs",
    ".stl",
    ".obj",
    ".3mf",
    ".fcstd",
    ".brep",
    ".npz",
    ".k",
    ".rad",
}


def study_dependency_fingerprint(cad_path: str) -> tuple:
    """Track declared input files and legacy geometry/deck references.

    Missing files are tracked too, so creating an input invalidates the cache.
    Export destinations are excluded: writing an output is not an input change.
    """
    project = Path(cad_path).resolve()
    try:
        session = json.loads(project.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    try:
        from pylcss.config import BASE_DIR

        repo_root = Path(BASE_DIR).resolve().parent
    except ImportError:
        repo_root = project.parent

    records = []
    for node_data in (session.get("nodes", {}) or {}).values():
        node_type = str(node_data.get("type_", ""))
        if node_type.startswith(("com.cad.export_step", "com.cad.export_stl")):
            continue
        identifier = study_node_identifier(node_type)
        schema = STUDY_NODE_SCHEMAS.get(identifier)
        declared = schema.input_files if schema is not None else ()
        custom = node_data.get("custom", {}) or {}
        for prop, value in custom.items():
            if (
                prop == "cad_export_filename"
                or not isinstance(value, str)
                or not value.strip()
            ):
                continue
            raw = os.path.expandvars(os.path.expanduser(value.strip()))
            if (
                prop not in declared
                and Path(raw).suffix.lower() not in _DEPENDENCY_SUFFIXES
            ):
                continue

            if identifier == "com.cad.freecad_part" and prop == "fcstd_filename":
                from pylcss.cae.design_studio.freecad_bridge.paths import (
                    freecad_data_dir,
                )

                stem = Path(raw).stem
                safe_stem = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in stem
                )
                safe_stem = safe_stem.strip("_") or "FreeCAD_Part"
                candidates = [freecad_data_dir(create=False) / f"{safe_stem}.FCStd"]
            else:
                raw_path = Path(raw)
                candidates = (
                    [raw_path]
                    if raw_path.is_absolute()
                    else [raw_path, project.parent / raw_path, repo_root / raw_path]
                )
            resolved = next(
                (path.resolve() for path in candidates if path.is_file()), None
            )
            if resolved is None:
                records.append((str(prop), raw, "missing"))
                continue
            try:
                stat = resolved.stat()
                records.append((str(resolved), stat.st_mtime_ns, stat.st_size))
            except OSError:
                records.append((str(resolved), "unreadable"))
    return tuple(sorted(set(records), key=repr))
