# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Read FreeCAD's exported BREP through CadQuery's OCCT bindings.

The returned container also carries spreadsheet parameters, object metadata,
and FEM definitions from the companion JSON. Missing or stale geometry is
represented separately from an unauthored document.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Lazy-import cadquery / OCP -- the bridge is loaded by the node registry
# at app startup and we don't want a slow OCP import to delay the splash
# screen if FreeCAD isn't being used this session.
def _occt_modules():
    """Return the BREP reader, shape, builder, and CadQuery wrapper classes."""
    from cadquery.occ_impl.shapes import Shape as ShapeCls  # type: ignore

    try:
        from OCP.BRepTools import BRepTools  # type: ignore
        from OCP.TopoDS import TopoDS_Shape  # type: ignore
        from OCP.BRep import BRep_Builder  # type: ignore
    except ImportError as exc:  # pragma: no cover - OCP shape mismatch
        raise ImportError(
            "OCP (CadQuery's OCCT bindings) not importable; cannot read BREP. "
            "Is cadquery installed? Original error: " + str(exc)
        ) from exc
    return BRepTools, TopoDS_Shape, BRep_Builder, ShapeCls


@dataclass
class FreeCadImportedShape:
    """Container for one BREP + sidecar pair PyLCSS got from FreeCAD."""

    shape: Any  # cadquery.Shape
    sidecar: Dict[str, Any] = field(default_factory=dict)
    brep_path: Optional[Path] = None
    fcstd_path: Optional[Path] = None

    @property
    def parameters(self) -> Dict[str, float]:
        return dict(self.sidecar.get("parameters", {}) or {})

    @property
    def per_shape_metadata(self) -> list:
        return list(self.sidecar.get("shapes", []) or [])

    @property
    def fem_summary(self) -> list:
        return list(self.sidecar.get("fem", []) or [])


def read_brep_from_fcstd(fcstd_path: Path | str) -> Optional[FreeCadImportedShape]:
    """Locate the BREP + sidecar siblings of ``fcstd_path`` and load them.

    Returns None if neither BREP nor metadata is available. When only metadata
    is available, or geometry is stale/unreadable, returns a container with
    shape=None so callers can distinguish pending documents from bad exports.
    """
    fcstd_path = Path(fcstd_path).resolve()
    brep_path = fcstd_path.with_suffix(".brep")
    sidecar_path = fcstd_path.with_suffix(".fcmeta.json")

    sidecar: Dict[str, Any] = {}
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Sidecar JSON %s unreadable: %s", sidecar_path, exc)

    if not brep_path.is_file():
        logger.debug(
            "No BREP sibling for %s; user hasn't saved in FreeCAD yet (or the "
            "auto-export macro didn't install).",
            fcstd_path.name,
        )
        if sidecar:
            # Still return the sidecar so e.g. parameter-only consumers can run.
            return FreeCadImportedShape(
                shape=None,
                sidecar=sidecar,
                brep_path=None,
                fcstd_path=fcstd_path,
            )
        return None

    # Never present geometry older than the authoritative FreeCAD document.
    # This happens when the save observer failed or FreeCAD was closed before
    # the BREP export completed.
    try:
        if (
            fcstd_path.is_file()
            and brep_path.stat().st_mtime_ns < fcstd_path.stat().st_mtime_ns
        ):
            logger.warning(
                "BREP %s is older than %s; refusing stale geometry.",
                brep_path.name,
                fcstd_path.name,
            )
            return FreeCadImportedShape(
                shape=None,
                sidecar=sidecar,
                brep_path=brep_path,
                fcstd_path=fcstd_path,
            )
    except OSError:
        pass

    try:
        BRepTools, TopoDS_Shape, BRep_Builder, ShapeCls = _occt_modules()
    except ImportError:
        logger.exception("Cannot read BREP because OCP is not importable.")
        return FreeCadImportedShape(
            shape=None,
            sidecar=sidecar,
            brep_path=brep_path,
            fcstd_path=fcstd_path,
        )

    occ_shape = TopoDS_Shape()
    builder = BRep_Builder()
    ok = BRepTools.Read_s(occ_shape, str(brep_path), builder)
    if not ok or occ_shape.IsNull():
        logger.warning("BRepTools.Read failed for %s", brep_path)
        # Preserve the distinction between an unauthored document (no BREP)
        # and a corrupt/unreadable export (BREP exists).  FreeCadPartNode can
        # keep the former pending while reporting the latter as a real error.
        return FreeCadImportedShape(
            shape=None,
            sidecar=sidecar,
            brep_path=brep_path,
            fcstd_path=fcstd_path,
        )

    cq_shape = ShapeCls.cast(occ_shape)
    return FreeCadImportedShape(
        shape=cq_shape,
        sidecar=sidecar,
        brep_path=brep_path,
        fcstd_path=fcstd_path,
    )
