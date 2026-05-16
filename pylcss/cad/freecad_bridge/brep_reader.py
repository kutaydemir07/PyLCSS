# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Read the BREP + sidecar JSON the FreeCAD startup macro writes.

Why BREP and not STEP
---------------------
STEP loses face identity round-trip in both FreeCAD's exporter and
CadQuery's importer (research notes in pylcss/cad/freecad_bridge/__init__.py).
BREP keeps OCCT topology + face indices intact, and CadQuery already ships
the OCCT bindings (``cadquery.occ_impl``), so PyLCSS can read what FreeCAD
just wrote without a new dependency.

The reader returns a :class:`FreeCadImportedShape` that bundles:
  - ``shape``: a ``cadquery.Shape`` PyLCSS's existing viewer + the rest of
    the node graph can consume exactly like a CadQuery-produced shape;
  - ``sidecar``: the parsed ``.fcmeta.json`` dict (parameters, FEM
    summary, per-shape face counts), or ``{}`` if it's missing.

Callers who only want the shape can ignore the sidecar; the FEM /
selection translation layers live in their own modules and consume it.
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
    """Return ``(BRepTools, TopoDS, Shape_factory)`` from CadQuery's OCP."""
    from cadquery import Shape as _Shape  # type: ignore
    from cadquery.occ_impl.shapes import Shape as ShapeCls  # type: ignore
    # OCP wraps BRepTools as BRepTools_BRepTools_Read in some builds; use
    # cadquery's helper which papers over the variants.
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

    Returns ``None`` if the BREP was never written (e.g. user opened
    FreeCAD, edited but never saved; or the startup macro failed to
    install). Always parses the sidecar JSON if present, even when the
    BREP itself is missing -- callers may still want the parameter set.
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
        # Was INFO -- demoted to DEBUG because the CAD engine re-walks the
        # graph on every property tweak / Run click, and this is the
        # expected "user hasn't saved yet" state, not an error.
        logger.debug(
            "No BREP sibling for %s; user hasn't saved in FreeCAD yet (or the "
            "auto-export macro didn't install).",
            fcstd_path.name,
        )
        if sidecar:
            # Still return the sidecar so e.g. parameter-only consumers can run.
            return FreeCadImportedShape(
                shape=None, sidecar=sidecar,
                brep_path=None, fcstd_path=fcstd_path,
            )
        return None

    try:
        BRepTools, TopoDS_Shape, BRep_Builder, ShapeCls = _occt_modules()
    except ImportError:
        logger.exception("Cannot read BREP because OCP is not importable.")
        return None

    occ_shape = TopoDS_Shape()
    builder = BRep_Builder()
    ok = BRepTools.Read_s(occ_shape, str(brep_path), builder)
    if not ok or occ_shape.IsNull():
        logger.warning("BRepTools.Read failed for %s", brep_path)
        return None

    cq_shape = ShapeCls.cast(occ_shape)
    return FreeCadImportedShape(
        shape=cq_shape, sidecar=sidecar,
        brep_path=brep_path, fcstd_path=fcstd_path,
    )
