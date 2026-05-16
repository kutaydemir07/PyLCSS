# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Translate FreeCAD FEM workbench constraint / load definitions into PyLCSS
``com.cad.sim.*`` node specs.

What the macro writes
---------------------
``pylcss_autoexport.FCMacro`` walks every ``Fem::`` object in the document
and writes its ``TypeId`` + flattened scalar properties into the
``fem`` array of ``<doc>.fcmeta.json``::

    [
      {"name": "ConstraintFixed", "label": "Fix Base",
       "type_id": "Fem::ConstraintFixed"},
      {"name": "ConstraintForce", "label": "Tip Load",
       "type_id": "Fem::ConstraintForce",
       "Force": 250.0, "DirectionVector_x": 0.0,
       "DirectionVector_y": -1.0, "DirectionVector_z": 0.0},
      ...
    ]

What we produce
---------------
PyLCSS node-graph specs in the same shape ``_build_node_graph`` consumes::

    {"id": "Fix_Base", "type": "com.cad.sim.constraint",
     "properties": {"constraint_type": "Fixed"}}

These specs do **not** carry the geometric "which face is this on?" link --
the face selection lives in the BREP + the macro's per-shape face index
metadata, and gets wired through a ``com.cad.select_face`` node in a
follow-up step (out of scope for the POC).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FreeCAD FEM constraint TypeId -> PyLCSS constraint_type property
# ---------------------------------------------------------------------------
# Only the ones with a 1:1 mapping live here; the rest fall through to the
# generic "Fixed" default + a warning so the user can see what was missed.
_CONSTRAINT_TYPE_MAP: Dict[str, str] = {
    "Fem::ConstraintFixed":          "Fixed",
    "Fem::ConstraintPinned":         "Pinned",
    "Fem::ConstraintDisplacement":   "Displacement",
    "Fem::ConstraintPlaneRotation":  "Symmetry Z",  # rough mirror -- user can tweak
}

# Load constraint TypeIds we can route to PyLCSS LoadNode / PressureLoadNode.
_LOAD_FORCE_TYPES = {"Fem::ConstraintForce"}
_LOAD_PRESSURE_TYPES = {"Fem::ConstraintPressure"}
_LOAD_GRAVITY_TYPES = {"Fem::ConstraintGear", "Fem::ConstraintSelfWeight"}


def _safe_id(label: str, fallback: str = "fem_item") -> str:
    """Sanitise a FreeCAD label into a PyLCSS-safe node id."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(label or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _f(entry: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Pull a scalar property from the sidecar entry, tolerant of missing keys."""
    val = entry.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-type translators
# ---------------------------------------------------------------------------
def _translate_constraint(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    type_id = str(entry.get("type_id", ""))
    if not type_id.startswith("Fem::Constraint") or type_id in _LOAD_FORCE_TYPES \
            or type_id in _LOAD_PRESSURE_TYPES or type_id in _LOAD_GRAVITY_TYPES:
        return None

    constraint_kind = _CONSTRAINT_TYPE_MAP.get(type_id)
    if constraint_kind is None:
        # No safe default: defaulting to "Fixed" would silently produce
        # a wrong analysis. Skip and surface the missed type in the log
        # so the user can either edit it in FreeCAD or extend this map.
        logger.info(
            "FEM translator: no PyLCSS equivalent for %s (label=%r); skipping. "
            "Extend _CONSTRAINT_TYPE_MAP if this comes up often.",
            type_id, entry.get("label"),
        )
        return None

    props: Dict[str, Any] = {
        "constraint_type": constraint_kind,
    }
    if constraint_kind == "Displacement":
        # FreeCAD's ConstraintDisplacement exposes individual axis values
        # under `xDisplacement`, `yDisplacement`, `zDisplacement` (mm).
        props["displacement_x"] = _f(entry, "xDisplacement")
        props["displacement_y"] = _f(entry, "yDisplacement")
        props["displacement_z"] = _f(entry, "zDisplacement")

    return {
        "id": _safe_id(entry.get("label") or entry.get("name") or "constraint"),
        "type": "com.cad.sim.constraint",
        "properties": props,
        # Provenance: where this spec came from so the inspector can show it.
        "source": {
            "kind": "freecad",
            "type_id": type_id,
            "name": entry.get("name"),
            "label": entry.get("label"),
        },
    }


def _translate_load(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    type_id = str(entry.get("type_id", ""))

    if type_id in _LOAD_FORCE_TYPES:
        magnitude = _f(entry, "Force")
        dx = _f(entry, "DirectionVector_x") or _f(entry, "Direction_x")
        dy = _f(entry, "DirectionVector_y") or _f(entry, "Direction_y")
        dz = _f(entry, "DirectionVector_z") or _f(entry, "Direction_z")
        # FreeCAD stores force as magnitude + unit direction. PyLCSS LoadNode
        # takes per-axis components, so multiply out.
        return {
            "id": _safe_id(entry.get("label") or entry.get("name") or "load"),
            "type": "com.cad.sim.load",
            "properties": {
                "load_type": "Force",
                "force_x": magnitude * dx,
                "force_y": magnitude * dy,
                "force_z": magnitude * dz,
            },
            "source": {"kind": "freecad", "type_id": type_id,
                       "name": entry.get("name"), "label": entry.get("label")},
        }

    if type_id in _LOAD_PRESSURE_TYPES:
        return {
            "id": _safe_id(entry.get("label") or entry.get("name") or "pressure"),
            "type": "com.cad.sim.pressure_load",
            "properties": {
                "pressure": _f(entry, "Pressure"),
            },
            "source": {"kind": "freecad", "type_id": type_id,
                       "name": entry.get("name"), "label": entry.get("label")},
        }

    if type_id in _LOAD_GRAVITY_TYPES:
        # PyLCSS LoadNode has a Gravity mode (accel + direction).  FreeCAD
        # SelfWeight uses the doc-global g vector under Newton/Meter.
        return {
            "id": _safe_id(entry.get("label") or entry.get("name") or "gravity"),
            "type": "com.cad.sim.load",
            "properties": {
                "load_type": "Gravity",
                "gravity_accel": _f(entry, "Gravity") or 9810.0,
                "gravity_direction": "-Z",
            },
            "source": {"kind": "freecad", "type_id": type_id,
                       "name": entry.get("name"), "label": entry.get("label")},
        }

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def translate_fem_summary(fem_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert the sidecar ``fem`` list to PyLCSS node specs.

    Returns a list of ``{"id", "type", "properties", "source"}`` dicts in the
    same shape ``_build_node_graph`` accepts, ready for the caller to either
    upsert into an existing graph or hand to the AI assistant's
    ``create_system_model`` / ``modify_system_node`` tools.

    Anything we can't recognise gets skipped with an INFO log -- the user
    can inspect those manually in FreeCAD and we don't silently produce
    garbage constraints.
    """
    specs: List[Dict[str, Any]] = []
    for entry in fem_entries or []:
        if not isinstance(entry, dict):
            continue
        type_id = str(entry.get("type_id", ""))
        if not type_id.startswith("Fem::"):
            continue
        spec = _translate_load(entry) or _translate_constraint(entry)
        if spec is None:
            logger.info(
                "FEM translator: skipping unrecognised %s (label=%r)",
                type_id, entry.get("label"),
            )
            continue
        specs.append(spec)
    return specs
