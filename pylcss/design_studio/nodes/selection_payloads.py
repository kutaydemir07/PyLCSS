# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.


"""Stable payload construction shared by CAD and mesh selection nodes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _entity_summary(entity: Any) -> dict[str, Any]:
    if isinstance(entity, dict):
        return {
            "center": entity.get("center"),
            "bbox": entity.get("bbox"),
            "node_count": entity.get("node_count"),
        }
    try:
        c = entity.Center()
        bb = entity.BoundingBox()
        summary = {
            "center": [float(c.x), float(c.y), float(c.z)],
            "bbox": {
                "xmin": float(bb.xmin),
                "xmax": float(bb.xmax),
                "ymin": float(bb.ymin),
                "ymax": float(bb.ymax),
                "zmin": float(bb.zmin),
                "zmax": float(bb.zmax),
            },
        }
        if hasattr(entity, "Area"):
            summary["area"] = float(entity.Area())
        if hasattr(entity, "Length"):
            summary["length"] = float(entity.Length())
        return summary
    except Exception:
        return {}


def _selection_payload(
    workplane: Any,
    entities: Iterable[Any] | None,
    selector_type: str,
    entity_type: str = "Face",
) -> dict[str, Any]:
    entities = list(entities or [])
    entity_type = str(entity_type or "Face").title()
    return {
        "workplane": workplane,
        "entity": entities[0] if entities else None,
        "entities": entities,
        "entity_type": entity_type,
        # Compatibility aliases: every existing support/load backend consumes
        # ``faces`` as a list of geometric regions.  Edges and vertices are
        # valid distance-match geometries too, so retaining these aliases lets
        # old downstream nodes accept the richer selection without migration.
        "face": entities[0] if entities else None,
        "faces": entities,
        "selector_type": selector_type,
        "entity_count": len(entities),
        "face_count": len(entities),
        "entity_summaries": [_entity_summary(entity) for entity in entities[:12]],
        "face_summaries": [_entity_summary(entity) for entity in entities[:12]],
    }
