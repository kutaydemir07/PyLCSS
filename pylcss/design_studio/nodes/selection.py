# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.


"""Geometry and mesh entity-selection nodes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import cadquery as cq
import numpy as np

from pylcss.design_studio.core.base_node import (
    CadQueryNode,
    resolve_any_input,
    resolve_shape_input,
)

from . import mesh_selection as _mesh_selection_module
from . import selection_payloads as _selection_payloads_module
from .mesh_selection import (
    _MESH_COMPONENT_INDEX_BASE,
    _MESH_FACE_DIRECTIONS,
    _mesh_axis_tolerance,
    _mesh_boundary_face_data,
    _mesh_component_selection,
    _mesh_direction_selection,
    _mesh_points,
    _mesh_selection_payload,
    _surface_component_metrics,
)
from .selection_payloads import _selection_payload

logger = logging.getLogger(__name__)

_COMPATIBILITY_MODULES = (_mesh_selection_module, _selection_payloads_module)


def __getattr__(name: str) -> object:
    """Resolve private helpers moved during the selection-module split."""
    for module in _COMPATIBILITY_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include moved compatibility names in interactive discovery."""
    names = set(globals())
    for module in _COMPATIBILITY_MODULES:
        names.update(dir(module))
    return sorted(names)


_SELECTOR_TYPE_ALIASES = {
    "direction": "Direction",
    "nearesttopoint": "NearestToPoint",
    "nearest point": "NearestToPoint",
    "nearest_point": "NearestToPoint",
    "index": "Index",
    "face index": "Index",
    "face_index": "Index",
    "largest area": "Largest Area",
    "largest_area": "Largest Area",
    "tag": "Tag",
    "box": "Box",
    "bounding box": "Box",
    "bounding_box": "Box",
    "coordinate range": "Coordinate Range",
    "range expression": "Coordinate Range",
    "range_expression": "Coordinate Range",
}


_DIRECTION_ALIASES = {
    "+X": ">X",
    "-X": "<X",
    "+Y": ">Y",
    "-Y": "<Y",
    "+Z": ">Z",
    "-Z": "<Z",
    "X+": ">X",
    "X-": "<X",
    "Y+": ">Y",
    "Y-": "<Y",
    "Z+": ">Z",
    "Z-": "<Z",
}


def _canonical_selector_type(value: object) -> str:
    text = str(value or "Direction").strip()
    return _SELECTOR_TYPE_ALIASES.get(text.lower(), text)


def _canonical_face_direction(value: object) -> str:
    text = str(value or ">Z").strip().upper()
    return _DIRECTION_ALIASES.get(text, text)


class SelectFaceNode(CadQueryNode):
    """Select CAD faces, edges, or vertices by geometric properties."""

    __identifier__ = "com.cad.select_face"
    NODE_NAME = "Select Geometry"

    def __init__(self) -> None:
        super(SelectFaceNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("workplane", color=(100, 200, 255))
        self.create_property(
            "entity_type",
            "Face",
            items=["Face", "Edge", "Vertex"],
            widget_type="combo",
        )
        self.create_property(
            "selector_type",
            "Direction",
            items=[
                "Direction",
                "NearestToPoint",
                "Index",
                "Largest Area",
                "Tag",
                "Box",
                "Coordinate Range",
            ],
            widget_type="combo",
        )
        self.create_property("direction", ">Z", widget_type="string")
        self.create_property("near_x", 0.0, widget_type="float")
        self.create_property("near_y", 0.0, widget_type="float")
        self.create_property("near_z", 0.0, widget_type="float")

        # New Box properties
        self.create_property("box_min_x", -10.0, widget_type="float")
        self.create_property("box_max_x", 10.0, widget_type="float")
        self.create_property("box_min_y", -10.0, widget_type="float")
        self.create_property("box_max_y", 10.0, widget_type="float")
        self.create_property("box_min_z", -10.0, widget_type="float")
        self.create_property("box_max_z", 10.0, widget_type="float")

        # New Coordinate Range property
        self.create_property("range_expr", "(x > 0) & (y < 20)", widget_type="string")

        self.create_property("face_index", 0, widget_type="int")
        self.create_property("tag", "top", widget_type="string")

    def run(self) -> dict[str, Any] | None:
        self.clear_error()
        raw_input = resolve_any_input(self.get_input("shape"))
        method = _canonical_selector_type(self.get_property("selector_type"))
        entity_type = str(self.get_property("entity_type") or "Face").title()
        if entity_type not in {"Face", "Edge", "Vertex"}:
            entity_type = "Face"

        mesh_points = _mesh_points(raw_input)
        if mesh_points is not None:
            return self._run_mesh_selection(
                raw_input,
                mesh_points,
                method,
                entity_type,
            )
        return self._run_cad_selection(raw_input, method, entity_type)

    def _run_mesh_selection(
        self,
        raw_input: Any,
        mesh_points: np.ndarray,
        method: str,
        entity_type: str,
    ) -> dict[str, Any] | None:
        try:
            import numpy as np

            p = mesh_points
            tol = _mesh_axis_tolerance(p)
            ids = np.array([], dtype=int)
            label = method
            surface_faces = None

            if entity_type == "Edge":
                raise ValueError(
                    "A volume mesh has no stable CAD-edge identity. Select "
                    "the edge on the upstream CAD shape, then connect that "
                    "selection to the mesh-based boundary condition."
                )

            if entity_type == "Vertex":
                if method == "Direction":
                    selector = _canonical_face_direction(self.get_property("direction"))
                    axis = {"X": 0, "Y": 1, "Z": 2}.get(selector[-1:])
                    if axis is None or selector[:1] not in {"<", ">"}:
                        raise ValueError(
                            "Vertex direction must be one of <X, >X, <Y, >Y, <Z, >Z."
                        )
                    target = (
                        float(np.max(p[axis]))
                        if selector.startswith(">")
                        else float(np.min(p[axis]))
                    )
                    ids = np.where(np.abs(p[axis] - target) <= tol)[0]
                    label = selector
                elif method == "NearestToPoint":
                    pt = np.asarray(
                        [
                            float(self.get_property("near_x") or 0.0),
                            float(self.get_property("near_y") or 0.0),
                            float(self.get_property("near_z") or 0.0),
                        ]
                    )
                    ids = np.asarray(
                        [int(np.argmin(np.linalg.norm(p.T - pt, axis=1)))],
                        dtype=int,
                    )
                    label = f"nearest {pt.tolist()}"
                elif method == "Index":
                    idx = int(self.get_property("face_index") or 0)
                    if not 0 <= idx < p.shape[1]:
                        raise ValueError(
                            f"Mesh vertex index {idx} is out of range "
                            f"(0..{p.shape[1] - 1})."
                        )
                    ids = np.asarray([idx], dtype=int)
                    label = f"vertex {idx}"
                elif method == "Box":
                    lo = np.minimum(
                        np.asarray(
                            [
                                float(self.get_property("box_min_x") or 0.0),
                                float(self.get_property("box_min_y") or 0.0),
                                float(self.get_property("box_min_z") or 0.0),
                            ]
                        ),
                        np.asarray(
                            [
                                float(self.get_property("box_max_x") or 0.0),
                                float(self.get_property("box_max_y") or 0.0),
                                float(self.get_property("box_max_z") or 0.0),
                            ]
                        ),
                    )
                    hi = np.maximum(
                        np.asarray(
                            [
                                float(self.get_property("box_min_x") or 0.0),
                                float(self.get_property("box_min_y") or 0.0),
                                float(self.get_property("box_min_z") or 0.0),
                            ]
                        ),
                        np.asarray(
                            [
                                float(self.get_property("box_max_x") or 0.0),
                                float(self.get_property("box_max_y") or 0.0),
                                float(self.get_property("box_max_z") or 0.0),
                            ]
                        ),
                    )
                    ids = np.where(np.all((p.T >= lo) & (p.T <= hi), axis=1))[0]
                    label = "box"
                elif method == "Coordinate Range":
                    from pylcss.solver_backends.selection import (
                        nodes_matching_condition,
                    )

                    label = str(self.get_property("range_expr") or "").strip()
                    ids = nodes_matching_condition(
                        raw_input,
                        label,
                        label="Select Vertex mesh range",
                    )
                else:
                    raise ValueError(
                        f"{method} is not meaningful for mesh vertices. "
                        "Use Direction, Nearest Point, Index, Bounding Box, "
                        "or Coordinate Range."
                    )

                payload = _mesh_selection_payload(
                    raw_input,
                    ids,
                    method,
                    label,
                    entity_type="Vertex",
                )
                if payload is None:
                    self.set_error("No mesh vertices matched the selector.")
                return payload

            if method == "Direction":
                selector = _canonical_face_direction(self.get_property("direction"))
                ids, surface_faces = _mesh_direction_selection(raw_input, selector)
                label = selector

            elif method == "NearestToPoint":
                pt = np.asarray(
                    [
                        float(self.get_property("near_x") or 0.0),
                        float(self.get_property("near_y") or 0.0),
                        float(self.get_property("near_z") or 0.0),
                    ]
                )
                boundary = _mesh_boundary_face_data(raw_input)
                if boundary is None:
                    raise ValueError("The mesh has no resolvable exterior surface.")
                distances = np.linalg.norm(boundary["centers"] - pt, axis=1)
                nearest = int(np.argmin(distances))
                surface_faces = boundary["faces"][[nearest]]
                ids = np.unique(surface_faces.reshape(-1))
                label = f"nearest {pt.tolist()}"

            elif method == "Index":
                idx = int(self.get_property("face_index") or 0)
                if not 0 <= idx < len(_MESH_FACE_DIRECTIONS):
                    raise ValueError(
                        f"Mesh face index {idx} is out of range; use 0..{len(_MESH_FACE_DIRECTIONS) - 1}."
                    )
                selector = _MESH_FACE_DIRECTIONS[idx]
                ids, surface_faces = _mesh_direction_selection(raw_input, selector)
                label = selector

            elif method == "Largest Area":
                best = None
                for selector in _MESH_FACE_DIRECTIONS:
                    current, current_faces = _mesh_direction_selection(
                        raw_input, selector
                    )
                    area = _surface_component_metrics(raw_input, current_faces)[0]
                    if best is None or area > best[0]:
                        best = (area, selector, current, current_faces)
                if best is not None:
                    _area, label, ids, surface_faces = best

            elif method == "Box":
                min_pt = np.asarray(
                    [
                        float(self.get_property("box_min_x") or 0.0),
                        float(self.get_property("box_min_y") or 0.0),
                        float(self.get_property("box_min_z") or 0.0),
                    ]
                )
                max_pt = np.asarray(
                    [
                        float(self.get_property("box_max_x") or 0.0),
                        float(self.get_property("box_max_y") or 0.0),
                        float(self.get_property("box_max_z") or 0.0),
                    ]
                )
                lo = np.minimum(min_pt, max_pt)
                hi = np.maximum(min_pt, max_pt)
                boundary = _mesh_boundary_face_data(raw_input)
                if boundary is None:
                    raise ValueError("The mesh has no resolvable exterior surface.")
                mask = np.all(
                    (boundary["centers"] >= lo) & (boundary["centers"] <= hi),
                    axis=1,
                )
                surface_faces = boundary["faces"][mask]
                ids = (
                    np.unique(surface_faces.reshape(-1))
                    if surface_faces.size
                    else np.array([], dtype=int)
                )
                label = "box"

            elif method == "Coordinate Range":
                expr = str(self.get_property("range_expr") or "").strip()
                from pylcss.solver_backends.selection import (
                    nodes_matching_condition,
                )
                from types import SimpleNamespace

                boundary = _mesh_boundary_face_data(raw_input)
                if boundary is None:
                    raise ValueError("The mesh has no resolvable exterior surface.")
                proxy = SimpleNamespace(p=boundary["centers"].T)
                face_ids = nodes_matching_condition(
                    proxy, expr, label="Select Face mesh range"
                )
                surface_faces = boundary["faces"][np.asarray(face_ids, dtype=int)]
                ids = (
                    np.unique(surface_faces.reshape(-1))
                    if surface_faces.size
                    else np.array([], dtype=int)
                )
                label = expr

            payload = _mesh_selection_payload(
                raw_input,
                ids,
                method,
                label,
                surface_faces=locals().get("surface_faces"),
                entity_type="Face",
            )
            if payload is None:
                self.set_error("No mesh nodes matched the selector")
                return None
            return payload
        except Exception as e:
            logger.error(
                "SelectFaceNode (%s): mesh selection failed: %s", self.NODE_NAME, e
            )
            self.set_error(f"Mesh {entity_type.lower()} selection failed: {e}")
            return None

    def _run_cad_selection(
        self,
        raw_input: Any,
        method: str,
        entity_type: str,
    ) -> dict[str, Any] | None:
        if raw_input is not None and any(
            hasattr(raw_input, attr)
            for attr in (
                "val",
                "tessellate",
                "faces",
                "extrude",
                "edges",
                "toCompound",
                "add",
            )
        ):
            shape_input = raw_input
        else:
            shape_input = resolve_shape_input(self.get_input("shape"))
        if not shape_input:
            self.set_error(f"Connect a CAD shape or mesh to Select {entity_type}.")
            return None

        # Convert Assembly to Compound if needed
        if hasattr(shape_input, "toCompound"):
            try:
                shape_val = shape_input.toCompound()
            except Exception:
                shape_val = shape_input
        else:
            shape_val = shape_input

        # Wrap in a Workplane to ensure .faces() returns a Workplane object with .vals()
        if isinstance(shape_val, cq.Workplane):
            obj = shape_val
        else:
            obj = cq.Workplane("XY").newObject([shape_val])

        collection = {
            "Face": obj.faces,
            "Edge": obj.edges,
            "Vertex": obj.vertices,
        }[entity_type]

        def _workplane(
            selection: Any,
            entities: Sequence[Any],
        ) -> Any:
            if entity_type != "Face":
                return None
            try:
                return selection.workplane()
            except Exception:
                try:
                    return obj.newObject(entities).workplane()
                except Exception:
                    return None

        try:
            if method == "Direction":
                selector = _canonical_face_direction(self.get_property("direction"))
                selection = collection(selector)
                entities = selection.vals()
                logger.debug(
                    "SelectFaceNode (%s): Direction %s found %d %s(s)",
                    self.NODE_NAME,
                    selector,
                    len(entities),
                    entity_type,
                )
                if not entities:
                    self.set_error(f"No {entity_type.lower()}s found with selector.")
                    return None
                return _selection_payload(
                    _workplane(selection, entities),
                    entities,
                    method,
                    entity_type,
                )

            elif method == "NearestToPoint":
                pt = (
                    self.get_property("near_x"),
                    self.get_property("near_y"),
                    self.get_property("near_z"),
                )
                selection = collection(cq.NearestToPointSelector(pt))
                entities = selection.vals()
                if not entities:
                    self.set_error(f"No {entity_type.lower()} found near the point.")
                    return None
                return _selection_payload(
                    _workplane(selection, entities),
                    entities,
                    method,
                    entity_type,
                )

            elif method == "Index":
                idx = int(self.get_property("face_index"))
                all_entities = collection().vals()
                if 0 <= idx < len(all_entities):
                    entities = [all_entities[idx]]
                    return _selection_payload(
                        _workplane(obj.newObject(entities), entities),
                        entities,
                        method,
                        entity_type,
                    )
                else:
                    self.set_error(
                        f"{entity_type} index {idx} is out of range "
                        f"(0..{len(all_entities) - 1})."
                    )
                    return None

            elif method == "Largest Area":
                if entity_type == "Vertex":
                    self.set_error(
                        "Largest Area is not meaningful for a vertex. "
                        "Use Nearest Point, Index, Direction, or a spatial range."
                    )
                    return None
                all_entities = collection().vals()
                if not all_entities:
                    self.set_error(
                        f"The connected shape has no {entity_type.lower()}s."
                    )
                    return None
                metric = (
                    (lambda entity: entity.Area())
                    if entity_type == "Face"
                    else (lambda entity: entity.Length())
                )
                selected = [max(all_entities, key=metric)]
                return _selection_payload(
                    _workplane(obj.newObject(selected), selected),
                    selected,
                    method,
                    entity_type,
                )

            elif method == "Tag":
                tag_name = self.get_property("tag")
                selection = collection(tag=tag_name)
                entities = selection.vals()
                if not entities:
                    self.set_error(
                        f"No {entity_type.lower()}s found with tag {tag_name!r}."
                    )
                    return None
                return _selection_payload(
                    _workplane(selection, entities),
                    entities,
                    method,
                    entity_type,
                )

            elif method == "Box":
                # Custom Box Selector
                a = np.asarray(
                    (
                        self.get_property("box_min_x"),
                        self.get_property("box_min_y"),
                        self.get_property("box_min_z"),
                    ),
                    dtype=float,
                )
                b = np.asarray(
                    (
                        self.get_property("box_max_x"),
                        self.get_property("box_max_y"),
                        self.get_property("box_max_z"),
                    ),
                    dtype=float,
                )
                min_pt = np.minimum(a, b)
                max_pt = np.maximum(a, b)

                def in_box(entity: Any) -> bool:
                    c = entity.Center()
                    return (
                        min_pt[0] <= c.x <= max_pt[0]
                        and min_pt[1] <= c.y <= max_pt[1]
                        and min_pt[2] <= c.z <= max_pt[2]
                    )

                entities = [entity for entity in collection().vals() if in_box(entity)]
                if not entities:
                    self.set_error(
                        f"No {entity_type.lower()} centers lie inside the specified box."
                    )
                    return None

                selection = obj.newObject(entities)
                return _selection_payload(
                    _workplane(selection, entities),
                    entities,
                    method,
                    entity_type,
                )

            elif method == "Coordinate Range":
                expr = self.get_property("range_expr")
                entities = []

                try:
                    from simpleeval import simple_eval
                except ImportError:
                    self.set_error(
                        "Coordinate Range requires the simpleeval dependency."
                    )
                    return None

                for entity in collection().vals():
                    c = entity.Center()
                    try:
                        res = simple_eval(expr, names={"x": c.x, "y": c.y, "z": c.z})
                        if res:
                            entities.append(entity)
                    except Exception:
                        continue

                if not entities:
                    self.set_error(
                        f"No {entity_type.lower()} centers matched coordinate "
                        f"expression {expr!r}."
                    )
                    return None

                selection = obj.newObject(entities)
                return _selection_payload(
                    _workplane(selection, entities),
                    entities,
                    method,
                    entity_type,
                )

        except Exception as e:
            logger.error("SelectFaceNode (%s): %s", self.NODE_NAME, e)
            self.set_error(f"{entity_type} selection failed: {e}")
            return None


class InteractiveSelectFaceNode(CadQueryNode):
    """
    Select faces, edges, or vertices by clicking them in the 3D viewport.

    This node stores a list of face indices (integers) chosen by the user
    when they click 'Pick Faces in 3D Viewer' in the Properties Panel.
    Its output is identical to SelectFaceNode — a dict with keys
    ``{'workplane', 'face', 'faces'}`` — so it is a drop-in replacement
    for any downstream FEA node.
    """

    __identifier__ = "com.cad.select_face_interactive"
    NODE_NAME = "Select Geometry (Interactive)"

    def __init__(self) -> None:
        super(InteractiveSelectFaceNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("workplane", color=(100, 200, 255))

        self.create_property(
            "entity_type",
            "Face",
            items=["Face", "Edge", "Vertex"],
            widget_type="combo",
        )
        # Comma-separated entity indices, e.g. "0,2,5".  The historical
        # property name is retained so old .cad files keep loading.
        # Updated programmatically by the Properties Panel picking session.
        self.create_property("picked_face_indices", "", widget_type="string")
        # Human-readable label shown in the Properties Panel.
        self.create_property(
            "selection_label", "No geometry selected", widget_type="string"
        )

    # ------------------------------------------------------------------
    # Public helper: called by the Properties Panel after picking
    # ------------------------------------------------------------------
    def set_picked_entities(self, entity_indices: Sequence[int]) -> None:
        """Store a list of entity indices and update the label."""
        indices_str = ",".join(str(i) for i in entity_indices)
        self.set_property("picked_face_indices", indices_str)
        entity_type = str(self.get_property("entity_type") or "Face").lower()
        n = len(entity_indices)
        if n == 0:
            label = "No geometry selected"
        elif n == 1:
            label = f"1 {entity_type} selected  (idx: {entity_indices[0]})"
        else:
            label = (
                f"{n} {entity_type}s selected  "
                f"(idx: {', '.join(str(i) for i in entity_indices)})"
            )
        self.set_property("selection_label", label)

    def set_picked_faces(self, face_indices: Sequence[int]) -> None:
        """Backward-compatible alias used by existing inspector code."""
        self.set_picked_entities(face_indices)

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------
    def run(self, preview: bool = False) -> dict[str, Any] | None:
        self.clear_error()
        entity_type = str(self.get_property("entity_type") or "Face").title()
        if entity_type not in {"Face", "Edge", "Vertex"}:
            entity_type = "Face"
        raw = self.get_property("picked_face_indices") or ""
        face_indices = [
            int(token) for token in raw.split(",") if token.strip().isdigit()
        ]
        if not face_indices:
            if preview:
                return None
            self.set_error(f"No {entity_type.lower()}s picked yet; use the 3D picker.")
            return None

        raw_input = resolve_any_input(self.get_input("shape"))
        if _mesh_points(raw_input) is not None:
            return self._run_mesh_selection(raw_input, face_indices, entity_type)
        return self._run_cad_selection(face_indices, entity_type)

    def _run_mesh_selection(
        self,
        raw_input: Any,
        face_indices: Sequence[int],
        entity_type: str,
    ) -> dict[str, Any] | None:
        if entity_type == "Vertex":
            payload = _mesh_selection_payload(
                raw_input,
                face_indices,
                "Interactive",
                "picked mesh vertices",
                entity_type="Vertex",
            )
            if payload is None:
                self.set_error(
                    "None of the stored mesh vertex indices exist on this mesh."
                )
            return payload
        if entity_type == "Edge":
            self.set_error(
                "Mesh edges do not retain stable CAD-edge identity after "
                "remeshing. Pick the edge on the upstream CAD shape."
            )
            return None
        selected = []
        for idx in face_indices:
            if idx >= _MESH_COMPONENT_INDEX_BASE:
                selector, ids, surface_faces = _mesh_component_selection(raw_input, idx)
                if selector is None or ids.size == 0:
                    logger.warning(
                        "InteractiveSelectFaceNode: mesh patch index %s could not "
                        "be resolved on this mesh - skipped",
                        idx,
                    )
                    continue
                label = f"patch {idx} / {selector}"

            elif 0 <= idx < len(_MESH_FACE_DIRECTIONS):
                selector = _MESH_FACE_DIRECTIONS[idx]
                label = f"idx {idx} / {selector}"
                # Legacy saved projects used 0..5 to mean whole virtual
                # directional faces.  Keep that behavior for old examples,
                # while new GUI picks store connected patch ids >= 1000.
                ids, surface_faces = _mesh_direction_selection(raw_input, selector)
            else:
                logger.warning(
                    "InteractiveSelectFaceNode: mesh face index %s out of range "
                    "(valid 0..%s or connected patch ids >= %s) - skipped",
                    idx,
                    len(_MESH_FACE_DIRECTIONS) - 1,
                    _MESH_COMPONENT_INDEX_BASE,
                )
                continue

            payload = _mesh_selection_payload(
                raw_input,
                ids,
                "Interactive",
                label,
                surface_faces=surface_faces,
                entity_type="Face",
            )
            if isinstance(payload, dict):
                selected.extend(
                    [f for f in payload.get("faces") or [] if f is not None]
                )

        if not selected:
            self.set_error("None of the stored mesh face indices matched this mesh")
            return None
        return _selection_payload(None, selected, "Interactive", entity_type="Face")

    def _run_cad_selection(
        self,
        face_indices: Sequence[int],
        entity_type: str,
    ) -> dict[str, Any] | None:
        shape_input = resolve_shape_input(self.get_input("shape"))
        if not shape_input:
            self.set_error(
                "Connect a CAD shape or mesh before using interactive selection."
            )
            return None

        # Resolve shape
        if hasattr(shape_input, "toCompound"):
            try:
                shape_val = shape_input.toCompound()
            except Exception:
                shape_val = shape_input
        else:
            shape_val = shape_input

        if isinstance(shape_val, cq.Workplane):
            obj = shape_val
        else:
            obj = cq.Workplane("XY").newObject([shape_val])

        collection = {
            "Face": obj.faces,
            "Edge": obj.edges,
            "Vertex": obj.vertices,
        }[entity_type]
        try:
            all_faces = collection().vals()
        except Exception as e:
            self.set_error(f"Cannot enumerate {entity_type.lower()}s: {e}")
            return None

        selected = []
        for idx in face_indices:
            if 0 <= idx < len(all_faces):
                selected.append(all_faces[idx])
            else:
                logger.warning(
                    f"InteractiveSelectFaceNode: face index {idx} out of range "
                    f"({len(all_faces)} faces total) — skipped"
                )

        if not selected:
            self.set_error(
                f"None of the stored {entity_type.lower()} indices are valid "
                f"(shape has {len(all_faces)} {entity_type.lower()}s)."
            )
            return None

        wp = None
        if entity_type == "Face":
            try:
                wp = obj.newObject(selected).workplane()
            except Exception:
                pass

        return _selection_payload(wp, selected, "Interactive", entity_type=entity_type)
