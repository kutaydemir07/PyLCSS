# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Modeling Nodes - 3D Operations and Transformations.
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input
import math
import logging

logger = logging.getLogger(__name__)


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


def _canonical_selector_type(value):
    text = str(value or "Direction").strip()
    return _SELECTOR_TYPE_ALIASES.get(text.lower(), text)


def _canonical_face_direction(value):
    text = str(value or ">Z").strip().upper()
    return _DIRECTION_ALIASES.get(text, text)


def _face_summary(face):
    try:
        c = face.Center()
        bb = face.BoundingBox()
        return {
            "center": [float(c.x), float(c.y), float(c.z)],
            "bbox": {
                "xmin": float(bb.xmin), "xmax": float(bb.xmax),
                "ymin": float(bb.ymin), "ymax": float(bb.ymax),
                "zmin": float(bb.zmin), "zmax": float(bb.zmax),
            },
            "area": float(face.Area()),
        }
    except Exception:
        return {}


def _selection_payload(workplane, faces, selector_type):
    faces = list(faces or [])
    return {
        "workplane": workplane,
        "face": faces[0] if faces else None,
        "faces": faces,
        "selector_type": selector_type,
        "face_count": len(faces),
        "face_summaries": [_face_summary(face) for face in faces[:12]],
    }

class SelectFaceNode(CadQueryNode):
    """Select a face robustly based on geometric properties."""
    __identifier__ = 'com.cad.select_face'
    NODE_NAME = 'Select Face'

    def __init__(self):
        super(SelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))
        self.create_property('selector_type', 'Direction', items=['Direction', 'NearestToPoint', 'Index', 'Largest Area', 'Tag', 'Box', 'Coordinate Range'], widget_type='combo')
        self.create_property('direction', '>Z', widget_type='string')
        self.create_property('near_x', 0.0, widget_type='float')
        self.create_property('near_y', 0.0, widget_type='float')
        self.create_property('near_z', 0.0, widget_type='float')
        
        # New Box properties
        self.create_property('box_min_x', -10.0, widget_type='float')
        self.create_property('box_max_x', 10.0, widget_type='float')
        self.create_property('box_min_y', -10.0, widget_type='float')
        self.create_property('box_max_y', 10.0, widget_type='float')
        self.create_property('box_min_z', -10.0, widget_type='float')
        self.create_property('box_max_z', 10.0, widget_type='float')
        
        # New Coordinate Range property
        self.create_property('range_expr', '(x > 0) & (y < 20)', widget_type='string')
        
        self.create_property('face_index', 0, widget_type='int')
        self.create_property('tag', 'top', widget_type='string')

    def run(self):
        shape_input = resolve_shape_input(self.get_input('shape'))
        if not shape_input:
            return None

        # Convert Assembly to Compound if needed
        if hasattr(shape_input, 'toCompound'):
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

        method = _canonical_selector_type(self.get_property('selector_type'))

        try:
            if method == 'Direction':
                selector = _canonical_face_direction(self.get_property('direction'))
                face_selection = obj.faces(selector)
                faces = face_selection.vals()
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Direction {selector} found {len(faces)} faces")
                if not faces:
                    self.set_error("No faces found with selector")
                    return None
                
                try:
                    wp = face_selection.workplane()
                except Exception:
                    wp = None
                return _selection_payload(wp, faces, method)

            elif method == 'NearestToPoint':
                pt = (self.get_property('near_x'), self.get_property('near_y'), self.get_property('near_z'))
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Point={pt}")
                face_selection = obj.faces(cq.NearestToPointSelector(pt))
                faces = face_selection.vals()
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): NearestToPoint found {len(faces)} faces")
                if not faces:
                    self.set_error("No faces found near point")
                    return None
                
                try:
                    wp = face_selection.workplane()
                except Exception:
                    wp = None
                return _selection_payload(wp, faces, method)

            elif method == 'Index':
                idx = int(self.get_property('face_index'))
                all_faces = obj.faces().vals()
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Index={idx}, Total faces={len(all_faces)}")
                if 0 <= idx < len(all_faces):
                    face = all_faces[idx]
                    wp = obj.newObject([face]).workplane()
                    return _selection_payload(wp, [face], method)
                else:
                    self.set_error(f"Face index {idx} out of range")
                    return None

            elif method == 'Largest Area':
                all_faces = obj.faces().vals()
                if not all_faces: 
                    print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): NO FACES FOUND AT ALL")
                    return None
                sorted_faces = sorted(all_faces, key=lambda f: f.Area(), reverse=True)
                largest_face = sorted_faces[0]
                wp = obj.newObject([largest_face]).workplane()
                return _selection_payload(wp, [largest_face], method)

            elif method == 'Tag':
                tag_name = self.get_property('tag')
                face_selection = obj.faces(tag=tag_name)
                faces = face_selection.vals()
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Tag {tag_name} found {len(faces)} faces")
                if not faces:
                    return None
                return _selection_payload(face_selection.workplane(), faces, method)

            elif method == 'Box':
                # Custom Box Selector
                min_pt = (self.get_property('box_min_x'), self.get_property('box_min_y'), self.get_property('box_min_z'))
                max_pt = (self.get_property('box_max_x'), self.get_property('box_max_y'), self.get_property('box_max_z'))
                
                # Check center of faces against box
                def in_box(f):
                    c = f.Center()
                    return (min_pt[0] <= c.x <= max_pt[0] and 
                            min_pt[1] <= c.y <= max_pt[1] and 
                            min_pt[2] <= c.z <= max_pt[2])
                
                all_faces = obj.faces().vals()
                faces = [f for f in all_faces if in_box(f)]
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Box found {len(faces)} faces")
                if not faces:
                    return None
                
                new_wp = obj.newObject(faces)
                return _selection_payload(new_wp, faces, method)

            elif method == 'Coordinate Range':
                # Selector by simpleeval expression on face center
                expr = self.get_property('range_expr')
                all_faces = obj.faces().vals()
                faces = []
                
                # Try to use simpleeval if available
                try:
                    from simpleeval import simple_eval
                except ImportError:
                    simple_eval = None
                
                for f in all_faces:
                    c = f.Center()
                    try:
                        if simple_eval:
                            res = simple_eval(expr, names={'x': c.x, 'y': c.y, 'z': c.z})
                        else:
                            res = eval(expr, {"__builtins__": None}, {'x': c.x, 'y': c.y, 'z': c.z})
                        if res:
                            faces.append(f)
                    except Exception:
                        continue
                
                print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): Coordinate Range found {len(faces)} faces")
                if not faces:
                    return None
                
                new_wp = obj.newObject(faces)
                return _selection_payload(new_wp, faces, method)

        except Exception as e:
            print(f"DEBUG SelectFaceNode ({self.NODE_NAME}): ERROR: {e}")
            self.set_error(f"Face selection failed: {e}")
            return None

class InteractiveSelectFaceNode(CadQueryNode):
    """
    Select faces by interactively clicking them in the 3D viewport.

    This node stores a list of face indices (integers) chosen by the user
    when they click 'Pick Faces in 3D Viewer' in the Properties Panel.
    Its output is identical to SelectFaceNode — a dict with keys
    ``{'workplane', 'face', 'faces'}`` — so it is a drop-in replacement
    for any downstream FEA node.
    """
    __identifier__ = 'com.cad.select_face_interactive'
    NODE_NAME = 'Select Face (Interactive)'

    def __init__(self):
        super(InteractiveSelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))

        # Comma-separated face indices, e.g. "0,2,5"
        # Updated programmatically by the Properties Panel picking session.
        self.create_property('picked_face_indices', '', widget_type='string')
        # Human-readable label shown in the Properties Panel.
        self.create_property('selection_label', 'No faces selected', widget_type='string')

    # ------------------------------------------------------------------
    # Public helper: called by the Properties Panel after picking
    # ------------------------------------------------------------------
    def set_picked_faces(self, face_indices):
        """Store a list of face indices and update the label."""
        indices_str = ','.join(str(i) for i in face_indices)
        self.set_property('picked_face_indices', indices_str)
        n = len(face_indices)
        if n == 0:
            label = 'No faces selected'
        elif n == 1:
            label = f'1 face selected  (idx: {face_indices[0]})'
        else:
            label = f'{n} faces selected  (idx: {", ".join(str(i) for i in face_indices)})'
        self.set_property('selection_label', label)

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------
    def run(self):
        shape_input = resolve_shape_input(self.get_input('shape'))
        if not shape_input:
            return None

        # Parse stored indices
        raw = self.get_property('picked_face_indices') or ''
        face_indices = []
        for tok in raw.split(','):
            tok = tok.strip()
            if tok.isdigit():
                face_indices.append(int(tok))

        if not face_indices:
            self.set_error('No faces picked yet — click "Pick Faces in 3D Viewer"')
            return None

        # Resolve shape
        if hasattr(shape_input, 'toCompound'):
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

        try:
            all_faces = obj.faces().vals()
        except Exception as e:
            self.set_error(f"Cannot enumerate faces: {e}")
            return None

        selected = []
        for idx in face_indices:
            if 0 <= idx < len(all_faces):
                selected.append(all_faces[idx])
            else:
                logger.warning(f"InteractiveSelectFaceNode: face index {idx} out of range "
                               f"({len(all_faces)} faces total) — skipped")

        if not selected:
            self.set_error(f"None of the stored face indices are valid for this shape "
                           f"(shape has {len(all_faces)} faces)")
            return None

        try:
            wp = obj.newObject(selected).workplane()
        except Exception:
            wp = None

        return _selection_payload(wp, selected, "Interactive")
