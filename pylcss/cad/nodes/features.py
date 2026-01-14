# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Feature Nodes - Engineering Features (Holes, Pockets, Slots).
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input
import ast
import math

class HoleAtCoordinatesNode(CadQueryNode):
    """Cut a hole at specific X, Y coordinates."""
    __identifier__ = "com.cad.hole_at_coords"
    NODE_NAME = "Hole (Point)"

    def __init__(self):
        super(HoleAtCoordinatesNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        self.create_property("x_position", 10.0, widget_type="float")
        self.create_property("y_position", 10.0, widget_type="float")
        self.create_property("diameter", 5.0, widget_type="float")
        self.create_property("depth", 10.0, widget_type="float")
        self.create_property("through_all", True, widget_type="bool")
        self.create_property("from_face", ">Z", widget_type="string")

    def run(self):
        shape = resolve_shape_input(self.get_input("shape"))
        if shape is None:
            return None
        
        x = float(self.get_property("x_position"))
        y = float(self.get_property("y_position"))
        d = float(self.get_property("diameter"))
        depth = float(self.get_property("depth"))
        through = self.get_property("through_all")
        face_selector = self.get_property("from_face")
        
        try:
            wp = shape.faces(face_selector).workplane()
            wp = wp.pushPoints([(x, y)])
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-depth)
        except Exception as e:
            self.set_error(f"Hole error: {e}")
            return shape


class MultiHoleNode(CadQueryNode):
    """Cut multiple holes at specified coordinates."""
    __identifier__ = "com.cad.multi_hole"
    NODE_NAME = "Holes (Multi)"

    def __init__(self):
        super(MultiHoleNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        self.create_property("coordinates", "[(10,10), (40,10), (10,30), (40,30)]", widget_type="string")
        self.create_property("diameter", 5.0, widget_type="float")
        self.create_property("depth", 10.0, widget_type="float")
        self.create_property("through_all", True, widget_type="bool")
        self.create_property("from_face", ">Z", widget_type="string")

    def run(self):
        shape = resolve_shape_input(self.get_input("shape"))
        if shape is None:
            return None
        
        coords_str = self.get_property("coordinates")
        d = float(self.get_property("diameter"))
        depth = float(self.get_property("depth"))
        through = self.get_property("through_all")
        face_selector = self.get_property("from_face")
        
        try:
            coords = ast.literal_eval(coords_str)
            if not isinstance(coords, list):
                return shape
            
            wp = shape.faces(face_selector).workplane()
            wp = wp.pushPoints(coords)
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-depth)
        except Exception as e:
            self.set_error(f"Multi-hole error: {e}")
            return shape


class PocketNode(CadQueryNode):
    """Cuts a hole (pocket) in a shape."""
    __identifier__ = 'com.cad.pocket'
    NODE_NAME = 'Pocket'

    def __init__(self):
        super(PocketNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('pocket_depth', color=(180, 180, 0))
        self.add_input('pocket_radius', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('pocket_depth', 5.0, widget_type='float')
        self.create_property('pocket_radius', 2.0, widget_type='float')

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        depth = resolve_numeric_input(self.get_input('pocket_depth'), self.get_property('pocket_depth'))
        radius = resolve_numeric_input(self.get_input('pocket_radius'), self.get_property('pocket_radius'))
        
        if shape is None:
            return None
        
        try:
            return shape.faces(">Z").workplane().circle(float(radius)).cutBlind(float(depth))
        except Exception as e:
            self.set_error(f"Pocket error: {e}")
            return shape


class RectangularCutNode(CadQueryNode):
    """Cut a rectangular pocket at specific coordinates."""
    __identifier__ = "com.cad.rectangular_cut"
    NODE_NAME = "Rectangular Pocket"

    def __init__(self):
        super(RectangularCutNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        self.create_property("x_position", 10.0, widget_type="float")
        self.create_property("y_position", 10.0, widget_type="float")
        self.create_property("cut_width", 15.0, widget_type="float")
        self.create_property("cut_length", 20.0, widget_type="float")
        self.create_property("depth", 5.0, widget_type="float")
        self.create_property("through_all", False, widget_type="bool")
        self.create_property("from_face", ">Z", widget_type="string")

    def run(self):
        shape = resolve_shape_input(self.get_input("shape"))
        if shape is None:
            return None
        
        x = float(self.get_property("x_position"))
        y = float(self.get_property("y_position"))
        w = float(self.get_property("cut_width"))
        l = float(self.get_property("cut_length"))
        depth = float(self.get_property("depth"))
        through = self.get_property("through_all")
        face_selector = self.get_property("from_face")
        
        try:
            wp = shape.faces(face_selector).workplane()
            wp = wp.moveTo(x, y).rect(w, l)
            if through:
                return wp.cutThruAll()
            else:
                return wp.cutBlind(-depth)
        except Exception as e:
            self.set_error(f"Rect cut error: {e}")
            return shape


class SlotCutNode(CadQueryNode):
    """Cut a slot."""
    __identifier__ = "com.cad.slot_cut"
    NODE_NAME = "Slot"

    def __init__(self):
        super(SlotCutNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        self.create_property("x_start", 10.0, widget_type="float")
        self.create_property("y_start", 10.0, widget_type="float")
        self.create_property("x_end", 30.0, widget_type="float")
        self.create_property("y_end", 10.0, widget_type="float")
        self.create_property("slot_width", 5.0, widget_type="float")
        self.create_property("depth", 5.0, widget_type="float")
        self.create_property("through_all", False, widget_type="bool")
        self.create_property("from_face", ">Z", widget_type="string")

    def run(self):
        shape = resolve_shape_input(self.get_input("shape"))
        if shape is None:
            return None
        
        x1 = float(self.get_property("x_start"))
        y1 = float(self.get_property("y_start"))
        x2 = float(self.get_property("x_end"))
        y2 = float(self.get_property("y_end"))
        width = float(self.get_property("slot_width"))
        depth = float(self.get_property("depth"))
        through = self.get_property("through_all")
        face_selector = self.get_property("from_face")
        
        try:
            wp = shape.faces(face_selector).workplane()
            
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 0.001:
                wp = wp.moveTo(x1, y1).circle(width/2)
            else:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                angle_deg = math.degrees(math.atan2(dy, dx))
                wp = wp.center(cx, cy).slot2D(length + width, width, angle_deg)
            
            if through:
                return wp.cutThruAll()
            else:
                return wp.cutBlind(-depth)
        except Exception as e:
            self.set_error(f"Slot error: {e}")
            return shape


class ArrayHolesNode(CadQueryNode):
    """Create a rectangular array of holes."""
    __identifier__ = "com.cad.array_holes"
    NODE_NAME = "Hole Array"

    def __init__(self):
        super(ArrayHolesNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        self.create_property("x_start", 10.0, widget_type="float")
        self.create_property("y_start", 10.0, widget_type="float")
        self.create_property("x_spacing", 20.0, widget_type="float")
        self.create_property("y_spacing", 15.0, widget_type="float")
        self.create_property("x_count", 3, widget_type="int")
        self.create_property("y_count", 2, widget_type="int")
        self.create_property("diameter", 5.0, widget_type="float")
        self.create_property("through_all", True, widget_type="bool")
        self.create_property("from_face", ">Z", widget_type="string")

    def run(self):
        shape = resolve_shape_input(self.get_input("shape"))
        if shape is None:
            return None
        
        x_start = float(self.get_property("x_start"))
        y_start = float(self.get_property("y_start"))
        x_spacing = float(self.get_property("x_spacing"))
        y_spacing = float(self.get_property("y_spacing"))
        x_count = int(self.get_property("x_count"))
        y_count = int(self.get_property("y_count"))
        d = float(self.get_property("diameter"))
        through = self.get_property("through_all")
        face_selector = self.get_property("from_face")
        
        try:
            coords = []
            for i in range(x_count):
                for j in range(y_count):
                    x = x_start + i * x_spacing
                    y = y_start + j * y_spacing
                    coords.append((x, y))
            
            wp = shape.faces(face_selector).workplane()
            wp = wp.pushPoints(coords)
            
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-5)
        except Exception as e:
            self.set_error(f"Array holes error: {e}")
            return shape


class CylinderCutNode(CadQueryNode):
    """Create a cylinder tool and cut it from a target solid."""
    __identifier__ = 'com.cad.cylinder_cut'
    NODE_NAME = 'Cylinder Cut'

    def __init__(self):
        super(CylinderCutNode, self).__init__()
        self.add_input('target', color=(100, 255, 100))
        self.add_input('cyl_radius', color=(180, 180, 0))
        self.add_input('cyl_height', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('cyl_radius', 5.0, widget_type='float')
        self.create_property('cyl_height', 10.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        target = resolve_shape_input(self.get_input('target'))
        if target is None:
            return None

        r = resolve_numeric_input(self.get_input('cyl_radius'), self.get_property('cyl_radius'))
        h = resolve_numeric_input(self.get_input('cyl_height'), self.get_property('cyl_height'))
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))

        try:
            r, h = float(r), float(h)
            x, y, z = float(x), float(y), float(z)
            tool = cq.Workplane("XY").cylinder(h, r).translate((x, y, z + h/2))
            return target.cut(tool)
        except Exception as e:
            self.set_error(f"CylinderCut error: {e}")
            return target
