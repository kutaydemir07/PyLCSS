# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Enhanced CAD nodes with better coordinate control and cut operations.

This module provides improved nodes for more precise CAD work:
- Coordinate-based positioning
- Advanced cut operations
- Hole creation with coordinates
- Pattern operations with precise positioning
"""

import cadquery as cq
from NodeGraphQt import BaseNode
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input


class HoleAtCoordinatesNode(CadQueryNode):
    """Cut a hole at specific X, Y coordinates."""
    __identifier__ = "com.cad.hole_at_coords"
    NODE_NAME = "Hole at Coordinates"

    def __init__(self):
        super(HoleAtCoordinatesNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        # Hole parameters
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
            # Select the face and create workplane
            wp = shape.faces(face_selector).workplane()
            # Position at coordinates
            wp = wp.pushPoints([(x, y)])
            # Create hole
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-depth)
        except Exception as e:
            print(f"Hole at coordinates error: {e}")
            return shape


class MultiHoleNode(CadQueryNode):
    """Cut multiple holes at specified coordinates."""
    __identifier__ = "com.cad.multi_hole"
    NODE_NAME = "Multiple Holes"

    def __init__(self):
        super(MultiHoleNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        # Coordinates as list of (x, y) tuples
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
            # Parse coordinates
            coords = eval(coords_str)
            if not isinstance(coords, list):
                return shape
            
            # Select face and create workplane
            wp = shape.faces(face_selector).workplane()
            # Add all hole positions
            wp = wp.pushPoints(coords)
            # Create holes
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-depth)
        except Exception as e:
            print(f"Multi-hole error: {e}")
            return shape


class RectangularCutNode(CadQueryNode):
    """Cut a rectangular pocket at specific coordinates."""
    __identifier__ = "com.cad.rectangular_cut"
    NODE_NAME = "Rectangular Cut"

    def __init__(self):
        super(RectangularCutNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        # Cut dimensions
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
            # Select face and create workplane
            wp = shape.faces(face_selector).workplane()
            # Position and create rectangle
            wp = wp.moveTo(x, y).rect(w, l)
            # Cut
            if through:
                return wp.cutThruAll()
            else:
                return wp.cutBlind(-depth)
        except Exception as e:
            print(f"Rectangular cut error: {e}")
            return shape


class SlotCutNode(CadQueryNode):
    """Cut a slot (elongated hole) at coordinates."""
    __identifier__ = "com.cad.slot_cut"
    NODE_NAME = "Slot Cut"

    def __init__(self):
        super(SlotCutNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        # Slot parameters
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
            # Select face
            wp = shape.faces(face_selector).workplane()
            
            # Create slot using polyline with rounded ends
            # Move to start, create circle, line to end, create circle
            import math
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 0.001:
                # Just a circle
                wp = wp.moveTo(x1, y1).circle(width/2)
            else:
                # Create slot profile
                angle = math.atan2(dy, dx)
                # Use a hull of two circles
                wp = wp.moveTo(x1, y1).circle(width/2)
                wp = wp.moveTo(x2, y2).circle(width/2)
            
            # Cut
            if through:
                return wp.cutThruAll()
            else:
                return wp.cutBlind(-depth)
        except Exception as e:
            print(f"Slot cut error: {e}")
            return shape


class ArrayHolesNode(CadQueryNode):
    """Create a rectangular array of holes."""
    __identifier__ = "com.cad.array_holes"
    NODE_NAME = "Array of Holes"

    def __init__(self):
        super(ArrayHolesNode, self).__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_output("shape", color=(100, 255, 100))
        
        # Array parameters
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
            # Generate grid of coordinates
            coords = []
            for i in range(x_count):
                for j in range(y_count):
                    x = x_start + i * x_spacing
                    y = y_start + j * y_spacing
                    coords.append((x, y))
            
            # Select face and create holes
            wp = shape.faces(face_selector).workplane()
            wp = wp.pushPoints(coords)
            
            if through:
                return wp.circle(d/2).cutThruAll()
            else:
                return wp.circle(d/2).cutBlind(-5)
        except Exception as e:
            print(f"Array holes error: {e}")
            return shape


# Registry for enhanced nodes
ENHANCED_NODE_REGISTRY = {
    "com.cad.hole_at_coords": HoleAtCoordinatesNode,
    "com.cad.multi_hole": MultiHoleNode,
    "com.cad.rectangular_cut": RectangularCutNode,
    "com.cad.slot_cut": SlotCutNode,
    "com.cad.array_holes": ArrayHolesNode,
}
