# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sketcher Nodes - 2D Profile Creation.
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input
import ast
import math

class SketchNode(CadQueryNode):
    """2D Sketch creation and editing base node."""
    __identifier__ = 'com.cad.sketch'
    NODE_NAME = 'Create Sketch'  # Renamed for clarity

    def __init__(self):
        super(SketchNode, self).__init__()
        self.add_output('sketch', color=(100, 200, 255))
        self.create_property('sketch_name', 'Sketch1', widget_type='string')
        self.create_property('plane', 'XY', widget_type='string')

    def run(self):
        plane = self.get_property('plane')
        return cq.Workplane(plane)


class LineSketchNode(CadQueryNode):
    """Creates a line segment."""
    __identifier__ = 'com.cad.sketch.line'
    NODE_NAME = 'Line'

    def __init__(self):
        super(LineSketchNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('x1', color=(180, 180, 0))
        self.add_input('y1', color=(180, 180, 0))
        self.add_input('x2', color=(180, 180, 0))
        self.add_input('y2', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))

        self.create_property('x1', 0.0, widget_type='float')
        self.create_property('y1', 0.0, widget_type='float')
        self.create_property('x2', 10.0, widget_type='float')
        self.create_property('y2', 10.0, widget_type='float')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")

        x1 = resolve_numeric_input(self.get_input('x1'), self.get_property('x1'))
        y1 = resolve_numeric_input(self.get_input('y1'), self.get_property('y1'))
        x2 = resolve_numeric_input(self.get_input('x2'), self.get_property('x2'))
        y2 = resolve_numeric_input(self.get_input('y2'), self.get_property('y2'))

        try:
            return sketch.moveTo(float(x1), float(y1)).lineTo(float(x2), float(y2))
        except Exception as e:
            self.set_error(f"Line creation failed: {e}")
            return None


class ArcSketchNode(CadQueryNode):
    """Creates an arc."""
    __identifier__ = 'com.cad.sketch.arc'
    NODE_NAME = 'Arc'

    def __init__(self):
        super(ArcSketchNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('radius', color=(180, 180, 0))
        self.add_input('start_angle', color=(180, 180, 0))
        self.add_input('end_angle', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))

        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('radius', 10.0, widget_type='float')
        self.create_property('start_angle', 0.0, widget_type='float')
        self.create_property('end_angle', 90.0, widget_type='float')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")

        cx = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        cy = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        r = resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        start = resolve_numeric_input(self.get_input('start_angle'), self.get_property('start_angle'))
        end = resolve_numeric_input(self.get_input('end_angle'), self.get_property('end_angle'))

        try:
            start_rad = math.radians(float(start))
            end_rad = math.radians(float(end))

            x1 = float(cx) + float(r) * math.cos(start_rad)
            y1 = float(cy) + float(r) * math.sin(start_rad)
            x2 = float(cx) + float(r) * math.cos(end_rad)
            y2 = float(cy) + float(r) * math.sin(end_rad)
            
            mid_rad = (start_rad + end_rad) / 2
            mid_x = float(cx) + float(r) * math.cos(mid_rad)
            mid_y = float(cy) + float(r) * math.sin(mid_rad)

            return sketch.moveTo(x1, y1).threePointArc((mid_x, mid_y), (x2, y2))
        except Exception as e:
            self.set_error(f"Arc creation failed: {e}")
            return None


class ParametricCircleSketchNode(CadQueryNode):
    """Creates a circle."""
    __identifier__ = 'com.cad.sketch.circle'
    NODE_NAME = 'Circle'

    def __init__(self):
        super(ParametricCircleSketchNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('radius', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))

        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('radius', 5.0, widget_type='float')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")

        cx = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        cy = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        r = resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))

        try:
            return sketch.moveTo(float(cx), float(cy)).circle(float(r))
        except Exception as e:
            self.set_error(f"Circle creation failed: {e}")
            return None


class ParametricRectangleSketchNode(CadQueryNode):
    """Creates a rectangle."""
    __identifier__ = 'com.cad.sketch.rectangle'
    NODE_NAME = 'Rectangle' # Renamed from Sketch Rectangle

    def __init__(self):
        super(ParametricRectangleSketchNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('x', color=(180, 180, 0))
        self.add_input('y', color=(180, 180, 0))
        self.add_input('width', color=(180, 180, 0))
        self.add_input('height', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))

        self.create_property('x', 0.0, widget_type='float')
        self.create_property('y', 0.0, widget_type='float')
        self.create_property('width', 10.0, widget_type='float')
        self.create_property('height', 10.0, widget_type='float')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")

        x = resolve_numeric_input(self.get_input('x'), self.get_property('x'))
        y = resolve_numeric_input(self.get_input('y'), self.get_property('y'))
        w = resolve_numeric_input(self.get_input('width'), self.get_property('width'))
        h = resolve_numeric_input(self.get_input('height'), self.get_property('height'))

        try:
            return sketch.moveTo(float(x), float(y)).rect(float(w), float(h))
        except Exception as e:
            self.set_error(f"Rectangle creation failed: {e}")
            return None


class PolygonSketchNode(CadQueryNode):
    """Creates a regular polygon."""
    __identifier__ = 'com.cad.sketch.polygon'
    NODE_NAME = 'Polygon'

    def __init__(self):
        super(PolygonSketchNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('radius', color=(180, 180, 0))
        self.add_input('sides', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))

        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('radius', 5.0, widget_type='float')
        self.create_property('sides', 6, widget_type='int')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")

        cx = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        cy = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        r = resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        sides = resolve_numeric_input(self.get_input('sides'), self.get_property('sides'))

        try:
            return sketch.moveTo(float(cx), float(cy)).polygon(int(sides), float(r))
        except Exception as e:
            self.set_error(f"Polygon creation failed: {e}")
            return None


class SplineNode(CadQueryNode):
    """Creates a spline through points."""
    __identifier__ = 'com.cad.spline'
    NODE_NAME = 'Spline'

    def __init__(self):
        super(SplineNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('points', '[(0,0), (5,5), (10,0), (15,5)]', widget_type='string')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        points_str = self.get_property('points')
        try:
            points = ast.literal_eval(points_str)
            return sketch.spline(points)
        except Exception as e:
            self.set_error(f"Spline error: {e}")
            return sketch


class EllipseNode(CadQueryNode):
    """Creates an ellipse."""
    __identifier__ = 'com.cad.ellipse'
    NODE_NAME = 'Ellipse'

    def __init__(self):
        super(EllipseNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('x_radius', color=(180, 180, 0))
        self.add_input('y_radius', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('x_radius', 10.0, widget_type='float')
        self.create_property('y_radius', 5.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')

    def run(self):
        sketch = resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        xr = resolve_numeric_input(self.get_input('x_radius'), self.get_property('x_radius'))
        yr = resolve_numeric_input(self.get_input('y_radius'), self.get_property('y_radius'))
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        
        return sketch.moveTo(float(x), float(y)).ellipse(float(xr), float(yr))
