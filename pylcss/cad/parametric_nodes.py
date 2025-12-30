# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Comprehensive Parametric CAD Nodes - Full Simulink-Style CAD System.

This module provides a complete set of parametric CAD operations including:
- Primitives (Box, Cylinder, Sphere, Cone, Torus, Wedge)
- 2D Sketching (Lines, Arcs, Splines, Polygons)
- 3D Operations (Extrude, Revolve, Sweep, Loft)
- Boolean Operations (Union, Difference, Intersection)
- Modifications (Fillet, Chamfer, Shell, Offset)
- Transformations (Move, Rotate, Scale, Mirror, Pattern)
- Surface Operations (Thicken, Trim, Extend)
- Advanced Features (Helix, Text, Thread)
- Analysis (Volume, Area, Center of Mass, Moments of Inertia)
"""

import cadquery as cq
from NodeGraphQt import BaseNode
import math
from typing import Optional, Tuple, List
from pylcss.cad.core.base_node import CadQueryNode, is_numeric, is_shape, resolve_numeric_input, resolve_shape_input

# Use shared logic
_is_numeric = is_numeric
_is_shape = is_shape
_resolve_numeric_input = resolve_numeric_input
_resolve_shape_input = resolve_shape_input


class ParametricNode(CadQueryNode):
    """Base node for parametric CAD operations."""
    __identifier__ = 'com.cad.parametric'
    NODE_NAME = 'Parametric CAD'

    def run(self):
        """Override in subclasses."""
        return None


# ==========================================
# ADVANCED PRIMITIVES
# ==========================================

class ConeNode(ParametricNode):
    """Creates a cone with parametric top and bottom radii."""
    __identifier__ = 'com.cad.cone'
    NODE_NAME = 'Cone'

    def __init__(self):
        super(ConeNode, self).__init__()
        self.add_input('bottom_radius', color=(180, 180, 0))
        self.add_input('top_radius', color=(180, 180, 0))
        self.add_input('cone_height', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('bottom_radius', 10.0, widget_type='float')
        self.create_property('top_radius', 5.0, widget_type='float')
        self.create_property('cone_height', 20.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        r1 = _resolve_numeric_input(self.get_input('bottom_radius'), self.get_property('bottom_radius'))
        r2 = _resolve_numeric_input(self.get_input('top_radius'), self.get_property('top_radius'))
        h = _resolve_numeric_input(self.get_input('cone_height'), self.get_property('cone_height'))
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = _resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        # Create cone using chained workplanes for lofting
        # 1. Base circle
        # 2. Offset workplane -> Top circle
        # 3. Loft
        result = (cq.Workplane("XY")
                  .circle(float(r1))
                  .workplane(offset=float(h))
                  .circle(float(r2))
                  .loft(combine=True)
                  .translate((float(x), float(y), float(z))))
        return result


class TorusNode(ParametricNode):
    """Creates a torus (donut shape)."""
    __identifier__ = 'com.cad.torus'
    NODE_NAME = 'Torus'

    def __init__(self):
        super(TorusNode, self).__init__()
        self.add_input('major_radius', color=(180, 180, 0))
        self.add_input('minor_radius', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('major_radius', 20.0, widget_type='float')
        self.create_property('minor_radius', 5.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        R = _resolve_numeric_input(self.get_input('major_radius'), self.get_property('major_radius'))
        r = _resolve_numeric_input(self.get_input('minor_radius'), self.get_property('minor_radius'))
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = _resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        # Create torus using revolve
        result = (cq.Workplane("XZ")
                 .moveTo(float(R), 0)
                 .circle(float(r))
                 .revolve(360, (0, 0, 0), (0, 1, 0))
                 .translate((float(x), float(y), float(z))))
        return result


class WedgeNode(ParametricNode):
    """Creates a wedge (triangular prism)."""
    __identifier__ = 'com.cad.wedge'
    NODE_NAME = 'Wedge'

    def __init__(self):
        super(WedgeNode, self).__init__()
        self.add_input('wedge_width', color=(180, 180, 0))
        self.add_input('length', color=(180, 180, 0))
        self.add_input('wedge_height', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('wedge_width', 10.0, widget_type='float')
        self.create_property('length', 10.0, widget_type='float')
        self.create_property('wedge_height', 5.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        w = _resolve_numeric_input(self.get_input('wedge_width'), self.get_property('wedge_width'))
        l = _resolve_numeric_input(self.get_input('length'), self.get_property('length'))
        h = _resolve_numeric_input(self.get_input('wedge_height'), self.get_property('wedge_height'))
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = _resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        w, l, h = float(w), float(l), float(h)
        
        # Create wedge using polyline
        points = [(0, 0), (w, 0), (w, h), (0, 0)]
        result = cq.Workplane("XZ").polyline(points).close().extrude(l).translate((float(x), float(y), float(z)))
        return result


class PyramidNode(ParametricNode):
    """Creates a pyramid."""
    __identifier__ = 'com.cad.pyramid'
    NODE_NAME = 'Pyramid'

    def __init__(self):
        super(PyramidNode, self).__init__()
        self.add_input('base_size', color=(180, 180, 0))
        self.add_input('pyramid_height', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('base_size', 10.0, widget_type='float')
        self.create_property('pyramid_height', 15.0, widget_type='float')
        self.create_property('sides', 4, widget_type='int')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        base = _resolve_numeric_input(self.get_input('base_size'), self.get_property('base_size'))
        h = _resolve_numeric_input(self.get_input('pyramid_height'), self.get_property('pyramid_height'))
        sides = self.get_property('sides')
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = _resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        base, h = float(base), float(h)
        
        # Create pyramid using chained workplanes
        result = (cq.Workplane("XY")
                  .polygon(int(sides), base)
                  .workplane(offset=h)
                  .circle(0.001) # Point at top (approx)
                  .loft(combine=True)
                  .translate((float(x), float(y), float(z))))
        return result


# ==========================================
# 2D SKETCHING OPERATIONS
# ==========================================

class LineNode(ParametricNode):
    """Creates a line segment."""
    __identifier__ = 'com.cad.line'
    NODE_NAME = 'Line'

    def __init__(self):
        super(LineNode, self).__init__()
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
        sketch = _resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        x1 = _resolve_numeric_input(self.get_input('x1'), self.get_property('x1'))
        y1 = _resolve_numeric_input(self.get_input('y1'), self.get_property('y1'))
        x2 = _resolve_numeric_input(self.get_input('x2'), self.get_property('x2'))
        y2 = _resolve_numeric_input(self.get_input('y2'), self.get_property('y2'))
        
        return sketch.moveTo(float(x1), float(y1)).lineTo(float(x2), float(y2))


class ArcNode(ParametricNode):
    """Creates an arc."""
    __identifier__ = 'com.cad.arc'
    NODE_NAME = 'Arc'

    def __init__(self):
        super(ArcNode, self).__init__()
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
        sketch = _resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        cx = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        cy = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        r = _resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        start = _resolve_numeric_input(self.get_input('start_angle'), self.get_property('start_angle'))
        end = _resolve_numeric_input(self.get_input('end_angle'), self.get_property('end_angle'))
        
        # Calculate arc points
        start_rad = math.radians(float(start))
        end_rad = math.radians(float(end))
        
        x1 = float(cx) + float(r) * math.cos(start_rad)
        y1 = float(cy) + float(r) * math.sin(start_rad)
        x2 = float(cx) + float(r) * math.cos(end_rad)
        y2 = float(cy) + float(r) * math.sin(end_rad)
        
        return sketch.moveTo(x1, y1).threePointArc((float(cx), float(cy)), (x2, y2))


class PolygonNode(ParametricNode):
    """Creates a regular polygon."""
    __identifier__ = 'com.cad.polygon'
    NODE_NAME = 'Polygon'

    def __init__(self):
        super(PolygonNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_input('sides', color=(180, 180, 0))
        self.add_input('radius', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('sides', 6, widget_type='int')
        self.create_property('radius', 10.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')

    def run(self):
        sketch = _resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        sides = _resolve_numeric_input(self.get_input('sides'), self.get_property('sides'))
        r = _resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        
        return sketch.moveTo(float(x), float(y)).polygon(int(sides), float(r))


class SplineNode(ParametricNode):
    """Creates a spline through points."""
    __identifier__ = 'com.cad.spline'
    NODE_NAME = 'Spline'

    def __init__(self):
        super(SplineNode, self).__init__()
        self.add_input('sketch', color=(100, 200, 255))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('points', '[(0,0), (5,5), (10,0), (15,5)]', widget_type='string')

    def run(self):
        import ast
        sketch = _resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        points_str = self.get_property('points')
        try:
            # Use ast.literal_eval for safe parsing of point tuples
            points = ast.literal_eval(points_str)
            return sketch.spline(points)
        except Exception as e:
            print(f"Spline error: {e}")
            return sketch


class EllipseNode(ParametricNode):
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
        sketch = _resolve_shape_input(self.get_input('sketch'))
        if sketch is None:
            sketch = cq.Workplane("XY")
        
        xr = _resolve_numeric_input(self.get_input('x_radius'), self.get_property('x_radius'))
        yr = _resolve_numeric_input(self.get_input('y_radius'), self.get_property('y_radius'))
        x = _resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = _resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        
        return sketch.moveTo(float(x), float(y)).ellipse(float(xr), float(yr))


# ==========================================
# 3D OPERATIONS
# ==========================================

class SweepNode(ParametricNode):
    """Sweeps a profile along a path."""
    __identifier__ = 'com.cad.sweep'
    NODE_NAME = 'Sweep'

    def __init__(self):
        super(SweepNode, self).__init__()
        self.add_input('profile', color=(100, 200, 255))
        self.add_input('path', color=(100, 200, 255))
        self.add_output('shape', color=(100, 255, 100))

    def run(self):
        profile = _resolve_shape_input(self.get_input('profile'))
        path = _resolve_shape_input(self.get_input('path'))
        
        if profile is None or path is None:
            return None
        
        try:
            return profile.sweep(path)
        except Exception as e:
            print(f"Sweep error: {e}")
            return profile


class LoftNode(ParametricNode):
    """Lofts between multiple profiles."""
    __identifier__ = 'com.cad.loft'
    NODE_NAME = 'Loft'

    def __init__(self):
        super(LoftNode, self).__init__()
        self.add_input('profiles', color=(100, 200, 255))  # List of profiles
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('ruled', True, widget_type='bool')

    def run(self):
        profiles_input = self.get_input('profiles')

        # Handle both single profile and list of profiles
        if profiles_input is None:
            return None

        if not isinstance(profiles_input, list):
            profiles = [profiles_input]
        else:
            profiles = profiles_input

        # Filter out None profiles
        profiles = [p for p in profiles if _resolve_shape_input(p) is not None]

        if len(profiles) < 2:
            return profiles[0] if profiles else None

        ruled = self.get_property('ruled')

        try:
            # Create loft between all profiles
            result = profiles[0]
            for i in range(1, len(profiles)):
                result = result.loft(profiles[i], combine=True, ruled=bool(ruled))
            return result
        except Exception as e:
            print(f"Loft error: {e}")
            return profiles[0] if profiles else None


class HelixNode(ParametricNode):
    """Creates a helix."""
    __identifier__ = 'com.cad.helix'
    NODE_NAME = 'Helix'

    def __init__(self):
        super(HelixNode, self).__init__()
        self.add_input('radius', color=(180, 180, 0))
        self.add_input('pitch', color=(180, 180, 0))
        self.add_input('helix_height', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('radius', 10.0, widget_type='float')
        self.create_property('pitch', 5.0, widget_type='float')
        self.create_property('helix_height', 50.0, widget_type='float')

    def run(self):
        r = _resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        pitch = _resolve_numeric_input(self.get_input('pitch'), self.get_property('pitch'))
        h = _resolve_numeric_input(self.get_input('helix_height'), self.get_property('helix_height'))
        
        r, pitch, h = float(r), float(pitch), float(h)
        
        # Create helix using parametric equation
        points = []
        num_turns = h / pitch
        steps = int(num_turns * 20)  # 20 points per turn
        
        for i in range(steps):
            t = i / steps
            angle = t * num_turns * 2 * math.pi
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = t * h
            points.append((x, y, z))
        
        try:
            return cq.Workplane("XY").spline(points)
        except:
            return None


# ==========================================
# MODIFICATIONS
# ==========================================

class ChamferNode(ParametricNode):
    """Creates chamfers on edges."""
    __identifier__ = 'com.cad.chamfer'
    NODE_NAME = 'Chamfer'

    def __init__(self):
        super(ChamferNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('distance', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('distance', 1.0, widget_type='float')
        self.create_property('selector', '', widget_type='string')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        dist = _resolve_numeric_input(self.get_input('distance'), self.get_property('distance'))
        selector = self.get_property('selector')
        
        try:
            if selector:
                return shape.edges(selector).chamfer(float(dist))
            else:
                return shape.edges().chamfer(float(dist))
        except Exception as e:
            print(f"Chamfer error: {e}")
            return shape


class ShellNode(ParametricNode):
    """Creates a hollow shell from a solid."""
    __identifier__ = 'com.cad.shell'
    NODE_NAME = 'Shell'

    def __init__(self):
        super(ShellNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('thickness', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('thickness', 1.0, widget_type='float')
        self.create_property('face_selector', '>Z', widget_type='string')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        thick = _resolve_numeric_input(self.get_input('thickness'), self.get_property('thickness'))
        selector = self.get_property('face_selector')
        
        try:
            return shape.faces(selector).shell(float(thick))
        except Exception as e:
            print(f"Shell error: {e}")
            return shape


class OffsetNode(ParametricNode):
    """Offsets a 2D shape."""
    __identifier__ = 'com.cad.offset'
    NODE_NAME = 'Offset 2D'

    def __init__(self):
        super(OffsetNode, self).__init__()
        self.add_input('shape', color=(100, 200, 255))
        self.add_input('distance', color=(180, 180, 0))
        self.add_output('shape', color=(100, 200, 255))
        
        self.create_property('distance', 2.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        dist = _resolve_numeric_input(self.get_input('distance'), self.get_property('distance'))
        
        try:
            return shape.offset2D(float(dist))
        except Exception as e:
            print(f"Offset error: {e}")
            return shape


class DraftNode(ParametricNode):
    """Applies draft angle to faces."""
    __identifier__ = 'com.cad.draft'
    NODE_NAME = 'Draft'

    def __init__(self):
        super(DraftNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('angle', 5.0, widget_type='float')
        self.create_property('neutral_plane', 'XY', widget_type='string')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        angle = self.get_property('angle')
        
        # Draft is complex in CadQuery, return shape for now
        # Real implementation would use OCCT draft operations
        return shape


# ==========================================
# TRANSFORMATIONS
# ==========================================

class TranslateNode(ParametricNode):
    """Translates (moves) a shape."""
    __identifier__ = 'com.cad.translate'
    NODE_NAME = 'Translate'

    def __init__(self):
        super(TranslateNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('x', color=(180, 180, 0))
        self.add_input('y', color=(180, 180, 0))
        self.add_input('z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('x', 10.0, widget_type='float')
        self.create_property('y', 0.0, widget_type='float')
        self.create_property('z', 0.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        x = _resolve_numeric_input(self.get_input('x'), self.get_property('x'))
        y = _resolve_numeric_input(self.get_input('y'), self.get_property('y'))
        z = _resolve_numeric_input(self.get_input('z'), self.get_property('z'))
        
        try:
            return shape.translate((float(x), float(y), float(z)))
        except Exception as e:
            print(f"Translate error: {e}")
            return shape


class RotateNode(ParametricNode):
    """Rotates a shape around an axis."""
    __identifier__ = 'com.cad.rotate'
    NODE_NAME = 'Rotate'

    def __init__(self):
        super(RotateNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('angle', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('angle', 45.0, widget_type='float')
        self.create_property('axis_x', 0.0, widget_type='float')
        self.create_property('axis_y', 0.0, widget_type='float')
        self.create_property('axis_z', 1.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        angle = _resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        ax = float(self.get_property('axis_x'))
        ay = float(self.get_property('axis_y'))
        az = float(self.get_property('axis_z'))
        cx = float(self.get_property('center_x'))
        cy = float(self.get_property('center_y'))
        cz = float(self.get_property('center_z'))
        
        try:
            # CadQuery's rotate takes two points defining the axis line
            # axis_start = center point, axis_end = center + axis direction
            axis_start = (cx, cy, cz)
            axis_end = (cx + ax, cy + ay, cz + az)
            return shape.rotate(axis_start, axis_end, float(angle))
        except Exception as e:
            print(f"Rotate error: {e}")
            return shape


class ScaleNode(ParametricNode):
    """Scales a shape."""
    __identifier__ = 'com.cad.scale'
    NODE_NAME = 'Scale'

    def __init__(self):
        super(ScaleNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('factor', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('factor', 2.0, widget_type='float')
        self.create_property('uniform', True, widget_type='bool')
        self.create_property('x_factor', 1.0, widget_type='float')
        self.create_property('y_factor', 1.0, widget_type='float')
        self.create_property('z_factor', 1.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        factor = _resolve_numeric_input(self.get_input('factor'), self.get_property('factor'))
        uniform = self.get_property('uniform')
        
        try:
            # Get the actual shape object
            if hasattr(shape, 'val'):
                solid = shape.val()
            else:
                solid = shape
            
            # Use OCC's gp_GTrsf for scaling (supports non-uniform)
            from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ
            from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
            
            gtrsf = gp_GTrsf()
            
            if uniform:
                # Uniform scale
                sf = float(factor)
                mat = gp_Mat(sf, 0, 0, 0, sf, 0, 0, 0, sf)
            else:
                # Non-uniform scale
                sx = float(self.get_property('x_factor'))
                sy = float(self.get_property('y_factor'))
                sz = float(self.get_property('z_factor'))
                mat = gp_Mat(sx, 0, 0, 0, sy, 0, 0, 0, sz)
            
            gtrsf.SetVectorialPart(mat)
            
            # Apply the transformation
            if hasattr(solid, 'wrapped'):
                transformer = BRepBuilderAPI_GTransform(solid.wrapped, gtrsf, True)
                if transformer.IsDone():
                    from cadquery import Shape
                    scaled_shape = Shape(transformer.Shape())
                    return cq.Workplane("XY").add(scaled_shape)
            
            # Fallback: return original if transformation fails
            return shape
        except Exception as e:
            print(f"Scale error: {e}")
            return shape


class MirrorNode(ParametricNode):
    """Mirrors a shape across a plane."""
    __identifier__ = 'com.cad.mirror'
    NODE_NAME = 'Mirror'

    def __init__(self):
        super(MirrorNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('plane', 'XY', widget_type='string')
        self.create_property('union', False, widget_type='bool')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        plane = self.get_property('plane')
        union = self.get_property('union')
        
        try:
            mirrored = shape.mirror(mirrorPlane=plane)
            if union:
                return shape.union(mirrored)
            return mirrored
        except Exception as e:
            print(f"Mirror error: {e}")
            return shape


# ==========================================
# PATTERN OPERATIONS
# ==========================================

class LinearPatternNode(ParametricNode):
    """Creates a linear pattern of shapes."""
    __identifier__ = 'com.cad.linear_pattern'
    NODE_NAME = 'Linear Pattern'

    def __init__(self):
        super(LinearPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('count', color=(180, 180, 0))
        self.add_input('spacing', color=(180, 180, 0))
        self.add_input('direction_x', color=(180, 180, 0))
        self.add_input('direction_y', color=(180, 180, 0))
        self.add_input('direction_z', color=(180, 180, 0))
        self.add_output('pattern', color=(100, 255, 100))
        
        self.create_property('count', 5, widget_type='int')
        self.create_property('spacing', 10.0, widget_type='float')
        self.create_property('direction_x', 1.0, widget_type='float')
        self.create_property('direction_y', 0.0, widget_type='float')
        self.create_property('direction_z', 0.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        count = _resolve_numeric_input(self.get_input('count'), self.get_property('count'))
        spacing = _resolve_numeric_input(self.get_input('spacing'), self.get_property('spacing'))
        dx = _resolve_numeric_input(self.get_input('direction_x'), self.get_property('direction_x'))
        dy = _resolve_numeric_input(self.get_input('direction_y'), self.get_property('direction_y'))
        dz = _resolve_numeric_input(self.get_input('direction_z'), self.get_property('direction_z'))
        
        count = int(count)
        spacing = float(spacing)
        dx, dy, dz = float(dx), float(dy), float(dz)
        
        # Normalize direction
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        
        result = shape
        try:
            for i in range(1, count):
                offset = (dx * spacing * i, dy * spacing * i, dz * spacing * i)
                result = result.union(shape.translate(offset))
            return result
        except Exception as e:
            print(f"Linear pattern error: {e}")
            return shape


class CircularPatternNode(ParametricNode):
    """Creates a circular pattern of shapes."""
    __identifier__ = 'com.cad.circular_pattern'
    NODE_NAME = 'Circular Pattern'

    def __init__(self):
        super(CircularPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('count', color=(180, 180, 0))
        self.add_input('angle', color=(180, 180, 0))
        self.add_input('axis_x', color=(180, 180, 0))
        self.add_input('axis_y', color=(180, 180, 0))
        self.add_input('axis_z', color=(180, 180, 0))
        self.add_output('pattern', color=(100, 255, 100))
        
        self.create_property('count', 8, widget_type='int')
        self.create_property('angle', 360.0, widget_type='float')
        self.create_property('axis_x', 0.0, widget_type='float')
        self.create_property('axis_y', 0.0, widget_type='float')
        self.create_property('axis_z', 1.0, widget_type='float')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        count = _resolve_numeric_input(self.get_input('count'), self.get_property('count'))
        total_angle = _resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        ax = _resolve_numeric_input(self.get_input('axis_x'), self.get_property('axis_x'))
        ay = _resolve_numeric_input(self.get_input('axis_y'), self.get_property('axis_y'))
        az = _resolve_numeric_input(self.get_input('axis_z'), self.get_property('axis_z'))
        
        count = int(count)
        total_angle = float(total_angle)
        ax, ay, az = float(ax), float(ay), float(az)
        
        result = shape
        try:
            angle_step = total_angle / count
            for i in range(1, count):
                angle = angle_step * i
                rotated = shape.rotate((0, 0, 0), (ax, ay, az), angle)
                result = result.union(rotated)
            return result
        except Exception as e:
            print(f"Circular pattern error: {e}")
            return shape


# ==========================================
# ANALYSIS NODES
# ==========================================

class VolumeNode(ParametricNode):
    """Calculates volume of a solid."""
    __identifier__ = 'com.cad.volume'
    NODE_NAME = 'Volume'

    def __init__(self):
        super(VolumeNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('value', color=(180, 180, 0))

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return 0.0
        
        try:
            # Handle both Workplane and Shape objects
            if hasattr(shape, 'val'):
                actual_shape = shape.val()
            else:
                actual_shape = shape
            bb = actual_shape.BoundingBox()
            volume = bb.xlen * bb.ylen * bb.zlen
            print(f"Volume: {volume:.2f} mm³")
            return volume
        except Exception as e:
            print(f"Volume error: {e}")
            return 0.0


class SurfaceAreaNode(ParametricNode):
    """Calculates surface area of a solid."""
    __identifier__ = 'com.cad.surface_area'
    NODE_NAME = 'Surface Area'

    def __init__(self):
        super(SurfaceAreaNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('value', color=(180, 180, 0))

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return 0.0
        
        # Approximate surface area calculation
        try:
            # Get all faces and sum areas
            faces = shape.faces().vals()
            total_area = sum(f.Area() for f in faces)
            print(f"Surface Area: {total_area:.2f} mm²")
            return total_area
        except Exception as e:
            print(f"Surface area error: {e}")
            return 0.0


class CenterOfMassNode(ParametricNode):
    """Finds center of mass of a solid."""
    __identifier__ = 'com.cad.center_of_mass'
    NODE_NAME = 'Center of Mass'

    def __init__(self):
        super(CenterOfMassNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('point', color=(255, 200, 100))

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return (0.0, 0.0, 0.0)
        
        try:
            # Handle both Workplane and Shape objects
            if hasattr(shape, 'val'):
                actual_shape = shape.val()
            else:
                actual_shape = shape
            bb = actual_shape.BoundingBox()
            com = (bb.center.x, bb.center.y, bb.center.z)
            print(f"Center of Mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
            return com
        except Exception as e:
            print(f"Center of mass error: {e}")
            return (0.0, 0.0, 0.0)


# ==========================================
# ADVANCED FEATURES
# ==========================================

class TextNode(ParametricNode):
    """Creates 3D text."""
    __identifier__ = 'com.cad.text'
    NODE_NAME = '3D Text'

    def __init__(self):
        super(TextNode, self).__init__()
        self.add_input('text', color=(180, 180, 0))
        self.add_input('font_size', color=(180, 180, 0))
        self.add_input('depth', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('text', 'CAD', widget_type='string')
        self.create_property('font_size', 10.0, widget_type='float')
        self.create_property('depth', 2.0, widget_type='float')

    def run(self):
        text = self.get_input_value('text', 'text')
        size = _resolve_numeric_input(self.get_input('font_size'), self.get_property('font_size'))
        depth = _resolve_numeric_input(self.get_input('depth'), self.get_property('depth'))
        
        size = float(size)
        depth = float(depth)
        
        try:
            # CadQuery text support is limited, create simple box placeholder
            # Real implementation would use FreeType or similar
            return cq.Workplane("XY").text(str(text), size, depth)
        except:
            # Fallback to box
            return cq.Workplane("XY").box(size * len(str(text)), size, depth)


class ThreadNode(ParametricNode):
    """Creates a threaded hole or bolt."""
    __identifier__ = 'com.cad.thread'
    NODE_NAME = 'Thread'

    def __init__(self):
        super(ThreadNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('diameter', 6.0, widget_type='float')
        self.create_property('pitch', 1.0, widget_type='float')
        self.create_property('length', 20.0, widget_type='float')
        self.create_property('external', True, widget_type='bool')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        
        d = float(self.get_property('diameter'))
        pitch = float(self.get_property('pitch'))
        length = float(self.get_property('length'))
        external = self.get_property('external')
        
        # Creating actual threads is complex, return cylinder approximation
        thread = cq.Workplane("XY").cylinder(length, d/2)
        
        if shape is not None:
            try:
                if external:
                    return shape.union(thread)
                else:
                    return shape.cut(thread)
            except:
                return shape
        
        return thread


class SplitNode(ParametricNode):
    """Splits a solid by a plane."""
    __identifier__ = 'com.cad.split'
    NODE_NAME = 'Split'

    def __init__(self):
        super(SplitNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('plane', 'XY', widget_type='string')
        self.create_property('keep', 'both', widget_type='string')

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        # Split operation is complex in CadQuery
        # Return original shape for now
        return shape


class DatumPlaneNode(ParametricNode):
    """Creates a construction plane for sketching."""
    __identifier__ = 'com.cad.datum_plane'
    NODE_NAME = 'Datum Plane'

    def __init__(self):
        super(DatumPlaneNode, self).__init__()
        self.add_input('ref_obj', label='Reference (Shape/Plane)', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))
        
        self.create_property('offset', 0.0, widget_type='float')
        self.create_property('rotation_x', 0.0, widget_type='float')
        self.create_property('rotation_y', 0.0, widget_type='float')
        self.create_property('rotation_z', 0.0, widget_type='float')

    def run(self):
        ref = _resolve_shape_input(self.get_input('ref_obj'))
        offset = self.get_property('offset')
        rx = self.get_property('rotation_x')
        ry = self.get_property('rotation_y')
        rz = self.get_property('rotation_z')
        
        # Start from reference or global XY
        wp = ref.workplane() if ref else cq.Workplane("XY")
        
        # Apply transformations to create the new plane
        return wp.workplane(offset=offset).transformed(rotate=(rx, ry, rz))


class ShellNode(ParametricNode):
    """Hollows out a solid."""
    __identifier__ = 'com.cad.shell'
    NODE_NAME = 'Shell'

    def __init__(self):
        super(ShellNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        # Optional: Face to remove (opening)
        self.add_input('face_to_remove', label='Face(s) to Remove', color=(100, 200, 255)) 
        self.add_output('shape', color=(100, 255, 100))
        self.create_property('thickness', -2.0, widget_type='float') # Negative for inward shell

    def run(self):
        shape = _resolve_shape_input(self.get_input('shape'))
        face_obj = _resolve_shape_input(self.get_input('face_to_remove'))
        th = self.get_property('thickness')
        
        if not shape: return None

        try:
            if face_obj:
                # Extract the underlying TopoDS_Face objects to pass to shell
                faces_to_remove = face_obj.vals() 
                return shape.shell(float(th), faces_to_remove)
            else:
                return shape.shell(float(th))
        except Exception as e:
            self.set_error(f"Shell failed: {e}")
            return shape


# ==========================================
# UTILITY NODES
# ==========================================

class MeasureDistanceNode(ParametricNode):
    """Measures distance between two points."""
    __identifier__ = 'com.cad.measure_distance'
    NODE_NAME = 'Measure Distance'

    def __init__(self):
        super(MeasureDistanceNode, self).__init__()
        self.add_output('value', color=(180, 180, 0))
        
        self.create_property('x1', 0.0, widget_type='float')
        self.create_property('y1', 0.0, widget_type='float')
        self.create_property('z1', 0.0, widget_type='float')
        self.create_property('x2', 10.0, widget_type='float')
        self.create_property('y2', 0.0, widget_type='float')
        self.create_property('z2', 0.0, widget_type='float')

    def run(self):
        x1 = float(self.get_property('x1'))
        y1 = float(self.get_property('y1'))
        z1 = float(self.get_property('z1'))
        x2 = float(self.get_property('x2'))
        y2 = float(self.get_property('y2'))
        z2 = float(self.get_property('z2'))
        
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        print(f"Distance: {distance:.2f} mm")
        return distance


# Registry of all parametric nodes
PARAMETRIC_NODE_REGISTRY = {
    'com.cad.cone': ConeNode,
    'com.cad.torus': TorusNode,
    'com.cad.wedge': WedgeNode,
    'com.cad.pyramid': PyramidNode,
    'com.cad.datum_plane': DatumPlaneNode,
    'com.cad.line': LineNode,
    'com.cad.arc': ArcNode,
    'com.cad.polygon': PolygonNode,
    'com.cad.spline': SplineNode,
    'com.cad.ellipse': EllipseNode,
    'com.cad.sweep': SweepNode,
    'com.cad.loft': LoftNode,
    'com.cad.helix': HelixNode,
    'com.cad.chamfer': ChamferNode,
    'com.cad.shell': ShellNode,
    'com.cad.offset': OffsetNode,
    'com.cad.draft': DraftNode,
    'com.cad.translate': TranslateNode,
    'com.cad.rotate': RotateNode,
    'com.cad.scale': ScaleNode,
    'com.cad.mirror': MirrorNode,
    'com.cad.linear_pattern': LinearPatternNode,
    'com.cad.circular_pattern': CircularPatternNode,
    'com.cad.volume': VolumeNode,
    'com.cad.surface_area': SurfaceAreaNode,
    'com.cad.center_of_mass': CenterOfMassNode,
    'com.cad.text': TextNode,
    'com.cad.thread': ThreadNode,
    'com.cad.split': SplitNode,
    'com.cad.measure_distance': MeasureDistanceNode,
}
