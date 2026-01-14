# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Geometry Nodes - Basic 3D Primitives.
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input

class BoxNode(CadQueryNode):
    """Creates a rectangular box (cube)."""
    __identifier__ = 'com.cad.box'
    NODE_NAME = 'Box'

    def __init__(self):
        super(BoxNode, self).__init__()
        self.add_input('box_length', color=(180, 180, 0))
        self.add_input('box_width', color=(180, 180, 0))
        self.add_input('box_depth', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('box_length', 10.0, widget_type='float')
        self.create_property('box_width', 10.0, widget_type='float')
        self.create_property('box_depth', 10.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        l = self.get_input_value('box_length', 'box_length')
        w = self.get_input_value('box_width', 'box_width')
        d = self.get_input_value('box_depth', 'box_depth')
        x = self.get_input_value('center_x', 'center_x')
        y = self.get_input_value('center_y', 'center_y')
        z = self.get_input_value('center_z', 'center_z')
        
        try:
            box = cq.Workplane("XY").box(float(l), float(w), float(d)).translate((float(x), float(y), float(z)))
            # Tag faces
            try:
                box = box.faces(">Z").tag("top").end() \
                         .faces("<Z").tag("bottom").end() \
                         .faces(">X").tag("right").end() \
                         .faces("<X").tag("left").end() \
                         .faces(">Y").tag("front").end() \
                         .faces("<Y").tag("back").end()
            except AttributeError:
                pass
            return box
        except Exception as e:
            self.set_error(f"Box error: {e}")
            return None


class CylinderNode(CadQueryNode):
    """Creates a cylinder."""
    __identifier__ = 'com.cad.cylinder'
    NODE_NAME = 'Cylinder'

    def __init__(self):
        super(CylinderNode, self).__init__()
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
        r = self.get_input_value('cyl_radius', 'cyl_radius')
        h = self.get_input_value('cyl_height', 'cyl_height')
        x = self.get_input_value('center_x', 'center_x')
        y = self.get_input_value('center_y', 'center_y')
        z = self.get_input_value('center_z', 'center_z')
        return cq.Workplane("XY").cylinder(float(h), float(r)).translate((float(x), float(y), float(z)))


class SphereNode(CadQueryNode):
    """Creates a sphere."""
    __identifier__ = 'com.cad.sphere'
    NODE_NAME = 'Sphere'

    def __init__(self):
        super(SphereNode, self).__init__()
        self.add_input('sphere_radius', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('sphere_radius', 5.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        r = self.get_input_value('sphere_radius', 'sphere_radius')
        x = self.get_input_value('center_x', 'center_x')
        y = self.get_input_value('center_y', 'center_y')
        z = self.get_input_value('center_z', 'center_z')
        return cq.Workplane("XY").sphere(float(r)).translate((float(x), float(y), float(z)))


class ConeNode(CadQueryNode):
    """Creates a cone."""
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
        r1 = resolve_numeric_input(self.get_input('bottom_radius'), self.get_property('bottom_radius'))
        r2 = resolve_numeric_input(self.get_input('top_radius'), self.get_property('top_radius'))
        h = resolve_numeric_input(self.get_input('cone_height'), self.get_property('cone_height'))
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        try:
             qt_cone = cq.Solid.makeCone(float(r1), float(r2), float(h))
             result = cq.Workplane("XY").newObject([qt_cone]).translate((float(x), float(y), float(z)))
             return result
        except Exception as e:
             self.set_error(f"Cone creation error: {e}")
             return None


class TorusNode(CadQueryNode):
    """Creates a torus."""
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
        R = resolve_numeric_input(self.get_input('major_radius'), self.get_property('major_radius'))
        r = resolve_numeric_input(self.get_input('minor_radius'), self.get_property('minor_radius'))
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        return (cq.Workplane("XZ")
                 .moveTo(float(R), 0)
                 .circle(float(r))
                 .revolve(360, (0, 0, 0), (0, 1, 0))
                 .translate((float(x), float(y), float(z))))


class WedgeNode(CadQueryNode):
    """Creates a wedge."""
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
        w = resolve_numeric_input(self.get_input('wedge_width'), self.get_property('wedge_width'))
        l = resolve_numeric_input(self.get_input('length'), self.get_property('length'))
        h = resolve_numeric_input(self.get_input('wedge_height'), self.get_property('wedge_height'))
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        w, l, h = float(w), float(l), float(h)
        points = [(0, 0), (w, 0), (w, h), (0, 0)]
        return cq.Workplane("XZ").polyline(points).close().extrude(l).translate((float(x), float(y), float(z)))


class PyramidNode(CadQueryNode):
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
        base = resolve_numeric_input(self.get_input('base_size'), self.get_property('base_size'))
        h = resolve_numeric_input(self.get_input('pyramid_height'), self.get_property('pyramid_height'))
        sides = self.get_property('sides')
        x = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        y = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        z = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))
        
        base, h = float(base), float(h)
        return (cq.Workplane("XY")
                  .polygon(int(sides), base)
                  .workplane(offset=h)
                  .circle(0.001)
                  .loft(combine=True)
                  .translate((float(x), float(y), float(z))))
