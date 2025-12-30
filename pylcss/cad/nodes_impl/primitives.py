# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode

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
        
        box = cq.Workplane("XY").box(float(l), float(w), float(d)).translate((float(x), float(y), float(z)))
        
        # Tag faces for robust selection downstream (prevents model breakage)
        try:
            # Tag faces by their orientation for reliable face selection
            box = box.faces(">Z").tag("top").end() \
                     .faces("<Z").tag("bottom").end() \
                     .faces(">X").tag("right").end() \
                     .faces("<X").tag("left").end() \
                     .faces(">Y").tag("front").end() \
                     .faces("<Y").tag("back").end()
        except AttributeError:
            # Fallback for older CadQuery versions that don't support tagging
            pass
        
        return box


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
