# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode

class MassPropertiesNode(CadQueryNode):
    """Calculate mass properties of a part."""
    __identifier__ = 'com.cad.analysis.mass_properties'
    NODE_NAME = 'Mass Properties'

    def __init__(self):
        super(MassPropertiesNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('properties', color=(255, 200, 100))
        
        self.create_property('density', 7.85e-9, widget_type='float')  # Steel tonne/mm^3

    def run(self):
        shape = self.get_input_shape('shape')
            
        if shape is None:
            return None
        
        density = self.get_property('density')
        
        try:
            # Handle Workplane vs value
            solid = shape.val() if hasattr(shape, 'val') else shape
            
            # Note: Volume() returns mm^3. Density typically tonne/mm^3 in this system.
            # So mass (tonne) = Volume(mm^3) * density(tonne/mm^3)
            # 1 tonne = 1000 kg.
            
            volume_mm3 = solid.Volume()
            mass_tonne = volume_mm3 * float(density)
            mass_kg = mass_tonne * 1000.0
            
            # Use actual center of mass if available
            try:
                com = solid.Center()
                center = (com.x, com.y, com.z)
            except:
                bb = solid.BoundingBox()
                center = (bb.center.x, bb.center.y, bb.center.z)
            
            return {
                'type': 'analysis',
                'property': 'mass_properties',
                'mass': mass_kg,
                'volume': volume_mm3,
                'density': float(density),
                'center_of_mass': center
            }
        except Exception as e:
            self.set_error(f"Mass properties error: {e}")
            return None


class BoundingBoxNode(CadQueryNode):
    """Get bounding box dimensions."""
    __identifier__ = 'com.cad.analysis.bounding_box'
    NODE_NAME = 'Bounding Box'

    def __init__(self):
        super(BoundingBoxNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('dimensions', color=(255, 200, 100))

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None:
            return None
        
        try:
            val = shape.val() if hasattr(shape, 'val') else shape
            bb = val.BoundingBox()
            return {
                'type': 'analysis',
                'property': 'bounding_box',
                'length': bb.xlen,
                'width': bb.ylen,
                'height': bb.zlen,
                'volume': bb.xlen * bb.ylen * bb.zlen
            }
        except Exception as e:
            self.set_error(f"Bounding box error: {e}")
            return None
