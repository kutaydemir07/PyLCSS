# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Advanced CAD Node Library - Sketch, Assembly, and Analysis nodes.

Includes:
- Sketch nodes (Rectangle, Circle)
- Assembly nodes (Assembly)
- Analysis nodes (Mass properties, Bounding box)
"""
import cadquery as cq
from NodeGraphQt import BaseNode
from datetime import datetime
from pylcss.cad.core.base_node import CadQueryNode


class AdvancedCadNode(CadQueryNode):
    """Base node for advanced CAD operations."""
    __identifier__ = 'com.cad.advanced'
    NODE_NAME = 'Advanced CAD'

    def run(self):
        return None


# ==========================================
# SKETCH NODES
# ==========================================

class SketchNode(AdvancedCadNode):
    """2D Sketch creation and editing."""
    __identifier__ = 'com.cad.sketch'
    NODE_NAME = 'Sketch'

    def __init__(self):
        super(SketchNode, self).__init__()
        self.add_output('sketch', color=(100, 200, 255))
        self.create_property('sketch_name', 'Sketch1', widget_type='string')
        self.create_property('plane', 'XY', widget_type='string')

    def run(self):
        plane = self.get_property('plane')
        return cq.Workplane(plane)



# ==========================================
# ASSEMBLY NODES
# ==========================================

class AssemblyNode(AdvancedCadNode):
    """Create an assembly from multiple parts."""
    __identifier__ = 'com.cad.assembly'
    NODE_NAME = 'Assembly'

    def __init__(self):
        super(AssemblyNode, self).__init__()
        self.add_input('part_1', color=(100, 255, 100))
        self.add_input('part_2', color=(100, 255, 100))
        self.add_input('part_3', color=(100, 255, 100))
        self.add_input('part_4', color=(100, 255, 100))
        
        self.add_output('assembly', color=(200, 150, 100))
        self.create_property('assembly_name', 'Assembly1', widget_type='string')

    def run(self):
        asm = cq.Assembly(name=self.get_property('assembly_name'))
        
        parts = []
        for i in range(1, 5):
            port_name = f'part_{i}'
            port = self.get_input(port_name)
            if port and port.connected_ports():
                try:
                    node = port.connected_ports()[0].node()
                    # Use cached result if available
                    part = getattr(node, '_last_result', None)
                    if part is None:
                        part = node.run()
                    if part:
                        parts.append(part)
                except Exception as e:
                    print(f"AssemblyNode: Error getting part {i}: {e}")
                    pass
        
        if not parts:
            return None
            
        for idx, part in enumerate(parts):
            asm.add(part, name=f"part_{idx}")
            
        return asm


# ==========================================
# ANALYSIS NODES
# ==========================================

class MassPropertiesNode(AdvancedCadNode):
    """Calculate mass properties of a part."""
    __identifier__ = 'com.cad.mass_properties'
    NODE_NAME = 'Mass Properties'

    def __init__(self):
        super(MassPropertiesNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('properties', color=(255, 200, 100))
        
        self.create_property('density', 7850.0, widget_type='float')  # Steel kg/m^3

    def run(self):
        shape = self._get_input_shape()
        if shape is None:
            return None
        
        density = self.get_property('density')
        
        try:
            solid = shape.val()
            # Use actual solid volume, not bounding box approximation
            volume = solid.Volume() / 1e9  # Convert mm^3 to m^3
            mass = volume * float(density)
            
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
                'mass': mass,
                'volume': volume,
                'density': float(density),
                'center_of_mass': center
            }
        except Exception as e:
            print(f"Mass properties error: {e}")
            return None

    def _get_input_shape(self):
        port = self.get_input('shape')
        if port and port.connected_ports():
            node = port.connected_ports()[0].node()
            # Use cached result if available
            res = getattr(node, '_last_result', None)
            if res is None:
                res = node.run()
            return res
        return None


class BoundingBoxNode(AdvancedCadNode):
    """Get bounding box dimensions."""
    __identifier__ = 'com.cad.bounding_box'
    NODE_NAME = 'Bounding Box'

    def __init__(self):
        super(BoundingBoxNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('dimensions', color=(255, 200, 100))

    def run(self):
        port = self.get_input('shape')
        if not port or not port.connected_ports():
            return None
        
        node = port.connected_ports()[0].node()
        # Use cached result if available
        shape = getattr(node, '_last_result', None)
        if shape is None:
            shape = node.run()
        if shape is None:
            return None
        
        try:
            bb = shape.val().BoundingBox()
            return {
                'type': 'analysis',
                'property': 'bounding_box',
                'length': bb.xlen,
                'width': bb.ylen,
                'height': bb.zlen,
                'volume': bb.xlen * bb.ylen * bb.zlen
            }
        except Exception as e:
            print(f"Bounding box error: {e}")
            return None
