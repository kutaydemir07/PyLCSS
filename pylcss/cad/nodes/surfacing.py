# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Surfacing Nodes - Advanced Surface and Shape Operations (Loft, Sweep, etc.)
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input
import math

class SweepNode(CadQueryNode):
    """Sweeps a profile along a path."""
    __identifier__ = 'com.cad.sweep'
    NODE_NAME = 'Sweep'

    def __init__(self):
        super(SweepNode, self).__init__()
        self.add_input('profile', color=(100, 200, 255))
        self.add_input('path', color=(100, 200, 255))
        self.add_output('shape', color=(100, 255, 100))

    def run(self):
        profile = resolve_shape_input(self.get_input('profile'))
        path = resolve_shape_input(self.get_input('path'))
        
        if profile is None or path is None:
            return None
        
        try:
            return profile.sweep(path)
        except Exception as e:
            self.set_error(f"Sweep error: {e}")
            return profile


class LoftNode(CadQueryNode):
    """Lofts between multiple profiles."""
    __identifier__ = 'com.cad.loft'
    NODE_NAME = 'Loft'

    def __init__(self):
        super(LoftNode, self).__init__()
        self.add_input('profiles', multi_input=True, color=(100, 200, 255))  # List of profiles
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('ruled', True, widget_type='bool')

    def run(self):
        # Get the profiles port (not the resolved value directly)
        profiles_port = self.get_input('profiles')

        if profiles_port is None or not profiles_port.connected_ports():
            return None

        # Collect shapes from all connected nodes
        profiles = []
        for connected_port in profiles_port.connected_ports():
            node = connected_port.node()
            # Use cached result if available
            shape = getattr(node, '_last_result', None)
            if shape is None:
                shape = node.run()
            if shape is not None:
                profiles.append(shape)


        
        if len(profiles) < 2:
            return profiles[0] if profiles else None

        ruled = self.get_property('ruled')

        try:
            # Extract underlying wire objects from profiles and get their Z positions
            wires_with_pos = []
            for i, p in enumerate(profiles):
                val = p.val() if hasattr(p, 'val') else p
                
                # Get bounding box center Z for sorting
                try:
                    bb = val.BoundingBox()
                    z_pos = bb.center.z
                except:
                    z_pos = i  # Fallback to order received
                
                # Convert Face to Wire if needed
                if hasattr(val, 'outerWire'): 
                    wire = val.outerWire()
                elif hasattr(val, 'Wires'):
                    wire = val
                else:
                    wire = val
                    
                wires_with_pos.append((z_pos, wire))

            # Sort by Z position for proper loft ordering
            wires_with_pos.sort(key=lambda x: x[0])
            wires = [w for _, w in wires_with_pos]
            
            # Create Solid using makeLoft (creates solid if wires are closed)
            lid = cq.Solid.makeLoft(wires, ruled=ruled)
            return cq.Workplane(obj=lid)

        except Exception as e:
            # Fallback for open wires / surfaces
            try:
                lid = cq.Shell.makeLoft(wires, ruled=ruled)
                return cq.Workplane(obj=lid)
            except Exception as e2:
                self.set_error(f"Loft failed: {e}")
                return profiles[0] if profiles else None


class HelixNode(CadQueryNode):
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
        r = resolve_numeric_input(self.get_input('radius'), self.get_property('radius'))
        pitch = resolve_numeric_input(self.get_input('pitch'), self.get_property('pitch'))
        h = resolve_numeric_input(self.get_input('helix_height'), self.get_property('helix_height'))
        
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
