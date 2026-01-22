# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Gear Generation Tools for the Agent.
Allows creating complex helical gears by procedural generation of the CAD graph.
"""

import math
from typing import Dict, Any, List

def create_helical_gear(
    module: float = 1.0,
    teeth: int = 20,
    width: float = 10.0,
    helix_angle: float = 20.0,
    pressure_angle: float = 20.0,
    center_x: float = 0.0,
    center_y: float = 0.0
) -> Dict[str, Any]:
    """
    Constructs a CAD graph for a helical gear.
    
    Args:
        module: Gear module (size scaler).
        teeth: Number of teeth.
        width: Gear width (thickness).
        helix_angle: Helix angle in degrees (0 for spur gear).
        pressure_angle: Pressure angle in degrees.
        center_x: X position.
        center_y: Y position.
        
    Returns:
        Dict with "nodes" and "connections" to build the graph.
    """
    
    # --- Gear Calculations ---
    # Pitch Diameter = Module * Teeth
    pd = module * teeth
    pitch_radius = pd / 2.0
    
    # Addendum = Module
    addendum = module
    # Dedendum = 1.25 * Module
    dedendum = 1.25 * module
    
    outer_radius = pitch_radius + addendum
    root_radius = pitch_radius - dedendum
    
    # Calculate angular pitch (360 / teeth)
    pitch_angle = 360.0 / teeth
    
    # Tooth thickness at pitch circle (approx half pitch)
    # Arc length = pi * module / 2
    # Angular thickness = (pi * module / 2) / pitch_radius * (180/pi) = 90 / teeth
    tooth_half_angle = 90.0 / teeth
    
    # Simple Trapezoidal Profile Points (Polar to Cartesian)
    # We define one tooth profile
    
    # Points on Root Circle
    r1 = root_radius
    a1 = -tooth_half_angle - (pressure_angle / teeth * 2) # Widen base slightly
    p1 = (r1 * math.cos(math.radians(a1)), r1 * math.sin(math.radians(a1)))
    
    a4 = tooth_half_angle + (pressure_angle / teeth * 2)
    p4 = (r1 * math.cos(math.radians(a4)), r1 * math.sin(math.radians(a4)))
    
    # Points on Outer Circle
    r2 = outer_radius
    a2 = -tooth_half_angle + (pressure_angle / teeth * 0.5) # Narrow tip
    p2 = (r2 * math.cos(math.radians(a2)), r2 * math.sin(math.radians(a2)))
    
    a3 = tooth_half_angle - (pressure_angle / teeth * 0.5)
    p3 = (r2 * math.cos(math.radians(a3)), r2 * math.sin(math.radians(a3)))
    
    # Scale points for PolylineNode (expected as list of tuples)
    points = [p1, p2, p3, p4, p1] # Closed loop
    
    # --- Graph Construction ---
    
    nodes = []
    connections = []
    
    # 1. Gear Body (Cylinder) at Root Radius
    nodes.append({
        "type": "com.cad.cylinder",
        "id": "gear_body",
        "properties": {
            "cyl_radius": root_radius,
            "cyl_height": width,
            "center_x": center_x,
            "center_y": center_y
        }
    })
    
    # 2. Tooth Profile (Polyline)
    # PolylineNode takes points as a list of (x, y) tuples
    nodes.append({
        "type": "com.cad.polyline",
        "id": "tooth_profile",
        "properties": {
            "points": str(points), # PolylineNode parses string rep of list
            "closed": True
        }
    })
    
    # 3. Extrude Tooth (Initial shape)
    nodes.append({
        "type": "com.cad.extrude",
        "id": "tooth_shape",
        "properties": {
            "extrude_distance": width
        }
    })
    
    # 4. Twist (for Helical)
    if abs(helix_angle) > 0.1:
        nodes.append({
            "type": "com.cad.twisted_extrude",
            "id": "helical_tooth",
            "properties": {
                "distance": width,
                "angle": helix_angle
            }
        })
        # Connect Profile -> Twist
        connections.append({"from": "tooth_profile.shape", "to": "helical_tooth.shape"})
        tooth_node_id = "helical_tooth"
    else:
         # Connect Profile -> Extrude
        connections.append({"from": "tooth_profile.shape", "to": "tooth_shape.shape"})
        tooth_node_id = "tooth_shape"
        
    # 5. Position Tooth
    # Translate to center
    nodes.append({
        "type": "com.cad.translate",
        "id": "positioned_tooth",
        "properties": {
            "x": center_x,
            "y": center_y,
            "z": 0.0
        }
    })
    connections.append({"from": f"{tooth_node_id}.shape", "to": "positioned_tooth.shape"})
    
    # 6. Circular Pattern
    nodes.append({
        "type": "com.cad.circular_pattern",
        "id": "all_teeth",
        "properties": {
            "count": teeth,
            "angle": 360.0,
            "axis_z": 1.0
        }
    })
    connections.append({"from": "positioned_tooth.shape", "to": "all_teeth.shape"})

    # 7. Union Body + Teeth
    nodes.append({
        "type": "com.cad.boolean",
        "id": "final_gear",
        "properties": {
            "operation": "Union"
        }
    })
    connections.append({"from": "gear_body.shape", "to": "final_gear.shape_a"})
    connections.append({"from": "all_teeth.shape", "to": "final_gear.shape_b"})
    
    return {
        "params": {
            "nodes": nodes,
            "connections": connections
        }
    }
