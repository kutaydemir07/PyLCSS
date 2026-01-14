"""
Boeing 747-400 CAD Model (Clean Version)
=========================================
Clean, properly aligned model with:
- Lofted fuselage with hump
- Swept wings with winglets
- 4 engine nacelles with pylons
- Vertical and horizontal stabilizers

Coordinate System:
  X = Longitudinal (Nose=0, Tail=+X, Length ~70)
  Y = Vertical (Down=-Y, Up=+Y)  
  Z = Lateral (Left=+Z, Right=-Z)
"""

import json
import uuid

class CadGraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self._x = 0
        self._y = 0

    def node(self, identifier, name, props=None, x=None, y=None):
        node_id = "0x" + uuid.uuid4().hex[:12]
        if x is None:
            x = self._x
            self._x += 200
        if y is None:
            y = self._y
        
        class_map = {
            "com.cad.box": "BoxNode",
            "com.cad.cylinder": "CylinderNode",
            "com.cad.sketch.circle": "ParametricCircleSketchNode",
            "com.cad.ellipse": "EllipseNode",
            "com.cad.loft": "LoftNode",
            "com.cad.translate": "TranslateNode",
            "com.cad.rotate": "RotateNode",
            "com.cad.boolean": "BooleanNode",
            "com.cad.assembly.assembly": "AssemblyNode",
        }
        class_name = class_map.get(identifier, "Node")
        
        self.nodes[node_id] = {
            "type_": f"{identifier}.{class_name}",
            "name": name,
            "custom": props if props else {},
            "pos": [x, y],
            "icon": None, "color": [13, 18, 23, 255], "border_color": [74, 84, 85, 255],
            "text_color": [255, 255, 255, 180], "disabled": False, "selected": False,
            "visible": True, "width": 160, "height": 100, "layout_direction": 0,
            "port_deletion_allowed": False, "subgraph_session": {}
        }
        return node_id

    def connect(self, src, dst, src_port="shape", dst_port="shape"):
        self.connections.append({"out": [src, src_port], "in": [dst, dst_port]})

    def row(self, dy=150):
        self._x = 0
        self._y += dy

    def save(self, path):
        data = {
            "graph": {"layout_direction": 0, "acyclic": True, "pipe_collision": False,
                      "pipe_slicing": True, "pipe_style": 1},
            "nodes": self.nodes,
            "connections": self.connections
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")


g = CadGraphBuilder()

# =============================================================================
# FUSELAGE (11-section loft with hump)
# =============================================================================
fuse_profiles = []

# Nose tip
p = g.node("com.cad.sketch.circle", "Fuse S0 Nose", {"radius": 0.5})
t = g.node("com.cad.translate", "Fuse S0 Pos", {"x": 0.0, "y": 0.0, "z": 0.0})
g.connect(p, t)
fuse_profiles.append(t)

# Nose expanding
p = g.node("com.cad.sketch.circle", "Fuse S3", {"radius": 2.0})
t = g.node("com.cad.translate", "Fuse S3 Pos", {"x": 0.0, "y": 0.0, "z": 3.0})
g.connect(p, t)
fuse_profiles.append(t)

# Approaching cockpit
p = g.node("com.cad.sketch.circle", "Fuse S7", {"radius": 3.0})
t = g.node("com.cad.translate", "Fuse S7 Pos", {"x": 0.0, "y": 0.0, "z": 7.0})
g.connect(p, t)
fuse_profiles.append(t)

# Start of hump
p = g.node("com.cad.ellipse", "Fuse S10 Hump", {"x_radius": 3.25, "y_radius": 4.0})
t = g.node("com.cad.translate", "Fuse S10 Pos", {"x": 0.0, "y": 0.5, "z": 10.0})
g.connect(p, t)
fuse_profiles.append(t)

# Max hump
p = g.node("com.cad.ellipse", "Fuse S15 Hump Max", {"x_radius": 3.25, "y_radius": 4.75})
t = g.node("com.cad.translate", "Fuse S15 Pos", {"x": 0.0, "y": 0.75, "z": 15.0})
g.connect(p, t)
fuse_profiles.append(t)

# Hump end
p = g.node("com.cad.ellipse", "Fuse S22 Hump End", {"x_radius": 3.25, "y_radius": 4.0})
t = g.node("com.cad.translate", "Fuse S22 Pos", {"x": 0.0, "y": 0.375, "z": 22.0})
g.connect(p, t)
fuse_profiles.append(t)

# Main body
p = g.node("com.cad.sketch.circle", "Fuse S28", {"radius": 3.25})
t = g.node("com.cad.translate", "Fuse S28 Pos", {"x": 0.0, "y": 0.0, "z": 28.0})
g.connect(p, t)
fuse_profiles.append(t)

# Main body continues
p = g.node("com.cad.sketch.circle", "Fuse S50", {"radius": 3.25})
t = g.node("com.cad.translate", "Fuse S50 Pos", {"x": 0.0, "y": 0.0, "z": 50.0})
g.connect(p, t)
fuse_profiles.append(t)

# Tail taper
p = g.node("com.cad.sketch.circle", "Fuse S58", {"radius": 2.5})
t = g.node("com.cad.translate", "Fuse S58 Pos", {"x": 0.0, "y": 0.3, "z": 58.0})
g.connect(p, t)
fuse_profiles.append(t)

# Tail cone
p = g.node("com.cad.sketch.circle", "Fuse S65", {"radius": 1.2})
t = g.node("com.cad.translate", "Fuse S65 Pos", {"x": 0.0, "y": 0.6, "z": 65.0})
g.connect(p, t)
fuse_profiles.append(t)

# Tail tip
p = g.node("com.cad.sketch.circle", "Fuse S70 Tail", {"radius": 0.3})
t = g.node("com.cad.translate", "Fuse S70 Pos", {"x": 0.0, "y": 0.8, "z": 70.0})
g.connect(p, t)
fuse_profiles.append(t)

g.row()

fuselage_loft = g.node("com.cad.loft", "Fuselage Loft", {"ruled": False})
for profile in fuse_profiles:
    g.connect(profile, fuselage_loft, "shape", "profiles")

fuselage = g.node("com.cad.rotate", "Fuselage",
    {"angle": 90.0, "axis_x": 0.0, "axis_y": 1.0, "axis_z": 0.0})
g.connect(fuselage_loft, fuselage)
g.row(200)

# =============================================================================
# DISPLAY STAND (elegant curved arm design)
# =============================================================================
# Elegant base (larger, thinner, further down)
stand_base = g.node("com.cad.box", "Stand Base", 
    {"box_length": 25.0, "box_width": 0.8, "box_depth": 15.0})
stand_base_pos = g.node("com.cad.translate", "Stand Base Pos",
    {"x": 30.0, "y": -15.0, "z": 0.0})  # Moved further down (far away)
g.connect(stand_base, stand_base_pos)

# Curved support arm (lofted from base up to aircraft)
# Profile at base
arm_base = g.node("com.cad.ellipse", "Arm Base", {"x_radius": 1.5, "y_radius": 0.8})
arm_base_pos = g.node("com.cad.translate", "Arm Base Pos", {"x": 0.0, "y": 0.0, "z": 0.0})
g.connect(arm_base, arm_base_pos)

# Profile at middle (swept back)
arm_mid = g.node("com.cad.ellipse", "Arm Mid", {"x_radius": 0.8, "y_radius": 0.5})
arm_mid_pos = g.node("com.cad.translate", "Arm Mid Pos", {"x": 0.0, "y": 0.0, "z": 6.0})
g.connect(arm_mid, arm_mid_pos)

# Profile at top (cradle shape)
arm_top = g.node("com.cad.ellipse", "Arm Top", {"x_radius": 2.0, "y_radius": 0.3})
arm_top_pos = g.node("com.cad.translate", "Arm Top Pos", {"x": 0.0, "y": 0.0, "z": 12.5})  # Taller arm to reach fuselage
g.connect(arm_top, arm_top_pos)

# Loft the arm
arm_loft = g.node("com.cad.loft", "Support Arm Loft", {"ruled": False})
g.connect(arm_base_pos, arm_loft, "shape", "profiles")
g.connect(arm_mid_pos, arm_loft, "shape", "profiles")
g.connect(arm_top_pos, arm_loft, "shape", "profiles")

# Rotate arm to be vertical (Z -> Y)
arm_rot = g.node("com.cad.rotate", "Support Arm Rot",
    {"angle": -90.0, "axis_x": 1.0, "axis_y": 0.0, "axis_z": 0.0})
g.connect(arm_loft, arm_rot)

# Position arm - starts inside base, goes up to fuselage
arm_pos = g.node("com.cad.translate", "Support Arm Pos",
    {"x": 30.0, "y": -15.0, "z": 0.0})  # Starts at base level
g.connect(arm_rot, arm_pos)

g.row(200)


# =============================================================================
# LEFT WING with WINGLET
# =============================================================================
lw_root = g.node("com.cad.ellipse", "LWing Root", {"x_radius": 7.0, "y_radius": 0.9})
lw_root_pos = g.node("com.cad.translate", "LWing Root Pos", 
    {"x": 22.0, "y": -0.5, "z": 3.25})
g.connect(lw_root, lw_root_pos)

lw_mid = g.node("com.cad.ellipse", "LWing Mid", {"x_radius": 4.0, "y_radius": 0.5})
lw_mid_pos = g.node("com.cad.translate", "LWing Mid Pos",
    {"x": 30.0, "y": 0.5, "z": 18.0})
g.connect(lw_mid, lw_mid_pos)

lw_tip = g.node("com.cad.ellipse", "LWing Tip", {"x_radius": 1.5, "y_radius": 0.2})
lw_tip_pos = g.node("com.cad.translate", "LWing Tip Pos",
    {"x": 38.0, "y": 2.0, "z": 32.0})
g.connect(lw_tip, lw_tip_pos)

left_wing = g.node("com.cad.loft", "Left Wing", {"ruled": True})
g.connect(lw_root_pos, left_wing, "shape", "profiles")
g.connect(lw_mid_pos, left_wing, "shape", "profiles")
g.connect(lw_tip_pos, left_wing, "shape", "profiles")

# Left Winglet
lwl_base = g.node("com.cad.ellipse", "LWinglet Base", {"x_radius": 1.5, "y_radius": 0.15})
lwl_base_pos = g.node("com.cad.translate", "LWinglet Base Pos",
    {"x": 38.0, "y": 2.0, "z": 32.0})
g.connect(lwl_base, lwl_base_pos)

lwl_tip = g.node("com.cad.ellipse", "LWinglet Tip", {"x_radius": 0.8, "y_radius": 0.08})
lwl_tip_pos = g.node("com.cad.translate", "LWinglet Tip Pos",
    {"x": 40.0, "y": 5.0, "z": 33.0})
g.connect(lwl_tip, lwl_tip_pos)

left_winglet = g.node("com.cad.loft", "Left Winglet", {"ruled": True})
g.connect(lwl_base_pos, left_winglet, "shape", "profiles")
g.connect(lwl_tip_pos, left_winglet, "shape", "profiles")

g.row()

# =============================================================================
# RIGHT WING with WINGLET
# =============================================================================
rw_root = g.node("com.cad.ellipse", "RWing Root", {"x_radius": 7.0, "y_radius": 0.9})
rw_root_pos = g.node("com.cad.translate", "RWing Root Pos",
    {"x": 22.0, "y": -0.5, "z": -3.25})
g.connect(rw_root, rw_root_pos)

rw_mid = g.node("com.cad.ellipse", "RWing Mid", {"x_radius": 4.0, "y_radius": 0.5})
rw_mid_pos = g.node("com.cad.translate", "RWing Mid Pos",
    {"x": 30.0, "y": 0.5, "z": -18.0})
g.connect(rw_mid, rw_mid_pos)

rw_tip = g.node("com.cad.ellipse", "RWing Tip", {"x_radius": 1.5, "y_radius": 0.2})
rw_tip_pos = g.node("com.cad.translate", "RWing Tip Pos",
    {"x": 38.0, "y": 2.0, "z": -32.0})
g.connect(rw_tip, rw_tip_pos)

right_wing = g.node("com.cad.loft", "Right Wing", {"ruled": True})
g.connect(rw_root_pos, right_wing, "shape", "profiles")
g.connect(rw_mid_pos, right_wing, "shape", "profiles")
g.connect(rw_tip_pos, right_wing, "shape", "profiles")

# Right Winglet
rwl_base = g.node("com.cad.ellipse", "RWinglet Base", {"x_radius": 1.5, "y_radius": 0.15})
rwl_base_pos = g.node("com.cad.translate", "RWinglet Base Pos",
    {"x": 38.0, "y": 2.0, "z": -32.0})
g.connect(rwl_base, rwl_base_pos)

rwl_tip = g.node("com.cad.ellipse", "RWinglet Tip", {"x_radius": 0.8, "y_radius": 0.08})
rwl_tip_pos = g.node("com.cad.translate", "RWinglet Tip Pos",
    {"x": 40.0, "y": 5.0, "z": -33.0})
g.connect(rwl_tip, rwl_tip_pos)

right_winglet = g.node("com.cad.loft", "Right Winglet", {"ruled": True})
g.connect(rwl_base_pos, right_winglet, "shape", "profiles")
g.connect(rwl_tip_pos, right_winglet, "shape", "profiles")

g.row(200)

# =============================================================================
# ENGINES - Detailed nacelles with pylons
# =============================================================================
def create_detailed_engine(name, eng_x, eng_y, eng_z, wing_y):
    engine_parts = []
    
    # 1. OUTER NACELLE (Lofted)
    inlet = g.node("com.cad.sketch.circle", f"{name} Inlet", {"radius": 1.5})
    inlet_pos = g.node("com.cad.translate", f"{name} Inlet Pos", {"x": 0.0, "y": 0.0, "z": -3.5})
    g.connect(inlet, inlet_pos)
    
    fan_sect = g.node("com.cad.sketch.circle", f"{name} FanSect", {"radius": 1.6})
    fan_pos = g.node("com.cad.translate", f"{name} FanSect Pos", {"x": 0.0, "y": 0.0, "z": -2.0})
    g.connect(fan_sect, fan_pos)
    
    exit_sect = g.node("com.cad.sketch.circle", f"{name} ExitSect", {"radius": 1.0})
    exit_pos = g.node("com.cad.translate", f"{name} ExitSect Pos", {"x": 0.0, "y": 0.0, "z": 3.5})
    g.connect(exit_sect, exit_pos)
    
    outer_loft = g.node("com.cad.loft", f"{name} Outer Loft", {"ruled": False})
    g.connect(inlet_pos, outer_loft, "shape", "profiles")
    g.connect(fan_pos, outer_loft, "shape", "profiles")
    g.connect(exit_pos, outer_loft, "shape", "profiles")
    
    # 2. INNER CUTOUT (Slightly smaller to create wall thickness)
    inner_inlet = g.node("com.cad.sketch.circle", f"{name} Inner Inlet", {"radius": 1.2})
    inner_inlet_pos = g.node("com.cad.translate", f"{name} Inner Inlet Pos", {"x": 0.0, "y": 0.0, "z": -3.6}) # Stick out slightly for clean cut
    g.connect(inner_inlet, inner_inlet_pos)
    
    inner_fan = g.node("com.cad.sketch.circle", f"{name} Inner Fan", {"radius": 1.3})
    inner_fan_pos = g.node("com.cad.translate", f"{name} Inner Fan Pos", {"x": 0.0, "y": 0.0, "z": -2.0})
    g.connect(inner_fan, inner_fan_pos)
    
    inner_exit = g.node("com.cad.sketch.circle", f"{name} Inner Exit", {"radius": 0.7})
    inner_exit_pos = g.node("com.cad.translate", f"{name} Inner Exit Pos", {"x": 0.0, "y": 0.0, "z": 3.6})
    g.connect(inner_exit, inner_exit_pos)
    
    inner_loft = g.node("com.cad.loft", f"{name} Inner Loft", {"ruled": False})
    g.connect(inner_inlet_pos, inner_loft, "shape", "profiles")
    g.connect(inner_fan_pos, inner_loft, "shape", "profiles")
    g.connect(inner_exit_pos, inner_loft, "shape", "profiles")
    
    # 3. HOLLOW NACELLE (Boolean Cut)
    hollow_nacelle = g.node("com.cad.boolean", f"{name} Hollow Nacelle", {"operation": "Cut"})
    g.connect(outer_loft, hollow_nacelle, "shape", "shape_a")
    g.connect(inner_loft, hollow_nacelle, "shape", "shape_b")
    
    # Rotate and Position Nacelle
    nacelle_rot = g.node("com.cad.rotate", f"{name} Nacelle Rot",
        {"angle": 90.0, "axis_x": 0.0, "axis_y": 1.0, "axis_z": 0.0})
    g.connect(hollow_nacelle, nacelle_rot)
    
    nacelle_final = g.node("com.cad.translate", f"{name} Nacelle Trans",
        {"x": eng_x, "y": eng_y, "z": eng_z})
    g.connect(nacelle_rot, nacelle_final)
    
    # 4. FAN BLADES (Realistic Airfoil Shape)
    # Create one master blade using lofted ellipses (airfoil-like)
    b_root = g.node("com.cad.ellipse", f"{name} BladeRoot", {"x_radius": 0.15, "y_radius": 0.02})
    b_root_rot = g.node("com.cad.rotate", f"{name} BladeRootRot", {"angle": 45.0, "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0})
    g.connect(b_root, b_root_rot)
    b_root_pos = g.node("com.cad.translate", f"{name} BladeRootPos", {"x": 0.4, "y": 0.0, "z": -2.2})
    g.connect(b_root_rot, b_root_pos)
    
    b_mid = g.node("com.cad.ellipse", f"{name} BladeMid", {"x_radius": 0.2, "y_radius": 0.02})
    b_mid_rot = g.node("com.cad.rotate", f"{name} BladeMidRot", {"angle": 30.0, "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0})
    g.connect(b_mid, b_mid_rot)
    b_mid_pos = g.node("com.cad.translate", f"{name} BladeMidPos", {"x": 0.9, "y": 0.0, "z": -2.2})
    g.connect(b_mid_rot, b_mid_pos)
    
    b_tip = g.node("com.cad.ellipse", f"{name} BladeTip", {"x_radius": 0.1, "y_radius": 0.01})
    b_tip_rot = g.node("com.cad.rotate", f"{name} BladeTipRot", {"angle": 15.0, "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0})
    g.connect(b_tip, b_tip_rot)
    b_tip_pos = g.node("com.cad.translate", f"{name} BladeTipPos", {"x": 1.35, "y": 0.0, "z": -2.2})
    g.connect(b_tip_rot, b_tip_pos)
    
    master_blade_loft = g.node("com.cad.loft", f"{name} MasterBlade", {"ruled": False})
    g.connect(b_root_pos, master_blade_loft, "shape", "profiles")
    g.connect(b_mid_pos, master_blade_loft, "shape", "profiles")
    g.connect(b_tip_pos, master_blade_loft, "shape", "profiles")
    
    # Replicate 12 blades
    for i in range(12):
        angle = i * 30.0
        # Rotate around Z axis (fan usage)
        blade_azimuth = g.node("com.cad.rotate", f"{name} Blade{i} Azi",
            {"angle": float(angle), "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0})
        g.connect(master_blade_loft, blade_azimuth)
        
        # Orient to engine (Rotate 90 deg Y)
        blade_eng_rot = g.node("com.cad.rotate", f"{name} Blade{i} EngRot",
            {"angle": 90.0, "axis_x": 0.0, "axis_y": 1.0, "axis_z": 0.0})
        g.connect(blade_azimuth, blade_eng_rot)
        
        # Position in engine
        blade_final = g.node("com.cad.translate", f"{name} Blade{i} Final",
            {"x": eng_x, "y": eng_y, "z": eng_z})
        g.connect(blade_eng_rot, blade_final)
        
        engine_parts.append(blade_final)
        
    # Spinner
    spinner = g.node("com.cad.sketch.circle", f"{name} SpinnerBase", {"radius": 0.4})
    spinner_ext = g.node("com.cad.loft", f"{name} SpinnerLoft", {"ruled": True}) # Need profile for cone
    # For simplicity, small cone using loft
    sp_tip = g.node("com.cad.sketch.circle", f"{name} SpinnerTip", {"radius": 0.05})
    sp_tip_pos = g.node("com.cad.translate", f"{name} SpinnerTipPos", {"x": 0.0, "y": 0.0, "z": -2.6})
    g.connect(sp_tip, sp_tip_pos)
    
    sp_base_pos = g.node("com.cad.translate", f"{name} SpinnerBasePos", {"x": 0.0, "y": 0.0, "z": -2.2})
    g.connect(spinner, sp_base_pos)
    
    spinner_cone = g.node("com.cad.loft", f"{name} SpinnerCone", {"ruled": True})
    g.connect(sp_base_pos, spinner_cone, "shape", "profiles")
    g.connect(sp_tip_pos, spinner_cone, "shape", "profiles")
    
    sp_rot = g.node("com.cad.rotate", f"{name} SpinnerRot", {"angle": 90.0, "axis_x": 0.0, "axis_y": 1.0, "axis_z": 0.0})
    g.connect(spinner_cone, sp_rot)
    sp_final = g.node("com.cad.translate", f"{name} SpinnerFinal", {"x": eng_x, "y": eng_y, "z": eng_z})
    g.connect(sp_rot, sp_final)
    engine_parts.append(sp_final)
    
    engine_parts.append(nacelle_final)
    
    # Pylon
    pylon_height = abs(wing_y - eng_y) - 1.0
    pylon = g.node("com.cad.box", f"{name} Pylon", 
        {"box_length": 4.0, "box_width": pylon_height, "box_depth": 0.3})
    pylon_y = eng_y + 1.0 + pylon_height/2
    pylon_pos = g.node("com.cad.translate", f"{name} Pylon Pos",
        {"x": eng_x + 1.0, "y": pylon_y, "z": eng_z})
    g.connect(pylon, pylon_pos)
    engine_parts.append(pylon_pos)
    
    return engine_parts


eng_L1_parts = create_detailed_engine("Eng L1", 25.0, -3.5, 11.0, -0.3)
eng_L2_parts = create_detailed_engine("Eng L2", 32.0, -2.5, 22.0, 0.3)
eng_R1_parts = create_detailed_engine("Eng R1", 25.0, -3.5, -11.0, -0.3)
eng_R2_parts = create_detailed_engine("Eng R2", 32.0, -2.5, -22.0, 0.3)

g.row(200)

# =============================================================================
# VERTICAL STABILIZER
# =============================================================================
vt_root = g.node("com.cad.ellipse", "VTail Root", {"x_radius": 6.0, "y_radius": 0.6})
vt_root_pos = g.node("com.cad.translate", "VTail Root Pos", {"x": 0.0, "y": 0.0, "z": 0.0})
g.connect(vt_root, vt_root_pos)

vt_mid = g.node("com.cad.ellipse", "VTail Mid", {"x_radius": 4.0, "y_radius": 0.4})
vt_mid_pos = g.node("com.cad.translate", "VTail Mid Pos", {"x": 5.0, "y": 0.0, "z": 9.0})
g.connect(vt_mid, vt_mid_pos)

vt_tip = g.node("com.cad.ellipse", "VTail Tip", {"x_radius": 2.0, "y_radius": 0.2})
vt_tip_pos = g.node("com.cad.translate", "VTail Tip Pos", {"x": 9.0, "y": 0.0, "z": 16.0})
g.connect(vt_tip, vt_tip_pos)

vtail_loft = g.node("com.cad.loft", "VTail Loft", {"ruled": True})
g.connect(vt_root_pos, vtail_loft, "shape", "profiles")
g.connect(vt_mid_pos, vtail_loft, "shape", "profiles")
g.connect(vt_tip_pos, vtail_loft, "shape", "profiles")

vtail_rot = g.node("com.cad.rotate", "VTail Rotate",
    {"angle": -90.0, "axis_x": 1.0, "axis_y": 0.0, "axis_z": 0.0})
g.connect(vtail_loft, vtail_rot)

vtail = g.node("com.cad.translate", "Vertical Stabilizer",
    {"x": 58.0, "y": 1.0, "z": 0.0})
g.connect(vtail_rot, vtail)

g.row()

# =============================================================================
# HORIZONTAL STABILIZERS
# =============================================================================
# Left H-Tail
ht_L_root = g.node("com.cad.ellipse", "HTail L Root", {"x_radius": 3.5, "y_radius": 0.35})
ht_L_root_pos = g.node("com.cad.translate", "HTail L Root Pos",
    {"x": 60.0, "y": 0.8, "z": 1.5})
g.connect(ht_L_root, ht_L_root_pos)

ht_L_tip = g.node("com.cad.ellipse", "HTail L Tip", {"x_radius": 1.5, "y_radius": 0.15})
ht_L_tip_pos = g.node("com.cad.translate", "HTail L Tip Pos",
    {"x": 67.0, "y": 1.0, "z": 11.0})
g.connect(ht_L_tip, ht_L_tip_pos)

htail_left = g.node("com.cad.loft", "Left H-Stab", {"ruled": True})
g.connect(ht_L_root_pos, htail_left, "shape", "profiles")
g.connect(ht_L_tip_pos, htail_left, "shape", "profiles")

# Right H-Tail
ht_R_root = g.node("com.cad.ellipse", "HTail R Root", {"x_radius": 3.5, "y_radius": 0.35})
ht_R_root_pos = g.node("com.cad.translate", "HTail R Root Pos",
    {"x": 60.0, "y": 0.8, "z": -1.5})
g.connect(ht_R_root, ht_R_root_pos)

ht_R_tip = g.node("com.cad.ellipse", "HTail R Tip", {"x_radius": 1.5, "y_radius": 0.15})
ht_R_tip_pos = g.node("com.cad.translate", "HTail R Tip Pos",
    {"x": 67.0, "y": 1.0, "z": -11.0})
g.connect(ht_R_tip, ht_R_tip_pos)

htail_right = g.node("com.cad.loft", "Right H-Stab", {"ruled": True})
g.connect(ht_R_root_pos, htail_right, "shape", "profiles")
g.connect(ht_R_tip_pos, htail_right, "shape", "profiles")

g.row(200)

# =============================================================================
# ASSEMBLY
# =============================================================================
# Wings + Winglets
asm_wings = g.node("com.cad.assembly.assembly", "Wings + Winglets", {"assembly_name": "Wings"})
g.connect(left_wing, asm_wings, "shape", "part_1")
g.connect(right_wing, asm_wings, "shape", "part_2")
g.connect(left_winglet, asm_wings, "shape", "part_3")
g.connect(right_winglet, asm_wings, "shape", "part_4")

# Body + Wings
asm_body = g.node("com.cad.assembly.assembly", "Body + Wings", {"assembly_name": "Body"})
g.connect(fuselage, asm_body, "shape", "part_1")
g.connect(asm_wings, asm_body, "assembly", "part_2")

g.row()

# Engines (Updated for blades)
# Each engine has: Nacelle + Pylon + Spinner + 12 Blades = 15 parts
# We need to chain assemblies

def create_engine_assembly(name, parts):
    # parts: [0..11]=blades, [12]=spinner, [13]=nacelle, [14]=pylon
    if len(parts) < 15:
        # Fallback if something missing
        asm = g.node("com.cad.assembly.assembly", f"{name} Basic", {"assembly_name": f"{name}Basic"})
        for i, p in enumerate(parts):
            if i < 4: g.connect(p, asm, "shape", f"part_{i+1}")
        return asm

    # Blade assembly (3 groups of 4)
    asm_blades1 = g.node("com.cad.assembly.assembly", f"{name} Blades 1", {"assembly_name": f"{name}B1"})
    for i in range(4): g.connect(parts[i], asm_blades1, "shape", f"part_{i+1}")
    
    asm_blades2 = g.node("com.cad.assembly.assembly", f"{name} Blades 2", {"assembly_name": f"{name}B2"})
    for i in range(4): g.connect(parts[i+4], asm_blades2, "shape", f"part_{i+1}")
    
    asm_blades3 = g.node("com.cad.assembly.assembly", f"{name} Blades 3", {"assembly_name": f"{name}B3"})
    for i in range(4): g.connect(parts[i+8], asm_blades3, "shape", f"part_{i+1}")
    
    # Combined blades
    asm_blades_all = g.node("com.cad.assembly.assembly", f"{name} All Blades", {"assembly_name": f"{name}Blades"})
    g.connect(asm_blades1, asm_blades_all, "assembly", "part_1")
    g.connect(asm_blades2, asm_blades_all, "assembly", "part_2")
    g.connect(asm_blades3, asm_blades_all, "assembly", "part_3")
    
    # Core engine (Nacelle + Pylon + Spinner)
    asm_core = g.node("com.cad.assembly.assembly", f"{name} Core", {"assembly_name": f"{name}Core"})
    g.connect(parts[13], asm_core, "shape", "part_1") # Nacelle
    g.connect(parts[14], asm_core, "shape", "part_2") # Pylon
    g.connect(parts[12], asm_core, "shape", "part_3") # Spinner
    
    # Full Engine
    asm_full = g.node("com.cad.assembly.assembly", f"{name} Full", {"assembly_name": name})
    g.connect(asm_core, asm_full, "assembly", "part_1")
    g.connect(asm_blades_all, asm_full, "assembly", "part_2")
    
    return asm_full

asm_eng1 = create_engine_assembly("EngL1", eng_L1_parts)
asm_eng2 = create_engine_assembly("EngL2", eng_L2_parts)
asm_eng3 = create_engine_assembly("EngR1", eng_R1_parts)
asm_eng4 = create_engine_assembly("EngR2", eng_R2_parts)

asm_engines = g.node("com.cad.assembly.assembly", "All Engines", {"assembly_name": "Engines"})
g.connect(asm_eng1, asm_engines, "assembly", "part_1")
g.connect(asm_eng2, asm_engines, "assembly", "part_2")
g.connect(asm_eng3, asm_engines, "assembly", "part_3")
g.connect(asm_eng4, asm_engines, "assembly", "part_4")

g.row()

# Tail
asm_tail = g.node("com.cad.assembly.assembly", "Tail Section", {"assembly_name": "Tail"})
g.connect(vtail, asm_tail, "shape", "part_1")
g.connect(htail_left, asm_tail, "shape", "part_2")
g.connect(htail_right, asm_tail, "shape", "part_3")

g.row()

# Display Stand (base + curved arm)
asm_stand = g.node("com.cad.assembly.assembly", "Display Stand", {"assembly_name": "Stand"})
g.connect(stand_base_pos, asm_stand, "shape", "part_1")
g.connect(arm_pos, asm_stand, "shape", "part_2")

g.row()

# Final Assembly (Body + Wings + Engines + Tail + Stand)
final = g.node("com.cad.assembly.assembly", "Boeing 747-400", {"assembly_name": "Boeing747"})
g.connect(asm_body, final, "assembly", "part_1")
g.connect(asm_engines, final, "assembly", "part_2")
g.connect(asm_tail, final, "assembly", "part_3")
g.connect(asm_stand, final, "assembly", "part_4")

g.save(r"d:\kutAI\data\Boeing_747.cad")

print("Boeing 747-400 with Display Stand - Generated successfully!")
