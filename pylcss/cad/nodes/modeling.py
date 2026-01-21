# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Modeling Nodes - 3D Operations and Transformations.
"""

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input, resolve_shape_input
import math

# ==========================================
# 3D CREATION & BOOLS
# ==========================================

class ExtrudeNode(CadQueryNode):
    """Extrudes a 2D shape or sketch."""
    __identifier__ = 'com.cad.extrude'
    NODE_NAME = 'Extrude'

    def __init__(self):
        super(ExtrudeNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('extrude_distance', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('extrude_distance', 5.0, widget_type='float')

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        distance = resolve_numeric_input(self.get_input('extrude_distance'), self.get_property('extrude_distance'))
        
        if shape is None:
            return None
        
        try:
            return shape.extrude(float(distance))
        except Exception as e:
            self.set_error(f"Extrude error: {e}")
            return shape


class TwistedExtrudeNode(CadQueryNode):
    """Twist extrudes a 2D shape."""
    __identifier__ = 'com.cad.twisted_extrude'
    NODE_NAME = 'Twisted Extrude'

    def __init__(self):
        super(TwistedExtrudeNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('distance', color=(180, 180, 0))
        self.add_input('angle', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('distance', 10.0, widget_type='float')
        self.create_property('angle', 45.0, widget_type='float')

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        distance = resolve_numeric_input(self.get_input('distance'), self.get_property('distance'))
        angle = resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        
        if shape is None:
            return None
        
        try:
            # IMPORTANT: CadQuery operations like twistExtrude consume pendingWires.
            # If we operate on the input 'shape' directly, we mutate the upstream node's result!
            # Next time we run (e.g. changing angle), 'shape' has no wires left.
            # FIX: Create a fresh Workplane with the same context and stack/wires
            
            # Helper to clone the workplane state safely
            if hasattr(shape, 'newObject'):
                # newObject creates a new current workplane with the same stack
                # but we actually need to ensure pendingWires are preserved/copied
                # Best way is to rely on CQ's immutability behavior if handled right,
                # but twistExtrude might be destructive to the context.
                
                # Copying the context is safest for these context-sensitive ops
                import copy
                safe_shape = copy.copy(shape)
                # Deep copy of context might be needed if pendingWires is a list ref
                if hasattr(shape, 'ctx'):
                    safe_shape.ctx = copy.copy(shape.ctx)
                    if hasattr(shape.ctx, 'pendingWires'):
                        safe_shape.ctx.pendingWires = list(shape.ctx.pendingWires)
            else:
                safe_shape = shape

            res = safe_shape.twistExtrude(float(distance), float(angle))
            return res
        except Exception as e:
            self.set_error(f"Twist Extrude error: {e}")
            return shape


class RevolveNode(CadQueryNode):
    """Revolve a sketch around an axis."""
    __identifier__ = 'com.cad.revolve'
    NODE_NAME = 'Revolve'

    def __init__(self):
        super(RevolveNode, self).__init__()
        self.add_input('profile', color=(100, 200, 255))
        self.add_input('angle', color=(180, 180, 0))
        # Axis definition: start point and direction
        self.add_input('axis_start_x', color=(180, 180, 0))
        self.add_input('axis_start_y', color=(180, 180, 0))
        self.add_input('axis_start_z', color=(180, 180, 0))
        self.add_input('axis_dir_x', color=(180, 180, 0))
        self.add_input('axis_dir_y', color=(180, 180, 0))
        self.add_input('axis_dir_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('angle', 360.0, widget_type='float')
        self.create_property('axis_start_x', 0.0, widget_type='float')
        self.create_property('axis_start_y', 0.0, widget_type='float')
        self.create_property('axis_start_z', 0.0, widget_type='float')
        self.create_property('axis_dir_x', 0.0, widget_type='float')
        self.create_property('axis_dir_y', 1.0, widget_type='float')
        self.create_property('axis_dir_z', 0.0, widget_type='float')

    def run(self):
        profile = resolve_shape_input(self.get_input('profile'))
        angle = resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        
        if not profile: return None
        
        sx = resolve_numeric_input(self.get_input('axis_start_x'), self.get_property('axis_start_x'))
        sy = resolve_numeric_input(self.get_input('axis_start_y'), self.get_property('axis_start_y'))
        sz = resolve_numeric_input(self.get_input('axis_start_z'), self.get_property('axis_start_z'))
        dx = resolve_numeric_input(self.get_input('axis_dir_x'), self.get_property('axis_dir_x'))
        dy = resolve_numeric_input(self.get_input('axis_dir_y'), self.get_property('axis_dir_y'))
        dz = resolve_numeric_input(self.get_input('axis_dir_z'), self.get_property('axis_dir_z'))

        try:
            axis_start = (float(sx), float(sy), float(sz))
            axis_dir = (float(dx), float(dy), float(dz))
            return profile.revolve(float(angle), axis_start, axis_dir)
        except Exception as e:
            self.set_error(f"Revolve error: {e}")
            return None


class CutExtrudeNode(CadQueryNode):
    """Cut a shape using a 2D profile (Extruded Cut)."""
    __identifier__ = 'com.cad.cut_extrude'
    NODE_NAME = 'Extruded Cut'

    def __init__(self):
        super(CutExtrudeNode, self).__init__()
        self.add_input('target', color=(100, 255, 100))
        self.add_input('profile', color=(100, 200, 255))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('distance', 10.0, widget_type='float')
        self.create_property('through_all', False, widget_type='bool')

    def run(self):
        target = resolve_shape_input(self.get_input('target'))
        profile = resolve_shape_input(self.get_input('profile'))
        dist = self.get_property('distance')

        if not target or not profile:
            return None

        try:
            # 1. Create the cutting tool
            if hasattr(profile, 'objects') and len(getattr(profile, 'objects', [])) == 0:
                return target

            if hasattr(profile, 'extrude'):
                tool = profile.extrude(float(dist))
            else:
                if hasattr(profile, 'val') and hasattr(profile.val(), 'extrude'):
                    tool = profile.val().extrude(float(dist))
                else:
                    raise RuntimeError('Profile cannot be extruded')

            # 2. Boolean Difference
            return target.cut(tool)
        except Exception as e:
            self.set_error(f"Cut Error: {e}")
            return target


class BooleanNode(CadQueryNode):
    """Union, Difference, or Intersection of two solids."""
    __identifier__ = 'com.cad.boolean'
    NODE_NAME = 'Boolean Op'

    def __init__(self):
        super(BooleanNode, self).__init__()
        self.add_input('shape_a', color=(100, 255, 100))
        self.add_input('shape_b', color=(100, 255, 100))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('operation', 'Union', widget_type='combo',
                             items=['Union', 'Cut', 'Intersect'])

    def run(self):
        a = resolve_shape_input(self.get_input('shape_a'))
        b = resolve_shape_input(self.get_input('shape_b'))
        op = self.get_property('operation').lower()
        
        if not a or not b: return a or b

        try:
            if 'union' in op:
                return a.union(b)
            elif 'cut' in op:
                return a.cut(b)
            elif 'intersect' in op:
                return a.intersect(b)
        except Exception as e:
            self.set_error(f"Boolean Error: {e}")
        return None


# ==========================================
# MODIFICATIONS
# ==========================================

class FilletNode(CadQueryNode):
    """Rounds edges (fillet)."""
    __identifier__ = 'com.cad.fillet'
    NODE_NAME = 'Fillet'

    def __init__(self):
        super(FilletNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('fillet_radius', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('fillet_radius', 1.0, widget_type='float')

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        radius = resolve_numeric_input(self.get_input('fillet_radius'), self.get_property('fillet_radius'))
        
        if shape is None:
            return None
        
        try:
            return shape.edges().fillet(float(radius))
        except Exception as e:
            self.set_error(f"Fillet error: {e}")
            return shape


class ChamferNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        dist = resolve_numeric_input(self.get_input('distance'), self.get_property('distance'))
        selector = self.get_property('selector')
        
        try:
            if selector:
                return shape.edges(selector).chamfer(float(dist))
            else:
                return shape.edges().chamfer(float(dist))
        except Exception as e:
            self.set_error(f"Chamfer error: {e}")
            return shape


class ShellNode(CadQueryNode):
    """Hollows out a solid."""
    __identifier__ = 'com.cad.shell'
    NODE_NAME = 'Shell'

    def __init__(self):
        super(ShellNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        # Optional: Face to remove (opening)
        self.add_input('face_to_remove', color=(100, 200, 255)) 
        self.add_output('shape', color=(100, 255, 100))
        self.create_property('thickness', -2.0, widget_type='float') # Negative for inward shell

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        face_obj = resolve_shape_input(self.get_input('face_to_remove'))
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


class OffsetNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        dist = resolve_numeric_input(self.get_input('distance'), self.get_property('distance'))
        
        try:
            return shape.offset2D(float(dist))
        except Exception as e:
            self.set_error(f"Offset error: {e}")
            return shape


# ==========================================
# TRANSFORMATIONS
# ==========================================

class TranslateNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        x = resolve_numeric_input(self.get_input('x'), self.get_property('x'))
        y = resolve_numeric_input(self.get_input('y'), self.get_property('y'))
        z = resolve_numeric_input(self.get_input('z'), self.get_property('z'))
        
        try:
            return shape.translate((float(x), float(y), float(z)))
        except Exception as e:
            self.set_error(f"Translate error: {e}")
            return shape


class RotateNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        angle = resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        ax = float(self.get_property('axis_x'))
        ay = float(self.get_property('axis_y'))
        az = float(self.get_property('axis_z'))
        cx = float(self.get_property('center_x'))
        cy = float(self.get_property('center_y'))
        cz = float(self.get_property('center_z'))
        
        try:
            axis_start = (cx, cy, cz)
            axis_end = (cx + ax, cy + ay, cz + az)
            return shape.rotate(axis_start, axis_end, float(angle))
        except Exception as e:
            self.set_error(f"Rotate error: {e}")
            return shape


class ScaleNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
        if shape is None:
            return None
        
        factor = resolve_numeric_input(self.get_input('factor'), self.get_property('factor'))
        uniform = self.get_property('uniform')
        
        try:
            if hasattr(shape, 'val'):
                solid = shape.val()
            else:
                solid = shape
            
            from OCP.gp import gp_GTrsf, gp_Mat
            from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
            
            gtrsf = gp_GTrsf()
            
            if uniform:
                sf = float(factor)
                mat = gp_Mat(sf, 0, 0, 0, sf, 0, 0, 0, sf)
            else:
                sx = float(self.get_property('x_factor'))
                sy = float(self.get_property('y_factor'))
                sz = float(self.get_property('z_factor'))
                mat = gp_Mat(sx, 0, 0, 0, sy, 0, 0, 0, sz)
            
            gtrsf.SetVectorialPart(mat)
            
            if hasattr(solid, 'wrapped'):
                transformer = BRepBuilderAPI_GTransform(solid.wrapped, gtrsf, True)
                if transformer.IsDone():
                    from cadquery import Shape
                    scaled_shape = Shape(transformer.Shape())
                    return cq.Workplane("XY").add(scaled_shape)
            
            return shape
        except Exception as e:
            self.set_error(f"Scale error: {e}")
            return shape


class MirrorNode(CadQueryNode):
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
        shape = resolve_shape_input(self.get_input('shape'))
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
            self.set_error(f"Mirror error: {e}")
            return shape


class SelectFaceNode(CadQueryNode):
    """Select a face robustly based on geometric properties."""
    __identifier__ = 'com.cad.select_face'
    NODE_NAME = 'Select Face'

    def __init__(self):
        super(SelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))
        self.create_property('selector_type', 'Direction', items=['Direction', 'NearestToPoint', 'Index', 'Largest Area', 'Tag'], widget_type='combo')
        self.create_property('direction', '>Z', widget_type='string')
        self.create_property('near_x', 0.0, widget_type='float')
        self.create_property('near_y', 0.0, widget_type='float')
        self.create_property('near_z', 0.0, widget_type='float')
        self.create_property('face_index', 0, widget_type='int')
        self.create_property('tag', 'top', widget_type='string')

    def run(self):
        shape = resolve_shape_input(self.get_input('shape'))
        if not shape:
            return None

        method = self.get_property('selector_type')

        try:
            if method == 'Direction':
                selector = self.get_property('direction')
                try:
                    face_selection = shape.faces(selector)
                except Exception:
                    face_selection = shape.workplane().faces(selector)
                faces = face_selection.vals()
                if not faces:
                    self.set_error("No faces found with selector")
                    return None
                return {'workplane': face_selection.workplane(), 'face': faces[0]}

            elif method == 'NearestToPoint':
                pt = (self.get_property('near_x'), self.get_property('near_y'), self.get_property('near_z'))
                face_selection = shape.faces(cq.NearestToPointSelector(pt))
                faces = face_selection.vals()
                if not faces:
                    self.set_error("No faces found near point")
                    return None
                return {'workplane': face_selection.workplane(), 'face': faces[0]}

            elif method == 'Index':
                idx = int(self.get_property('face_index'))
                all_faces = shape.faces().vals()
                if 0 <= idx < len(all_faces):
                    face = all_faces[idx]
                    wp = shape.newObject([face]).workplane()
                    return {'workplane': wp, 'face': face}
                else:
                    self.set_error(f"Face index {idx} out of range")
                    return None

            elif method == 'Largest Area':
                all_faces = shape.faces().vals()
                if not all_faces: return None
                sorted_faces = sorted(all_faces, key=lambda f: f.Area(), reverse=True)
                largest_face = sorted_faces[0]
                wp = shape.newObject([largest_face]).workplane()
                return {'workplane': wp, 'face': largest_face}

            elif method == 'Tag':
                tag_name = self.get_property('tag')
                try:
                    face_selection = shape.faces(tag=tag_name)
                except Exception:
                    face_selection = shape.workplane().faces(tag=tag_name)
                faces = face_selection.vals()
                if not faces:
                    return None
                return {'workplane': face_selection.workplane(), 'face': faces[0]}

        except Exception as e:
            self.set_error(f"Face selection failed: {e}")
            return None
