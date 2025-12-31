# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode

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
        shape = self.get_input_shape('shape')
        distance = self.get_input_value('extrude_distance', 'extrude_distance')
        
        if shape is None:
            return None
        
        # Extrude the shape
        try:
            return shape.extrude(float(distance))
        except Exception as e:
            print(f"Extrude error: {e}")
            return shape


class PocketNode(CadQueryNode):
    """Cuts a hole (pocket) in a shape."""
    __identifier__ = 'com.cad.pocket'
    NODE_NAME = 'Pocket'

    def __init__(self):
        super(PocketNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('pocket_depth', color=(180, 180, 0))
        self.add_input('pocket_radius', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))
        
        self.create_property('pocket_depth', 5.0, widget_type='float')
        self.create_property('pocket_radius', 2.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        depth = self.get_input_value('pocket_depth', 'pocket_depth')
        radius = self.get_input_value('pocket_radius', 'pocket_radius')
        
        if shape is None:
            return None
        
        try:
            # Cut a cylindrical hole
            return shape.faces(">Z").workplane().circle(float(radius)).cutBlind(float(depth))
        except Exception as e:
            print(f"Pocket error: {e}")
            return shape


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
        shape = self.get_input_shape('shape')
        radius = self.get_input_value('fillet_radius', 'fillet_radius')
        
        if shape is None:
            return None
        
        try:
            return shape.edges().fillet(float(radius))
        except Exception as e:
            print(f"Fillet error: {e}")
            return shape


class SelectFaceNode(CadQueryNode):
    """Select a face robustly based on geometric properties."""
    __identifier__ = 'com.cad.select_face'
    NODE_NAME = 'Select Face (Robust)'

    def __init__(self):
        super(SelectFaceNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_output('workplane', color=(100, 200, 255))
        self.create_property('selector_type', 'Direction', items=['Direction', 'NearestToPoint', 'Index', 'Largest Area', 'Tag'], widget_type='combo')
        # For Direction: >Z, <X etc.
        self.create_property('direction', '>Z', widget_type='string')
        # For Point: Select face closest to this point
        self.create_property('near_x', 0.0, widget_type='float')
        self.create_property('near_y', 0.0, widget_type='float')
        self.create_property('near_z', 0.0, widget_type='float')
        # For Index: Select explicit face index
        self.create_property('face_index', 0, widget_type='int')
        # For Tag: Select face by tag name
        self.create_property('tag', 'top', widget_type='string')

    def run(self):
        shape = self.get_input_shape('shape')
        if not shape:
            return None

        method = self.get_property('selector_type')

        try:
            if method == 'Direction':
                selector = self.get_property('direction')
                print(f"[SelectFace DEBUG] Method: Direction, Selector: {selector}")
                try:
                    face_selection = shape.faces(selector)
                except Exception:
                    # Fallback to .workplane().faces() if direct access fails
                    print("[SelectFace DEBUG] Direct faces selection failed, trying .workplane().faces()")
                    face_selection = shape.workplane().faces(selector)

                faces = face_selection.vals()
                print(f"[SelectFace DEBUG] Found {len(faces)} faces")
                if not faces:
                    self.set_error("No faces found with selector")
                    return None
                # Return dict with both workplane and face geometry
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
                    self.set_error(f"Face index {idx} out of range (0-{len(all_faces)-1})")
                    return None

            elif method == 'Largest Area':
                all_faces = shape.faces().vals()
                if not all_faces:
                    self.set_error("No faces found in shape")
                    return None
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
                    self.set_error(f"No faces found with tag '{tag_name}'")
                    return None
                return {'workplane': face_selection.workplane(), 'face': faces[0]}

        except Exception as e:
            self.set_error(f"Face selection failed: {e}")
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
        target = self.get_input_shape('target')
        profile = self.get_input_shape('profile')
        dist = self.get_property('distance')
        thru = self.get_property('through_all')

        if not target or not profile:
            return None

        # Approach: Extrude the profile into a tool, then boolean cut
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
            print(f"Cut Error: {e}")
            return target


class BooleanNode(CadQueryNode):
    """Union, Difference, or Intersection of two solids."""
    __identifier__ = 'com.cad.boolean'
    NODE_NAME = 'Boolean Op'

    def __init__(self):
        super(BooleanNode, self).__init__()
        self.add_input('shape_a', color=(100, 255, 100))
        self.add_input('shape_b', color=(100, 255, 100))
        self.add_output('result', color=(100, 255, 100))
        
        self.create_property('operation', 'Union', widget_type='string') # Union, Cut, Intersect

    def run(self):
        a = self.get_input_shape('shape_a')
        b = self.get_input_shape('shape_b')
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
            print(f"Boolean Error: {e}")
        return None


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
        profile = self.get_input_shape('profile')
        angle = self.get_input_value('angle', 'angle')
        
        if not profile: return None
        
        # Get axis parameters
        sx = self.get_input_value('axis_start_x', 'axis_start_x')
        sy = self.get_input_value('axis_start_y', 'axis_start_y')
        sz = self.get_input_value('axis_start_z', 'axis_start_z')
        dx = self.get_input_value('axis_dir_x', 'axis_dir_x')
        dy = self.get_input_value('axis_dir_y', 'axis_dir_y')
        dz = self.get_input_value('axis_dir_z', 'axis_dir_z')

        try:
            axis_start = (float(sx), float(sy), float(sz))
            axis_dir = (float(dx), float(dy), float(dz))
            return profile.revolve(float(angle), axis_start, axis_dir)
        except Exception as e:
            print(f"Revolve error: {e}")
            return None


class CylinderCutNode(CadQueryNode):
    """Create a cylinder tool and cut it from a target solid."""
    __identifier__ = 'com.cad.cylinder_cut'
    NODE_NAME = 'Cylinder Cut'

    def __init__(self):
        super(CylinderCutNode, self).__init__()
        self.add_input('target', color=(100, 255, 100))
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
        import cadquery as cq
        target = self.get_input_shape('target')
        if target is None:
            return None

        r = self.get_input_value('cyl_radius', 'cyl_radius')
        h = self.get_input_value('cyl_height', 'cyl_height')
        x = self.get_input_value('center_x', 'center_x')
        y = self.get_input_value('center_y', 'center_y')
        z = self.get_input_value('center_z', 'center_z')

        try:
            r, h = float(r), float(h)
            x, y, z = float(x), float(y), float(z)
            
            # Create a cylinder tool at the specified position
            tool = cq.Workplane("XY").cylinder(h, r).translate((x, y, z + h/2))
            
            # Cut from target
            return target.cut(tool)
        except Exception as e:
            print(f"CylinderCut error: {e}")
            return target