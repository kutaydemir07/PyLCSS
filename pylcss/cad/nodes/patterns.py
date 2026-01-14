# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Advanced patterning nodes for parametric CAD design."""
import cadquery as cq
from pylcss.cad.core.base_node import CadQueryNode, resolve_numeric_input

class RadialPatternNode(CadQueryNode):
    """Creates a radial pattern of shapes around an axis."""
    __identifier__ = 'com.cad.pattern.radial'
    NODE_NAME = 'Radial Pattern'

    def __init__(self):
        super(RadialPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('count', color=(180, 180, 0))
        self.add_input('angle', color=(180, 180, 0))
        self.add_input('center_x', color=(180, 180, 0))
        self.add_input('center_y', color=(180, 180, 0))
        self.add_input('center_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('count', 6, widget_type='int')
        self.create_property('angle', 360.0, widget_type='float')
        self.create_property('center_x', 0.0, widget_type='float')
        self.create_property('center_y', 0.0, widget_type='float')
        self.create_property('center_z', 0.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None:
            return None

        count = resolve_numeric_input(self.get_input('count'), self.get_property('count'))
        angle = resolve_numeric_input(self.get_input('angle'), self.get_property('angle'))
        cx = resolve_numeric_input(self.get_input('center_x'), self.get_property('center_x'))
        cy = resolve_numeric_input(self.get_input('center_y'), self.get_property('center_y'))
        cz = resolve_numeric_input(self.get_input('center_z'), self.get_property('center_z'))

        try:
            count = int(count)
            angle = float(angle)
            center = (float(cx), float(cy), float(cz))

            # Create radial pattern
            result = shape
            for i in range(1, count):
                rotation_angle = (angle / (count - 1)) * i
                rotated = shape.rotate(center, (0, 0, 1), rotation_angle)
                result = result.union(rotated)

            return result
        except Exception as e:
            self.set_error(f"Radial pattern failed: {e}")
            return None


class MirrorPatternNode(CadQueryNode):
    """Creates mirror patterns across planes."""
    __identifier__ = 'com.cad.pattern.mirror'
    NODE_NAME = 'Mirror Pattern'

    def __init__(self):
        super(MirrorPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('plane_normal_x', color=(180, 180, 0))
        self.add_input('plane_normal_y', color=(180, 180, 0))
        self.add_input('plane_normal_z', color=(180, 180, 0))
        self.add_input('plane_point_x', color=(180, 180, 0))
        self.add_input('plane_point_y', color=(180, 180, 0))
        self.add_input('plane_point_z', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('plane_normal_x', 1.0, widget_type='float')
        self.create_property('plane_normal_y', 0.0, widget_type='float')
        self.create_property('plane_normal_z', 0.0, widget_type='float')
        self.create_property('plane_point_x', 0.0, widget_type='float')
        self.create_property('plane_point_y', 0.0, widget_type='float')
        self.create_property('plane_point_z', 0.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None:
            return None

        nx = resolve_numeric_input(self.get_input('plane_normal_x'), self.get_property('plane_normal_x'))
        ny = resolve_numeric_input(self.get_input('plane_normal_y'), self.get_property('plane_normal_y'))
        nz = resolve_numeric_input(self.get_input('plane_normal_z'), self.get_property('plane_normal_z'))
        px = resolve_numeric_input(self.get_input('plane_point_x'), self.get_property('plane_point_x'))
        py = resolve_numeric_input(self.get_input('plane_point_y'), self.get_property('plane_point_y'))
        pz = resolve_numeric_input(self.get_input('plane_point_z'), self.get_property('plane_point_z'))

        try:
            normal = (float(nx), float(ny), float(nz))
            point = (float(px), float(py), float(pz))

            # Create mirror pattern
            mirrored = shape.mirror(normal, point)
            return shape.union(mirrored)
        except Exception as e:
            self.set_error(f"Mirror pattern failed: {e}")
            return None


class GridPatternNode(CadQueryNode):
    """Creates a 2D grid pattern of shapes."""
    __identifier__ = 'com.cad.pattern.grid'
    NODE_NAME = 'Grid Pattern'

    def __init__(self):
        super(GridPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('rows', color=(180, 180, 0))
        self.add_input('columns', color=(180, 180, 0))
        self.add_input('row_spacing', color=(180, 180, 0))
        self.add_input('col_spacing', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('rows', 3, widget_type='int')
        self.create_property('columns', 3, widget_type='int')
        self.create_property('row_spacing', 10.0, widget_type='float')
        self.create_property('col_spacing', 10.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None:
            return None

        rows = resolve_numeric_input(self.get_input('rows'), self.get_property('rows'))
        cols = resolve_numeric_input(self.get_input('columns'), self.get_property('columns'))
        row_space = resolve_numeric_input(self.get_input('row_spacing'), self.get_property('row_spacing'))
        col_space = resolve_numeric_input(self.get_input('col_spacing'), self.get_property('col_spacing'))

        try:
            rows, cols = int(rows), int(cols)
            row_space, col_space = float(row_space), float(col_space)

            result = shape
            for i in range(rows):
                for j in range(cols):
                    if i == 0 and j == 0:
                        continue  # Skip original position

                    dx = j * col_space
                    dy = i * row_space
                    dz = 0

                    translated = shape.translate((dx, dy, dz))
                    result = result.union(translated)

            return result
        except Exception as e:
            self.set_error(f"Grid pattern failed: {e}")
            return None
class LinearPatternNode(CadQueryNode):
    """Creates a linear pattern of shapes."""
    __identifier__ = 'com.cad.linear_pattern'
    NODE_NAME = 'Linear Pattern'

    def __init__(self):
        super(LinearPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('count', color=(180, 180, 0))
        self.add_input('distance', color=(180, 180, 0))
        self.add_input('dx', color=(180, 180, 0))
        self.add_input('dy', color=(180, 180, 0))
        self.add_input('dz', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('count', 3, widget_type='int')
        self.create_property('distance', 10.0, widget_type='float')
        self.create_property('dx', 1.0, widget_type='float')
        self.create_property('dy', 0.0, widget_type='float')
        self.create_property('dz', 0.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None: return None

        count = int(resolve_numeric_input(self.get_input('count'), self.get_property('count')))
        dist = float(resolve_numeric_input(self.get_input('distance'), self.get_property('distance')))
        dx = float(resolve_numeric_input(self.get_input('dx'), self.get_property('dx')))
        dy = float(resolve_numeric_input(self.get_input('dy'), self.get_property('dy')))
        dz = float(resolve_numeric_input(self.get_input('dz'), self.get_property('dz')))

        # Normalize direction vector
        import math
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-9: length = 1.0
        
        vec = (dx/length * dist, dy/length * dist, dz/length * dist)

        try:
            result = shape
            for i in range(1, count):
                # Translate i * vec
                t = (vec[0]*i, vec[1]*i, vec[2]*i)
                result = result.union(shape.translate(t))
            return result
        except Exception as e:
            self.set_error(f"Linear pattern error: {e}")
            return shape


class CircularPatternNode(CadQueryNode):
    """Creates a circular pattern of shapes."""
    __identifier__ = 'com.cad.circular_pattern'
    NODE_NAME = 'Circular Pattern'

    def __init__(self):
        super(CircularPatternNode, self).__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('count', color=(180, 180, 0))
        self.add_input('angle', color=(180, 180, 0))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('count', 6, widget_type='int')
        self.create_property('angle', 360.0, widget_type='float')
        self.create_property('axis_x', 0.0, widget_type='float')
        self.create_property('axis_y', 0.0, widget_type='float')
        self.create_property('axis_z', 1.0, widget_type='float')

    def run(self):
        shape = self.get_input_shape('shape')
        if shape is None: return None

        count = int(resolve_numeric_input(self.get_input('count'), self.get_property('count')))
        angle = float(resolve_numeric_input(self.get_input('angle'), self.get_property('angle')))
        ax = float(self.get_property('axis_x'))
        ay = float(self.get_property('axis_y'))
        az = float(self.get_property('axis_z'))

        center = (0,0,0) # Assuming rotating around origin/axis passing through origin for now
        axis = (ax, ay, az)

        try:
            result = shape
            step = angle / count if angle == 360 else angle / (count - 1)
            # If 360, we usually want to distribute evenly, but if count includes 0, logic varies.
            # CadQuery rotate is absolute? No.
            # Simple union loop:
            
            for i in range(1, count):
                d_ang = step * i
                # If 360 and i==0 is implied as start.
                if angle == 360 and i == count: break # Don't overlap start
                
                rotated = shape.rotate(center, (ax+center[0], ay+center[1], az+center[2]), d_ang)
                result = result.union(rotated)
            return result
        except Exception as e:
            self.set_error(f"Circular pattern error: {e}")
            return shape
