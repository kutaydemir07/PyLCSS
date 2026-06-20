# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM boundary condition nodes — constraints, forces, and pressure loads."""
import numpy as np
import logging
from pylcss.design_studio.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


def _target_face_objects(target_wp):
    """Normalize OCC face and mesh-selection payloads from Select Face nodes."""
    if isinstance(target_wp, dict):
        faces = target_wp.get('faces', None)
        if faces:
            return [f for f in faces if f is not None]
        if target_wp.get('mesh_selection') or target_wp.get('node_ids') is not None:
            return [target_wp]
        face = target_wp.get('face', None)
        return [face] if face is not None else []
    return target_wp.vals() if hasattr(target_wp, 'vals') else []


def _selection_center(item):
    if isinstance(item, dict):
        center = item.get('center')
        if center is not None:
            return [float(v) for v in center[:3]]
        bbox = item.get('bbox') or {}
        try:
            return [
                (float(bbox['xmin']) + float(bbox['xmax'])) / 2.0,
                (float(bbox['ymin']) + float(bbox['ymax'])) / 2.0,
                (float(bbox['zmin']) + float(bbox['zmax'])) / 2.0,
            ]
        except Exception:
            return None
    try:
        bb = item.BoundingBox()
        return [
            (bb.xmin + bb.xmax) / 2,
            (bb.ymin + bb.ymax) / 2,
            (bb.zmin + bb.zmax) / 2,
        ]
    except Exception:
        return None


def _selection_bbox(item):
    if isinstance(item, dict):
        return item.get('bbox')
    try:
        bb = item.BoundingBox()
        return {
            'xmin': bb.xmin, 'xmax': bb.xmax,
            'ymin': bb.ymin, 'ymax': bb.ymax,
            'zmin': bb.zmin, 'zmax': bb.zmax,
        }
    except Exception:
        return None


class ConstraintNode(CadQueryNode):
    """Applies boundary constraints (fixed, roller, pinned, displacement) to a face."""
    __identifier__ = 'com.cad.sim.constraint'
    NODE_NAME = 'FEA Constraint (Face)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to constrain
        self.add_input('target_face', color=(100, 200, 255))
        self.add_output('constraints', color=(255, 100, 100))
        
        # Constraint type selection
        self.create_property('constraint_type', 'Fixed', widget_type='combo',
                             items=['Fixed', 'Pinned', 'Roller X', 'Roller Y', 'Roller Z',
                                    'Symmetry X', 'Symmetry Y', 'Symmetry Z',
                                    'Displacement'])
        
        # Displacement values for prescribed BC (used when type is 'Displacement')
        self.create_property('displacement_x', 0.0, widget_type='float')
        self.create_property('displacement_y', 0.0, widget_type='float')
        self.create_property('displacement_z', 0.0, widget_type='float')
        # Axis toggles allow a prescribed displacement without accidentally
        # locking the other two translational degrees of freedom.
        self.create_property('displacement_x_enabled', True, widget_type='checkbox')
        self.create_property('displacement_y_enabled', True, widget_type='checkbox')
        self.create_property('displacement_z_enabled', True, widget_type='checkbox')
        
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        constraint_type = self.get_property('constraint_type')
        fallback_condition = self.get_property('condition')
        if constraint_type == 'Pinned (Fixed for solids)':
            constraint_type = 'Pinned'


        if mesh is None:
            self.set_error("Connect a mesh to the constraint node.")
            return None

        # Get displacement values
        disp_x = float(self.get_property('displacement_x'))
        disp_y = float(self.get_property('displacement_y'))
        disp_z = float(self.get_property('displacement_z'))

        # Map constraint type to DOF constraints
        disp_enabled = [
            bool(self.get_property('displacement_x_enabled')),
            bool(self.get_property('displacement_y_enabled')),
            bool(self.get_property('displacement_z_enabled')),
        ]
        prescribed_dofs = [idx for idx, enabled in enumerate(disp_enabled) if enabled]
        if constraint_type == 'Displacement' and not prescribed_dofs:
            self.set_error("Enable at least one displacement axis (UX, UY, or UZ).")
            return None

        # 'fixed_dofs' indicates which DOFs (0=x, 1=y, 2=z) are fixed
        # NOTE on 'Pinned': for 3-D solid elements there are no rotational DOFs —
        # rotation is captured through relative translations of adjacent nodes.  A
        # Pinned BC therefore constrains the same three translational DOFs as Fixed.
        # The practical difference only matters in beam/shell formulations.
        constraint_mapping = {
            'Fixed':      {'fixed_dofs': [0, 1, 2], 'displacement': None},
            'Roller X':   {'fixed_dofs': [0],        'displacement': None},  # Normal-X blocked
            'Roller Y':   {'fixed_dofs': [1],        'displacement': None},  # Normal-Y blocked
            'Roller Z':   {'fixed_dofs': [2],        'displacement': None},  # Normal-Z blocked
            'Pinned':     {'fixed_dofs': [0, 1, 2],  'displacement': None},  # = Fixed for solids
            'Symmetry X': {'fixed_dofs': [0],        'displacement': None},  # No motion in X
            'Symmetry Y': {'fixed_dofs': [1],        'displacement': None},  # No motion in Y
            'Symmetry Z': {'fixed_dofs': [2],        'displacement': None},  # No motion in Z
            'Displacement': {'fixed_dofs': prescribed_dofs, 'displacement': [disp_x, disp_y, disp_z]},
        }
        # Colour coding used by the viewer for pre-solve BC overlay:
        #   Fixed / Pinned → blue    Roller → cyan    Symmetry → green
        #   Displacement   → orange
        _viz_color_map = {
            'Fixed': '#2979FF',      'Pinned': '#2979FF',
            'Roller X': '#00E5FF',   'Roller Y': '#00E5FF',   'Roller Z': '#00E5FF',
            'Symmetry X': '#00E676', 'Symmetry Y': '#00E676', 'Symmetry Z': '#00E676',
            'Displacement': '#FF9100',
        }
        constraint_info = constraint_mapping.get(constraint_type, constraint_mapping['Fixed'])
        viz_color = _viz_color_map.get(constraint_type, '#2979FF')

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                self.set_error("No target face or condition")
                return None
            return {
                'type': constraint_type.lower().replace(' ', '_'),
                'condition': fallback_condition,
                'fixed_dofs': constraint_info['fixed_dofs'],
                'displacement': constraint_info['displacement'],
                # No geometry — viewer cannot draw pre-solve overlay for string conditions
                'viz': None,
            }

        # Extract faces from SelectFaceNode dict format {'workplane': ..., 'faces': [...]}
        try:
            face_objs = _target_face_objects(target_wp)

            if not face_objs or face_objs[0] is None:
                self.set_error("No faces found in target face input")
                return None

            # Build per-face bboxes for the viewer pre-solve overlay
            viz_faces = []
            for f in face_objs:
                if f is None:
                    continue
                bbox = _selection_bbox(f)
                center = _selection_center(f)
                if bbox is not None and center is not None:
                    viz_faces.append({
                        'bbox': bbox,
                        'center': center,
                        'points': f.get('points') if isinstance(f, dict) else None,
                    })

            return {
                'type': constraint_type.lower().replace(' ', '_'),
                'geometries': face_objs,  # Pass the list of faces
                'fixed_dofs': constraint_info['fixed_dofs'],
                'displacement': constraint_info['displacement'],
                # Pre-solve viewer overlay metadata
                'viz': {
                    'constraint_type': constraint_type,
                    'color': viz_color,
                    'fixed_dofs': list(constraint_info['fixed_dofs']),
                    'displacement': constraint_info['displacement'],
                    'faces': viz_faces,
                },
            }

        except Exception as e:
            print(f"DEBUG ConstraintNode ({self.NODE_NAME}): ERROR during run: {e}")
            self.set_error(f"Constraint setup failed: {e}")
            return None

class LoadNode(CadQueryNode):
    """Applies a load to a specific geometric face."""
    __identifier__ = 'com.cad.sim.load'
    NODE_NAME = 'FEA Load (Face)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to load (not used for Gravity)
        self.add_input('target_face', color=(100, 200, 255))

        # Add inputs for parametric force components
        self.add_input('force_x', color=(255, 255, 0))
        self.add_input('force_y', color=(255, 255, 0))
        self.add_input('force_z', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))

        # Load type selection — Moment and Remote Force removed: not yet implemented.
        # They were listed in the dropdown but returned None with an error, giving users
        # the impression they work.  Re-add them when implemented.
        self.create_property('load_type', 'Force', widget_type='combo',
                             items=['Force', 'Gravity'])

        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

        # Force/load values
        self.create_property('force_x', 0.0, widget_type='float')
        self.create_property('force_y', 0.0, widget_type='float')
        self.create_property('force_z', 0.0, widget_type='float')
        self.create_property('moment_x', 0.0, widget_type='float')
        self.create_property('moment_y', 0.0, widget_type='float')
        self.create_property('moment_z', 0.0, widget_type='float')

        # Gravity parameters
        self.create_property('gravity_accel', 9810.0, widget_type='float')
        self.create_property('gravity_direction', '-Y', widget_type='combo',
                             items=['-Y', '-Z', '-X', '+Y', '+Z', '+X'])

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)  # Not used for Gravity

        if mesh is None:
            self.set_error("Connect a mesh to the load node.")
            return None

        # Resolve force inputs with fallback to properties
        fx = self.get_input_value('force_x', 'force_x')
        fy = self.get_input_value('force_y', 'force_y')
        fz = self.get_input_value('force_z', 'force_z')

        load_type          = self.get_property('load_type')
        fallback_condition = self.get_property('condition')

        # ── Gravity body force: no face needed — acts on the whole body ──────────
        if load_type == 'Gravity':
            return {
                'type':      'gravity',
                'accel':     float(self.get_property('gravity_accel')),
                'direction': self.get_property('gravity_direction'),
                # Pre-solve viz: gravity uses a global body-force icon, no face
                'viz': {'load_type': 'Gravity', 'direction': self.get_property('gravity_direction')},
            }

        # ── Pressure load ───────────────────────────────────────────────────────
        # NOTE: 'Pressure' is surfaced here for routing — actual assembly is done
        # inside SolverNode.  LoadNode routes the face geometry and pressure value;
        # SolverNode assembles the proper Neumann BC via FacetBasis.

        # ── Force (default) ────────────────────────────────────────────────────
        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                self.set_error("No target face or condition")
                return None
            return {
                'type': 'force',
                'condition': fallback_condition,
                'vector': [float(fx), float(fy), float(fz)],
                'viz': None,
            }

        # Extract faces from SelectFaceNode dict format {'workplane': ..., 'faces': [...]}
        try:
            face_objs = _target_face_objects(target_wp)

            if not face_objs or face_objs[0] is None:
                self.set_error("No faces found in target face input")
                return None

            force_vec = [float(fx), float(fy), float(fz)]
            force_mag = float(np.linalg.norm(force_vec))

            # Build per-face bbox for pre-solve force arrow overlay
            viz_faces = []
            for f in face_objs:
                if f is None:
                    continue
                center = _selection_center(f)
                if center is not None:
                    viz_faces.append({
                        'center': center,
                        'points': f.get('points') if isinstance(f, dict) else None,
                    })

            return {
                'type': 'force',
                'geometries': face_objs,
                'vector': force_vec,
                # Pre-solve viewer overlay metadata — normalized direction + real magnitude
                'viz': {
                    'load_type':   'Force',
                    'vector':      force_vec,
                    'magnitude_N': force_mag,
                    'faces':       viz_faces,
                },
            }

        except Exception:
            self.set_error("Load setup failed")
            return None


class PressureLoadNode(CadQueryNode):
    """Applies a pressure load to a specific geometric face."""
    __identifier__ = 'com.cad.sim.pressure_load'
    NODE_NAME = 'FEA Pressure Load'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to apply pressure
        self.add_input('target_face', color=(100, 200, 255))
        # Pressure magnitude (positive)
        self.add_input('pressure', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))
        self.create_property('pressure', 1.0, widget_type='float')  # 1 MPa default
        self.create_property('direction', 'Inward', widget_type='combo', items=['Inward', 'Outward'])

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        pressure = self.get_input_value('pressure', 'pressure')

        if mesh is None:
            self.set_error("Connect a mesh to the pressure-load node.")
            return None
        if target_wp is None:
            self.set_error("Connect a selected target face to the pressure-load node.")
            return None

        try:
            # Extract the face geometry
            face_objs = _target_face_objects(target_wp)

            if not face_objs:
                self.set_error("No faces found in target face input")
                return None

            # CalculiX *DLOAD P is positive in compression (opposite the
            # element's outward surface normal); negative pressure is tension.
            # Therefore Inward is positive and Outward is negative.
            direction = self.get_property('direction')
            sign = 1.0 if direction == 'Inward' else -1.0

            return {
                'type': 'pressure',
                'geometries': face_objs,   # all selected faces (may be multiple)
                'geometry': face_objs[0],  # kept for backward compatibility
                'pressure': float(pressure) * sign,
                'viz': {
                    'load_type': 'Pressure',
                    'bbox': _selection_bbox(face_objs[0]),
                    'color': '#ff00ff'
                }
            }

        except Exception as e:
            self.set_error(f"Pressure load setup failed: {e}")
            return None


