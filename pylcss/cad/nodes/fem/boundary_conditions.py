# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM boundary condition nodes — constraints, forces, and pressure loads."""
import numpy as np
import logging
from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)

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
                             items=['Fixed', 'Pinned (Fixed for solids)', 'Roller X', 'Roller Y', 'Roller Z',
                                    'Symmetry X', 'Symmetry Y', 'Symmetry Z',
                                    'Displacement'])
        
        # Displacement values for prescribed BC (used when type is 'Displacement')
        self.create_property('displacement_x', 0.0, widget_type='float')
        self.create_property('displacement_y', 0.0, widget_type='float')
        self.create_property('displacement_z', 0.0, widget_type='float')
        
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        constraint_type = self.get_property('constraint_type')
        fallback_condition = self.get_property('condition')

        if mesh is None:
            print(f"DEBUG ConstraintNode ({self.NODE_NAME}): ABORTING - NO MESH")
            return None

        # Get displacement values
        disp_x = float(self.get_property('displacement_x'))
        disp_y = float(self.get_property('displacement_y'))
        disp_z = float(self.get_property('displacement_z'))

        # Map constraint type to DOF constraints
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
            'Displacement': {'fixed_dofs': [0, 1, 2], 'displacement': [disp_x, disp_y, disp_z]},
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
            if isinstance(target_wp, dict):
                # Use 'faces' list if available, otherwise fallback to 'face'
                face_objs = target_wp.get('faces', [target_wp.get('face')])
            else:
                # Fallback: try to get vals from workplane (legacy support)
                face_objs = target_wp.vals() if hasattr(target_wp, 'vals') else []

            if not face_objs or face_objs[0] is None:
                self.set_error("No faces found in target face input")
                return None

            # Build per-face bboxes for the viewer pre-solve overlay
            viz_faces = []
            for f in face_objs:
                if f is None:
                    continue
                try:
                    bb = f.BoundingBox()
                    viz_faces.append({
                        'bbox': {
                            'xmin': bb.xmin, 'xmax': bb.xmax,
                            'ymin': bb.ymin, 'ymax': bb.ymax,
                            'zmin': bb.zmin, 'zmax': bb.zmax,
                        },
                        'center': [
                            (bb.xmin + bb.xmax) / 2,
                            (bb.ymin + bb.ymax) / 2,
                            (bb.zmin + bb.zmax) / 2,
                        ]
                    })
                except Exception:
                    pass

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
            print(f"DEBUG LoadNode ({self.NODE_NAME}): ABORTING - NO MESH")
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
            if isinstance(target_wp, dict):
                face_objs = target_wp.get('faces', [target_wp.get('face')])
            else:
                face_objs = target_wp.vals() if hasattr(target_wp, 'vals') else []

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
                try:
                    bb = f.BoundingBox()
                    viz_faces.append({
                        'center': [
                            (bb.xmin + bb.xmax) / 2,
                            (bb.ymin + bb.ymax) / 2,
                            (bb.zmin + bb.zmax) / 2,
                        ]
                    })
                except Exception:
                    pass

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

        if mesh is None or target_wp is None:
            return None

        try:
            # Extract the face geometry
            if isinstance(target_wp, dict):
                # Handle SelectFaceNode output dict {'workplane', 'face', 'faces'}
                face_objs = target_wp.get('faces', [])
                if not face_objs and 'face' in target_wp:
                    face_objs = [target_wp['face']]
            else:
                # Handle direct Workplane input
                face_objs = target_wp.vals() if hasattr(target_wp, 'vals') else []

            if not face_objs:
                self.set_error("No faces found in target face input")
                return None

            # Apply direction sign (Outward = positive normal, Inward = negative normal)
            direction = self.get_property('direction')
            sign = 1.0 if direction == 'Outward' else -1.0

            return {
                'type': 'pressure',
                'geometries': face_objs,   # all selected faces (may be multiple)
                'geometry': face_objs[0],  # kept for backward compatibility
                'pressure': float(pressure) * sign,
                'viz': {
                    'load_type': 'Pressure',
                    'bbox': face_objs[0].BoundingBox(),
                    'color': '#ff00ff'
                }
            }

        except Exception as e:
            self.set_error(f"Pressure load setup failed: {e}")
            return None


