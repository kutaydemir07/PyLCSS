# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM boundary condition nodes — constraints, forces, and pressure loads."""
import numpy as np
from pylcss.cae.design_studio.core.base_node import CadQueryNode
from pylcss.cae.solver_backends.selection import normalize_geometries


def _target_face_objects(target_wp):
    """Normalize geometry-selection payloads from Select Geometry nodes."""
    value = target_wp.vals() if hasattr(target_wp, 'vals') else target_wp
    return normalize_geometries(value)


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


def _pressure_load_payload(mesh, selection, magnitude, direction):
    """Build the surface traction shared by both pressure node interfaces."""
    if selection is None:
        raise ValueError("Connect a selected target face to the pressure load.")
    if (
        isinstance(selection, dict)
        and str(selection.get('entity_type') or 'Face').title() != 'Face'
    ):
        raise ValueError(
            "Pressure is a surface traction and requires a face selection; "
            "an edge or vertex has no loaded area."
        )
    try:
        pressure = float(magnitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pressure magnitude must be numeric.") from exc
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("Pressure magnitude must be a finite value greater than zero.")
    faces = _target_face_objects(selection)
    if not faces:
        raise ValueError("No faces found in target face input")
    # CalculiX pressure is positive in compression and negative in tension.
    sign = 1.0 if direction == 'Inward' else -1.0
    return {
        'type': 'pressure',
        'mesh': mesh,
        'geometries': faces,
        'geometry': faces[0],
        'pressure': pressure * sign,
        'viz': {
            'load_type': 'Pressure',
            'bbox': _selection_bbox(faces[0]),
            'color': '#ff00ff',
        },
    }


class ConstraintNode(CadQueryNode):
    """Apply a structural support in either FEA or topology optimization.

    A finite-element mesh is optional because TopOpt creates its analysis grid
    internally.  Static FEA studies may still connect the mesh so legacy
    coordinate expressions and mesh-scoped selections keep working.
    """
    __identifier__ = 'com.cad.sim.constraint'
    NODE_NAME = 'Support'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to constrain
        self.add_input('target_face', color=(100, 200, 255))
        self.add_output('constraints', color=(255, 100, 100))
        
        # Constraint type selection
        self.create_property('constraint_type', 'Fixed', widget_type='combo',
                             items=['Fixed',
                                    'Block X Translation',
                                    'Block Y Translation',
                                    'Block Z Translation',
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
        from pylcss.input_values import as_bool

        self.clear_error()
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        constraint_type = self.get_property('constraint_type') or 'Fixed'
        fallback_condition = self.get_property('condition')
        # Translational degrees of freedom: 0=X, 1=Y, 2=Z.
        blocked_axes = {
            'Fixed': [0, 1, 2],
            'Block X Translation': [0],
            'Block Y Translation': [1],
            'Block Z Translation': [2],
        }
        displacement = None
        if constraint_type == 'Displacement':
            fixed_dofs = [
                idx for idx, axis in enumerate('xyz')
                if as_bool(self.get_property(f'displacement_{axis}_enabled'))
            ]
            if not fixed_dofs:
                self.set_error("Enable at least one displacement axis (UX, UY, or UZ).")
                return None
            displacement = [0.0, 0.0, 0.0]
            try:
                for idx in fixed_dofs:
                    displacement[idx] = float(
                        self.get_property(f'displacement_{"xyz"[idx]}')
                    )
            except (TypeError, ValueError):
                self.set_error("Prescribed displacement values must be numeric.")
                return None
            if not np.all(np.isfinite(displacement)):
                self.set_error("Prescribed displacement values must be finite.")
                return None
            viz_color = '#FF9100'
        elif constraint_type in blocked_axes:
            fixed_dofs = blocked_axes[constraint_type]
            viz_color = '#2979FF' if constraint_type == 'Fixed' else '#00E5FF'
        else:
            self.set_error(f"Unknown support type: {constraint_type}.")
            return None

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                self.set_error("Connect selected geometry or enter a coordinate condition.")
                return None
            return {
                'type': constraint_type.lower().replace(' ', '_'),
                'mesh': mesh,
                'condition': fallback_condition,
                'fixed_dofs': fixed_dofs,
                'displacement': displacement,
                # No geometry — viewer cannot draw pre-solve overlay for string conditions
                'viz': None,
            }

        # Extract faces from SelectFaceNode dict format {'workplane': ..., 'faces': [...]}
        try:
            face_objs = _target_face_objects(target_wp)

            if not face_objs or face_objs[0] is None:
                self.set_error("The connected geometry selection is empty.")
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
                'mesh': mesh,
                'geometries': face_objs,  # Pass the list of faces
                'fixed_dofs': fixed_dofs,
                'displacement': displacement,
                # Pre-solve viewer overlay metadata
                'viz': {
                    'constraint_type': constraint_type,
                    'color': viz_color,
                    'fixed_dofs': list(fixed_dofs),
                    'displacement': displacement,
                    'faces': viz_faces,
                },
            }

        except Exception as e:
            self.set_error(f"Constraint setup failed: {e}")
            return None

class LoadNode(CadQueryNode):
    """Apply a shared FEA/TopOpt force, or FEA-only gravity.

    Topology optimization maps the selected CAD region to its internal voxel
    grid, so a mesh connection is optional for force loads. Gravity remains a
    whole-mesh FEA load and therefore still requires one.
    """
    __identifier__ = 'com.cad.sim.load'
    NODE_NAME = 'Load'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to load (not used for Gravity)
        self.add_input('target_face', color=(100, 200, 255))

        # Add inputs for parametric force components
        self.add_input('force_x', color=(255, 255, 0))
        self.add_input('force_y', color=(255, 255, 0))
        self.add_input('force_z', color=(255, 255, 0))
        # Input for parametric surface pressure magnitude
        self.add_input('pressure', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))

        # Load type selection — supports Force, Pressure, Gravity, and Bearing Load.
        self.create_property('load_type', 'Force', widget_type='combo',
                             items=['Force', 'Pressure', 'Gravity', 'Bearing Load'])

        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

        # Force/load values
        self.create_property('force_x', 0.0, widget_type='float')
        self.create_property('force_y', 0.0, widget_type='float')
        self.create_property('force_z', 0.0, widget_type='float')
        # Pressure values
        self.create_property('pressure', 1.0, widget_type='float')
        self.create_property('direction', 'Inward', widget_type='combo',
                             items=['Inward', 'Outward'])
        # Gravity parameters
        self.create_property('gravity_accel', 9810.0, widget_type='float')
        self.create_property('gravity_direction', '-Y', widget_type='combo',
                             items=['-Y', '-Z', '-X', '+Y', '+Z', '+X'])
        # Multi-load case weighting for topology optimization
        self.create_property('case_weight', 1.0, widget_type='float')
        self.create_property('case_name', '', widget_type='text')

    def run(self):
        self.clear_error()
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)  # Not used for Gravity

        load_type          = self.get_property('load_type')
        fallback_condition = self.get_property('condition')
        case_weight        = float(self.get_property('case_weight') or 1.0)
        case_name          = str(self.get_property('case_name') or '').strip()

        # ── Gravity body force: no face needed — acts on the whole body ──────────
        if load_type == 'Gravity':
            if mesh is None:
                self.set_error("Gravity is an FEA body load and requires an FE mesh.")
                return None
            accel = float(self.get_property('gravity_accel'))
            if not np.isfinite(accel) or accel <= 0.0:
                self.set_error("Gravity acceleration must be a finite value greater than zero.")
                return None
            return {
                'type':      'gravity',
                'mesh':      mesh,
                'accel':     accel,
                'direction': self.get_property('gravity_direction'),
                'weight':    case_weight,
                'name':      case_name,
                # Pre-solve viz: gravity uses a global body-force icon, no face
                'viz': {'load_type': 'Gravity', 'direction': self.get_property('gravity_direction')},
            }

        # ── Pressure load: surface traction on selected face ───────────────────
        if load_type == 'Pressure':
            try:
                result = _pressure_load_payload(
                    mesh, target_wp,
                    self.get_input_value('pressure', 'pressure'),
                    self.get_property('direction'),
                )
            except ValueError as exc:
                self.set_error(str(exc))
                return None
            except Exception as exc:
                self.set_error(f"Pressure load setup failed: {exc}")
                return None
            result.update(weight=case_weight, name=case_name)
            return result

        # ── Bearing Load: cosine compressive contact on cylindrical hole ───────
        if load_type == 'Bearing Load':
            if target_wp is None and not fallback_condition:
                self.set_error("Connect a cylindrical target face or enter a condition for the bearing load.")
                return None
            fx = self.get_input_value('force_x', 'force_x')
            fy = self.get_input_value('force_y', 'force_y')
            fz = self.get_input_value('force_z', 'force_z')
            try:
                force_vec = np.asarray([fx, fy, fz], dtype=float)
            except (TypeError, ValueError):
                self.set_error("Bearing load force components must be numeric.")
                return None
            if not np.all(np.isfinite(force_vec)) or float(np.linalg.norm(force_vec)) <= 1e-12:
                self.set_error("Bearing load must contain a finite, non-zero force vector.")
                return None

            face_objs = _target_face_objects(target_wp) if target_wp is not None else []
            force_mag = float(np.linalg.norm(force_vec))
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
                'type': 'bearing_load',
                'mesh': mesh,
                'geometries': face_objs if face_objs else None,
                'condition': fallback_condition if not face_objs else None,
                'vector': force_vec.tolist(),
                'weight': case_weight,
                'name': case_name,
                'viz': {
                    'load_type': 'Bearing Load',
                    'vector': force_vec.tolist(),
                    'magnitude_N': force_mag,
                    'faces': viz_faces,
                },
            }

        # ── Force: vector components on selected face ──────────────────────────
        fx = self.get_input_value('force_x', 'force_x')
        fy = self.get_input_value('force_y', 'force_y')
        fz = self.get_input_value('force_z', 'force_z')

        try:
            force_vec = np.asarray([fx, fy, fz], dtype=float)
        except (TypeError, ValueError):
            self.set_error("Force components must be numeric.")
            return None
        if not np.all(np.isfinite(force_vec)) or float(np.linalg.norm(force_vec)) <= 1e-12:
            self.set_error("Force must contain a finite, non-zero component.")
            return None

        # ── Force (default) ────────────────────────────────────────────────────
        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                self.set_error("Connect selected geometry or enter a coordinate condition.")
                return None
            return {
                'type': 'force',
                'mesh': mesh,
                'condition': fallback_condition,
                'vector': force_vec.tolist(),
                'weight': case_weight,
                'name': case_name,
                'viz': None,
            }

        # Extract faces from SelectFaceNode dict format {'workplane': ..., 'faces': [...]}
        try:
            face_objs = _target_face_objects(target_wp)

            if not face_objs or face_objs[0] is None:
                self.set_error("The connected geometry selection is empty.")
                return None

            force_vec = force_vec.tolist()
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
                'mesh': mesh,
                'geometries': face_objs,
                'vector': force_vec,
                'weight': case_weight,
                'name': case_name,
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
    """Apply pressure to a selected face in FEA or topology optimization."""
    __identifier__ = 'com.cad.sim.pressure_load'
    NODE_NAME = 'Pressure'

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
        self.clear_error()
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        try:
            return _pressure_load_payload(
                mesh, target_wp,
                self.get_input_value('pressure', 'pressure'),
                self.get_property('direction'),
            )
        except ValueError as exc:
            self.set_error(str(exc))
            return None
        except Exception as exc:
            self.set_error(f"Pressure load setup failed: {exc}")
            return None
