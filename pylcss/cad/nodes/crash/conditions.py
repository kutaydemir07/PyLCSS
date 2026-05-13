# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash impact condition node: initial velocity field definition."""
import numpy as np
from pylcss.cad.core.base_node import CadQueryNode

class ImpactConditionNode(CadQueryNode):
    """
    Defines the crash / impact loading condition.

    In ``Impact Face`` scope, the selected face is struck by a moving rigid
    wall/impactor while connected constraints remain active.  In
    ``Moving Body`` scope, the whole mesh receives an initial velocity and
    hits a generated rigid wall.  This covers fixed-rear crush tests and
    free-body barrier/drop tests without using the same boundary model for
    both cases.

    Units: mm / ms = m/s (consistent with the mm-tonne-N-MPa-ms system).

    Tip:
    - 10 km/h ~= 2.778 mm/ms -> velocity_z = -2.778
    - 56 km/h (NCAP) ~= 15.556 mm/ms -> velocity_z = -15.556
    - Keep magnitude <= ~50 mm/ms for typical structural crash.
    """

    __identifier__ = 'com.cad.sim.impact'
    NODE_NAME = 'Impact Condition'

    def __init__(self):
        super().__init__()
        self.add_input('impact_face', color=(255, 100, 100))
        self.add_output('impact', color=(255, 200, 0))

        # Velocity components in mm/ms (= m/s).
        # Rule of thumb: 10 km/h ~= 2.78 mm/ms; 56 km/h ~= 15.6 mm/ms.
        self.create_property('velocity_x', 0.0,  widget_type='float')
        self.create_property('velocity_y', 0.0,  widget_type='float')
        self.create_property('velocity_z', -1.0, widget_type='float')
        self.create_property(
            'application_scope', 'Impact Face',
            widget_type='combo',
            items=['Impact Face', 'Moving Body'],
        )
        # Node-selection tolerance (mm): nodes within this distance of the
        # impact face receive the initial velocity.
        self.create_property('node_tolerance', 2.0, widget_type='float')

    def run(self):
        face_data = self.get_input_value('impact_face', None)

        face_list = []
        if face_data is not None:
            if isinstance(face_data, dict):
                flist = face_data.get('faces', [])
                if not flist and face_data.get('face') is not None:
                    flist = [face_data['face']]
                face_list = flist
            elif hasattr(face_data, 'vals'):
                face_list = face_data.vals()
            else:
                face_list = [face_data]

        return {
            'face_list':      face_list,
            'velocity':       np.array([
                float(self.get_property('velocity_x')),
                float(self.get_property('velocity_y')),
                float(self.get_property('velocity_z')),
            ]),
            'application_scope': str(self.get_property('application_scope') or 'Impact Face'),
            'node_tolerance': float(self.get_property('node_tolerance')),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: CrashSolverNode
# ─────────────────────────────────────────────────────────────────────────────

