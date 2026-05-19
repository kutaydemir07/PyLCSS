# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash impact condition node: initial velocity / wall scenario definition."""
import numpy as np
from pylcss.cad.core.base_node import CadQueryNode


CRASH_SCENARIOS = [
    "Fixed specimen + moving impactor",
    "Moving body + fixed wall",
    "Prescribed moving wall",
]


class ImpactConditionNode(CadQueryNode):
    """
    Defines the crash / impact loading condition.

    Scenarios:
    - ``Fixed specimen + moving impactor``: the selected face is struck by a
      moving rigid wall/impactor while connected constraints remain active.
      With zero impactor mass, OpenRadioss treats the wall speed as imposed
      velocity rather than an inertial impact.
    - ``Moving body + fixed wall``: the whole mesh receives an initial velocity
      and hits a generated stationary rigid wall. Connected constraints are
      ignored because the structure is a free-flying projectile.
    - ``Prescribed moving wall``: the selected face is driven by a massless
      moving wall/platen. This is useful for controlled crush, not for a
      free impactor whose velocity should decay as energy is absorbed.

    Legacy saved values ``Impact Face`` and ``Moving Body`` are still accepted.

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
            'application_scope', CRASH_SCENARIOS[0],
            widget_type='combo',
            items=CRASH_SCENARIOS,
        )
        # Node-selection tolerance (mm): nodes within this distance of the
        # selected face become rigid-wall secondary candidates.
        self.create_property('node_tolerance', 2.0, widget_type='float')
        # Negative means scenario default: frictionless fixed barrier for the
        # moving-body case, low-friction platen for the moving-wall cases.
        self.create_property('wall_friction', -1.0, widget_type='float')
        # Zero/negative means auto gap based on model size.
        self.create_property('wall_gap_mm', 0.0, widget_type='float')

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

        wall_friction = self.get_property('wall_friction')
        wall_gap_mm = self.get_property('wall_gap_mm')
        return {
            'face_list':      face_list,
            'velocity':       np.array([
                float(self.get_property('velocity_x')),
                float(self.get_property('velocity_y')),
                float(self.get_property('velocity_z')),
            ]),
            'application_scope': str(self.get_property('application_scope') or CRASH_SCENARIOS[0]),
            'node_tolerance': float(self.get_property('node_tolerance')),
            'wall_friction': float(wall_friction if wall_friction is not None else -1.0),
            'wall_gap_mm': float(wall_gap_mm if wall_gap_mm is not None else 0.0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: CrashSolverNode
# ─────────────────────────────────────────────────────────────────────────────

