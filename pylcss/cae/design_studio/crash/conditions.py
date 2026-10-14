# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash impact condition node: initial velocity / wall scenario definition."""

import numpy as np
from pylcss.cae.design_studio.core.base_node import CadQueryNode
from pylcss.cae.solver_backends.selection import normalize_geometries


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
      With zero impactor mass, OpenCourant treats the wall speed as imposed
      velocity rather than an inertial impact.
    - ``Moving body + fixed wall``: the whole mesh receives an initial velocity
      and hits a generated stationary rigid wall. Disconnect supports for this
      free-flying projectile scenario.
    - ``Prescribed moving wall``: the selected face is driven by a massless
      moving wall/platen. This is useful for controlled crush, not for a
      free impactor whose velocity should decay as energy is absorbed.

    Velocity units: mm / ms = m/s. User material inputs use MPa and
    tonne/mm^3. The backend converts stress to its tonne-mm-ms deck units.

    Tip:
    - 10 km/h ~= 2.778 mm/ms -> velocity_z = -2.778
    - 56 km/h (NCAP) ~= 15.556 mm/ms -> velocity_z = -15.556
    - Keep magnitude <= ~50 mm/ms for typical structural crash.
    """

    __identifier__ = "com.cad.sim.impact"
    NODE_NAME = "Impact Condition"

    def __init__(self):
        super().__init__()
        self.add_input("impact_face", color=(255, 100, 100))
        self.add_output("impact", color=(255, 200, 0))

        # Velocity components in mm/ms (= m/s).
        # Rule of thumb: 10 km/h ~= 2.78 mm/ms; 56 km/h ~= 15.6 mm/ms.
        self.create_property("velocity_x", 0.0, widget_type="float")
        self.create_property("velocity_y", 0.0, widget_type="float")
        self.create_property("velocity_z", -1.0, widget_type="float")
        self.create_property(
            "application_scope",
            CRASH_SCENARIOS[0],
            widget_type="combo",
            items=CRASH_SCENARIOS,
        )
        # Physical impactor mass is part of the event definition, not a
        # numerical solver control. Zero retains prescribed wall velocity.
        self.create_property("impactor_mass_kg", 0.0, widget_type="float")
        # Node-selection tolerance (mm): nodes within this distance of the
        # selected face become rigid-wall secondary candidates.
        self.create_property("node_tolerance", 2.0, widget_type="float")
        self.create_property("wall_friction", 0.0, widget_type="float")
        # Zero/negative means auto gap based on model size.
        self.create_property("wall_gap_mm", 0.0, widget_type="float")
        self.create_property("self_contact_enabled", False, widget_type="checkbox")
        self.create_property("self_contact_friction", 0.0, widget_type="float")

    def run(self):
        self.clear_error()
        face_data = self.get_input_value("impact_face", None)

        face_list = []
        if face_data is not None:
            if isinstance(face_data, dict):
                if str(face_data.get("entity_type") or "Face").title() != "Face":
                    self.set_error(
                        "An impact wall needs a selected face to define its "
                        "plane and normal; edge/vertex selections are not valid."
                    )
                    return None
                face_list = normalize_geometries(face_data)
            elif hasattr(face_data, "vals"):
                face_list = face_data.vals()
            else:
                face_list = [face_data]

        scope = str(self.get_property("application_scope") or CRASH_SCENARIOS[0])
        if not face_list and not scope.lower().replace("_", " ").startswith(
            "moving body"
        ):
            self.set_error(
                "Select an impact face for fixed-specimen or prescribed-wall crash."
            )
            return None

        try:
            velocity = np.array(
                [
                    float(self.get_property("velocity_x")),
                    float(self.get_property("velocity_y")),
                    float(self.get_property("velocity_z")),
                ],
                dtype=float,
            )
            tolerance = float(self.get_property("node_tolerance"))
            friction = float(self.get_property("wall_friction"))
            gap = float(self.get_property("wall_gap_mm"))
            impactor_mass = float(self.get_property("impactor_mass_kg"))
            self_contact_friction = float(self.get_property("self_contact_friction"))
        except (TypeError, ValueError):
            self.set_error("Impact velocity, mass, and wall settings must be numeric.")
            return None
        if (
            not np.all(np.isfinite(velocity))
            or float(np.linalg.norm(velocity)) <= 1e-12
        ):
            self.set_error("Impact velocity must contain a finite, non-zero component.")
            return None
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            self.set_error("Impact surface search tolerance must be greater than zero.")
            return None
        if not np.isfinite(friction) or friction < 0.0:
            self.set_error("Wall friction must be finite and non-negative.")
            return None
        if not np.isfinite(gap):
            self.set_error("Wall gap must be finite.")
            return None
        if not np.isfinite(impactor_mass) or impactor_mass < 0.0:
            self.set_error("Impactor mass must be finite and non-negative.")
            return None
        if not np.isfinite(self_contact_friction) or self_contact_friction < 0.0:
            self.set_error("Self-contact friction must be finite and non-negative.")
            return None
        from pylcss.input_values import as_bool

        return {
            "face_list": face_list,
            "velocity": velocity,
            "application_scope": scope,
            "node_tolerance": tolerance,
            "wall_friction": friction,
            "wall_gap_mm": gap,
            "impactor_mass_kg": impactor_mass,
            "self_contact_enabled": as_bool(self.get_property("self_contact_enabled")),
            "self_contact_friction": self_contact_friction,
        }
