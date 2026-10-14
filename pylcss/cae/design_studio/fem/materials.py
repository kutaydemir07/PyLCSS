# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM material node — elastic material properties with preset database."""

import math

from pylcss.cae.design_studio.core.base_node import CadQueryNode, resolve_numeric_input
from pylcss.cae.design_studio.crash.materials import (
    CRASH_MATERIAL_PRESETS,
    IMPACT_MATERIAL_PRESETS,
    add_impact_material_properties,
    impact_material_payload,
)
from pylcss.cae.design_studio.fem._helpers import MATERIAL_DATABASE


class MaterialNode(CadQueryNode):
    """Shared elastic and impact material properties with preset databases."""

    __identifier__ = "com.cad.sim.material"
    NODE_NAME = "Material"

    def __init__(self):
        super().__init__()
        self.add_output("material", color=(200, 200, 200))

        self.create_property(
            "analysis_purpose",
            "Structural FEA",
            widget_type="combo",
            items=["Structural FEA", "Topology / Lattice", "Impact"],
        )

        # Add Inputs for parametric material properties
        self.add_input("youngs_modulus", color=(180, 180, 0))
        self.add_input("poissons_ratio", color=(180, 180, 0))
        self.add_input("density", color=(180, 180, 0))

        # Preset dropdown. The impact catalogue is offered alongside the
        # elastic one: choosing an impact preset brings its calibrated flow
        # curve and rate constants, choosing an elastic preset leaves
        # plasticity to the explicit fields below.
        self.create_property(
            "preset",
            "Steel (Structural)",
            widget_type="combo",
            items=list(
                dict.fromkeys(
                    list(MATERIAL_DATABASE.keys()) + list(IMPACT_MATERIAL_PRESETS)
                )
            ),
        )

        # Keep properties as defaults (editable for Custom)
        self.create_property("youngs_modulus", 210000.0, widget_type="float")  # MPa
        self.create_property("poissons_ratio", 0.3, widget_type="float")
        self.create_property("density", 7.85e-9, widget_type="float")  # tonne/mm^3
        # Yield is an allowable in a linear study. It becomes an isotropic
        # bilinear-hardening law only when the solver explicitly selects
        # Nonlinear (Plastic).
        self.create_property("yield_strength", 0.0, widget_type="float")  # MPa
        self.create_property("tangent_modulus", 0.0, widget_type="float")  # MPa

        # Failure strain, rate sensitivity and lot traceability — used only
        # when this material feeds an impact study.
        add_impact_material_properties(self)

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo=push_undo)
        if name == "preset" and value != "Custom":
            mat = CRASH_MATERIAL_PRESETS.get(value, MATERIAL_DATABASE.get(value, {}))
            if "E" in mat:
                super().set_property(
                    "youngs_modulus", float(mat["E"]), push_undo=push_undo
                )
            if "nu" in mat:
                super().set_property(
                    "poissons_ratio", float(mat["nu"]), push_undo=push_undo
                )
            if "rho" in mat:
                super().set_property("density", float(mat["rho"]), push_undo=push_undo)
            for field in ("yield_strength", "tangent_modulus", "failure_strain"):
                if field in mat:
                    super().set_property(field, float(mat[field]), push_undo=push_undo)

    def run(self):
        self.clear_error()
        # Check if using preset or custom
        preset = self.get_property("preset")

        mat = (
            CRASH_MATERIAL_PRESETS.get(preset, MATERIAL_DATABASE.get(preset, {}))
            if preset != "Custom"
            else {}
        )
        try:
            E = float(
                resolve_numeric_input(
                    self.get_input("youngs_modulus"),
                    mat.get("E", self.get_property("youngs_modulus")),
                )
            )
            nu = float(
                resolve_numeric_input(
                    self.get_input("poissons_ratio"),
                    mat.get("nu", self.get_property("poissons_ratio")),
                )
            )
            rho = float(
                resolve_numeric_input(
                    self.get_input("density"),
                    mat.get("rho", self.get_property("density")),
                )
            )
            sigma_y = float(self.get_property("yield_strength") or 0.0)
            Et = float(self.get_property("tangent_modulus") or 0.0)
        except (TypeError, ValueError):
            self.set_error("Material properties must be numeric.")
            return None
        values = (E, nu, rho, sigma_y, Et)
        if not all(math.isfinite(value) for value in values):
            self.set_error("Material properties must be finite numbers.")
            return None
        if E <= 0.0:
            self.set_error("Young's modulus must be greater than zero.")
            return None
        if not (-1.0 < nu < 0.5):
            self.set_error("Poisson's ratio must be between -1 and 0.5 (exclusive).")
            return None
        if rho <= 0.0:
            self.set_error("Density must be greater than zero.")
            return None
        if sigma_y < 0.0 or Et < 0.0:
            self.set_error("Yield strength and tangent modulus cannot be negative.")
            return None
        if sigma_y > 0.0 and Et >= E:
            self.set_error("Tangent modulus must be smaller than Young's modulus.")
            return None

        out = {
            "name": str(preset or "Custom"),
            "E": E,
            "nu": nu,
            "rho": rho,
        }
        if sigma_y > 0.0:
            out["yield_strength"] = sigma_y
            out["tangent_modulus"] = Et

        purpose = str(self.get_property("analysis_purpose") or "Structural FEA")
        if purpose == "Impact":
            out = impact_material_payload(self, elastic=(E, nu, rho))
            if self.has_error():
                return None
        return {"material": out}
