# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM material node — elastic material properties with preset database."""
import math

from pylcss.design_studio.core.base_node import CadQueryNode, resolve_numeric_input
from pylcss.design_studio.crash.materials import (
    IMPACT_MATERIAL_PRESETS,
    add_impact_material_properties,
    impact_material_payload,
)
from pylcss.design_studio.fem._helpers import MATERIAL_DATABASE

class MaterialNode(CadQueryNode):
    """Defines material properties with preset database.

    One material node serves the static, topology and impact studies. The
    impact card used to be a node of its own; a part has one material, and
    keeping two nodes meant a static study and an impact study could describe
    the same steel differently without anything noticing.
    """
    __identifier__ = 'com.cad.sim.material'
    NODE_NAME = 'Material'

    def __init__(self):
        super().__init__()
        self.add_output('material', color=(200, 200, 200))
        impact_output = self.add_output(
            'impact_material', display_name=False, color=(255, 150, 50)
        )
        impact_output.view.setVisible(False)
        # Retained so studies saved against the separate Impact Material node
        # keep their wiring when they are reopened.
        legacy_output = self.add_output(
            'crash_material', display_name=False, color=(255, 150, 50)
        )
        legacy_output.view.setVisible(False)

        self.create_property(
            'analysis_purpose',
            'Structural FEA',
            widget_type='combo',
            items=['Structural FEA', 'Topology / Lattice', 'Impact'],
        )

        # Add Inputs for parametric material properties
        self.add_input('youngs_modulus', color=(180, 180, 0))
        self.add_input('poissons_ratio', color=(180, 180, 0))
        self.add_input('density', color=(180, 180, 0))
        self.add_input('thermal_conductivity', color=(255, 110, 50))

        # Preset dropdown. The impact catalogue is offered alongside the
        # elastic one: choosing an impact preset brings its calibrated flow
        # curve and rate constants, choosing an elastic preset leaves
        # plasticity to the explicit fields below.
        self.create_property(
            'preset', 'Steel (Structural)', widget_type='combo',
            items=list(
                dict.fromkeys(
                    list(MATERIAL_DATABASE.keys()) + list(IMPACT_MATERIAL_PRESETS)
                )
            ),
        )
        
        # Keep properties as defaults (editable for Custom)
        self.create_property('youngs_modulus', 210000.0, widget_type='float')  # MPa
        self.create_property('poissons_ratio', 0.3, widget_type='float')
        self.create_property('density', 7.85e-9, widget_type='float')  # tonne/mm^3
        self.create_property(
            'thermal_conductivity',
            45.0,
            widget_type='float',
        )

        # Yield is an allowable in a linear study. It becomes an isotropic
        # bilinear-hardening law only when the solver explicitly selects
        # Nonlinear (Plastic).
        self.create_property('yield_strength',  0.0, widget_type='float')   # MPa
        self.create_property('tangent_modulus', 0.0, widget_type='float')   # MPa

        # Failure strain, rate sensitivity and lot traceability — used only
        # when this material feeds an impact study.
        add_impact_material_properties(self)

    def run(self):
        self.clear_error()
        # Check if using preset or custom
        preset = self.get_property('preset')

        if preset != 'Custom' and preset in MATERIAL_DATABASE:
            mat = MATERIAL_DATABASE[preset]
            E = resolve_numeric_input(self.get_input('youngs_modulus'), mat['E'])
            nu = resolve_numeric_input(self.get_input('poissons_ratio'), mat['nu'])
            rho = resolve_numeric_input(self.get_input('density'), mat['rho'])
            thermal_conductivity = resolve_numeric_input(
                self.get_input('thermal_conductivity'),
                mat['k'],
            )
        else:
            # Resolve inputs with fallback to properties
            E = self.get_input_value('youngs_modulus', 'youngs_modulus')
            nu = self.get_input_value('poissons_ratio', 'poissons_ratio')
            rho = self.get_input_value('density', 'density')
            thermal_conductivity = self.get_input_value(
                'thermal_conductivity',
                'thermal_conductivity',
            )

        # Plasticity is independent of the preset choice — surface as
        # explicit overrides so a user can mix Steel (preset) + custom
        # yield strength without typing all the elastic fields.
        sigma_y = float(self.get_property('yield_strength')  or 0.0)
        Et      = float(self.get_property('tangent_modulus') or 0.0)

        try:
            E = float(E)
            nu = float(nu)
            rho = float(rho)
            thermal_conductivity = float(thermal_conductivity)
        except (TypeError, ValueError):
            self.set_error("Material properties must be numeric.")
            return None
        values = (E, nu, rho, thermal_conductivity, sigma_y, Et)
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
        if thermal_conductivity <= 0.0:
            self.set_error("Thermal conductivity must be greater than zero.")
            return None
        if sigma_y < 0.0 or Et < 0.0:
            self.set_error("Yield strength and tangent modulus cannot be negative.")
            return None
        if sigma_y > 0.0 and Et >= E:
            self.set_error("Tangent modulus must be smaller than Young's modulus.")
            return None

        out = {
            'name': str(preset or 'Custom'),
            'E': E,
            'nu': nu,
            'rho': rho,
            'thermal_conductivity': thermal_conductivity,
        }
        if sigma_y > 0.0:
            out['yield_strength']  = sigma_y
            out['tangent_modulus'] = Et

        # One visible output serves the selected study purpose. The hidden
        # impact ports preserve graphs saved before Material became
        # purpose-aware.
        impact = impact_material_payload(self, elastic=(E, nu, rho))
        if self.has_error():
            return None
        purpose = str(self.get_property('analysis_purpose') or 'Structural FEA')
        material_output = impact if purpose == 'Impact' else out
        return {
            'material': material_output,
            'impact_material': impact,
            'crash_material': impact,
        }

