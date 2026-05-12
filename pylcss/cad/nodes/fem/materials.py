# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM material node — elastic material properties with preset database."""
from pylcss.cad.core.base_node import CadQueryNode
from pylcss.cad.nodes.fem._helpers import MATERIAL_DATABASE

class MaterialNode(CadQueryNode):
    """Defines material properties with preset database."""
    __identifier__ = 'com.cad.sim.material'
    NODE_NAME = 'Material'

    def __init__(self):
        super().__init__()
        self.add_output('material', color=(200, 200, 200))
        
        # Add Inputs for parametric material properties
        self.add_input('youngs_modulus', color=(180, 180, 0))
        self.add_input('poissons_ratio', color=(180, 180, 0))
        self.add_input('density', color=(180, 180, 0))
        
        # Preset dropdown
        self.create_property('preset', 'Steel (Structural)', widget_type='combo',
                             items=list(MATERIAL_DATABASE.keys()))
        
        # Keep properties as defaults (editable for Custom)
        self.create_property('youngs_modulus', 210000.0, widget_type='float')  # MPa
        self.create_property('poissons_ratio', 0.3, widget_type='float')
        self.create_property('density', 7.85e-9, widget_type='float')  # tonne/mm^3

        # Optional plasticity — pure elastic when yield_strength == 0.
        # When > 0 the CalculiX backend writes a *PLASTIC card (isotropic
        # bilinear hardening) and CCX automatically enables NLGEOM, giving
        # a nonlinear material+geometric static solve.
        self.create_property('yield_strength',  0.0, widget_type='float')   # MPa
        self.create_property('tangent_modulus', 0.0, widget_type='float')   # MPa

    def run(self):
        # Check if using preset or custom
        preset = self.get_property('preset')

        if preset != 'Custom' and preset in MATERIAL_DATABASE:
            mat = MATERIAL_DATABASE[preset]
            E = mat['E']
            nu = mat['nu']
            rho = mat['rho']
        else:
            # Resolve inputs with fallback to properties
            E = self.get_input_value('youngs_modulus', 'youngs_modulus')
            nu = self.get_input_value('poissons_ratio', 'poissons_ratio')
            rho = self.get_input_value('density', 'density')

        # Plasticity is independent of the preset choice — surface as
        # explicit overrides so a user can mix Steel (preset) + custom
        # yield strength without typing all the elastic fields.
        sigma_y = float(self.get_property('yield_strength')  or 0.0)
        Et      = float(self.get_property('tangent_modulus') or 0.0)

        out = {
            'E':   float(E),
            'nu':  float(nu),
            'rho': float(rho),
        }
        if sigma_y > 0.0:
            out['yield_strength']  = sigma_y
            out['tangent_modulus'] = Et
        return out

