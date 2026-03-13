# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash material node — elastic + plasticity properties for impact simulation."""
from pylcss.cad.core.base_node import CadQueryNode


# ─────────────────────────────────────────────────────────────────────────────
# Crash material presets (yield strength / tangent modulus added to base data)
# ─────────────────────────────────────────────────────────────────────────────

CRASH_MATERIAL_PRESETS = {
    # Preset name: {E [MPa], nu, rho [t/mm³], yield [MPa], H [MPa], eps_f}
    'Custom': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2100.0, 'failure_strain': 0.20,
    },
    'Steel (Structural A36)': {
        'E': 200000.0, 'nu': 0.29, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2000.0, 'failure_strain': 0.20,
    },
    'Steel (High-Strength DP780)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 480.0, 'tangent_modulus': 3000.0, 'failure_strain': 0.15,
    },
    # Alias used in legacy examples (same parameters, higher measured yield/hardening from tensile data)
    'DP780 Dual-Phase': {
        'E': 210000.0, 'nu': 0.28, 'rho': 7.83e-9,
        'yield_strength': 560.0, 'tangent_modulus': 1800.0, 'failure_strain': 0.22,
    },
    'Steel (Ultra-High UHSS 1500)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 1200.0, 'tangent_modulus': 4000.0, 'failure_strain': 0.08,
    },
    'Aluminum 6061-T6': {
        'E': 68900.0, 'nu': 0.33, 'rho': 2.70e-9,
        'yield_strength': 276.0, 'tangent_modulus': 690.0,  'failure_strain': 0.12,
    },
    'Aluminum 5052-H32 (Crush)': {
        'E': 70300.0, 'nu': 0.33, 'rho': 2.68e-9,
        'yield_strength': 193.0, 'tangent_modulus': 500.0,  'failure_strain': 0.14,
    },
    'CFRP (Quasi-Isotropic)': {
        'E': 70000.0, 'nu': 0.30, 'rho': 1.55e-9,
        'yield_strength': 600.0, 'tangent_modulus': 0.0,    'failure_strain': 0.015,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: CrashMaterialNode
# ─────────────────────────────────────────────────────────────────────────────

class CrashMaterialNode(CadQueryNode):
    """
    Material definition for crash / impact simulation.

    Extends the standard elastic material with plasticity parameters:
    - Yield strength (von Mises)
    - Isotropic hardening modulus (tangent slope after yield)
    - Failure / fracture strain (element deletion threshold)

    Presets cover common automotive and structural crash materials.
    """

    __identifier__ = 'com.cad.sim.crash_material'
    NODE_NAME = 'Crash Material'

    def __init__(self):
        super().__init__()
        self.add_output('crash_material', color=(255, 150, 50))

        self.create_property(
            'preset', 'Steel (Structural A36)',
            widget_type='combo',
            items=list(CRASH_MATERIAL_PRESETS.keys())
        )
        # ---------- elastic ----------
        self.create_property('youngs_modulus',  210000.0, widget_type='float')  # MPa
        self.create_property('poissons_ratio',  0.3,      widget_type='float')
        self.create_property('density',         7.85e-9,  widget_type='float')  # t/mm³
        # ---------- plasticity ----------
        self.create_property('yield_strength',  250.0,    widget_type='float')  # MPa
        self.create_property('tangent_modulus', 2000.0,   widget_type='float')  # MPa
        self.create_property('failure_strain',  0.20,     widget_type='float')  # m/m
        self.create_property('enable_fracture', True,     widget_type='checkbox')

    def run(self):
        preset = self.get_property('preset')
        if preset != 'Custom' and preset in CRASH_MATERIAL_PRESETS:
            p = CRASH_MATERIAL_PRESETS[preset]
            E   = p['E']
            nu  = p['nu']
            rho = p['rho']
            sy  = p['yield_strength']
            H   = p['tangent_modulus']
            ef  = p['failure_strain']
        else:
            E   = float(self.get_property('youngs_modulus'))
            nu  = float(self.get_property('poissons_ratio'))
            rho = float(self.get_property('density'))
            sy  = float(self.get_property('yield_strength'))
            H   = float(self.get_property('tangent_modulus'))
            ef  = float(self.get_property('failure_strain'))

        return {
            'E':              E,
            'nu':             nu,
            'rho':            rho,
            'yield_strength': sy,
            'tangent_modulus': H,
            'failure_strain': ef,
            'enable_fracture': bool(self.get_property('enable_fracture')),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: ImpactConditionNode
# ─────────────────────────────────────────────────────────────────────────────

