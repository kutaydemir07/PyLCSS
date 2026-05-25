# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash material node — elastic + plasticity properties for impact simulation."""
from pylcss.design_studio.core.base_node import CadQueryNode


# ─────────────────────────────────────────────────────────────────────────────
# Crash material presets (yield strength / tangent modulus added to base data)
# ─────────────────────────────────────────────────────────────────────────────

CRASH_MATERIAL_PRESETS = {
    # Preset name: {E [MPa], nu, rho [t/mm³], yield [MPa], H [MPa], eps_f,
    #               strain_rate_c [1/s], strain_rate_p}
    # Cowper-Symonds rate hardening: σ_y(ε̇) = σ_y0 · (1 + (ε̇/C)^(1/p)).
    # Defaults are the classical mild-steel/aluminum values from Jones (1989);
    # composites use C=0 (rate-insensitive).
    'Custom': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2100.0, 'failure_strain': 0.20,
        'strain_rate_c': 40.0, 'strain_rate_p': 5.0,
    },
    'Steel (Structural A36)': {
        'E': 200000.0, 'nu': 0.29, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2000.0, 'failure_strain': 0.20,
        'strain_rate_c': 40.0, 'strain_rate_p': 5.0,
    },
    'Steel (High-Strength DP780)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 480.0, 'tangent_modulus': 3000.0, 'failure_strain': 0.15,
        'strain_rate_c': 200.0, 'strain_rate_p': 5.0,
    },
    # Alias used in legacy examples (same parameters, higher measured yield/hardening from tensile data)
    'DP780 Dual-Phase': {
        'E': 210000.0, 'nu': 0.28, 'rho': 7.83e-9,
        'yield_strength': 560.0, 'tangent_modulus': 1800.0, 'failure_strain': 0.22,
        'strain_rate_c': 200.0, 'strain_rate_p': 5.0,
    },
    'Steel (Ultra-High UHSS 1500)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 1200.0, 'tangent_modulus': 4000.0, 'failure_strain': 0.08,
        'strain_rate_c': 800.0, 'strain_rate_p': 5.0,
    },
    'Aluminum 6061-T6': {
        'E': 68900.0, 'nu': 0.33, 'rho': 2.70e-9,
        'yield_strength': 276.0, 'tangent_modulus': 690.0,  'failure_strain': 0.12,
        'strain_rate_c': 6500.0, 'strain_rate_p': 4.0,
    },
    'Aluminum 5052-H32 (Crush)': {
        'E': 70300.0, 'nu': 0.33, 'rho': 2.68e-9,
        'yield_strength': 193.0, 'tangent_modulus': 500.0,  'failure_strain': 0.14,
        'strain_rate_c': 6500.0, 'strain_rate_p': 4.0,
    },
    # CFRP is treated here as a rate-insensitive elastic + brittle-failure
    # proxy — composites do not yield like metals; for production crash work
    # use an orthotropic damage law (LAW25 Tsai-Wu) on a shell section.
    'CFRP (Quasi-Isotropic, proxy)': {
        'E': 70000.0, 'nu': 0.30, 'rho': 1.55e-9,
        'yield_strength': 600.0, 'tangent_modulus': 0.0,    'failure_strain': 0.015,
        'strain_rate_c': 0.0, 'strain_rate_p': 0.0,
    },
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return bool(value)


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
        # ---------- strain-rate sensitivity ----------
        # Engineering-facing switch only. Cowper-Symonds constants are selected
        # internally from the material preset.
        self.create_property('strain_rate_sensitive', True, widget_type='checkbox')

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
            if _as_bool(self.get_property('strain_rate_sensitive')):
                src = float(p.get('strain_rate_c', 0.0) or 0.0)
                srp = float(p.get('strain_rate_p', 0.0) or 0.0)
            else:
                src = 0.0
                srp = 0.0
        else:
            E   = float(self.get_property('youngs_modulus'))
            nu  = float(self.get_property('poissons_ratio'))
            rho = float(self.get_property('density'))
            sy  = float(self.get_property('yield_strength'))
            H   = float(self.get_property('tangent_modulus'))
            ef  = float(self.get_property('failure_strain'))
            if _as_bool(self.get_property('strain_rate_sensitive')):
                p = CRASH_MATERIAL_PRESETS['Custom']
                src = float(p.get('strain_rate_c', 0.0) or 0.0)
                srp = float(p.get('strain_rate_p', 0.0) or 0.0)
            else:
                src = 0.0
                srp = 0.0

        return {
            'E':              E,
            'nu':             nu,
            'rho':            rho,
            'yield_strength': sy,
            'tangent_modulus': H,
            'failure_strain': ef,
            'enable_fracture': _as_bool(self.get_property('enable_fracture')),
            # Cowper-Symonds (consumed by the OpenRadioss deck writer; 0 = off).
            'strain_rate_c': src,
            'strain_rate_p': srp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: ImpactConditionNode
# ─────────────────────────────────────────────────────────────────────────────

