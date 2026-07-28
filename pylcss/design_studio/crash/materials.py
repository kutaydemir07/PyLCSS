# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash material node — elastic + plasticity properties for impact simulation."""
import hashlib
import json
import math
from pathlib import Path

from pylcss.design_studio.core.base_node import CadQueryNode


# ─────────────────────────────────────────────────────────────────────────────
# Crash material presets (yield strength / tangent modulus added to base data)
# ─────────────────────────────────────────────────────────────────────────────

CRASH_MATERIAL_PRESETS = {
    # These are numerical starting values for workflow examples, not certified
    # material cards. Production crash models require coupon/rate calibration
    # for the actual alloy, temper, thickness, forming history, and failure law.
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
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return bool(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_material_dossier(
    report_path,
    *,
    expected_lot_id,
    configured_rate_min,
    configured_rate_max,
    rate_model_required,
    failure_model_required,
):
    """Validate a traceable JSON sidecar and its referenced coupon report."""
    source = Path(report_path) if report_path else None
    result = {
        'status': 'fail',
        'reason': 'A traceable JSON material validation dossier is required.',
        'validation_report_path': str(source) if source else '',
    }
    if source is None or not source.is_file():
        result['reason'] = 'The material validation dossier file does not exist.'
        return result
    result['validation_report_path'] = str(source.resolve())
    result['validation_report_sha256'] = _sha256(source)
    try:
        payload = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result['reason'] = (
            f'The material validation dossier is not valid JSON: {exc}'
        )
        return result

    required = {
        'material_id',
        'test_standard',
        'lot_id',
        'thickness_mm',
        'rate_range_per_s',
        'temperature_c',
        'curve_source',
        'true_stress_plastic_strain_verified',
        'strain_rate_model_verified',
        'failure_model_verified',
        'coupon_report',
        'approved_by',
        'approval_date',
        'status',
    }
    missing = sorted(required - set(payload))
    if missing:
        result['reason'] = (
            'Material dossier fields missing: ' + ', '.join(missing)
        )
        return result

    placeholders = {'', 'required', 'replace_with_test_id', 'none', 'null'}

    def meaningful(name):
        return (
            str(payload.get(name, '')).strip().lower() not in placeholders
        )

    try:
        rate_range = [float(value) for value in payload['rate_range_per_s']]
        thickness = float(payload['thickness_mm'])
        temperature = float(payload['temperature_c'])
    except (TypeError, ValueError):
        result['reason'] = (
            'Material thickness, temperature, and rate range must be numeric.'
        )
        return result
    if len(rate_range) != 2 or not all(
        math.isfinite(value) for value in rate_range
    ):
        result['reason'] = (
            'rate_range_per_s must contain two finite values.'
        )
        return result

    coupon_value = str(payload.get('coupon_report') or '').strip()
    coupon_path = Path(coupon_value).expanduser()
    if not coupon_path.is_absolute():
        coupon_path = source.parent / coupon_path
    evidence = {
        'declared_pass': (
            str(payload.get('status')).strip().lower() == 'pass'
        ),
        'identity_complete': all(
            meaningful(name)
            for name in (
                'material_id',
                'test_standard',
                'curve_source',
                'approved_by',
                'approval_date',
            )
        ),
        'lot_matches': (
            str(payload.get('lot_id')).strip()
            == str(expected_lot_id).strip()
        ),
        'physical_conditions_valid': (
            math.isfinite(thickness)
            and thickness > 0.0
            and math.isfinite(temperature)
        ),
        'rate_range_covers_model': (
            rate_range[0] >= 0.0
            and rate_range[1] > rate_range[0]
            and rate_range[0] <= float(configured_rate_min)
            and rate_range[1] >= float(configured_rate_max)
        ),
        'true_curve_verified': (
            payload.get('true_stress_plastic_strain_verified') is True
        ),
        'rate_model_verified': (
            not rate_model_required
            or payload.get('strain_rate_model_verified') is True
        ),
        'failure_model_verified': (
            not failure_model_required
            or payload.get('failure_model_verified') is True
        ),
        'coupon_report_exists': coupon_path.is_file(),
    }
    result.update(
        {
            'material_id': payload.get('material_id'),
            'material_lot_id': payload.get('lot_id'),
            'test_standard': payload.get('test_standard'),
            'validated_rate_range_per_s': rate_range,
            'temperature_c': temperature,
            'thickness_mm': thickness,
            'coupon_report_path': (
                str(coupon_path.resolve())
                if coupon_path.is_file()
                else coupon_value
            ),
            'coupon_report_sha256': (
                _sha256(coupon_path) if coupon_path.is_file() else None
            ),
            'approved_by': payload.get('approved_by'),
            'approval_date': payload.get('approval_date'),
            'evidence': evidence,
        }
    )
    failed = [name for name, passed in evidence.items() if not passed]
    if failed:
        result['reason'] = (
            'Material dossier evidence failed: ' + ', '.join(failed)
        )
        return result
    result['status'] = 'pass'
    result['reason'] = (
        'Traceable material-lot validation evidence verified.'
    )
    return result


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
        # Element deletion is unsafe without a calibrated failure model, so it
        # is deliberately opt-in.
        self.create_property('enable_fracture', False,    widget_type='checkbox')
        # ---------- strain-rate sensitivity ----------
        # Engineering-facing switch only. Cowper-Symonds constants are selected
        # internally from the material preset.
        self.create_property('strain_rate_sensitive', True, widget_type='checkbox')
        # Traceability is deliberately separate from the material preset.
        # Preset values are engineering starting points, not a certification
        # that the exact sheet lot/thickness/rate range has been characterised.
        self.create_property(
            'validation_status', 'Unvalidated',
            widget_type='combo',
            items=['Unvalidated', 'Validated for this lot and rate range'],
        )
        self.create_property('material_lot_id', '', widget_type='text')
        self.create_property('validation_report_path', '', widget_type='text')
        self.create_property('validated_rate_min_per_s', 0.0, widget_type='float')
        self.create_property('validated_rate_max_per_s', 0.0, widget_type='float')

    def run(self):
        self.clear_error()
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
            try:
                E   = float(self.get_property('youngs_modulus'))
                nu  = float(self.get_property('poissons_ratio'))
                rho = float(self.get_property('density'))
                sy  = float(self.get_property('yield_strength'))
                H   = float(self.get_property('tangent_modulus'))
                ef  = float(self.get_property('failure_strain'))
            except (TypeError, ValueError):
                self.set_error("Crash material properties must be numeric.")
                return None
            if _as_bool(self.get_property('strain_rate_sensitive')):
                p = CRASH_MATERIAL_PRESETS['Custom']
                src = float(p.get('strain_rate_c', 0.0) or 0.0)
                srp = float(p.get('strain_rate_p', 0.0) or 0.0)
            else:
                src = 0.0
                srp = 0.0

        values = (E, nu, rho, sy, H, ef, src, srp)
        if not all(math.isfinite(float(value)) for value in values):
            self.set_error("Crash material properties must be finite numbers.")
            return None
        if E <= 0.0:
            self.set_error("Young's modulus must be greater than zero.")
            return None
        if not (-1.0 < nu < 0.5):
            self.set_error("Poisson's ratio must be between -1 and 0.5 (exclusive).")
            return None
        if rho <= 0.0 or sy <= 0.0:
            self.set_error("Density and yield strength must be greater than zero.")
            return None
        if H < 0.0 or H >= E:
            self.set_error("Tangent modulus must be non-negative and smaller than Young's modulus.")
            return None
        if _as_bool(self.get_property('enable_fracture')) and ef <= 0.0:
            self.set_error("Failure strain must be greater than zero when fracture is enabled.")
            return None
        if src < 0.0 or srp < 0.0:
            self.set_error("Strain-rate constants cannot be negative.")
            return None

        validation_label = str(self.get_property('validation_status') or 'Unvalidated')
        lot_id = str(self.get_property('material_lot_id') or '').strip()
        report_value = str(self.get_property('validation_report_path') or '').strip()
        report_path = Path(report_value).expanduser() if report_value else None
        project_dir = getattr(self, '_project_dir', None)
        if report_path is not None and not report_path.is_absolute() and project_dir:
            report_path = Path(project_dir) / report_path
        try:
            rate_min = float(self.get_property('validated_rate_min_per_s') or 0.0)
            rate_max = float(self.get_property('validated_rate_max_per_s') or 0.0)
        except (TypeError, ValueError):
            rate_min = rate_max = 0.0
        validation_requested = validation_label.lower().startswith('validated')
        validation_inputs_complete = bool(
            validation_requested
            and lot_id
            and rate_min >= 0.0
            and rate_max > rate_min
        )
        if validation_inputs_complete:
            validation = validate_material_dossier(
                report_path,
                expected_lot_id=lot_id,
                configured_rate_min=rate_min,
                configured_rate_max=rate_max,
                rate_model_required=src > 0.0 and srp > 0.0,
                failure_model_required=_as_bool(
                    self.get_property('enable_fracture')
                ),
            )
            validation['declared_status'] = validation_label
        else:
            validation = {
                'status': 'fail',
                'declared_status': validation_label,
                'material_lot_id': lot_id,
                'validation_report_path': report_value,
                'validated_rate_range_per_s': [rate_min, rate_max],
                'reason': (
                    'Select validated status, enter the matching material lot, '
                    'a positive covered rate range, and a traceable JSON dossier.'
                ),
            }

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
            'preset_name': preset,
            'validation': validation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: ImpactConditionNode
# ─────────────────────────────────────────────────────────────────────────────

