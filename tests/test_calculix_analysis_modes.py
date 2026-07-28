import json
from pathlib import Path

import pytest

from pylcss.design_studio.fem._helpers import MATERIAL_DATABASE
from pylcss.solver_backends.calculix_deck import _material_block, _step_header
from pylcss.solver_backends.common import SolverBackendError


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIAL = {
    "E": 210000.0,
    "nu": 0.3,
    "rho": 7.85e-9,
    "yield_strength": 250.0,
    "tangent_modulus": 1000.0,
}


def test_linear_study_keeps_yield_strength_as_allowable_only():
    material_lines = _material_block(MATERIAL, include_plasticity=False)

    assert "*PLASTIC, HARDENING=ISOTROPIC" not in material_lines
    assert _step_header("Linear") == ["*STEP", "*STATIC"]


def test_geometric_nonlinearity_does_not_enable_plasticity():
    material_lines = _material_block(MATERIAL, include_plasticity=False)
    step_lines = _step_header("Nonlinear (Geometric)")

    assert "*PLASTIC, HARDENING=ISOTROPIC" not in material_lines
    assert step_lines[0] == "*STEP, NLGEOM, INC=200"


def test_plastic_nonlinearity_emits_explicit_material_law():
    material_lines = _material_block(MATERIAL, include_plasticity=True)

    assert "*PLASTIC, HARDENING=ISOTROPIC" in material_lines
    plastic_index = material_lines.index("*PLASTIC, HARDENING=ISOTROPIC")
    assert material_lines[plastic_index + 1] == "250, 0.0"
    assert material_lines[plastic_index + 2] == "350, 0.1"


def test_plastic_nonlinearity_requires_positive_yield_strength():
    elastic_only = {**MATERIAL, "yield_strength": 0.0}

    with pytest.raises(SolverBackendError, match="yield strength"):
        _material_block(elastic_only, include_plasticity=True)


def test_every_material_preset_has_positive_thermal_conductivity():
    assert MATERIAL_DATABASE
    assert all(float(material["k"]) > 0.0 for material in MATERIAL_DATABASE.values())


def test_fea_solution_space_references_a_versioned_cad_example():
    model_path = (
        REPO_ROOT
        / "data"
        / "modeling_environment"
        / "FEA Plate Solution Space.json"
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    serialized = json.dumps(model)
    relative_cad_path = (
        "data/cad_environment/01_fea/03_fea_plate_solution_space.cad"
    )

    assert relative_cad_path in serialized
    assert (REPO_ROOT / relative_cad_path).is_file()
