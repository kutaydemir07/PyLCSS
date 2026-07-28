from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pylcss.design_studio.crash.provenance import (
    build_crash_provenance,
    write_crash_manifest,
)
from pylcss.design_studio.crash.materials import validate_material_dossier
from pylcss.design_studio.crash.quality import evaluate_crash_quality
from pylcss.design_studio.crash.signals import (
    build_crash_measurements,
    cfc_filter,
)
from pylcss.design_studio.crash.validation import (
    assess_convergence,
    assess_repeatability,
    compare_time_histories,
    correlate_crash_benchmark,
    summarize_solver_quality,
)
from pylcss.solver_backends.openradioss import (
    _build_animation_frames_with_mesh,
    _build_engine_deck,
    _keyword_card,
)
from pylcss.solver_backends.radioss_reader import _vtk_point_data
from pylcss.solver_backends.radioss_time_history import parse_time_history_csv


def _qualified_synthetic_measurements():
    time_ms = np.linspace(0.0, 100.0, 1001)
    force_kn = np.full(time_ms.size, 10.0)
    acceleration = np.full(time_ms.size, 100.0)
    speed = np.maximum(10.0 - acceleration * time_ms * 1.0e-3, 0.0)
    kinetic = 0.5 * 100.0 * speed**2 / 1000.0
    internal = 5.0 - kinetic
    history = {
        "time_ms": time_ms.tolist(),
        # One tonne-mm-ms force unit is 1000 kN.
        "rigid_wall_force_x_raw": (force_kn / 1000.0).tolist(),
        "rigid_wall_force_y_raw": np.zeros(time_ms.size).tolist(),
        "rigid_wall_force_z_raw": np.zeros(time_ms.size).tolist(),
        "kinetic_energy_kj": kinetic.tolist(),
        "internal_energy_kj": internal.tolist(),
        "rotational_kinetic_energy_kj": np.zeros(time_ms.size).tolist(),
        "contact_energy_kj": np.zeros(time_ms.size).tolist(),
        "hourglass_energy_kj": np.zeros(time_ms.size).tolist(),
        "external_work_kj": np.zeros(time_ms.size).tolist(),
        "delta_total_energy_relative": np.zeros(time_ms.size).tolist(),
        "mass_tonne": np.full(time_ms.size, 0.1).tolist(),
        "timestep_ms": np.full(time_ms.size, 1.0e-4).tolist(),
    }
    measurement = {
        "scenario": "fixed_specimen_moving_impactor",
        "impact_axis": [1.0, 0.0, 0.0],
        "initial_speed_m_s": 10.0,
        "impactor_mass_kg": 100.0,
        "structural_mass_kg": 10.0,
        "reference_node_ids": [],
        "support_node_ids": [],
        "material_validation": {
            "status": "pass",
            "reason": "Synthetic test fixture with exact constitutive response.",
        },
        "solver_diagnostics": {
            "starter": {
                "available": True,
                "normal_termination": True,
                "error_count": 0,
                "warning_count": 0,
            },
            "engine": {
                "available": True,
                "normal_termination": True,
                "error_count": 0,
                "warning_count": 0,
            },
        },
    }
    time_s = time_ms * 1.0e-3
    wall_displacement = (
        10.0 * time_s - 0.5 * 100.0 * time_s**2
    ) * 1000.0
    frames = [
        {
            "time": float(t),
            "displacement": np.zeros(3),
            "velocity": np.zeros((1, 3)),
            "acceleration": np.zeros((1, 3)),
            "rigid_wall_reference": {
                "node_id": 2,
                "displacement": [float(x), 0.0, 0.0],
                "velocity": [float(v), 0.0, 0.0],
                "acceleration": [-0.1, 0.0, 0.0],
            },
        }
        for t, x, v in zip(time_ms, wall_displacement, speed)
    ]
    return build_crash_measurements(
        history,
        frames=frames,
        measurement=measurement,
        acceleration_cfc=60,
        force_cfc=600,
    )


def test_engine_deck_separates_animation_and_high_rate_history():
    deck = _build_engine_deck(
        "qualification",
        end_time=100.0,
        output_dt=2.0,
        history_dt=0.1,
    )
    assert "/ANIM/DT\n0.  2" in deck
    assert "/TH/TITLE\n/TFILE\n0.1" in deck
    assert "/ANIM/VECT/ACC" in deck
    assert "/DT/NODA/STOP/0\n0.9  0.0" in deck


def test_engine_deck_distinguishes_mass_scaling_from_temporal_refinement():
    refined = _build_engine_deck(
        "qualification",
        end_time=6.0,
        output_dt=0.1,
        time_step_scale=0.5,
    )
    assert "/DT/NODA/STOP/0\n0.5  0.0" in refined
    assert "/DT/NODA/CST" not in refined

    scaled = _build_engine_deck(
        "qualification",
        end_time=6.0,
        output_dt=0.1,
        mass_scaling_dt=0.00015,
        mass_scaling_scale=0.67,
        time_step_scale=0.5,
    )
    assert "/DT/NODA/CST/0\n0.67  0.00015" in scaled
    assert "/DT/NODA/STOP" not in scaled


def test_keyword_card_keeps_rigid_wall_geometry_inside_80_columns():
    card = _keyword_card(
        90.6680239517,
        -0.54668740678,
        -0.77956220339,
        89.6680239517,
        -0.54668740678,
        -0.77956220339,
        0.08,
    )
    assert len(card) == 70
    assert all(
        len(card[index : index + 10]) == 10
        for index in range(0, len(card), 10)
    )
    assert float(card[20:30]) == pytest.approx(-0.779562)
    assert float(card[50:60]) == pytest.approx(-0.779562)
    assert float(card[60:70]) == pytest.approx(0.08)


def test_time_history_csv_maps_global_and_rigid_wall_channels(tmp_path):
    csv_path = tmp_path / "runT01.csv"
    csv_path.write_text(
        "time,INTERNAL ENERGY,KINETIC ENERGY,HOURGLASS ENERGY,"
        "CONTACT ENERGY,GLOBAL MASS,TIME STEP,"
        "NORMAL FORCE X - RWALL 1,NORMAL FORCE Y - RWALL 1,"
        "NORMAL FORCE Z - RWALL 1\n"
        "0.0,0.0,5.0,0.0,0.0,0.1,0.001,0.01,0.0,0.0\n"
        "0.1,1.0,4.0,0.01,0.02,0.1,0.001,0.02,0.0,0.0\n",
        encoding="utf-8",
    )
    parsed = parse_time_history_csv(csv_path)
    assert parsed["time_ms"] == [0.0, 0.1]
    assert parsed["internal_energy_kj"] == [0.0, 1.0]
    assert parsed["kinetic_energy_kj"] == [5.0, 4.0]
    assert parsed["mass_tonne"] == [0.1, 0.1]
    assert parsed["rigid_wall_impulse_x_raw"] == [0.01, 0.02]


def test_time_history_does_not_mislabel_interface_force_as_wall_force(
    tmp_path,
):
    csv_path = tmp_path / "runT01.csv"
    csv_path.write_text(
        "time,TH-INTER 1 CONTACT FNX,TH-INTER 1 CONTACT FNY,"
        "TH-INTER 1 CONTACT FNZ\n"
        "0.0,0.01,0.0,0.0\n"
        "0.1,0.02,0.0,0.0\n",
        encoding="utf-8",
    )
    parsed = parse_time_history_csv(csv_path)
    assert "rigid_wall_impulse_x_raw" not in parsed


def test_measurement_contract_computes_crashworthiness_metrics():
    result = _qualified_synthetic_measurements()
    metrics = result["metrics"]
    assert metrics["peak_crushing_force_kN"] == pytest.approx(10.0, rel=1e-6)
    assert metrics["mean_crushing_force_kN"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["crush_force_efficiency"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["force_displacement_energy_kJ"] == pytest.approx(5.0, rel=1e-3)
    assert metrics["useful_crush_stroke_mm"] == pytest.approx(500.0, rel=1e-3)
    assert metrics["delta_v_m_s"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["force_impulse_N_s"] == pytest.approx(1000.0, rel=1e-3)
    assert result["processing"]["raw_preserved"] is True


def test_measurement_contract_differentiates_tfile_rigid_wall_impulse():
    time_ms = np.linspace(0.0, 10.0, 1001)
    # 10 kN = 0.01 tonne*mm/ms², so TFILE impulse grows by 0.01 per ms.
    history = {
        "time_ms": time_ms.tolist(),
        "rigid_wall_impulse_x_raw": (0.01 * time_ms).tolist(),
        "rigid_wall_impulse_y_raw": np.zeros(time_ms.size).tolist(),
        "rigid_wall_impulse_z_raw": np.zeros(time_ms.size).tolist(),
    }
    measurement = {
        "scenario": "fixed_specimen_moving_impactor",
        "impact_axis": [1.0, 0.0, 0.0],
        "initial_speed_m_s": 10.0,
        "impactor_mass_kg": 50.0,
    }
    result = build_crash_measurements(
        history,
        frames=[],
        measurement=measurement,
        force_cfc=600,
    )
    assert np.asarray(result["raw"]["rigid_wall_force_kN"]) == pytest.approx(
        10.0
    )
    assert result["metrics"]["force_impulse_N_s"] == pytest.approx(100.0)


def test_quality_gate_passes_consistent_solver_history():
    measurements = _qualified_synthetic_measurements()
    quality = evaluate_crash_quality(
        measurements,
        external_status="engine_completed",
        end_time_ms=100.0,
    )
    assert quality["status"] == "pass"
    assert quality["ml_eligible"] is True
    assert quality["failed_checks"] == []


def test_quality_gate_rejects_missing_force_and_excess_added_mass():
    measurements = _qualified_synthetic_measurements()
    measurements["processed"]["rigid_wall_force_kN"] = [0.0] * 1001
    measurements["raw"]["mass_kg"] = np.linspace(100.0, 105.0, 1001).tolist()
    quality = evaluate_crash_quality(
        measurements,
        external_status="engine_completed",
        end_time_ms=100.0,
    )
    assert quality["status"] == "fail"
    assert "rigid_wall_force_channel" in quality["failed_checks"]
    assert "added_mass" in quality["failed_checks"]
    assert quality["ml_eligible"] is False


def test_quality_gate_rejects_unvalidated_material_preset():
    measurements = _qualified_synthetic_measurements()
    measurements["measurement"]["material_validation"] = {
        "status": "fail",
        "reason": "Reference preset only.",
    }
    quality = evaluate_crash_quality(
        measurements,
        external_status="engine_completed",
        end_time_ms=100.0,
    )
    assert "material_validation" in quality["failed_checks"]
    assert quality["ml_eligible"] is False


def test_material_dossier_requires_matching_traceable_evidence(tmp_path):
    coupon_report = tmp_path / "coupon_report.pdf"
    coupon_report.write_bytes(b"controlled coupon evidence")
    dossier = tmp_path / "material_validation.json"
    dossier.write_text(
        json.dumps(
            {
                "material_id": "Aluminum 5052-H32",
                "status": "pass",
                "test_standard": "ISO 6892-1 + dynamic protocol",
                "lot_id": "LOT-42",
                "thickness_mm": 2.0,
                "rate_range_per_s": [0.001, 1000.0],
                "temperature_c": 23.0,
                "curve_source": "controlled true stress-plastic strain curves",
                "true_stress_plastic_strain_verified": True,
                "strain_rate_model_verified": True,
                "failure_model_verified": True,
                "coupon_report": coupon_report.name,
                "approved_by": "Test engineer",
                "approval_date": "2026-07-24",
            }
        ),
        encoding="utf-8",
    )
    validation = validate_material_dossier(
        dossier,
        expected_lot_id="LOT-42",
        configured_rate_min=0.01,
        configured_rate_max=500.0,
        rate_model_required=True,
        failure_model_required=True,
    )
    assert validation["status"] == "pass"
    assert validation["validation_report_sha256"]
    assert validation["coupon_report_sha256"]

    mismatch = validate_material_dossier(
        dossier,
        expected_lot_id="OTHER-LOT",
        configured_rate_min=0.01,
        configured_rate_max=500.0,
        rate_model_required=True,
        failure_model_required=True,
    )
    assert mismatch["status"] == "fail"
    assert mismatch["evidence"]["lot_matches"] is False


def test_cfc_filter_reduces_high_frequency_noise_without_phase_shift():
    time_ms = np.linspace(0.0, 100.0, 5001)
    low = np.sin(2.0 * np.pi * 20.0 * time_ms * 1.0e-3)
    high = 0.4 * np.sin(2.0 * np.pi * 1500.0 * time_ms * 1.0e-3)
    filtered, metadata = cfc_filter(time_ms, low + high, 60)
    assert metadata["applied"] is True
    assert np.sqrt(np.mean((filtered - low) ** 2)) < 0.05
    first_period = time_ms <= 50.0
    assert abs(
        np.argmax(filtered[first_period]) - np.argmax(low[first_period])
    ) < 5


def test_animation_reader_extracts_acceleration_vector():
    mesh = SimpleNamespace(
        point_data={
            "Displacement": np.zeros((2, 3)),
            "Velocity": np.ones((2, 3)),
            "Acceleration": np.full((2, 3), 2.0),
        },
        cell_data={},
        cell_data_dict={},
        cells=[],
        points=np.zeros((2, 3)),
    )
    values = _vtk_point_data(mesh)
    assert np.allclose(values[3], 1.0)
    assert np.allclose(values[4], 2.0)


def test_animation_wrapper_preserves_native_rigid_wall_reference():
    source_mesh = SimpleNamespace(
        p=np.zeros((3, 2)),
        t=np.zeros((3, 1), dtype=int),
    )
    raw = {
        "node_ids": np.array([1, 2, 3]),
        "displacement": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0]
        ),
        "velocity": np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        ),
        "acceleration": np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]]
        ),
        "stress_vm": np.zeros(3),
        "time": 1.0,
    }
    frame = _build_animation_frames_with_mesh(source_mesh, [raw])[0]
    wall = frame["rigid_wall_reference"]
    assert wall["node_id"] == 3
    assert wall["displacement"] == [12.0, 0.0, 0.0]
    assert wall["velocity"] == [4.0, 0.0, 0.0]


def test_validation_metrics_detect_convergence_and_repeatability():
    cases = [
        {"case_id": "coarse", "metrics": {"peak_crushing_force_kN": 105.0}},
        {"case_id": "fine", "metrics": {"peak_crushing_force_kN": 100.0}},
    ]
    convergence = assess_convergence(
        cases,
        metrics=("peak_crushing_force_kN",),
    )
    assert convergence["status"] == "pass"
    repeatability = assess_repeatability(
        [
            {"peak_crushing_force_kN": 100.0},
            {"peak_crushing_force_kN": 100.5},
            {"peak_crushing_force_kN": 99.5},
        ],
        metrics=("peak_crushing_force_kN",),
    )
    assert repeatability["status"] == "pass"


def test_solver_quality_summary_separates_numerical_and_material_gates():
    summary = summarize_solver_quality(
        [
            {
                "case_id": "run_1",
                "external_status": "engine_completed",
                "quality_status": "fail",
                "quality": {
                    "numerical_status": "pass",
                    "failed_checks": ["material_validation"],
                    "warning_checks": [],
                },
                "provenance": {"run_id": "abc"},
            }
        ]
    )
    assert summary["status"] == "pass"
    assert summary["cases"][0]["failed_checks"] == []


def test_curve_correlation_scores_identical_histories_as_pass():
    time_ms = np.linspace(0.0, 20.0, 201)
    values = np.sin(np.pi * time_ms / 20.0)
    comparison = compare_time_histories(
        time_ms,
        values,
        time_ms,
        values,
    )
    assert comparison["status"] == "pass"
    assert comparison["engineering_score"] == pytest.approx(1.0)
    assert comparison["normalized_rmse"] == pytest.approx(0.0)


def test_physical_correlation_requires_traceable_matching_test_metadata():
    simulation = _qualified_synthetic_measurements()
    processed = simulation["processed"]
    benchmark = {
        "benchmark_id": "TEST-42",
        "time_ms": processed["time_ms"],
        "force_kN": processed["rigid_wall_force_kN"],
        "acceleration_g": processed["acceleration_g"],
        "displacement_mm": processed["crush_displacement_mm"],
        "force_displacement_force_kN": processed["rigid_wall_force_kN"],
        "traceability": {
            "status": "pass",
            "benchmark_id": "TEST-42",
            "specimen_id": "SPEC-1",
            "test_date": "2026-07-25",
            "test_laboratory": "Controlled lab",
            "geometry_revision": "REV-A",
            "material_lot_id": "LOT-42",
            "thickness_mm": 1.5,
            "test_replicate_count": 3,
            "boundary_condition": "fixed rear, moving 100 kg impactor",
            "impact_velocity_m_s": 10.0,
            "impactor_mass_kg": 100.0,
            "trigger_definition": "first force threshold crossing",
            "time_zero_definition": "trigger time",
            "source_document": "controlled_test_report.pdf",
            "source_document_sha256": "a" * 64,
            "csv_sha256": "b" * 64,
            "metadata_sha256": "c" * 64,
            "license_or_permission": "project-controlled evidence",
            "approved_by": "Validation engineer",
            "approval_date": "2026-07-25",
            "force_channel": {
                "sensor_id": "LC-1",
                "calibration_id": "CAL-F-1",
                "sample_rate_hz": 20000,
                "filter_cfc": 600,
                "positive_convention": "compression_positive",
            },
            "acceleration_channel": {
                "sensor_id": "ACC-1",
                "location": "impactor CG",
                "axis": "global X",
                "calibration_id": "CAL-A-1",
                "sample_rate_hz": 20000,
                "filter_cfc": 60,
                "positive_convention": "deceleration_positive",
            },
            "displacement_channel": {
                "sensor_id": "LVDT-1",
                "calibration_id": "CAL-D-1",
                "sample_rate_hz": 20000,
                "positive_convention": "crush_positive",
            },
        },
    }
    result = correlate_crash_benchmark(simulation, benchmark)
    assert result["status"] == "pass"
    assert result["metadata_validation"]["status"] == "pass"

    benchmark["traceability"]["status"] = "fail"
    rejected = correlate_crash_benchmark(simulation, benchmark)
    assert rejected["status"] == "fail"
    assert rejected["metadata_validation"]["status"] == "fail"


def test_provenance_fingerprint_is_stable_and_manifest_is_auditable(tmp_path):
    mesh = SimpleNamespace(
        p=np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        t=np.array([[0], [1], [1]]),
        shell_thickness=1.5,
        shell_nip=5,
    )
    deck = tmp_path / "run.k"
    engine = tmp_path / "run_0001.rad"
    deck.write_text("*KEYWORD\n*END\n", encoding="utf-8")
    engine.write_text("#RADIOSS ENGINE INPUT\n", encoding="utf-8")
    kwargs = dict(
        mesh=mesh,
        material={"E": 70000.0},
        impact={"velocity": [-10.0, 0.0, 0.0]},
        constraints=[],
        solver_settings={"end_time_ms": 6.0},
        deck_path=deck,
        engine_path=engine,
        starter_executable=None,
        engine_executable=None,
        time_history_converter=None,
    )
    first = build_crash_provenance(**kwargs)
    second = build_crash_provenance(**kwargs)
    assert first["run_id"] == second["run_id"]
    manifest = write_crash_manifest(
        tmp_path,
        provenance=first,
        quality={"status": "pass"},
        metrics={"absorbed_energy_kJ": 5.0},
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["provenance"]["run_id"] == first["run_id"]
    assert payload["quality"]["status"] == "pass"
