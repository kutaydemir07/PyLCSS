import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    REPO_ROOT
    / "experiments"
    / "active_learning"
    / "hard_nonlinear_fea_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pylcss_hard_nonlinear_fea_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_hard_nonlinear_design_is_seeded_independent_and_bounded():
    first = BENCHMARK.design_points(pool_size=8, test_size=4, seed=17)
    second = BENCHMARK.design_points(pool_size=8, test_size=4, seed=17)
    assert first == second
    assert [row["sample_id"] for row in first[:8]] == [
        f"pool_{index:03d}" for index in range(8)
    ]
    assert [row["sample_id"] for row in first[8:]] == [
        f"test_{index:03d}" for index in range(4)
    ]
    for row in first:
        for name, (lower, upper) in zip(BENCHMARK.PARAMETER_NAMES, BENCHMARK.BOUNDS):
            assert lower <= float(row[name]) <= upper


def test_shallow_arch_deck_is_real_incremented_nlgeom_beam_analysis():
    deck, apex_id, target = BENCHMARK.build_shallow_arch_deck(
        {
            "rise_mm": 10.0,
            "thickness_mm": 1.8,
            "displacement_ratio": 1.2,
            "imperfection_ratio": 0.01,
        },
        mesh_elements=20,
        max_increment=0.02,
    )
    assert "*ELEMENT, TYPE=B31" in deck
    assert "*STEP, NLGEOM" in deck
    assert "*NODE PRINT, NSET=APEX" in deck
    assert apex_id == 11
    assert target == 12.0


def test_two_bar_closed_form_has_expected_equilibrium_zeros():
    for ratio in (0.0, 1.0, 2.0):
        force = BENCHMARK.two_bar_analytical_force(
            rise_mm=10.0,
            area_mm2=10.0,
            displacement_mm=ratio * 10.0,
        )
        assert abs(force) < 1e-9


def test_reaction_parser_extracts_path_metrics(tmp_path):
    dat = tmp_path / "case.dat"
    dat.write_text(
        """
 displacements (vx,vy,vz) for set APEX and time  0.2000000E+00

         2  0.000000E+00 -2.000000E-01  0.000000E+00
 forces (fx,fy,fz) for set APEX and time  0.2000000E+00

         2  0.000000E+00 -1.000000E+01  0.000000E+00
 displacements (vx,vy,vz) for set APEX and time  0.5000000E+00

         2  0.000000E+00 -5.000000E-01  0.000000E+00
 forces (fx,fy,fz) for set APEX and time  0.5000000E+00

         2  0.000000E+00 -2.000000E+01  0.000000E+00
 displacements (vx,vy,vz) for set APEX and time  0.1000000E+01

         2  0.000000E+00 -1.000000E+00  0.000000E+00
 forces (fx,fy,fz) for set APEX and time  0.1000000E+01

         2  0.000000E+00 -5.000000E+00  0.000000E+00
""",
        encoding="utf-8",
    )
    result = BENCHMARK.parse_reaction_history(
        dat,
        apex_id=2,
        target_displacement_mm=1.0,
        rise_mm=1.0,
    )
    assert result["final_force_n"] == 5.0
    assert result["pre_peak_force_n"] == 20.0
    assert 0.49 <= result["pre_peak_displacement_mm"] <= 0.51
    assert result["minimum_tangent_n_per_mm"] < 0.0


def test_committee_replay_uses_same_initial_design_and_hides_pool_labels():
    rng = np.random.default_rng(12)
    X = np.column_stack(
        [
            rng.uniform(lower, upper, 40)
            for lower, upper in BENCHMARK.BOUNDS
        ]
    )
    y = np.column_stack(
        [
            np.sin(X[:, 0]),
            X[:, 1] ** 2,
            X[:, 0] * X[:, 2],
            X[:, 3] + X[:, 2],
        ]
    )
    visible_counts = []

    def acquisition(strategy, X_train, y_train, X_pool, **kwargs):
        assert strategy == "committee"
        assert len(X_train) == len(y_train)
        visible_counts.append(len(y_train))
        return SimpleNamespace(scores=np.linspace(0.0, 1.0, len(X_pool)))

    selected, _ = BENCHMARK.committee_replay_indices(
        X,
        y,
        seed=4,
        budget=20,
        acquisition_fn=acquisition,
    )
    static = BENCHMARK.farthest_point_order(
        BENCHMARK.normalize_to_unit(X, BENCHMARK.BOUNDS), seed=4
    )
    np.testing.assert_array_equal(
        selected[: BENCHMARK.ACTIVE_INITIAL],
        static[: BENCHMARK.ACTIVE_INITIAL],
    )
    assert visible_counts == [12, 16]
    assert len(set(selected.tolist())) == 20


def test_prediction_metrics_are_exact_for_exact_predictions():
    X = np.array(
        [
            [8.0, 1.2, 0.2, 0.0],
            [10.0, 1.8, 0.5, 0.01],
            [12.0, 2.2, 1.2, -0.01],
        ]
    )
    y = np.array(
        [
            [10.0, 20.0, 2.0, 5.0],
            [30.0, 40.0, 5.0, 15.0],
            [5.0, 50.0, 6.0, 25.0],
        ]
    )
    metrics = BENCHMARK.prediction_metrics(y, y.copy(), X, np.array([0, 1, 2]))
    assert metrics["aggregate_nrmse"] == 0.0
    assert metrics["aggregate_r2"] == 1.0
    assert metrics["transition_aggregate_nrmse"] == 0.0


def test_paired_statistics_counts_seed_as_the_inferential_unit():
    rows = []
    for seed, static, committee in ((0, 0.20, 0.18), (1, 0.22, 0.19), (2, 0.21, 0.20)):
        for sampling, error in (
            ("static_maximin", static),
            ("committee", committee),
        ):
            rows.append(
                {
                    "seed": seed,
                    "budget": 32,
                    "sampling": sampling,
                    "aggregate_nrmse": error,
                    "transition_aggregate_nrmse": error * 1.2,
                }
            )
    stats = BENCHMARK.paired_statistics(rows, bootstrap_samples=200, bootstrap_seed=1)
    assert len(stats) == 1
    assert stats[0]["seeds"] == 3
    assert stats[0]["committee_wins"] == 3
    assert stats[0]["paired_improvement_mean_pct"] > 0.0


def test_validation_gates_are_budget_specific_and_require_every_gate():
    statistic = {
        "seeds": 20,
        "paired_improvement_mean_pct": 10.0,
        "paired_bootstrap_ci95_low_pct": 2.0,
        "committee_wins": 12,
        "transition_static_nrmse_mean": 0.10,
        "transition_committee_nrmse_mean": 0.09,
    }
    gates = BENCHMARK.validation_gates(statistic)
    assert all(gates.values())

    statistic["transition_committee_nrmse_mean"] = 0.11
    gates = BENCHMARK.validation_gates(statistic)
    assert not gates["transition_not_worse"]
    assert not all(gates.values())


def test_model_quality_summary_averages_seeds_without_mixing_methods():
    rows = []
    for seed, static, committee in ((0, 0.20, 0.10), (1, 0.40, 0.20)):
        for sampling, value in (
            ("static_maximin", static),
            ("committee", committee),
        ):
            rows.append(
                {
                    "seed": seed,
                    "budget": 32,
                    "sampling": sampling,
                    "final_force_n_nrmse": value,
                    "pre_peak_force_n_nrmse": value,
                    "pre_peak_displacement_mm_nrmse": value,
                    "strain_energy_nmm_nrmse": value,
                    "aggregate_r2": 1.0 - value,
                    "regime_accuracy": 1.0,
                }
            )
    summary = BENCHMARK.summarize_model_quality(rows)
    assert np.isclose(
        summary["32"]["static_maximin"]["final_force_n_nrmse"], 0.30
    )
    assert np.isclose(
        summary["32"]["committee"]["final_force_n_nrmse"], 0.15
    )


def test_replacement_statistics_enforces_every_frozen_margin():
    rows = []
    metric_names = tuple(
        f"{name}_nrmse" for name in BENCHMARK.OUTPUT_NAMES
    )
    for seed in range(4):
        for method, error in (("committee_64", 0.040), ("maximin_100", 0.041)):
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "aggregate_nrmse": error,
                    "aggregate_r2": 0.995,
                    "transition_aggregate_nrmse": error,
                    **{name: error for name in metric_names},
                }
            )
    comparison = BENCHMARK.replacement_statistics(
        rows,
        reference_budgets=(100,),
        bootstrap_samples=200,
        bootstrap_seed=1,
    )[0]
    assert comparison["replacement_passed"]
    assert all(comparison["gates"].values())

    for row in rows:
        if row["method"] == "committee_64":
            row["pre_peak_displacement_mm_nrmse"] = 0.060
    comparison = BENCHMARK.replacement_statistics(
        rows,
        reference_budgets=(100,),
        bootstrap_samples=200,
        bootstrap_seed=1,
    )[0]
    assert not comparison["gates"]["every_output_within_margin"]
    assert not comparison["replacement_passed"]
