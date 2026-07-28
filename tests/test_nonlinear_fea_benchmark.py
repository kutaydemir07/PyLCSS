import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    REPO_ROOT / "experiments" / "active_learning" / "nonlinear_fea_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pylcss_nonlinear_fea_benchmark",
    BENCHMARK_PATH,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
BOUNDS = BENCHMARK.BOUNDS
design_points = BENCHMARK.design_points
farthest_point_order = BENCHMARK.farthest_point_order
prediction_metrics = BENCHMARK.prediction_metrics


def test_nonlinear_benchmark_design_is_seeded_and_bounded():
    first = design_points(pool_size=8, test_size=4, seed=17)
    second = design_points(pool_size=8, test_size=4, seed=17)
    assert first == second
    assert len(first) == 12
    for row in first:
        for name, (lower, upper) in zip(
            ("pressure_mpa", "thickness_mm", "hole_radius_mm"),
            BOUNDS,
        ):
            assert lower <= float(row[name]) <= upper


def test_farthest_point_order_is_a_deterministic_permutation():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    )
    order = farthest_point_order(points, seed=9)
    assert order.tolist() == farthest_point_order(points, seed=9).tolist()
    assert sorted(order.tolist()) == list(range(len(points)))


def test_prediction_metrics_are_exact_for_exact_predictions():
    y = np.array([[100.0, 0.1], [250.0, 0.5], [300.0, 1.0]])
    metrics = prediction_metrics(y, y.copy())
    assert metrics["aggregate_nrmse"] == 0.0
    assert metrics["aggregate_r2"] == 1.0
    assert metrics["yield_regime_accuracy"] == 1.0


def test_benchmark_cad_enables_real_plastic_nonlinearity():
    path = (
        REPO_ROOT
        / "data"
        / "cad_environment"
        / "01_fea"
        / "04_nonlinear_fea_benchmark_plate.cad"
    )
    session = json.loads(path.read_text(encoding="utf-8"))
    nodes = list(session["nodes"].values())
    solver = next(
        node for node in nodes
        if node["type_"] == "com.cad.sim.solver.SolverNode"
    )
    material = next(
        node for node in nodes
        if node["type_"] == "com.cad.sim.material.MaterialNode"
    )
    assert solver["custom"]["analysis_type"] == "Nonlinear (Plastic)"
    assert material["custom"]["yield_strength"] == 250.0
    assert material["custom"]["tangent_modulus"] > 0.0

    pressure_node_id, pressure_property = BENCHMARK.PRESSURE_KEY.rsplit("::", 1)
    pressure_node = session["nodes"][pressure_node_id]
    assert pressure_node["type_"] == "com.cad.sim.pressure_load.PressureLoadNode"
    assert pressure_property in pressure_node["custom"]
