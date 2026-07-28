"""Reproducible scalar-surrogate benchmark on a real nonlinear CalculiX study.

The benchmark uses ``data/cad_environment/01_fea/04_nonlinear_fea_benchmark_plate.cad``: a perforated
steel plate solved with NLGEOM and bilinear plasticity.  Pressure, thickness,
and centre-hole radius vary; maximum von Mises stress and peak displacement are
learned.

Two questions are answered independently:

1. Which PyLCSS scalar surrogate architecture works best at small, medium, and
   larger FEA budgets?
2. At an equal 32-FEA budget, does GP-RF committee sampling beat a static
   space-filling design on this real nonlinear response?

The expensive CalculiX dataset is persistent and resumable.  Offline active
learning reveals a pre-computed pool response only after that point is selected;
this permits repeatable multi-seed comparisons without re-running identical
deterministic FEA jobs for every architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import qmc
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pylcss.surrogate_modeling.active_learning import (
    acquisition_scores,
    diverse_top_k,
    normalize_to_unit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAD_PATH = (
    REPO_ROOT
    / "data"
    / "cad_environment"
    / "01_fea"
    / "04_nonlinear_fea_benchmark_plate.cad"
)
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "experiments" / "active_learning" / "results"
    / "nonlinear_fea" / "dataset.csv"
)
DEFAULT_RESULTS_DIR = DEFAULT_DATASET_PATH.parent

PARAMETER_NAMES = ("pressure_mpa", "thickness_mm", "hole_radius_mm")
OUTPUT_NAMES = ("max_stress_mpa", "peak_displacement_mm")
BOUNDS = ((15.0, 160.0), (6.0, 14.0), (12.0, 28.0))
PRESSURE_KEY = "0x21d8149d390::pressure"
YIELD_STRENGTH_MPA = 250.0

DEFAULT_POOL_SIZE = 80
DEFAULT_TEST_SIZE = 24
DEFAULT_DATA_SEED = 20260723
DEFAULT_BUDGETS = (16, 32, 64)
DEFAULT_BENCHMARK_SEEDS = 5
ACTIVE_INITIAL = 12
ACTIVE_BATCH = 4
ACTIVE_BUDGET = 32

ARCHITECTURES = (
    "Gaussian Process",
    "Random Forest",
    "Gradient Boosting",
    "MLP Regressor",
    "PyTorch DNN",
)

DATASET_FIELDS = (
    "sample_id",
    "split",
    *PARAMETER_NAMES,
    *OUTPUT_NAMES,
    "mass_tonne",
    "analysis_type",
    "yielded",
    "elapsed_s",
    "success",
    "error",
)


def design_points(
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
) -> list[dict[str, object]]:
    """Return deterministic, independent LHS pool and holdout-test designs."""

    if pool_size < 1 or test_size < 1:
        raise ValueError("pool_size and test_size must be positive.")
    bounds = np.asarray(BOUNDS, dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    rows: list[dict[str, object]] = []
    for split, count, split_seed in (
        ("pool", pool_size, seed),
        ("test", test_size, seed + 1),
    ):
        unit = qmc.LatinHypercube(d=len(BOUNDS), seed=split_seed).random(count)
        physical = qmc.scale(unit, lower, upper)
        for index, point in enumerate(physical):
            rows.append(
                {
                    "sample_id": f"{split}_{index:03d}",
                    "split": split,
                    **{
                        name: float(value)
                        for name, value in zip(PARAMETER_NAMES, point)
                    },
                }
            )
    return rows


def _read_latest_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read a resumable CSV, keeping the newest retry for each sample id."""

    if not path.is_file():
        return {}
    latest: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                latest[sample_id] = row
    return latest


def _row_succeeded(row: dict[str, str] | None) -> bool:
    return bool(row) and str(row.get("success", "")).strip() == "1"


def _validate_resume_design(
    expected: Sequence[dict[str, object]],
    existing: dict[str, dict[str, str]],
) -> None:
    for design in expected:
        previous = existing.get(str(design["sample_id"]))
        if not previous:
            continue
        for name in PARAMETER_NAMES:
            if not math.isclose(
                float(previous[name]),
                float(design[name]),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"Existing dataset {design['sample_id']} does not match the "
                    "requested pool/test sizes or seed. Choose a new dataset path."
                )


def generate_dataset(
    cad_path: Path,
    dataset_path: Path,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Run/resume the real CalculiX design and append every completed sample."""

    cad_path = cad_path.resolve()
    dataset_path = dataset_path.resolve()
    if not cad_path.is_file():
        raise FileNotFoundError(f"Nonlinear CAD study was not found: {cad_path}")

    expected = design_points(pool_size, test_size, seed)
    existing = _read_latest_rows(dataset_path)
    _validate_resume_design(expected, existing)
    pending = [
        row for row in expected
        if not _row_succeeded(existing.get(str(row["sample_id"])))
    ]
    if not pending:
        progress(
            f"Dataset already complete: {pool_size} pool + {test_size} test."
        )
        return {"completed": len(expected), "failed": 0, "pending": 0}

    from pylcss.design_studio import runtime as cad

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not dataset_path.is_file() or dataset_path.stat().st_size == 0
    failures = 0
    with dataset_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        if needs_header:
            writer.writeheader()
            handle.flush()

        for pending_index, design in enumerate(pending, start=1):
            started = time.perf_counter()
            output: dict[str, object] = {
                **design,
                "max_stress_mpa": "",
                "peak_displacement_mm": "",
                "mass_tonne": "",
                "analysis_type": "",
                "yielded": "",
                "elapsed_s": "",
                "success": 0,
                "error": "",
            }
            try:
                result = cad.fea(
                    str(cad_path),
                    H=float(design["thickness_mm"]),
                    big_R=float(design["hole_radius_mm"]),
                    _settings={
                        PRESSURE_KEY: float(design["pressure_mpa"]),
                    },
                )
                if result.analysis_type != "Nonlinear (Plastic)":
                    raise RuntimeError(
                        "Study did not execute as Nonlinear (Plastic): "
                        f"{result.analysis_type!r}"
                    )
                max_stress = float(result.max_stress)
                peak_disp = float(result.peak_disp)
                if not np.all(np.isfinite([max_stress, peak_disp])):
                    raise ValueError("CalculiX returned NaN or infinite outputs.")
                output.update(
                    {
                        "max_stress_mpa": max_stress,
                        "peak_displacement_mm": peak_disp,
                        "mass_tonne": float(result.mass),
                        "analysis_type": str(result.analysis_type),
                        "yielded": int(max_stress >= 0.995 * YIELD_STRENGTH_MPA),
                        "success": 1,
                    }
                )
            except Exception as exc:
                failures += 1
                output["error"] = f"{type(exc).__name__}: {exc}"
            output["elapsed_s"] = time.perf_counter() - started
            writer.writerow(output)
            handle.flush()
            progress(
                f"[{pending_index:03d}/{len(pending):03d}] "
                f"{design['sample_id']} p={float(design['pressure_mpa']):7.2f}, "
                f"t={float(design['thickness_mm']):5.2f}, "
                f"r={float(design['hole_radius_mm']):5.2f} -> "
                + (
                    f"stress={float(output['max_stress_mpa']):8.3f}, "
                    f"disp={float(output['peak_displacement_mm']):.6f}"
                    if output["success"]
                    else f"FAILED: {output['error']}"
                )
            )

    completed = sum(
        1
        for row in _read_latest_rows(dataset_path).values()
        if _row_succeeded(row)
    )
    return {
        "completed": completed,
        "failed": failures,
        "pending": max(0, len(expected) - completed),
    }


def load_dataset(
    dataset_path: Path,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_DATA_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load and validate the completed deterministic pool/test dataset."""

    expected = design_points(pool_size, test_size, seed)
    latest = _read_latest_rows(dataset_path.resolve())
    _validate_resume_design(expected, latest)
    missing = [
        str(row["sample_id"])
        for row in expected
        if not _row_succeeded(latest.get(str(row["sample_id"])))
    ]
    if missing:
        raise RuntimeError(
            f"Dataset is incomplete ({len(missing)} missing/failed); run --mode "
            f"generate first. First missing ids: {missing[:5]}"
        )

    def arrays(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        designs = [row for row in expected if row["split"] == split]
        rows = [latest[str(row["sample_id"])] for row in designs]
        X = np.asarray(
            [[float(row[name]) for name in PARAMETER_NAMES] for row in rows],
            dtype=float,
        )
        y = np.asarray(
            [[float(row[name]) for name in OUTPUT_NAMES] for row in rows],
            dtype=float,
        )
        ids = [str(row["sample_id"]) for row in rows]
        return X, y, ids

    X_pool, y_pool, pool_ids = arrays("pool")
    X_test, y_test, test_ids = arrays("test")
    return X_pool, y_pool, X_test, y_test, pool_ids, test_ids


def farthest_point_order(points_unit: np.ndarray, seed: int) -> np.ndarray:
    """Create a deterministic nested space-filling order for a finite pool."""

    points = np.asarray(points_unit, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("points_unit must be a non-empty 2D array.")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(points)))
    selected = np.zeros(len(points), dtype=bool)
    min_dist = np.full(len(points), np.inf, dtype=float)
    order: list[int] = []
    next_index = first
    for _ in range(len(points)):
        order.append(next_index)
        selected[next_index] = True
        distances = np.linalg.norm(points - points[next_index], axis=1)
        min_dist = np.minimum(min_dist, distances)
        min_dist[selected] = -np.inf
        if len(order) < len(points):
            next_index = int(np.argmax(min_dist))
    return np.asarray(order, dtype=int)


def committee_replay_indices(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    seed: int,
    budget: int = ACTIVE_BUDGET,
    n_initial: int = ACTIVE_INITIAL,
    batch_size: int = ACTIVE_BATCH,
    min_dist: float = 0.06,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Replay committee selection on a finite pool without revealing labels."""

    if not 2 <= n_initial < budget <= len(X_pool):
        raise ValueError("Require 2 <= n_initial < budget <= len(X_pool).")
    if (budget - n_initial) % batch_size:
        raise ValueError("budget - n_initial must be divisible by batch_size.")

    pool_unit = normalize_to_unit(X_pool, BOUNDS)
    order = farthest_point_order(pool_unit, seed)
    selected = order[:n_initial].tolist()
    taken = np.zeros(len(X_pool), dtype=bool)
    taken[selected] = True
    trace: list[dict[str, object]] = []
    n_rounds = (budget - n_initial) // batch_size
    for round_index in range(n_rounds):
        selected_arr = np.asarray(selected, dtype=int)
        result = acquisition_scores(
            "committee",
            X_pool[selected_arr],
            y_pool[selected_arr],
            X_pool,
            explore_floor=0.3,
            random_state=seed + round_index,
            gp_restarts=1,
        )
        indices = diverse_top_k(
            result.scores,
            pool_unit,
            batch_size,
            taken_mask=taken,
            min_dist=min_dist,
        )
        if len(indices) < batch_size:
            extras = [
                int(index)
                for index in np.argsort(-result.scores, kind="stable")
                if not taken[int(index)] and int(index) not in set(indices.tolist())
            ][: batch_size - len(indices)]
            indices = np.concatenate([indices, np.asarray(extras, dtype=int)])
        if len(indices) != batch_size:
            raise RuntimeError("Committee replay could not fill its FEA batch.")
        taken[indices] = True
        selected.extend(indices.tolist())
        trace.append(
            {
                "round": round_index + 1,
                "indices": indices.tolist(),
                "scores": result.scores[indices].tolist(),
                "source": result.source,
            }
        )
    return np.asarray(selected, dtype=int), trace


def _scaled_regressor(regressor: object) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            ("input_scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def _sklearn_model(name: str, input_dim: int, seed: int) -> object:
    if name == "Gaussian Process":
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(input_dim),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(1e-4, (1e-8, 1e-1))
        )
        return _scaled_regressor(
            GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=False,
                n_restarts_optimizer=2,
                random_state=seed,
            )
        )
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "Gradient Boosting":
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                loss="squared_error",
                random_state=seed,
            )
        )
    if name == "MLP Regressor":
        return _scaled_regressor(
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.005,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=60,
                random_state=seed,
            )
        )
    raise KeyError(name)


def _torch_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train the PyLCSS tabular DNN with a train-only validation split."""

    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch DNN benchmark requested but torch is unavailable.") from exc

    from pylcss.surrogate_modeling.models import ConfigurableNet

    torch.manual_seed(seed)
    np.random.seed(seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=max(2, int(round(0.2 * len(X_train)))),
        random_state=seed,
    )
    x_scaler = StandardScaler().fit(X_fit)
    y_scaler = StandardScaler().fit(y_fit)
    X_fit_t = torch.tensor(x_scaler.transform(X_fit), dtype=torch.float32)
    X_val_t = torch.tensor(x_scaler.transform(X_val), dtype=torch.float32)
    y_fit_t = torch.tensor(y_scaler.transform(y_fit), dtype=torch.float32)
    y_val_t = torch.tensor(y_scaler.transform(y_val), dtype=torch.float32)
    X_test_t = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)

    net = ConfigurableNet(X_train.shape[1], [64, 64], y_train.shape[1], 0.05)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _epoch in range(2000):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(X_fit_t), y_fit_t)
        loss.backward()
        optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(X_val_t), y_val_t).item())
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in net.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 120:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_scaled = net(X_test_t).numpy()
    return y_scaler.inverse_transform(pred_scaled)


def fit_predict(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    if architecture == "PyTorch DNN":
        prediction = _torch_predict(X_train, y_train, X_test, seed)
        fit_seconds = time.perf_counter() - started
        return prediction, fit_seconds, 0.0

    model = _sklearn_model(architecture, X_train.shape[1], seed)
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    prediction = np.asarray(model.predict(X_test), dtype=float)
    predict_seconds = time.perf_counter() - predict_started
    return prediction, fit_seconds, predict_seconds


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Prediction shape {y_pred.shape} does not match target {y_true.shape}."
        )
    metrics: dict[str, float] = {}
    nrmse_values = []
    r2_values = []
    for index, output_name in enumerate(OUTPUT_NAMES):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        rmse = float(np.sqrt(mean_squared_error(truth, pred)))
        scale = float(np.std(truth))
        nrmse = rmse / scale if scale > 1e-12 else float("inf")
        r2 = float(r2_score(truth, pred))
        metrics[f"{output_name}_rmse"] = rmse
        metrics[f"{output_name}_nrmse"] = nrmse
        metrics[f"{output_name}_mae"] = float(mean_absolute_error(truth, pred))
        metrics[f"{output_name}_r2"] = r2
        nrmse_values.append(nrmse)
        r2_values.append(r2)
    actual_yielded = y_true[:, 0] >= 0.995 * YIELD_STRENGTH_MPA
    predicted_yielded = y_pred[:, 0] >= 0.995 * YIELD_STRENGTH_MPA
    metrics["yield_regime_accuracy"] = float(
        np.mean(actual_yielded == predicted_yielded)
    )
    metrics["aggregate_nrmse"] = float(np.mean(nrmse_values))
    metrics["aggregate_r2"] = float(np.mean(r2_values))
    return metrics


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows were produced for {path.name}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(
    rows: Sequence[dict[str, object]],
    group_keys: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    summary = []
    for key, group in grouped.items():
        nrmse = np.asarray([float(row["aggregate_nrmse"]) for row in group])
        r2 = np.asarray([float(row["aggregate_r2"]) for row in group])
        fit = np.asarray([float(row["fit_seconds"]) for row in group])
        accuracy = np.asarray([float(row["yield_regime_accuracy"]) for row in group])
        summary.append(
            {
                **dict(zip(group_keys, key)),
                "runs": len(group),
                "aggregate_nrmse_mean": float(np.mean(nrmse)),
                "aggregate_nrmse_std": float(np.std(nrmse)),
                "aggregate_r2_mean": float(np.mean(r2)),
                "yield_regime_accuracy_mean": float(np.mean(accuracy)),
                "fit_seconds_mean": float(np.mean(fit)),
            }
        )
    return summary


def _best_by_budget(
    summary: Sequence[dict[str, object]],
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    budgets = sorted({int(row["budget"]) for row in summary})
    for budget in budgets:
        candidates = [row for row in summary if int(row["budget"]) == budget]
        result[budget] = min(
            candidates,
            key=lambda row: float(row["aggregate_nrmse_mean"]),
        )
    return result


def _markdown_report(
    *,
    architecture_summary: Sequence[dict[str, object]],
    sampling_summary: Sequence[dict[str, object]],
    sampling_diagnostics: Sequence[dict[str, object]],
    dataset_info: dict[str, object],
) -> str:
    best = _best_by_budget(architecture_summary)
    lines = [
        "# Nonlinear CalculiX FEA — surrogate architecture benchmark",
        "",
        "## Scope",
        "",
        (
            "Real CalculiX `NLGEOM` + bilinear-plastic analysis of a perforated "
            "steel plate. Inputs are pressure, plate thickness, and centre-hole "
            "radius. Outputs are maximum von Mises stress and peak displacement."
        ),
        "",
        (
            f"Dataset: {dataset_info['pool_size']} reusable design-pool FEA runs + "
            f"{dataset_info['test_size']} untouched holdout FEA runs; "
            f"{dataset_info['yielded_test_points']} holdout points reached the "
            "yield regime."
        ),
        "",
        "## Architecture ranking",
        "",
        "| FEA budget | Architecture | Aggregate NRMSE | Aggregate R² | Yield accuracy | Fit time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        architecture_summary,
        key=lambda item: (
            int(item["budget"]),
            float(item["aggregate_nrmse_mean"]),
        ),
    ):
        lines.append(
            f"| {row['budget']} | {row['architecture']} | "
            f"{float(row['aggregate_nrmse_mean']):.4f} ± "
            f"{float(row['aggregate_nrmse_std']):.4f} | "
            f"{float(row['aggregate_r2_mean']):.4f} | "
            f"{float(row['yield_regime_accuracy_mean']):.3f} | "
            f"{float(row['fit_seconds_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Equal-budget sampling comparison",
            "",
            "| Sampling | Architecture | Aggregate NRMSE | vs static | Aggregate R² | Yield accuracy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        sampling_summary,
        key=lambda item: (
            str(item["sampling"]),
            float(item["aggregate_nrmse_mean"]),
        ),
    ):
        lines.append(
            f"| {row['sampling']} | {row['architecture']} | "
            f"{float(row['aggregate_nrmse_mean']):.4f} ± "
            f"{float(row['aggregate_nrmse_std']):.4f} | "
            f"{float(row.get('vs_static_improvement_pct', 0.0)):+.1f}% | "
            f"{float(row['aggregate_r2_mean']):.4f} | "
            f"{float(row['yield_regime_accuracy_mean']):.3f} |"
        )

    static_by_arch = {
        str(row["architecture"]): row
        for row in sampling_summary
        if row["sampling"] == "static_maximin"
    }
    committee_rows = [
        row for row in sampling_summary if row["sampling"] == "committee"
    ]
    committee_wins = sum(
        float(row["aggregate_nrmse_mean"])
        < float(static_by_arch[str(row["architecture"])]["aggregate_nrmse_mean"])
        for row in committee_rows
    )
    mean_improvement = float(
        np.mean(
            [
                float(row.get("vs_static_improvement_pct", 0.0))
                for row in committee_rows
            ]
        )
    )
    diagnostics_by_sampling: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sampling_diagnostics:
        diagnostics_by_sampling[str(row["sampling"])].append(row)
    diagnostic_text = []
    for sampling in ("static_maximin", "committee"):
        rows = diagnostics_by_sampling.get(sampling, [])
        if not rows:
            continue
        diagnostic_text.append(
            (
                f"- `{sampling}` selected "
                f"{np.mean([float(row['yielded_selected']) for row in rows]):.1f} "
                "yielded points on average; mean/max normalized holdout distance "
                f"was {np.mean([float(row['coverage_mean']) for row in rows]):.3f}/"
                f"{np.mean([float(row['coverage_max']) for row in rows]):.3f}."
            )
        )

    low_budget = min(best)
    high_budget = max(best)
    middle_budget = sorted(best)[len(best) // 2]
    lines.extend(
        [
            "",
            "## Automatic scalar-surrogate guideline",
            "",
            (
                f"- Up to {low_budget} successful FEA samples: start with "
                f"**{best[low_budget]['architecture']}**."
            ),
            (
                f"- Around {middle_budget} samples: start with "
                f"**{best[middle_budget]['architecture']}**."
            ),
            (
                f"- Around {high_budget} or more samples: start with "
                f"**{best[high_budget]['architecture']}**, then verify by "
                "holdout error or cross-validation."
            ),
            (
                (
                    f"- In this real case, committee beat static sampling for "
                    f"{committee_wins}/{len(committee_rows)} final architectures "
                    f"and its architecture-averaged NRMSE change was "
                    f"{mean_improvement:+.1f}%."
                )
            ),
            (
                "- Therefore use **static maximin/LHS as the safe default for "
                "low-dimensional globally smooth FEA responses**. Enable "
                "committee when a pilot benchmark or known contact/buckling/"
                "failure transition shows a repeatable gain."
                if committee_wins < math.ceil(len(committee_rows) / 2)
                else
                "- Committee won this case overall; use it for adaptive FEA "
                "placement while retaining a space-filling exploration floor."
            ),
            (
                "- GINO and Geom-DeepONet are not ranked here: they predict "
                "nodal/field quantities on geometry and require a separate "
                "field-surrogate benchmark."
            ),
            "",
            "## Sampling diagnostics",
            "",
            *diagnostic_text,
            "",
            "## Interpretation limits",
            "",
            (
                "This closes the scalar FEA architecture question for this "
                "three-parameter elastic–plastic plate study. It is a real "
                "material/geometric nonlinear solve, but it is not a universal "
                "claim for contact, buckling, fracture, or high-dimensional "
                "topology-changing problems. Re-run this harness on a new "
                "representative CAD study before changing the product default."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    dataset_path: Path,
    results_dir: Path,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    data_seed: int = DEFAULT_DATA_SEED,
    benchmark_seeds: int = DEFAULT_BENCHMARK_SEEDS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    architectures: Sequence[str] = ARCHITECTURES,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    X_pool, y_pool, X_test, y_test, pool_ids, _ = load_dataset(
        dataset_path,
        pool_size=pool_size,
        test_size=test_size,
        seed=data_seed,
    )
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or min(budgets) < 4 or max(budgets) > len(X_pool):
        raise ValueError("Budgets must be between 4 and the pool size.")
    unknown = sorted(set(architectures) - set(ARCHITECTURES))
    if unknown:
        raise ValueError(f"Unknown architectures: {unknown}")

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pool_unit = normalize_to_unit(X_pool, BOUNDS)
    architecture_rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    selection_traces: dict[str, object] = {}

    total_arch_runs = benchmark_seeds * len(budgets) * len(architectures)
    arch_run = 0
    for seed in range(benchmark_seeds):
        order = farthest_point_order(pool_unit, seed)
        for budget in budgets:
            selected = order[:budget]
            for architecture in architectures:
                arch_run += 1
                progress(
                    f"[architecture {arch_run:03d}/{total_arch_runs:03d}] "
                    f"seed={seed} budget={budget} model={architecture}"
                )
                prediction, fit_seconds, predict_seconds = fit_predict(
                    architecture,
                    X_pool[selected],
                    y_pool[selected],
                    X_test,
                    seed,
                )
                architecture_rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "architecture": architecture,
                        "sampling": "static_maximin",
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        **prediction_metrics(y_test, prediction),
                    }
                )

    sampling_run = 0
    total_sampling_runs = benchmark_seeds * 2 * len(architectures)
    for seed in range(benchmark_seeds):
        static_order = farthest_point_order(pool_unit, seed)
        static_indices = static_order[:ACTIVE_BUDGET]
        active_indices, trace = committee_replay_indices(
            X_pool,
            y_pool,
            seed=seed,
        )
        selection_traces[str(seed)] = {
            "initial_sample_ids": [
                pool_ids[index] for index in active_indices[:ACTIVE_INITIAL]
            ],
            "rounds": [
                {
                    **round_trace,
                    "sample_ids": [
                        pool_ids[index] for index in round_trace["indices"]
                    ],
                }
                for round_trace in trace
            ],
        }
        for sampling, indices in (
            ("static_maximin", static_indices),
            ("committee", active_indices),
        ):
            for architecture in architectures:
                sampling_run += 1
                progress(
                    f"[sampling {sampling_run:03d}/{total_sampling_runs:03d}] "
                    f"seed={seed} method={sampling} model={architecture}"
                )
                prediction, fit_seconds, predict_seconds = fit_predict(
                    architecture,
                    X_pool[indices],
                    y_pool[indices],
                    X_test,
                    seed,
                )
                sampling_rows.append(
                    {
                        "seed": seed,
                        "budget": ACTIVE_BUDGET,
                        "architecture": architecture,
                        "sampling": sampling,
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        **prediction_metrics(y_test, prediction),
                    }
                )

    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(results_dir / "architecture_runs.csv", architecture_rows)
    _write_rows(results_dir / "sampling_runs.csv", sampling_rows)
    architecture_summary = _group_summary(
        architecture_rows,
        ("budget", "architecture"),
    )
    sampling_summary = _group_summary(
        sampling_rows,
        ("sampling", "architecture"),
    )
    static_summary = {
        str(row["architecture"]): row
        for row in sampling_summary
        if row["sampling"] == "static_maximin"
    }
    for row in sampling_summary:
        if row["sampling"] == "static_maximin":
            row["vs_static_improvement_pct"] = 0.0
            continue
        reference = static_summary[str(row["architecture"])]
        static_error = float(reference["aggregate_nrmse_mean"])
        row["vs_static_improvement_pct"] = (
            100.0
            * (static_error - float(row["aggregate_nrmse_mean"]))
            / static_error
        )
    _write_rows(results_dir / "architecture_summary.csv", architecture_summary)
    _write_rows(results_dir / "sampling_summary.csv", sampling_summary)
    (results_dir / "selection_traces.json").write_text(
        json.dumps(selection_traces, indent=2) + "\n",
        encoding="utf-8",
    )

    dataset_info = {
        "pool_size": len(X_pool),
        "test_size": len(X_test),
        "yielded_pool_points": int(
            np.count_nonzero(y_pool[:, 0] >= 0.995 * YIELD_STRENGTH_MPA)
        ),
        "yielded_test_points": int(
            np.count_nonzero(y_test[:, 0] >= 0.995 * YIELD_STRENGTH_MPA)
        ),
        "input_bounds": {
            name: list(bound) for name, bound in zip(PARAMETER_NAMES, BOUNDS)
        },
        "output_ranges": {
            name: [
                float(np.min(np.concatenate([y_pool[:, index], y_test[:, index]]))),
                float(np.max(np.concatenate([y_pool[:, index], y_test[:, index]]))),
            ]
            for index, name in enumerate(OUTPUT_NAMES)
        },
    }
    test_unit = normalize_to_unit(X_test, BOUNDS)
    sampling_diagnostics: list[dict[str, object]] = []
    for seed in range(benchmark_seeds):
        static_indices = farthest_point_order(pool_unit, seed)[:ACTIVE_BUDGET]
        active_indices, _ = committee_replay_indices(
            X_pool,
            y_pool,
            seed=seed,
        )
        for sampling, indices in (
            ("static_maximin", static_indices),
            ("committee", active_indices),
        ):
            distances = np.linalg.norm(
                test_unit[:, None, :] - pool_unit[indices][None, :, :],
                axis=2,
            )
            nearest = np.min(distances, axis=1)
            sampling_diagnostics.append(
                {
                    "seed": seed,
                    "sampling": sampling,
                    "yielded_selected": int(
                        np.count_nonzero(
                            y_pool[indices, 0]
                            >= 0.995 * YIELD_STRENGTH_MPA
                        )
                    ),
                    "near_yield_selected": int(
                        np.count_nonzero(y_pool[indices, 0] >= 235.0)
                    ),
                    "coverage_mean": float(np.mean(nearest)),
                    "coverage_max": float(np.max(nearest)),
                }
            )
    _write_rows(
        results_dir / "sampling_diagnostics.csv",
        sampling_diagnostics,
    )
    summary = {
        "dataset": dataset_info,
        "architecture_summary": architecture_summary,
        "sampling_summary": sampling_summary,
        "sampling_diagnostics": sampling_diagnostics,
        "best_by_budget": {
            str(budget): row
            for budget, row in _best_by_budget(architecture_summary).items()
        },
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "REPORT.md").write_text(
        _markdown_report(
            architecture_summary=architecture_summary,
            sampling_summary=sampling_summary,
            sampling_diagnostics=sampling_diagnostics,
            dataset_info=dataset_info,
        ),
        encoding="utf-8",
    )
    return summary


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return values


def _parse_architectures(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - set(ARCHITECTURES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown architectures {unknown}; available: {ARCHITECTURES}"
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("generate", "benchmark", "all"),
        default="all",
    )
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--benchmark-seeds", type=int, default=DEFAULT_BENCHMARK_SEEDS)
    parser.add_argument(
        "--budgets",
        type=_parse_int_list,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument(
        "--architectures",
        type=_parse_architectures,
        default=ARCHITECTURES,
    )
    args = parser.parse_args()

    if args.mode in {"generate", "all"}:
        status = generate_dataset(
            args.cad_path,
            args.dataset,
            pool_size=args.pool_size,
            test_size=args.test_size,
            seed=args.data_seed,
        )
        print("dataset:", json.dumps(status))
        if status["pending"]:
            return 2
    if args.mode in {"benchmark", "all"}:
        summary = run_benchmark(
            args.dataset,
            args.results_dir,
            pool_size=args.pool_size,
            test_size=args.test_size,
            data_seed=args.data_seed,
            benchmark_seeds=args.benchmark_seeds,
            budgets=args.budgets,
            architectures=args.architectures,
        )
        print(json.dumps(summary["best_by_budget"], indent=2))
        print(f"report: {(args.results_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
