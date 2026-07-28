"""Small end-to-end active-learning validation against a real CalculiX run.

The study varies the pressure in a CAD graph and learns the maximum von Mises
stress.  It intentionally uses a tiny budget; the goal is integration validation
(candidate selection -> cad.fea -> dataset update -> final model), not a
statistically meaningful comparison with static LHS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pylcss.solver_backends.calculix import resolve_calculix_executable
from pylcss.surrogate_modeling.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSelector,
    latin_hypercube_pool,
)
from pylcss.surrogate_modeling.training_engine import SurrogateTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAD_PATH = (
    REPO_ROOT
    / "data"
    / "cad_environment"
    / "01_fea"
    / "03_fea_plate_solution_space.cad"
)
PRESSURE_KEY = "0x00d::pressure"
PRESSURE_BOUNDS = (0.5, 3.0)


def _spy_code(cad_path: Path) -> str:
    return f"""
from pylcss.design_studio import runtime as cad

def spy_model(*args):
    pressure = float(args[0])
    result = cad.fea(
        r'{cad_path}',
        _settings={{r'{PRESSURE_KEY}': pressure}},
    )
    return {{'input_0': pressure}}, {{'output_0': result.max_stress}}
"""


def run_validation(
    n_init: int,
    rounds: int,
    batch_size: int,
    seed: int,
    cad_path: Path = DEFAULT_CAD_PATH,
) -> dict:
    cad_path = cad_path.resolve()
    if not cad_path.is_file():
        raise FileNotFoundError(f"CAD graph was not found: {cad_path}")

    executable = resolve_calculix_executable()
    if not executable:
        raise RuntimeError(
            "CalculiX was not found. Run `python scripts/install_solvers.py --only ccx`."
        )

    trainer = SurrogateTrainer()
    metadata = [{"name": "pressure"}]
    output_metadata = [{"name": "max_stress"}]
    initial_X, _ = latin_hypercube_pool([PRESSURE_BOUNDS], n_init, seed)
    X, y, failures = trainer.evaluate_points(
        _spy_code(cad_path), metadata, output_metadata, initial_X
    )
    if failures:
        raise RuntimeError(f"Initial FEA evaluations failed: {failures}")
    y = y.ravel()

    config = ActiveLearningConfig(
        strategy="committee",
        n_rounds=rounds,
        batch_size=batch_size,
        n_candidates=max(20, rounds * batch_size * 5),
        explore_floor=0.3,
        min_dist=0.06,
        random_state=seed,
        gp_restarts=0,
    )
    selector = ActiveLearningSelector([PRESSURE_BOUNDS], config)
    selections = []
    for round_index in range(rounds):
        batch = selector.select(X, y)
        new_X, new_y, failures = trainer.evaluate_points(
            _spy_code(cad_path), metadata, output_metadata, batch.points
        )
        if failures:
            raise RuntimeError(f"FEA round {round_index + 1} failed: {failures}")
        X = np.vstack([X, new_X])
        y = np.concatenate([y, new_y.ravel()])
        selections.append({
            "round": round_index + 1,
            "pressures": new_X.ravel().tolist(),
            "max_stress": new_y.ravel().tolist(),
            "source": batch.acquisition_source,
        })

    test_X = np.array([[0.75], [1.75], [2.75]])
    _, test_y, failures = trainer.evaluate_points(
        _spy_code(cad_path), metadata, output_metadata, test_X
    )
    if failures:
        raise RuntimeError(f"FEA validation points failed: {failures}")
    test_y = test_y.ravel()

    model, metrics = trainer.train_model(
        X,
        y,
        {
            "model_type": "Random Forest",
            "n_estimators": 300,
            "random_state": seed,
        },
        test_X,
        test_y,
    )
    return {
        "solver": executable,
        "cad_path": str(cad_path),
        "strategy": config.strategy,
        "initial_samples": n_init,
        "adaptive_samples": rounds * batch_size,
        "total_training_samples": len(X),
        "selections": selections,
        "test_pressures": test_X.ravel().tolist(),
        "test_max_stress": test_y.tolist(),
        "predicted_max_stress": np.asarray(model.predict(test_X)).ravel().tolist(),
        "rmse": float(metrics["RMSE"]),
        "r2": float(metrics["R2"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cad-path",
        type=Path,
        default=DEFAULT_CAD_PATH,
        help=(
            "CAD graph to evaluate (default: "
            "data/cad_environment/01_fea/03_fea_plate_solution_space.cad)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the JSON result will be saved.",
    )
    args = parser.parse_args()
    result = run_validation(
        args.initial,
        args.rounds,
        args.batch_size,
        args.seed,
        cad_path=args.cad_path,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
