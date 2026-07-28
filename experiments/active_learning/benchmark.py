# PyLCSS active-learning sandbox — Phase 2.5: MULTI-SEED benchmark (v2).
"""
Races ALL acquisition strategies against static LHS across many seeds.
Reports mean +/- std, win rate vs LHS, mean improvement, sims-to-match.
Run:  python benchmark.py     (~1-2 min, prints table + writes CSV)
"""

import csv
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from baseline import run_baseline
from active_learning import run_active_learning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_SEEDS = 12
N_INIT, BATCH, ITERS = 20, 10, 8
TOTAL = N_INIT + BATCH * ITERS
STRATEGIES = ["uncertainty", "gradient", "committee", "random"]


def main():
    rows = []
    t0 = time.time()
    print(f"=== Multi-seed benchmark v2 | {N_SEEDS} seeds | budget={TOTAL} ===")
    for seed in range(N_SEEDS):
        base = run_baseline(surrogate="gp", n_train=TOTAL, seed=seed, quiet=True)
        row = {"seed": seed, "lhs": base["rmse"]}
        for strat in STRATEGIES:
            res = run_active_learning(strat, N_INIT, BATCH, ITERS,
                                      seed=seed, verbose=False)
            row[strat] = res["rmse"]
            row[f"{strat}_match"] = next(
                (n for n, r, _ in res["history"] if r <= base["rmse"]), None)
        rows.append(row)
        print(f"  seed {seed:>2} | " + "  ".join(
            f"{s[:4].upper()}={row[s]:.4f}" for s in STRATEGIES)
            + f"  LHS={row['lhs']:.4f}")

    lhs = np.array([r["lhs"] for r in rows])
    print(f"\n=== SUMMARY over {N_SEEDS} seeds (final RMSE, lower=better) ===")
    print(f"  {'LHS (static)':<18}: {lhs.mean():.4f} +/- {lhs.std():.4f}")
    for s in STRATEGIES:
        arr = np.array([r[s] for r in rows])
        wins = int((arr < lhs).sum())
        impr = 100 * (lhs - arr).mean() / lhs.mean()
        matches = [r[f"{s}_match"] for r in rows if r[f"{s}_match"] is not None]
        m_str = (f"match@{np.mean(matches):.0f} ({len(matches)}/{N_SEEDS})"
                 if matches else "match: never")
        print(f"  {s:<18}: {arr.mean():.4f} +/- {arr.std():.4f} | "
              f"wins {wins}/{N_SEEDS} | +{impr:.1f}% | {m_str}")

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] -> benchmark_results.csv | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()