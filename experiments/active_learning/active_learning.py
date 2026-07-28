# PyLCSS active-learning sandbox — Phase 2: SELF-TRAINING loop (acq v2).
"""
Self-training loop with pluggable ACQUISITION strategies:

  uncertainty : score = GP predictive std                  (v1)
  gradient    : score = std * (0.3 + 0.7*|grad(mu)|_norm)  (v2a)
  committee   : score = std * (0.3 + 0.7*disagreement)     (v2b, GP vs RF)
  random      : ablation control

Why v2: plain std measures DISTANCE-TO-DATA, not DIFFICULTY. It wastes
budget on domain edges. Weighting by the steepness of the GP mean (or by
model disagreement) redirects budget to where the response is actually
hard, while the 0.3 floor keeps some pure exploration alive (the
needle-in-haystack lesson: never go full exploitation).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from baseline import (expensive_function, lhs_sample, build_surrogate,
                      evaluate, run_baseline, CLIFF_CENTER)

EXPLORE_FLOOR = 0.3   # fraction of the score that stays pure-exploration


class ALGP:
    """GP with honest uncertainty (return_std) — the AL engine."""

    def __init__(self, seed=42):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(0.2, (1e-2, 1e1))
                  + WhiteKernel(1e-4, (1e-8, 1e-1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=3,
                                           random_state=seed)
        self.xs = StandardScaler()

    def fit(self, X, y):
        self.gp.fit(self.xs.fit_transform(X), y)
        return self

    def predict(self, X, return_std=False):
        return self.gp.predict(self.xs.transform(X), return_std=return_std)


def grad_magnitude(model, X, h=0.01):
    """|grad(mu)| via central differences on the surrogate mean (cheap:
    2*d extra predicts per call, no expensive sims involved)."""
    n, d = X.shape
    g2 = np.zeros(n)
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        up = model.predict(np.clip(X + e, 0.0, 1.0))
        dn = model.predict(np.clip(X - e, 0.0, 1.0))
        g2 += ((up - dn) / (2 * h)) ** 2
    return np.sqrt(g2)


def _norm(v):
    return v / (v.max() + 1e-12)


def acquisition_scores(strategy, model, X_pool, X_train, y_train, rng, seed):
    if strategy == "random":
        return rng.random(len(X_pool))
    mu, std = model.predict(X_pool, return_std=True)
    if strategy == "uncertainty":
        return std
    if strategy == "gradient":
        g = grad_magnitude(model, X_pool)
        return _norm(std) * (EXPLORE_FLOOR + (1 - EXPLORE_FLOOR) * _norm(g))
    if strategy == "committee":
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X_train, y_train)
        dis = np.abs(mu - rf.predict(X_pool))
        return _norm(std) * (EXPLORE_FLOOR + (1 - EXPLORE_FLOOR) * _norm(dis))
    raise ValueError(strategy)


def diverse_top_k(scores, pool, k, taken_mask, min_dist=0.06):
    """Greedy top-K by score with a min-distance filter so the batch
    doesn't collapse onto one hotspot."""
    picked = []
    for idx in np.argsort(-scores):
        if taken_mask[idx]:
            continue
        if all(np.linalg.norm(pool[idx] - pool[j]) >= min_dist for j in picked):
            picked.append(idx)
        if len(picked) == k:
            break
    return np.asarray(picked, dtype=int)


def cliff_hits(X, band=0.06):
    return int(np.sum(np.abs(np.mean(X, axis=1) - CLIFF_CENTER) < band))


def run_active_learning(strategy="uncertainty", n_init=20, batch=10, iters=8,
                        n_dims=2, pool_size=3000, seed=42, verbose=True):
    rng = np.random.default_rng(seed)

    X_pool = lhs_sample(pool_size, n_dims, seed=seed + 100)
    taken = np.zeros(pool_size, dtype=bool)

    X_train = lhs_sample(n_init, n_dims, seed=seed)
    y_train = expensive_function(X_train)                 # expensive calls: n_init

    X_test = lhs_sample(2000, n_dims, seed=seed + 1)
    y_test = expensive_function(X_test)

    model = ALGP(seed=seed).fit(X_train, y_train)
    rmse, r2 = evaluate(model, X_test, y_test)
    history = [(len(y_train), rmse, r2)]
    if verbose:
        print(f"  it=0 | sims={len(y_train):>3} | RMSE={rmse:.4f} R2={r2:.4f} "
              f"| cliff_hits={cliff_hits(X_train)}")

    for it in range(1, iters + 1):
        scores = acquisition_scores(strategy, model, X_pool,
                                    X_train, y_train, rng, seed)
        idx = diverse_top_k(scores, X_pool, batch, taken)
        taken[idx] = True
        X_new = X_pool[idx]
        y_new = expensive_function(X_new)                 # expensive calls: batch

        X_train = np.vstack([X_train, X_new])
        y_train = np.concatenate([y_train, y_new])
        model = ALGP(seed=seed).fit(X_train, y_train)

        rmse, r2 = evaluate(model, X_test, y_test)
        history.append((len(y_train), rmse, r2))
        if verbose:
            print(f"  it={it} | sims={len(y_train):>3} | RMSE={rmse:.4f} "
                  f"R2={r2:.4f} | cliff_hits={cliff_hits(X_train)}")

    return {"model": model, "X": X_train, "y": y_train, "history": history,
            "rmse": rmse, "r2": r2, "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    N_INIT, BATCH, ITERS = 20, 10, 8
    TOTAL = N_INIT + BATCH * ITERS

    print(f"=== Phase 2: active learning | budget = {TOTAL} sims ===\n")
    print("[A] Uncertainty loop (v1):")
    al = run_active_learning("uncertainty", N_INIT, BATCH, ITERS)

    print("\n[A2] Gradient-weighted loop (v2):")
    alg = run_active_learning("gradient", N_INIT, BATCH, ITERS)

    print("\n[B] Random loop (ablation control):")
    rnd = run_active_learning("random", N_INIT, BATCH, ITERS, verbose=False)
    print(f"  final | sims={TOTAL} | RMSE={rnd['rmse']:.4f} R2={rnd['r2']:.4f}")

    print("\n[C] Static LHS baseline, same budget:")
    base = run_baseline(surrogate="gp", n_train=TOTAL, quiet=True)
    print(f"  final | sims={TOTAL} | RMSE={base['rmse']:.4f} R2={base['r2']:.4f}")

    print("\n=== SUMMARY (equal budget) ===")
    print(f"  AL uncertainty : RMSE={al['rmse']:.4f}")
    print(f"  AL gradient    : RMSE={alg['rmse']:.4f}")
    print(f"  Random loop    : RMSE={rnd['rmse']:.4f}")
    print(f"  Static LHS     : RMSE={base['rmse']:.4f}")