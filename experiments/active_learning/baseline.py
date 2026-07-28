# PyLCSS active-learning sandbox — Phase 1: STATIC baseline.
"""
Mirrors PyLCSS's CURRENT one-shot surrogate pipeline (training_engine.py):
LHS-sample once -> run expensive sim at every point -> fit surrogate -> score.

The expensive sim (cad.fea / cad.crash) is replaced by a synthetic function
with known ground truth so improvements are measurable. Swap it for a real
solver call at integration time; everything else stays identical.
"""

import numpy as np
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score

CLIFF_CENTER = 0.75   # the "mode switch" sits where mean(x) = 0.75
CLIFF_WIDTH = 0.03    # how sharp the switch is (crash-like)


def expensive_function(X, noise_std=0.0, seed=None):
    """Smooth global bowl + a sharp CLIFF (crash-like regime switch).

    Real FEA/crash responses are mostly smooth but contain steep
    transitions (buckling mode switches, contact onset). Uniform LHS
    spends points everywhere; a good active learner should discover the
    cliff from the seed data and swarm it. X in [0,1]^d -> y (n,).
    """
    X = np.atleast_2d(X)
    smooth = np.sum((X - 0.5) ** 2, axis=1)
    cliff = 1.5 * np.tanh((np.mean(X, axis=1) - CLIFF_CENTER) / CLIFF_WIDTH)
    y = smooth + cliff
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise_std, size=y.shape)
    return y


def lhs_sample(n_samples, n_dims, seed=42):
    return qmc.LatinHypercube(d=n_dims, seed=seed).random(n=n_samples)


def build_surrogate(kind="mlp", seed=42):
    """Same scaling strategy as PyLCSS: scale inputs AND target."""
    if kind == "mlp":
        reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu",
                           solver="adam", max_iter=2000, early_stopping=True,
                           n_iter_no_change=20, random_state=seed)
    elif kind == "gp":
        kernel = ConstantKernel(1.0) * RBF(0.2) + WhiteKernel(1e-3)
        reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       random_state=seed)
    elif kind == "rf":
        reg = RandomForestRegressor(n_estimators=200, random_state=seed)
    else:
        raise ValueError(kind)
    pipe = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (float(np.sqrt(mean_squared_error(y_test, y_pred))),
            float(r2_score(y_test, y_pred)))


def run_baseline(n_dims=2, n_train=100, n_test=2000, noise_std=0.0,
                 surrogate="gp", seed=42, quiet=False):
    X_train = lhs_sample(n_train, n_dims, seed=seed)
    y_train = expensive_function(X_train, noise_std=noise_std, seed=seed)
    X_test = lhs_sample(n_test, n_dims, seed=seed + 1)
    y_test = expensive_function(X_test)

    model = build_surrogate(surrogate, seed=seed)
    model.fit(X_train, y_train)
    rmse, r2 = evaluate(model, X_test, y_test)
    if not quiet:
        print(f"[baseline] {surrogate:>3} | train={n_train:>4} "
              f"-> RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"surrogate": surrogate, "n_train": n_train, "rmse": rmse, "r2": r2,
            "X_train": X_train, "y_train": y_train}


if __name__ == "__main__":
    print("=== Phase 1: static baseline (one-shot LHS) ===")
    for sur in ("mlp", "gp", "rf"):
        run_baseline(surrogate=sur, n_train=100)