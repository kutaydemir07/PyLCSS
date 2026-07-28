# Verification for the UncertaintyWrapper Gaussian-Process fix.
"""Checks the GP uncertainty path against the real production strategies.

Run:  python experiments/active_learning/test_gp_uncertainty.py
Exits non-zero if any assertion fails.
"""

import sys
import os
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

from pylcss.surrogate_modeling.training_engine import (  # noqa: E402
    GaussianProcessStrategy,
    RandomForestStrategy,
)

RNG = np.random.default_rng(0)
FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_data(n=40, d=3, multi=False):
    X = RNG.random((n, d)) * np.array([2.0, 500.0, 0.01])  # deliberately mixed scales
    y = X[:, 0] ** 2 + 0.01 * X[:, 1] + 300.0 * X[:, 2]
    if multi:
        y = np.column_stack([y, 1e5 * y])  # second output on a wildly different scale
    return X, y


def test_gp(multi):
    label = "multi-output" if multi else "single-output"
    print(f"\nGaussian Process ({label})")
    X, y = make_data(multi=multi)
    model, _ = GaussianProcessStrategy().train(X, y, {"random_state": 0}, X, y)

    mean, std = model.predict(X, return_std=True)

    # 1. It must not raise -- this is the regression being fixed.
    check("return_std=True does not raise", True)

    # 2. The mean must match the plain predict path exactly.
    plain = model.predict(X)
    check("mean matches plain predict()", np.allclose(mean, plain, rtol=1e-9, atol=1e-9),
          f"max diff {np.max(np.abs(np.asarray(mean) - np.asarray(plain))):.3e}")

    # 3. Shapes must mirror the target.
    check("shape mirrors y", np.shape(std) == np.shape(y),
          f"std{np.shape(std)} vs y{np.shape(y)}")

    # 4. Std must be finite and non-negative. Exact zeros are legitimate: a GP
    #    interpolates its training data, and sklearn clips tiny negative
    #    variances to zero. Those candidates simply score 0 and are not picked.
    std_arr = np.asarray(std)
    n_zero = int(np.count_nonzero(std_arr == 0))
    check("std finite and non-negative",
          bool(np.all(np.isfinite(std_arr)) and np.all(std_arr >= 0)),
          f"{n_zero}/{std_arr.size} exactly zero (expected at training points)")

    # 5. Std must be in ENGINEERING units, not scaled units. Recompute the
    #    ground truth independently via the target scaler's scale_ factor.
    pipeline = model.model.regressor_
    gp = pipeline.named_steps["regressor"]
    scaler_x = pipeline.named_steps["scaler"]
    _, std_scaled = gp.predict(scaler_x.transform(X), return_std=True)
    scale = np.asarray(model.model.transformer_.scale_, dtype=float)
    expected = np.asarray(std_scaled, dtype=float)
    if expected.ndim == 1:
        expected = expected.reshape(-1, 1)
    expected = (expected * scale.reshape(1, -1))
    if not multi:
        expected = expected.ravel()
    check("std rescaled to engineering units",
          np.allclose(std, expected, rtol=1e-8, atol=1e-12),
          f"max rel err {np.max(np.abs(np.asarray(std) - expected) / (expected + 1e-30)):.3e}")

    # 6. The acquisition function needs VARIATION, not just a value.
    spread = float(np.ptp(np.asarray(std).reshape(len(X), -1)[:, 0]))
    check("std varies across points (rankable)", spread > 1e-12, f"ptp {spread:.3e}")

    # 7. Uncertainty must grow away from the training data.
    far = X.mean(axis=0) + 50.0 * X.std(axis=0)
    _, std_far = model.predict(far.reshape(1, -1), return_std=True)
    near_med = float(np.median(np.asarray(std).reshape(len(X), -1)[:, 0]))
    far_val = float(np.asarray(std_far).reshape(1, -1)[0, 0])
    check("std larger far from the training data", far_val > near_med,
          f"far {far_val:.4g} vs median {near_med:.4g}")


def test_rf():
    print("\nRandom Forest (regression guard -- must stay working)")
    X, y = make_data()
    model, _ = RandomForestStrategy().train(X, y, {"random_state": 0}, X, y)
    mean, std = model.predict(X, return_std=True)
    check("return_std=True does not raise", True)
    check("shapes match y", np.shape(mean) == np.shape(y) and np.shape(std) == np.shape(y))
    check("std finite and non-negative", bool(np.all(np.isfinite(std)) and np.all(std >= 0)))


if __name__ == "__main__":
    print("=== UncertaintyWrapper GP fix verification ===")
    test_gp(multi=False)
    test_gp(multi=True)
    test_rf()
    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
