import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pylcss.surrogate_modeling.training_engine import (
    SurrogateTrainer,
    UncertaintyWrapper,
)


def test_evaluate_points_skips_failed_simulations_instead_of_fabricating_zero():
    spy_code = """
def spy_model(*args):
    x = float(args[0])
    if x < 0.0:
        raise RuntimeError("synthetic solver failure")
    return {'input_0': x}, {'output_0': x ** 2}
"""
    trainer = SurrogateTrainer()
    X, y, failures = trainer.evaluate_points(
        spy_code,
        [{'name': 'x'}],
        [{'name': 'y'}],
        np.array([[2.0], [-1.0], [3.0]]),
    )
    np.testing.assert_allclose(X, [[2.0], [3.0]])
    np.testing.assert_allclose(y, [[4.0], [9.0]])
    assert len(failures) == 1
    assert "synthetic solver failure" in failures[0]
    assert not np.any(y == 0.0)


def test_gp_uncertainty_wrapper_unwraps_target_and_input_scalers():
    X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    y = 100.0 * np.sin(X[:, 0])
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GaussianProcessRegressor(
            kernel=RBF(length_scale=0.4),
            optimizer=None,
            alpha=1e-8,
        )),
    ])
    transformed = TransformedTargetRegressor(
        regressor=base,
        transformer=StandardScaler(),
    ).fit(X, y)
    wrapper = UncertaintyWrapper(transformed, 'Gaussian Process (Kriging)')

    query = np.array([[0.15], [0.55], [0.95]])
    mean, std = wrapper.predict(query, return_std=True)
    np.testing.assert_allclose(mean, transformed.predict(query), rtol=1e-8, atol=1e-8)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.all(np.isfinite(std))
    assert np.all(std > 0.0)
