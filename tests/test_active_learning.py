import numpy as np
import pytest

from pylcss.surrogate_modeling.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSelector,
    acquisition_scores,
    committee_scores,
    diverse_top_k,
    normalize_to_unit,
)


def test_config_rejects_pool_smaller_than_total_budget():
    with pytest.raises(ValueError, match=r"n_rounds \* batch_size"):
        ActiveLearningConfig(n_rounds=3, batch_size=4, n_candidates=10)


def test_normalize_to_unit_removes_physical_scale_difference():
    points = np.array([[1.0, 1000.0], [3.0, 50000.0], [2.0, 25500.0]])
    unit = normalize_to_unit(points, [(1.0, 3.0), (1000.0, 50000.0)])
    np.testing.assert_allclose(unit, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])


def test_committee_score_matches_validated_formula():
    std = np.array([1.0, 2.0, 4.0])
    gp = np.array([0.0, 0.0, 0.0])
    rf = np.array([4.0, 2.0, 0.0])
    actual = committee_scores(std, gp, rf, explore_floor=0.3)
    expected = np.array([0.25, 0.325, 0.3])
    np.testing.assert_allclose(actual, expected)


def test_multioutput_score_is_invariant_to_output_units():
    std = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    gp = np.zeros((3, 2))
    rf = np.array([[1.0, 4.0], [2.0, 1.0], [0.5, 3.0]])
    base = committee_scores(std, gp, rf)

    scale = np.array([1.0, 1000.0])
    rescaled = committee_scores(std * scale, gp * scale, rf * scale)
    np.testing.assert_allclose(base, rescaled)


def test_diverse_top_k_uses_distance_filter():
    scores = np.array([10.0, 9.0, 8.0, 7.0])
    pool = np.array([[0.0, 0.0], [0.01, 0.01], [0.8, 0.8], [0.82, 0.82]])
    selected = diverse_top_k(scores, pool, k=2, min_dist=0.1)
    np.testing.assert_array_equal(selected, [0, 2])


class _NoUncertaintyModel:
    def predict(self, X):
        return np.zeros(len(X))


def test_uncertainty_explicitly_falls_back_to_committee_disagreement():
    rng = np.random.default_rng(7)
    X = rng.random((12, 2))
    y = np.sin(4.0 * X[:, 0]) + X[:, 1] ** 2
    pool = rng.random((20, 2))
    result = acquisition_scores(
        "uncertainty",
        X,
        y,
        pool,
        primary_model=_NoUncertaintyModel(),
        gp_restarts=0,
    )
    assert result.fallback_used is True
    assert result.source == "GP–RF disagreement fallback"
    assert np.ptp(result.scores) > 0.0


def test_selector_reuses_one_pool_without_reselecting_candidates():
    config = ActiveLearningConfig(
        strategy="random",
        n_rounds=2,
        batch_size=3,
        n_candidates=30,
        min_dist=0.05,
        random_state=9,
    )
    selector = ActiveLearningSelector([(0.0, 1.0), (0.0, 1.0)], config)
    X = np.array([[0.1, 0.1], [0.9, 0.9]])
    y = np.array([0.0, 1.0])
    first = selector.select(X, y)
    second = selector.select(X, y)
    assert set(first.indices).isdisjoint(set(second.indices))
    assert selector.taken.sum() == 6
