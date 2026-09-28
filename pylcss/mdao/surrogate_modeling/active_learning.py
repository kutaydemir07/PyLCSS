# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Acquisition and batch-selection utilities for active surrogate learning.

The module intentionally has no Qt dependencies.  The GUI worker owns the
expensive simulation loop; this module only builds acquisition models, scores a
fixed candidate pool, and selects a spatially diverse batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


EXPLORE_FLOOR = 0.3
ACTIVE_LEARNING_STRATEGIES = ("uncertainty", "committee", "random")


class ActiveLearningError(RuntimeError):
    """Raised when an acquisition batch cannot be scored or selected."""


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Configuration for an adaptive sampling run.

    ``min_dist`` is measured in the normalized unit design space, never in raw
    engineering units.  This keeps, for example, force and thickness variables
    from dominating the diversity filter merely because their scales differ.
    """

    strategy: str = "committee"
    n_rounds: int = 5
    batch_size: int = 10
    n_candidates: int = 1000
    explore_floor: float = EXPLORE_FLOOR
    min_dist: float = 0.06
    random_state: int = 42
    gp_restarts: int = 3

    def __post_init__(self) -> None:
        strategy = str(self.strategy).lower().strip()
        object.__setattr__(self, "strategy", strategy)
        if strategy not in ACTIVE_LEARNING_STRATEGIES:
            raise ValueError(
                f"Unknown active-learning strategy {self.strategy!r}; expected "
                f"one of {ACTIVE_LEARNING_STRATEGIES}."
            )
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.n_candidates < self.batch_size:
            raise ValueError("n_candidates must be greater than or equal to batch_size.")
        if self.n_candidates < self.n_rounds * self.batch_size:
            raise ValueError(
                "The fixed candidate pool must contain at least "
                "n_rounds * batch_size points."
            )
        if not 0.0 <= self.explore_floor <= 1.0:
            raise ValueError("explore_floor must be between 0 and 1.")
        if not 0.0 <= self.min_dist <= 1.0:
            raise ValueError("min_dist must be between 0 and 1 in unit-space coordinates.")
        if self.gp_restarts < 0:
            raise ValueError("gp_restarts cannot be negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ActiveLearningConfig":
        """Create a validated config from GUI/API values.

        Both the public names and the historical ``al_*`` GUI names are
        accepted so project settings remain forwards-compatible.
        """

        values = values or {}

        def get(name: str, default: Any) -> Any:
            return values.get(name, values.get(f"al_{name}", default))

        return cls(
            strategy=str(get("strategy", "committee")),
            n_rounds=int(get("n_rounds", 5)),
            batch_size=int(get("batch_size", 10)),
            n_candidates=int(get("n_candidates", 1000)),
            explore_floor=float(get("explore_floor", EXPLORE_FLOOR)),
            min_dist=float(get("min_dist", 0.06)),
            random_state=int(get("random_state", 42)),
            gp_restarts=int(get("gp_restarts", 3)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_rounds": self.n_rounds,
            "batch_size": self.batch_size,
            "n_candidates": self.n_candidates,
            "explore_floor": self.explore_floor,
            "min_dist": self.min_dist,
            "random_state": self.random_state,
            "gp_restarts": self.gp_restarts,
        }


@dataclass(frozen=True)
class AcquisitionResult:
    scores: np.ndarray
    source: str
    fallback_used: bool = False


@dataclass(frozen=True)
class BatchSelection:
    points: np.ndarray
    indices: np.ndarray
    scores: np.ndarray
    acquisition_source: str
    fallback_used: bool = False
    diversity_relaxed: bool = False


def _as_2d(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}.")
    return arr


def _normalize_columns(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Normalize each output separately before combining multiple outputs."""

    arr = np.abs(_as_2d(values))
    maxima = np.max(arr, axis=0, keepdims=True)
    maxima = np.where(maxima > 1e-12, maxima, 1.0)
    return arr / maxima


def _aggregate_outputs(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a unitless per-row magnitude for single or multiple outputs."""

    normalized = _normalize_columns(values)
    return np.sqrt(np.mean(normalized**2, axis=1))


def _normalize_vector(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    maximum = float(np.max(np.abs(arr))) if arr.size else 0.0
    if maximum <= 1e-12:
        return np.zeros_like(arr)
    return arr / maximum


def normalize_to_unit(
    points: np.ndarray | Sequence[Sequence[float]],
    bounds: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Map physical design points to the unit cube used for distances."""

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a two-dimensional array.")
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (arr.shape[1], 2):
        raise ValueError(
            f"Expected {arr.shape[1]} (lower, upper) bounds, got {bounds_arr.shape}."
        )
    lower = bounds_arr[:, 0]
    span = bounds_arr[:, 1] - lower
    if np.any(~np.isfinite(bounds_arr)) or np.any(span <= 0.0):
        raise ValueError("Every active-learning bound must be finite with upper > lower.")
    return (arr - lower) / span


def latin_hypercube_pool(
    bounds: Sequence[tuple[float, float]],
    n_candidates: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic fixed LHS pool in physical and unit space."""

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must contain one (lower, upper) pair per variable.")
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive.")
    lower, upper = bounds_arr[:, 0], bounds_arr[:, 1]
    if np.any(~np.isfinite(bounds_arr)) or np.any(upper <= lower):
        raise ValueError("Every active-learning bound must be finite with upper > lower.")
    sampler = qmc.LatinHypercube(d=len(bounds_arr), seed=random_state)
    unit = sampler.random(n_candidates)
    return qmc.scale(unit, lower, upper), unit


def committee_scores(
    gp_std: np.ndarray,
    gp_mean: np.ndarray,
    rf_mean: np.ndarray,
    explore_floor: float = EXPLORE_FLOOR,
) -> np.ndarray:
    """Compute the validated GP–RF acquisition score.

    For multiple outputs, each output is normalized independently and their
    root-mean-square magnitude is used.  This prevents a high-unit response
    (for example stress in Pa) from hiding a low-unit response.
    """

    if not 0.0 <= explore_floor <= 1.0:
        raise ValueError("explore_floor must be between 0 and 1.")
    std = _normalize_vector(_aggregate_outputs(gp_std))
    disagreement = _normalize_vector(
        _aggregate_outputs(_as_2d(gp_mean) - _as_2d(rf_mean))
    )
    if std.shape != disagreement.shape:
        raise ValueError("GP uncertainty and committee disagreement shapes do not match.")
    return std * (explore_floor + (1.0 - explore_floor) * disagreement)


def _committee_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    random_state: int,
    gp_restarts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the acquisition-only GP/RF pair in standardized coordinates."""

    X_train = np.asarray(X_train, dtype=float)
    X_pool = np.asarray(X_pool, dtype=float)
    y_2d = _as_2d(y_train)
    if X_train.ndim != 2 or X_pool.ndim != 2:
        raise ValueError("X_train and X_pool must be two-dimensional arrays.")
    if len(X_train) != len(y_2d):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if X_train.shape[1] != X_pool.shape[1]:
        raise ValueError("Training and candidate points must have the same feature count.")
    if len(X_train) < 2:
        raise ActiveLearningError("At least two training samples are required for committee acquisition.")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_2d)):
        raise ActiveLearningError("Committee training data contains NaN or infinite values.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_train)
    pool_scaled = x_scaler.transform(X_pool)
    y_scaled = y_scaler.fit_transform(y_2d)
    y_fit = y_scaled.ravel() if y_scaled.shape[1] == 1 else y_scaled

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=gp_restarts,
        random_state=random_state,
    )
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        gp.fit(X_scaled, y_fit)
        rf.fit(X_scaled, y_fit)
        gp_mean, gp_std = gp.predict(pool_scaled, return_std=True)
        rf_mean = rf.predict(pool_scaled)
    except Exception as exc:  # sklearn raises several numerical exception types
        raise ActiveLearningError(f"GP–RF acquisition committee could not be fitted: {exc}") from exc

    return _as_2d(gp_mean), _as_2d(gp_std), _as_2d(rf_mean)


def _primary_uncertainty(primary_model: Any, X_pool: np.ndarray) -> np.ndarray:
    if primary_model is None:
        raise TypeError("No primary model was supplied.")
    prediction = primary_model.predict(X_pool, return_std=True)
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise TypeError("Model.predict(return_std=True) did not return (mean, std).")
    std = np.asarray(prediction[1], dtype=float)
    if std.size == 0 or not np.all(np.isfinite(std)):
        raise ValueError("Model returned empty or non-finite uncertainty values.")
    aggregated = _aggregate_outputs(std)
    if float(np.ptp(aggregated)) <= 1e-12:
        raise ValueError("Model uncertainty is constant and cannot rank candidates.")
    return aggregated


def acquisition_scores(
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    *,
    primary_model: Any = None,
    explore_floor: float = EXPLORE_FLOOR,
    random_state: int = 42,
    gp_restarts: int = 3,
) -> AcquisitionResult:
    """Score a candidate pool using uncertainty, committee, or random search.

    ``uncertainty`` first asks the user's current model for predictive standard
    deviation.  If that model cannot provide an informative standard
    deviation, GP–RF disagreement is used explicitly and reported as a
    fallback; the method never silently becomes random sampling.
    """

    strategy = str(strategy).lower().strip()
    X_pool = np.asarray(X_pool, dtype=float)
    if strategy not in ACTIVE_LEARNING_STRATEGIES:
        raise ValueError(f"Unknown active-learning strategy {strategy!r}.")
    if strategy == "random":
        rng = np.random.default_rng(random_state)
        return AcquisitionResult(rng.random(len(X_pool)), "random")

    if strategy == "uncertainty":
        try:
            scores = _normalize_vector(_primary_uncertainty(primary_model, X_pool))
            return AcquisitionResult(scores, "primary-model uncertainty")
        except (TypeError, ValueError, AttributeError):
            gp_mean, _gp_std, rf_mean = _committee_predictions(
                X_train, y_train, X_pool, random_state, gp_restarts
            )
            disagreement = _normalize_vector(
                _aggregate_outputs(gp_mean - rf_mean)
            )
            return AcquisitionResult(
                disagreement,
                "GP–RF disagreement fallback",
                fallback_used=True,
            )

    gp_mean, gp_std, rf_mean = _committee_predictions(
        X_train, y_train, X_pool, random_state, gp_restarts
    )
    scores = committee_scores(gp_std, gp_mean, rf_mean, explore_floor)
    return AcquisitionResult(scores, "GP uncertainty × GP–RF disagreement")


def diverse_top_k(
    scores: np.ndarray,
    pool_unit: np.ndarray,
    k: int,
    taken_mask: np.ndarray | None = None,
    min_dist: float = 0.06,
) -> np.ndarray:
    """Greedily select high-scoring points separated in normalized space."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    pool_unit = np.asarray(pool_unit, dtype=float)
    if pool_unit.ndim != 2 or len(pool_unit) != len(scores):
        raise ValueError("scores and pool_unit must describe the same candidate rows.")
    if k < 1:
        raise ValueError("k must be positive.")
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("min_dist must be between 0 and 1.")
    if taken_mask is None:
        taken = np.zeros(len(scores), dtype=bool)
    else:
        taken = np.asarray(taken_mask, dtype=bool)
        if taken.shape != scores.shape:
            raise ValueError("taken_mask must have one value per candidate.")

    picked: list[int] = []
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    for idx in np.argsort(-safe_scores, kind="stable"):
        idx = int(idx)
        if taken[idx]:
            continue
        if all(np.linalg.norm(pool_unit[idx] - pool_unit[j]) >= min_dist for j in picked):
            picked.append(idx)
        if len(picked) == k:
            break
    return np.asarray(picked, dtype=int)


class ActiveLearningSelector:
    """Stateful selector backed by one fixed, seeded candidate pool."""

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        config: ActiveLearningConfig,
    ) -> None:
        self.config = config
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.pool, self.pool_unit = latin_hypercube_pool(
            self.bounds, config.n_candidates, config.random_state
        )
        self.taken = np.zeros(config.n_candidates, dtype=bool)
        self.round_index = 0

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        primary_model: Any = None,
    ) -> BatchSelection:
        if self.round_index >= self.config.n_rounds:
            raise ActiveLearningError("All configured active-learning rounds are complete.")
        available = int(np.count_nonzero(~self.taken))
        if available < self.config.batch_size:
            raise ActiveLearningError(
                f"Only {available} unused candidates remain for a batch of "
                f"{self.config.batch_size}."
            )

        result = acquisition_scores(
            self.config.strategy,
            X_train,
            y_train,
            self.pool,
            primary_model=primary_model,
            explore_floor=self.config.explore_floor,
            random_state=self.config.random_state + self.round_index,
            gp_restarts=self.config.gp_restarts,
        )
        indices = diverse_top_k(
            result.scores,
            self.pool_unit,
            self.config.batch_size,
            self.taken,
            self.config.min_dist,
        )
        diversity_relaxed = len(indices) < self.config.batch_size
        if diversity_relaxed:
            # Preserve the simulation budget even for an over-constrained
            # distance setting, but report the relaxation to the caller.
            already = set(indices.tolist())
            extras = []
            for idx in np.argsort(-result.scores, kind="stable"):
                idx = int(idx)
                if self.taken[idx] or idx in already:
                    continue
                extras.append(idx)
                if len(indices) + len(extras) == self.config.batch_size:
                    break
            if extras:
                indices = np.concatenate([indices, np.asarray(extras, dtype=int)])
        if len(indices) != self.config.batch_size:
            raise ActiveLearningError("Unable to fill the requested active-learning batch.")

        self.taken[indices] = True
        self.round_index += 1
        return BatchSelection(
            points=self.pool[indices].copy(),
            indices=indices,
            scores=result.scores[indices].copy(),
            acquisition_source=result.source,
            fallback_used=result.fallback_used,
            diversity_relaxed=diversity_relaxed,
        )


def select_next_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    config: ActiveLearningConfig,
    *,
    primary_model: Any = None,
) -> BatchSelection:
    """Stateless convenience wrapper for selecting the first adaptive batch."""

    return ActiveLearningSelector(bounds, config).select(
        X_train, y_train, primary_model=primary_model
    )


__all__ = [
    "ACTIVE_LEARNING_STRATEGIES",
    "EXPLORE_FLOOR",
    "AcquisitionResult",
    "ActiveLearningConfig",
    "ActiveLearningError",
    "ActiveLearningSelector",
    "BatchSelection",
    "acquisition_scores",
    "committee_scores",
    "diverse_top_k",
    "latin_hypercube_pool",
    "normalize_to_unit",
    "select_next_batch",
]
