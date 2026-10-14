# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Benchmark engineering regressors and optionally combine leading models."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import LossCallback, Metrics, ProgressCallback, StopFlag
from .dataset import prepare_training_data
from .mdao_algorithms import (
    MdaoEnsembleOfSurrogates,
    MdaoKPLSRegressor,
    MdaoKrigingRegressor,
    MdaoPolynomialRegressor,
    MdaoRBFRegressor,
    MdaoSVRRegressor,
    compute_mdao_metrics,
)
from .metrics import (
    as_target_matrix,
    evaluate_predictions,
    format_prediction,
    prediction_features,
)
from .strategies import SurrogateModelStrategy

logger = logging.getLogger(__name__)


class AutoSurrogateModel:
    """Selected estimator and its validation leaderboard."""

    def __init__(
        self,
        champion_model: Any,
        champion_name: str,
        champion_family: str,
        citation: str,
        leaderboard: list[dict[str, Any]],
        optimal_weights: list[float] | None = None,
    ) -> None:
        self.champion_model = champion_model
        self.champion_name = champion_name
        self.champion_family = champion_family
        self.citation = citation
        self.leaderboard = leaderboard
        self.optimal_weights = optimal_weights

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        features, single = prediction_features(X)
        if return_std:
            try:
                result = self.champion_model.predict(features, return_std=True)
            except TypeError:
                result = None
            if isinstance(result, tuple) and len(result) == 2:
                mean, std = (as_target_matrix(values) for values in result)
                if mean.shape != std.shape:
                    raise ValueError("Prediction and uncertainty shapes differ.")
            else:
                mean = as_target_matrix(self.champion_model.predict(features))
                std = np.zeros_like(mean)
            return format_prediction(mean, single=single), format_prediction(
                std, single=single
            )
        return format_prediction(self.champion_model.predict(features), single=single)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            champion = object.__getattribute__(self, "champion_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(champion, name)


class AutoSurrogateStrategy(SurrogateModelStrategy):
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
    ) -> tuple[Any, Metrics]:
        data = prepare_training_data(X, y, config, X_test, y_test)
        # A provided test set is reserved for final evaluation, not selection.
        selection = (
            prepare_training_data(X, y, config) if data.external_holdout else data
        )
        preset = str(config.get("auto_preset", "quick")).lower()
        primary_metric = str(config.get("auto_metric", "R2")).upper()
        metric_keys = {"R2": "r2", "RMSE": "rmse", "MAE": "mae", "RMAE": "rmae"}
        if primary_metric not in metric_keys:
            raise ValueError(f"Unknown Auto Mode metric {primary_metric!r}.")
        candidates = self._build_mdao_candidates(
            preset,
            selection.train_X.shape[1],
            int(config.get("random_state", 42)),
            config.get("auto_max_polynomial_terms"),
        )
        entries = []
        failures = []
        target = (
            selection.train_y[:, 0]
            if selection.train_y.shape[1] == 1
            else selection.train_y
        )
        for index, (name, family, citation, factory) in enumerate(candidates):
            if stop_flag and stop_flag():
                raise RuntimeError("Auto Mode training was cancelled.")
            if callback:
                callback(
                    10 + int(70 * index / len(candidates)),
                    f"Evaluating {name} ({index + 1}/{len(candidates)})...",
                )
            start = time.perf_counter()
            try:
                model = factory().fit(selection.train_X, target)
                fit_time = time.perf_counter() - start
                prediction = as_target_matrix(model.predict(selection.eval_X))
                scores = compute_mdao_metrics(selection.eval_y, prediction)
            except Exception as exc:
                logger.warning("Auto candidate %s failed: %s", name, exc)
                failures.append({"name": name, "error": str(exc)})
                continue
            entries.append(
                {
                    "name": name,
                    "family": family,
                    "citation": citation,
                    "r2": scores["R2"],
                    "rmse": scores["RMSE"],
                    "mae": scores["MAE"],
                    "rmae": scores["RMAE"],
                    "fit_time": round(fit_time, 3),
                    "_model": model,
                    "_errors": selection.eval_y - prediction,
                }
            )
        if not entries:
            raise RuntimeError(
                "All Auto Mode candidates failed: "
                + "; ".join(f["name"] + ": " + f["error"] for f in failures)
            )
        metric = metric_keys[primary_metric]
        reverse = primary_metric == "R2"
        entries.sort(key=lambda entry: entry[metric], reverse=reverse)
        champion = entries[0]
        optimal_weights = None
        if bool(config.get("auto_ensemble", True)) and len(entries) >= 2:
            members = entries[:4]
            # Keep each residual attached to its model through sorting. Signed
            # residuals also preserve useful cancellation between models.
            scales = (
                selection.eval_y.std(axis=0)
                if primary_metric in {"R2", "RMAE"}
                else np.ones(selection.eval_y.shape[1])
            )
            scales = np.where(scales > 1e-12, scales, 1.0)
            errors = np.column_stack(
                [(entry["_errors"] / scales).ravel() for entry in members]
            )
            try:
                ensemble = MdaoEnsembleOfSurrogates(
                    [(entry["name"], entry["_model"]) for entry in members],
                    error_matrix=errors,
                )
                scores = compute_mdao_metrics(
                    selection.eval_y, ensemble.predict(selection.eval_X)
                )
                wins = (
                    scores[primary_metric] >= champion[metric]
                    if reverse
                    else scores[primary_metric] <= champion[metric]
                )
                if wins:
                    champion = {
                        "name": f"Weighted ensemble ({len(members)} models)",
                        "family": "Ensemble",
                        "citation": "Validation-weighted convex combination",
                        "r2": scores["R2"],
                        "rmse": scores["RMSE"],
                        "mae": scores["MAE"],
                        "rmae": scores["RMAE"],
                        "fit_time": sum(e["fit_time"] for e in members),
                        "_model": ensemble,
                    }
                    optimal_weights = ensemble.weights_.tolist()
                    entries.insert(0, champion)
            except Exception as exc:
                logger.warning("Auto ensemble failed: %s", exc)

        champion_model = champion["_model"]
        if data.external_holdout:
            # Refit selected members on all training rows after choosing weights.
            full_target = (
                data.train_y[:, 0] if data.train_y.shape[1] == 1 else data.train_y
            )
            members = (
                champion_model.models
                if isinstance(champion_model, MdaoEnsembleOfSurrogates)
                else [(champion["name"], champion_model)]
            )
            for _, member in members:
                if stop_flag and stop_flag():
                    raise RuntimeError("Auto Mode training was cancelled.")
                member.fit(data.train_X, full_target)
        leaderboard = []
        for rank, entry in enumerate(entries, 1):
            leaderboard.append(
                {
                    "rank": rank,
                    **{k: v for k, v in entry.items() if not k.startswith("_")},
                }
            )
        wrapped = AutoSurrogateModel(
            champion_model,
            champion["name"],
            champion["family"],
            champion["citation"],
            leaderboard,
            optimal_weights,
        )
        prediction = wrapped.predict(data.eval_X)
        metrics = evaluate_predictions(data.eval_y, prediction)
        scores = compute_mdao_metrics(data.eval_y, prediction)
        metrics.update(
            {
                "MAE": scores["MAE"],
                "RMAE": scores["RMAE"],
                "champion_name": champion["name"],
                "champion_family": champion["family"],
                "citation": champion["citation"],
                "leaderboard": leaderboard,
                "auto_preset": preset,
                "auto_metric": primary_metric,
                "failed_candidates": failures,
                "selection_source": selection.evaluation_source,
                "evaluation_source": data.evaluation_source
                if data.external_holdout
                else data.evaluation_source + " (model selection)",
            }
        )
        if callback:
            callback(95, f"Auto Mode complete: {champion['name']}")
        return wrapped, metrics

    def _build_mdao_candidates(
        self,
        preset: str,
        n_features: int,
        random_state: int = 42,
        max_terms: int | None = None,
    ):
        if preset not in {"quick", "standard", "thorough"}:
            raise ValueError(f"Unknown Auto Mode preset {preset!r}.")
        candidates = [
            (
                "Kriging (Matern 5/2 ARD)",
                "Kriging / GP",
                "Sacks et al. (1989)",
                lambda: MdaoKrigingRegressor(n_restarts=4, random_state=random_state),
            ),
            (
                "RBF (Thin-plate spline)",
                "Radial basis",
                "Regis & Shoemaker (2007)",
                lambda: MdaoRBFRegressor(),
            ),
            (
                "Sparse polynomial response surface (degree 2)",
                "Polynomial",
                "Polynomial features + ElasticNet",
                lambda: MdaoPolynomialRegressor(degree=2, max_terms=max_terms),
            ),
            (
                "SVR (RBF kernel)",
                "Support vector regression",
                "Clarke et al. (2005)",
                lambda: MdaoSVRRegressor(),
            ),
        ]
        if n_features >= 3 or preset in {"standard", "thorough"}:
            candidates.append(
                (
                    "PLS + Kriging",
                    "PLS / GP",
                    "PLS projection followed by Kriging",
                    lambda: MdaoKPLSRegressor(
                        n_comp=min(3, n_features),
                        n_restarts=3,
                        random_state=random_state,
                    ),
                )
            )
        if preset in {"standard", "thorough"}:
            candidates.extend(
                [
                    (
                        "Kriging (Matern 3/2 ARD)",
                        "Kriging / GP",
                        "Matern covariance",
                        lambda: MdaoKrigingRegressor(
                            kernel_type="matern32",
                            n_restarts=4,
                            random_state=random_state,
                        ),
                    ),
                    (
                        "RBF (Multiquadric)",
                        "Radial basis",
                        "Gutmann (2001)",
                        lambda: MdaoRBFRegressor(kernel="multiquadric", smoothing=1e-3),
                    ),
                ]
            )
        if preset == "thorough":
            candidates.extend(
                [
                    (
                        "Sparse polynomial response surface (degree 3)",
                        "Polynomial",
                        "Polynomial features + ElasticNet",
                        lambda: MdaoPolynomialRegressor(
                            degree=3, alpha=0.005, l1_ratio=0.7, max_terms=max_terms
                        ),
                    ),
                    (
                        "SVR (Linear kernel)",
                        "Support vector regression",
                        "Clarke et al. (2005)",
                        lambda: MdaoSVRRegressor(C=5.0, epsilon=0.02, kernel="linear"),
                    ),
                    (
                        "Kriging (Gaussian ARD)",
                        "Kriging / GP",
                        "Sacks et al. (1989)",
                        lambda: MdaoKrigingRegressor(
                            kernel_type="gaussian",
                            n_restarts=5,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        return candidates
