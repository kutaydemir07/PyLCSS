# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Scientific Auto Mode for Multidisciplinary Design Analysis and Optimization (MDAO).

Benchmarks and optimizes state-of-the-art surrogate modeling architectures
published in AIAA, Structural and Multidisciplinary Optimization (SMO),
Surrogate Modeling Toolbox (SMT), and ASME literature:
- Ordinary and Universal Kriging with Anisotropic ARD (Sacks et al. 1989, Jones et al. 1998)
- Kriging by Partial Least Squares / KPLS (Bouhlel et al. 2016)
- Radial Basis Functions with Polyharmonic / Thin Plate Splines and Polynomial Drift (Gutmann 2001, Regis & Shoemaker 2007)
- Sparse Orthogonal Polynomial Chaos Expansion (Blatman & Sudret 2011, Xiu & Karniadakis 2002)
- Anisotropic epsilon-Support Vector Regression (Clarke et al. 2005)
- Goel-Haftka Optimal Convex Ensemble of Surrogates minimizing cross-validation error covariance (Goel et al. 2007, Viana et al. 2009)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from .contracts import LossCallback, Metrics, ProgressCallback, StopFlag
from .mdao_algorithms import (
    MdaoEnsembleOfSurrogates,
    MdaoKPLSRegressor,
    MdaoKrigingRegressor,
    MdaoRBFRegressor,
    MdaoSVRRegressor,
    MdaoSparsePCERegressor,
    compute_mdao_metrics,
)
from .metrics import evaluate_predictions, validate_training_data
from .models import TORCH_AVAILABLE
from .strategies import SurrogateModelStrategy

logger = logging.getLogger(__name__)

__all__ = ["AutoSurrogateModel", "AutoSurrogateStrategy"]


class AutoSurrogateModel:
    """Champion MDAO surrogate model or optimal ensemble with full scientific benchmarking metadata."""

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
        features = np.asarray(X, dtype=np.float64)
        single_sample = features.ndim == 1
        if single_sample:
            features = features.reshape(1, -1)

        if hasattr(self.champion_model, "predict"):
            if return_std:
                try:
                    res = self.champion_model.predict(features, return_std=True)
                    if isinstance(res, tuple) and len(res) == 2:
                        mean, std = res
                        if single_sample:
                            return mean.squeeze(), std.squeeze()
                        return mean, std
                except (TypeError, ValueError):
                    pass
                mean = np.asarray(self.champion_model.predict(features), dtype=np.float64)
                std = np.zeros_like(mean)
                if single_sample:
                    return mean.squeeze(), std.squeeze()
                return mean, std

            pred = np.asarray(self.champion_model.predict(features), dtype=np.float64)
            return pred.squeeze() if single_sample else pred

        raise RuntimeError("Champion model does not implement predict().")

    def __getattr__(self, name: str) -> Any:
        """Expose estimator attributes without forwarding serialization hooks."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Unpickling probes attributes before restoring champion_model. Bypass
        # __getattr__ here so a missing champion cannot recursively look itself up.
        try:
            champion = object.__getattribute__(self, "champion_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(champion, name)


class AutoSurrogateStrategy(SurrogateModelStrategy):
    """AutoML strategy dedicated to scientific MDAO architectures."""

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
        features, targets = validate_training_data(X, y)
        n_samples, n_features = features.shape
        n_targets = targets.shape[1]

        # Holdout split if not provided
        if X_test is not None and y_test is not None and len(X_test) > 0 and len(y_test) > 0:
            eval_X, eval_y = validate_training_data(X_test, y_test, minimum_samples=1)
            train_X, train_y = features, targets
        else:
            val_split = float(config.get("validation_split", 0.2))
            train_X, eval_X, train_y, eval_y = train_test_split(
                features, targets, test_size=val_split, random_state=42
            )

        target_1d = train_y.ravel() if n_targets == 1 else train_y
        eval_target_1d = eval_y.ravel() if n_targets == 1 else eval_y

        preset = str(config.get("auto_preset", "standard")).lower()
        ensemble_enabled = bool(config.get("auto_ensemble", True))
        primary_metric = str(config.get("auto_metric", "R2")).upper()

        candidates = self._build_mdao_candidates(preset, n_samples, n_features, n_targets)
        total_candidates = len(candidates)

        leaderboard: list[dict[str, Any]] = []
        cv_error_columns: list[np.ndarray] = []
        fitted_candidate_models: list[tuple[str, Any]] = []

        for idx, (name, family, citation, model_factory) in enumerate(candidates):
            if stop_flag and stop_flag():
                logger.info("MDAO surrogate benchmark cancelled by user.")
                break

            progress_pct = 10 + int(70 * idx / max(1, total_candidates))
            if callback:
                callback(progress_pct, f"Evaluating [{family}]: {name} ({idx + 1}/{total_candidates})...")

            start_t = time.perf_counter()
            try:
                model = model_factory()
                model.fit(train_X, target_1d)
                fit_time = time.perf_counter() - start_t

                preds = model.predict(eval_X)
                pred_arr = np.asarray(preds, dtype=np.float64)
                if pred_arr.ndim > 1 and pred_arr.shape[1] == 1:
                    pred_arr = pred_arr.ravel()

                eval_m = compute_mdao_metrics(eval_target_1d, pred_arr)
                error_col = np.abs(eval_target_1d - pred_arr)

                entry = {
                    "name": name,
                    "family": family,
                    "citation": citation,
                    "r2": eval_m["R2"],
                    "rmse": eval_m["RMSE"],
                    "mae": eval_m["MAE"],
                    "rmae": eval_m["RMAE"],
                    "fit_time": round(fit_time, 3),
                    "model": model,
                }
                leaderboard.append(entry)
                cv_error_columns.append(error_col)
                fitted_candidate_models.append((name, model))

                if loss_callback:
                    loss_callback({"epoch": idx, "train": eval_m["RMSE"], "val": eval_m["RMSE"]})

            except Exception as e:
                logger.warning("MDAO candidate '%s' failed: %s", name, e)
                continue

        if not leaderboard:
            raise RuntimeError("All automatic surrogate candidates failed to fit.")

        # Rank candidates according to chosen optimization metric
        if primary_metric == "RMSE":
            leaderboard.sort(key=lambda x: x["rmse"])
        elif primary_metric == "MAE":
            leaderboard.sort(key=lambda x: x["mae"])
        elif primary_metric == "RMAE":
            leaderboard.sort(key=lambda x: x["rmae"])
        else:  # Default: R2 Score (Maximize)
            leaderboard.sort(key=lambda x: x["r2"], reverse=True)

        for rank, item in enumerate(leaderboard, 1):
            item["rank"] = rank

        champion_item = leaderboard[0]
        champion_model = champion_item["model"]
        champion_name = champion_item["name"]
        champion_family = champion_item["family"]
        champion_citation = champion_item["citation"]
        optimal_weights = None

        # Build Goel-Haftka Optimal Convex Ensemble if enabled and candidate models succeeded
        if ensemble_enabled and len(fitted_candidate_models) >= 2:
            try:
                # Select top 3-4 viable models with positive R2
                top_candidates = [
                    item for item in leaderboard[:4] if item["r2"] > 0.0
                ]
                if len(top_candidates) >= 2:
                    sub_models = [(it["name"], it["model"]) for it in top_candidates]
                    indices = [leaderboard.index(it) for it in top_candidates]
                    sub_errors = np.column_stack([cv_error_columns[i] for i in indices])

                    eos_model = MdaoEnsembleOfSurrogates(
                        models=sub_models,
                        cv_error_matrix=sub_errors,
                    )
                    eos_preds = eos_model.predict(eval_X)
                    eos_metrics = compute_mdao_metrics(eval_target_1d, eos_preds)

                    # If ensemble matches or outperforms the best individual model
                    if eos_metrics["R2"] >= champion_item["r2"] - 0.005:
                        champion_model = eos_model
                        champion_name = f"Goel-Haftka Optimal Ensemble ({len(top_candidates)} Surrogates)"
                        champion_family = "Ensemble of Surrogates"
                        champion_citation = "Goel, Haftka, Shyy (2007) SMO; Viana et al. (2009) AIAA"
                        optimal_weights = [float(w) for w in eos_model.weights_]

                        ens_entry = {
                            "rank": 1,
                            "name": champion_name,
                            "family": champion_family,
                            "citation": champion_citation,
                            "r2": eos_metrics["R2"],
                            "rmse": eos_metrics["RMSE"],
                            "mae": eos_metrics["MAE"],
                            "rmae": eos_metrics["RMAE"],
                            "fit_time": sum(it["fit_time"] for it in top_candidates),
                        }
                        leaderboard.insert(0, ens_entry)
                        for r, it in enumerate(leaderboard, 1):
                            it["rank"] = r
            except Exception as e:
                logger.warning("Goel-Haftka ensemble construction failed: %s", e)

        clean_leaderboard = [
            {k: v for k, v in item.items() if k != "model"}
            for item in leaderboard
        ]

        if callback:
            callback(
                95,
                f"MDAO Benchmarking Complete. Champion: {champion_name} (R2={leaderboard[0]['r2']:.4f}, RMAE={leaderboard[0]['rmae']:.4f})",
            )

        wrapped = AutoSurrogateModel(
            champion_model=champion_model,
            champion_name=champion_name,
            champion_family=champion_family,
            citation=champion_citation,
            leaderboard=clean_leaderboard,
            optimal_weights=optimal_weights,
        )

        final_preds = wrapped.predict(eval_X)
        final_metrics = evaluate_predictions(eval_target_1d, final_preds)
        final_metrics["champion_name"] = champion_name
        final_metrics["champion_family"] = champion_family
        final_metrics["citation"] = champion_citation
        final_metrics["leaderboard"] = clean_leaderboard
        final_metrics["auto_preset"] = preset
        final_metrics["RMAE"] = leaderboard[0].get("rmae", 0.0)

        return wrapped, final_metrics

    def _build_mdao_candidates(
        self,
        preset: str,
        n_samples: int,
        n_features: int,
        n_targets: int,
    ) -> list[tuple[str, str, str, Any]]:
        """Construct state-of-the-art MDAO surrogate candidates with literature citations."""
        candidates = []

        # 1. Ordinary Kriging with ARD Matern 5/2 (C2 Smooth Physical Response)
        candidates.append(
            (
                "Ordinary Kriging (Matern 5/2 ARD)",
                "Kriging / GP",
                "Sacks et al. (1989); Jones et al. (1998)",
                lambda: MdaoKrigingRegressor(kernel_type="matern52", n_restarts=4),
            )
        )

        # 2. Radial Basis Function (Polyharmonic Thin-Plate Spline with Linear Drift)
        candidates.append(
            (
                "RBF (Thin-Plate Polyharmonic Spline)",
                "Radial Basis Function",
                "Regis & Shoemaker (2007); Powell (1992)",
                lambda: MdaoRBFRegressor(kernel="thin_plate_spline", smoothing=1e-4, degree=1),
            )
        )

        # 3. Sparse Polynomial Chaos Expansion (Orthogonal Legendre Polynomials)
        candidates.append(
            (
                "Sparse Polynomial Chaos (Legendre PCE)",
                "Polynomial Chaos",
                "Blatman & Sudret (2011); Xiu & Karniadakis (2002)",
                lambda: MdaoSparsePCERegressor(degree=2, alpha=0.01, l1_ratio=0.5),
            )
        )

        # 4. Anisotropic epsilon-Support Vector Regression
        candidates.append(
            (
                "Anisotropic epsilon-SVR (RBF Kernel)",
                "Support Vector Regression",
                "Clarke, Griebsch, Simpson (2005) ASME",
                lambda: MdaoSVRRegressor(C=10.0, epsilon=0.05, kernel="rbf"),
            )
        )

        # 5. KPLS (Kriging by Partial Least Squares) - essential for MDAO when features >= 4
        if n_features >= 3 or preset in ("standard", "thorough"):
            candidates.append(
                (
                    "KPLS (Kriging with Partial Least Squares)",
                    "KPLS",
                    "Bouhlel et al. (2016) Struct. Multidisc. Optim.",
                    lambda: MdaoKPLSRegressor(n_comp=min(3, n_features), n_restarts=3),
                )
            )

        if preset in ("standard", "thorough"):
            # 6. Ordinary Kriging with Matern 3/2 (C1 Rough/Shock/Boundary Response)
            candidates.append(
                (
                    "Ordinary Kriging (Matern 3/2 ARD)",
                    "Kriging / GP",
                    "Jones et al. (1998)",
                    lambda: MdaoKrigingRegressor(kernel_type="matern32", n_restarts=4),
                )
            )

            # 7. Multiquadric RBF with Optimized Smoothing
            candidates.append(
                (
                    "Multiquadric RBF (Polynomial Drift)",
                    "Radial Basis Function",
                    "Gutmann (2001) J. Global Optim.",
                    lambda: MdaoRBFRegressor(kernel="multiquadric", smoothing=1e-3, degree=1),
                )
            )

        if preset == "thorough":
            # 8. High-Degree Sparse PCE (Cubic interactions with L1 Sparsity)
            if n_features <= 10:
                candidates.append(
                    (
                        "Sparse PCE (Degree 3 Interaction)",
                        "Polynomial Chaos",
                        "Sudret (2008); Blatman & Sudret (2011)",
                        lambda: MdaoSparsePCERegressor(degree=3, alpha=0.005, l1_ratio=0.7),
                    )
                )

            # 9. SVR with Linear Margin
            candidates.append(
                (
                    "Anisotropic epsilon-SVR (Linear Margin)",
                    "Support Vector Regression",
                    "Clarke et al. (2005)",
                    lambda: MdaoSVRRegressor(C=5.0, epsilon=0.02, kernel="linear"),
                )
            )

            # 10. Gaussian / Squared-Exponential Kriging
            candidates.append(
                (
                    "Gaussian Kriging (Squared Exponential)",
                    "Kriging / GP",
                    "Sacks et al. (1989)",
                    lambda: MdaoKrigingRegressor(kernel_type="gaussian", n_restarts=5),
                )
            )

        return candidates
