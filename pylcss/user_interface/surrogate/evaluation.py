# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateEvaluationMixin behavior for surrogate training."""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from pylcss.surrogate_modeling.validation import (
    CrossValidator,
    FeatureImportanceAnalyzer,
    ModelComparator,
)


logger = logging.getLogger(__name__)

__all__ = ["SurrogateEvaluationMixin"]


class SurrogateEvaluationMixin:
    def _run_cross_validation(self):
        """Run K-fold cross-validation on the current dataset and model config."""
        if not hasattr(self, "X_train") or self.X_train is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Generate or load data first."
            )
            return

        n_folds = self.spin_cv_folds.value()
        model_type = self.combo_model.currentText()

        self.lbl_metrics.setText(f"Running {n_folds}-fold CV for {model_type}...")
        QtWidgets.QApplication.processEvents()

        try:
            cv = CrossValidator()

            def factory():
                return self._create_sklearn_model(model_type)

            result = cv.kfold_cv(
                factory,
                self.X_train,
                self.y_train,
                n_folds=n_folds,
                model_type=model_type,
            )

            self._display_cv_results([result])
            self.lbl_metrics.setText(
                f"CV Result: R² = {result.r2_mean:.4f} ± {result.r2_std:.4f}, "
                f"RMSE = {result.rmse_mean:.4f}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CV Failed", str(e))
            self.lbl_metrics.setText("CV failed.")

    def _compare_models(self):
        """Compare all available model types on the current dataset."""
        if not hasattr(self, "X_train") or self.X_train is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Generate or load data first."
            )
            return

        n_folds = self.spin_cv_folds.value()
        self.lbl_metrics.setText("Comparing all model types...")
        QtWidgets.QApplication.processEvents()

        try:
            comparator = ModelComparator()
            factories = {}
            for name in [
                "MLP Regressor",
                "Random Forest",
                "Gradient Boosting",
                "Gaussian Process",
            ]:
                try:

                    def make_factory(n=name):
                        return lambda: self._create_sklearn_model(n)

                    factories[name] = make_factory()
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)

            results = comparator.compare_models(
                factories, self.X_train, self.y_train, n_folds=n_folds
            )
            self._display_cv_results(results)

            if results:
                best = results[0]
                self.lbl_metrics.setText(
                    f"Best: {best.model_type} (R² = {best.r2_mean:.4f} ± {best.r2_std:.4f})"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Comparison Failed", str(e))
            self.lbl_metrics.setText("Model comparison failed.")

    def _display_cv_results(self, results):
        """Populate the CV results table."""
        self.cv_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.cv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(r.model_type))
            self.cv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{r.r2_mean:.4f}"))
            self.cv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{r.r2_std:.4f}"))
            self.cv_table.setItem(
                i, 3, QtWidgets.QTableWidgetItem(f"{r.rmse_mean:.4f}")
            )
            self.cv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{r.mae_mean:.4f}"))
        self.cv_table.resizeColumnsToContents()
        self.tab_widget.setCurrentWidget(self.cv_tab)

    def _compute_feature_importance(self):
        """Compute and display permutation feature importance."""
        if not hasattr(self, "current_model") or self.current_model is None:
            QtWidgets.QMessageBox.warning(self, "No Model", "Train a model first.")
            return
        if not hasattr(self, "X_test") or self.X_test is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No test data available.")
            return

        self.lbl_metrics.setText("Computing feature importance...")
        QtWidgets.QApplication.processEvents()

        try:
            feature_names = getattr(self, "input_names", None) or [
                f"X{i}" for i in range(self.X_test.shape[1])
            ]

            # Try tree-based importance first (fast)
            fi_result = None
            if hasattr(self.current_model, "feature_importances_"):
                fi_result = FeatureImportanceAnalyzer.tree_feature_importance(
                    self.current_model, feature_names
                )
                imp_key = "importances"
            else:
                fi_result = FeatureImportanceAnalyzer.permutation_importance(
                    self.current_model,
                    self.X_test,
                    self.y_test,
                    feature_names=feature_names,
                    n_repeats=10,
                )
                imp_key = "importances_mean"

            if "error" in fi_result:
                QtWidgets.QMessageBox.warning(self, "Error", fi_result["error"])
                return

            # Plot
            self.fi_plot.clear()
            names = fi_result.get("ranking", feature_names)
            values = fi_result.get("ranking_values", fi_result.get(imp_key, []))
            x = np.arange(len(names))

            bars = pg.BarGraphItem(
                x=x,
                height=values,
                width=0.6,
                brush=pg.mkBrush("#27AE60"),
                pen=pg.mkPen("k", width=1),
            )
            self.fi_plot.addItem(bars)
            ax = self.fi_plot.getAxis("bottom")
            ax.setTicks([[(i, n) for i, n in enumerate(names)]])
            self.fi_plot.setTitle(
                "Feature Importance (Permutation)"
                if imp_key == "importances_mean"
                else "Feature Importance (Built-in)"
            )

            self.tab_widget.setCurrentWidget(self.fi_tab)
            self.lbl_metrics.setText("Feature importance computed.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Feature Importance Failed", str(e))
            self.lbl_metrics.setText("Feature importance failed.")

    def _create_sklearn_model(self, model_type: str):
        """Create an unfitted scikit-learn model based on type and current UI settings."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        if model_type == "MLP Regressor":
            return MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=42,
            )
        elif model_type == "Random Forest":
            return RandomForestRegressor(
                n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
            )
        elif model_type == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )
        elif model_type in ("Gaussian Process", "Gaussian Process (Kriging)"):
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
            return GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, random_state=42
            )
        else:
            # Fallback to MLP
            return MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42
            )
