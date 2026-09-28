# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Responsive validation and feature-analysis UI behavior."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from .workers import EvaluationWorker, create_evaluation_model

__all__ = ["SurrogateEvaluationMixin"]


class SurrogateEvaluationMixin:
    def _evaluation_is_running(self):
        worker = getattr(self, "evaluation_worker", None)
        return worker is not None and worker.isRunning()

    def _start_evaluation(self, operation, **kwargs):
        if self._evaluation_is_running():
            QtWidgets.QMessageBox.information(
                self,
                "Evaluation Running",
                "Wait for the current evaluation to finish.",
            )
            return

        worker = EvaluationWorker(operation, parent=self, **kwargs)
        self.evaluation_worker = worker
        worker.progress_sig.connect(self.update_progress)
        worker.done_sig.connect(self._evaluation_finished)
        worker.error_sig.connect(self._evaluation_failed)
        worker.cancelled_sig.connect(self._evaluation_cancelled)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._evaluation_thread_stopped)
        self._evaluation_control_states = [
            (control, control.isEnabled())
            for control in (
                self.btn_train,
                self.btn_adaptive,
                self.btn_generate,
                self.btn_browse,
                self.btn_save,
                self.combo_algo,
                self.spin_cv_folds,
            )
        ]
        for control, _was_enabled in self._evaluation_control_states:
            control.setEnabled(False)
        for button in (self.btn_run_cv, self.btn_compare, self.btn_feature_imp):
            button.setEnabled(False)
        self.btn_stop.setText("Stop Evaluation")
        self.btn_stop.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        worker.start()

    def _evaluation_thread_stopped(self):
        self.evaluation_worker = None

    def _restore_evaluation_controls(self):
        for control, was_enabled in getattr(
            self, "_evaluation_control_states", ()
        ):
            control.setEnabled(was_enabled)
        self._evaluation_control_states = []
        has_data = self.X_train is not None
        self.btn_run_cv.setEnabled(has_data)
        self.btn_compare.setEnabled(has_data)
        self.btn_feature_imp.setEnabled(
            self.current_model is not None and self.X_test is not None
        )
        self.btn_stop.setText("Stop Training")
        self.btn_stop.setEnabled(False)

    def _run_cross_validation(self):
        """Run K-fold cross-validation outside the GUI thread."""
        if self.X_train is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Generate or load data first."
            )
            return
        n_folds = self.spin_cv_folds.value()
        model_type = self.combo_algo.currentText()
        self.lbl_metrics.setText(f"Running {n_folds}-fold CV for {model_type}...")
        self._start_evaluation(
            "cross_validation",
            X=self.X_train,
            y=self.y_train,
            n_folds=n_folds,
            model_type=model_type,
        )

    def _compare_models(self):
        """Compare all available model types outside the GUI thread."""
        if self.X_train is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Generate or load data first."
            )
            return
        n_folds = self.spin_cv_folds.value()
        self.lbl_metrics.setText("Comparing all model types...")
        self._start_evaluation(
            "compare_models",
            X=self.X_train,
            y=self.y_train,
            n_folds=n_folds,
        )

    def _compute_feature_importance(self):
        """Compute feature importance outside the GUI thread."""
        if self.current_model is None:
            QtWidgets.QMessageBox.warning(self, "No Model", "Train a model first.")
            return
        if self.X_test is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No test data available.")
            return
        feature_names = getattr(self, "input_names", None) or [
            f"X{i}" for i in range(self.X_test.shape[1])
        ]
        self.lbl_metrics.setText("Computing feature importance...")
        self._start_evaluation(
            "feature_importance",
            X=self.X_test,
            y=self.y_test,
            model=self.current_model,
            feature_names=feature_names,
        )

    def _evaluation_finished(self, operation, result):
        self._restore_evaluation_controls()
        self.progress.setValue(100)
        if operation == "cross_validation":
            self._display_cv_results([result])
            self.lbl_metrics.setText(
                f"CV Result: R² = {result.r2_mean:.4f} ± {result.r2_std:.4f}, "
                f"RMSE = {result.rmse_mean:.4f}"
            )
        elif operation == "compare_models":
            self._display_cv_results(result)
            successful = next((item for item in result if item.succeeded), None)
            self.lbl_metrics.setText(
                f"Best: {successful.model_type} "
                f"(R² = {successful.r2_mean:.4f} ± {successful.r2_std:.4f})"
                if successful
                else "No model comparison completed successfully."
            )
        else:
            self._display_feature_importance(result)

    def _evaluation_failed(self, operation, error):
        self._restore_evaluation_controls()
        title = {
            "cross_validation": "CV Failed",
            "compare_models": "Comparison Failed",
            "feature_importance": "Feature Importance Failed",
        }.get(operation, "Evaluation Failed")
        self.lbl_metrics.setText(f"{title}.")
        QtWidgets.QMessageBox.critical(self, title, error)

    def _evaluation_cancelled(self, _operation):
        self._restore_evaluation_controls()
        self.lbl_metrics.setText("Evaluation cancelled.")

    def _display_cv_results(self, results):
        self.cv_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.cv_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(result.model_type)
            )
            self.cv_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{result.r2_mean:.4f}")
            )
            self.cv_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(f"{result.r2_std:.4f}")
            )
            self.cv_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(f"{result.rmse_mean:.4f}")
            )
            self.cv_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(f"{result.mae_mean:.4f}")
            )
        self.cv_table.resizeColumnsToContents()
        self.tab_widget.setCurrentWidget(self.cv_tab)

    def _display_feature_importance(self, result):
        if "error" in result:
            self._evaluation_failed("feature_importance", result["error"])
            return
        feature_names = getattr(self, "input_names", None) or [
            f"X{i}" for i in range(self.X_test.shape[1])
        ]
        importance_key = (
            "importances_mean" if "importances_mean" in result else "importances"
        )
        names = result.get("ranking", feature_names)
        values = result.get(
            "ranking_values",
            result.get(importance_key, []),
        )
        self.fi_plot.clear()
        x_positions = np.arange(len(names))
        self.fi_plot.addItem(
            pg.BarGraphItem(
                x=x_positions,
                height=values,
                width=0.6,
                brush=pg.mkBrush("#27AE60"),
                pen=pg.mkPen("k", width=1),
            )
        )
        self.fi_plot.getAxis("bottom").setTicks(
            [[(index, name) for index, name in enumerate(names)]]
        )
        self.fi_plot.setTitle(
            "Feature Importance (Permutation)"
            if importance_key == "importances_mean"
            else "Feature Importance (Built-in)"
        )
        self.tab_widget.setCurrentWidget(self.fi_tab)
        self.lbl_metrics.setText("Feature importance computed.")

    def _create_sklearn_model(self, model_type: str):
        """Compatibility helper used by existing tests and integrations."""
        return create_evaluation_model(model_type)
