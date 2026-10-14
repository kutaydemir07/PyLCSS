# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateTrainingMixin behavior for surrogate training."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from pylcss.mdao.surrogate_modeling.training_engine import (
    TORCH_AVAILABLE,
    SurrogateTrainer,
)
from pylcss.mdao.system_modeling.compiler import GraphBuilder

from .workers import (
    AdaptiveTrainingWorker,
    ModelTrainingWorker,
)

if TORCH_AVAILABLE:
    import torch

logger = logging.getLogger(__name__)

__all__ = ["SurrogateTrainingMixin"]


class SurrogateTrainingMixin:
    def stop_training(self):
        evaluation_worker = getattr(self, "evaluation_worker", None)
        if evaluation_worker is not None and evaluation_worker.isRunning():
            evaluation_worker.stop()
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop_flag = True
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)
        if hasattr(self, "adaptive_worker") and self.adaptive_worker.isRunning():
            self.adaptive_worker.stop_flag = True
            self.btn_stop.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def update_loss_plot(self, data):
        epoch = data["epoch"]
        train_loss = data["train"]
        val_loss = data["val"]
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Hide progress text when we start getting real data
        self.progress_text.hide()

        # Throttle GUI updates to prevent freezing during fast training
        current_time = time.time()
        if not hasattr(self, "_last_plot_time"):
            self._last_plot_time = 0

        if current_time - self._last_plot_time > 0.1:  # 10 FPS limit
            # Check if items are actually in the plot item list
            plot_items = self.curve_plot.listDataItems()

            if not hasattr(self, "train_curve") or self.train_curve not in plot_items:
                self.train_curve = self.curve_plot.plot(
                    pen=pg.mkPen("r", width=2), name="Train Loss"
                )

            if not hasattr(self, "val_curve") or self.val_curve not in plot_items:
                self.val_curve = self.curve_plot.plot(
                    pen=pg.mkPen("g", width=2), name="Val Loss"
                )

            # Safe updates
            self.train_curve.setData(np.array(self.epochs), np.array(self.train_losses))
            self.val_curve.setData(np.array(self.epochs), np.array(self.val_losses))
            self.curve_plot.update()
            self.curve_plot.autoRange()
            self._last_plot_time = current_time

    def get_config(self) -> dict[str, Any]:
        """Gather configuration from UI."""
        algo = self.combo_algo.currentText()
        config = {"model_type": algo}

        if algo.startswith("Auto Mode"):
            config["model_type"] = "Auto Mode"
            if hasattr(self, "combo_auto_preset"):
                config["auto_preset"] = self.combo_auto_preset.currentData() or "quick"
                config["auto_metric"] = self.combo_auto_metric.currentData() or "R2"
                config["auto_ensemble"] = self.chk_auto_ensemble.isChecked()

        elif algo in (
            "Your Own Architecture (Deep Learning)",
            "Custom Architecture (Deep Learning)",
            "Your Own Architecture",
            "Custom Architecture",
        ):
            config["model_type"] = "Your Own Architecture (Deep Learning)"
            if hasattr(self, "arch_builder"):
                config.update(self.arch_builder.get_config())
        elif algo in ("Geom-DeepONet", "GINO"):
            # Geometric backbones drive the CAD pipeline directly; they need
            # the CAD path, solver kind, target nodal field, parameter bounds,
            # and sample count. Bounds + names come from the system graph via
            # the existing target-node mechanism.
            config["cad_path"] = self.txt_cad_path.text().strip()
            config["cad_kind"] = self.combo_cad_kind.currentText()
            config["field_name"] = self.combo_field.currentText().strip() or "von_mises"
            config["n_samples"] = self.spin_geom_samples.value()
            config["epochs"] = self.spin_geom_epochs.value()
            config["learning_rate"] = self.spin_geom_lr.value()
            if algo == "Geom-DeepONet":
                config["latent_dim"] = self.spin_donet_latent.value()
                config["trunk_hidden"] = self.spin_donet_trunk.value()
            else:  # GINO
                config["hidden_channels"] = self.spin_gino_channels.value()
                config["grid_size"] = self.spin_gino_grid.value()
                config["fno_modes"] = self.spin_gino_modes.value()
            # input_names / input_bounds are populated by start_training from
            # the current spy-model context (same source the LHS data-gen uses).

        # Default debug mode
        config["debug_mode"] = False
        config["active_learning"] = self.active_learning_values()

        return config

    def start_training(self) -> None:
        config = self.get_config()
        algo = config.get("model_type", "")
        is_geometric = algo in ("Geom-DeepONet", "GINO")

        if is_geometric:
            # Geometric backbones drive the CAD pipeline themselves; they
            # don't use (X, y) generated from a spy model. Validate config
            # instead.
            if not config.get("cad_path"):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Geometric surrogates need a CAD graph file.\n"
                    "Pick one via 'Browse...' in step 3 (Model).",
                )
                return
            if not config.get("field_name"):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Geometric surrogates need a nodal field name (e.g. 'von_mises').",
                )
                return
            # Pull input bounds + names from the same source the LHS data-gen
            # uses: the currently selected target node's spy-model context.
            try:
                bounds, names = self._collect_input_bounds_for_target()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Couldn't determine input bounds from the system graph: {exc}\n"
                    "Make sure a target node is selected and its inputs have ranges set.",
                )
                return
            config["input_bounds"] = bounds
            config["input_names"] = names
        else:
            if not hasattr(self, "X_train") or self.X_train is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "No training data available. Please generate or upload data first.",
                )
                return

        self.btn_train.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.plot_widget.clear()
        self.curve_plot.clear()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        # Recreate curves after clearing
        self.train_curve = self.curve_plot.plot(
            pen=pg.mkPen("r", width=2), name="Train Loss"
        )
        self.val_curve = self.curve_plot.plot(
            pen=pg.mkPen("g", width=2), name="Val Loss"
        )

        # Show appropriate tab and message based on architecture
        model_type = config.get("model_type", "Auto Mode")
        is_neural = any(
            k in model_type
            for k in ("Custom", "Your Own", "PyTorch", "Architecture")
        )

        if is_neural:
            self.tab_widget.setCurrentWidget(self.curve_tab)
            self.progress_text.hide()
        elif "Auto" in model_type:
            self.tab_widget.setCurrentWidget(self.lb_tab)
            self.progress_text.setText("Comparing models…")
            self.progress_text.show()
            self.curve_plot.setXRange(-1, 1)
            self.curve_plot.setYRange(-1, 1)
        else:
            self.tab_widget.setCurrentWidget(self.parity_tab)
            self.progress_text.hide()
            self.curve_plot.setYRange(-1, 1)

        # --- START WORKER (BACKGROUND THREAD) ---
        # For geometric backbones we pass placeholder (X, y) -- the strategy
        # ignores them and uses cad_path / input_bounds from config instead.
        if is_geometric:
            X_tr = np.zeros((1, len(config["input_names"])))
            y_tr = np.zeros((1, 1))
            X_te = np.zeros((0, len(config["input_names"])))
            y_te = np.zeros((0, 1))
        else:
            X_tr, y_tr, X_te, y_te = (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
            )
        self.worker = ModelTrainingWorker(X_tr, y_tr, X_te, y_te, config)
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.loss_sig.connect(self.update_loss_plot)
        self.worker.done_sig.connect(self.training_finished)
        self.worker.start()
        self.btn_stop.setEnabled(True)

    def update_progress(self, val, msg):
        self.progress.setValue(val)
        self.lbl_metrics.setText(msg)

    def training_finished(self, model, metrics, error):
        self.btn_train.setEnabled(True)
        self.btn_adaptive.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stop Training")

        # Hide progress text
        self.progress_text.hide()

        if error:
            QtWidgets.QMessageBox.critical(self, "Training Failed", error)
            self.lbl_metrics.setText("Error occurred.")
            return

        self.current_model = model
        self.current_metrics = metrics
        self.btn_save.setEnabled(True)
        self.progress.setValue(100)
        self.update_training_summary()

        # --- FIX 1: Convert lists back to numpy arrays for plotting ---
        y_test = np.array(metrics["y_test"])
        y_pred = np.array(metrics["y_pred"])
        y_std = (
            np.array(metrics["y_std"])
            if "y_std" in metrics and metrics["y_std"] is not None
            else None
        )

        # --- Learning Curve Display ---
        # Loss curves are already collected during training via callbacks
        # No need to dig into internal model attributes

        # Update the plot if we have loss data
        if self.epochs and self.train_losses:
            # Ensure curves exist
            if (
                not hasattr(self, "train_curve")
                or self.train_curve not in self.curve_plot.items()
            ):
                self.train_curve = self.curve_plot.plot(
                    pen=pg.mkPen("r", width=2), name="Train Loss"
                )
            if (
                not hasattr(self, "val_curve")
                or self.val_curve not in self.curve_plot.items()
            ):
                self.val_curve = self.curve_plot.plot(
                    pen=pg.mkPen("g", width=2), name="Val Loss"
                )

            self.train_curve.setData(np.array(self.epochs), np.array(self.train_losses))
            if self.val_losses and len(self.val_losses) == len(self.train_losses):
                self.val_curve.setData(np.array(self.epochs), np.array(self.val_losses))

            self.curve_plot.setTitle("Learning Curve")
            self.curve_plot.update()
            self.curve_plot.autoRange()

        # Update metrics text
        msg = f"<b>Completed</b> · RMSE: {metrics['RMSE']:.4f} · R\u00b2: {metrics['R2']:.4f}"
        if "RMAE" in metrics:
            msg += f" · RMAE: {metrics['RMAE']:.4f}"
        if "champion_name" in metrics:
            msg += f"<br>Selected model: <b>{metrics['champion_name']}</b>"
        if y_std is not None:
            msg += f"<br>Mean Uncertainty: {np.mean(y_std):.4f}"

        if metrics.get("debug_mode", False):
            msg += "<br>Evaluation: debug mode"

        self.lbl_metrics.setText(msg)

        # Populate AutoML Leaderboard if available
        if hasattr(self, "lb_table"):
            self.lb_table.setRowCount(0)
        if "leaderboard" in metrics and hasattr(self, "lb_table"):
            lb_data = metrics["leaderboard"]
            self.lb_table.setRowCount(len(lb_data))
            for row_idx, item in enumerate(lb_data):
                self.lb_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(item.get("rank", row_idx + 1))))
                self.lb_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(str(item.get("name", "-"))))
                self.lb_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{item.get('r2', 0.0):.4f}"))
                self.lb_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(f"{item.get('rmse', 0.0):.4f}"))
                self.lb_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(f"{item.get('rmae', 0.0):.4f}"))
                self.lb_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(f"{item.get('fit_time', 0.0):.3f}"))
            self.tab_widget.setCurrentWidget(self.lb_tab)
        else:
            self.tab_widget.setCurrentWidget(self.parity_tab)

        # Handle multi-output visualization
        self.plot_widget.clear()
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            # Multi-output: plot each output separately with different colors
            colors = [
                (255, 100, 100),
                (100, 255, 100),
                (100, 100, 255),
                (255, 255, 100),
                (255, 100, 255),
            ]
            for i in range(min(y_test.shape[1], len(colors))):
                y_test_i = y_test[:, i]
                y_pred_i = y_pred[:, i]
                color = colors[i % len(colors)]

                # Plot points for this output
                self.plot_widget.plot(
                    y_test_i,
                    y_pred_i,
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=color + (150,),
                    name=f"Output {i + 1}",
                )

                # Add diagonal line for this output
                if len(y_test_i) > 0:
                    mn, mx = float(np.min(y_test_i)), float(np.max(y_test_i))
                    self.plot_widget.plot(
                        [mn, mx],
                        [mn, mx],
                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                    )

            self.plot_widget.setTitle(
                f"Parity Plot (Multi-Output) - R² = {metrics['R2']:.4f}"
            )
            self.plot_widget.addLegend()
        else:
            # Single output: flatten if needed and plot normally
            if y_test.ndim > 1:
                y_test = y_test.flatten()
                y_pred = y_pred.flatten()
                if y_std is not None and y_std.ndim > 1:
                    y_std = y_std.flatten()

            self.plot_widget.plot(
                y_test,
                y_pred,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=(100, 100, 255, 150),
            )

            # Add diagonal line
            if len(y_test) > 0:
                mn, mx = float(np.min(y_test)), float(np.max(y_test))
                self.plot_widget.plot(
                    [mn, mx],
                    [mn, mx],
                    pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine),
                )

            self.plot_widget.setTitle(
                f"Parity Plot (Predicted vs Actual) - R² = {metrics['R2']:.4f}"
            )

        # Add uncertainty bands (mostly for Gaussian Process) - only for single output
        if (
            y_std is not None
            and len(y_std) > 0
            and (y_test.ndim == 1 or (y_test.ndim == 2 and y_test.shape[1] == 1))
        ):
            if y_test.ndim > 1:
                y_test_flat = y_test.flatten()
                y_pred_flat = y_pred.flatten()
                y_std_flat = y_std.flatten()
            else:
                y_test_flat, y_pred_flat, y_std_flat = y_test, y_pred, y_std

            sort_idx = np.argsort(y_test_flat)
            y_test_sorted = y_test_flat[sort_idx]
            y_pred_sorted = y_pred_flat[sort_idx]
            y_std_sorted = y_std_flat[sort_idx]

            upper = y_pred_sorted + 2 * y_std_sorted
            lower = y_pred_sorted - 2 * y_std_sorted

            fill = pg.FillBetweenItem(
                pg.PlotDataItem(y_test_sorted, upper, pen="g"),
                pg.PlotDataItem(y_test_sorted, lower, pen="g"),
                brush=(0, 255, 0, 50),
            )
            self.plot_widget.addItem(fill)
            self.plot_widget.plot(
                y_test_sorted, y_pred_sorted, pen=pg.mkPen("g", width=2)
            )

    def save_model(self) -> None:
        if not self.current_model:
            return

        idx = self.combo_nodes.currentIndex()
        target_node = self.combo_nodes.itemData(idx)

        # Generate filename based on node ID, save in user data directory
        safe_id = target_node.id.replace("-", "_")

        from pylcss.config import BASE_DIR
        base_dir = BASE_DIR
        folder = os.path.join(base_dir, "data_surrogate")

        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                # Fallback to current working directory if permission denied
                folder = os.path.join(os.getcwd(), "data_surrogate")
                os.makedirs(folder, exist_ok=True)

        fname = os.path.join(folder, f"surrogate_{safe_id}.joblib")

        try:
            if (
                TORCH_AVAILABLE
                and hasattr(self.current_model, "model")
                and isinstance(self.current_model.model, torch.nn.Module)
            ):
                self.current_model.model.cpu()  # Move to CPU before serialization
                self.current_model.device = torch.device("cpu")  # Update wrapper state

            joblib.dump(self.current_model, fname)

            # Update Node Properties automatically
            target_node.set_property("surrogate_model_path", fname)
            target_node.set_property("use_surrogate", True)
            target_node.set_property(
                "surrogate_status",
                f"Trained ({self.combo_algo.currentText()}, R\u00b2={self.current_metrics['R2']:.2f})",
            )

            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Model saved to '{fname}' and attached to node.\n"
                "The node is now set to use the surrogate model.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def start_adaptive_training(self) -> None:
        if not hasattr(self, "X_train") or self.X_train is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "No training data available. Please generate or upload data first.",
            )
            return

        idx = self.combo_nodes.currentIndex()
        if idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No node selected.")
            return

        target_node = self.combo_nodes.itemData(idx)
        config = self.get_config()

        # Check for PyTorch with zero dropout (breaks uncertainty estimation)
        if (
            config.get("model_type") == "Deep Neural Network (PyTorch)"
            and config.get("dropout", 0.0) == 0.0
        ):
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Zero Dropout Warning",
                "PyTorch model has 0% dropout, so uncertainty estimation for adaptive sampling is disabled.\n\n"
                "Continue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        self.btn_adaptive.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.plot_widget.clear()
        self.curve_plot.clear()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.tab_widget.setCurrentWidget(self.curve_tab)

        # Recreate curves after clearing
        self.train_curve = self.curve_plot.plot(
            pen=pg.mkPen("r", width=2), name="Train Loss"
        )
        self.val_curve = self.curve_plot.plot(
            pen=pg.mkPen("g", width=2), name="Val Loss"
        )

        # Show adaptive training message
        self.progress_text.setText("Adaptive training…")
        self.progress_text.show()
        self.curve_plot.setXRange(-1, 1)
        self.curve_plot.setYRange(-1, 1)

        # Prepare spy model for evaluation
        try:
            graph = self.modeling_widget.current_graph
            nodes = graph.all_nodes()
            input_nodes = [n for n in nodes if n.type_.startswith("com.pfd.input")]
            output_nodes = [n for n in nodes if n.type_.startswith("com.pfd.output")]

            builder = GraphBuilder(graph)
            spy_code, spy_inputs, spy_outputs = builder.build_spy_model(
                nodes, input_nodes, output_nodes, target_node.id, "spy_model"
            )

            input_bounds = []
            for inp_node in input_nodes:
                if inp_node.has_property("input_props"):
                    props = inp_node.get_property("input_props")
                    min_val = float(props.get("min", "0.0"))
                    max_val = float(props.get("max", "10.0"))
                else:
                    min_val = float(inp_node.get_property("min"))
                    max_val = float(inp_node.get_property("max"))
                input_bounds.append((min_val, max_val))

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preparation Error", str(e))
            self.btn_adaptive.setEnabled(True)
            self.btn_train.setEnabled(True)
            return

        # Start adaptive training worker
        self.adaptive_worker = AdaptiveTrainingWorker(
            SurrogateTrainer(),
            spy_code,
            spy_inputs,
            spy_outputs,
            input_bounds,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            config,
        )
        self.adaptive_worker.progress_sig.connect(self.update_progress)
        self.adaptive_worker.done_sig.connect(self.adaptive_training_finished)
        self.adaptive_worker.start()
        self.btn_stop.setEnabled(True)

    def adaptive_training_finished(self, model, metrics, train_X, train_y, error):
        self.btn_adaptive.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stop Training")

        # Hide progress text
        self.progress_text.hide()

        if error:
            QtWidgets.QMessageBox.critical(self, "Adaptive Training Failed", error)
            self.lbl_metrics.setText("Adaptive training failed.")
            return

        self.current_model = model
        self.current_metrics = metrics
        self.X_train = np.asarray(train_X)
        self.y_train = np.asarray(train_y)
        if hasattr(self, "update_data_table"):
            self.update_data_table()
        self.btn_save.setEnabled(True)
        self.progress.setValue(100)
        self.update_training_summary()

        # Update metrics display (similar to training_finished)
        msg = f"<b>Completed</b> · RMSE: {metrics['RMSE']:.4f} · R²: {metrics['R2']:.4f}"
        if metrics.get("debug_mode", False):
            msg += "<br>Evaluation: debug mode"
        if "y_std" in metrics and metrics["y_std"] is not None:
            y_std = np.array(metrics["y_std"])
            msg += f"<br>Mean Uncertainty: {np.mean(y_std):.4f}"
        active_info = metrics.get("active_learning", {})
        if active_info:
            msg += (
                f"<br>New simulations: {active_info.get('new_samples', 0)}"
            )
            if active_info.get("failed_evaluations"):
                msg += (
                    "<br>Failed simulations: "
                    f"{active_info['failed_evaluations']}"
                )

        self.lbl_metrics.setText(msg)

        # Switch to parity plot tab to show results
        self.tab_widget.setCurrentWidget(self.parity_tab)

        # Handle multi-output visualization
        y_test = np.array(metrics["y_test"])
        y_pred = np.array(metrics["y_pred"])
        y_std = (
            np.array(metrics["y_std"])
            if "y_std" in metrics and metrics["y_std"] is not None
            else None
        )

        self.plot_widget.clear()
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            # Multi-output: plot each output separately with different colors
            colors = [
                (255, 100, 100),
                (100, 255, 100),
                (100, 100, 255),
                (255, 255, 100),
                (255, 100, 255),
            ]
            for i in range(min(y_test.shape[1], len(colors))):
                y_test_i = y_test[:, i]
                y_pred_i = y_pred[:, i]
                color = colors[i % len(colors)]

                # Plot points for this output
                self.plot_widget.plot(
                    y_test_i,
                    y_pred_i,
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=color + (150,),
                    name=f"Output {i + 1}",
                )

                # Add diagonal line for this output
                if len(y_test_i) > 0:
                    mn, mx = float(np.min(y_test_i)), float(np.max(y_test_i))
                    self.plot_widget.plot(
                        [mn, mx],
                        [mn, mx],
                        pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                    )

            self.plot_widget.setTitle(
                f"Parity Plot (Multi-Output Adaptive) - R² = {metrics['R2']:.4f}"
            )
            self.plot_widget.addLegend()
        else:
            # Single output: flatten if needed and plot normally
            if y_test.ndim > 1:
                y_test = y_test.flatten()
                y_pred = y_pred.flatten()
                if y_std is not None and y_std.ndim > 1:
                    y_std = y_std.flatten()

            self.plot_widget.plot(
                y_test,
                y_pred,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=(100, 100, 255, 150),
            )

            # Add diagonal line
            if len(y_test) > 0:
                mn, mx = float(np.min(y_test)), float(np.max(y_test))
                self.plot_widget.plot(
                    [mn, mx],
                    [mn, mx],
                    pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine),
                )

            self.plot_widget.setTitle(
                f"Parity Plot (Adaptive) - R² = {metrics['R2']:.4f}"
            )
