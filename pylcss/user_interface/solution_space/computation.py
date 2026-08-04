# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionComputationMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.solution_space.models import (
    MMSSParameters,
    MultiModalResult,
)
from pylcss.solution_space.solver import SolutionSpaceSolver
from pylcss.user_interface.solution_space.resample_worker import ResampleThread
from pylcss.user_interface.solution_space.solver_workers import (
    MultiModalResampleWorker,
    MultiModalSolverWorker,
    SolverWorker,
)


logger = logging.getLogger(__name__)

__all__ = ["SolutionComputationMixin"]


class SolutionComputationMixin:
    def _multimodal_problem_arrays(self):
        dsl = np.array(
            [
                self._safe_get_float(self.dv_table.item(i, 2), np.nan)
                for i in range(self.dv_table.rowCount())
            ]
        )
        dsu = np.array(
            [
                self._safe_get_float(self.dv_table.item(i, 3), np.nan)
                for i in range(self.dv_table.rowCount())
            ]
        )
        req_l = np.array(
            [
                self._safe_get_float(self.qoi_table.item(i, 2), -np.inf)
                for i in range(self.qoi_table.rowCount())
            ]
        )
        req_u = np.array(
            [
                self._safe_get_float(self.qoi_table.item(i, 3), np.inf)
                for i in range(self.qoi_table.rowCount())
            ]
        )
        if (
            dsl.size == 0
            or not np.all(np.isfinite(dsl))
            or not np.all(np.isfinite(dsu))
        ):
            raise ValueError("Design-space bounds must be finite numbers.")
        if np.any(dsu < dsl):
            raise ValueError("Every design-space maximum must be at least its minimum.")
        if req_l.size == 0 or req_l.shape != req_u.shape:
            raise ValueError("At least one quantity-of-interest requirement is needed.")
        if np.any(req_u < req_l):
            raise ValueError("Every requirement maximum must be at least its minimum.")
        return dsl, dsu, req_l, req_u

    def run_multimodal_computation(self):
        """Run the paper's five-stage MMSS algorithm."""
        if self.problem is None:
            return
        if self.multi_modal_worker is not None and self.multi_modal_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Multi-Modal computation",
                "A Multi-Modal computation is already running.",
            )
            return

        self._reset_multimodal_state()

        try:
            import copy

            dsl, dsu, req_l, req_u = self._multimodal_problem_arrays()
            self.dsl, self.dsu = dsl.copy(), dsu.copy()
            self._has_sampled = True

            params = MMSSParameters(
                solver_type=self.mm_solver_combo.currentData(),
                decoupling_enabled=True,
            )

            self.multi_modal_worker = MultiModalSolverWorker(
                problem=copy.deepcopy(self.problem),
                dsl=dsl,
                dsu=dsu,
                reqL=req_l,
                reqU=req_u,
                parameters=None,
                params=params,
                weight=np.ones(len(dsl)),
            )
            self.multi_modal_worker.progress_signal.connect(self.on_multimodal_progress)
            self.multi_modal_worker.finished_signal.connect(self.on_multimodal_finished)
            self.multi_modal_worker.error_signal.connect(self.on_multimodal_error)

            self.btn_compute_multimodal.setEnabled(False)
            self.btn_resample_multimodal.setEnabled(False)
            self.status_msg = QtWidgets.QProgressDialog(
                "Discovering Multi-Modal solution spaces...", "Cancel", 0, 0, self
            )
            self.status_msg.setWindowModality(QtCore.Qt.WindowModal)
            self.status_msg.canceled.connect(self.on_multimodal_cancelled)
            self.status_msg.show()
            self.multi_modal_worker.start()
        except Exception as exc:
            self.btn_compute_multimodal.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "Multi-Modal computation", str(exc))

    def on_multimodal_progress(self, message):
        if hasattr(self, "status_msg") and self.status_msg is not None:
            self.status_msg.setLabelText(str(message))
        self.lbl_multimodal_info.setText(str(message))

    def on_multimodal_cancelled(self):
        if self.multi_modal_worker is not None:
            self.multi_modal_worker.stop()
        if hasattr(self, "status_msg"):
            self.status_msg.close()
        self.btn_compute_multimodal.setEnabled(self.problem is not None)
        self.lbl_multimodal_info.setText("Multi-Modal computation cancelled.")

    def on_multimodal_error(self, error_message):
        if hasattr(self, "status_msg"):
            self.status_msg.close()
        self.btn_compute_multimodal.setEnabled(self.problem is not None)
        self.lbl_multimodal_info.setText("Multi-Modal computation failed.")
        QtWidgets.QMessageBox.critical(
            self, "Multi-Modal computation", f"Computation failed:\n{error_message}"
        )

    def on_multimodal_finished(self, result: MultiModalResult):
        if hasattr(self, "status_msg"):
            self.status_msg.close()
        self.btn_compute_multimodal.setEnabled(self.problem is not None)
        self.multi_modal_result = result
        self.multi_modal_boxes = list(result.boxes)
        self.active_box_index = -1
        self.multimodal_view_mode = "all"

        self.combo_active_box.blockSignals(True)
        self.combo_active_box.clear()
        self.combo_active_box.addItem("All boxes", "all")
        for box in self.multi_modal_boxes:
            self.combo_active_box.addItem(
                f"Box {box.box_id + 1} (Mode {box.box_id + 1})",
                f"box:{box.box_id}",
            )
        if self._get_shared_display_family() is not None:
            self.combo_active_box.addItem(
                "Recommended box (Decoupled form)", "recommended"
            )
        self.combo_active_box.setCurrentIndex(0)
        self.combo_active_box.blockSignals(False)

        self.multibox_table.setRowCount(len(self.multi_modal_boxes))
        for row, box in enumerate(self.multi_modal_boxes):
            values = (
                f"Box {box.box_id + 1} / Mode {box.box_id + 1}",
                f"{box.volume:.3e}",
                f"{box.good_fraction_lower_bound:.4f}",
                str(box.validation_samples),
            )
            tint = QtGui.QColor(self._get_branch_color(row, box))
            tint.setAlpha(45)
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setBackground(tint)
                self.multibox_table.setItem(row, col, item)
        self.multibox_table.resizeColumnsToContents()

        if not self.multi_modal_boxes:
            self.btn_resample_multimodal.setEnabled(False)
            reason = (
                "Feasible finds were clustered, but no positive-volume "
                "solution space survived computation."
                if result.n_clusters_found
                else "No feasible basin was discovered."
            )
            self.lbl_multimodal_info.setText(reason)
            QtWidgets.QMessageBox.warning(self, "Multi-Modal result", reason)
            return

        decoupled = self._get_shared_display_family()
        decoupled_text = ""
        if decoupled is not None:
            names = [dv["name"] for dv in self.problem.design_variables]
            common_names = [names[i] for i in decoupled.common_variable_indices]
            separating_names = [names[i] for i in decoupled.separating_variable_indices]
            common_text = ", ".join(common_names) if common_names else "none"
            separating_text = (
                ", ".join(separating_names) if separating_names else "none"
            )
            decoupled_text = f" | common: {common_text} | separating: {separating_text}"
        self.lbl_multimodal_info.setText(
            f"{len(self.multi_modal_boxes)} mode(s) / box-shaped solution spaces, "
            "total volume "
            f"{result.total_volume:.3e}, {result.computation_time:.1f} s, "
            f"{result.clustering_method}{decoupled_text}"
        )
        self.btn_resample_multimodal.setEnabled(True)
        self._display_all_boxes()
        self.resample_multimodal(silent=True)

    def _get_shared_display_family(self):
        result = self.multi_modal_result
        family = getattr(result, "decoupled_form", None) if result is not None else None
        return family if family is not None and family.is_valid() else None

    def _get_multimodal_display_boxes(self):
        if self.multimodal_view_mode == "recommended":
            family = self._get_shared_display_family()
            return list(family.mode_boxes) if family is not None else []
        if self.multimodal_view_mode == "box":
            return [
                box
                for box in self.multi_modal_boxes
                if box.box_id == self.active_box_index
            ]
        return list(self.multi_modal_boxes)

    def _get_branch_color(self, branch_index, box=None):
        display_boxes = self._get_multimodal_display_boxes()
        if len(display_boxes) <= 1:
            return "#000000"
        if self.multimodal_view_mode == "recommended":
            family = self._get_shared_display_family()
            if family is not None and not family.separating_variable_indices:
                return "#000000"
            return self.box_colors[branch_index % len(self.box_colors)]
        color_index = box.box_id if box is not None else branch_index
        return self.box_colors[color_index % len(self.box_colors)]

    @staticmethod
    def _combine_multimodal_samples(boxes):
        samples = [
            box.samples for box in boxes if box.samples and box.samples["points"].size
        ]
        if not samples:
            return None
        return {
            "points": np.hstack([sample["points"] for sample in samples]),
            "is_good": np.concatenate([sample["is_good"] for sample in samples]),
            "is_bad": np.concatenate([sample["is_bad"] for sample in samples]),
            "violation_idx": np.concatenate(
                [sample["violation_idx"] for sample in samples]
            ),
            "qoi_values": np.hstack([sample["qoi_values"] for sample in samples]),
        }

    def _show_multimodal_samples(self, samples):
        if samples is None:
            self.update_all_plots()
            return
        if isinstance(samples, dict):
            for plot_widget in self.plot_widgets:
                plot_widget.samples = samples
        self.process_results(samples)
        self.lbl_global_title.setText(
            f"Multi-Modal Solution Spaces for {self.problem.name}"
        )

    def on_active_box_changed(self, _index):
        if not self.multi_modal_boxes:
            return
        selection = str(self.combo_active_box.currentData())
        if selection == "all":
            self.multimodal_view_mode = "all"
            self.active_box_index = -1
            self._display_all_boxes()
        elif selection == "recommended":
            self.multimodal_view_mode = "recommended"
            self.active_box_index = -1
            self._display_recommended_box()
        elif selection.startswith("box:"):
            box_id = int(selection.split(":", 1)[1])
            self.multimodal_view_mode = "box"
            for box in self.multi_modal_boxes:
                if box.box_id == box_id:
                    self._display_box(box)
                    break
        self.resample_multimodal(silent=True)

    def on_multibox_table_clicked(self, row, _column):
        if 0 <= row < len(self.multi_modal_boxes):
            self.combo_active_box.setCurrentIndex(row + 1)

    def _display_all_boxes(self):
        self.multimodal_view_mode = "all"
        boxes = list(self.multi_modal_boxes)
        if not boxes:
            return
        self.active_box_index = -1
        self.dv_par_box = boxes[0].bounds.copy()
        self.dv_table.blockSignals(True)
        for row in range(min(self.dv_table.rowCount(), len(self.dv_par_box))):
            self.dv_table.setItem(row, 4, QtWidgets.QTableWidgetItem("multiple"))
            self.dv_table.setItem(row, 5, QtWidgets.QTableWidgetItem("multiple"))
        self.dv_table.blockSignals(False)
        samples = self._combine_multimodal_samples(boxes)
        if samples is None and self.multi_modal_result is not None:
            samples = self.multi_modal_result.samples_all
        self._show_multimodal_samples(samples)

    def _display_recommended_box(self):
        boxes = self._get_multimodal_display_boxes()
        if not boxes:
            return
        self.active_box_index = -1
        self.dv_par_box = boxes[0].bounds.copy()
        stacked = np.stack([box.bounds for box in boxes])
        self.dv_table.blockSignals(True)
        for row in range(min(self.dv_table.rowCount(), stacked.shape[1])):
            lower = stacked[:, row, 0]
            upper = stacked[:, row, 1]
            if np.allclose(lower, lower[0]) and np.allclose(upper, upper[0]):
                low_text, high_text = f"{lower[0]:.6g}", f"{upper[0]:.6g}"
            else:
                low_text = high_text = "multiple"
            self.dv_table.setItem(row, 4, QtWidgets.QTableWidgetItem(low_text))
            self.dv_table.setItem(row, 5, QtWidgets.QTableWidgetItem(high_text))
        self.dv_table.blockSignals(False)
        self._show_multimodal_samples(self._combine_multimodal_samples(boxes))

    def _display_box(self, box):
        self.active_box_index = box.box_id
        # Keep the selected result box connected to the standard black ROI,
        # as in the reference MMSS individual-mode plot.
        self.dv_par_box = box.bounds
        self.dv_table.blockSignals(True)
        for row, bounds in enumerate(box.bounds):
            self.dv_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(f"{bounds[0]:.6g}")
            )
            self.dv_table.setItem(
                row, 5, QtWidgets.QTableWidgetItem(f"{bounds[1]:.6g}")
            )
        self.dv_table.blockSignals(False)
        self._show_multimodal_samples(box.samples)

    def resample_multimodal(self, silent=True):
        """Evaluate fresh per-plot samples around the visible MMSS boxes."""
        boxes = self._get_multimodal_display_boxes()
        if self.problem is None or not boxes:
            return
        self._multimodal_resample_request += 1
        request_id = self._multimodal_resample_request
        current_worker = self.multimodal_resample_worker
        if current_worker is not None and current_worker.isRunning():
            self._multimodal_resample_pending = True
            return
        try:
            dsl, dsu, req_l, req_u = self._multimodal_problem_arrays()
            num_inputs = len(self.inputs)
            active_plots = []
            for widget in self.plot_widgets:
                x_idx = (
                    self.inputs.index(widget.x_name)
                    if widget.x_name in self.inputs
                    else num_inputs + self.outputs.index(widget.x_name)
                )
                y_idx = (
                    self.inputs.index(widget.y_name)
                    if widget.y_name in self.inputs
                    else num_inputs + self.outputs.index(widget.y_name)
                )
                active_plots.append((x_idx, y_idx))

            center_slice = bool(
                hasattr(self, "chk_center_slice") and self.chk_center_slice.isChecked()
            )
            sample_size = self.mm_plot_sample_size_spin.value()
            worker = MultiModalResampleWorker(
                self.problem,
                [box.bounds for box in boxes],
                dsl,
                dsu,
                req_u,
                req_l,
                sample_size,
                active_plots=active_plots,
                center_slice=center_slice,
            )
            worker.request_id = request_id
            worker.silent = bool(silent)
            worker.box_count = len(boxes)
            worker.sample_size = sample_size
            self.multimodal_resample_worker = worker
            self._multimodal_resample_pending = False
            self.btn_resample_multimodal.setEnabled(False)
            worker.result_signal.connect(
                lambda samples, active_worker=worker: self._on_multimodal_resampled(
                    active_worker, samples
                )
            )
            worker.error_signal.connect(
                lambda message, active_worker=worker: self._on_multimodal_resample_error(
                    active_worker, message
                )
            )
            worker.finished.connect(
                lambda active_worker=worker: self._on_multimodal_resample_stopped(
                    active_worker
                )
            )
            worker.finished.connect(worker.deleteLater)
            worker.start()
        except Exception as exc:
            logger.exception("Multi-Modal resampling failed")
            if not silent:
                QtWidgets.QMessageBox.critical(self, "Multi-Modal resampling", str(exc))

    def _on_multimodal_resampled(self, worker, samples):
        if worker is not self.multimodal_resample_worker:
            return
        if worker.request_id != self._multimodal_resample_request:
            return
        self._show_multimodal_samples(samples)
        if not worker.silent:
            self.lbl_multimodal_info.setText(
                f"Plotted {worker.box_count} visible box(es) with "
                f"approximately {worker.sample_size} samples."
            )

    def _on_multimodal_resample_error(self, worker, message):
        if worker is not self.multimodal_resample_worker:
            return
        logger.error("Multi-Modal resampling failed: %s", message)
        if not worker.silent:
            QtWidgets.QMessageBox.critical(self, "Multi-Modal resampling", message)

    def _on_multimodal_resample_stopped(self, worker):
        if worker is not self.multimodal_resample_worker:
            return
        self.multimodal_resample_worker = None
        self.btn_resample_multimodal.setEnabled(bool(self.multi_modal_boxes))
        if self._multimodal_resample_pending:
            self._multimodal_resample_pending = False
            QtCore.QTimer.singleShot(0, lambda: self.resample_multimodal(silent=True))

    def run_computation(self, include_objectives=False):
        # 1. Prepare Problem Object (if not already set)
        if not self.problem:
            return
        self._reset_multimodal_state()
        # Compute is an explicit user action — allow its post-compute sample.
        self._has_sampled = True

        # 2. Gather Parameters for compute_solution_space
        try:
            # DVs
            dsl = []
            dsu = []
            lower_bounds = []
            upper_bounds = []

            for i in range(self.dv_table.rowCount()):
                dsl.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
                dsu.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
                lower_bounds.append(
                    self._safe_get_float(self.dv_table.item(i, 2), -1e9)
                )
                upper_bounds.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))

            dsl = np.array(dsl)
            dsu = np.array(dsu)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            # Store for plotting
            self.dsl = dsl
            self.dsu = dsu

            # QoIs
            reqL = []
            reqU = []

            for i in range(self.qoi_table.rowCount()):
                reqL.append(self._safe_get_float(self.qoi_table.item(i, 2), -1e9))
                reqU.append(self._safe_get_float(self.qoi_table.item(i, 3), 1e9))

            reqL = np.array(reqL)
            reqU = np.array(reqU)

            # Other params
            weight = np.ones(len(dsl))
            parameters = None  # Assuming no fixed parameters for now
            sample_size = self.sample_size_spin.value()
            solver_type = self.solver_combo.currentData()

            # Create solver - use dill for robust problem serialization
            import copy

            # With dill, we can safely pass the problem object directly without manual reconstruction
            problem_to_use = copy.deepcopy(self.problem)

            if not include_objectives:
                for qoi in problem_to_use.quantities_of_interest:
                    qoi["minimize"] = False
                    qoi["maximize"] = False

            status_text = "Computing Solution Space..."

            solver = SolutionSpaceSolver(
                problem_to_use,
                weight,
                dsl,
                dsu,
                lower_bounds,
                upper_bounds,
                reqU,
                reqL,
                parameters,
                solver_type=solver_type,
                include_objectives=include_objectives,
            )
            solver.final_sample_size = sample_size

            self.solver_worker = SolverWorker(solver)
            self.solver_worker.finished_signal.connect(self.on_compute_finished)
            self.solver_worker.progress_signal.connect(self.on_compute_progress)
            self.solver_worker.error_signal.connect(self.on_compute_error)

            # Disable button during computation
            self.btn_compute_feasible.setEnabled(False)
            self.status_msg = QtWidgets.QProgressDialog(
                status_text, "Cancel", 0, 0, self
            )
            self.status_msg.setWindowModality(QtCore.Qt.WindowModal)
            self.status_msg.canceled.connect(self.on_computation_cancelled)
            self.status_msg.show()

            self.solver_worker.start()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Computation Error", str(e))

    def on_compute_progress(self, msg):
        self.status_msg.setLabelText(msg)

    def on_compute_finished(self, box, elapsed_time, samples):
        self.status_msg.close()
        self.btn_compute_feasible.setEnabled(True)
        self.btn_compute_adg.setEnabled(True)
        self.dv_par_box = box

        # Update DV Table with Solution Bounds
        if box is not None:
            self.dv_table.blockSignals(True)
            for i in range(self.dv_table.rowCount()):
                if i < len(box):
                    self.dv_table.setItem(
                        i, 4, QtWidgets.QTableWidgetItem(f"{box[i, 0]:.4f}")
                    )
                    self.dv_table.setItem(
                        i, 5, QtWidgets.QTableWidgetItem(f"{box[i, 1]:.4f}")
                    )
            self.dv_table.blockSignals(False)

        if samples is None and hasattr(self.solver_worker.solver, "latest_results"):
            samples = self.solver_worker.solver.latest_results

        # Store optimal point if objectives were included
        if (
            hasattr(self.solver_worker.solver, "include_objectives")
            and self.solver_worker.solver.include_objectives
        ):
            # The optimal point is the last point added to the samples (the extra_point)
            if (
                samples is not None
                and "points" in samples
                and samples["points"].shape[1] > 0
            ):
                self.optimal_point = samples["points"][:, -1]
            else:
                self.optimal_point = None
        else:
            self.optimal_point = None

        self.process_results(samples)
        self.btn_resample.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "Success", f"Computation complete in {elapsed_time:.2f}s!"
        )

        # Auto-resample to show samples
        self.resample_box(silent=True)

    def on_compute_error(self, error_msg):
        self.status_msg.close()
        self.btn_compute_feasible.setEnabled(True)
        self.btn_compute_adg.setEnabled(True)
        # Check if there are any objectives defined
        has_objectives = self.problem is not None and any(
            qoi.get("minimize", False) or qoi.get("maximize", False)
            for qoi in self.problem.quantities_of_interest
        )
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        QtWidgets.QMessageBox.critical(
            self, "Error", f"Computation failed: {error_msg}"
        )

    def on_computation_cancelled(self):
        self.solver_worker.stop()
        self.status_msg.close()

    def resample_box(self, silent=False):
        # Auto/silent resamples (box drag, first render, add-plot, table writes)
        # are suppressed until the user has sampled once explicitly (Resample
        # button or Compute), so a freshly forwarded model never samples on arrival.
        if silent and not self._has_sampled:
            return
        if self.resampling:
            self.pending_restart = True
            return
        self.resampling = True

        if self.problem is None:
            self.resampling = False
            if not silent:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "No valid model loaded for resampling."
                )
            return

        has_box = False
        self.dv_par_box_mutex.lock()
        try:
            if self.dv_par_box is not None:
                has_box = True
                # Make a deep copy of the box while locked to ensure thread safety
                dv_par_box_copy = self.dv_par_box.copy()
        finally:
            self.dv_par_box_mutex.unlock()

        if not has_box:
            self.resampling = False
            return

        # An explicit sample is now happening — from here on, box drags may live-update.
        self._has_sampled = True

        # Wait for any existing resample thread to finish
        if self.resample_thread and self.resample_thread.isRunning():
            return

        try:
            # Sync tables with problem
            if self.problem and self.qoi_table.rowCount() != len(
                self.problem.quantities_of_interest
            ):
                self.populate_tables_from_problem()

            # Gather bounds again
            def get_val(table, row, col, default):
                item = table.item(row, col)
                if item is None:
                    return default
                text = item.text().strip()
                if not text:
                    return default
                try:
                    return float(text)
                except ValueError:
                    return default

            dsl = []
            dsu = []
            for i in range(self.dv_table.rowCount()):
                dsl.append(get_val(self.dv_table, i, 2, -1e9))
                dsu.append(get_val(self.dv_table, i, 3, 1e9))
            dsl = np.array(dsl)
            dsu = np.array(dsu)

            reqL = []
            reqU = []
            for i in range(self.qoi_table.rowCount()):
                reqL.append(get_val(self.qoi_table, i, 2, -1e9))
                reqU.append(get_val(self.qoi_table, i, 3, 1e9))
            reqL = np.array(reqL)
            reqU = np.array(reqU)

            parameters = None
            sample_size = self.sample_size_spin.value()

            active_plots = []
            num_inputs = len(self.inputs)
            for widget in self.plot_widgets:
                x_name = widget.x_name
                y_name = widget.y_name

                x_idx = -1
                y_idx = -1

                if x_name in self.inputs:
                    x_idx = self.inputs.index(x_name)
                elif x_name in self.outputs:
                    x_idx = num_inputs + self.outputs.index(x_name)

                if y_name in self.inputs:
                    y_idx = self.inputs.index(y_name)
                elif y_name in self.outputs:
                    y_idx = num_inputs + self.outputs.index(y_name)

                if x_idx != -1 and y_idx != -1:
                    active_plots.append((x_idx, y_idx))

            self.btn_resample.setEnabled(False)

            if not silent:
                self.status_msg = QtWidgets.QProgressDialog(
                    "Resampling...", "Cancel", 0, 0, self
                )
                self.status_msg.setWindowModality(QtCore.Qt.WindowModal)
                self.status_msg.show()
            else:
                self.status_msg = None

            # Pass the COPIED box to the thread
            center_slice = bool(
                hasattr(self, "chk_center_slice") and self.chk_center_slice.isChecked()
            )
            self.resample_thread = ResampleThread(
                self.problem,
                dv_par_box_copy,
                dsl,
                dsu,
                reqU,
                reqL,
                parameters,
                sample_size,
                active_plots,
                None,
                center_slice=center_slice,
            )
            self.resample_thread.result_ready.connect(
                lambda s: self.on_resample_finished(s, silent)
            )
            self.resample_thread.error_signal.connect(self.on_resample_error)
            self.resample_thread.finished.connect(self._on_resample_thread_stopped)
            self.resample_thread.finished.connect(self.resample_thread.deleteLater)
            self.resample_thread.start()

        except Exception as e:
            self.resampling = False
            if not silent:
                QtWidgets.QMessageBox.critical(self, "Error", f"Resampling failed: {e}")

    def on_resample_finished(self, samples, silent=False):
        if self.status_msg:
            self.status_msg.close()

        self.resampling = False

        self.btn_resample.setEnabled(True)

        self.process_results(samples, update_table=not silent)
        if not silent:
            QtWidgets.QMessageBox.information(self, "Success", "Resampling complete!")

    def on_resample_error(self, error_msg):
        if self.status_msg:
            self.status_msg.close()

        self.resampling = False

        self.btn_resample.setEnabled(True)
        self.pending_restart = False
        QtWidgets.QMessageBox.critical(self, "Error", f"Resampling failed: {error_msg}")

    def _on_resample_thread_stopped(self):
        self.resample_thread = None
        if self.pending_restart:
            self.pending_restart = False
            QtCore.QTimer.singleShot(0, lambda: self.resample_box(silent=True))
