# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionRequirementsMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["SolutionRequirementsMixin"]


class SolutionRequirementsMixin:
    def _refresh_design_space(self):
        """Re-read the design-space bounds (cols 2/3) and re-range the plots.

        Columns 2/3 define the sampling domain and the plot axis ranges (via
        self.dsl/self.dsu in PlotWidget.get_bounds), so editing them must update
        the plots — otherwise the design space appears to do nothing.
        """
        dsl, dsu = [], []
        for i in range(self.dv_table.rowCount()):
            dsl.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
            dsu.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
        self.dsl = np.array(dsl)
        self.dsu = np.array(dsu)
        # Keep the problem's design variables in sync with the table.
        if self.problem:
            for i, dv in enumerate(self.problem.design_variables):
                if i < len(dsl):
                    dv["min"] = dsl[i]
                    dv["max"] = dsu[i]
        # The plot axes also span the feasible box, so an old (larger) box keeps
        # the axes from shrinking. Clamp the box into the new design space and
        # mirror it in the box columns (4/5).
        if self.dv_par_box is not None:
            self.dv_par_box_mutex.lock()
            try:
                n = min(len(self.dv_par_box), len(dsl))
                for i in range(n):
                    lo = min(max(self.dv_par_box[i, 0], dsl[i]), dsu[i])
                    hi = min(max(self.dv_par_box[i, 1], dsl[i]), dsu[i])
                    if lo > hi:
                        lo, hi = dsl[i], dsu[i]
                    self.dv_par_box[i, 0] = lo
                    self.dv_par_box[i, 1] = hi
                box_copy = self.dv_par_box.copy()
            finally:
                self.dv_par_box_mutex.unlock()
            self.dv_table.blockSignals(True)
            for i in range(min(self.dv_table.rowCount(), len(box_copy))):
                self.dv_table.setItem(
                    i, 4, QtWidgets.QTableWidgetItem(f"{box_copy[i, 0]:.4g}")
                )
                self.dv_table.setItem(
                    i, 5, QtWidgets.QTableWidgetItem(f"{box_copy[i, 1]:.4g}")
                )
            self.dv_table.blockSignals(False)
        self.update_all_plots()

    def on_dv_table_changed(self, item):
        if self.dv_par_box is None:
            return

        row = item.row()
        col = item.column()

        # Columns 2 (design-space min) and 3 (design-space max) set the sampling
        # domain and the plot axis ranges — re-range the plots so the edit shows.
        if col == 2 or col == 3:
            try:
                float(item.text())
            except (TypeError, ValueError):
                return
            self._refresh_design_space()
            return

        # Columns 4 (Min Sol) and 5 (Max Sol) are editable box bounds
        if col == 4 or col == 5:
            try:
                val = float(item.text())
                # Update dv_par_box - minimize time under lock
                idx = 0 if col == 4 else 1

                self.dv_par_box_mutex.lock()
                try:
                    self.dv_par_box[row, idx] = val
                    # Make a copy of the updated box for thread-safe use
                    self.dv_par_box.copy()
                finally:
                    self.dv_par_box_mutex.unlock()

                # Perform GUI updates and resampling outside the lock
                # Redraw plots to show new box
                self.update_all_plots()

                # Auto resample with the copied data
                self._resample_current_view(silent=True)

            except ValueError:
                pass  # Ignore invalid input

    def on_qoi_table_changed(self, item):
        row = item.row()
        col = item.column()

        if col == 6 or col == 7:  # Minimize or Maximize checkbox
            # Make them mutually exclusive
            self.qoi_table.blockSignals(True)
            if col == 6 and item.checkState() == QtCore.Qt.Checked:
                # Uncheck maximize
                max_item = self.qoi_table.item(row, 7)
                if max_item:
                    max_item.setCheckState(QtCore.Qt.Unchecked)
            elif col == 7 and item.checkState() == QtCore.Qt.Checked:
                # Uncheck minimize
                min_item = self.qoi_table.item(row, 6)
                if min_item:
                    min_item.setCheckState(QtCore.Qt.Unchecked)
            self.qoi_table.blockSignals(False)

            # Update req min max if checked
            if item.checkState() == QtCore.Qt.Checked:
                self.qoi_table.setItem(row, 2, QtWidgets.QTableWidgetItem("-inf"))
                self.qoi_table.setItem(row, 3, QtWidgets.QTableWidgetItem("inf"))
                # Disable the fields
                min_req_item = self.qoi_table.item(row, 2)
                max_req_item = self.qoi_table.item(row, 3)
                if min_req_item:
                    min_req_item.setFlags(
                        min_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )
                if max_req_item:
                    max_req_item.setFlags(
                        max_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )
            else:
                # Enable the fields if neither is checked
                min_checked = (
                    self.qoi_table.item(row, 6).checkState() == QtCore.Qt.Checked
                    if self.qoi_table.item(row, 6)
                    else False
                )
                max_checked = (
                    self.qoi_table.item(row, 7).checkState() == QtCore.Qt.Checked
                    if self.qoi_table.item(row, 7)
                    else False
                )
                if not min_checked and not max_checked:
                    min_req_item = self.qoi_table.item(row, 2)
                    max_req_item = self.qoi_table.item(row, 3)
                    if min_req_item:
                        min_req_item.setFlags(
                            min_req_item.flags() | QtCore.Qt.ItemIsEditable
                        )
                    if max_req_item:
                        max_req_item.setFlags(
                            max_req_item.flags() | QtCore.Qt.ItemIsEditable
                        )

            # Update the problem if it exists
            if self.problem and row < len(self.problem.quantities_of_interest):
                qoi = self.problem.quantities_of_interest[row]
                qoi["minimize"] = (
                    self.qoi_table.item(row, 6).checkState() == QtCore.Qt.Checked
                    if self.qoi_table.item(row, 6)
                    else False
                )
                qoi["maximize"] = (
                    self.qoi_table.item(row, 7).checkState() == QtCore.Qt.Checked
                    if self.qoi_table.item(row, 7)
                    else False
                )

        elif col == 8:  # Weight column
            # Update the problem weight if it exists
            if self.problem and row < len(self.problem.quantities_of_interest):
                try:
                    weight_value = float(item.text())
                    self.problem.quantities_of_interest[row]["weight"] = weight_value
                except ValueError:
                    # Reset to default if invalid
                    item.setText("1.0")
                    if self.problem and row < len(self.problem.quantities_of_interest):
                        self.problem.quantities_of_interest[row]["weight"] = 1.0

    def update_single_dv_row(self, row_idx):
        """Update only a single row in the DV table for performance during dragging."""
        self.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = (
                self.dv_par_box.copy() if self.dv_par_box is not None else None
            )
        finally:
            self.dv_par_box_mutex.unlock()

        if dv_par_box_copy is None or row_idx >= len(dv_par_box_copy):
            return

        self.dv_table.blockSignals(True)
        self.dv_table.setItem(
            row_idx, 4, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[row_idx, 0]:.4f}")
        )
        self.dv_table.setItem(
            row_idx, 5, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[row_idx, 1]:.4f}")
        )
        self.dv_table.blockSignals(False)

    def _reset_multimodal_state(self):
        self._multimodal_resample_request = (
            getattr(self, "_multimodal_resample_request", 0) + 1
        )
        self._multimodal_resample_pending = False
        worker = getattr(self, "multi_modal_worker", None)
        if worker is not None and worker.isRunning():
            worker.stop()
        self.multi_modal_result = None
        self.multi_modal_boxes = []
        self.active_box_index = -1
        self.multimodal_view_mode = "all"
        if hasattr(self, "combo_active_box"):
            self.combo_active_box.blockSignals(True)
            self.combo_active_box.clear()
            self.combo_active_box.blockSignals(False)
        if hasattr(self, "multibox_table"):
            self.multibox_table.setRowCount(0)
        if hasattr(self, "lbl_multimodal_info"):
            self.lbl_multimodal_info.setText("No Multi-Modal result yet.")
        if hasattr(self, "btn_resample_multimodal"):
            self.btn_resample_multimodal.setEnabled(False)
        self.last_samples = None
        for plot_widget in getattr(self, "plot_widgets", []):
            plot_widget.samples = None
