# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""ProductFamilyMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import numpy as np
from PySide6 import QtCore, QtWidgets

from pylcss.user_interface.solution_space.solver_workers import (
    ProductFamilyWorker,
)

from .plotting import (
    VariantRequirementsDialog,
)

logger = logging.getLogger(__name__)

__all__ = ["ProductFamilyMixin"]


class ProductFamilyMixin:
    def compute_product_family(self):
        """
        Compute product family analysis with progress dialog.

        Runs solution space computation for each variant and calculates
        the platform (common feasible region).
        """
        if not self.problem:
            QtWidgets.QMessageBox.warning(self, "Warning", "No valid model loaded.")
            return

        # Check if variants exist
        if (
            not hasattr(self.problem, "requirement_sets")
            or not self.problem.requirement_sets
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "No product variants defined. Please add variants first.",
            )
            return

        # Gather parameters
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

            # QoIs (base requirements)
            reqL = []
            reqU = []

            for i in range(self.qoi_table.rowCount()):
                reqL.append(self._safe_get_float(self.qoi_table.item(i, 2), -1e9))
                reqU.append(self._safe_get_float(self.qoi_table.item(i, 3), 1e9))

            reqL = np.array(reqL)
            reqU = np.array(reqU)

            # Other params
            weight = np.ones(len(dsl))
            parameters = None
            solver_type = self.family_solver_combo.currentData()

            # Create progress dialog
            num_variants = len(self.problem.requirement_sets)
            self.family_progress = QtWidgets.QProgressDialog(
                "Computing Product Family...", "Cancel", 0, num_variants, self
            )
            self.family_progress.setWindowModality(QtCore.Qt.WindowModal)
            self.family_progress.setMinimumDuration(0)
            self.family_progress.show()

            # Create worker thread for product family computation
            self.family_worker = ProductFamilyWorker(
                self.problem,
                weight,
                dsl,
                dsu,
                lower_bounds,
                upper_bounds,
                reqU,
                reqL,
                parameters,
                solver_type,
            )
            self.family_worker.progress_signal.connect(self.on_family_progress)
            self.family_worker.finished_signal.connect(self.on_family_finished)
            self.family_worker.error_signal.connect(self.on_family_error)

            self.btn_compute_family.setEnabled(False)
            self.family_progress.canceled.connect(self.on_family_cancelled)

            self.family_worker.start()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Computation Error", str(e))

    def on_family_progress(self, variant_name, current, total, progress_msg):
        """Update progress dialog for product family computation."""
        if progress_msg:
            self.family_progress.setLabelText(f"{variant_name}: {progress_msg}")
        else:
            self.family_progress.setLabelText(f"Computing variant: {variant_name}")
        self.family_progress.setValue(current)

    def on_family_finished(self, results):
        """Handle completion of product family computation."""
        self.family_progress.close()
        self.btn_compute_family.setEnabled(True)

        if results:
            # Store results for visualization
            self.family_results = results

            # Update family plots using the detailed plotting method
            self.plot_product_family(results)

            # Display communality information if available
            if "Communality" in results and results["Communality"] is not None:
                self.display_communality_info(results["Communality"])

            # Switch to Product Family Analysis tab
            # Find the index of the Product Family Analysis tab
            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == "Product Family Analysis":
                    self.right_tabs.setCurrentIndex(i)
                    break

            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Product family computation complete!\nComputed {len(self.problem.requirement_sets)} variants and platform.",
            )
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No valid results obtained.")

    def display_communality_info(self, communality):
        """Display communality information for design variables."""
        if communality is None or len(communality) == 0:
            return

        # Create a dialog to show communality information
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Design Variable Communality")
        dialog.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Title
        title = QtWidgets.QLabel("Communality per Variable")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Description
        desc = QtWidgets.QLabel(
            "Communality measures the degree to which each design variable is shared/common "
            "across all product variants. A value of 1.0 indicates complete commonality "
            "(same value/range across all variants), while lower values indicate differentiation."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("margin-bottom: 15px;")
        layout.addWidget(desc)

        # Table for communality values
        table = QtWidgets.QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Variable", "Communality", "Interpretation"])
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setRowCount(len(communality))

        # Get variable names
        var_names = []
        if self.problem and self.problem.design_variables:
            var_names = [dv["name"] for dv in self.problem.design_variables]
        else:
            var_names = [f"DV{i + 1}" for i in range(len(communality))]

        for i, comm_val in enumerate(communality):
            # Variable name
            var_name = var_names[i] if i < len(var_names) else f"DV{i + 1}"
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(var_name))

            # Communality value
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{comm_val:.4f}"))

            # Interpretation
            if comm_val >= 0.9:
                interp = "High commonality"
            elif comm_val >= 0.7:
                interp = "Moderate commonality"
            elif comm_val >= 0.5:
                interp = "Low commonality"
            else:
                interp = "High differentiation"
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(interp))

        layout.addWidget(table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        dialog.exec_()

    def on_family_error(self, error_msg):
        """Handle errors in product family computation."""
        self.family_progress.close()
        self.btn_compute_family.setEnabled(True)
        QtWidgets.QMessageBox.critical(
            self, "Error", f"Product family computation failed: {error_msg}"
        )

    def on_family_cancelled(self):
        """Handle cancellation of product family computation."""
        if hasattr(self, "family_worker"):
            self.family_worker.stop()
        self.family_progress.close()
        self.btn_compute_family.setEnabled(True)

    def add_variant(self):
        """Add a new product variant."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Variant", "Variant Name:")
        if ok and name:
            # Check if variant already exists
            for row in range(self.variant_table.rowCount()):
                if self.variant_table.item(row, 0).text() == name:
                    QtWidgets.QMessageBox.warning(
                        self, "Warning", f"Variant '{name}' already exists."
                    )
                    return

            row = self.variant_table.rowCount()
            self.variant_table.insertRow(row)
            self.variant_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.variant_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

            # Add to problem if it exists
            if self.problem:
                self.problem.add_requirement_set(name, {})

    def remove_variant(self):
        """Remove selected product variant."""
        current_row = self.variant_table.currentRow()
        if current_row >= 0:
            name = self.variant_table.item(current_row, 0).text()
            reply = QtWidgets.QMessageBox.question(
                self,
                "Remove Variant",
                f"Are you sure you want to remove variant '{name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.variant_table.removeRow(current_row)
                # Remove from problem if it exists
                if self.problem and name in self.problem.requirement_sets:
                    del self.problem.requirement_sets[name]

    def edit_variant_requirements(self):
        """Edit requirements for selected variant."""
        current_row = self.variant_table.currentRow()
        if current_row < 0:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select a variant first."
            )
            return

        variant_name = self.variant_table.item(current_row, 0).text()

        if not self.problem:
            QtWidgets.QMessageBox.warning(self, "Warning", "No problem loaded.")
            return

        # Create dialog for editing requirements
        dialog = VariantRequirementsDialog(variant_name, self.problem, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Update the requirement set
            overrides = dialog.get_overrides()
            self.problem.requirement_sets[variant_name] = overrides
