# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionPlotMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets


from ..common.qt_patches import NumericTableWidgetItem
from ..common.text_utils import format_html
from .plotting import (
    PlotWidget,
)

logger = logging.getLogger(__name__)

__all__ = ["SolutionPlotMixin"]


class SolutionPlotMixin:
    def clear_all_plots(self):
        for widget in self.plot_widgets:
            self.plots_layout.removeWidget(widget)
            widget.deleteLater()
        self.plot_widgets = []

    def _export_vector_layout(
        self, filename, widgets, main_title, legend_type="solution"
    ):
        """
        Helper to export a grid of plots to a vector format (PDF/SVG).
        Renders the actual scene items rather than a raster screenshot.
        """
        from PySide6.QtPrintSupport import QPrinter
        from PySide6.QtSvg import QSvgGenerator

        # 1. Setup Device
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(filename)
            printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
            device = printer
            # Use printer's rect (high resolution pixels)
            rect = printer.pageRect(QPrinter.DevicePixel)
            width = rect.width()
            height = rect.height()
        else:
            device = QSvgGenerator()
            device.setFileName(filename)
            # Arbitrary high-res canvas for SVG
            width, height = 1200, 1600
            device.setSize(QtCore.QSize(width, height))
            device.setViewBox(QtCore.QRect(0, 0, width, height))

        painter = QtGui.QPainter(device)
        if not painter.isActive():
            raise RuntimeError("Could not initialize the vector export device.")

        try:
            # 2. Layout Constants
            margin = width * 0.05
            content_w = width - 2 * margin

            # 3. Draw Main Title
            font_title = QtGui.QFont("Segoe UI", int(width * 0.025), QtGui.QFont.Bold)
            painter.setFont(font_title)
            title_rect = QtCore.QRectF(0, margin, width, height * 0.05)
            painter.drawText(title_rect, QtCore.Qt.AlignCenter, main_title)

            cursor_y = margin + height * 0.06

            # 4. Grid Layout for Plots
            if not widgets:
                return

            cols = max(1, getattr(self, "plot_columns", 2))
            rows = (len(widgets) + cols - 1) // cols

            # Calculate areas
            legend_height_est = height * 0.15
            plots_height = height - cursor_y - legend_height_est - margin

            cell_gap = margin * 0.5
            cell_w = (content_w - (cols - 1) * cell_gap) / cols
            cell_h = (plots_height - (rows - 1) * cell_gap) / rows

            # Prevent squashing
            if cell_h < cell_w * 0.5:
                cell_h = cell_w * 0.5
                # In a real app we might handle pagination here,
                # but for single-page summary we let it overflow or clip if too many plots.

            for i, widget in enumerate(widgets):
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ExcludeUserInputEvents
                )
                r = i // cols
                c = i % cols

                x = margin + c * (cell_w + cell_gap)
                y = cursor_y + r * (cell_h + cell_gap)

                # Draw Sub-Title
                sub_title = widget.lbl_title.text()
                sub_font = QtGui.QFont("Segoe UI", int(cell_w * 0.05), QtGui.QFont.Bold)
                painter.setFont(sub_font)

                # Title area
                sub_title_h = cell_h * 0.1
                sub_rect = QtCore.QRectF(x, y, cell_w, sub_title_h)
                painter.drawText(sub_rect, QtCore.Qt.AlignCenter, sub_title)

                # Render Plot Scene
                # We target the specific QGraphicsScene of the widget
                scene = widget.plot_widget.scene()
                plot_target = QtCore.QRectF(
                    x, y + sub_title_h, cell_w, cell_h - sub_title_h
                )

                # Save painter state before render
                painter.save()
                # Clip to target rect to prevent spillover
                painter.setClipRect(plot_target)

                # Render
                scene.render(painter, plot_target, scene.sceneRect())
                painter.restore()

                # Draw border
                painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(plot_target)

            # 5. Draw Legend
            legend_y = cursor_y + rows * (cell_h + cell_gap) + margin

            # Legend Title
            painter.setFont(QtGui.QFont("Segoe UI", int(width * 0.015), QtGui.QFont.Bold))
            painter.drawText(
                QtCore.QRectF(margin, legend_y, content_w, 30),
                QtCore.Qt.AlignLeft,
                "Legend",
            )

            legend_content_y = legend_y + 40
            item_h = 30
            icon_size = 20

            # Prepare Items
            items = []
            if legend_type == "solution":
                items.append(("Good Design", "#00aa00"))

                # Get violations
                if hasattr(self, "outputs") and self.outputs:
                    names = self.outputs
                elif self.problem:
                    names = [q["name"] for q in self.problem.quantities_of_interest]
                else:
                    names = []

                for name in names:
                    color = self.qoi_colors.get(name, "red")
                    items.append((f"Violating {name}", color))
            else:
                items.append(("Good Design", "#00aa00"))
                items.append(("Bad Design", "#ff0000"))

            # Draw items (Simple Flow Layout)
            lx = margin
            ly = legend_content_y

            font_leg = QtGui.QFont("Segoe UI", int(width * 0.012))
            painter.setFont(font_leg)
            metrics = QtGui.QFontMetrics(font_leg)

            for label, color_code in items:
                # Icon
                painter.setBrush(QtGui.QColor(color_code))
                painter.setPen(QtCore.Qt.black)
                painter.drawRect(lx, ly, icon_size, icon_size)

                # Text
                text_w = metrics.horizontalAdvance(label)
                text_rect = QtCore.QRectF(
                    lx + icon_size + 10, ly, text_w + 10, icon_size
                )
                painter.setPen(QtCore.Qt.black)
                painter.drawText(
                    text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label
                )

                # Advance cursor
                lx += icon_size + 10 + text_w + 30
                # Wrap line if needed
                if lx > width - margin:
                    lx = margin
                    ly += item_h + 5

        finally:
            painter.end()

    def save_all_plots(self):
        if getattr(self, "_plot_export_busy", False):
            QtWidgets.QMessageBox.information(
                self, "Save All Plots", "A plot export is already running."
            )
            return
        if self.has_active_background_tasks():
            QtWidgets.QMessageBox.information(
                self,
                "Save All Plots",
                "Wait for solution-space calculations and plot updates to finish.",
            )
            return
        if not self.plot_widgets:
            QtWidgets.QMessageBox.warning(self, "Warning", "No plots to save.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save All Plots",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf);;SVG Image (*.svg)",
        )
        if not filename:
            return

        self._plot_export_busy = True
        self.btn_save_all.setEnabled(False)
        try:
            # Prepare UI (hide buttons) for raster screenshot consistency
            for widget in self.plot_widgets:
                widget.btn_remove.hide()
                widget.btn_zoom.hide()
                widget.btn_save.hide()
                widget.lbl_title.hide()

            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ExcludeUserInputEvents
            )

            if filename.lower().endswith(".pdf") or filename.lower().endswith(".svg"):
                # Use new vector export
                title = "Solution Spaces"
                if hasattr(self, "lbl_global_title"):
                    title = self.lbl_global_title.text()
                self._export_vector_layout(
                    filename, self.plot_widgets, title, "solution"
                )
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Plots saved to {filename}"
                )
            else:
                # Existing raster logic for PNG/JPG
                scale_factor = 2.0  # High quality raster

                plots_pixmap = self.plots_container.grab()
                if scale_factor > 1.0:
                    plots_pixmap = plots_pixmap.scaled(
                        plots_pixmap.size() * scale_factor,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )

                # Legend grab
                old_style = self.legend_group.styleSheet()
                self.legend_group.setStyleSheet(
                    "QGroupBox { background-color: white; color: black; border: none; } QLabel { color: black; }"
                )
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ExcludeUserInputEvents
                )
                legend_pixmap = self.legend_group.grab()
                if scale_factor > 1.0:
                    legend_pixmap = legend_pixmap.scaled(
                        legend_pixmap.size() * scale_factor,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                self.legend_group.setStyleSheet(old_style)

                # Compose
                padding = int(20 * scale_factor)
                title_height = int(60 * scale_factor)
                total_width = plots_pixmap.width() + legend_pixmap.width() + padding
                total_height = (
                    max(plots_pixmap.height(), legend_pixmap.height()) + title_height
                )

                final_pixmap = QtGui.QPixmap(total_width, total_height)
                final_pixmap.fill(QtCore.Qt.white)

                painter = QtGui.QPainter(final_pixmap)
                title_text = (
                    self.lbl_global_title.text()
                    if hasattr(self, "lbl_global_title")
                    else "Solution Spaces"
                )
                font = QtGui.QFont("Segoe UI", int(20 * scale_factor), QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtCore.Qt.black)
                rect = QtCore.QRect(0, 0, total_width, title_height)
                painter.drawText(rect, QtCore.Qt.AlignCenter, title_text)

                painter.drawPixmap(0, title_height, plots_pixmap)
                painter.drawPixmap(
                    plots_pixmap.width() + padding, title_height, legend_pixmap
                )
                painter.end()

                if not final_pixmap.save(filename):
                    raise RuntimeError(
                        "Qt could not encode the selected image format or write the file."
                    )
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Plots saved to {filename}"
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save plots: {e}")
        finally:
            self._plot_export_busy = False
            self.btn_save_all.setEnabled(True)
            # Restore UI
            for widget in self.plot_widgets:
                widget.btn_remove.show()
                widget.btn_zoom.show()
                widget.btn_save.show()
                widget.lbl_title.show()

    def _on_plot_columns_changed(self, value):
        self.plot_columns = max(1, int(value))
        # Re-place existing plots into the new column count.
        while self.plots_layout.count():
            self.plots_layout.takeAt(0)
        for i, widget in enumerate(self.plot_widgets):
            self.plots_layout.addWidget(
                widget, i // self.plot_columns, i % self.plot_columns
            )
        self.plots_container.update()

    def add_plot(self, x_name=None, y_name=None, do_resample=True):
        if x_name is None or not isinstance(x_name, str):
            x_name = self.combo_add_x.currentText()
        if y_name is None or not isinstance(y_name, str):
            y_name = self.combo_add_y.currentText()
        if not x_name or not y_name:
            return

        # Allow plotting both inputs and outputs
        valid_vars = self.inputs + self.outputs
        if x_name not in valid_vars or y_name not in valid_vars:
            return

        plot_widget = PlotWidget(x_name, y_name)
        plot_widget.parent_widget = self
        plot_widget.setFixedHeight(350)
        plot_widget.btn_remove.clicked.connect(lambda: self.remove_plot(plot_widget))
        plot_widget.plot()
        count = len(self.plot_widgets)
        row = count // self.plot_columns
        col = count % self.plot_columns
        self.plots_layout.addWidget(plot_widget, row, col)
        self.plot_widgets.append(plot_widget)
        self.plots_container.update()
        if do_resample:
            self._resample_current_view(silent=True)

    def remove_plot(self, plot_widget):
        if plot_widget in self.plot_widgets:
            self.plot_widgets.remove(plot_widget)
            plot_widget.deleteLater()

            while self.plots_layout.count():
                self.plots_layout.takeAt(0)

            for i, widget in enumerate(self.plot_widgets):
                row = i // self.plot_columns
                col = i % self.plot_columns
                self.plots_layout.addWidget(widget, row, col)

            self.plots_container.update()
            self._resample_current_view(silent=True)

    def process_results(self, samples, update_table=True):
        # 1. Store Global Samples
        if isinstance(samples, list) and len(samples) > 0:
            self.last_samples = {
                "points": np.hstack([s["points"] for s in samples]),
                "is_good": np.concatenate([s["is_good"] for s in samples]),
                "is_bad": np.concatenate([s["is_bad"] for s in samples]),
                "violation_idx": np.concatenate([s["violation_idx"] for s in samples]),
                "qoi_values": np.hstack([s["qoi_values"] for s in samples]),
            }
        elif isinstance(samples, dict):
            self.last_samples = samples
        else:
            self.last_samples = None

        # 2. Assign samples to Design Space Plots (Keep existing logic)
        if isinstance(samples, list):
            for i, widget in enumerate(self.plot_widgets):
                if i < len(samples):
                    widget.samples = samples[i]
                else:
                    widget.samples = None

        if self.last_samples and isinstance(self.last_samples, dict):
            self.problem.samples = {}
            for i, dv in enumerate(self.problem.design_variables):
                self.problem.samples[dv["name"]] = self.last_samples["points"][i, :]
            self.problem.results = {}
            for i, qoi in enumerate(self.problem.quantities_of_interest):
                self.problem.results[qoi["name"]] = self.last_samples["qoi_values"][
                    i, :
                ]

        self.update_all_plots()
        if update_table:
            self.update_data_table()

        model_name = "Unknown Model"
        if self.problem and hasattr(self.problem, "name"):
            model_name = self.problem.name
        if self.multi_modal_boxes:
            self.lbl_global_title.setText(
                f"Multi-Modal Solution Spaces for {model_name}"
            )
        else:
            self.lbl_global_title.setText(f"Solution Spaces for {model_name}")

    def update_all_plots(self):
        if self.updating_plots:
            return
        self.updating_plots = True

        items = self.inputs + self.outputs
        self.combo_add_x.clear()
        self.combo_add_y.clear()
        self.combo_add_x.addItems(items)
        self.combo_add_y.addItems(items)
        if self.combo_add_x.count() > 0:
            if len(self.inputs) >= 2:
                self.combo_add_x.setCurrentText(self.inputs[0])
                self.combo_add_y.setCurrentText(self.inputs[1])
            elif len(self.inputs) == 1:
                self.combo_add_x.setCurrentText(self.inputs[0])
                self.combo_add_y.setCurrentText(self.inputs[0])

        if not self.plot_widgets:
            self.add_plot()

        # Clear cached interpolation data when switching to Points mode
        viz_mode = self.combo_viz_mode.currentText()
        if viz_mode == "Points":
            for pw in self.plot_widgets:
                pw.cached_categorical_img = None
                pw.cached_data_hash = None
                pw.cached_bounds_hash = None

                # [CRITICAL FIX]: Cancel interpolation threads WITHOUT waiting
                if (
                    pw.interpolation_thread is not None
                    and pw.interpolation_thread.isRunning()
                ):
                    pw.interpolation_thread.cancel()

                    # Disconnect signals so it dies quietly
                    try:
                        pw.interpolation_thread.finished.disconnect()
                        pw.interpolation_thread.quick_result.disconnect()
                        pw.interpolation_thread.error.disconnect()
                    except (RuntimeError, TypeError):
                        pass

                    # Store reference to prevent garbage collection while running
                    pw.old_threads.append(pw.interpolation_thread)
                    pw.interpolation_thread = None

        for pw in self.plot_widgets:
            pw.plot()

        self.update_legend()
        self.updating_plots = False

    def update_legend(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        container = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(container)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(10)

        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        left_layout.setAlignment(QtCore.Qt.AlignTop)
        self.add_legend_item("Good Design", "#00aa00", legend_layout=left_layout)

        if self.problem and self.problem.quantities_of_interest:
            qois_to_show = self.problem.quantities_of_interest
        else:
            qois_to_show = [{"name": name} for name in self.outputs]
        for qoi in qois_to_show:
            if not qoi.get("show_in_legend", True):
                continue
            name = qoi["name"]
            display_name = qoi.get("display_name", name)
            color = self.qoi_colors.get(name, "red")
            self.add_legend_item(
                f"Violating {format_html(display_name)}",
                color,
                legend_layout=left_layout,
            )
        left_layout.addStretch()
        columns.addWidget(left_column, stretch=1)

        if self.multi_modal_boxes:
            right_column = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_column)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(2)
            right_layout.setAlignment(QtCore.Qt.AlignTop)
            boxes = self._get_multimodal_display_boxes()

            if self.multimodal_view_mode == "box":
                for box in boxes:
                    self.add_legend_item(
                        f"Ω<sub>{box.box_id + 1}</sub>",
                        "#000000",
                        legend_layout=right_layout,
                    )
            elif self.multimodal_view_mode == "recommended":
                colors = [self._get_branch_color(i, box) for i, box in enumerate(boxes)]
                if colors and all(color == "#000000" for color in colors):
                    self.add_legend_item(
                        "D<sub>MMSS</sub> (common)",
                        "#000000",
                        legend_layout=right_layout,
                    )
                else:
                    for index, box in enumerate(boxes):
                        self.add_legend_item(
                            f"D<sub>MMSS</sub>: Ω<sub>{index + 1}</sub>",
                            colors[index],
                            legend_layout=right_layout,
                        )
            else:
                for index, box in enumerate(boxes):
                    self.add_legend_item(
                        f"Ω<sub>{box.box_id + 1}</sub>",
                        self._get_branch_color(index, box),
                        legend_layout=right_layout,
                    )
            right_layout.addStretch()
            columns.addWidget(right_column, stretch=1)

        self.legend_layout.addWidget(container)

    def add_legend_item(self, name, color, legend_layout=None):
        item = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(item)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(10)
        lbl_color = QtWidgets.QLabel()
        lbl_color.setFixedSize(18, 18)
        lbl_color.setStyleSheet(
            f"background-color: {color}; border: 1px solid #666; border-radius: 4px;"
        )
        lbl_text = QtWidgets.QLabel(name)
        lbl_text.setTextFormat(QtCore.Qt.RichText)
        lbl_text.setStyleSheet("font-size: 9pt;")
        layout.addWidget(lbl_color)
        layout.addWidget(lbl_text)
        layout.addStretch()
        if legend_layout is None:
            legend_layout = self.legend_layout
        legend_layout.addWidget(item)

    def update_data_table(self):
        if not self.last_samples:
            return
        data = {}
        for name, values in self.problem.samples.items():
            data[name] = values
        for name, values in self.problem.results.items():
            data[name] = values
        data["Is Good"] = self.last_samples["is_good"]
        data["Is Bad"] = self.last_samples["is_bad"]
        df = pd.DataFrame(data)
        self.data_table.setRowCount(df.shape[0])
        self.data_table.setColumnCount(df.shape[1])
        self.data_table.setHorizontalHeaderLabels(df.columns)
        limit = min(10000, df.shape[0])
        self.data_table.setRowCount(limit)
        for i in range(limit):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                item = NumericTableWidgetItem(f"{val:.4f}")
                self.data_table.setItem(i, j, item)

    def export_csv(self):
        if not self.last_samples:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No data available to export."
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "results.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                # Reconstruct DataFrame (logic copied from update_data_table)
                data = {}
                # Add Samples
                for name, values in self.problem.samples.items():
                    data[name] = values
                # Add Results (QoIs)
                for name, values in self.problem.results.items():
                    data[name] = values
                # Add Metadata
                data["Is Good"] = self.last_samples["is_good"]
                data["Is Bad"] = self.last_samples["is_bad"]

                df = pd.DataFrame(data)
                df.to_csv(path, index=False)

                QtWidgets.QMessageBox.information(
                    self, "Success", f"Successfully exported {len(df)} rows to {path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Failed to export: {e}"
                )
