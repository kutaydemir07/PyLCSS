# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Reusable plotting widgets and dialogs for solution-space analysis."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..common.text_utils import format_html
from ..common.theme_manager import COLORS
from .plot_rendering import PlotRenderingMixin

__all__ = [
    "ArrowLine",
    "ColorConfigDialog",
    "PlotWidget",
    "ScalableText",
    "VariantRequirementsDialog",
]


class ScalableText(pg.GraphicsObject):
    """
    Text item that scales with the view (unlike pg.TextItem which stays fixed size).
    Uses data coordinates so it zooms in/out with the graph.
    """

    def __init__(self, text, x, y, size=0.15, color="k"):
        super().__init__()
        self.text = text
        self.x = x
        self.y = y
        self.size = size  # Text height in data coordinates
        self.color = pg.mkColor(color)
        self._generate_path()

    def _generate_path(self):
        """Generate text as a QPainterPath for scalable rendering."""
        self.path = QtGui.QPainterPath()
        font = QtGui.QFont("Arial", 100, QtGui.QFont.Bold)  # Use large font then scale
        self.path.addText(0, 0, font, self.text)

        # Get bounding rect and calculate scale factor
        br = self.path.boundingRect()
        if br.height() > 0:
            self.scale_factor = self.size / br.height()
        else:
            self.scale_factor = 0.001

        # Store bounding rect info for centering
        self.path_width = br.width()
        self.path_height = br.height()
        self.path_x = br.x()  # Left edge (usually 0 or small positive)
        self.path_y = br.y()  # Top edge (usually negative due to baseline)

    def paint(self, p, *args):
        p.save()
        # Translate to node center, then offset to center the text
        # In Qt coordinates: Y increases downward, but we flip Y in scale
        # Center calculation: after scaling, text width = path_width * scale_factor
        self.path_width * self.scale_factor
        self.path_height * self.scale_factor

        # Move to position where text center aligns with (self.x, self.y)
        # We draw at origin then translate
        p.translate(self.x, self.y)
        p.scale(self.scale_factor, -self.scale_factor)  # Flip Y for proper orientation

        # Offset to center: move left by half width, and vertically center
        # path_y is typically negative (above baseline), so center = path_y + path_height/2
        center_x = self.path_x + self.path_width / 2
        center_y = self.path_y + self.path_height / 2

        # Translate so center of path is at origin (which is now at node center)
        p.translate(-center_x, -center_y)

        p.fillPath(self.path, self.color)
        p.restore()

    def boundingRect(self):
        w = self.path_width * self.scale_factor
        h = self.path_height * self.scale_factor
        return QtCore.QRectF(self.x - w / 2, self.y - h / 2, w, h)


class ArrowLine(pg.GraphicsObject):
    """
    Arrow line that scales with the view (line width, arrow head all in data coords).
    Uses cosmetic=False so everything scales together when zooming.
    """

    def __init__(
        self,
        start_pos,
        end_pos,
        pen="k",
        head_len=0.12,
        node_radius=0.4,
        line_width=0.03,
    ):
        super().__init__()
        self.start_pos = pg.Point(start_pos)
        self.end_pos = pg.Point(end_pos)
        self.head_len = head_len  # Arrow head length in data coordinates
        self.node_radius = node_radius
        self.line_width = line_width  # Line width in data coordinates
        self.color = pg.mkColor(pen)

        self._calculate_geometry()

    def _calculate_geometry(self):
        """Calculate line and arrow head geometry."""
        diff = self.end_pos - self.start_pos
        dist = np.sqrt(diff.x() ** 2 + diff.y() ** 2)
        if dist < 0.001:
            self.valid = False
            return
        self.valid = True

        # Unit vector from start to end
        ux = diff.x() / dist
        uy = diff.y() / dist

        # Stop line at node boundary
        self.adj_end_x = self.end_pos.x() - ux * self.node_radius
        self.adj_end_y = self.end_pos.y() - uy * self.node_radius

        # Arrow head points (triangle)
        # Tip is at adj_end, two base points spread perpendicular
        tip_x = self.adj_end_x
        tip_y = self.adj_end_y

        # Base of arrow head (behind tip)
        base_x = tip_x - ux * self.head_len
        base_y = tip_y - uy * self.head_len

        # Perpendicular vector
        px = -uy
        py = ux

        # Arrow head width
        head_width = self.head_len * 0.5

        # Two base corners
        self.arrow_points = [
            (tip_x, tip_y),
            (base_x + px * head_width, base_y + py * head_width),
            (base_x - px * head_width, base_y - py * head_width),
        ]

        # Line ends before arrow head base
        self.line_end_x = base_x
        self.line_end_y = base_y

    def paint(self, p, *args):
        if not self.valid:
            return

        # Use a pen with width in data coordinates (cosmetic=False)
        pen = QtGui.QPen(self.color)
        pen.setWidthF(self.line_width)
        pen.setCosmetic(False)  # Scale with view
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)

        # Draw line
        p.drawLine(
            QtCore.QPointF(self.start_pos.x(), self.start_pos.y()),
            QtCore.QPointF(self.line_end_x, self.line_end_y),
        )

        # Draw arrow head as filled triangle
        arrow_polygon = QtGui.QPolygonF(
            [QtCore.QPointF(pt[0], pt[1]) for pt in self.arrow_points]
        )
        p.setBrush(self.color)
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(arrow_polygon)

    def boundingRect(self):
        return (
            QtCore.QRectF(self.start_pos, self.end_pos)
            .normalized()
            .adjusted(-0.2, -0.2, 0.2, 0.2)
        )


class VariantRequirementsDialog(QtWidgets.QDialog):
    def __init__(self, variant_name, problem, parent=None):
        super().__init__(parent)
        self.variant_name = variant_name
        self.problem = problem
        self.setWindowTitle(f"Edit Requirements - {variant_name}")
        self.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(self)

        # Instructions
        instructions = QtWidgets.QLabel("Define requirement bounds for this variant.")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Table for QoI requirements
        self.req_table = QtWidgets.QTableWidget()
        self.req_table.setColumnCount(4)
        self.req_table.setHorizontalHeaderLabels(
            ["QoI Name", "Unit", "Min Requirement", "Max Requirement"]
        )
        layout.addWidget(self.req_table)

        # Populate table
        self.req_table.setRowCount(len(self.problem.quantities_of_interest))
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            self.req_table.setItem(i, 0, QtWidgets.QTableWidgetItem(qoi["name"]))
            self.req_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(qoi.get("unit", "-"))
            )

            # Get current values (default or override)
            current_overrides = self.problem.requirement_sets.get(variant_name, {})
            qoi_overrides = current_overrides.get(qoi["name"], {})

            min_val = qoi_overrides.get("req_min", qoi["min"])
            max_val = qoi_overrides.get("req_max", qoi["max"])

            min_edit = QtWidgets.QLineEdit(str(min_val))
            max_edit = QtWidgets.QLineEdit(str(max_val))

            self.req_table.setCellWidget(i, 2, min_edit)
            self.req_table.setCellWidget(i, 3, max_edit)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

        # Adjust table column widths
        self.req_table.resizeColumnsToContents()

    def get_overrides(self):
        """Get the requirement overrides from the dialog."""
        overrides = {}
        for i in range(self.req_table.rowCount()):
            qoi_name = self.req_table.item(i, 0).text()
            min_edit = self.req_table.cellWidget(i, 2)
            max_edit = self.req_table.cellWidget(i, 3)

            min_val = min_edit.text().strip()
            max_val = max_edit.text().strip()

            override_data = {}
            if min_val:
                try:
                    override_data["req_min"] = float(min_val)
                except ValueError:
                    pass
            if max_val:
                try:
                    override_data["req_max"] = float(max_val)
                except ValueError:
                    pass

            if override_data:
                overrides[qoi_name] = override_data

        return overrides


class ColorConfigDialog(QtWidgets.QDialog):
    def __init__(self, qoi_names, current_colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Colors")
        self.resize(400, 500)
        self.colors = current_colors.copy()

        layout = QtWidgets.QVBoxLayout(self)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout(container)

        # Add buttons for each QOI
        keys = list(qoi_names)
        self.buttons = {}

        for key in keys:
            color = self.colors.get(key, "#ff0000")
            btn = QtWidgets.QPushButton()
            btn.setStyleSheet(f"background-color: {color};")
            btn.clicked.connect(lambda checked, k=key: self.pick_color(k))
            self.buttons[key] = btn
            self.form_layout.addRow(key, btn)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def pick_color(self, key):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self.colors.get(key, "#ff0000")), self
        )
        if color.isValid():
            hex_color = color.name()
            self.colors[key] = hex_color
            self.buttons[key].setStyleSheet(f"background-color: {hex_color};")

    def get_colors(self):
        return self.colors


class PlotWidget(PlotRenderingMixin, QtWidgets.QWidget):
    """Interactive plot with ROI and interpolation support."""

    def __init__(self, x_name, y_name, parent=None):
        super().__init__(parent)
        self.x_name = x_name
        self.y_name = y_name
        self.samples = None  # Per-plot samples
        self.plotting = False

        # References to plot items to prevent deletion
        self.scatter_good = None
        self.scatter_bad = None
        self.scatter_optimal = None  # Star marker for optimal point
        self.img_item = None
        self.roi_item = None
        self.roi_lines = []
        self.limit_lines = []
        self.hull_item = None  # Visual item for candidate space hull

        # KNN interpolation caching for performance
        self.cached_categorical_img = None
        self.cached_data_hash = None
        self.cached_bounds_hash = None

        # Interpolation thread
        self.interpolation_thread = None
        self.old_threads = []  # Keep track of cancelled threads until they finish
        self.current_generation_id = 0  # Track interpolation request generations

        # Box update throttling timer
        self.box_update_timer = QtCore.QTimer()
        self.box_update_timer.setSingleShot(True)
        self.box_update_timer.timeout.connect(self._perform_box_update)
        self.pending_roi = None

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Controls (Simplified)
        ctrl_layout = QtWidgets.QHBoxLayout()

        # Title Label
        title = f"{format_html(y_name)} vs {format_html(x_name)}"
        self.lbl_title = QtWidgets.QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: bold;")

        self.btn_remove = QtWidgets.QPushButton("X")
        self.btn_remove.setFixedSize(18, 18)
        self.btn_remove.setStyleSheet(
            "background-color: #ff4444; color: white; font-weight: bold;"
        )

        ctrl_layout.addWidget(self.lbl_title)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_remove)

        self.layout.addLayout(ctrl_layout)

        # Plot
        self.plot_widget = pg.PlotWidget()
        # Solution-space charts have always used a paper canvas.  They are
        # engineering figures (and export targets), not application chrome,
        # so keep them white in both application themes.
        self.plot_widget.setBackground(COLORS["chart_bg"])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)

        # Get plot item for customization
        self.plot_item = self.plot_widget.getPlotItem()
        for axis_name in ("left", "right", "top", "bottom"):
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(COLORS["chart_fg"])
            axis.setTextPen(COLORS["chart_fg"])
        self.plot_item.titleLabel.setAttr("color", COLORS["chart_fg"])
        self.plot_item.setTitle(title)

        # Enable mouse interactions
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(
            False
        )  # Disable right-click menu for cleaner interface

        # Add Zoom/Save buttons to control layout
        self.btn_zoom = QtWidgets.QPushButton("Zoom")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setToolTip("Toggle zoom/pan on this plot")
        self.btn_zoom.clicked.connect(self.toggle_zoom)
        self.btn_zoom.setFixedHeight(18)
        self.btn_zoom.setStyleSheet(
            "QPushButton { background-color:#5865F2; color:white; border:none;"
            " padding:0 5px; border-radius:3px; font-size:10px; min-width:0; }"
            " QPushButton:hover { background-color:#4752c4; }"
            " QPushButton:pressed, QPushButton:checked { background-color:#383a40; }"
        )

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setToolTip("Save this plot as an image")
        self.btn_save.clicked.connect(self.save_plot)
        self.btn_save.setFixedHeight(18)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color:#5865F2; color:white; border:none;"
            " padding:0 5px; border-radius:3px; font-size:10px; min-width:0; }"
            " QPushButton:hover { background-color:#4752c4; }"
            " QPushButton:pressed { background-color:#383a40; }"
        )

        ctrl_layout.insertWidget(2, self.btn_zoom)
        ctrl_layout.insertWidget(3, self.btn_save)

        # Data reference
        self.parent_widget = None  # To access data

    def toggle_zoom(self):
        if self.btn_zoom.isChecked():
            # Enable zoom mode in PyQtGraph
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        else:
            # Return to pan mode
            self.plot_widget.setMouseEnabled(x=False, y=False)
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)

    def save_plot(self):
        # Use PyQtGraph's export functionality
        from PySide6.QtWidgets import QFileDialog
        import pyqtgraph.exporters as pg_exporters

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf);;All Files (*)",
        )

        if filename:
            try:
                if filename.lower().endswith(".svg"):
                    # Use PyQtGraph's native SVG export for true vector graphics
                    try:
                        exporter = pg_exporters.SVGExporter(self.plot_widget.plotItem)
                        exporter.export(filename)
                    except Exception:
                        # Fallback to Qt SVG generation if PyQtGraph export fails
                        from PySide6.QtSvg import QSvgGenerator
                        from PySide6.QtGui import QPainter

                        rect = self.plot_widget.sceneRect()
                        svg_gen = QSvgGenerator()
                        svg_gen.setFileName(filename)
                        svg_gen.setSize(rect.size().toSize())
                        svg_gen.setViewBox(rect)
                        svg_gen.setTitle("PyQtGraph Plot")

                        painter = QPainter(svg_gen)
                        self.plot_widget.scene().render(painter)
                        painter.end()

                elif filename.lower().endswith(".pdf"):
                    # High-quality PDF export using PyQtGraph's vector capabilities
                    try:
                        # Try PyQtGraph's export if available
                        if hasattr(self.plot_widget, "export"):
                            self.plot_widget.export(filename, format="pdf")
                        else:
                            # Fallback to Qt PDF generation
                            from PySide6.QtPrintSupport import QPrinter
                            from PySide6.QtGui import QPainter

                            rect = self.plot_widget.sceneRect()
                            printer = QPrinter(QPrinter.HighResolution)
                            printer.setOutputFormat(QPrinter.PdfFormat)
                            printer.setOutputFileName(filename)
                            printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
                            printer.setResolution(600)  # Higher DPI for better quality

                            painter = QPainter(printer)
                            page_rect = printer.pageRect()

                            # Calculate scaling to fit page with margins
                            margin = 50  # pixels
                            available_width = page_rect.width() - 2 * margin
                            available_height = page_rect.height() - 2 * margin

                            scale_x = available_width / rect.width()
                            scale_y = available_height / rect.height()
                            scale = min(scale_x, scale_y)

                            # Center the plot on the page
                            painter.translate(page_rect.center())
                            painter.scale(scale, scale)
                            painter.translate(-rect.center())

                            self.plot_widget.scene().render(painter)
                            painter.end()
                    except Exception:
                        # Final fallback
                        from PySide6.QtPrintSupport import QPrinter

                        printer = QPrinter(QPrinter.HighResolution)
                        printer.setOutputFormat(QPrinter.PdfFormat)
                        printer.setOutputFileName(filename)
                        printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))

                        # Create high-res pixmap and print it
                        pixmap = self.plot_widget.grab()
                        scaled_pixmap = pixmap.scaledToWidth(
                            4000, QtCore.Qt.SmoothTransformation
                        )

                        painter = QPainter(printer)
                        page_rect = printer.pageRect()
                        painter.drawPixmap(
                            (page_rect.width() - scaled_pixmap.width()) / 2,
                            (page_rect.height() - scaled_pixmap.height()) / 2,
                            scaled_pixmap,
                        )
                        painter.end()

                else:
                    # Default to PNG with high quality
                    if not filename.lower().endswith(".png"):
                        filename += ".png"
                    # Export at very high resolution for quality
                    exporter = pg_exporters.ImageExporter(self.plot_widget.plotItem)
                    exporter.parameters()["width"] = 3000
                    exporter.parameters()["height"] = 2000
                    exporter.export(filename)

                QtWidgets.QMessageBox.information(
                    self, "Success", f"Plot saved to {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to save plot: {e}"
                )

    def update_roi_visuals(self):
        """Force update ROI position from global state without full replot."""
        if not self.roi_item or not self.parent_widget:
            return

        self.parent_widget.dv_par_box_mutex.lock()
        try:
            if self.parent_widget.dv_par_box is None:
                return
            box = self.parent_widget.dv_par_box
            dvs = [dv["name"] for dv in self.parent_widget.problem.design_variables]
            if self.x_name not in dvs or self.y_name not in dvs:
                return
            x_idx = dvs.index(self.x_name)
            y_idx = dvs.index(self.y_name)
            bx_min, bx_max = box[x_idx, 0], box[x_idx, 1]
            by_min, by_max = box[y_idx, 0], box[y_idx, 1]
        finally:
            self.parent_widget.dv_par_box_mutex.unlock()

        # Update Rect
        new_pos = QtCore.QPointF(bx_min, by_min)
        new_size = QtCore.QPointF(bx_max - bx_min, by_max - by_min)

        self.roi_item.blockSignals(True)
        self.roi_item.setPos(new_pos)
        self.roi_item.setSize(new_size)
        self.roi_item.blockSignals(False)

        # Update Lines
        if len(self.roi_lines) == 4:
            self.roi_lines[0].setPos(bx_min)
            self.roi_lines[1].setPos(bx_max)
            self.roi_lines[2].setPos(by_min)
            self.roi_lines[3].setPos(by_max)

    def on_box_moved(self, roi):
        """Throttle box updates to every 100ms for responsive UI."""
        self.pending_roi = roi
        self.box_update_timer.start(100)

    def _perform_box_update(self):
        """Actually perform the box update after throttling delay."""
        roi = self.pending_roi
        if not self.parent_widget or self.parent_widget.dv_par_box is None or not roi:
            return

        # Get ROI position and size
        pos = roi.pos()
        size = roi.size()
        x, y = pos.x(), pos.y()
        w, h = size.x(), size.y()

        # Identify indices for x_name and y_name
        x_idx = -1
        y_idx = -1
        for i, dv in enumerate(self.parent_widget.problem.design_variables):
            if dv["name"] == self.x_name:
                x_idx = i
            if dv["name"] == self.y_name:
                y_idx = i

        if x_idx != -1 or y_idx != -1:
            self.parent_widget.dv_par_box_mutex.lock()
            try:
                if x_idx != -1:
                    self.parent_widget.dv_par_box[x_idx, 0] = x
                    self.parent_widget.dv_par_box[x_idx, 1] = x + w

                if y_idx != -1:
                    self.parent_widget.dv_par_box[y_idx, 0] = y
                    self.parent_widget.dv_par_box[y_idx, 1] = y + h
            finally:
                self.parent_widget.dv_par_box_mutex.unlock()

        # Force other plots to update their ROI visuals immediately (without full replot)
        self.parent_widget.sync_plots_roi(self)

        # Update table UI for only the changed variables (block signals to avoid loop)
        if x_idx != -1:
            self.parent_widget.update_single_dv_row(x_idx)
        if y_idx != -1:
            self.parent_widget.update_single_dv_row(y_idx)

        # Trigger auto-resample
        # self.parent_widget.resample_box(silent=True)
        # Debounce resampling to avoid freezing during drag
        self.parent_widget.trigger_debounced_resample()
