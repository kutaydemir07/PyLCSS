# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Solution space analysis and visualization interface for PyLCSS.

This module provides the main GUI interface for solution space exploration,
including multi-objective optimization, Monte Carlo sampling, visualization,
and interactive analysis tools. It integrates with the computation engine
and solver engine to provide comprehensive design space analysis capabilities.

Key Features:
- Interactive solution space visualization with parallel coordinates
- Monte Carlo sampling for uncertainty analysis
- Multi-objective optimization with Pareto front analysis
- Real-time convergence monitoring
- Data export/import capabilities
- Interactive filtering and exploration tools

Classes:
    - SolutionSpaceWidget: Main GUI widget for solution space analysis
    - ColorConfigDialog: Dialog for configuring visualization colors
    - PlotWidget: Base class for plot widgets with common functionality
"""

import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore as PGQtCore
import dill as pickle  # Use dill for better serialization of dynamic functions
import json
import h5py
import os
import time
import networkx as nx
import tempfile
import importlib.util
from scipy.interpolate import NearestNDInterpolator
import logging

from ..problem_definition.problem_setup import XRayProblem
from .computation_engine import compute_solution_space, resample_solution_space
from .solver_engine import SolutionSpaceSolver
from typing import List, Dict, Any, Optional, Tuple, Union
from .solver_worker_thread import SolverWorker, ProductFamilyWorker
from .resample_worker_thread import ResampleThread
from .interpolation_worker_thread import InterpolationThread

from ..user_interface.text_utils import format_latex, format_html

logger = logging.getLogger(__name__)

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
        self.req_table.setHorizontalHeaderLabels(["QoI Name", "Unit", "Min Requirement", "Max Requirement"])
        layout.addWidget(self.req_table)
        
        # Populate table
        self.req_table.setRowCount(len(self.problem.quantities_of_interest))
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            self.req_table.setItem(i, 0, QtWidgets.QTableWidgetItem(qoi['name']))
            self.req_table.setItem(i, 1, QtWidgets.QTableWidgetItem(qoi.get('unit', '-')))
            
            # Get current values (default or override)
            current_overrides = self.problem.requirement_sets.get(variant_name, {})
            qoi_overrides = current_overrides.get(qoi['name'], {})
            
            min_val = qoi_overrides.get('req_min', qoi['min'])
            max_val = qoi_overrides.get('req_max', qoi['max'])
            
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
                    override_data['req_min'] = float(min_val)
                except ValueError:
                    pass
            if max_val:
                try:
                    override_data['req_max'] = float(max_val)
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
            color = self.colors.get(key, '#ff0000')
            btn = QtWidgets.QPushButton()
            btn.setStyleSheet(f"background-color: {color};")
            btn.clicked.connect(lambda checked, k=key: self.pick_color(k))
            self.buttons[key] = btn
            self.form_layout.addRow(key, btn)
            
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    def pick_color(self, key):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.colors.get(key, '#ff0000')), self)
        if color.isValid():
            hex_color = color.name()
            self.colors[key] = hex_color
            self.buttons[key].setStyleSheet(f"background-color: {hex_color};")
            
    def get_colors(self):
        return self.colors

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, x_name, y_name, parent=None):
        super(PlotWidget, self).__init__(parent)
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
        self.hull_item = None # Visual item for candidate space hull
        
        # KNN interpolation caching for performance
        self.cached_categorical_img = None
        self.cached_data_hash = None
        self.cached_bounds_hash = None
        
        # Interpolation thread
        self.interpolation_thread = None
        self.old_threads = [] # Keep track of cancelled threads until they finish
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
        self.btn_remove.setFixedSize(24, 24)
        self.btn_remove.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold;")
        
        ctrl_layout.addWidget(self.lbl_title)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_remove)
        
        self.layout.addLayout(ctrl_layout)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background for better readability
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)
        
        # Get plot item for customization
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setTitle(title)
        
        # Enable mouse interactions
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)  # Disable right-click menu for cleaner interface
        
        # Add Zoom/Save buttons to control layout
        self.btn_zoom = QtWidgets.QPushButton("Zoom")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.clicked.connect(self.toggle_zoom)
        self.btn_zoom.setMinimumWidth(60)
        self.btn_zoom.setMaximumWidth(80)
        # Apply dark theme styling explicitly
        self.btn_zoom.setStyleSheet("""
            QPushButton {
                background-color: #5865F2;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 60px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:pressed {
                background-color: #383a40;
            }
            QPushButton:checked {
                background-color: #383a40;
                color: #b5bac1;
            }
        """)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.clicked.connect(self.save_plot)
        self.btn_save.setMinimumWidth(60)
        self.btn_save.setMaximumWidth(80)
        # Apply dark theme styling explicitly
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #5865F2;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 60px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:pressed {
                background-color: #383a40;
            }
        """)
        
        ctrl_layout.insertWidget(2, self.btn_zoom)
        ctrl_layout.insertWidget(3, self.btn_save)
        
        # Data reference
        self.parent_widget = None # To access data

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
        import os
        from PySide6.QtWidgets import QFileDialog
        import pyqtgraph.exporters as pg_exporters
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", 
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if filename:
            try:
                if filename.lower().endswith('.svg'):
                    # Use PyQtGraph's native SVG export for true vector graphics
                    try:
                        exporter = pg_exporters.SVGExporter(self.plot_widget.plotItem)
                        exporter.export(filename)
                    except Exception as e:
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
                    
                elif filename.lower().endswith('.pdf'):
                    # High-quality PDF export using PyQtGraph's vector capabilities
                    try:
                        # Try PyQtGraph's export if available
                        if hasattr(self.plot_widget, 'export'):
                            self.plot_widget.export(filename, format='pdf')
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
                    except Exception as e:
                        # Final fallback
                        from PySide6.QtPrintSupport import QPrinter
                        printer = QPrinter(QPrinter.HighResolution)
                        printer.setOutputFormat(QPrinter.PdfFormat)
                        printer.setOutputFileName(filename)
                        printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
                        
                        # Create high-res pixmap and print it
                        pixmap = self.plot_widget.grab()
                        scaled_pixmap = pixmap.scaledToWidth(4000, QtCore.Qt.SmoothTransformation)
                        
                        painter = QPainter(printer)
                        page_rect = printer.pageRect()
                        painter.drawPixmap((page_rect.width() - scaled_pixmap.width()) / 2,
                                         (page_rect.height() - scaled_pixmap.height()) / 2,
                                         scaled_pixmap)
                        painter.end()
                    
                else:
                    # Default to PNG with high quality
                    if not filename.lower().endswith('.png'):
                        filename += '.png'
                    # Export at very high resolution for quality
                    exporter = pg_exporters.ImageExporter(self.plot_widget.plotItem)
                    exporter.parameters()['width'] = 3000
                    exporter.parameters()['height'] = 2000
                    exporter.export(filename)
                    
                QtWidgets.QMessageBox.information(self, "Success", f"Plot saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save plot: {e}")

    def update_roi_visuals(self):
        """Force update ROI position from global state without full replot."""
        if not self.roi_item or not self.parent_widget: return
        
        self.parent_widget.dv_par_box_mutex.lock()
        try:
            if self.parent_widget.dv_par_box is None: return
            box = self.parent_widget.dv_par_box
            dvs = [dv['name'] for dv in self.parent_widget.problem.design_variables]
            if self.x_name not in dvs or self.y_name not in dvs: return
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

    def plot(self):
        if self.plotting:
            return
        self.plotting = True

        if not self.parent_widget or not self.parent_widget.problem:
            self.plotting = False
            return

        # SMART CLEAR - Remove data but keep structure
        if self.scatter_good:
            self.plot_widget.removeItem(self.scatter_good)
            self.scatter_good = None
        if self.scatter_bad:
            self.plot_widget.removeItem(self.scatter_bad)
            self.scatter_bad = None
        if self.scatter_optimal:
            self.plot_widget.removeItem(self.scatter_optimal)
            self.scatter_optimal = None
        if self.img_item:
            self.plot_widget.removeItem(self.img_item)
            self.img_item = None
        for item in self.limit_lines:
            self.plot_widget.removeItem(item)
        self.limit_lines = []
        
        if self.hull_item:
            self.plot_widget.removeItem(self.hull_item)
            self.hull_item = None
        
        # Clean up multiple hull items if they exist
        if hasattr(self, 'hull_items') and self.hull_items:
            for item in self.hull_items:
                try:
                    self.plot_widget.removeItem(item)
                except:
                    pass
            self.hull_items = []

        # Force update to ensure items are removed
        self.plot_widget.update()

        x_name = self.x_name
        y_name = self.y_name
        # Detect if this is an objective plot (both axes are outputs)
        is_objective_plot = False

        points = self.samples['points'] if self.samples is not None else None
        qoi_values = self.samples['qoi_values'] if self.samples is not None else None
        is_good = self.samples['is_good'] if self.samples is not None else None
        violation_idx = self.samples['violation_idx'] if self.samples is not None else None

        def get_data(name):
            if points is not None and name in self.parent_widget.inputs:
                idx = self.parent_widget.inputs.index(name)
                return points[idx, :]
            elif qoi_values is not None and name in [q['name'] for q in self.parent_widget.problem.quantities_of_interest]:
                idx = [q['name'] for q in self.parent_widget.problem.quantities_of_interest].index(name)
                return qoi_values[idx, :]
            return None

        x_data = get_data(x_name)
        y_data = get_data(y_name)
        # Continue even if no data, for optimized point

        def get_bounds(name):
            if name in self.parent_widget.inputs:
                idx = self.parent_widget.inputs.index(name)
                if (hasattr(self.parent_widget, 'dsl') and hasattr(self.parent_widget, 'dsu') and 
                    self.parent_widget.dsl is not None and 
                    idx < len(self.parent_widget.dsl)):  # Check size!
                    
                    lower = self.parent_widget.dsl[idx]
                    upper = self.parent_widget.dsu[idx]
                    # Handle infinite bounds
                    if not np.isfinite(lower):
                        lower = -1e6  # Use a large finite value
                    if not np.isfinite(upper):
                        upper = 1e6   # Use a large finite value
                    return lower, upper
            data = get_data(name)
            if data is not None:
                # Filter out NaN and infinite values for min/max calculation
                valid_data = data[np.isfinite(data)]
                if len(valid_data) > 0:
                    d_min, d_max = np.min(valid_data), np.max(valid_data)
                    rng = d_max - d_min
                    if rng == 0: rng = 1.0
                    return d_min, d_max
                else:
                    # All values are NaN or infinite, return default range
                    return 0, 1
            return 0, 1

        x_min, x_max = get_bounds(x_name)
        y_min, y_max = get_bounds(y_name)
        
        # Include ROI box bounds in axis ranges
        self.parent_widget.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = self.parent_widget.dv_par_box.copy() if self.parent_widget.dv_par_box is not None else None
        finally:
            self.parent_widget.dv_par_box_mutex.unlock()
            
        if dv_par_box_copy is not None:
            dvs = [dv['name'] for dv in self.parent_widget.problem.design_variables]
            if x_name in dvs and y_name in dvs:
                x_idx = dvs.index(x_name)
                y_idx = dvs.index(y_name)
                box_x_min, box_x_max = dv_par_box_copy[x_idx, 0], dv_par_box_copy[x_idx, 1]
                box_y_min, box_y_max = dv_par_box_copy[y_idx, 0], dv_par_box_copy[y_idx, 1]
                x_min = min(x_min, box_x_min)
                x_max = max(x_max, box_x_max)
                y_min = min(y_min, box_y_min)
                y_max = max(y_max, box_y_max)
        
        # Set axis ranges
        self.plot_item.setXRange(x_min, x_max, padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)
        
        # Force ViewBox ranges with no padding and disable auto-ranging
        view_box = self.plot_widget.getViewBox()
        view_box.setRange(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min), padding=0)
        view_box.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        view_box.enableAutoRange(enable=False)

        # Create scatter plot data
        if self.samples is not None and x_data is not None and y_data is not None:
            # Filter out NaN and infinite values
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            if is_objective_plot:
                is_good = is_good[valid_mask]
            
            # Objective plot: green for good, red for bad
            if is_objective_plot:
                # Separate good and bad points
                good_mask = is_good
                bad_mask = ~is_good
                
                # Plot bad points first (red)
                if np.any(bad_mask):
                    bad_x = x_data[bad_mask]
                    bad_y = y_data[bad_mask]
                    self.scatter_bad = pg.ScatterPlotItem(
                        x=bad_x, y=bad_y, 
                        pen=pg.mkPen('w', width=0.5), 
                        brush=pg.mkBrush('#ff0000'),
                        size=6, alpha=0.6
                    )
                    self.plot_widget.addItem(self.scatter_bad)
                
                # Plot good points (green)
                if np.any(good_mask):
                    good_x = x_data[good_mask]
                    good_y = y_data[good_mask]
                    self.scatter_good = pg.ScatterPlotItem(
                        x=good_x, y=good_y, 
                        pen=pg.mkPen('w', width=0.5), 
                        brush=pg.mkBrush('#00aa00'),
                        size=6, alpha=0.6
                    )
                    self.plot_widget.addItem(self.scatter_good)
            else:
                # Solution space plot: color by QOI violation
                viz_mode = self.parent_widget.combo_viz_mode.currentText()
                num_qoi = len(self.parent_widget.problem.quantities_of_interest)
                qoi_names = [q['name'] for q in self.parent_widget.problem.quantities_of_interest]
                
                if viz_mode == "Categorical Areas":
                    # Create interpolated filled areas using KNN interpolation
                    try:
                        # Create grid for interpolation
                        x_grid, y_grid = np.meshgrid(
                            np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500)
                        )
                        
                        # Prepare data for interpolation
                        points = np.column_stack((x_data, y_data))
                        
                        # Create color index array (0=good, 1+=violation types)
                        color_indices = np.zeros(len(x_data))
                        for i in range(len(x_data)):
                            if not is_good[i]:
                                if violation_idx is not None and i < len(violation_idx):
                                    v_idx = int(violation_idx[i]) % num_qoi
                                    color_indices[i] = v_idx + 1  # 1, 2, 3... for different violations
                                else:
                                    color_indices[i] = num_qoi + 1  # Unknown violation
                        
                        # Create hash of current data and bounds for caching
                        import hashlib
                        data_hash = hashlib.md5()
                        data_hash.update(points.tobytes())
                        data_hash.update(color_indices.tobytes())
                        data_hash.update(np.array([x_min, x_max, y_min, y_max]).tobytes())
                        current_hash = data_hash.hexdigest()
                        
                        # Check if we can use cached result
                        if (self.cached_data_hash == current_hash and 
                            self.cached_categorical_img is not None):
                            # Use cached image
                            self.img_item = pg.ImageItem(self.cached_categorical_img)
                            self.img_item.setRect(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
                            self.img_item.setZValue(0)
                            self.plot_widget.addItem(self.img_item)
                        
                        # Check if interpolation is already running
                        if self.interpolation_thread is not None and self.interpolation_thread.isRunning():
                            # Cancel the existing thread
                            self.interpolation_thread.cancel()
                            # Do NOT wait() as it freezes GUI. Move to old threads list.
                            old_thread = self.interpolation_thread
                            self.old_threads.append(old_thread)
                            # Connect cleanup when finished (ignoring result)
                            old_thread.finished.connect(lambda _: self._cleanup_thread(old_thread))
                            old_thread.error.connect(lambda _: self._cleanup_thread(old_thread))
                            self.interpolation_thread = None
                        
                        # Increment generation ID for this request
                        self.current_generation_id += 1
                        generation_id = self.current_generation_id
                        
                        # Start background interpolation with generation tracking
                        self.interpolation_thread = InterpolationThread(points, color_indices, x_grid, y_grid, generation_id)
                        self.interpolation_thread.quick_result.connect(
                            lambda quick_interp, gen_id=generation_id: self._on_quick_interpolation_result(
                                quick_interp, x_min, x_max, y_min, y_max, 
                                num_qoi, qoi_names, current_hash, gen_id
                            )
                        )
                        self.interpolation_thread.finished.connect(
                            lambda interpolated, gen_id=generation_id: self._on_interpolation_finished(
                                interpolated, x_min, x_max, y_min, y_max, 
                                num_qoi, qoi_names, current_hash, gen_id
                            )
                        )
                        self.interpolation_thread.error.connect(
                            lambda err: self._on_interpolation_error(err)
                        )
                        self.interpolation_thread.start()
                        
                    except Exception as e:
                        logger.exception("Interpolation failed")
                else:
                    # Points mode: original implementation
                    colors = []
                    for i in range(len(x_data)):
                        if is_good[i]:
                            colors.append(pg.mkBrush('#00aa00'))  # Green for good
                        else:
                            # Color by violating constraint
                            if violation_idx is not None and i < len(violation_idx):
                                v_idx = int(violation_idx[i]) % num_qoi
                                if v_idx < len(qoi_names):
                                    q_name = qoi_names[v_idx]
                                    color = self.parent_widget.qoi_colors.get(q_name, 'red')
                                    colors.append(pg.mkBrush(color))
                                else:
                                    colors.append(pg.mkBrush('red'))
                            else:
                                colors.append(pg.mkBrush('red'))
                    
                    self.scatter_good = pg.ScatterPlotItem(
                        x=x_data, y=y_data, 
                        pen=pg.mkPen('w', width=0.5), 
                        brush=colors,
                        size=6, alpha=0.6
                    )
                    self.scatter_good.setZValue(1)
                    self.plot_widget.addItem(self.scatter_good)

        # Draw Optimal Point (if objectives are included and we have the optimized point)
        if hasattr(self.parent_widget, 'optimal_point') and self.parent_widget.optimal_point is not None:
            optimal_pt = self.parent_widget.optimal_point
            # Check if we can plot this point on this plot
            opt_x = None
            opt_y = None
            
            # Get optimal point value for x-axis
            if x_name in self.parent_widget.inputs:
                x_idx = self.parent_widget.inputs.index(x_name)
                if x_idx < len(optimal_pt):
                    opt_x = optimal_pt[x_idx]
            elif qoi_values is not None and x_name in [q['name'] for q in self.parent_widget.problem.quantities_of_interest]:
                # For QoI on x-axis, we'd need to evaluate - skip for now
                pass
            
            # Get optimal point value for y-axis  
            if y_name in self.parent_widget.inputs:
                y_idx = self.parent_widget.inputs.index(y_name)
                if y_idx < len(optimal_pt):
                    opt_y = optimal_pt[y_idx]
            elif qoi_values is not None and y_name in [q['name'] for q in self.parent_widget.problem.quantities_of_interest]:
                # For QoI on y-axis, we'd need to evaluate - skip for now
                pass
            
            # Plot star marker if we have both coordinates
            if opt_x is not None and opt_y is not None:
                self.scatter_optimal = pg.ScatterPlotItem(
                    x=[opt_x], y=[opt_y],
                    pen=pg.mkPen('k', width=2),
                    brush=pg.mkBrush(255, 215, 0),  # Gold color
                    size=20,
                    symbol='star'
                )
                self.scatter_optimal.setZValue(100)  # Ensure it's on top
                self.plot_widget.addItem(self.scatter_optimal)

        # Draw Solution Box (if available and both axes are DVs)
        if dv_par_box_copy is not None:
            dvs = [dv['name'] for dv in self.parent_widget.problem.design_variables]
            if x_name in dvs and y_name in dvs:
                x_idx = dvs.index(x_name)
                y_idx = dvs.index(y_name)
                bx_min, bx_max = dv_par_box_copy[x_idx, 0], dv_par_box_copy[x_idx, 1]
                by_min, by_max = dv_par_box_copy[y_idx, 0], dv_par_box_copy[y_idx, 1]
                
                # Helper for scalar conversion
                def _s(v):
                    if hasattr(v, 'item'):
                        if v.size > 1: return float(v.flatten()[0])
                        return float(v.item())
                    if hasattr(v, '__len__') and not isinstance(v, str):
                        return float(v[0])
                    return float(v)

                if self.roi_item is None:
                    # Create draggable ROI rectangle
                    roi = pg.ROI([bx_min, by_min], [bx_max - bx_min, by_max - by_min], 
                                pen=pg.mkPen('black', width=2), rotatable=False)
                    roi.maxBounds = QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min))
                    roi.addScaleHandle([1, 1], [0, 0])  # Bottom-right
                    roi.addScaleHandle([0, 0], [1, 1])  # Top-left
                    roi.addScaleHandle([1, 0], [0, 1])  # Bottom-left
                    roi.addScaleHandle([0, 1], [1, 0])  # Top-right
                    
                    roi.sigRegionChanged.connect(lambda: self.on_box_moved(roi))
                    roi.setZValue(10) # Ensure on top
                    self.plot_widget.addItem(roi)
                    self.roi_item = roi
                else:
                    self.roi_item.maxBounds = QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min))
                    
                    # FIX: Update ROI position if external change detected (computation result), 
                    # but check threshold to avoid jitter during drag.
                    current_pos = self.roi_item.pos()
                    current_size = self.roi_item.size()
                    new_pos = QtCore.QPointF(bx_min, by_min)
                    new_size = QtCore.QPointF(bx_max - bx_min, by_max - by_min)
                    
                    if (abs(current_pos.x() - new_pos.x()) > 1e-6 or abs(current_pos.y() - new_pos.y()) > 1e-6 or
                        abs(current_size.x() - new_size.x()) > 1e-6 or abs(current_size.y() - new_size.y()) > 1e-6):
                        
                        self.roi_item.blockSignals(True)
                        self.roi_item.setPos(new_pos)
                        self.roi_item.setSize(new_size)
                        self.roi_item.blockSignals(False)

                # Update dotted lines connecting box to axes
                for l in self.roi_lines:
                    self.plot_widget.removeItem(l)
                self.roi_lines = []
                
                # Vertical lines from box to x-axis
                vline_left = pg.InfiniteLine(pos=bx_min, angle=90, pen=pg.mkPen('black', style=QtCore.Qt.DashLine, width=1))
                vline_right = pg.InfiniteLine(pos=bx_max, angle=90, pen=pg.mkPen('black', style=QtCore.Qt.DashLine, width=1))
                # Horizontal lines from box to y-axis  
                hline_bottom = pg.InfiniteLine(pos=by_min, angle=0, pen=pg.mkPen('black', style=QtCore.Qt.DashLine, width=1))
                hline_top = pg.InfiniteLine(pos=by_max, angle=0, pen=pg.mkPen('black', style=QtCore.Qt.DashLine, width=1))
                
                self.plot_widget.addItem(vline_left)
                self.plot_widget.addItem(vline_right)
                self.plot_widget.addItem(hline_bottom)
                self.plot_widget.addItem(hline_top)
                self.roi_lines = [vline_left, vline_right, hline_bottom, hline_top]

        # Set axis labels
        x_unit = self.parent_widget.input_units.get(x_name) or self.parent_widget.output_units.get(x_name) or '-'
        y_unit = self.parent_widget.input_units.get(y_name) or self.parent_widget.output_units.get(y_name) or '-'
        x_label_text = format_html(x_name)
        y_label_text = format_html(y_name)
        x_label = f"{x_label_text} ({x_unit})"
        y_label = f"{y_label_text} ({y_unit})"
        
        self.plot_item.setLabel('bottom', x_label)
        self.plot_item.setLabel('left', y_label)
        self.plot_item.setTitle(f"{x_label_text} vs {y_label_text}")
        
        # X-Axis Requirements (Vertical Lines)
        qois = [q['name'] for q in self.parent_widget.problem.quantities_of_interest]
        if x_name in qois:
            # Find row in qoi_table
            for i in range(self.parent_widget.qoi_table.rowCount()):
                if self.parent_widget.qoi_table.item(i, 0).text() == x_name:
                    try:
                        l_val = float(self.parent_widget.qoi_table.item(i, 2).text())
                        u_val = float(self.parent_widget.qoi_table.item(i, 3).text())
                        
                        if l_val > -1e8: # Arbitrary large number check
                            l_line = pg.InfiniteLine(pos=l_val, angle=90, pen=pg.mkPen('red', style=QtCore.Qt.DashLine, alpha=0.5))
                            self.plot_widget.addItem(l_line)
                            self.limit_lines.append(l_line)
                        if u_val < 1e8:
                            u_line = pg.InfiniteLine(pos=u_val, angle=90, pen=pg.mkPen('red', style=QtCore.Qt.DashLine, alpha=0.5))
                            self.plot_widget.addItem(u_line)
                            self.limit_lines.append(u_line)
                    except: pass
                    break
                    
        # Y-Axis Requirements (Horizontal Lines)
        if y_name in qois:
            for i in range(self.parent_widget.qoi_table.rowCount()):
                if self.parent_widget.qoi_table.item(i, 0).text() == y_name:
                    try:
                        l_val = float(self.parent_widget.qoi_table.item(i, 2).text())
                        u_val = float(self.parent_widget.qoi_table.item(i, 3).text())
                        
                        if l_val > -1e8:
                            l_line = pg.InfiniteLine(pos=l_val, angle=0, pen=pg.mkPen('red', style=QtCore.Qt.DashLine, alpha=0.5))
                            self.plot_widget.addItem(l_line)
                            self.limit_lines.append(l_line)
                        if u_val < 1e8:
                            u_line = pg.InfiniteLine(pos=u_val, angle=0, pen=pg.mkPen('red', style=QtCore.Qt.DashLine, alpha=0.5))
                            self.plot_widget.addItem(u_line)
                            self.limit_lines.append(u_line)
                    except: pass
                    break
            
        # Force final update
        self.plot_widget.update()
        self.plotting = False

    def _cleanup_thread(self, thread):
        """Clean up finished background threads."""
        if thread in self.old_threads:
            self.old_threads.remove(thread)
        thread.deleteLater()

    def _on_interpolation_finished(self, interpolated, x_min, x_max, y_min, y_max, num_qoi, qoi_names, current_hash, generation_id):
        """Handle completion of background interpolation."""
        try:
            # Check if this result is from the current generation (not stale)
            if generation_id != self.current_generation_id:
                return  # Discard stale results
            
            # Check if we're still in categorical mode
            current_viz_mode = self.parent_widget.combo_viz_mode.currentText()
            if current_viz_mode != "Categorical Areas":
                return  # Don't add categorical regions if mode has changed
            
            # Create color map
            color_map = np.zeros((interpolated.shape[0], interpolated.shape[1], 4))
            
            # Good regions (green)
            good_mask = interpolated == 0
            color_map[good_mask] = [0, 170, 0, 180]  # Semi-transparent green
            
            # Violation regions
            for v_idx in range(num_qoi):
                violation_mask = interpolated == (v_idx + 1)
                if np.any(violation_mask):
                    q_name = qoi_names[v_idx] if v_idx < len(qoi_names) else 'unknown'
                    color = self.parent_widget.qoi_colors.get(q_name, 'red')
                    # Convert color name to RGB
                    color_rgb = pg.mkColor(color)
                    color_map[violation_mask] = [color_rgb.red(), color_rgb.green(), color_rgb.blue(), 180]
            
            # Unknown violations
            unknown_mask = interpolated == (num_qoi + 1)
            color_map[unknown_mask] = [255, 0, 0, 180]  # Semi-transparent red
            
            # Helper for scalar conversion
            def _s(v):
                if hasattr(v, 'item'):
                    if v.size > 1: return float(v.flatten()[0])
                    return float(v.item())
                if hasattr(v, '__len__') and not isinstance(v, str):
                    return float(v[0])
                return float(v)

            # Update or create image item (avoid flicker by reusing existing item)
            if hasattr(self, 'img_item') and self.img_item is not None:
                # Update existing image data to avoid flicker
                self.img_item.setImage(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)))
            else:
                # Create new image item
                self.img_item = pg.ImageItem(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)))
                self.img_item.setZValue(0)
                self.plot_widget.addItem(self.img_item)
        except RuntimeError as e:
            # Ignore errors resulting from deleted C++ objects
            if "Internal C++ object" not in str(e):
                logger.debug("RuntimeError in interpolation finished", exc_info=True)
        except Exception as e:
            logger.exception("Error in interpolation finished")
            # Cache the result
            self.cached_categorical_img = color_map.astype(np.uint8).transpose(1, 0, 2)
            self.cached_data_hash = current_hash
                    
        except Exception as e:
            logger.exception("Error creating interpolated image")
        finally:
            self.interpolation_thread = None

    def _on_quick_interpolation_result(self, quick_interp, x_min, x_max, y_min, y_max, num_qoi, qoi_names, current_hash, generation_id):
        """Handle quick interpolation result for immediate visual feedback."""
        try:
            # Check if this result is from the current generation (not stale)
            if generation_id != self.current_generation_id:
                return  # Discard stale results
            
            # Check if we're still in categorical mode
            current_viz_mode = self.parent_widget.combo_viz_mode.currentText()
            if current_viz_mode != "Categorical Areas":
                return  # Don't add categorical regions if mode has changed
            
            # Create color map for quick result
            color_map = np.zeros((quick_interp.shape[0], quick_interp.shape[1], 4))
            
            # Good regions (green)
            good_mask = quick_interp == 0
            color_map[good_mask] = [0, 170, 0, 120]  # More transparent for quick result
            
            # Violation regions
            for v_idx in range(num_qoi):
                violation_mask = quick_interp == (v_idx + 1)
                if np.any(violation_mask):
                    q_name = qoi_names[v_idx] if v_idx < len(qoi_names) else 'unknown'
                    color = self.parent_widget.qoi_colors.get(q_name, 'red')
                    # Convert color name to RGB
                    color_rgb = pg.mkColor(color)
                    color_map[violation_mask] = [color_rgb.red(), color_rgb.green(), color_rgb.blue(), 120]
            
            # Unknown violations
            unknown_mask = quick_interp == (num_qoi + 1)
            color_map[unknown_mask] = [255, 0, 0, 120]  # More transparent for quick result
            
            # Helper for scalar conversion
            def _s(v):
                if hasattr(v, 'item'):
                    if v.size > 1: return float(v.flatten()[0])
                    return float(v.item())
                if hasattr(v, '__len__') and not isinstance(v, str):
                    return float(v[0])
                return float(v)

            # Update or create image item for quick result (avoid flicker)
            if hasattr(self, 'img_item') and self.img_item is not None:
                # Update existing image data with quick result
                self.img_item.setImage(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)))
            else:
                # Create new image item if none exists
                self.img_item = pg.ImageItem(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(QtCore.QRectF(_s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)))
                self.img_item.setZValue(0)
                self.plot_widget.addItem(self.img_item)
            
        except RuntimeError as e:
            # Ignore errors resulting from deleted C++ objects
            if "Internal C++ object" not in str(e):
                logger.debug("RuntimeError in quick interpolation", exc_info=True)
        except Exception as e:
            logger.exception("Error creating quick interpolated image")

    def _on_interpolation_error(self, error_msg):
        """Handle interpolation thread errors."""
        logger.error("Interpolation failed: %s", error_msg)
        self.interpolation_thread = None

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
            if dv['name'] == self.x_name: x_idx = i
            if dv['name'] == self.y_name: y_idx = i
            
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

class SolutionSpaceWidget(QtWidgets.QWidget):
    import numpy as np

    def _to_serializable(self, obj):
        """Recursively convert numpy arrays in obj to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._to_serializable(v) for v in obj)
        else:
            return obj
    """
    Main widget for solution space analysis and visualization.

    Provides a comprehensive interface for multi-objective optimization,
    Monte Carlo sampling, and interactive exploration of design spaces.
    Integrates multiple visualization components and background computation
    threads for efficient analysis of complex engineering problems.

    Key Features:
        - Design variable and quantity of interest management
        - Multi-objective optimization with Pareto front analysis
        - Monte Carlo sampling for uncertainty quantification
        - Interactive parallel coordinates plots
        - Convergence monitoring and diagnostics
        - Data import/export capabilities
        - Real-time progress tracking

    Attributes:
        problem: XRayProblem instance defining the optimization problem
        system_code: Compiled system model code
        plot_widgets: List of active visualization widgets
        solver_worker: Background thread for optimization
        resample_thread: Background thread for resampling
    """

    def trigger_debounced_resample(self):
        """Trigger resampling after a short delay to prevent freezing during rapid updates."""
        self.resample_timer.start()

    def _safe_get_float(self, item, default=0.0):
        """Safely convert table item text to float."""
        if item is None: return default
        text = item.text().strip()
        if not text: return default
        try:
            return float(text)
        except ValueError:
            return default

    def apply_optimization_constraints(self, updates: Dict[str, Any]) -> None:
        """
        Apply relaxed optimization results as new constraints.

        Updates the quantity of interest table with relaxed bounds from
        optimization results, disabling optimization flags for those variables.

        Args:
            updates: Dictionary mapping variable names to constraint updates
        """
        self.qoi_table.blockSignals(True)

        for row in range(self.qoi_table.rowCount()):
            name = self.qoi_table.item(row, 0).text()
            if name in updates:
                data = updates[name]

                # Uncheck optimization flags (Cols 6 & 7)
                self.qoi_table.item(row, 6).setCheckState(QtCore.Qt.Unchecked)
                self.qoi_table.item(row, 7).setCheckState(QtCore.Qt.Unchecked)

                if 'req_min' in data:
                    self.qoi_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{data['req_min']:.4f}"))
                    self.qoi_table.item(row, 2).setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

                if 'req_max' in data:
                    self.qoi_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{data['req_max']:.4f}"))
                    self.qoi_table.item(row, 3).setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

        self.qoi_table.blockSignals(False)
        # Optional: Auto-run computation
        # self.run_computation()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize the solution space analysis widget."""
        super(SolutionSpaceWidget, self).__init__(parent)

        self.problem = None
        self.system_code = None
        self.inputs = []
        self.outputs = []
        self.models = []
        self.dv_par_box = None
        self.last_samples = None
        self.plot_widgets = []
        self.updating_plots = False
        self.resample_thread = None
        self.candidate_worker = None
        self.resampling = False
        self.pending_restart = False
        self.qoi_colors = {}
        self.optimal_point = None
        
        # Debounce timer for resampling
        self.resample_timer = QtCore.QTimer()
        self.resample_timer.setSingleShot(True)
        self.resample_timer.setInterval(300) # 300ms delay
        self.resample_timer.timeout.connect(lambda: self.resample_box(silent=True))
        
        # Thread safety: Mutex for dv_par_box access
        self.dv_par_box_mutex = QtCore.QRecursiveMutex()
        # Distinct colors excluding green (e.g. #3cb44b, #00aa00)
        self.default_colors = [
            '#e6194b', # Red
            '#4363d8', # Blue
            '#ffe119', # Yellow
            '#f58231', # Orange
            '#911eb4', # Purple
            '#42d4f4', # Cyan
            '#f032e6', # Magenta
            '#a9a9a9', # Grey
            '#fabebe', # Pink
            '#000075', # Navy
            '#9a6324', # Brown
            '#800000', # Maroon
            '#e6beff', # Lavender
            '#fffac8', # Beige
        ]
        self.input_units = {}
        self.output_units = {}
        
        # Connect to application quit to clean up threads
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.on_app_quit)
        
        self.init_ui()
        
    def closeEvent(self, event):
        # Wait for any running threads to finish
        if self.resample_thread and self.resample_thread.isRunning():
            self.resample_thread.wait()
        if self.candidate_worker and self.candidate_worker.isRunning():
            self.candidate_worker.wait()
        event.accept()
        
    def on_app_quit(self):
        # Wait for any running threads to finish before app quits
        if self.resample_thread and self.resample_thread.isRunning():
            self.resample_thread.wait()
        if self.candidate_worker and self.candidate_worker.isRunning():
            self.candidate_worker.wait()
        
    def init_ui(self):
        # Main Layout: Splitter
        main_layout = QtWidgets.QHBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # --- Left Panel: Configuration ---
        self.config_panel = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(self.config_panel)
        
        # Config Tabs
        self.config_tabs = QtWidgets.QTabWidget()
        config_layout.addWidget(self.config_tabs)
        
        # Tab 1: Model & Controls
        tab_model = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(tab_model)
        
        # --- Top Section: Controls (Always Visible) ---
        
        # System Selection
        sys_group = QtWidgets.QGroupBox("System Model")
        sys_layout = QtWidgets.QHBoxLayout(sys_group)
        
        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.setToolTip("Select the system model to analyze.")
        self.system_combo.currentIndexChanged.connect(self.on_system_changed)
        sys_layout.addWidget(self.system_combo, stretch=1)
        
        self.btn_view_code = QtWidgets.QPushButton("Code")
        self.btn_view_code.setFixedWidth(50)
        self.btn_view_code.clicked.connect(self.view_source_code)
        sys_layout.addWidget(self.btn_view_code)
        
        model_layout.addWidget(sys_group)
        
        # Actions & Settings
        actions_group = QtWidgets.QGroupBox("Analysis Controls")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        
        # Row 1: Project Actions
        row1 = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Project")
        self.btn_load = QtWidgets.QPushButton("Load Project")
        self.btn_save.clicked.connect(self.save_project)
        self.btn_load.clicked.connect(self.load_project)
        row1.addWidget(self.btn_save)
        row1.addWidget(self.btn_load)
        actions_layout.addLayout(row1)
        
        # Row 2: Solver & Samples
        row2 = QtWidgets.QHBoxLayout()
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItem("SLSQP", "goal_attainment")
        self.solver_combo.addItem("Nevergrad", "nevergrad")
        self.solver_combo.addItem("Diff. Evol.", "differential_evolution")
        self.solver_combo.setToolTip("Choose optimization solver")
        row2.addWidget(self.solver_combo, stretch=1)
        
        self.sample_size_spin = QtWidgets.QSpinBox()
        self.sample_size_spin.setRange(100, 100000)
        self.sample_size_spin.setValue(3000)
        self.sample_size_spin.setPrefix("N=")
        self.sample_size_spin.setToolTip("Sample Size")
        row2.addWidget(self.sample_size_spin)
        actions_layout.addLayout(row2)
        
        # Row 3: Compute Buttons
        row3 = QtWidgets.QHBoxLayout()
        self.btn_compute_feasible = QtWidgets.QPushButton("Compute Solution Space")
        self.btn_compute_feasible.clicked.connect(lambda: self.run_computation(include_objectives=self.chk_include_optimization.isChecked()))
        self.btn_compute_feasible.setEnabled(False)
        row3.addWidget(self.btn_compute_feasible)
        
        self.chk_include_optimization = QtWidgets.QCheckBox("Opt.")
        self.chk_include_optimization.setToolTip("Include Optimization Objectives")
        self.chk_include_optimization.setEnabled(False)
        row3.addWidget(self.chk_include_optimization)
        actions_layout.addLayout(row3)
        
        # Row 4: Refinement
        row4 = QtWidgets.QHBoxLayout()
        self.btn_resample = QtWidgets.QPushButton("Resample")
        self.btn_resample.clicked.connect(self.resample_box)
        self.btn_resample.setEnabled(False)
        row4.addWidget(self.btn_resample)
        
        actions_layout.addLayout(row4)
        
        # Slider
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Box Size:"))
        self.slider_mosse = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_mosse.setRange(0, 100)
        self.slider_mosse.setValue(100)
        slider_layout.addWidget(self.slider_mosse)
        actions_layout.addLayout(slider_layout)
        
        model_layout.addWidget(actions_group)
        
        # --- Bottom Section: Variables (Tabbed) ---
        self.vars_tabs = QtWidgets.QTabWidget()
        
        # Design Variables Table
        self.dv_table = QtWidgets.QTableWidget()
        self.dv_table.setColumnCount(6)
        self.dv_table.setHorizontalHeaderLabels(["Name", "Unit", "Min (DS)", "Max (DS)", "Min (Sol)", "Max (Sol)"])
        self.dv_table.itemChanged.connect(self.on_dv_table_changed)
        self.vars_tabs.addTab(self.dv_table, "Design Variables")
        
        # QoI Table
        self.qoi_table = QtWidgets.QTableWidget()
        self.qoi_table.setColumnCount(9)
        self.qoi_table.setHorizontalHeaderLabels(["Name", "Unit", "Min (Req)", "Max (Req)", "Min (Plot)", "Max (Plot)", "Min", "Max", "W"])
        self.qoi_table.itemChanged.connect(self.on_qoi_table_changed)
        self.vars_tabs.addTab(self.qoi_table, "Quantities of Interest")
        
        model_layout.addWidget(self.vars_tabs)
        
        self.config_tabs.addTab(tab_model, "Model Control")
        
        # Tab 4: Product Family
        tab_family = QtWidgets.QWidget()
        family_layout = QtWidgets.QVBoxLayout(tab_family)
        
        # Variant Management
        variant_group = QtWidgets.QGroupBox("Product Variants")
        variant_layout = QtWidgets.QVBoxLayout(variant_group)
        
        self.variant_table = QtWidgets.QTableWidget()
        self.variant_table.setColumnCount(2)
        self.variant_table.setHorizontalHeaderLabels(["Variant Name", "Description"])
        variant_layout.addWidget(self.variant_table)
        
        btn_variant_layout = QtWidgets.QHBoxLayout()
        self.btn_add_variant = QtWidgets.QPushButton("Add Variant")
        self.btn_add_variant.clicked.connect(self.add_variant)
        self.btn_remove_variant = QtWidgets.QPushButton("Remove Variant")
        self.btn_remove_variant.clicked.connect(self.remove_variant)
        self.btn_edit_variant = QtWidgets.QPushButton("Edit Requirements")
        self.btn_edit_variant.clicked.connect(self.edit_variant_requirements)
        btn_variant_layout.addWidget(self.btn_add_variant)
        btn_variant_layout.addWidget(self.btn_remove_variant)
        btn_variant_layout.addWidget(self.btn_edit_variant)
        variant_layout.addLayout(btn_variant_layout)
        
        family_layout.addWidget(variant_group)
        
        # Solver Selection for Product Family
        solver_group = QtWidgets.QGroupBox("Solver Options")
        solver_layout = QtWidgets.QVBoxLayout(solver_group)
        
        # Solver Selection
        family_solver_layout = QtWidgets.QHBoxLayout()
        family_solver_layout.addWidget(QtWidgets.QLabel("Solver:"))
        self.family_solver_combo = QtWidgets.QComboBox()
        self.family_solver_combo.addItem("SLSQP (fast)", "goal_attainment")
        self.family_solver_combo.addItem("Nevergrad (robust)", "nevergrad")
        self.family_solver_combo.addItem("Differential Evolution (evolutionary)", "differential_evolution")
        self.family_solver_combo.setToolTip("Choose optimization solver:\n- SLSQP: Fast and reliable (recommended)\n- Nevergrad: Gradient-free optimization with native constraint support\n- Differential Evolution: Population-based evolutionary algorithm")
        family_solver_layout.addWidget(self.family_solver_combo)
        solver_layout.addLayout(family_solver_layout)
        
        family_layout.addWidget(solver_group)
        
        # Compute Button
        self.btn_compute_family = QtWidgets.QPushButton("Compute Product Family")
        self.btn_compute_family.setToolTip("Analyze multiple product variants simultaneously to find the common feasible design space (platform) across all variants.")
        self.btn_compute_family.clicked.connect(self.compute_product_family)
        self.btn_compute_family.setEnabled(False)
        family_layout.addWidget(self.btn_compute_family)
        
        family_layout.addStretch()
        
        self.config_tabs.addTab(tab_family, "Product Family")
        
        # Connect config tab changes to mode switching
        self.config_tabs.currentChanged.connect(self.on_config_tab_changed)
        
        self.splitter.addWidget(self.config_panel)
        
        # --- Right Panel: Visualization & Data ---
        self.right_tabs = QtWidgets.QTabWidget()

        # --- Solution Spaces Tab (Design Variables) ---
        self.solution_tab = QtWidgets.QWidget()
        solution_layout = QtWidgets.QVBoxLayout(self.solution_tab)
        # Controls for Solution Spaces
        self.combo_add_x = QtWidgets.QComboBox()
        self.combo_add_x.setToolTip("Select the design variable or output to plot on the X-axis. Shows how the selected variable affects the solution space.")
        self.combo_add_y = QtWidgets.QComboBox()
        self.combo_add_y.setToolTip("Select the design variable or output to plot on the Y-axis. Shows how the selected variable affects the solution space.")
        self.combo_viz_mode = QtWidgets.QComboBox()
        self.combo_viz_mode.addItem("Points")
        self.combo_viz_mode.addItem("Categorical Areas")
        self.combo_viz_mode.setToolTip("Choose visualization mode:\n• Points: Show individual sample points colored by feasibility\n• Categorical Areas: Show interpolated regions colored by constraint violations")
        self.combo_viz_mode.currentIndexChanged.connect(self.update_all_plots)
        self.btn_add_plot = QtWidgets.QPushButton("Add Plot")
        self.btn_add_plot.clicked.connect(self.add_plot)
        self.btn_clear_plots = QtWidgets.QPushButton("Clear All")
        self.btn_clear_plots.clicked.connect(self.clear_all_plots)
        self.btn_save_all = QtWidgets.QPushButton("Save All Plots")
        self.btn_save_all.clicked.connect(self.save_all_plots)
        self.btn_colors = QtWidgets.QPushButton("Colors")
        self.btn_colors.clicked.connect(self.configure_colors)
        sol_ctrl_layout = QtWidgets.QHBoxLayout()
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("X:"))
        sol_ctrl_layout.addWidget(self.combo_add_x)
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("Y:"))
        sol_ctrl_layout.addWidget(self.combo_add_y)
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"))
        sol_ctrl_layout.addWidget(self.combo_viz_mode)
        sol_ctrl_layout.addWidget(self.btn_add_plot)
        sol_ctrl_layout.addWidget(self.btn_clear_plots)
        sol_ctrl_layout.addWidget(self.btn_save_all)
        sol_ctrl_layout.addWidget(self.btn_colors)
        sol_ctrl_layout.addStretch()
        solution_layout.addLayout(sol_ctrl_layout)
        # Title
        self.lbl_global_title = QtWidgets.QLabel("Solution Spaces for Unknown Model")
        self.lbl_global_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_global_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        solution_layout.addWidget(self.lbl_global_title)
        # Main Content Area (Scroll + Legend)
        sol_content_layout = QtWidgets.QHBoxLayout()
        # Scroll Area for Plots
        sol_scroll = QtWidgets.QScrollArea()
        sol_scroll.setWidgetResizable(True)
        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setStyleSheet("background-color: white;")
        self.plots_layout = QtWidgets.QGridLayout(self.plots_container)
        self.plots_layout.setAlignment(QtCore.Qt.AlignTop)
        sol_scroll.setWidget(self.plots_container)
        sol_content_layout.addWidget(sol_scroll, stretch=4)
        # Global Legend Area for Solution Spaces
        self.legend_group = QtWidgets.QGroupBox("Legend")
        self.legend_layout = QtWidgets.QVBoxLayout(self.legend_group)
        self.legend_layout.setAlignment(QtCore.Qt.AlignTop)
        sol_content_layout.addWidget(self.legend_group, stretch=1)
        solution_layout.addLayout(sol_content_layout)
        self.right_tabs.addTab(self.solution_tab, "Solution Spaces")

        # Tab 3: Data Table
        self.data_panel = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(self.data_panel)
        
        # Add Export Button
        btn_export = QtWidgets.QPushButton("Export Data to CSV")
        btn_export.clicked.connect(self.export_csv)
        data_layout.addWidget(btn_export)
        
        self.data_table = QtWidgets.QTableWidget()
        self.data_table.setSortingEnabled(True)
        data_layout.addWidget(self.data_table)
        self.right_tabs.addTab(self.data_panel, "Data Table")

        # Tab 4: Product Family Analysis
        self.family_tab = QtWidgets.QWidget()
        family_tab_layout = QtWidgets.QVBoxLayout(self.family_tab)
        
        # Title
        self.lbl_family_title = QtWidgets.QLabel("Product Family Analysis")
        self.lbl_family_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_family_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        family_tab_layout.addWidget(self.lbl_family_title)
        
        # Scroll Area for Family Plots
        family_scroll = QtWidgets.QScrollArea()
        family_scroll.setWidgetResizable(True)
        self.family_plots_container = QtWidgets.QWidget()
        self.family_plots_container.setStyleSheet("background-color: white;")
        self.family_plots_layout = QtWidgets.QGridLayout(self.family_plots_container)
        self.family_plots_layout.setAlignment(QtCore.Qt.AlignTop)
        family_scroll.setWidget(self.family_plots_container)
        family_tab_layout.addWidget(family_scroll)
        
        self.right_tabs.addTab(self.family_tab, "Product Family Analysis")
        
        # Connect tab change to update data table on demand
        self.right_tabs.currentChanged.connect(self.on_right_tab_changed)

        self.splitter.addWidget(self.right_tabs)
        # Set initial sizes and stretch factors
        self.splitter.setSizes([400, 800])
        self.splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.splitter.setStretchFactor(1, 1)  # Right panel stretches
        
        # Initialize in normal mode (not product family mode)
        self.product_family_mode = False
        self.update_right_tabs_visibility()

    def on_config_tab_changed(self, index: int):
        """
        Handle config tab changes to switch between normal and product family modes.
        """
        current_tab_text = self.config_tabs.tabText(index)
        if current_tab_text == "Product Family":
            self.set_product_family_mode(True)
        else:
            self.set_product_family_mode(False)

    def on_right_tab_changed(self, index: int):
        """Handle right panel tab changes."""
        if self.right_tabs.tabText(index) == "Data Table":
            self.update_data_table()

    def set_product_family_mode(self, enabled: bool):
        """
        Switch between normal solution space mode and product family analysis mode.
        
        Args:
            enabled: True to enable product family mode, False for normal mode
        """
        self.product_family_mode = enabled
        self.update_right_tabs_visibility()
        
        if enabled:
            # Switch to product family tab
            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == "Product Family Analysis":
                    self.right_tabs.setCurrentIndex(i)
                    break
        else:
            # Switch to solution spaces tab
            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == "Solution Spaces":
                    self.right_tabs.setCurrentIndex(i)
                    break
    
    def update_right_tabs_visibility(self):
        """
        Update the visibility of right panel tabs based on current mode.
        """
        for i in range(self.right_tabs.count()):
            tab_text = self.right_tabs.tabText(i)
            if self.product_family_mode:
                # In product family mode, only show Product Family Analysis
                visible = (tab_text == "Product Family Analysis")
            else:
                # In normal mode, show all except Product Family Analysis
                visible = (tab_text != "Product Family Analysis")
            
            self.right_tabs.setTabVisible(i, visible)

    def sync_plots_roi(self, source_widget):
        """Force update ROI visuals for all plots except source."""
        for widget in self.plot_widgets:
            if widget != source_widget:
                widget.update_roi_visuals()

    def populate_tables_from_problem(self):
        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.qoi_table.blockSignals(True)
        
        self.dv_table.setRowCount(len(self.problem.design_variables))
        self.dv_par_box = np.zeros((len(self.problem.design_variables), 2))
        
        for i, dv in enumerate(self.problem.design_variables):
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(dv['name']))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(dv.get('unit', '-')))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(dv['min'])))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(dv['max'])))
            
            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(dv['min'])))
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(dv['max'])))
            
            self.dv_par_box[i, 0] = dv['min']
            self.dv_par_box[i, 1] = dv['max']
            
        # Populate QoI Table
        self.qoi_table.setRowCount(len(self.problem.quantities_of_interest))
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(qoi['name']))
            self.qoi_table.setItem(i, 1, QtWidgets.QTableWidgetItem(qoi.get('unit', '-')))
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(qoi['min'])))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(qoi['max'])))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto"))
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto"))
            
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(QtCore.Qt.Checked if qoi.get('minimize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 6, min_item)
            
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(QtCore.Qt.Checked if qoi.get('maximize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 7, max_item)
            
            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(qoi.get('weight', 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)
            
            # Disable req fields if minimize or maximize is checked
            if qoi.get('minimize', False) or qoi.get('maximize', False):
                min_req_item = self.qoi_table.item(i, 2)
                max_req_item = self.qoi_table.item(i, 3)
                if min_req_item:
                    min_req_item.setFlags(min_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if max_req_item:
                    max_req_item.setFlags(max_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.dv_table.blockSignals(False)
        self.qoi_table.blockSignals(False)

    def _execute_code_safely(self, code):
        """Execute code safely using exec with a custom filename.
        Avoids temporary files which can cause issues with pickling/dill."""
        namespace = {}
        try:
            # Compile with a descriptive filename for tracebacks
            bytecode = compile(code, "<dynamic_system_model>", 'exec')
            exec(bytecode, namespace)
        except Exception as e:
            logger.exception("Error executing code")
            raise e
            
        if 'system_function' in namespace:
            return namespace['system_function']
        else:
            # Fallback: look for any callable
            for k, v in namespace.items():
                if callable(v) and k not in ['__builtins__', 'np', 'joblib', 'os', 'sys']:
                    return v
            raise AttributeError("system_function not found in generated code")

    def load_model_from_system_model(self, m):
        self.load_model(m.source_code, m.inputs, m.outputs, m.name)

    def load_model(self, code, inputs, outputs, name=None):
        """
        Loads the model code and populates the tables (from Modeling Tab).
        inputs: List of dicts {'name', 'unit', 'min', 'max'}
        outputs: List of dicts {'name', 'unit', 'req_min', 'req_max'}
        name: Optional system name
        """
        self.system_code = code
        self.system_name = name
        
        # Initialize Problem Object immediately for resampling
        try:
            # Use SystemModel to create a persistent file for multiprocessing support
            from pylcss.system_modeling.system_model import SystemModel
            model_name = name if name else "Loaded_Model"
            
            sm = SystemModel.from_code_string(model_name, self.system_code, inputs, outputs)
            system_func = sm.system_function
            
            # Use sample_size_spin if available, otherwise default to 3000
            if hasattr(self, 'sample_size_spin'):
                n_samples = self.sample_size_spin.value()
            else:
                n_samples = 3000
            # Use provided name, else default
            model_name = name if name else "Loaded_Model"
            self.problem = XRayProblem(model_name, n_samples)
            self.problem.set_system_model(system_func)
            
            for inp in inputs:
                self.problem.add_design_variable(inp['name'], inp.get('unit', '-'), inp['min'], inp['max'])
                
            for out in outputs:
                minimize = out.get('minimize', False)
                maximize = out.get('maximize', False)
                self.problem.add_quantity_of_interest(out['name'], out.get('unit', '-'), out['req_min'], out['req_max'], minimize=minimize, maximize=maximize, weight=1.0)
                
        except Exception as e:
            logger.warning("Failed to initialize problem object", exc_info=True)
            self.problem = None

        # Extract names for internal use
        self.inputs = [i['name'] for i in inputs]
        self.outputs = [o['name'] for o in outputs]
        # Store units for axis labels
        self.input_units = {i['name']: i.get('unit', '-') for i in inputs}
        self.output_units = {o['name']: o.get('unit', '-') for o in outputs}
        
        # Update Combo Boxes
        self.combo_add_x.clear()
        self.combo_add_y.clear()
        
        all_vars = self.inputs + self.outputs
        self.combo_add_x.addItems(all_vars)
        self.combo_add_y.addItems(all_vars)
        
        # Set defaults if available
        if len(self.inputs) >= 1:
            self.combo_add_x.setCurrentIndex(0)
        if len(self.inputs) >= 2:
            self.combo_add_y.setCurrentIndex(1)
        elif len(self.inputs) >= 1:
            self.combo_add_y.setCurrentIndex(0)
            
        # --- FIX START: Initialize bounds to prevent plot crash ---
        try:
            self.dsl = np.array([float(i.get('min', 0)) for i in inputs])
            self.dsu = np.array([float(i.get('max', 1)) for i in inputs])
        except:
            self.dsl = None
            self.dsu = None
        # --- FIX END ---
        
        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(inputs))
        self.dv_par_box = np.zeros((len(inputs), 2))
        
        def safe_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                if val.lower() == 'inf':
                    return float('inf')
                if val.lower() == '-inf':
                    return float('-inf')
                try:
                    return float(val)
                except Exception:
                    return val
            return val

        for i, inp in enumerate(inputs):
            min_val = safe_float(inp.get('min', 0))
            max_val = safe_float(inp.get('max', 0))
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(inp['name']))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(inp.get('unit', '-')))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val))) 
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))
            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val
            
        self.dv_table.blockSignals(False)
            
        # Populate QoI Table
        self.qoi_table.setRowCount(len(outputs))
        for i, out in enumerate(outputs):
            req_min_val = safe_float(out.get('req_min', 0))
            req_max_val = safe_float(out.get('req_max', 0))
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(out['name']))
            self.qoi_table.setItem(i, 1, QtWidgets.QTableWidgetItem(out.get('unit', '-')))
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(req_min_val)))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(req_max_val)))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto")) # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto")) # Plot Max
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(QtCore.Qt.Checked if out.get('minimize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 6, min_item)
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(QtCore.Qt.Checked if out.get('maximize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 7, max_item)
            
            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(out.get('weight', 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)
            
            # Disable req fields if minimize or maximize is checked
            if out.get('minimize', False) or out.get('maximize', False):
                min_req_item = self.qoi_table.item(i, 2)
                max_req_item = self.qoi_table.item(i, 3)
                if min_req_item:
                    min_req_item.setFlags(min_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if max_req_item:
                    max_req_item.setFlags(max_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
            
        # Init colors
        self.qoi_colors = {}
        forbidden_greens = ['#00aa00', '#3cb44b', '#bcf60c', '#008080', '#aaffc3', '#808000', '#00ff00', '#008000']
        for i, out in enumerate(outputs):
            name = out['name']
            color = None
            if 'color' in out and out['color']:
                color = out['color']
                if color.lower() in forbidden_greens:
                    color = None # Reset if green
            
            if color:
                self.qoi_colors[name] = color
            else:
                self.qoi_colors[name] = self.default_colors[i % len(self.default_colors)]

        self.btn_compute_feasible.setEnabled(True)
        # Check if there are any objectives defined
        has_objectives = self.problem and any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                           for qoi in self.problem.quantities_of_interest)
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        self.btn_compute_family.setEnabled(True)
        self.update_all_plots()
        
        # Auto-create default plot if none exists
        if not self.plot_widgets and len(self.inputs) >= 2:
            self.add_plot(self.inputs[0], self.inputs[1])
        elif not self.plot_widgets and len(self.inputs) == 1:
             if self.outputs:
                 self.add_plot(self.inputs[0], self.outputs[0])
             else:
                 self.add_plot(self.inputs[0], self.inputs[0])

        # Auto-resample
        self.resample_box(silent=True)

    def load_model_from_system_model(self, system_model):
        """
        Load a model from a SystemModel instance.
        """
        # Use the compiled function directly
        self.system_code = system_model.source_code

        # Initialize Problem Object
        try:
            # Use sample_size_spin if available, otherwise default to 3000
            if hasattr(self, 'sample_size_spin'):
                n_samples = self.sample_size_spin.value()
            else:
                n_samples = 3000
            model_name = system_model.name
            self.problem = XRayProblem(model_name, n_samples)
            self.problem.set_system_model(system_model.system_function)
            self.problem.set_system_code(system_model.source_code)

            for inp in system_model.inputs:
                self.problem.add_design_variable(inp['name'], inp.get('unit', '-'), inp['min'], inp['max'])

            for out in system_model.outputs:
                minimize = out.get('minimize', False)
                maximize = out.get('maximize', False)
                # Get weight from table if available, otherwise use default
                weight = 1.0
                if hasattr(self, 'qoi_table') and self.qoi_table.rowCount() > 0:
                    for i, table_out in enumerate(system_model.outputs):
                        if i < self.qoi_table.rowCount() and table_out['name'] == out['name']:
                            try:
                                weight = float(self.qoi_table.item(i, 8).text())
                            except (ValueError, AttributeError):
                                weight = 1.0
                            break
                self.problem.add_quantity_of_interest(out['name'], out.get('unit', '-'), out['req_min'], out['req_max'], minimize=minimize, maximize=maximize, weight=weight)

        except Exception as e:
            logger.warning("Failed to initialize problem object", exc_info=True)
            self.problem = None

        # Extract names for internal use
        self.inputs = system_model.get_input_names()
        self.outputs = system_model.get_output_names()
        # Store units for axis labels
        self.input_units = {i['name']: i.get('unit', '-') for i in system_model.inputs}
        self.output_units = {o['name']: o.get('unit', '-') for o in system_model.outputs}
        
        # --- FIX START: Initialize bounds to prevent plot crash ---
        try:
            def safe_val(v, default):
                try: return float(v)
                except: return default
            
            self.dsl = np.array([safe_val(i.get('min', 0), 0.0) for i in system_model.inputs])
            self.dsu = np.array([safe_val(i.get('max', 1), 1.0) for i in system_model.inputs])
        except:
            self.dsl = None
            self.dsu = None
        # --- FIX END ---

        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(system_model.inputs))
        self.dv_par_box = np.zeros((len(system_model.inputs), 2))

        def safe_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                if val.lower() == 'inf':
                    return float('inf')
                if val.lower() == '-inf':
                    return float('-inf')
                try:
                    return float(val)
                except Exception:
                    return val
            return val

        for i, inp in enumerate(system_model.inputs):
            min_val = safe_float(inp.get('min', 0))
            max_val = safe_float(inp.get('max', 1))
            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val

            # Set table values
            display_name = inp.get('display_name', inp['name'])
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(display_name))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(inp.get('unit', '-')))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val))) 
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))

        self.dv_table.blockSignals(False)
        self.qoi_table.blockSignals(False)

        # Populate QoI Table
        self.qoi_table.setRowCount(len(system_model.outputs))
        for i, out in enumerate(system_model.outputs):
            req_min_val = safe_float(out.get('req_min', 0))
            req_max_val = safe_float(out.get('req_max', 0))
            display_name = out.get('display_name', out['name'])
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(display_name))
            self.qoi_table.setItem(i, 1, QtWidgets.QTableWidgetItem(out.get('unit', '-')))
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(req_min_val)))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(req_max_val)))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto")) # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto")) # Plot Max
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(QtCore.Qt.Checked if out.get('minimize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 6, min_item)
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(QtCore.Qt.Checked if out.get('maximize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 7, max_item)
            
            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(out.get('weight', 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)
            
            # Disable req fields if minimize or maximize is checked
            if out.get('minimize', False) or out.get('maximize', False):
                min_req_item = self.qoi_table.item(i, 2)
                max_req_item = self.qoi_table.item(i, 3)
                if min_req_item:
                    min_req_item.setFlags(min_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if max_req_item:
                    max_req_item.setFlags(max_req_item.flags() & ~QtCore.Qt.ItemIsEditable)

        # Init colors
        self.qoi_colors = {}
        forbidden_greens = ['#00aa00', '#3cb44b', '#bcf60c', '#008080', '#aaffc3', '#808000', '#00ff00', '#008000']
        for i, out in enumerate(system_model.outputs):
            name = out['name']
            color = None
            if 'color' in out and out['color']:
                color = out['color']
                if color.lower() in forbidden_greens:
                    color = None # Reset if green
            
            if color:
                self.qoi_colors[name] = color
            else:
                self.qoi_colors[name] = self.default_colors[i % len(self.default_colors)]

        self.btn_compute_feasible.setEnabled(True)
        # Check if there are any objectives defined
        has_objectives = self.problem and any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                           for qoi in self.problem.quantities_of_interest)
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        self.btn_compute_family.setEnabled(True)
        self.update_all_plots()
        
        # Auto-create default plot if none exists
        if not self.plot_widgets and len(self.inputs) >= 2:
            self.add_plot(self.inputs[0], self.inputs[1])
        elif not self.plot_widgets and len(self.inputs) == 1:
             if self.outputs:
                 self.add_plot(self.inputs[0], self.outputs[0])
             else:
                 self.add_plot(self.inputs[0], self.inputs[0])

        # Auto-resample
        self.resample_box(silent=True)

    def load_models(self, models):
        """
        Loads multiple models and allows selection.
        Models can be either SystemModel instances or legacy dict format.
        """
        self.models = models
        self.system_combo.clear()
        for m in models:
            # Handle both SystemModel instances and legacy dicts
            name = m.name if hasattr(m, 'name') else m['name']
            self.system_combo.addItem(name)
        
        if self.models:
            self.system_combo.setCurrentIndex(0)
            self.load_selected_system()

    def create_merged_model(self, models):
        """
        Creates a merged model from multiple subsystems.
        Detects dependencies based on shared variable names.
        """
        # Collect all variables
        all_inputs = {}
        all_outputs = {}
        
        for model in models:
            for inp in model['inputs']:
                name = inp['name']
                if name not in all_inputs:
                    all_inputs[name] = inp
            for out in model['outputs']:
                name = out['name']
                if name not in all_outputs:
                    all_outputs[name] = out
        
        # Build dependency graph
        G = nx.DiGraph()
        for i in range(len(models)):
            G.add_node(i)
        
        for i, model_a in enumerate(models):
            for inp in model_a['inputs']:
                inp_name = inp['name']
                for j, model_b in enumerate(models):
                    if i != j:
                        for out in model_b['outputs']:
                            if inp_name == out['name']:
                                G.add_edge(j, i)  # b provides input to a
        
        # Topological sort for execution order
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            raise ValueError("Circular dependency detected in models")
        
        # Identify global inputs and outputs
        input_names = set(all_inputs.keys())
        output_names = set(all_outputs.keys())
        
        global_inputs = sorted(list(input_names - output_names))
        global_outputs = sorted(list(output_names - input_names))
        
        if not global_outputs:
            raise ValueError("No global outputs found (all outputs are used as inputs)")
        
        # Generate merged code
        code = "import numpy as np\n\n"
        
        # Add each model's function with unique name
        for i, model in enumerate(models):
            model_code = model['code']
            code += model_code + "\n\n"
        
        # Merged system_function
        code += "def system_function(**kwargs):\n"
        
        # Extract global inputs
        for name in global_inputs:
            code += f"    {name} = kwargs['{name}']\n"
        
        code += "    intermediates = {}\n\n"
        
        # Execute models in order
        for idx in order:
            model = models[idx]
            code += f"    # Execute model {idx} ({model['name']})\n"
            
            # Build call arguments
            call_args = []
            for inp in model['inputs']:
                name = inp['name']
                if name in global_inputs:
                    call_args.append(f"{name}={name}")
                else:
                    call_args.append(f"{name}=intermediates['{name}']")
            
            call_str = ", ".join(call_args)
            code += f"    outputs_{idx} = system_function_{idx}({call_str})\n"
            
            # Store outputs in intermediates
            for out in model['outputs']:
                name = out['name']
                code += f"    intermediates['{name}'] = outputs_{idx}['{name}']\n"
            
            code += "\n"
        
        # Return global outputs
        code += "    return {\n"
        for name in global_outputs:
            code += f"        '{name}': intermediates['{name}'],\n"
        code += "    }\n"
        
        # Create merged model dict
        merged_inputs = [all_inputs[name] for name in global_inputs]
        merged_outputs = [all_outputs[name] for name in global_outputs]
        
        return {
            'name': 'Merged',
            'code': code,
            'inputs': merged_inputs,
            'outputs': merged_outputs
        }

    def on_system_changed(self):
        self.load_selected_system()

    def load_selected_system(self):
        idx = self.system_combo.currentIndex()
        if idx >= 0 and idx < len(self.models):
            m = self.models[idx]
            # Clear existing plots when switching systems
            self.clear_all_plots()
            # Clear cached samples to avoid QoI count mismatches
            self.last_samples = None

            # Handle both SystemModel instances and legacy dicts
            if hasattr(m, 'name'):  # SystemModel instance
                self.system_code = m.source_code
                self.load_model_from_system_model(m)
            else:  # Legacy dict format
                if m['name'] == 'Merged':
                    self.system_code = m['code']
                    self.load_model(m['code'], m['inputs'], m['outputs'], m['name'])
                else:
                    self.load_model(m['code'], m['inputs'], m['outputs'], m['name'])

            # Update title with system name
            model_name = m.name if hasattr(m, 'name') else m['name']
            if hasattr(self, 'lbl_global_title') and model_name:
                self.lbl_global_title.setText(f"Solution Spaces for {model_name}")

    def configure_colors(self):
        if not self.problem:
            return
        qoi_names = [q['name'] for q in self.problem.quantities_of_interest]
        dialog = ColorConfigDialog(qoi_names, self.qoi_colors, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.qoi_colors = dialog.get_colors()
            self.update_all_plots()

    def view_source_code(self):
        if not self.system_code:
            QtWidgets.QMessageBox.warning(self, "Warning", "No system loaded.")
            return
            
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Source Code")
        dialog.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText(self.system_code)
        text_edit.setReadOnly(True)
        text_edit.setFont(QtGui.QFont("Consolas", 10))
        layout.addWidget(text_edit)
        dialog.exec_()

    def on_dv_table_changed(self, item):
        if self.dv_par_box is None:
            return
            
        row = item.row()
        col = item.column()
        
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
                    dv_par_box_copy = self.dv_par_box.copy()
                finally:
                    self.dv_par_box_mutex.unlock()
                
                # Perform GUI updates and resampling outside the lock
                # Redraw plots to show new box
                self.update_all_plots()
                
                # Auto resample with the copied data
                self.resample_box(silent=True)
                
            except ValueError:
                pass # Ignore invalid input

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
                    min_req_item.setFlags(min_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
                if max_req_item:
                    max_req_item.setFlags(max_req_item.flags() & ~QtCore.Qt.ItemIsEditable)
            else:
                # Enable the fields if neither is checked
                min_checked = self.qoi_table.item(row, 6).checkState() == QtCore.Qt.Checked if self.qoi_table.item(row, 6) else False
                max_checked = self.qoi_table.item(row, 7).checkState() == QtCore.Qt.Checked if self.qoi_table.item(row, 7) else False
                if not min_checked and not max_checked:
                    min_req_item = self.qoi_table.item(row, 2)
                    max_req_item = self.qoi_table.item(row, 3)
                    if min_req_item:
                        min_req_item.setFlags(min_req_item.flags() | QtCore.Qt.ItemIsEditable)
                    if max_req_item:
                        max_req_item.setFlags(max_req_item.flags() | QtCore.Qt.ItemIsEditable)
            
            # Update the problem if it exists
            if self.problem and row < len(self.problem.quantities_of_interest):
                qoi = self.problem.quantities_of_interest[row]
                qoi['minimize'] = self.qoi_table.item(row, 6).checkState() == QtCore.Qt.Checked if self.qoi_table.item(row, 6) else False
                qoi['maximize'] = self.qoi_table.item(row, 7).checkState() == QtCore.Qt.Checked if self.qoi_table.item(row, 7) else False
                
        elif col == 8:  # Weight column
            # Update the problem weight if it exists
            if self.problem and row < len(self.problem.quantities_of_interest):
                try:
                    weight_value = float(item.text())
                    self.problem.quantities_of_interest[row]['weight'] = weight_value
                except ValueError:
                    # Reset to default if invalid
                    item.setText("1.0")
                    if self.problem and row < len(self.problem.quantities_of_interest):
                        self.problem.quantities_of_interest[row]['weight'] = 1.0
        
        # Update button states based on objectives
        if self.problem:
            has_objectives = any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                               for qoi in self.problem.quantities_of_interest)

    def update_table_from_box(self):
        self.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = self.dv_par_box.copy() if self.dv_par_box is not None else None
        finally:
            self.dv_par_box_mutex.unlock()
            
        if dv_par_box_copy is None: return
        
        self.dv_table.blockSignals(True)
        for i in range(self.dv_table.rowCount()):
            if i < len(dv_par_box_copy):
                self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[i, 0]:.4f}"))
                self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[i, 1]:.4f}"))
        self.dv_table.blockSignals(False)

    def update_single_dv_row(self, row_idx):
        """Update only a single row in the DV table for performance during dragging."""
        self.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = self.dv_par_box.copy() if self.dv_par_box is not None else None
        finally:
            self.dv_par_box_mutex.unlock()
            
        if dv_par_box_copy is None or row_idx >= len(dv_par_box_copy):
            return
        
        self.dv_table.blockSignals(True)
        self.dv_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[row_idx, 0]:.4f}"))
        self.dv_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(f"{dv_par_box_copy[row_idx, 1]:.4f}"))
        self.dv_table.blockSignals(False)

    def run_computation(self, include_objectives=False):
        # 1. Prepare Problem Object (if not already set)
        if not self.problem:
            return

        # 2. Gather Parameters for compute_solution_space
        try:
            # DVs
            dsl = []
            dsu = []
            l = [] 
            u = []
            
            for i in range(self.dv_table.rowCount()):
                dsl.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
                dsu.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
                l.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
                u.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
                
            dsl = np.array(dsl)
            dsu = np.array(dsu)
            l = np.array(l)
            u = np.array(u)
            
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
            parameters = None # Assuming no fixed parameters for now
            slider_val = self.slider_mosse.value() / 100.0
            sample_size = self.sample_size_spin.value()
            solver_type = self.solver_combo.currentData()
            
            # Create solver - use dill for robust problem serialization
            import copy
            # With dill, we can safely pass the problem object directly without manual reconstruction
            problem_to_use = copy.deepcopy(self.problem)
            
            # FIX: Only remove objectives if the user did NOT check "Include Optimization"
            if not include_objectives:
                for qoi in problem_to_use.quantities_of_interest:
                    qoi['minimize'] = False
                    qoi['maximize'] = False
            
            status_text = "Computing Solution Space..."
            
            solver = SolutionSpaceSolver(problem_to_use, weight, dsl, dsu, l, u, reqU, reqL, parameters, slider_value=slider_val, solver_type=solver_type, include_objectives=include_objectives)
            solver.final_sample_size = sample_size
            
            self.solver_worker = SolverWorker(solver)
            self.solver_worker.finished_signal.connect(self.on_compute_finished)
            self.solver_worker.progress_signal.connect(self.on_compute_progress)
            self.solver_worker.error_signal.connect(self.on_compute_error)
            
            # Disable button during computation
            self.btn_compute_feasible.setEnabled(False)
            self.status_msg = QtWidgets.QProgressDialog(status_text, "Cancel", 0, 0, self)
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
        self.dv_par_box = box
        
        # Update DV Table with Solution Bounds
        if box is not None:
            self.dv_table.blockSignals(True)
            for i in range(self.dv_table.rowCount()):
                if i < len(box):
                    self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{box[i, 0]:.4f}"))
                    self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{box[i, 1]:.4f}"))
            self.dv_table.blockSignals(False)
        
        # FIX: Retrieve samples from solver if not provided (optimization)
        if samples is None and hasattr(self.solver_worker.solver, 'latest_results'):
            samples = self.solver_worker.solver.latest_results
        
        # Store optimal point if objectives were included
        if hasattr(self.solver_worker.solver, 'include_objectives') and self.solver_worker.solver.include_objectives:
            # The optimal point is the last point added to the samples (the extra_point)
            if samples is not None and 'points' in samples and samples['points'].shape[1] > 0:
                self.optimal_point = samples['points'][:, -1]
            else:
                self.optimal_point = None
        else:
            self.optimal_point = None

        self.process_results(samples)
        self.btn_resample.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Success", f"Computation complete in {elapsed_time:.2f}s!")
        
        # Auto-resample to show samples
        self.resample_box(silent=True)

    def on_compute_error(self, error_msg):
        self.status_msg.close()
        self.btn_compute_feasible.setEnabled(True)
        # Check if there are any objectives defined
        has_objectives = self.problem and any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                           for qoi in self.problem.quantities_of_interest)
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        QtWidgets.QMessageBox.critical(self, "Error", f"Computation failed: {error_msg}")

    def on_computation_cancelled(self):
        self.solver_worker.stop()
        self.status_msg.close()

    def resample_box(self, silent=False):
        if self.resampling:
            self.pending_restart = True
            return
        self.resampling = True
        
        if self.problem is None:
            self.resampling = False
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Warning", "No valid model loaded for resampling.")
            return
        
        # FIX: Robust Thread Safety for Box Copy
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
            
        # Wait for any existing resample thread to finish
        if self.resample_thread and self.resample_thread.isRunning():
            return
        
        try:
            # Sync tables with problem
            if self.problem and self.qoi_table.rowCount() != len(self.problem.quantities_of_interest):
                self.populate_tables_from_problem()

            # Gather bounds again
            def get_val(table, row, col, default):
                item = table.item(row, col)
                if item is None: return default
                text = item.text().strip()
                if not text: return default
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
                self.status_msg = QtWidgets.QProgressDialog("Resampling...", "Cancel", 0, 0, self)
                self.status_msg.setWindowModality(QtCore.Qt.WindowModal)
                self.status_msg.show()
            else:
                self.status_msg = None
            
            # Pass the COPIED box to the thread
            self.resample_thread = ResampleThread(
                self.problem, dv_par_box_copy, dsl, dsu, reqU, reqL, parameters, sample_size, active_plots, None
            )
            self.resample_thread.finished.connect(lambda s: self.on_resample_finished(s, silent))
            self.resample_thread.error.connect(self.on_resample_error)
            self.resample_thread.start()
            
        except Exception as e:
            self.resampling = False
            if not silent:
                QtWidgets.QMessageBox.critical(self, "Error", f"Resampling failed: {e}")
            
    def on_resample_finished(self, samples, silent=False):
        if self.status_msg:
            self.status_msg.close()
        self.btn_resample.setEnabled(True)
        self.resample_thread = None
        self.resampling = False
        self.process_results(samples, update_table=not silent)
        if not silent:
            QtWidgets.QMessageBox.information(self, "Success", "Resampling complete!")
        
        if self.pending_restart:
            self.pending_restart = False
            self.resample_box(silent=True)
        
    def on_resample_error(self, error_msg):
        if self.status_msg:
            self.status_msg.close()
        self.btn_resample.setEnabled(True)
        self.resample_thread = None
        self.resampling = False
        self.pending_restart = False
        QtWidgets.QMessageBox.critical(self, "Error", f"Resampling failed: {error_msg}")

    def clear_all_plots(self):
        for widget in self.plot_widgets:
            self.plots_layout.removeWidget(widget)
            widget.deleteLater()
        self.plot_widgets = []
        
    def _export_vector_layout(self, filename, widgets, main_title, legend_type="solution"):
        """
        Helper to export a grid of plots to a vector format (PDF/SVG).
        Renders the actual scene items rather than a raster screenshot.
        """
        from PySide6.QtPrintSupport import QPrinter
        from PySide6.QtSvg import QSvgGenerator
        
        # 1. Setup Device
        is_pdf = filename.lower().endswith('.pdf')
        
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
        
        try:
            # 2. Layout Constants
            margin = width * 0.05
            content_w = width - 2 * margin
            
            # 3. Draw Main Title
            font_title = QtGui.QFont("Arial", int(width * 0.025), QtGui.QFont.Bold)
            painter.setFont(font_title)
            title_rect = QtCore.QRectF(0, margin, width, height * 0.05)
            painter.drawText(title_rect, QtCore.Qt.AlignCenter, main_title)
            
            cursor_y = margin + height * 0.06
            
            # 4. Grid Layout for Plots
            if not widgets:
                return

            cols = 2
            rows = (len(widgets) + 1) // 2
            
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
                r = i // cols
                c = i % cols
                
                x = margin + c * (cell_w + cell_gap)
                y = cursor_y + r * (cell_h + cell_gap)
                
                # Draw Sub-Title
                sub_title = widget.lbl_title.text()
                sub_font = QtGui.QFont("Arial", int(cell_w * 0.05), QtGui.QFont.Bold)
                painter.setFont(sub_font)
                
                # Title area
                sub_title_h = cell_h * 0.1
                sub_rect = QtCore.QRectF(x, y, cell_w, sub_title_h)
                painter.drawText(sub_rect, QtCore.Qt.AlignCenter, sub_title)
                
                # Render Plot Scene
                # We target the specific QGraphicsScene of the widget
                scene = widget.plot_widget.scene()
                plot_target = QtCore.QRectF(x, y + sub_title_h, cell_w, cell_h - sub_title_h)
                
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
            painter.setFont(QtGui.QFont("Arial", int(width * 0.015), QtGui.QFont.Bold))
            painter.drawText(QtCore.QRectF(margin, legend_y, content_w, 30), QtCore.Qt.AlignLeft, "Legend")
            
            legend_content_y = legend_y + 40
            item_h = 30
            icon_size = 20
            
            # Prepare Items
            items = []
            if legend_type == "solution":
                items.append(("Good Design", "#00aa00"))

                # Get violations
                if hasattr(self, 'outputs') and self.outputs:
                    names = self.outputs
                elif self.problem:
                    names = [q['name'] for q in self.problem.quantities_of_interest]
                else:
                    names = []
                    
                for name in names:
                    color = self.qoi_colors.get(name, 'red')
                    items.append((f"Violating {name}", color))
            else:
                items.append(("Good Design", "#00aa00"))
                items.append(("Bad Design", "#ff0000"))
                
            # Draw items (Simple Flow Layout)
            lx = margin
            ly = legend_content_y
            
            font_leg = QtGui.QFont("Arial", int(width * 0.012))
            painter.setFont(font_leg)
            metrics = QtGui.QFontMetrics(font_leg)
            
            for label, color_code in items:
                # Icon
                painter.setBrush(QtGui.QColor(color_code))
                painter.setPen(QtCore.Qt.black)
                painter.drawRect(lx, ly, icon_size, icon_size)
                
                # Text
                text_w = metrics.horizontalAdvance(label)
                text_rect = QtCore.QRectF(lx + icon_size + 10, ly, text_w + 10, icon_size)
                painter.setPen(QtCore.Qt.black)
                painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)
                
                # Advance cursor
                lx += icon_size + 10 + text_w + 30
                # Wrap line if needed
                if lx > width - margin:
                    lx = margin
                    ly += item_h + 5

        finally:
            painter.end()

    def save_all_plots(self):
        if not self.plot_widgets:
            QtWidgets.QMessageBox.warning(self, "Warning", "No plots to save.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save All Plots", "", 
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf);;SVG Image (*.svg)"
        )
        if not filename:
            return

        try:
            # Prepare UI (hide buttons) for raster screenshot consistency
            for widget in self.plot_widgets:
                widget.btn_remove.hide()
                widget.btn_zoom.hide()
                widget.btn_save.hide()
                widget.lbl_title.hide()
            
            QtWidgets.QApplication.processEvents()
            
            if filename.lower().endswith('.pdf') or filename.lower().endswith('.svg'):
                # Use new vector export
                title = "Solution Spaces"
                if hasattr(self, 'lbl_global_title'):
                    title = self.lbl_global_title.text()
                self._export_vector_layout(filename, self.plot_widgets, title, "solution")
                QtWidgets.QMessageBox.information(self, "Success", f"Plots saved to {filename}")
            else:
                # Existing raster logic for PNG/JPG
                scale_factor = 2.0  # High quality raster
                
                plots_pixmap = self.plots_container.grab()
                if scale_factor > 1.0:
                    plots_pixmap = plots_pixmap.scaled(
                        plots_pixmap.size() * scale_factor,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation
                    )
                
                # Legend grab
                old_style = self.legend_group.styleSheet()
                self.legend_group.setStyleSheet("QGroupBox { background-color: white; color: black; border: none; } QLabel { color: black; }")
                QtWidgets.QApplication.processEvents() 
                legend_pixmap = self.legend_group.grab()
                if scale_factor > 1.0:
                    legend_pixmap = legend_pixmap.scaled(
                        legend_pixmap.size() * scale_factor,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation
                    )
                self.legend_group.setStyleSheet(old_style)
                
                # Compose
                padding = int(20 * scale_factor)
                title_height = int(60 * scale_factor)
                total_width = plots_pixmap.width() + legend_pixmap.width() + padding
                total_height = max(plots_pixmap.height(), legend_pixmap.height()) + title_height
                
                final_pixmap = QtGui.QPixmap(total_width, total_height)
                final_pixmap.fill(QtCore.Qt.white)
                
                painter = QtGui.QPainter(final_pixmap)
                title_text = self.lbl_global_title.text() if hasattr(self, 'lbl_global_title') else "Solution Spaces"
                font = QtGui.QFont("Arial", int(20 * scale_factor), QtGui.QFont.Bold)
                painter.setFont(font)
                painter.setPen(QtCore.Qt.black)
                rect = QtCore.QRect(0, 0, total_width, title_height)
                painter.drawText(rect, QtCore.Qt.AlignCenter, title_text)
                
                painter.drawPixmap(0, title_height, plots_pixmap)
                painter.drawPixmap(plots_pixmap.width() + padding, title_height, legend_pixmap)
                painter.end()
                
                final_pixmap.save(filename)
                QtWidgets.QMessageBox.information(self, "Success", f"Plots saved to {filename}")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save plots: {e}")
        finally:
            # Restore UI
            for widget in self.plot_widgets:
                widget.btn_remove.show()
                widget.btn_zoom.show()
                widget.btn_save.show()
                widget.lbl_title.show()

    def add_plot(self, x_name=None, y_name=None):
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
        row = count // 2
        col = count % 2
        self.plots_layout.addWidget(plot_widget, row, col)
        self.plot_widgets.append(plot_widget)
        self.plots_container.update()
        self.resample_box(silent=True)
        
    def remove_plot(self, plot_widget):
        if plot_widget in self.plot_widgets:
            self.plot_widgets.remove(plot_widget)
            plot_widget.deleteLater()
            
            while self.plots_layout.count():
                item = self.plots_layout.takeAt(0)
            
            for i, widget in enumerate(self.plot_widgets):
                row = i // 2
                col = i % 2
                self.plots_layout.addWidget(widget, row, col)
            
            self.plots_container.update()
            self.resample_box(silent=True)

    def process_results(self, samples, update_table=True):
        # 1. Store Global Samples
        if isinstance(samples, list) and len(samples) > 0:
            self.last_samples = {
                "points": np.hstack([s["points"] for s in samples]),
                "is_good": np.concatenate([s["is_good"] for s in samples]),
                "is_bad": np.concatenate([s["is_bad"] for s in samples]),
                "violation_idx": np.concatenate([s["violation_idx"] for s in samples]),
                "qoi_values": np.hstack([s["qoi_values"] for s in samples])
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
                self.problem.samples[dv['name']] = self.last_samples['points'][i, :]
            self.problem.results = {}
            for i, qoi in enumerate(self.problem.quantities_of_interest):
                self.problem.results[qoi['name']] = self.last_samples['qoi_values'][i, :]

        self.update_all_plots()
        if update_table:
            self.update_data_table()
        
        model_name = "Unknown Model"
        if self.problem and hasattr(self.problem, 'name'):
            model_name = self.problem.name
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
                # Cancel any running interpolation threads
                if pw.interpolation_thread is not None and pw.interpolation_thread.isRunning():
                    pw.interpolation_thread.cancel()
                    pw.interpolation_thread.wait()
        
        for pw in self.plot_widgets:
            pw.plot()

        self.update_legend()
        self.updating_plots = False

    def update_legend(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.add_legend_item("Good Design", '#00aa00', legend_layout=self.legend_layout)
        
        qoi_names = []
        if hasattr(self, 'outputs') and self.outputs:
            qoi_names = self.outputs
        elif self.problem:
            qoi_names = [q['name'] for q in self.problem.quantities_of_interest]
        for name in qoi_names:
            color = self.qoi_colors.get(name, 'red')
            self.add_legend_item(f"Violating {format_html(name)}", color, legend_layout=self.legend_layout)

    def add_legend_item(self, name, color, legend_layout=None):
        item = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(item)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(10)
        lbl_color = QtWidgets.QLabel()
        lbl_color.setFixedSize(18, 18)
        lbl_color.setStyleSheet(f"background-color: {color}; border: 1px solid #666; border-radius: 4px;")
        lbl_text = QtWidgets.QLabel(name)
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
        data['Is Good'] = self.last_samples['is_good']
        data['Is Bad'] = self.last_samples['is_bad']
        df = pd.DataFrame(data)
        self.data_table.setRowCount(df.shape[0])
        self.data_table.setColumnCount(df.shape[1])
        self.data_table.setHorizontalHeaderLabels(df.columns)
        limit = min(10000, df.shape[0])
        self.data_table.setRowCount(limit)
        for i in range(limit):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                item = QtWidgets.QTableWidgetItem(f"{val:.4f}")
                self.data_table.setItem(i, j, item)

    def export_csv(self):
        if not self.last_samples:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data available to export.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "results.csv", "CSV Files (*.csv)")
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
                data['Is Good'] = self.last_samples['is_good']
                data['Is Bad'] = self.last_samples['is_bad']
                
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
                
                QtWidgets.QMessageBox.information(self, "Success", f"Successfully exported {len(df)} rows to {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def compute_product_family(self):
        if not self.problem or not self.problem.requirement_sets:
            QtWidgets.QMessageBox.warning(self, "Warning", "No variants defined.")
            return
        
        try:
            # Switch to product family mode
            self.set_product_family_mode(True)
            
            # Get current bounds (ensure float type)
            # dsl/dsu are the physical limits of the Design Space
            dsl = np.array([dv['min'] for dv in self.problem.design_variables], dtype=float)
            dsu = np.array([dv['max'] for dv in self.problem.design_variables], dtype=float)
            
            # l/u are the "current search box" limits. 
            # For a fresh start, these should equal dsl/dsu (the full Design Space).
            # They MUST match the number of Design Variables (DVs).
            l = np.array([dv['min'] for dv in self.problem.design_variables], dtype=float)
            u = np.array([dv['max'] for dv in self.problem.design_variables], dtype=float)
            parameters = None  # Assuming no parameters for now
            
            from .computation_engine import compute_product_family
            solver_type = self.family_solver_combo.currentData()
            results = compute_product_family(self.problem, dsl, dsu, l, u, parameters, slider_value=self.slider_mosse.value() / 100.0, solver_type=solver_type)
            
            # Visualize results
            self.plot_product_family(results)
            
            QtWidgets.QMessageBox.information(self, "Success", f"Computed {len(results)-1} variants and platform.")
            
        except Exception as e:
            # If computation fails, return to normal mode
            self.set_product_family_mode(False)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute product family: {e}")

    def plot_product_family(self, results):
        """
        Visualizes the ranges for each product variant and the common platform.
        Now includes explicit Axis Labels and Titles for clearer DV identification.
        """
        # Check for valid results
        if not results or not self.problem or not self.problem.design_variables:
            return
            
        # 1. Clear existing plots in the scroll area
        while self.family_plots_layout.count():
            item = self.family_plots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 2. Setup Colors for Variants
        variant_names = sorted([name for name in results.keys() if name not in ['Platform', 'Platform_Infeasible', 'Communality']])
        var_colors = {}
        for i, name in enumerate(variant_names):
            var_colors[name] = self.default_colors[i % len(self.default_colors)]

        # 3. Create a Plot for EACH Design Variable
        cols = 3
        for i, dv in enumerate(self.problem.design_variables):
            
            # Container for layout
            container = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(container)
            vbox.setContentsMargins(0, 10, 0, 20) # Add spacing between plots
            
            # PyQtGraph Widget
            win = pg.GraphicsLayoutWidget()
            win.setFixedHeight(220) # Slightly taller for labels
            win.setBackground('w')
            vbox.addWidget(win)
            
            # Setup Plot Item
            p = win.addPlot()
            p.showGrid(x=True, y=False, alpha=0.3)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=False)
            
            # --- IMPROVEMENT 1: X-Axis Label with DV Name ---
            unit_str = f" [{dv.get('unit', '-')}]" if dv.get('unit') else ""
            label_text = f"{dv['name']}{unit_str}"
            # Use HTML for bold labeling
            p.setLabel('bottom', text=label_text, **{'font-size': '11pt', 'font-weight': 'bold', 'color': 'black'})

            # Y-Axis Labels (Variant Names)
            y_ticks = [(0, "PLATFORM")]
            for idx, name in enumerate(variant_names):
                y_ticks.append((idx + 1, name))
            
            ax_left = p.getAxis('left')
            ax_left.setTicks([y_ticks])
            ax_left.setPen('k')
            ax_left.setTextPen('k')
            p.getAxis('bottom').setPen('k')
            p.getAxis('bottom').setTextPen('k')
            
            p.setYRange(-0.5, len(variant_names) + 0.5)
            
            # --- DRAWING BARS ---
            min_view = float('inf')
            max_view = float('-inf')
            
            # 1. Draw Variant Ranges
            for idx, var_name in enumerate(variant_names):
                box = results[var_name]
                if box is not None and i < box.shape[0]:
                    x_min = box[i, 0]
                    x_max = box[i, 1]
                    min_view = min(min_view, x_min)
                    max_view = max(max_view, x_max)
                    
                    color = QtGui.QColor(var_colors[var_name])
                    color.setAlpha(150)
                    
                    bar = pg.BarGraphItem(x0=[x_min], y=[idx + 1], width=[x_max - x_min], height=0.6, brush=pg.mkBrush(color), pen=pg.mkPen('k'))
                    p.addItem(bar)
                    
                    # Numeric Label
                    text = pg.TextItem(f"{x_min:.2f} - {x_max:.2f}", anchor=(0, 0.5), color='k')
                    text.setPos(x_max, idx + 1)
                    p.addItem(text)

            # 2. Draw Platform (Intersection)
            platform_exists = False
            if 'Platform' in results and results['Platform'] is not None:
                platform = results['Platform']
                if i < platform.shape[0]:
                    p_min = platform[i, 0]
                    p_max = platform[i, 1]
                    
                    if p_min < p_max: # Valid intersection
                        platform_exists = True
                        bar = pg.BarGraphItem(x0=[p_min], y=[0], width=[p_max - p_min], height=0.8, brush=pg.mkBrush('#00cc00'), pen=pg.mkPen('k', width=2))
                        p.addItem(bar)
                        
                        text = pg.TextItem(f"{p_min:.2f} - {p_max:.2f}", anchor=(0, 0.5), color='#006600')
                        text.setPos(p_max, 0)
                        text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
                        p.addItem(text)
                        
                        # Alignment Guides
                        p.addItem(pg.InfiniteLine(pos=p_min, angle=90, pen=pg.mkPen('g', style=QtCore.Qt.DotLine, width=2)))
                        p.addItem(pg.InfiniteLine(pos=p_max, angle=90, pen=pg.mkPen('g', style=QtCore.Qt.DotLine, width=2)))

            # --- IMPROVEMENT 2: Combined Title (Name + Status) ---
            if platform_exists:
                status_text = "Common Platform Feasible"
                status_color = "#008800"
            elif 'Platform_Infeasible' in results and results['Platform_Infeasible']:
                status_text = "Platform Infeasible (No Commonality)"
                status_color = "#cc0000"
            else:
                status_text = "No Commonality"
                status_color = "#cc0000"
            
            # HTML Title: Name in Large Bold, Status smaller below
            title_html = (
                f"<span style='font-size: 14pt; font-weight: bold; color: black;'>{dv['name']}</span>"
                f"<br><span style='font-size: 9pt; color: {status_color};'>{status_text}</span>"
            )
            p.setTitle(title_html)

            # Auto-scale View - ensure platform is always visible
            if min_view != float('inf'):
                # Include platform bounds in the range calculation
                if platform_exists:
                    min_view = min(min_view, p_min)
                    max_view = max(max_view, p_max)
                
                width = max_view - min_view
                padding = width * 0.1 if width > 0 else 1.0
                p.setXRange(min_view - padding, max_view + padding * 3)

            # Add to Grid Layout
            row = i // cols
            col = i % cols
            self.family_plots_layout.addWidget(container, row, col)

    def save_to_folder(self, folder_path):
        """Save solution space state to a folder."""
        json_path = os.path.join(folder_path, 'solution_space.json')
        h5_path = os.path.join(folder_path, 'solution_space.h5')
        
        # Update problem from UI
        if self.problem:
            for i in range(self.dv_table.rowCount()):
                min_val = self._safe_get_float(self.dv_table.item(i, 2), -1e9)
                max_val = self._safe_get_float(self.dv_table.item(i, 3), 1e9)
                if i < len(self.problem.design_variables):
                    self.problem.design_variables[i]['min'] = min_val
                    self.problem.design_variables[i]['max'] = max_val

            for i in range(self.qoi_table.rowCount()):
                req_min = self._safe_get_float(self.qoi_table.item(i, 2), -1e9)
                req_max = self._safe_get_float(self.qoi_table.item(i, 3), 1e9)
                if i < len(self.problem.quantities_of_interest):
                    self.problem.quantities_of_interest[i]['min'] = req_min
                    self.problem.quantities_of_interest[i]['max'] = req_max

        problem_data = None
        if self.problem:
            problem_data = {
                'name': self.problem.name,
                'design_variables': self.problem.design_variables,
                'quantities_of_interest': self.problem.quantities_of_interest,
                'requirement_sets': self.problem.requirement_sets,
                'samples': self.problem.samples,
                'results': self.problem.results,
                'n_samples': self.problem.sample_size
            }

        data = {
            'code': self.system_code,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'problem_data': problem_data,
            'last_samples': self.last_samples,
            'dv_par_box': None,
            'qoi_colors': self.qoi_colors,
            'design_plots': [(w.x_name, w.y_name) for w in self.plot_widgets],
            'version': '1.0'
        }
        
        self.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = self.dv_par_box.copy() if self.dv_par_box is not None else None
            data['dv_par_box'] = dv_par_box_copy.tolist() if dv_par_box_copy is not None else None
        finally:
            self.dv_par_box_mutex.unlock()
        
        try:
            serializable_data = self._to_serializable(data)
            with open(json_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            if self.problem and (self.problem.samples or self.problem.results):
                with h5py.File(h5_path, 'w') as h5f:
                    if self.problem.samples:
                        for key, value in self.problem.samples.items():
                            h5f.create_dataset(f"samples/{key}", data=value)
                    if self.problem.results:
                        for key, value in self.problem.results.items():
                            h5f.create_dataset(f"results/{key}", data=value)
        except Exception as e:
            raise e

    def update_ui_from_problem(self):
        """Update UI elements based on the current problem definition."""
        if not self.problem:
            return

        # Update Sample Count
        if hasattr(self, 'sample_size_spin'):
            # Handle attribute name mismatch (sample_size vs n_samples)
            if hasattr(self.problem, 'sample_size'):
                self.sample_size_spin.setValue(self.problem.sample_size)
            elif hasattr(self.problem, 'n_samples'):
                self.sample_size_spin.setValue(self.problem.n_samples)

        # Update Design Variables Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(self.problem.design_variables))
        self.dv_par_box = np.zeros((len(self.problem.design_variables), 2))
        
        self.inputs = []
        self.input_units = {}
        
        for i, dv in enumerate(self.problem.design_variables):
            name = dv['name']
            self.inputs.append(name)
            self.input_units[name] = dv.get('unit', '-')
            
            min_val = dv['min']
            max_val = dv['max']
            
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(dv.get('unit', '-')))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space (or restore if we had it saved separately)
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val))) 
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))
            
            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val
            
        self.dv_table.blockSignals(False)

        # Update Quantities of Interest Table
        self.qoi_table.blockSignals(True)
        self.qoi_table.setRowCount(len(self.problem.quantities_of_interest))
        
        self.outputs = []
        self.output_units = {}
        
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            name = qoi['name']
            self.outputs.append(name)
            self.output_units[name] = qoi.get('unit', '-')
            
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.qoi_table.setItem(i, 1, QtWidgets.QTableWidgetItem(qoi.get('unit', '-')))
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(qoi.get('min', ''))))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(qoi.get('max', ''))))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto")) # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto")) # Plot Max
            
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(QtCore.Qt.Checked if qoi.get('minimize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 6, min_item)
            
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(QtCore.Qt.Checked if qoi.get('maximize', False) else QtCore.Qt.Unchecked)
            self.qoi_table.setItem(i, 7, max_item)
            
            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(qoi.get('weight', 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)
            
        self.qoi_table.blockSignals(False)

        # Update Axis Combos
        all_vars = self.inputs + self.outputs
        self.combo_add_x.clear()
        self.combo_add_x.addItems(all_vars)
        self.combo_add_y.clear()
        self.combo_add_y.addItems(all_vars)
        
        # Update Bounds for plotting
        try:
            self.dsl = np.array([float(dv['min']) for dv in self.problem.design_variables])
            self.dsu = np.array([float(dv['max']) for dv in self.problem.design_variables])
        except:
            self.dsl = None
            self.dsu = None

    def load_from_folder(self, folder_path):
        """Load solution space state from a folder."""
        json_path = os.path.join(folder_path, 'solution_space.json')
        h5_path = os.path.join(folder_path, 'solution_space.h5')
        
        if not os.path.exists(json_path):
            return # Nothing to load
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.system_code = data.get('code')
            self.inputs = data.get('inputs', [])
            self.outputs = data.get('outputs', [])
            self.qoi_colors = data.get('qoi_colors', {})
            
            # Restore problem
            p_data = data.get('problem_data')
            if p_data:
                self.problem = XRayProblem(p_data['name'], p_data['n_samples'])
                self.problem.design_variables = p_data['design_variables']
                self.problem.quantities_of_interest = p_data['quantities_of_interest']
                self.problem.requirement_sets = p_data.get('requirement_sets', {})
                
                # Load large data from H5 if available
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as h5f:
                        if 'samples' in h5f:
                            self.problem.samples = {k: np.array(v) for k, v in h5f['samples'].items()}
                        if 'results' in h5f:
                            self.problem.results = {k: np.array(v) for k, v in h5f['results'].items()}
                else:
                    # Fallback to JSON data (might be slow/large)
                    self.problem.samples = {k: np.array(v) for k, v in p_data.get('samples', {}).items()}
                    self.problem.results = {k: np.array(v) for k, v in p_data.get('results', {}).items()}
                
                # Recompile system model if code exists
                if self.system_code:
                    try:
                        # Use SystemModel to create a persistent file for multiprocessing support
                        # This ensures 'dill' can pickle the function correctly
                        from pylcss.system_modeling.system_model import SystemModel
                        
                        # Create dummy inputs/outputs for SystemModel creation if needed
                        # (SystemModel needs them but we only need the function here)
                        dummy_inputs = [{'name': n, 'min': 0, 'max': 1} for n in self.inputs]
                        dummy_outputs = [{'name': n, 'req_min': 0, 'req_max': 0} for n in self.outputs]
                        
                        sm = SystemModel.from_code_string(
                            self.problem.name, 
                            self.system_code, 
                            dummy_inputs, 
                            dummy_outputs
                        )
                        self.problem.set_system_model(sm.system_function)
                        
                    except Exception as e:
                        logger.warning("Failed to recompile system model", exc_info=True)
                        # Fallback to exec if file creation fails
                        try:
                            local_vars = {}
                            exec(self.system_code, globals(), local_vars)
                            if 'system_function' in local_vars:
                                self.problem.set_system_model(local_vars['system_function'])
                        except:
                            pass

            # Restore UI state
            self.update_ui_from_problem()
            
            # Restore System Combo and Models List
            if self.problem and self.system_code:
                # Reconstruct a model definition so the UI behaves correctly
                inputs_list = []
                if hasattr(self.problem, 'design_variables'):
                    inputs_list = self.problem.design_variables
                else:
                    inputs_list = [{'name': name, 'min': 0, 'max': 1} for name in self.inputs]

                outputs_list = []
                if hasattr(self.problem, 'quantities_of_interest'):
                    # Map min/max to req_min/req_max for compatibility with OptimizationWidget
                    for qoi in self.problem.quantities_of_interest:
                        q_dict = qoi.copy()
                        if 'min' in q_dict: q_dict['req_min'] = q_dict['min']
                        if 'max' in q_dict: q_dict['req_max'] = q_dict['max']
                        outputs_list.append(q_dict)
                else:
                    outputs_list = [{'name': name, 'req_min': 0, 'req_max': 0} for name in self.outputs]

                reconstructed_model = {
                    'name': self.problem.name,
                    'code': self.system_code,
                    'inputs': inputs_list,
                    'outputs': outputs_list
                }
                self.models = [reconstructed_model]
                
                # Save the loaded problem with data before it gets overwritten
                loaded_problem = self.problem
                
                self.system_combo.blockSignals(True)
                self.system_combo.clear()
                self.system_combo.addItem(self.problem.name)
                self.system_combo.blockSignals(False)
                
                # Trigger load_selected_system to initialize UI, bounds, and buttons
                self.load_selected_system()
                
                # Restore the problem with loaded data
                self.problem = loaded_problem
                
                # Refresh UI with restored data
                self.update_ui_from_problem()
                self.update_all_plots()
                
                # Enable buttons
                self.btn_compute_feasible.setEnabled(True)
                
                has_objectives = any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                                   for qoi in self.problem.quantities_of_interest)
                self.chk_include_optimization.setEnabled(has_objectives)
                self.btn_compute_family.setEnabled(True)
                
                # Enable resample if we have samples
                if self.problem.samples:
                    self.btn_resample.setEnabled(True)
            
            # Restore last samples
            ls = data.get('last_samples')
            if ls:
                self.last_samples = {
                    'points': np.array(ls['points']),
                    'qoi_values': np.array(ls['qoi_values']),
                    'is_good': np.array(ls['is_good']),
                    'violation_idx': np.array(ls['violation_idx'])
                }
            
            # Restore ROI box
            box_list = data.get('dv_par_box')
            if box_list:
                self.dv_par_box_mutex.lock()
                self.dv_par_box = np.array(box_list)
                self.dv_par_box_mutex.unlock()
            
            # Update plots
            self.update_all_plots()
            
            # Restore Design Space Plots
            design_plots = data.get('design_plots', [])
            if design_plots:
                self.clear_all_plots()
                for x_name, y_name in design_plots:
                    self.add_plot(x_name, y_name)
            
            for pw in self.plot_widgets:
                pw.plot()
                
            # Trigger replot if possible
            # self.update_plots() # Might be too early if UI not fully ready
            
        except Exception as e:
            raise e

    def save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project", "", "PFD Project (*.pfd)")
        if path:
            if self.problem:
                for i in range(self.dv_table.rowCount()):
                    min_val = self._safe_get_float(self.dv_table.item(i, 2), -1e9)
                    max_val = self._safe_get_float(self.dv_table.item(i, 3), 1e9)
                    if i < len(self.problem.design_variables):
                        self.problem.design_variables[i]['min'] = min_val
                        self.problem.design_variables[i]['max'] = max_val

                for i in range(self.qoi_table.rowCount()):
                    req_min = self._safe_get_float(self.qoi_table.item(i, 2), -1e9)
                    req_max = self._safe_get_float(self.qoi_table.item(i, 3), 1e9)
                    if i < len(self.problem.quantities_of_interest):
                        self.problem.quantities_of_interest[i]['min'] = req_min
                        self.problem.quantities_of_interest[i]['max'] = req_max

            problem_data = None
            if self.problem:
                problem_data = {
                    'name': self.problem.name,
                    'design_variables': self.problem.design_variables,
                    'quantities_of_interest': self.problem.quantities_of_interest,
                    'samples': self.problem.samples,
                    'results': self.problem.results,
                    'n_samples': self.problem.sample_size
                }

            data = {
                'code': self.system_code,
                'inputs': self.inputs,
                'outputs': self.outputs,
                'problem_data': problem_data,
                'last_samples': self.last_samples,
                'dv_par_box': None,
                'qoi_colors': self.qoi_colors,
                'version': '1.0'
            }
            
            self.dv_par_box_mutex.lock()
            try:
                dv_par_box_copy = self.dv_par_box.copy() if self.dv_par_box is not None else None
                data['dv_par_box'] = dv_par_box_copy.tolist() if dv_par_box_copy is not None else None
            finally:
                self.dv_par_box_mutex.unlock()
            
            json_path = path
            h5_path = path.replace('.pfd', '.h5')
            
            try:
                serializable_data = self._to_serializable(data)
                with open(json_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                if self.problem and (self.problem.samples or self.problem.results):
                    with h5py.File(h5_path, 'w') as h5f:
                        if self.problem.samples:
                            for key, value in self.problem.samples.items():
                                h5f.create_dataset(f'samples/{key}', data=value)
                        if self.problem.results:
                            for key, value in self.problem.results.items():
                                h5f.create_dataset(f'results/{key}', data=value)
                QtWidgets.QMessageBox.information(self, "Saved", "Project saved successfully.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Project", "", "PFD Project (*.pfd)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                self.system_code = data.get('code')
                self.inputs = data.get('inputs', [])
                self.outputs = data.get('outputs', [])
                self.last_samples = data.get('last_samples')
                
                # Convert lists back to NumPy arrays for last_samples
                if self.last_samples:
                    self.last_samples['points'] = np.array(self.last_samples['points'])
                    self.last_samples['qoi_values'] = np.array(self.last_samples['qoi_values'])
                    self.last_samples['is_good'] = np.array(self.last_samples['is_good'])
                    self.last_samples['is_bad'] = np.array(self.last_samples['is_bad'])
                    self.last_samples['violation_idx'] = np.array(self.last_samples['violation_idx'])
                
                dv_par_box_data = data.get('dv_par_box')
                self.dv_par_box = np.array(dv_par_box_data) if dv_par_box_data is not None else None
                
                self.qoi_colors = data.get('qoi_colors', {})
                forbidden_greens = ['#00aa00', '#3cb44b', '#bcf60c', '#008080', '#aaffc3', '#808000', '#00ff00', '#008000']
                for name, color in self.qoi_colors.items():
                    if color.lower() in forbidden_greens:
                        idx = 0
                        if self.outputs and name in self.outputs:
                            idx = self.outputs.index(name)
                        self.qoi_colors[name] = self.default_colors[idx % len(self.default_colors)]
                
                h5_path = path.replace('.pfd', '.h5')
                samples_data = {}
                results_data = {}
                
                if os.path.exists(h5_path):
                    try:
                        with h5py.File(h5_path, 'r') as h5f:
                            if 'samples' in h5f:
                                for key in h5f['samples'].keys():
                                    samples_data[key] = h5f[f'samples/{key}'][:]
                            if 'results' in h5f:
                                for key in h5f['results'].keys():
                                    results_data[key] = h5f[f'results/{key}'][:]
                    except Exception as e:
                        logger.warning("Could not load HDF5 data", exc_info=True)
                
                p_data = data.get('problem_data')
                if p_data:
                    system_func = None
                    if self.system_code:
                        try:
                            # Use safe execution instead of exec()
                            system_func = self._execute_code_safely(self.system_code)
                        except:
                            pass
                    
                    self.problem = XRayProblem(p_data.get('name', "Loaded_Model"), p_data['n_samples'])
                    if system_func:
                        self.problem.set_system_model(system_func)
                        
                    self.problem.design_variables = p_data['design_variables']
                    self.problem.quantities_of_interest = p_data['quantities_of_interest']
                    
                    if samples_data:
                        self.problem.samples = samples_data
                    elif 'samples' in p_data:
                        self.problem.samples = p_data['samples']
                        # Convert lists to NumPy arrays
                        for key in self.problem.samples:
                            self.problem.samples[key] = np.array(self.problem.samples[key])
                        
                    if results_data:
                        self.problem.results = results_data
                    elif 'results' in p_data:
                        self.problem.results = p_data['results']
                        # Convert lists to NumPy arrays
                        for key in self.problem.results:
                            self.problem.results[key] = np.array(self.problem.results[key])
                
                self.populate_tables_from_problem()
                self.update_all_plots()
                self.update_data_table()
                
                if self.problem and self.problem.system_model:
                     self.btn_compute_feasible.setEnabled(True)
                     # Check if there are any objectives defined
                     has_objectives = any(qoi.get('minimize', False) or qoi.get('maximize', False) 
                                        for qoi in self.problem.quantities_of_interest)
                     self.btn_compute_family.setEnabled(True)
                     self.resample_box(silent=True)
                     
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")

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
        if not hasattr(self.problem, 'requirement_sets') or not self.problem.requirement_sets:
            QtWidgets.QMessageBox.warning(self, "Warning", "No product variants defined. Please add variants first.")
            return

        # Gather parameters
        try:
            # DVs
            dsl = []
            dsu = []
            l = [] 
            u = []
            
            for i in range(self.dv_table.rowCount()):
                dsl.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
                dsu.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
                l.append(self._safe_get_float(self.dv_table.item(i, 2), -1e9))
                u.append(self._safe_get_float(self.dv_table.item(i, 3), 1e9))
                
            dsl = np.array(dsl)
            dsu = np.array(dsu)
            l = np.array(l)
            u = np.array(u)
            
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
            slider_val = self.slider_mosse.value() / 100.0
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
                self.problem, weight, dsl, dsu, l, u, reqU, reqL, 
                parameters, slider_val, solver_type
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
            if 'Communality' in results and results['Communality'] is not None:
                self.display_communality_info(results['Communality'])
            
            # Switch to Product Family Analysis tab
            # Find the index of the Product Family Analysis tab
            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == "Product Family Analysis":
                    self.right_tabs.setCurrentIndex(i)
                    break
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Product family computation complete!\nComputed {len(self.problem.requirement_sets)} variants and platform."
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
            var_names = [dv['name'] for dv in self.problem.design_variables]
        else:
            var_names = [f"DV{i+1}" for i in range(len(communality))]
        
        for i, comm_val in enumerate(communality):
            # Variable name
            var_name = var_names[i] if i < len(var_names) else f"DV{i+1}"
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
        QtWidgets.QMessageBox.critical(self, "Error", f"Product family computation failed: {error_msg}")

    def on_family_cancelled(self):
        """Handle cancellation of product family computation."""
        if hasattr(self, 'family_worker'):
            self.family_worker.stop()
        self.family_progress.close()
        self.btn_compute_family.setEnabled(True)

    def update_family_plots(self):
        """Update product family visualization plots."""
        if not hasattr(self, 'family_results') or not self.family_results:
            return
            
        # Clear existing family plots
        for i in reversed(range(self.family_plots_layout.count())):
            widget = self.family_plots_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Create plots for each variant and platform
        plot_widgets = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        row = 0
        col = 0
        max_cols = 2
        
        for i, (variant_name, box) in enumerate(self.family_results.items()):
            if box is None:
                continue
                
            # Create a simple plot showing the variant box
            plot_widget = pg.PlotWidget()
            plot_widget.setTitle(f"{variant_name}")
            plot_widget.setLabel('left', 'Design Variable 2')
            plot_widget.setLabel('bottom', 'Design Variable 1')
            
            # Add rectangle for the box
            if box.shape[0] >= 2:  # At least 2D
                x_min, x_max = box[0, 0], box[0, 1]
                y_min, y_max = box[1, 0], box[1, 1]
                
                # Create rectangle
                rect = QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
                rect_item = pg.QtGui.QGraphicsRectItem(rect)
                rect_item.setPen(pg.mkPen(colors[i % len(colors)], width=2))
                rect_item.setBrush(pg.mkBrush(colors[i % len(colors)], alpha=100))
                plot_widget.addItem(rect_item)
                
                # Set axis ranges with some padding
                padding = 0.1
                x_range = x_max - x_min
                y_range = y_max - y_min
                plot_widget.setXRange(x_min - padding * x_range, x_max + padding * x_range)
                plot_widget.setYRange(y_min - padding * y_range, y_max + padding * y_range)
            
            plot_widgets.append(plot_widget)
            
            # Add to layout
            self.family_plots_layout.addWidget(plot_widget, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Update the container
        self.family_plots_container.update()

    def add_variant(self):
        """Add a new product variant."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Variant", "Variant Name:")
        if ok and name:
            # Check if variant already exists
            for row in range(self.variant_table.rowCount()):
                if self.variant_table.item(row, 0).text() == name:
                    QtWidgets.QMessageBox.warning(self, "Warning", f"Variant '{name}' already exists.")
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
                self, "Remove Variant", 
                f"Are you sure you want to remove variant '{name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
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
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a variant first.")
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






