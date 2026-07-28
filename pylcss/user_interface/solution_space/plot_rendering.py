# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Rendering and interpolation behavior for solution-space plots."""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.solution_space.interpolation_worker import (
    InterpolationThread,
)

from ..common.text_utils import format_html

logger = logging.getLogger(__name__)

__all__ = ["PlotRenderingMixin"]


class PlotRenderingMixin:
    def plot(self):
        if self.plotting:
            return
        self.plotting = True

        # This prevents "zombie" threads from updating the plot while we are drawing new points.
        if self.interpolation_thread is not None:
            if self.interpolation_thread.isRunning():
                self.interpolation_thread.cancel()
                # Disconnect signals to prevent late updates
                try:
                    self.interpolation_thread.finished.disconnect()
                    self.interpolation_thread.quick_result.disconnect()
                    self.interpolation_thread.error.disconnect()
                except (RuntimeError, TypeError):
                    pass
                # Move to old threads list to keep reference until it finishes naturally
                self.old_threads.append(self.interpolation_thread)
            self.interpolation_thread = None

        if not self.parent_widget or not self.parent_widget.problem:
            self.plotting = False
            return

        # SMART CLEAR - Remove data but keep structure
        self.plot_widget.clear()  # Clears items managed by PlotItem

        # Explicit cleanup of heavy references
        self.scatter_good = None
        self.scatter_bad = None
        self.scatter_optimal = None
        self.img_item = None
        self.limit_lines = []
        self.hull_item = None
        self.roi_item = None  # ROI is re-added if needed
        self.roi_lines = []

        if hasattr(self, "color_scatter_items"):
            self.color_scatter_items = []

        x_name = self.x_name
        y_name = self.y_name
        # Detect if this is an objective plot (both axes are outputs)
        is_objective_plot = False

        points = self.samples["points"] if self.samples is not None else None
        qoi_values = self.samples["qoi_values"] if self.samples is not None else None
        is_good = self.samples["is_good"] if self.samples is not None else None
        violation_idx = (
            self.samples["violation_idx"] if self.samples is not None else None
        )

        def get_data(name):
            if points is not None and name in self.parent_widget.inputs:
                idx = self.parent_widget.inputs.index(name)
                return points[idx, :]
            elif qoi_values is not None and name in [
                q["name"] for q in self.parent_widget.problem.quantities_of_interest
            ]:
                idx = [
                    q["name"] for q in self.parent_widget.problem.quantities_of_interest
                ].index(name)
                return qoi_values[idx, :]
            return None

        x_data = get_data(x_name)
        y_data = get_data(y_name)
        # Continue even if no data, for optimized point

        def get_bounds(name):
            if name in self.parent_widget.inputs:
                idx = self.parent_widget.inputs.index(name)
                if (
                    hasattr(self.parent_widget, "dsl")
                    and hasattr(self.parent_widget, "dsu")
                    and self.parent_widget.dsl is not None
                    and idx < len(self.parent_widget.dsl)
                ):  # Check size!
                    lower = self.parent_widget.dsl[idx]
                    upper = self.parent_widget.dsu[idx]
                    # Handle infinite bounds
                    if not np.isfinite(lower):
                        lower = -1e6  # Use a large finite value
                    if not np.isfinite(upper):
                        upper = 1e6  # Use a large finite value
                    return lower, upper
            data = get_data(name)
            if data is not None:
                # Filter out NaN and infinite values for min/max calculation
                valid_data = data[np.isfinite(data)]
                if len(valid_data) > 0:
                    d_min, d_max = np.min(valid_data), np.max(valid_data)
                    rng = d_max - d_min
                    if rng == 0:
                        rng = 1.0
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
            dv_par_box_copy = (
                self.parent_widget.dv_par_box.copy()
                if self.parent_widget.dv_par_box is not None
                else None
            )
        finally:
            self.parent_widget.dv_par_box_mutex.unlock()

        if dv_par_box_copy is not None:
            dvs = [dv["name"] for dv in self.parent_widget.problem.design_variables]
            if x_name in dvs and y_name in dvs:
                x_idx = dvs.index(x_name)
                y_idx = dvs.index(y_name)
                box_x_min, box_x_max = (
                    dv_par_box_copy[x_idx, 0],
                    dv_par_box_copy[x_idx, 1],
                )
                box_y_min, box_y_max = (
                    dv_par_box_copy[y_idx, 0],
                    dv_par_box_copy[y_idx, 1],
                )
                x_min = min(x_min, box_x_min)
                x_max = max(x_max, box_x_max)
                y_min = min(y_min, box_y_min)
                y_max = max(y_max, box_y_max)

        # Set axis ranges
        self.plot_item.setXRange(x_min, x_max, padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)

        # Force ViewBox ranges with no padding and disable auto-ranging
        view_box = self.plot_widget.getViewBox()
        view_box.setRange(
            QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min), padding=0
        )
        view_box.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        view_box.enableAutoRange(enable=False)

        # Ensure all arrays are 1D before any masking or processing
        if x_data is not None:
            x_data = np.asarray(x_data).ravel()
        if y_data is not None:
            y_data = np.asarray(y_data).ravel()
        if is_good is not None:
            is_good = np.asarray(is_good).ravel()
        if violation_idx is not None:
            violation_idx = np.asarray(violation_idx).ravel()

        # Apply filtering
        if self.samples is not None and x_data is not None and y_data is not None:
            # 1. Create the mask for valid (finite) numbers
            # Now safe because x_data and y_data are guaranteed 1D
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)

            # 2. Apply mask (Results are 1D)
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]

            # 3. Apply mask to status arrays
            if is_good is not None:
                # Resize is_good to match valid points (safe because both are 1D)
                if len(is_good) == len(valid_mask):
                    is_good = is_good[valid_mask]
                else:
                    # Fallback if lengths mismatch (should not happen with strict sync)
                    is_good = is_good[: len(valid_mask)][valid_mask]

            if violation_idx is not None:
                if len(violation_idx) == len(valid_mask):
                    violation_idx = violation_idx[valid_mask]

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
                        x=bad_x,
                        y=bad_y,
                        pen=pg.mkPen("w", width=0.5),
                        brush=pg.mkBrush("#ff0000"),
                        size=6,
                        alpha=0.6,
                    )
                    self.plot_widget.addItem(self.scatter_bad)

                # Plot good points (green)
                if np.any(good_mask):
                    good_x = x_data[good_mask]
                    good_y = y_data[good_mask]
                    self.scatter_good = pg.ScatterPlotItem(
                        x=good_x,
                        y=good_y,
                        pen=pg.mkPen("w", width=0.5),
                        brush=pg.mkBrush("#00aa00"),
                        size=6,
                        alpha=0.6,
                    )
                    self.plot_widget.addItem(self.scatter_good)
            else:
                # Solution space plot: color by QOI violation
                viz_mode = self.parent_widget.combo_viz_mode.currentText()
                num_qoi = len(self.parent_widget.problem.quantities_of_interest)
                qoi_names = [
                    q["name"] for q in self.parent_widget.problem.quantities_of_interest
                ]

                if viz_mode == "Categorical Areas":
                    # Create interpolated filled areas using KNN interpolation
                    try:
                        # Create grid for interpolation
                        x_grid, y_grid = np.meshgrid(
                            np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500),
                        )

                        # Prepare data for interpolation
                        points = np.column_stack((x_data, y_data))

                        # Create color index array (0=good, 1+=violation types)
                        color_indices = np.zeros(len(x_data))
                        for i in range(len(x_data)):
                            if not is_good[i]:
                                if violation_idx is not None and i < len(violation_idx):
                                    v_idx = int(violation_idx[i]) % num_qoi
                                    color_indices[i] = (
                                        v_idx + 1
                                    )  # 1, 2, 3... for different violations
                                else:
                                    color_indices[i] = num_qoi + 1  # Unknown violation

                        # Create hash of current data and bounds for caching
                        import hashlib

                        data_hash = hashlib.md5()
                        data_hash.update(points.tobytes())
                        data_hash.update(color_indices.tobytes())
                        data_hash.update(
                            np.array([x_min, x_max, y_min, y_max]).tobytes()
                        )
                        current_hash = data_hash.hexdigest()

                        # Check if we can use cached result
                        if (
                            self.cached_data_hash == current_hash
                            and self.cached_categorical_img is not None
                        ):
                            # Use cached image
                            self.img_item = pg.ImageItem(self.cached_categorical_img)
                            self.img_item.setRect(
                                QtCore.QRectF(
                                    x_min, y_min, x_max - x_min, y_max - y_min
                                )
                            )
                            self.img_item.setZValue(0)
                            self.plot_widget.addItem(self.img_item)

                        # Check if interpolation is already running
                        if (
                            self.interpolation_thread is not None
                            and self.interpolation_thread.isRunning()
                        ):
                            # Cancel the existing thread
                            self.interpolation_thread.cancel()
                            # Do NOT wait() as it freezes GUI. Move to old threads list.
                            old_thread = self.interpolation_thread
                            self.old_threads.append(old_thread)
                            # Connect cleanup when finished (ignoring result)
                            old_thread.finished.connect(
                                lambda _: self._cleanup_thread(old_thread)
                            )
                            old_thread.error.connect(
                                lambda _: self._cleanup_thread(old_thread)
                            )
                            self.interpolation_thread = None

                        # Increment generation ID for this request
                        self.current_generation_id += 1
                        generation_id = self.current_generation_id

                        # Start background interpolation with generation tracking
                        self.interpolation_thread = InterpolationThread(
                            points, color_indices, x_grid, y_grid, generation_id
                        )
                        self.interpolation_thread.quick_result.connect(
                            lambda quick_interp,
                            gen_id=generation_id: self._on_quick_interpolation_result(
                                quick_interp,
                                x_min,
                                x_max,
                                y_min,
                                y_max,
                                num_qoi,
                                qoi_names,
                                current_hash,
                                gen_id,
                            )
                        )
                        self.interpolation_thread.finished.connect(
                            lambda interpolated,
                            gen_id=generation_id: self._on_interpolation_finished(
                                interpolated,
                                x_min,
                                x_max,
                                y_min,
                                y_max,
                                num_qoi,
                                qoi_names,
                                current_hash,
                                gen_id,
                            )
                        )
                        self.interpolation_thread.error.connect(
                            lambda err: self._on_interpolation_error(err)
                        )
                        self.interpolation_thread.start()

                    except Exception:
                        logger.exception("Interpolation failed")
                else:
                    # --- FAST VECTORIZED POINTS MODE ---
                    n_points = len(x_data)

                    # 1. Create array of RGBA values (Default: Red for bad points)
                    # Shape: (N, 4), type: uint8 (0-255)
                    colors = np.zeros((n_points, 4), dtype=np.uint8)
                    colors[:] = [255, 0, 0, 255]  # Default red for all

                    # 2. Set Green for Good points (overwrite default)
                    # Safe because is_good is guaranteed 1D
                    if is_good is not None and np.any(is_good):
                        colors[is_good] = [0, 170, 0, 255]

                    # 3. Handle Violation Colors (Vectorized)
                    # Only process bad points that have a violation index
                    if violation_idx is not None and is_good is not None:
                        # Safety: ensure arrays are same length
                        if len(violation_idx) != len(is_good):
                            logger.warning(
                                f"Array length mismatch: violation_idx={len(violation_idx)}, is_good={len(is_good)}"
                            )
                        else:
                            # Get mask of bad points
                            bad_mask = ~is_good

                            # Only proceed if there are bad points
                            if np.any(bad_mask):
                                # Extract violation indices for bad points only
                                bad_violation_idx = violation_idx[bad_mask]

                                # Filter out NaN/infinite values
                                valid_mask = np.isfinite(bad_violation_idx)

                                if np.any(valid_mask):
                                    # Get valid violation indices and clamp to range [0, num_qoi)
                                    valid_violations = bad_violation_idx[
                                        valid_mask
                                    ].astype(int)

                                    # Safety: ensure num_qoi > 0 to avoid division by zero
                                    if num_qoi > 0:
                                        valid_violations = valid_violations % num_qoi

                                        # Get the indices in the original array where we need to set colors
                                        bad_indices = np.where(bad_mask)[0]
                                        valid_bad_indices = bad_indices[valid_mask]

                                        # Loop through constraints (small loop) instead of points (large loop)
                                        for v_idx_val in range(num_qoi):
                                            # Find all points that violated this specific constraint
                                            constraint_mask = (
                                                valid_violations == v_idx_val
                                            )

                                            if np.any(constraint_mask):
                                                # Get the color for this constraint
                                                q_name = qoi_names[v_idx_val]
                                                hex_color = (
                                                    self.parent_widget.qoi_colors.get(
                                                        q_name, "#ff0000"
                                                    )
                                                )

                                                # Convert hex to RGB
                                                c = QtGui.QColor(hex_color)
                                                rgb = np.array(
                                                    [c.red(), c.green(), c.blue(), 255],
                                                    dtype=np.uint8,
                                                )

                                                # Apply color to all points that violated this constraint
                                                colors[
                                                    valid_bad_indices[constraint_mask]
                                                ] = rgb

                    # 4. Group points by color and create batched ScatterPlotItems for performance
                    # Instead of creating thousands of individual brushes, group points by color
                    unique_colors, inverse_indices = np.unique(
                        colors, axis=0, return_inverse=True
                    )

                    # [CRITICAL FIX]: Ensure inverse_indices is 1D.
                    # If np.unique returns a column vector (N, 1), this flattening fixes it.
                    inverse_indices = inverse_indices.ravel()

                    # Create a ScatterPlotItem for each unique color
                    for color_idx, unique_color in enumerate(unique_colors):
                        # Get mask for points with this color
                        # Now guaranteed to be 1D because inverse_indices is 1D
                        color_mask = inverse_indices == color_idx

                        if np.any(color_mask):
                            # Extract points for this color
                            # [SAFE]: x_data is 1D, color_mask is 1D.
                            # We double-check x_data dimensionality to be absolutely safe
                            if x_data.ndim > 1:
                                x_data = x_data.ravel()
                            if y_data.ndim > 1:
                                y_data = y_data.ravel()

                            color_x = x_data[color_mask]
                            color_y = y_data[color_mask]

                            # Create single brush for this color group
                            brush = pg.mkBrush(
                                QtGui.QColor(
                                    int(unique_color[0]),
                                    int(unique_color[1]),
                                    int(unique_color[2]),
                                    int(unique_color[3]),
                                )
                            )

                            # Create ScatterPlotItem for this color group
                            scatter_item = pg.ScatterPlotItem(
                                x=color_x,
                                y=color_y,
                                pen=pg.mkPen("w", width=0.5),  # White border
                                brush=brush,
                                size=6,
                            )
                            scatter_item.setZValue(1)
                            self.plot_widget.addItem(scatter_item)

                            # Store reference to avoid garbage collection
                            if not hasattr(self, "color_scatter_items"):
                                self.color_scatter_items = []
                            self.color_scatter_items.append(scatter_item)

        # Draw Optimal Point (if objectives are included and we have the optimized point)
        if (
            hasattr(self.parent_widget, "optimal_point")
            and self.parent_widget.optimal_point is not None
        ):
            optimal_pt = self.parent_widget.optimal_point
            # Check if we can plot this point on this plot
            opt_x = None
            opt_y = None

            # Get optimal point value for x-axis
            if x_name in self.parent_widget.inputs:
                x_idx = self.parent_widget.inputs.index(x_name)
                if x_idx < len(optimal_pt):
                    opt_x = optimal_pt[x_idx]
            elif qoi_values is not None and x_name in [
                q["name"] for q in self.parent_widget.problem.quantities_of_interest
            ]:
                # For QoI on x-axis, we'd need to evaluate - skip for now
                pass

            # Get optimal point value for y-axis
            if y_name in self.parent_widget.inputs:
                y_idx = self.parent_widget.inputs.index(y_name)
                if y_idx < len(optimal_pt):
                    opt_y = optimal_pt[y_idx]
            elif qoi_values is not None and y_name in [
                q["name"] for q in self.parent_widget.problem.quantities_of_interest
            ]:
                # For QoI on y-axis, we'd need to evaluate - skip for now
                pass

            # Plot star marker if we have both coordinates
            if opt_x is not None and opt_y is not None:
                self.scatter_optimal = pg.ScatterPlotItem(
                    x=[opt_x],
                    y=[opt_y],
                    pen=pg.mkPen("k", width=2),
                    brush=pg.mkBrush(255, 215, 0),  # Gold color
                    size=20,
                    symbol="star",
                )
                self.scatter_optimal.setZValue(100)  # Ensure it's on top
                self.plot_widget.addItem(self.scatter_optimal)

        # Match the MMSS result views: one selected mode uses the standard
        # black ROI, while All boxes and D_MMSS use read-only coloured overlays.
        has_multimodal_result = bool(
            getattr(self.parent_widget, "multi_modal_boxes", [])
        )
        multimodal_view_mode = getattr(
            self.parent_widget, "multimodal_view_mode", "all"
        )
        draw_multimodal_overlays = has_multimodal_result and multimodal_view_mode in (
            "all",
            "recommended",
        )
        if dv_par_box_copy is not None and not draw_multimodal_overlays:
            dvs = [dv["name"] for dv in self.parent_widget.problem.design_variables]
            if x_name in dvs and y_name in dvs:
                x_idx = dvs.index(x_name)
                y_idx = dvs.index(y_name)
                bx_min, bx_max = dv_par_box_copy[x_idx, 0], dv_par_box_copy[x_idx, 1]
                by_min, by_max = dv_par_box_copy[y_idx, 0], dv_par_box_copy[y_idx, 1]

                # Helper for scalar conversion
                def _s(v):
                    if hasattr(v, "item"):
                        if v.size > 1:
                            return float(v.flatten()[0])
                        return float(v.item())
                    if hasattr(v, "__len__") and not isinstance(v, str):
                        return float(v[0])
                    return float(v)

                if self.roi_item is None:
                    pen_style = QtCore.Qt.SolidLine
                    center_slice = bool(
                        hasattr(self.parent_widget, "chk_center_slice")
                        and self.parent_widget.chk_center_slice.isChecked()
                    )
                    if center_slice:
                        pen_style = QtCore.Qt.DashLine

                    # Create draggable ROI rectangle
                    roi = pg.ROI(
                        [bx_min, by_min],
                        [bx_max - bx_min, by_max - by_min],
                        pen=pg.mkPen("black", width=2, style=pen_style),
                        rotatable=False,
                    )
                    roi.maxBounds = QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )
                    roi.addScaleHandle([1, 1], [0, 0])  # Bottom-right
                    roi.addScaleHandle([0, 0], [1, 1])  # Top-left
                    roi.addScaleHandle([1, 0], [0, 1])  # Bottom-left
                    roi.addScaleHandle([0, 1], [1, 0])  # Top-right

                    roi.sigRegionChanged.connect(lambda: self.on_box_moved(roi))
                    roi.setZValue(10)  # Ensure on top
                    self.plot_widget.addItem(roi)
                    self.roi_item = roi

                    if center_slice:
                        center_item = pg.ScatterPlotItem(
                            x=[(bx_min + bx_max) / 2.0],
                            y=[(by_min + by_max) / 2.0],
                            pen=pg.mkPen("black", width=2),
                            brush=pg.mkBrush("black"),
                            size=12,
                            symbol="+",
                        )
                        center_item.setZValue(11)
                        self.plot_widget.addItem(center_item)
                else:
                    self.roi_item.maxBounds = QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )

                    # but check threshold to avoid jitter during drag.
                    current_pos = self.roi_item.pos()
                    current_size = self.roi_item.size()
                    new_pos = QtCore.QPointF(bx_min, by_min)
                    new_size = QtCore.QPointF(bx_max - bx_min, by_max - by_min)

                    if (
                        abs(current_pos.x() - new_pos.x()) > 1e-6
                        or abs(current_pos.y() - new_pos.y()) > 1e-6
                        or abs(current_size.x() - new_size.x()) > 1e-6
                        or abs(current_size.y() - new_size.y()) > 1e-6
                    ):
                        self.roi_item.blockSignals(True)
                        self.roi_item.setPos(new_pos)
                        self.roi_item.setSize(new_size)
                        self.roi_item.blockSignals(False)

                # Update dotted lines connecting box to axes
                for line in self.roi_lines:
                    self.plot_widget.removeItem(line)
                self.roi_lines = []

                # Vertical lines from box to x-axis
                vline_left = pg.InfiniteLine(
                    pos=bx_min,
                    angle=90,
                    pen=pg.mkPen("black", style=QtCore.Qt.DashLine, width=1),
                )
                vline_right = pg.InfiniteLine(
                    pos=bx_max,
                    angle=90,
                    pen=pg.mkPen("black", style=QtCore.Qt.DashLine, width=1),
                )
                # Horizontal lines from box to y-axis
                hline_bottom = pg.InfiniteLine(
                    pos=by_min,
                    angle=0,
                    pen=pg.mkPen("black", style=QtCore.Qt.DashLine, width=1),
                )
                hline_top = pg.InfiniteLine(
                    pos=by_max,
                    angle=0,
                    pen=pg.mkPen("black", style=QtCore.Qt.DashLine, width=1),
                )

                # Set Z-Value higher than points (points are usually 1 or 0)
                vline_left.setZValue(2)
                vline_right.setZValue(2)
                hline_bottom.setZValue(2)
                hline_top.setZValue(2)

                self.plot_widget.addItem(vline_left)
                self.plot_widget.addItem(vline_right)
                self.plot_widget.addItem(hline_bottom)
                self.plot_widget.addItem(hline_top)
                self.roi_lines = [vline_left, vline_right, hline_bottom, hline_top]

        if draw_multimodal_overlays:
            dvs = [dv["name"] for dv in self.parent_widget.problem.design_variables]
            if x_name in dvs and y_name in dvs:
                x_idx = dvs.index(x_name)
                y_idx = dvs.index(y_name)
                for mode_index, box in enumerate(
                    self.parent_widget._get_multimodal_display_boxes()
                ):
                    bx_min, bx_max = box.bounds[x_idx]
                    by_min, by_max = box.bounds[y_idx]
                    width = max(0.0, float(bx_max - bx_min))
                    height = max(0.0, float(by_max - by_min))
                    color = QtGui.QColor(
                        self.parent_widget._get_branch_color(mode_index, box)
                    )
                    fill = QtGui.QColor(color)
                    fill.setAlpha(40)
                    fill_item = QtWidgets.QGraphicsRectItem(
                        float(bx_min), float(by_min), width, height
                    )
                    fill_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    fill_item.setBrush(QtGui.QBrush(fill))
                    fill_item.setZValue(0)
                    self.plot_widget.addItem(fill_item)

                    pen_style = QtCore.Qt.SolidLine
                    center_slice = bool(
                        hasattr(self.parent_widget, "chk_center_slice")
                        and self.parent_widget.chk_center_slice.isChecked()
                    )
                    if center_slice:
                        pen_style = QtCore.Qt.DashLine

                    border_item = pg.ROI(
                        [float(bx_min), float(by_min)],
                        [width, height],
                        pen=pg.mkPen(color, width=3, style=pen_style),
                        movable=False,
                        rotatable=False,
                    )
                    border_item.setZValue(100)
                    self.plot_widget.addItem(border_item)

                    if center_slice:
                        center_item = pg.ScatterPlotItem(
                            x=[(float(bx_min) + float(bx_max)) / 2.0],
                            y=[(float(by_min) + float(by_max)) / 2.0],
                            pen=pg.mkPen(color, width=2),
                            brush=pg.mkBrush(color),
                            size=12,
                            symbol="+",
                        )
                        center_item.setZValue(101)
                        self.plot_widget.addItem(center_item)

        # Set axis labels
        x_unit = (
            self.parent_widget.input_units.get(x_name)
            or self.parent_widget.output_units.get(x_name)
            or "-"
        )
        y_unit = (
            self.parent_widget.input_units.get(y_name)
            or self.parent_widget.output_units.get(y_name)
            or "-"
        )
        x_label_text = format_html(x_name)
        y_label_text = format_html(y_name)
        x_label = f"{x_label_text} ({x_unit})"
        y_label = f"{y_label_text} ({y_unit})"

        self.plot_item.setLabel("bottom", x_label)
        self.plot_item.setLabel("left", y_label)
        self.plot_item.setTitle(f"{x_label_text} vs {y_label_text}")

        # X-Axis Requirements (Vertical Lines)
        qois = [q["name"] for q in self.parent_widget.problem.quantities_of_interest]
        if x_name in qois:
            # Find row in qoi_table
            for i in range(self.parent_widget.qoi_table.rowCount()):
                if self.parent_widget.qoi_table.item(i, 0).text() == x_name:
                    try:
                        l_val = float(self.parent_widget.qoi_table.item(i, 2).text())
                        u_val = float(self.parent_widget.qoi_table.item(i, 3).text())

                        if l_val > -1e8:  # Arbitrary large number check
                            l_line = pg.InfiniteLine(
                                pos=l_val,
                                angle=90,
                                pen=pg.mkPen(
                                    "red", style=QtCore.Qt.DashLine, alpha=0.5
                                ),
                            )
                            l_line.setZValue(2)
                            self.plot_widget.addItem(l_line)
                            self.limit_lines.append(l_line)
                        if u_val < 1e8:
                            u_line = pg.InfiniteLine(
                                pos=u_val,
                                angle=90,
                                pen=pg.mkPen(
                                    "red", style=QtCore.Qt.DashLine, alpha=0.5
                                ),
                            )
                            u_line.setZValue(2)
                            self.plot_widget.addItem(u_line)
                            self.limit_lines.append(u_line)
                    except (AttributeError, TypeError, ValueError):
                        logger.debug(
                            "Could not draw the x-axis requirement limits.",
                            exc_info=True,
                        )
                    break

        # Y-Axis Requirements (Horizontal Lines)
        if y_name in qois:
            for i in range(self.parent_widget.qoi_table.rowCount()):
                if self.parent_widget.qoi_table.item(i, 0).text() == y_name:
                    try:
                        l_val = float(self.parent_widget.qoi_table.item(i, 2).text())
                        u_val = float(self.parent_widget.qoi_table.item(i, 3).text())

                        if l_val > -1e8:
                            l_line = pg.InfiniteLine(
                                pos=l_val,
                                angle=0,
                                pen=pg.mkPen(
                                    "red", style=QtCore.Qt.DashLine, alpha=0.5
                                ),
                            )
                            l_line.setZValue(2)
                            self.plot_widget.addItem(l_line)
                            self.limit_lines.append(l_line)
                        if u_val < 1e8:
                            u_line = pg.InfiniteLine(
                                pos=u_val,
                                angle=0,
                                pen=pg.mkPen(
                                    "red", style=QtCore.Qt.DashLine, alpha=0.5
                                ),
                            )
                            u_line.setZValue(2)
                            self.plot_widget.addItem(u_line)
                            self.limit_lines.append(u_line)
                    except (AttributeError, TypeError, ValueError):
                        logger.debug(
                            "Could not draw the y-axis requirement limits.",
                            exc_info=True,
                        )
                    break

        # Force final update
        self.plot_widget.update()
        self.plotting = False

    def _cleanup_thread(self, thread):
        """Clean up finished background threads."""
        if thread in self.old_threads:
            self.old_threads.remove(thread)
        thread.deleteLater()

    def _on_interpolation_finished(
        self,
        interpolated,
        x_min,
        x_max,
        y_min,
        y_max,
        num_qoi,
        qoi_names,
        current_hash,
        generation_id,
    ):
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
                    q_name = qoi_names[v_idx] if v_idx < len(qoi_names) else "unknown"
                    color = self.parent_widget.qoi_colors.get(q_name, "red")
                    # Convert color name to RGB
                    color_rgb = pg.mkColor(color)
                    color_map[violation_mask] = [
                        color_rgb.red(),
                        color_rgb.green(),
                        color_rgb.blue(),
                        180,
                    ]

            # Unknown violations
            unknown_mask = interpolated == (num_qoi + 1)
            color_map[unknown_mask] = [255, 0, 0, 180]  # Semi-transparent red

            # Helper for scalar conversion
            def _s(v):
                if hasattr(v, "item"):
                    if v.size > 1:
                        return float(v.flatten()[0])
                    return float(v.item())
                if hasattr(v, "__len__") and not isinstance(v, str):
                    return float(v[0])
                return float(v)

            # Update or create image item (avoid flicker by reusing existing item)
            if hasattr(self, "img_item") and self.img_item is not None:
                # Update existing image data to avoid flicker
                self.img_item.setImage(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(
                    QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )
                )
            else:
                # Create new image item
                self.img_item = pg.ImageItem(
                    color_map.astype(np.uint8).transpose(1, 0, 2)
                )
                self.img_item.setRect(
                    QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )
                )
                self.img_item.setZValue(0)
                self.plot_widget.addItem(self.img_item)
        except RuntimeError as e:
            # Ignore errors resulting from deleted C++ objects
            if "Internal C++ object" not in str(e):
                logger.debug("RuntimeError in interpolation finished", exc_info=True)
        except Exception:
            logger.exception("Error in interpolation finished")
            # Cache the result
            self.cached_categorical_img = color_map.astype(np.uint8).transpose(1, 0, 2)
            self.cached_data_hash = current_hash

        except Exception:
            logger.exception("Error creating interpolated image")
        finally:
            self.interpolation_thread = None

    def _on_quick_interpolation_result(
        self,
        quick_interp,
        x_min,
        x_max,
        y_min,
        y_max,
        num_qoi,
        qoi_names,
        current_hash,
        generation_id,
    ):
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
                    q_name = qoi_names[v_idx] if v_idx < len(qoi_names) else "unknown"
                    color = self.parent_widget.qoi_colors.get(q_name, "red")
                    # Convert color name to RGB
                    color_rgb = pg.mkColor(color)
                    color_map[violation_mask] = [
                        color_rgb.red(),
                        color_rgb.green(),
                        color_rgb.blue(),
                        120,
                    ]

            # Unknown violations
            unknown_mask = quick_interp == (num_qoi + 1)
            color_map[unknown_mask] = [
                255,
                0,
                0,
                120,
            ]  # More transparent for quick result

            # Helper for scalar conversion
            def _s(v):
                if hasattr(v, "item"):
                    if v.size > 1:
                        return float(v.flatten()[0])
                    return float(v.item())
                if hasattr(v, "__len__") and not isinstance(v, str):
                    return float(v[0])
                return float(v)

            # Update or create image item for quick result (avoid flicker)
            if hasattr(self, "img_item") and self.img_item is not None:
                # Update existing image data with quick result
                self.img_item.setImage(color_map.astype(np.uint8).transpose(1, 0, 2))
                self.img_item.setRect(
                    QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )
                )
            else:
                # Create new image item if none exists
                self.img_item = pg.ImageItem(
                    color_map.astype(np.uint8).transpose(1, 0, 2)
                )
                self.img_item.setRect(
                    QtCore.QRectF(
                        _s(x_min), _s(y_min), _s(x_max - x_min), _s(y_max - y_min)
                    )
                )
                self.img_item.setZValue(0)
                self.plot_widget.addItem(self.img_item)

        except RuntimeError as e:
            # Ignore errors resulting from deleted C++ objects
            if "Internal C++ object" not in str(e):
                logger.debug("RuntimeError in quick interpolation", exc_info=True)
        except Exception:
            logger.exception("Error creating quick interpolated image")

    def _on_interpolation_error(self, error_msg):
        """Handle interpolation thread errors."""
        logger.error("Interpolation failed: %s", error_msg)
        self.interpolation_thread = None
