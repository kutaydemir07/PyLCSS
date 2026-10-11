# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""1-D and 3-D solution-space plots.

Both follow the contract the 2-D ``PlotWidget`` has with ``SolutionSpaceWidget``:
per-plot ``samples``, ``plot()``, ``update_roi_visuals()`` and an editable
solution-space box that writes back to ``dv_par_box``.  The 3-D view is projected
by hand into a pyqtgraph scene, so it needs no OpenGL and exports like any other plot.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.common.text_utils import format_html
from pylcss.user_interface.common.theme_manager import COLORS

from .plotting import PlotWidget as Plot2DWidget

logger = logging.getLogger(__name__)

__all__ = ["Plot1DWidget", "Plot3DWidget"]

_GOOD_RGBA = (0, 170, 0, 255)
_GHOST_QSS = (
    "QPushButton { background: transparent; color: #555555;"
    " border: 1px solid #c8ccd0; padding: 0 6px; border-radius: 3px;"
    " font-size: 10px; min-width: 0; }"
    " QPushButton:hover { background: #eef0f3; }"
    " QPushButton:checked { background: #d0d5dd; }"
)
_REMOVE_QSS = (
    "QPushButton { background: transparent; color: #666666; border: none;"
    " padding: 0; min-width: 0; font-size: 11px; }"
    " QPushButton:hover { color: #cf222e; }"
)


# --- helpers shared by both widgets -------------------------------------------------


def _axis_values(parent, samples, name):
    """Sampled values of an input or output, or None when unavailable."""
    if samples is None:
        return None
    if name in parent.inputs:
        return np.asarray(samples["points"])[parent.inputs.index(name), :]
    qoi_names = [q["name"] for q in parent.problem.quantities_of_interest]
    if name in qoi_names:
        return np.asarray(samples["qoi_values"])[qoi_names.index(name), :]
    return None


def _box_rows(parent, names):
    """Design-variable rows of ``names``; None unless all are design variables."""
    if parent.dv_par_box is None:
        return None
    dvs = [dv["name"] for dv in parent.problem.design_variables]
    if not all(name in dvs for name in names):
        return None
    return [dvs.index(name) for name in names]


def _read_box(parent, rows):
    parent.dv_par_box_mutex.lock()
    try:
        box = parent.dv_par_box.copy() if parent.dv_par_box is not None else None
    finally:
        parent.dv_par_box_mutex.unlock()
    if box is None:
        return None
    return [(float(box[r, 0]), float(box[r, 1])) for r in rows]


def _write_box(parent, rows, intervals):
    parent.dv_par_box_mutex.lock()
    try:
        for row, (low, high) in zip(rows, intervals):
            parent.dv_par_box[row, 0] = low
            parent.dv_par_box[row, 1] = high
    finally:
        parent.dv_par_box_mutex.unlock()


def _axis_range(parent, name, data, box_interval):
    """Axis extent: design space (or data range) widened to include the box."""
    low, high = 0.0, 1.0
    dsl = getattr(parent, "dsl", None)
    dsu = getattr(parent, "dsu", None)
    if name in parent.inputs and dsl is not None:
        idx = parent.inputs.index(name)
        if idx < len(dsl):
            low, high = float(dsl[idx]), float(dsu[idx])
            low = low if np.isfinite(low) else -1e6
            high = high if np.isfinite(high) else 1e6
    elif data is not None:
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
    if box_interval is not None:
        low, high = min(low, box_interval[0]), max(high, box_interval[1])
    if high <= low:
        high = low + 1.0
    return low, high


def _point_colors(parent, is_good, violation_idx):
    """RGBA per sample using the legend colours; hidden legend entries get alpha 0."""
    n = len(is_good)
    hidden = getattr(parent, "hidden_legend_keys", ())
    colors = np.empty((n, 4), dtype=np.uint8)
    colors[:] = (255, 0, 0, 255)
    good = np.asarray(is_good, dtype=bool)
    colors[good] = _GOOD_RGBA
    if "__good__" in hidden:
        colors[good, 3] = 0
    qois = parent.problem.quantities_of_interest
    if qois and violation_idx is not None and len(violation_idx) == n:
        violation = np.asarray(violation_idx, dtype=float)
        rows = np.flatnonzero(~good & np.isfinite(violation))
        which = violation[rows].astype(int) % len(qois)
        for k, qoi in enumerate(qois):
            selected = rows[which == k]
            if selected.size:
                color = QtGui.QColor(parent.qoi_colors.get(qoi["name"], "#ff0000"))
                alpha = 0 if qoi["name"] in hidden else 255
                colors[selected] = (color.red(), color.green(), color.blue(), alpha)
    return colors


def _requirement_limits(parent, name):
    """Finite (low, high) requirement limits of an output, else (None, None)."""
    table = parent.qoi_table
    for row in range(table.rowCount()):
        cell = table.item(row, 0)
        if cell is not None and cell.text() == name:
            try:
                low = float(table.item(row, 2).text())
                high = float(table.item(row, 3).text())
            except (AttributeError, TypeError, ValueError):
                return None, None
            return (low if low > -1e8 else None), (high if high < 1e8 else None)
    return None, None


def _multimodal_overlays(parent):
    """[(colour, bounds)] when read-only multi-modal boxes replace the editable box."""
    boxes = getattr(parent, "multi_modal_boxes", [])
    mode = getattr(parent, "multimodal_view_mode", "all")
    if not boxes or mode not in ("all", "recommended"):
        return []
    return [
        (parent._get_branch_color(index, box), box.bounds)
        for index, box in enumerate(parent._get_multimodal_display_boxes())
    ]


def _axis_label(parent, name):
    unit = parent.input_units.get(name) or parent.output_units.get(name) or "-"
    return f"{format_html(name)} ({unit})"


def _style_plot(plot_widget):
    plot_widget.setBackground(COLORS["chart_bg"])
    item = plot_widget.getPlotItem()
    for axis_name in ("left", "right", "top", "bottom"):
        axis = item.getAxis(axis_name)
        axis.setPen(COLORS["chart_fg"])
        axis.setTextPen(COLORS["chart_fg"])
    item.titleLabel.setAttr("color", COLORS["chart_fg"])
    return item


class _AxisPlotBase(QtWidgets.QWidget):
    """Header, bookkeeping attributes and box write-back shared by 1-D and 3-D plots."""

    save_plot = Plot2DWidget.save_plot

    def __init__(self, axis_names, parent=None):
        super().__init__(parent)
        names = tuple(axis_names)
        self.axis_names = names
        self.x_name = names[0]
        self.y_name = names[1] if len(names) > 1 else None
        self.z_name = names[2] if len(names) > 2 else None
        self.samples = None
        self.plotting = False
        self.parent_widget = None
        self.roi_item = None
        self.title_text = ""

        # The parent resets these on 2-D plots; they stay inert here.
        self.interpolation_thread = None
        self.old_threads = []
        self.cached_categorical_img = None
        self.cached_data_hash = None
        self.cached_bounds_hash = None

        self.box_update_timer = QtCore.QTimer(self)
        self.box_update_timer.setSingleShot(True)
        self.box_update_timer.timeout.connect(self._apply_box_update)

        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        header.addStretch()
        self.btn_zoom = QtWidgets.QPushButton("Zoom")
        self.btn_zoom.setFixedHeight(18)
        self.btn_zoom.setStyleSheet(_GHOST_QSS)
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setToolTip("Save this plot as an image")
        self.btn_save.setFixedHeight(18)
        self.btn_save.setStyleSheet(_GHOST_QSS)
        self.btn_save.clicked.connect(self.save_plot)
        self.btn_remove = QtWidgets.QPushButton("\u2715")
        self.btn_remove.setFixedSize(20, 18)
        self.btn_remove.setToolTip("Remove this plot")
        self.btn_remove.setStyleSheet(_REMOVE_QSS)
        for button in (self.btn_zoom, self.btn_save, self.btn_remove):
            header.addWidget(button)
        self._root.addLayout(header)

    # -- parent contract ---------------------------------------------------------

    def plot(self):
        if self.plotting:
            return
        parent = self.parent_widget
        if not parent or not parent.problem:
            return
        self.plotting = True
        try:
            self._draw()
        finally:
            self.plotting = False

    def _draw(self):
        raise NotImplementedError

    def update_roi_visuals(self):
        raise NotImplementedError

    def _apply_box_update(self):
        raise NotImplementedError

    def _commit_box(self, rows, intervals):
        """Write the edited box back and let every other plot follow it."""
        parent = self.parent_widget
        if not parent or parent.dv_par_box is None:
            return
        _write_box(parent, rows, intervals)
        parent.sync_plots_roi(self)
        for row in rows:
            parent.update_single_dv_row(row)
        parent.trigger_debounced_resample()

    def _center_slice(self):
        check = getattr(self.parent_widget, "chk_center_slice", None)
        return bool(check is not None and check.isChecked())


# --- 1-D ---------------------------------------------------------------------------


class Plot1DWidget(_AxisPlotBase):
    """One variable on a strip: sampled points and its (draggable) solution interval."""

    def __init__(self, name, parent=None):
        super().__init__((name,), parent)
        self.title_text = f"{format_html(name)} (1D)"
        self.plot_widget = pg.PlotWidget()
        self.plot_item = _style_plot(self.plot_widget)
        self.plot_item.hideAxis("left")
        self.plot_item.showGrid(x=True, y=False, alpha=0.3)
        self.plot_item.setTitle(self.title_text)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self._root.addWidget(self.plot_widget)

        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setToolTip("Toggle zoom/pan on this plot")
        self.btn_zoom.clicked.connect(self._toggle_zoom)
        self._pending = None

    def _toggle_zoom(self):
        self.plot_widget.setMouseEnabled(x=self.btn_zoom.isChecked(), y=False)

    def _draw(self):
        parent = self.parent_widget
        name = self.x_name
        self.plot_widget.clear()
        self.roi_item = None

        data = _axis_values(parent, self.samples, name)
        overlays = _multimodal_overlays(parent)
        rows = None if overlays else _box_rows(parent, (name,))
        box = _read_box(parent, rows) if rows else None
        low, high = _axis_range(parent, name, data, box[0] if box else None)

        self.plot_item.setXRange(low, high, padding=0)
        self.plot_item.setYRange(0, 1, padding=0)
        view_box = self.plot_widget.getViewBox()
        view_box.setLimits(xMin=low, xMax=high, yMin=0, yMax=1)
        view_box.enableAutoRange(enable=False)

        if data is not None:
            self._draw_points(np.asarray(data, dtype=float).ravel())

        index = _box_rows(parent, (name,))
        for color, bounds in overlays:
            if index is None:
                break
            lo, hi = (float(v) for v in bounds[index[0]])
            fill = QtGui.QColor(color)
            fill.setAlpha(40)
            region = pg.LinearRegionItem(
                values=(lo, hi),
                orientation="vertical",
                brush=pg.mkBrush(fill),
                pen=pg.mkPen(color, width=3),
                movable=False,
            )
            region.setZValue(5)
            self.plot_widget.addItem(region)

        if box:
            style = QtCore.Qt.DashLine if self._center_slice() else QtCore.Qt.SolidLine
            region = pg.LinearRegionItem(
                values=box[0],
                orientation="vertical",
                brush=pg.mkBrush(0, 0, 0, 25),
                pen=pg.mkPen("k", width=2, style=style),
                movable=True,
                bounds=(low, high),
            )
            region.setZValue(10)
            region.sigRegionChanged.connect(self._on_region_changed)
            self.plot_widget.addItem(region)
            self.roi_item = region

        if name in parent.outputs:
            req_low, req_high = _requirement_limits(parent, name)
            for limit in (req_low, req_high):
                if limit is not None:
                    line = pg.InfiniteLine(
                        pos=limit,
                        angle=90,
                        pen=pg.mkPen("red", style=QtCore.Qt.DashLine, alpha=0.5),
                    )
                    line.setZValue(2)
                    self.plot_widget.addItem(line)

        self.plot_item.setLabel("bottom", _axis_label(parent, name))

    def _draw_points(self, values):
        samples = self.samples
        is_good = np.asarray(samples["is_good"]).ravel().astype(bool)
        violation = np.asarray(samples["violation_idx"]).ravel()
        if len(is_good) != len(values):
            return
        colors = _point_colors(self.parent_widget, is_good, violation)
        keep = np.isfinite(values) & (colors[:, 3] > 0)
        values, colors = values[keep], colors[keep]
        if not values.size:
            return
        # Vertical position carries no meaning; a fixed jitter just separates points.
        jitter = np.random.default_rng(7).uniform(0.06, 0.94, size=len(values))
        unique, inverse = np.unique(colors, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        for index, rgba in enumerate(unique):
            mask = inverse == index
            scatter = pg.ScatterPlotItem(
                x=values[mask],
                y=jitter[mask],
                pen=pg.mkPen("w", width=0.5),
                brush=pg.mkBrush(QtGui.QColor(*(int(c) for c in rgba))),
                size=6,
            )
            scatter.setZValue(2 if tuple(rgba) == _GOOD_RGBA else 1)
            self.plot_widget.addItem(scatter)

    def _on_region_changed(self):
        self.box_update_timer.start(100)

    def _apply_box_update(self):
        parent = self.parent_widget
        if self.roi_item is None or not parent:
            return
        rows = _box_rows(parent, self.axis_names)
        if rows:
            low, high = self.roi_item.getRegion()
            self._commit_box(rows, [(float(low), float(high))])

    def update_roi_visuals(self):
        parent = self.parent_widget
        if self.roi_item is None or not parent:
            return
        rows = _box_rows(parent, self.axis_names)
        box = _read_box(parent, rows) if rows else None
        if box:
            self.roi_item.blockSignals(True)
            self.roi_item.setRegion(box[0])
            self.roi_item.blockSignals(False)


# --- 3-D ---------------------------------------------------------------------------

_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)
_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(_CORNERS[i] != _CORNERS[j]) == 1
]
_EDGE_INDEX = [index for edge in _EDGES for index in edge]


def _cuboid_edges(low, high):
    """Endpoints (24, 3) of the 12 edges of the cuboid spanned by low/high."""
    corners = low + (_CORNERS + 1.0) / 2.0 * (high - low)
    return corners[_EDGE_INDEX]


class _Scene3DViewBox(pg.ViewBox):
    """Left-drag rotates (or drags a box handle); the wheel zooms."""

    def __init__(self, owner):
        super().__init__(enableMenu=False)
        self._owner = owner
        self._handle = None
        self.setMouseEnabled(False, False)

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if ev.isStart():
            self._handle = self._owner.hit_handle(ev.buttonDownScenePos())
        if self._handle is not None:
            now = self.mapSceneToView(ev.scenePos())
            before = self.mapSceneToView(ev.lastScenePos())
            self._owner.drag_handle(self._handle, now.x() - before.x(), now.y() - before.y())
        else:
            delta = ev.scenePos() - ev.lastScenePos()
            self._owner.rotate_view(delta.x(), delta.y())
        if ev.isFinish():
            self._handle = None

    def wheelEvent(self, ev, axis=None):
        ev.accept()
        self._owner.zoom_view(ev.delta())

    def mouseClickEvent(self, ev):
        ev.accept()


class Plot3DWidget(_AxisPlotBase):
    """Three variables in a rotatable cube with an editable solution-space cuboid."""

    _HANDLE_RADIUS_PX = 12.0

    def __init__(self, axis_names, parent=None):
        super().__init__(axis_names, parent)
        self.title_text = (
            f"{format_html(self.z_name)} vs {format_html(self.y_name)}"
            f" vs {format_html(self.x_name)}"
        )
        self._view_box = _Scene3DViewBox(self)
        self.plot_widget = pg.PlotWidget(viewBox=self._view_box)
        self.plot_item = _style_plot(self.plot_widget)
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plot_item.setTitle(self.title_text)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setToolTip(
            "Drag to rotate, scroll to zoom, drag the square handles to resize the box."
        )
        self._view_box.setAspectLocked(True)
        self._view_box.setRange(xRange=(-2.3, 2.3), yRange=(-2.3, 2.3), padding=0)
        self._view_box.disableAutoRange()
        self._root.addWidget(self.plot_widget)

        self.btn_zoom.setText("Reset view")
        self.btn_zoom.setToolTip("Restore the default viewing angle and zoom")
        self.btn_zoom.clicked.connect(self._reset_view)

        self._azimuth = -55.0
        self._elevation = 22.0
        self._zoom = 1.0
        self._ranges = [(0.0, 1.0)] * 3
        self._rows = None
        self._box = None
        self._overlays = []
        self._norm_points = None
        self._brushes = []
        self._color_index = None
        self._items_ready = False
        self._handle_ids = []
        self._handle_xy = []

    # -- view controls -----------------------------------------------------------

    def _reset_view(self):
        self._azimuth, self._elevation, self._zoom = -55.0, 22.0, 1.0
        self._project()

    def rotate_view(self, dx, dy):
        self._azimuth -= dx * 0.6
        self._elevation = float(np.clip(self._elevation + dy * 0.6, -89.0, 89.0))
        self._project()

    def zoom_view(self, delta):
        self._zoom = float(np.clip(self._zoom * (1.0015 ** delta), 0.4, 3.0))
        self._project()

    def _project_points(self, points):
        """Orthographic projection; returns screen x, screen y and depth (larger = farther)."""
        azimuth, elevation = np.radians(self._azimuth), np.radians(self._elevation)
        ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elevation), np.sin(elevation)
        x1 = points[:, 0] * ca - points[:, 1] * sa
        y1 = points[:, 0] * sa + points[:, 1] * ca
        z1 = points[:, 2]
        return (
            x1 * self._zoom,
            (z1 * ce + y1 * se) * self._zoom,
            y1 * ce - z1 * se,
        )

    # -- drawing -----------------------------------------------------------------

    def _normalise(self, values, axis):
        low, high = self._ranges[axis]
        return 2.0 * (np.asarray(values, dtype=float) - low) / (high - low) - 1.0

    def _normalise_box(self, intervals):
        low = np.array([self._normalise(iv[0], a) for a, iv in enumerate(intervals)])
        high = np.array([self._normalise(iv[1], a) for a, iv in enumerate(intervals)])
        return low, high

    def _draw(self):
        parent = self.parent_widget
        names = self.axis_names
        self.plot_widget.clear()
        self.roi_item = None
        self._items_ready = False

        data = [_axis_values(parent, self.samples, name) for name in names]
        overlays = _multimodal_overlays(parent)
        self._rows = None if overlays else _box_rows(parent, names)
        self._box = _read_box(parent, self._rows) if self._rows else None
        self._overlays = []
        for color, bounds in overlays:
            rows = _box_rows(parent, names)
            if rows:
                self._overlays.append(
                    (color, [tuple(float(v) for v in bounds[r]) for r in rows])
                )

        ranges = []
        for axis, name in enumerate(names):
            interval = self._box[axis] if self._box else None
            ranges.append(_axis_range(parent, name, data[axis], interval))
        self._ranges = ranges

        self._norm_points = None
        if self.samples is not None and all(d is not None for d in data):
            self._prepare_points([np.asarray(d, dtype=float).ravel() for d in data])

        self._frame_curve = pg.PlotCurveItem(pen=pg.mkPen("#b0b0b0", width=1))
        self._frame_curve.setZValue(0)
        self._scatter = pg.ScatterPlotItem()
        self._scatter.setZValue(1)
        self._box_curve = pg.PlotCurveItem()
        self._box_curve.setZValue(5)
        self._handles = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=1), brush=pg.mkBrush("w"), symbol="s", size=9
        )
        self._handles.setZValue(6)
        self._overlay_curves = [
            pg.PlotCurveItem(pen=pg.mkPen(color, width=3)) for color, _ in self._overlays
        ]
        for item in (self._frame_curve, self._scatter, self._box_curve, self._handles):
            self.plot_widget.addItem(item)
        for curve in self._overlay_curves:
            curve.setZValue(5)
            self.plot_widget.addItem(curve)
        if self._box:
            style = QtCore.Qt.DashLine if self._center_slice() else QtCore.Qt.SolidLine
            self._box_curve.setPen(pg.mkPen("k", width=2, style=style))

        self._name_labels = []
        self._tick_labels = []
        for axis, name in enumerate(names):
            label = pg.TextItem(
                html=f"<span style='color:{COLORS['chart_fg']};'>{_axis_label(parent, name)}</span>",
                anchor=(0.5, 0.5),
            )
            label.setZValue(7)
            self.plot_widget.addItem(label)
            self._name_labels.append(label)
            ticks = []
            for _ in range(2):
                tick = pg.TextItem(color=COLORS["chart_fg"], anchor=(0.5, 0.5))
                tick.setFont(QtGui.QFont("Segoe UI", 7))
                tick.setZValue(7)
                self.plot_widget.addItem(tick)
                ticks.append(tick)
            self._tick_labels.append(ticks)

        self._items_ready = True
        self._project()

    def _prepare_points(self, arrays):
        is_good = np.asarray(self.samples["is_good"]).ravel().astype(bool)
        violation = np.asarray(self.samples["violation_idx"]).ravel()
        if len(is_good) != len(arrays[0]):
            return
        colors = _point_colors(self.parent_widget, is_good, violation)
        finite = np.all([np.isfinite(a) for a in arrays], axis=0)
        keep = finite & (colors[:, 3] > 0)
        if not np.any(keep):
            return
        columns = [self._normalise(a[keep], axis) for axis, a in enumerate(arrays)]
        self._norm_points = np.column_stack(columns)
        unique, inverse = np.unique(colors[keep], axis=0, return_inverse=True)
        self._color_index = inverse.ravel()
        self._brushes = [
            pg.mkBrush(QtGui.QColor(*(int(c) for c in rgba))) for rgba in unique
        ]

    def _axis_edge(self, axis):
        """The cube edge along ``axis`` that sits nearest the bottom (left for z) of the view."""
        others = [a for a in range(3) if a != axis]
        best, best_key = None, None
        for s_first in (-1.0, 1.0):
            for s_second in (-1.0, 1.0):
                start = np.zeros(3)
                start[axis], start[others[0]], start[others[1]] = -1.0, s_first, s_second
                end = start.copy()
                end[axis] = 1.0
                sx, sy, _ = self._project_points(((start + end) / 2.0)[None, :])
                key = sx[0] if axis == 2 else sy[0]
                if best_key is None or key < best_key:
                    best, best_key = (start, end), key
        return best

    def _project(self):
        if not self._items_ready:
            return
        edges = _cuboid_edges(np.full(3, -1.0), np.full(3, 1.0))
        fx, fy, _ = self._project_points(edges)
        self._frame_curve.setData(x=fx, y=fy, connect="pairs")

        if self._norm_points is not None:
            sx, sy, depth = self._project_points(self._norm_points)
            order = np.argsort(-depth)  # far points first so near ones overlap them
            sizes = np.clip(6.0 - 2.0 * depth[order] / 1.73, 4.0, 8.0)
            self._scatter.setData(
                x=sx[order],
                y=sy[order],
                brush=[self._brushes[i] for i in self._color_index[order]],
                pen=pg.mkPen("w", width=0.5),
                size=sizes,
            )
        else:
            self._scatter.setData(x=[], y=[])

        self._project_box()
        for curve, (_, intervals) in zip(self._overlay_curves, self._overlays):
            low, high = self._normalise_box(intervals)
            ox, oy, _ = self._project_points(_cuboid_edges(low, high))
            curve.setData(x=ox, y=oy, connect="pairs")
        self._project_labels()

    def _project_box(self):
        if not self._box:
            self._box_curve.setData(x=[], y=[])
            self._handles.setData(x=[], y=[])
            self._handle_ids, self._handle_xy = [], []
            return
        low, high = self._normalise_box(self._box)
        bx, by, _ = self._project_points(_cuboid_edges(low, high))
        self._box_curve.setData(x=bx, y=by, connect="pairs")
        centre = (low + high) / 2.0
        points, ids = [], []
        for axis in range(3):
            for side, bound in enumerate((low[axis], high[axis])):
                point = centre.copy()
                point[axis] = bound
                points.append(point)
                ids.append((axis, side))
        hx, hy, _ = self._project_points(np.array(points))
        self._handles.setData(x=hx, y=hy)
        self._handle_ids = ids
        self._handle_xy = list(zip(hx.tolist(), hy.tolist()))

    def _project_labels(self):
        for axis in range(3):
            start, end = self._axis_edge(axis)
            pts = np.array([start, (start + end) / 2.0, end])
            sx, sy, _ = self._project_points(pts)
            mid = np.array([sx[1], sy[1]])
            norm = np.linalg.norm(mid)
            outward = mid / norm if norm > 1e-6 else np.array([0.0, -1.0])
            name_pos = mid + outward * 0.3
            self._name_labels[axis].setPos(float(name_pos[0]), float(name_pos[1]))
            low, high = self._ranges[axis]
            for tick, index, value in zip(self._tick_labels[axis], (0, 2), (low, high)):
                pos = np.array([sx[index], sy[index]]) + outward * 0.14
                tick.setText(f"{value:.4g}")
                tick.setPos(float(pos[0]), float(pos[1]))

    # -- box editing -------------------------------------------------------------

    def hit_handle(self, scene_pos):
        """Handle id (axis, side) under ``scene_pos`` or None."""
        best, best_distance = None, self._HANDLE_RADIUS_PX
        for handle, (x, y) in zip(self._handle_ids, self._handle_xy):
            target = self._view_box.mapViewToScene(QtCore.QPointF(x, y))
            distance = np.hypot(target.x() - scene_pos.x(), target.y() - scene_pos.y())
            if distance < best_distance:
                best, best_distance = handle, distance
        return best

    def drag_handle(self, handle, dx, dy):
        """Move one cuboid face along its axis by the screen-space drag (dx, dy)."""
        if not self._box:
            return
        axis, side = handle
        unit = np.zeros((1, 3))
        unit[0, axis] = 1.0
        ux, uy, _ = self._project_points(unit)
        ux, uy = float(ux[0]), float(uy[0])
        denominator = ux * ux + uy * uy
        if denominator < 1e-6:
            return  # axis points straight at the viewer
        low, high = self._ranges[axis]
        step = (dx * ux + dy * uy) / denominator * (high - low) / 2.0
        gap = (high - low) * 1e-3
        current_low, current_high = self._box[axis]
        if side == 0:
            current_low = min(max(current_low + step, low), current_high - gap)
        else:
            current_high = max(min(current_high + step, high), current_low + gap)
        self._box[axis] = (current_low, current_high)
        self._project_box()
        self.box_update_timer.start(100)

    def _apply_box_update(self):
        if self._rows and self._box:
            self._commit_box(self._rows, list(self._box))

    def update_roi_visuals(self):
        parent = self.parent_widget
        if not self._rows or not parent or not self._items_ready:
            return
        box = _read_box(parent, self._rows)
        if box:
            self._box = box
            self._project_box()
