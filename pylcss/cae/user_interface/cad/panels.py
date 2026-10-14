# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Results and node-library panels for Design Studio."""

from __future__ import annotations

import logging

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag

__all__ = ["EventLog", "LibraryPanel", "ResultsPanel"]


class EventLog:
    """Session event sink, replacing the former History panel.

    These events are useful in ``pylcss.log`` when reconstructing what a user
    did before a problem, but they are not engineering results and were only
    competing with the results dock for space. Kept as a plain object rather
    than a hidden widget so nothing is rendered or retained by Qt.
    """

    def __init__(self, limit: int = 500):
        self._events = deque(maxlen=limit)

    def add_event(self, event_text):
        """Record one session event and mirror it to the application log."""
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {event_text}"
        self._events.append(stamped)
        logging.getLogger(__name__).info("%s", stamped)

    def events(self):
        """Return the retained events, oldest first."""
        return list(self._events)


class _ElidedLabel(QtWidgets.QLabel):
    """Single-line label that ellipsizes rather than wrapping.

    The results dock is narrow, so a long metric name like "Mean-ratio quality
    (5th percentile)" wrapped onto a second line and made that row twice as
    tall as its neighbours. Eliding keeps one row per metric and leaves the
    full text on the tooltip.
    """

    def __init__(self, text="", parent=None, is_key=False):
        super().__init__(parent)
        self._full_text = ""
        self._eliding = False
        self._is_key = is_key
        self.setWordWrap(False)
        self.setText(text)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._apply_elision()

    def full_text(self):
        """The unabbreviated text, whatever is currently displayed."""
        return self._full_text

    def sizeHint(self):
        # Ask for the untruncated width so a row that fits is never shortened.
        return QtCore.QSize(
            self.fontMetrics().horizontalAdvance(self._full_text) + 2,
            super().sizeHint().height(),
        )

    def minimumSizeHint(self):
        if self._is_key:
            # Guarantee key labels (e.g. Type, Backend, Visualization, Work directory)
            # reserve enough space so they are never squeezed into "T...", "Back...".
            adv = self.fontMetrics().horizontalAdvance(self._full_text) + 2
            return QtCore.QSize(min(adv, 130), super().minimumSizeHint().height())
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self):
        if self._eliding:
            return
        self._eliding = True
        try:
            metrics = self.fontMetrics()
            rect = self.contentsRect()
            avail_w = rect.width() if rect.width() > 0 else self.width()
            avail_w = max(0, avail_w - 4)
            if avail_w <= 0 or metrics.horizontalAdvance(self._full_text) <= avail_w:
                shown = self._full_text
            else:
                shown = metrics.elidedText(
                    self._full_text, QtCore.Qt.ElideRight, avail_w
                )
            QtWidgets.QLabel.setText(self, shown)
        finally:
            self._eliding = False


class ResultsPanel(QtWidgets.QWidget):
    """Summary of the most recent FEA / Impact / TopOpt solve.

    Pulled from the result dict that the solver nodes already produce — so we
    do not duplicate any computation; this is purely a presentation surface
    for what otherwise only goes to stdout.
    """

    view_mode_requested = QtCore.Signal(str)
    undeformed_requested = QtCore.Signal(bool)
    extrema_requested = QtCore.Signal(bool)
    deform_scale_requested = QtCore.Signal(str)

    # Mirrors the inspector's group styling; this panel sits directly beneath
    # it in the same dock and should not look like a different application.
    _RESULTS_QSS = """
        #ResultsPanel { background: #1c1e22; }
        QScrollArea { background: transparent; border: none; }
        #qt_scrollarea_viewport { background: transparent; }
        #Results3DModeButton, #ResultsTablesGraphsButton {
            background-color: transparent;
            color: #e0e4ea;
            border: 1px solid #383e4a;
            border-radius: 5px;
            padding: 5px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        #Results3DModeButton:hover, #ResultsTablesGraphsButton:hover {
            background-color: #383f4d;
            border-color: #4f5869;
            color: #ffffff;
        }
        #Results3DModeButton:pressed, #ResultsTablesGraphsButton:pressed {
            background-color: #1e2229;
        }
        QGroupBox {
            background: #24272d;
            border: 1px solid #2f333a;
            border-radius: 7px;
            margin-top: 13px;
            padding: 9px 7px 7px 7px;
            font-size: 11px;
            font-weight: 600;
            color: #cdd2d9;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            color: #d29922;
            font-weight: 700;
        }
        QLabel { color: #aab0b8; font-size: 11px; background: transparent; }
    """

    result_received = QtCore.Signal(object, object)
    open_results_page_requested = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        # Matches the inspector so the two halves of the right-hand dock read
        # as one panel.
        self.setObjectName("ResultsPanel")
        self.setStyleSheet(self._RESULTS_QSS)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(0)

        self._empty = QtWidgets.QLabel("No results yet.")
        self._empty.setStyleSheet("color:#8f98a5; font-style:italic;")
        self._empty.setVisible(False)
        # Measure-category values are collected separately from solver runs so
        # they stay on screen while the user inspects a solver result.
        self._measurements = []
        self._last_data = None
        self._results_dialog = None
        self._undeformed_checked = False
        self._extrema_checked = False

        # Header action bar with single full-width 3D view popup menu button
        self._header_bar = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(self._header_bar)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self._btn_3d_mode = QtWidgets.QToolButton()
        self._btn_3d_mode.setObjectName("Results3DModeButton")
        self._btn_3d_mode.setText("3D View")
        self._btn_3d_mode.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._btn_3d_mode.setToolTip("Select 3D simulation field to show in VTK viewer")
        self._btn_3d_mode.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._btn_3d_mode.setStyleSheet(
            "QToolButton { padding: 5px 10px; font-weight: 600; font-size: 11px; text-align: left; }"
        )
        try:
            import qtawesome as qta
            self._btn_3d_mode.setIcon(qta.icon("fa5s.cube", color="#8b949e"))
        except Exception:
            pass

        self._btn_tables_graphs = QtWidgets.QPushButton("Tables & Graphs")
        self._btn_tables_graphs.setObjectName("ResultsTablesGraphsButton")
        self._btn_tables_graphs.setVisible(False)
        self._btn_tables_graphs.clicked.connect(
            lambda: self.open_detailed_results("tables")
        )

        h_layout.addWidget(self._btn_3d_mode)
        self._header_bar.setVisible(False)
        outer.addWidget(self._header_bar)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._content = QtWidgets.QWidget()
        self._content.setMinimumWidth(0)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._scroll.setWidget(self._content)
        self._scroll.setVisible(False)

        class _ResultsViewportFilter(QtCore.QObject):
            def __init__(self, content_widget, scroll_widget):
                super().__init__(scroll_widget)
                self._content = content_widget
                self._scroll = scroll_widget

            def eventFilter(self, obj, event):
                if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
                    vp_w = self._scroll.viewport().width()
                    if vp_w > 0:
                        self._content.setMaximumWidth(vp_w)
                return super().eventFilter(obj, event)

        self._scroll.viewport().installEventFilter(
            _ResultsViewportFilter(self._content, self._scroll)
        )

    def apply_theme(self, theme_name):
        from pylcss.user_interface.common.theme_manager import THEMES
        self._theme_name = "light" if str(theme_name).lower() == "light" else "dark"
        colors = THEMES[self._theme_name]
        self.setStyleSheet(
            f"#ResultsPanel {{ background: {colors['bg_panel']}; }}"
            f"QToolButton {{ background: {colors['bg_input']}; color: {colors['text_main']}; border: 1px solid {colors['border']}; border-radius: 4px; padding: 5px 10px; }}"
            f"QToolButton:hover {{ color: {colors['primary']}; }}"
            f"QLabel {{ color: {colors['text_dim']}; }}"
        )
        self._update_3d_menu()

    def _get_available_3d_modes(self, data: Optional[Dict[str, Any]]) -> List[str]:
        from .result_fields import available_result_fields
        return available_result_fields(data)

    def _update_3d_menu(self):
        data = self._last_data or {}
        modes = self._get_available_3d_modes(data)
        current = str(data.get("visualization_mode") or modes[0])
        if data.get("type") == "topopt_voxel":
            from .result_fields import topopt_view_mode
            current = topopt_view_mode(data, current)
        if current not in modes:
            current = modes[0]
        self._btn_3d_mode.setText(f"3D View: {current}")

        menu = QtWidgets.QMenu(self)
        from pylcss.user_interface.common.theme_manager import THEMES
        colors = THEMES[getattr(self, "_theme_name", "dark")]
        menu.setStyleSheet(
            "QMenu { background-color: #21242b; border: 1px solid #383e4a; color: #e0e4ea; padding: 4px; }"
            "QMenu::item { padding: 6px 20px 6px 24px; font-size: 11px; border-radius: 3px; }"
            "QMenu::item:selected { background-color: #2d3748; color: #d29922; }"
            "QMenu::item:disabled { color: #64748b; font-weight: bold; background: transparent; }"
            "QMenu::separator { height: 1px; background-color: #2e3440; margin: 4px 6px; }"
        )
        menu.setStyleSheet(
            f"QMenu {{ background: {colors['bg_panel']}; color: {colors['text_main']}; border: 1px solid {colors['border']}; padding: 4px; }}"
            "QMenu::item { padding: 6px 20px; }"
            f"QMenu::item:selected {{ background: {colors['bg_input']}; color: {colors['primary']}; }}"
            f"QMenu::item:disabled {{ color: {colors['text_dim']}; }}"
        )

        header = menu.addAction("3D Display Mode")
        header.setEnabled(False)

        for mode in modes:
            act = menu.addAction(mode)
            act.setCheckable(True)
            if mode == current:
                act.setChecked(True)
            act.triggered.connect(
                lambda checked=False, m=mode: self._on_mode_selected(m)
            )

        menu.addSeparator()

        act_undef = menu.addAction("Undeformed Wireframe")
        act_undef.setCheckable(True)
        act_undef.setEnabled(data.get("displacement") is not None or bool(data.get("frames")))
        act_undef.setChecked(self._undeformed_checked)
        act_undef.toggled.connect(self._on_undeformed_toggled)

        act_extrema = menu.addAction("Min / Max Extrema Markers")
        act_extrema.setCheckable(True)
        act_extrema.setEnabled(modes != ["Default View"] and current not in {"CAD", "STL Shape", "Lattice", "Manufactured Mesh", "Deformed Animation"})
        act_extrema.setChecked(self._extrema_checked)
        act_extrema.toggled.connect(self._on_extrema_toggled)

        menu.addSeparator()

        act_tables = menu.addAction("Data Tables...")
        act_tables.triggered.connect(
            lambda: self.open_detailed_results("tables")
        )

        act_graphs = menu.addAction("Graph Results...")
        act_graphs.triggered.connect(
            lambda: self.open_detailed_results("graphs")
        )

        self._btn_3d_mode.setMenu(menu)

    def _on_mode_selected(self, mode: str):
        self._btn_3d_mode.setText(f"3D View: {mode}")
        if self._last_data is not None:
            self._last_data["visualization_mode"] = mode
        self.view_mode_requested.emit(mode)

    def _on_undeformed_toggled(self, checked: bool):
        self._undeformed_checked = bool(checked)
        self.undeformed_requested.emit(bool(checked))

    def _on_extrema_toggled(self, checked: bool):
        self._extrema_checked = bool(checked)
        self.extrema_requested.emit(bool(checked))

    def open_detailed_results(self, initial_tab: str = "tables"):
        """Switch to the Results tab in the main CAE window without popping up a dialog."""
        self.open_results_page_requested.emit(str(initial_tab))

    @staticmethod
    def _fmt(value, unit=""):
        if value is None:
            return "—"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if v == 0.0:
            return f"0 {unit}".strip()
        if abs(v) >= 1e3 or abs(v) < 1e-2:
            return f"{v:.3e} {unit}".strip()
        return f"{v:.4g} {unit}".strip()

    def _clear(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

    def clear_results(self):
        """Forget solver and measurement data when the project changes."""
        self._last_data = None
        self._measurements = []
        self._clear()
        self.setVisible(False)
        self._empty.setVisible(False)
        self._header_bar.setVisible(False)

    def _add_section(self, title, rows):
        """rows: list of (label, value-string)."""
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(10, 6, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        # Proportional 56/44 grid split so key names like "Type", "Backend", "Nodes",
        # "Volume", "Mass", "Solver status" are never crushed by long value strings.
        grid.setColumnStretch(0, 56)
        grid.setColumnStretch(1, 44)

        for row_idx, (label, val) in enumerate(rows):
            lk = _ElidedLabel(label, is_key=True)
            lk.setStyleSheet("color: #B0BEC5;")
            lk.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Preferred,
            )
            lv = _ElidedLabel(val, is_key=False)
            lv.setStyleSheet("color: #FAFAFA; font-weight: bold;")
            lv.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lv.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Preferred,
            )
            lv.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(lk, row_idx, 0)
            grid.addWidget(lv, row_idx, 1)
        self._content_layout.addWidget(group)

    def _add_topology_history_plot(self, data):
        """Plot normalized convergence responses already cached by TopOpt."""
        try:
            import pyqtgraph as pg
        except Exception:
            logging.getLogger(__name__).debug(
                "pyqtgraph is unavailable; convergence chart omitted.",
                exc_info=True,
            )
            return

        curves = []

        def _normalized(values):
            array = np.asarray(
                [] if values is None else values,
                dtype=float,
            ).reshape(-1)
            array = array[np.isfinite(array)]
            if array.size < 2:
                return None
            reference = max(abs(float(array[0])), 1.0e-30)
            return array / reference

        objective = _normalized(data.get("objective_history"))
        if objective is not None:
            curves.append(("Objective / initial", objective, "#58a6ff"))

        raw_change = data.get("change_history")
        change = np.asarray(
            [] if raw_change is None else raw_change,
            dtype=float,
        ).reshape(-1)
        change = change[np.isfinite(change)]
        if change.size >= 2:
            curves.append(("Design change", change, "#d29922"))

        raw_stress = data.get("stress_history")
        stress = np.asarray(
            [] if raw_stress is None else raw_stress,
            dtype=float,
        ).reshape(-1)
        stress = stress[np.isfinite(stress)]
        stress_meta = data.get("stress_aggregation")
        allowable = (
            float(stress_meta.get("allowable_stress") or 0.0)
            if isinstance(stress_meta, dict)
            else 0.0
        )
        if stress.size >= 2 and allowable > 0.0:
            curves.append(("Stress utilization", stress / allowable, "#f85149"))

        if not curves:
            return

        group = QtWidgets.QGroupBox("Convergence History")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        plot = pg.PlotWidget(background=None)
        plot.setMinimumHeight(190)
        plot.showGrid(x=True, y=True, alpha=0.20)
        plot.setLabel("bottom", "Iteration")
        plot.setLabel("left", "Normalized response")
        plot.addLegend(offset=(8, 8))
        for label, values, color in curves:
            iterations = np.arange(1, len(values) + 1, dtype=float)
            plot.plot(
                iterations,
                values,
                name=label,
                pen=pg.mkPen(color=color, width=2),
            )
        if any(label == "Stress utilization" for label, *_rest in curves):
            plot.addLine(
                y=1.0,
                pen=pg.mkPen(
                    "#f85149",
                    width=1,
                    style=QtCore.Qt.DashLine,
                ),
            )
        layout.addWidget(plot)
        self._content_layout.addWidget(group)

    # Energy channels drawn on the balance chart, in legend order.  Kinetic and
    # internal energy are the physics; hourglass and contact energy are the
    # numerical-health traces a reviewer checks before trusting the run.
    _CRASH_ENERGY_CHANNELS = (
        ("kinetic_energy_kj", "Kinetic", "#58a6ff"),
        ("internal_energy_kj", "Internal (absorbed)", "#3fb950"),
        ("total_energy_kj", "Total", "#c9d1d9"),
        ("external_work_kj", "External work", "#9e8ac9"),
        ("contact_energy_kj", "Contact", "#d29922"),
        ("hourglass_energy_kj", "Hourglass", "#f85149"),
    )

    @staticmethod
    def _crash_channel(source, name):
        """Return a 1-D channel from a measurement group, or None.

        Sample order is preserved rather than compacted: these channels are
        plotted against each other, so dropping a non-finite sample from one
        would silently shift it against its partner.  pyqtgraph already renders
        a NaN as a break in the line, which is the honest representation.
        """
        if not isinstance(source, dict):
            return None
        raw = source.get(name)
        if raw is None:
            return None
        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 2 or not np.any(np.isfinite(values)):
            return None
        return values

    def _add_crash_history_plots(self, data):
        """Plot the SAE J211 crash channels the solver already recorded.

        The backend stores both unfiltered and CFC-filtered traces, so the
        crush curve and pulse show the filtered signal used for peak reporting
        with the raw trace behind it — standard crash engineering convention.
        """
        histories = data.get("histories")
        if not isinstance(histories, dict):
            return
        raw = histories.get("raw")
        processed = histories.get("processed")
        if not isinstance(processed, dict):
            return
        try:
            import pyqtgraph as pg
        except Exception:
            logging.getLogger(__name__).debug(
                "pyqtgraph is unavailable; crash history charts omitted.",
                exc_info=True,
            )
            return

        def _new_plot(title, x_label, y_label):
            plot = pg.PlotWidget(background=None, title=title)
            plot.setMinimumHeight(190)
            plot.showGrid(x=True, y=True, alpha=0.20)
            plot.setLabel("bottom", x_label)
            plot.setLabel("left", y_label)
            plot.addLegend(offset=(8, 8))
            return plot

        plots = []

        # 1. Force vs crush — the curve that defines mean force, CFE, and SEA.
        crush = self._crash_channel(processed, "crush_displacement_mm")
        force = self._crash_channel(processed, "rigid_wall_force_kN")
        if crush is not None and force is not None:
            count = min(crush.size, force.size)
            plot = _new_plot(
                "Crush curve", "Crush displacement (mm)", "Wall force (kN)"
            )
            raw_force = self._crash_channel(raw, "rigid_wall_force_kN")
            if raw_force is not None and raw_force.size >= count:
                plot.plot(
                    crush[:count],
                    raw_force[:count],
                    name="Unfiltered",
                    pen=pg.mkPen(color="#475569", width=1),
                )
            filter_label = self._crash_filter_label(histories, "force_filter")
            plot.plot(
                crush[:count],
                force[:count],
                name=filter_label,
                pen=pg.mkPen(color="#58a6ff", width=2),
            )
            mean_force = self._crash_metric(histories, "mean_crushing_force_kN")
            if mean_force:
                plot.addLine(
                    y=mean_force,
                    pen=pg.mkPen("#d29922", width=1, style=QtCore.Qt.DashLine),
                )
            plots.append(plot)

        time_ms = self._crash_channel(processed, "time_ms")

        # 2. Energy balance — the primary numerical-validity check.
        if time_ms is not None:
            energy_plot = _new_plot("Energy balance", "Time (ms)", "Energy (kJ)")
            drawn = 0
            for name, label, color in self._CRASH_ENERGY_CHANNELS:
                channel = self._crash_channel(processed, name)
                if channel is None:
                    continue
                count = min(time_ms.size, channel.size)
                energy_plot.plot(
                    time_ms[:count],
                    channel[:count],
                    name=label,
                    pen=pg.mkPen(color=color, width=2),
                )
                drawn += 1
            if drawn:
                plots.append(energy_plot)

        # 3. Deceleration pulse — what an occupant-safety reviewer reads first.
        pulse = self._crash_channel(processed, "acceleration_g")
        if time_ms is not None and pulse is not None:
            count = min(time_ms.size, pulse.size)
            plot = _new_plot(
                "Deceleration pulse", "Time (ms)", "Acceleration (g)"
            )
            raw_pulse = self._crash_channel(raw, "acceleration_g")
            if raw_pulse is not None and raw_pulse.size >= count:
                plot.plot(
                    time_ms[:count],
                    raw_pulse[:count],
                    name="Unfiltered",
                    pen=pg.mkPen(color="#475569", width=1),
                )
            plot.plot(
                time_ms[:count],
                pulse[:count],
                name=self._crash_filter_label(histories, "acceleration_filter"),
                pen=pg.mkPen(color="#f85149", width=2),
            )
            plots.append(plot)

        if not plots:
            return
        group = QtWidgets.QGroupBox("Impact time histories (SAE J211)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        for plot in plots:
            layout.addWidget(plot)
        self._content_layout.addWidget(group)

    @staticmethod
    def _crash_filter_label(histories, key):
        """Name the CFC class the backend actually applied, e.g. 'CFC 600'."""
        processing = histories.get("processing")
        label = ""
        if isinstance(processing, dict):
            label = str(processing.get(key) or "").strip()
        return label or "Filtered"

    @staticmethod
    def _crash_metric(histories, name):
        """Return a finite scalar metric, or 0.0 when unavailable."""
        metrics = histories.get("metrics")
        if not isinstance(metrics, dict):
            return 0.0
        try:
            value = float(metrics.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    _SOLVER_RESULT_TYPES = (
        "fea",
        "modal",
        "crash",
        "external_solver",
        "topopt_voxel",
        "mesh",
        "remesh",
    )

    @classmethod
    def measurement_rows(cls, node_name, class_name, result):
        """Format one Measure node's output as ``(label, value)`` rows.

        Measure Distance and Surface Area return a bare number so they can feed
        numeric inputs downstream; Mass Properties and Bounding Box return a
        dictionary.  Both shapes are normalized here.
        """
        label = str(node_name or class_name)
        if isinstance(result, dict):
            prop = result.get("property")
            if prop == "mass_properties":
                center = result.get("center_of_mass") or ()
                rows = [
                    (f"{label} — mass", cls._fmt(result.get("mass"), "kg")),
                    (f"{label} — volume", cls._fmt(result.get("volume"), "mm³")),
                ]
                if len(center) >= 3:
                    rows.append(
                        (
                            f"{label} — centre of mass",
                            ", ".join(cls._fmt(v, "mm") for v in center[:3]),
                        )
                    )
                return rows
            if prop == "bounding_box":
                return [
                    (
                        f"{label} — size (L × W × H)",
                        " × ".join(
                            cls._fmt(result.get(key))
                            for key in ("length", "width", "height")
                        )
                        + " mm",
                    ),
                    (
                        f"{label} — enclosed volume",
                        cls._fmt(result.get("volume"), "mm³"),
                    ),
                ]
            return []
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            unit = "mm²" if class_name == "SurfaceAreaNode" else "mm"
            return [(label, cls._fmt(result, unit))]
        return []

    def set_measurements(self, measurements):
        """Show the Measure-node values collected after a graph run."""
        self._measurements = [
            (str(key), str(value)) for key, value in (measurements or [])
        ]
        self.show_result(self._last_data)

    def show_result(self, data):
        """Populate from a solver result dict.  Safe to call with None."""
        solver_data = (
            data
            if isinstance(data, dict)
            and data.get("type") in self._SOLVER_RESULT_TYPES
            else None
        )
        if solver_data is not None:
            self._last_data = solver_data
        if solver_data is None and not self._measurements:
            self.setVisible(False)
            self._empty.setVisible(False)
            self._header_bar.setVisible(False)
            return

        self._clear()
        self._empty.setVisible(False)
        if solver_data is not None:
            self.setVisible(True)
            self._header_bar.setVisible(True)
            self._update_3d_menu()
            self.result_received.emit(self._last_data, self._measurements)
        else:
            self.setVisible(False)
            self._header_bar.setVisible(False)
        return



class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""

    _CATEGORY_ICONS = {
        "Geometry":       ("fa5s.cube",           "#7a9bbf"),  # Muted slate blue
        "Selection":      ("fa5s.mouse-pointer",  "#6e9e7a"),  # Muted sage
        "Analysis Setup": ("fa5s.sliders-h",      "#8b949e"),  # Neutral text-dim
        "Solvers":        ("fa5s.play-circle",    "#d29922"),  # MDAO amber (primary action)
        "Measure":        ("fa5s.ruler-combined", "#9e8ac9"),  # Muted plum
        "Data":           ("fa5s.file-export",    "#8b949e"),  # Neutral text-dim
    }
    # Glyph only. The colour always comes from the category the node is listed
    # under, with no per-node exceptions, so a category cannot end up half one
    # colour and half another — which is what happened when each entry carried
    # its own hex value and Combine CAD Bodies, Import STEP and Import Mesh
    # kept the colour of the category they were originally filed in. Port and
    # wire colours are a separate, per-connection palette; the library groups
    # by discipline, so it is the discipline that colours it.
    _NODE_ICONS = (
        ("com.cad.geometry.box", "fa5s.cube"),
        ("com.cad.geometry.cylinder", "fa5s.database"),
        ("com.cad.geometry.boolean", "fa5s.object-group"),
        ("com.cad.geometry.transform", "fa5s.arrows-alt"),
        ("com.cad.geometry.linear_pattern", "fa5s.grip-horizontal"),
        ("com.cad.code_part", "fa5s.code"),
        ("com.cad.freecad_part", "fa5s.drafting-compass"),
        ("com.cad.import_step", "fa5s.file-import"),
        ("com.cad.import_stl", "fa5s.file-import"),
        ("com.cad.select_face_interactive", "fa5s.mouse-pointer"),
        ("com.cad.select_face", "fa5s.hand-pointer"),
        ("com.cad.sim.mesh", "fa5s.project-diagram"),
        ("com.cad.sim.material", "fa5s.layer-group"),
        ("com.cad.sim.component", "fa5s.puzzle-piece"),
        ("com.cad.sim.constraint", "fa5s.anchor"),
        ("com.cad.sim.load", "fa5s.arrow-down"),
        ("com.cad.sim.gravity", "fa5s.globe"),
        ("com.cad.sim.pressure_load", "fa5s.compress-arrows-alt"),
        ("com.cad.sim.solver", "fa5s.play-circle"),
        ("com.cad.sim.modal_solver", "fa5s.wave-square"),
        ("com.cad.topopt.support", "fa5s.anchor"),
        ("com.cad.topopt.load", "fa5s.arrow-down"),
        ("com.cad.sim.topopt_voxel", "fa5s.project-diagram"),
        ("com.cad.sim.lattice_infill", "fa5s.border-all"),
        ("com.cad.sim.impact", "fa5s.car-crash"),
        ("com.cad.sim.crash_solver", "fa5s.play-circle"),
        ("com.cad.assembly", "fa5s.cubes"),
        ("com.cad.mass_properties", "fa5s.weight"),
        ("com.cad.bounding_box", "fa5s.vector-square"),
        ("com.cad.measure_distance", "fa5s.ruler"),
        ("com.cad.surface_area", "fa5s.draw-polygon"),
        ("com.cad.number", "fa5s.hashtag"),
        ("com.cad.math_expression", "fa5s.square-root-alt"),
        ("com.cad.export_step", "fa5s.file-export"),
        ("com.cad.export_stl", "fa5s.file-export"),
    )
    _DEFAULT_CATEGORY_COLOR = "#8b949e"

    @classmethod
    def _category_color(cls, category: str) -> str:
        spec = cls._CATEGORY_ICONS.get(category)
        return spec[1] if spec else cls._DEFAULT_CATEGORY_COLOR

    @staticmethod
    def _icon_for(label: str):
        try:
            import qtawesome as qta
        except Exception:
            qta = None
        icon_spec = LibraryPanel._CATEGORY_ICONS.get(label)
        if qta is not None and icon_spec is not None:
            try:
                return qta.icon(icon_spec[0], color=icon_spec[1])
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        return QtWidgets.QApplication.style().standardIcon(
            QtWidgets.QStyle.SP_DirClosedIcon
        )

    @classmethod
    def _icon_for_node(cls, node_id: str, category: str = ""):
        """Glyph for one library entry, tinted with its category colour.

        A node with no glyph of its own falls back to the category's own icon
        rather than the platform "blank document", which is how Code Part,
        Select Geometry, FEA Component and Math Expression ended up looking
        like unrecognised files instead of tools.
        """
        try:
            import qtawesome as qta
        except Exception:
            return QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_FileIcon
            )

        color = cls._category_color(category)
        glyph = None
        for prefix, icon_name in cls._NODE_ICONS:
            if node_id == prefix or node_id.startswith(prefix + "."):
                glyph = icon_name
                break
        if glyph is None:
            category_spec = cls._CATEGORY_ICONS.get(category)
            glyph = category_spec[0] if category_spec else "fa5s.shapes"
        try:
            return qta.icon(glyph, color=color)
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        return QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)

    @staticmethod
    def _calculix_status():
        """Return a concise, non-launching CalculiX availability check."""
        try:
            from pylcss.cae.solver_backends.calculix import resolve_calculix_executable

            executable = resolve_calculix_executable()
        except Exception as exc:
            return False, f"CalculiX availability check failed: {exc}"
        if not executable:
            return False, (
                "CalculiX unavailable. Deck-only generation remains available; "
                "install with `python scripts/install_solvers.py --only ccx`."
            )
        return True, f"CalculiX ready: {executable}"

    @staticmethod
    def _opencourant_status():
        """Return a concise, non-launching availability check for the palette."""
        try:
            from pylcss.cae.solver_backends.execution import resolve_executable
            from pylcss.cae.solver_backends.courant_reader import resolve_anim_to_vtk

            starter = resolve_executable(
                None,
                ("PYLCSS_OPENCOURANT_STARTER", "OPENCOURANT_STARTER"),
                ("starter_win64.exe", "starter_win64", "starter_linux64_gf"),
            )
            engine = resolve_executable(
                None,
                ("PYLCSS_OPENCOURANT_ENGINE", "OPENCOURANT_ENGINE"),
                ("engine_win64.exe", "engine_win64", "engine_linux64_gf"),
            )
            converter = resolve_anim_to_vtk()
        except Exception as exc:
            return False, f"Availability check failed: {exc}"

        missing = [
            label
            for label, value in (
                ("Starter", starter),
                ("Engine", engine),
                ("anim_to_vtk", converter),
            )
            if not value
        ]
        if missing:
            return False, (
                "OpenCourant unavailable: missing " + ", ".join(missing) + ".\n"
                "Deck-only generation remains available; install with "
                "`python scripts/install_solvers.py --only opencourant`."
            )
        return True, "OpenCourant ready: Starter, Engine, and anim_to_vtk detected."

    def __init__(self, spawn_callback):
        super().__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)
        courant_ready, courant_status = self._opencourant_status()
        calculix_ready, calculix_status = self._calculix_status()

        # Search box — title was redundant with the dock title, dropped.
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search tools")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        self.layout.addWidget(self.search)

        # Tree view for categories
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QtCore.QSize(16, 16))
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(True)
        self.tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree.setTextElideMode(QtCore.Qt.ElideRight)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                border: none;
                background: transparent;
                outline: none;
            }
            QTreeView::item {
                padding: 4px 2px;
                border: 0;
            }
            QTreeView::item:hover {
                background: rgba(255, 255, 255, 18);
            }
            QTreeView::item:selected {
                background: rgba(210, 153, 34, 55);
                color: white;
            }
            """
        )
        # enable dragging from the library into the graph
        self.tree.setDragEnabled(True)
        self.tree.itemPressed.connect(self._start_drag)

        # Compact public palette. Legacy/script nodes remain registered so old
        # studies load, but they do not duplicate the native interactive tools.
        # Format: (Label, node_id, tooltip_description)
        categories = {
            "Geometry": [
                (
                    "Box",
                    "com.cad.geometry.box",
                    "Create a box.",
                ),
                (
                    "Cylinder",
                    "com.cad.geometry.cylinder",
                    "Create a cylinder.",
                ),
                (
                    "Boolean",
                    "com.cad.geometry.boolean",
                    "Join, cut, or intersect solids.",
                ),
                (
                    "Transform",
                    "com.cad.geometry.transform",
                    "Move or rotate geometry.",
                ),
                (
                    "Linear Pattern",
                    "com.cad.geometry.linear_pattern",
                    "Repeat geometry in a line.",
                ),
                (
                    "Group Bodies",
                    "com.cad.assembly",
                    "Group bodies without fusing them.",
                ),
                (
                    "Import CAD",
                    "com.cad.import_step",
                    "Import STEP, IGES, or BREP.",
                ),
                (
                    "Import Mesh",
                    "com.cad.import_stl",
                    "Import STL, OBJ, or 3MF.",
                ),
                (
                    "FreeCAD Part",
                    "com.cad.freecad_part",
                    "Edit a linked part in FreeCAD.",
                ),
                (
                    "Code Part",
                    "com.cad.code_part",
                    "Build geometry with CadQuery.",
                ),
            ],
            "Selection": [
                (
                    "Select Geometry",
                    "com.cad.select_face",
                    "Select geometry by rule.",
                ),
                (
                    "Pick Geometry",
                    "com.cad.select_face_interactive",
                    "Pick geometry in the 3D view.",
                ),
            ],
            "Analysis Setup": [
                (
                    "Material",
                    "com.cad.sim.material",
                    "Shared material for static FEA and topology optimization.",
                ),
                (
                    "Mesh",
                    "com.cad.sim.mesh",
                    "Create the FE mesh used by static FEA.",
                ),
                (
                    "Assembly Component",
                    "com.cad.sim.component",
                    "Optional: pair a mesh and material for multi-body assemblies.",
                ),
                (
                    "Support",
                    "com.cad.sim.constraint",
                    "Shared support for static FEA and topology optimization. "
                    "Connect FE Mesh for FEA; Topology maps the selection internally.",
                ),
                (
                    "Load",
                    "com.cad.sim.load",
                    "Shared load (force, pressure, or body acceleration) for static FEA and topology optimization.",
                ),
                (
                    "Impact Condition",
                    "com.cad.sim.impact",
                    "For crash analysis: set impact velocity, barrier, and platen mass.",
                ),
            ],
            "Solvers": [
                (
                    "Static FEA",
                    "com.cad.sim.solver",
                    "Run CalculiX.",
                ),
                (
                    "Modal Solver",
                    "com.cad.sim.modal_solver",
                    "Run CalculiX modal analysis to extract natural frequencies and mode shapes.",
                ),
                (
                    "Topology Optimization",
                    "com.cad.sim.topopt_voxel",
                    "Optimize material for the selected objective and constraints. "
                    "Produces a solid envelope.",
                ),
                (
                    "Lattice Infill",
                    "com.cad.sim.lattice_infill",
                    "Fill an existing solid with a lattice at a chosen cell "
                    "size and relative density. No loads, no supports, no "
                    "optimization.",
                ),
                (
                    "Impact Solver",
                    "com.cad.sim.crash_solver",
                    "Run OpenCourant.",
                ),
            ],
            "Measure": [
                ("Mass Properties", "com.cad.mass_properties", "Measure mass and volume."),
                ("Bounding Box", "com.cad.bounding_box", "Measure overall size."),
                (
                    "Measure Distance",
                    "com.cad.measure_distance",
                    "Measure the closest distance.",
                ),
                (
                    "Surface Area",
                    "com.cad.surface_area",
                    "Measure surface area.",
                ),
            ],
            "Data": [
                (
                    "Parameter",
                    "com.cad.number",
                    "Add a numeric input.",
                ),
                (
                    "Math Expression",
                    "com.cad.math_expression",
                    "Calculate a value.",
                ),
                (
                    "Export STEP",
                    "com.cad.export_step",
                    "Export STEP.",
                ),
                (
                    "Export STL",
                    "com.cad.export_stl",
                    "Export STL.",
                ),
            ],
        }

        cat_font = QtGui.QFont()
        cat_font.setBold(True)
        for category, items in categories.items():
            cat_item = QtWidgets.QTreeWidgetItem([category])
            cat_item.setFont(0, cat_font)
            icon = self._icon_for(category)
            if icon is not None:
                cat_item.setIcon(0, icon)
            # Category rows are not draggable / not selectable as a target.
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemIsSelectable)

            for item_data in items:
                label, node_id, tooltip = item_data
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, QtCore.Qt.UserRole, node_id)
                item.setIcon(0, self._icon_for_node(node_id, category))
                if node_id in ("com.cad.sim.solver", "com.cad.sim.modal_solver"):
                    tooltip += "\n\n" + calculix_status
                    item.setData(0, QtCore.Qt.UserRole + 1, calculix_ready)
                    if not calculix_ready:
                        item.setForeground(0, QtGui.QColor("#e3b341"))
                elif node_id in (
                    "com.cad.sim.crash_solver",
                ):
                    tooltip += "\n\n" + courant_status
                    item.setData(0, QtCore.Qt.UserRole + 1, courant_ready)
                    if not courant_ready:
                        item.setForeground(0, QtGui.QColor("#e3b341"))
                item.setToolTip(0, tooltip)  # Show description on hover
                cat_item.addChild(item)

            self.tree.addTopLevelItem(cat_item)

        # A compact category overview fits the default panel. Searching
        # expands only matching groups; clearing the search closes them again.
        self.tree.collapseAll()
        self.tree.itemDoubleClicked.connect(self._on_component_selected)
        self.layout.addWidget(self.tree)

    def _filter_tree(self, text):
        """Filter tree items based on search text."""
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(category.childCount()):
                item = category.child(j)
                matches = text in item.text(0).lower() if text else True
                item.setHidden(not matches)
                if matches:
                    visible_children += 1
            # Show category if any children match, or if search is empty
            category.setHidden(visible_children == 0 and bool(text))
            if visible_children > 0 or not text:
                category.setExpanded(bool(text))  # Auto-expand when searching

    def _on_component_selected(self, item, column):
        """Handle component selection from library."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if node_id:
            self.spawn_callback(node_id, item.text(0))

    def _start_drag(self, item, column):
        """Start a drag operation carrying the node identifier."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if not node_id:
            return
        drag = QDrag(self.tree)
        mime = QMimeData()
        mime.setData("application/x-node-id", str(node_id).encode("utf-8"))
        mime.setText(item.text(0))
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.CopyAction)
