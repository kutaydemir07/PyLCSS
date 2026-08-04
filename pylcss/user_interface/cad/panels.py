# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Results and node-library panels for Design Studio."""

from __future__ import annotations

import logging

from collections import deque
from datetime import datetime

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag

__all__ = ["EventLog", "LibraryPanel", "ResultsPanel"]


def _surface_handoff_rows(quality: dict) -> list[tuple[str, str]]:
    """Rows describing whether a recovered surface can be handed downstream.

    Watertightness and open-edge count are reported by the caller because the
    study gate keys off them. These are the measurements that decide the *next*
    step instead: winding and non-manifold seams decide whether a mesher will
    accept the file at all, aspect ratio decides whether it needs remeshing
    first, and sampled thickness is the first screen against a wall too thin to
    manufacture. Anything the recovery could not measure is omitted rather than
    shown as zero, and the warning list is surfaced verbatim.
    """
    rows: list[tuple[str, str]] = []

    if quality.get("winding_consistent") is False:
        rows.append(("Face winding", "Inconsistent"))
    nonmanifold = quality.get("nonmanifold_edges")
    if nonmanifold:
        rows.append(("Non-manifold edges", str(nonmanifold)))
    degenerate = quality.get("degenerate_faces")
    if degenerate:
        rows.append(("Zero-area faces", str(degenerate)))

    faces = quality.get("faces")
    if faces:
        rows.append(("Recovered triangles", f"{int(faces):,}"))
    volume = quality.get("enclosed_volume")
    if isinstance(volume, (int, float)):
        rows.append(("Enclosed volume", f"{float(volume):,.3g}"))
    area = quality.get("surface_area")
    if isinstance(area, (int, float)):
        rows.append(("Surface area", f"{float(area):,.3g}"))
    genus = quality.get("estimated_genus")
    if isinstance(genus, (int, float)):
        rows.append(("Estimated genus (through-holes)", f"{float(genus):.0f}"))

    p95 = quality.get("p95_triangle_aspect_ratio")
    median = quality.get("median_triangle_aspect_ratio")
    if isinstance(p95, (int, float)) and p95 > 0.0:
        detail = f"{float(p95):.1f} (p95)"
        if isinstance(median, (int, float)) and median > 0.0:
            detail += f", {float(median):.1f} (median)"
        rows.append(("Triangle aspect ratio", detail))

    thin = quality.get("sampled_p05_thickness")
    thinnest = quality.get("sampled_minimum_thickness")
    if isinstance(thin, (int, float)) and isinstance(thinnest, (int, float)):
        rows.append(
            (
                "Sampled wall thickness",
                f"{float(thinnest):.3g} min, {float(thin):.3g} at p05",
            )
        )

    warnings = quality.get("warnings") or ()
    for index, warning in enumerate(warnings):
        label = "Surface warnings" if index == 0 else ""
        rows.append((label, str(warning)))

    return rows


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

    # Mirrors the inspector's group styling; this panel sits directly beneath
    # it in the same dock and should not look like a different application.
    _RESULTS_QSS = """
        #ResultsPanel { background: #1c1e22; }
        QScrollArea { background: transparent; border: none; }
        #qt_scrollarea_viewport { background: transparent; }
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
            color: #6fb3ff;
            font-weight: 700;
        }
        QLabel { color: #aab0b8; font-size: 11px; background: transparent; }
    """

    def __init__(self):
        super().__init__()
        # Matches the inspector so the two halves of the right-hand dock read
        # as one panel.
        self.setObjectName("ResultsPanel")
        self.setStyleSheet(self._RESULTS_QSS)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 7)
        outer.setSpacing(5)

        self._empty = QtWidgets.QLabel("No results yet.")
        self._empty.setStyleSheet("color:#8f98a5; font-style:italic;")
        outer.addWidget(self._empty)
        # Measure-category values are collected separately from solver runs so
        # they stay on screen while the user inspects a solver result.
        self._measurements = []
        self._last_data = None
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Like the inspector: the panel is narrow and resizable, so content
        # wraps to the available width instead of growing a horizontal bar.
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._content = QtWidgets.QWidget()
        self._content.setMinimumWidth(0)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(5)
        self._scroll.setWidget(self._content)
        self._scroll.setVisible(False)
        outer.addWidget(self._scroll, 1)

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
        self._empty.setVisible(True)
        self._scroll.setVisible(False)

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

    def _add_warnings(self, warnings):
        if not warnings:
            return
        group = QtWidgets.QGroupBox(f"Warnings ({len(warnings)})")
        v = QtWidgets.QVBoxLayout(group)
        v.setContentsMargins(10, 6, 10, 10)
        for w in warnings:
            label = QtWidgets.QLabel(f"• {w}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #FFCC80;")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            v.addWidget(label)
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
            curves.append(("Objective / initial", objective, "#42A5F5"))

        raw_change = data.get("change_history")
        change = np.asarray(
            [] if raw_change is None else raw_change,
            dtype=float,
        ).reshape(-1)
        change = change[np.isfinite(change)]
        if change.size >= 2:
            curves.append(("Design change", change, "#FFB74D"))

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
            curves.append(("Stress utilization", stress / allowable, "#EF5350"))

        thermal = _normalized(data.get("thermal_compliance_history"))
        if (
            thermal is not None
            and str(data.get("physics_mode") or "").lower() == "thermo_mechanical"
        ):
            curves.append(("Thermal response / initial", thermal, "#66BB6A"))

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
                    "#EF5350",
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
        ("kinetic_energy_kj", "Kinetic", "#42A5F5"),
        ("internal_energy_kj", "Internal (absorbed)", "#66BB6A"),
        ("total_energy_kj", "Total", "#ECEFF1"),
        ("external_work_kj", "External work", "#AB47BC"),
        ("contact_energy_kj", "Contact", "#FFB74D"),
        ("hourglass_energy_kj", "Hourglass", "#EF5350"),
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
                    pen=pg.mkPen(color="#546E7A", width=1),
                )
            filter_label = self._crash_filter_label(histories, "force_filter")
            plot.plot(
                crush[:count],
                force[:count],
                name=filter_label,
                pen=pg.mkPen(color="#42A5F5", width=2),
            )
            mean_force = self._crash_metric(histories, "mean_crushing_force_kN")
            if mean_force:
                plot.addLine(
                    y=mean_force,
                    pen=pg.mkPen("#FFB74D", width=1, style=QtCore.Qt.DashLine),
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
                    pen=pg.mkPen(color="#546E7A", width=1),
                )
            plot.plot(
                time_ms[:count],
                pulse[:count],
                name=self._crash_filter_label(histories, "acceleration_filter"),
                pen=pg.mkPen(color="#EF5350", width=2),
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
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return

        self._clear()
        self._empty.setVisible(False)
        self._scroll.setVisible(True)
        if self._measurements:
            self._add_section("Measurements", self._measurements)
        if solver_data is None:
            return
        data = solver_data
        rtype = data.get("type")

        backend = data.get("backend") or (
            "CalculiX" if rtype == "fea" else "OpenRadioss" if rtype == "crash" else "—"
        )
        if rtype == "topopt_voxel":
            backend = data.get("backend") or "pyMOTO"
        meta_rows = [
            ("Type", rtype.upper()),
            ("Backend", str(backend)),
        ]
        if "visualization_mode" in data:
            meta_rows.append(("Visualization", str(data["visualization_mode"])))
        if "analysis_type" in data:
            meta_rows.append(("Analysis", str(data["analysis_type"])))
        if "external_status" in data:
            meta_rows.append(("Solver status", str(data["external_status"])))
        if "work_dir" in data:
            meta_rows.append(("Work directory", str(data["work_dir"])))
        self._add_section("Run", meta_rows)

        mesh_obj = data.get("mesh")
        quality = data.get("mesh_quality")
        if quality is None and mesh_obj is not None:
            quality = getattr(mesh_obj, "quality_report", None)
        if isinstance(quality, dict):
            mesh_rows = []
            if mesh_obj is not None:
                try:
                    mesh_rows.extend(
                        [
                            ("Nodes", f"{int(np.asarray(mesh_obj.p).shape[1]):,}"),
                            ("Elements", f"{int(np.asarray(mesh_obj.t).shape[1]):,}"),
                        ]
                    )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            mesh_rows.extend(
                [
                    ("Assessment", str(quality.get("assessment") or "—")),
                    (
                        "Mean-ratio quality (minimum)",
                        self._fmt(quality.get("min_mean_ratio")),
                    ),
                    (
                        "Mean-ratio quality (5th percentile)",
                        self._fmt(quality.get("p05_mean_ratio")),
                    ),
                    (
                        "Mean-ratio quality (mean)",
                        self._fmt(quality.get("mean_mean_ratio")),
                    ),
                    (
                        "Maximum edge ratio",
                        self._fmt(quality.get("max_edge_ratio")),
                    ),
                    (
                        "Collapsed/invalid elements",
                        str(int(quality.get("degenerate_elements") or 0)),
                    ),
                ]
            )
            self._add_section("Mesh quality (1.0 = ideal)", mesh_rows)

        if rtype == "fea" or rtype == "external_solver":
            metrics = []
            peak_disp = data.get("peak_displacement")
            if peak_disp is None and data.get("displacement") is not None:
                try:
                    arr = np.asarray(data["displacement"], dtype=float)
                    if arr.size:
                        if arr.ndim == 1 and arr.size % 3 == 0:
                            arr = arr.reshape((-1, 3))
                        peak_disp = float(np.max(np.linalg.norm(arr, axis=1)))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if peak_disp is not None:
                metrics.append(("Peak displacement", self._fmt(peak_disp, "mm")))

            peak_stress = data.get("peak_stress_nodal")
            if peak_stress is None and data.get("stress") is not None:
                try:
                    arr = np.asarray(data["stress"], dtype=float)
                    if arr.size:
                        peak_stress = float(np.max(arr))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if peak_stress is not None:
                metrics.append(
                    ("Peak stress (nodal extrapolated)", self._fmt(peak_stress, "MPa"))
                )
            if data.get("stress_location") == "gauss" and "max_stress_gauss" in data:
                metrics.append(
                    ("Peak stress (Gauss)", self._fmt(data["max_stress_gauss"], "MPa"))
                )
            if "strain_energy" in data:
                metrics.append(
                    ("Strain energy", self._fmt(data["strain_energy"], "N mm"))
                )
            if data.get("compliance") is not None:
                metrics.append(("Compliance", self._fmt(data["compliance"], "N mm")))
            if data.get("reaction_force") is not None:
                try:
                    reaction = np.asarray(data["reaction_force"], dtype=float).reshape(
                        -1
                    )
                    if reaction.size >= 3:
                        metrics.append(
                            (
                                "Support reaction (X, Y, Z)",
                                ", ".join(
                                    self._fmt(value, "N") for value in reaction[:3]
                                ),
                            )
                        )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if "volume" in data:
                metrics.append(("Volume", self._fmt(data["volume"], "mm^3")))
            if "mass" in data:
                metrics.append(("Mass", self._fmt(float(data["mass"]) * 1000.0, "kg")))
            solver_elapsed = data.get("solver_elapsed_s")
            if solver_elapsed is not None:
                try:
                    elapsed_value = float(solver_elapsed)
                    metrics.append(("Solver wall time", self._fmt(elapsed_value, "s")))
                    mesh = data.get("mesh")
                    element_count = int(
                        np.asarray(mesh.t).shape[1]
                        if mesh is not None and hasattr(mesh, "t")
                        else 0
                    )
                    if elapsed_value > 0.0 and element_count:
                        metrics.append(
                            (
                                "Solver throughput",
                                f"{element_count / elapsed_value:,.0f} elements/s",
                            )
                        )
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Could not format solver timing.", exc_info=True
                    )
            if data.get("minimum_yield_safety_factor") is not None:
                metrics.append(
                    (
                        "Minimum yield safety factor",
                        self._fmt(data["minimum_yield_safety_factor"]),
                    )
                )
            if "deformation_scale" in data:
                try:
                    raw_scale = data["deformation_scale"]
                    if isinstance(raw_scale, str):
                        text_scale = raw_scale.strip().lower()
                        scale = (
                            1.0
                            if text_scale == "auto"
                            else float(text_scale.rstrip("x"))
                        )
                    else:
                        scale = float(raw_scale)
                    metrics.append(("Deformation scale", f"{scale:.1f}×"))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if metrics:
                self._add_section("Result", metrics)

            components = data.get("components")
            if isinstance(components, list) and components:
                component_rows = []
                for component in components:
                    if not isinstance(component, dict):
                        continue
                    name = str(component.get("name") or "Component")
                    material = str(component.get("material") or "Material")
                    detail = [material]
                    if component.get("elements") is not None:
                        detail.append(f"{int(component['elements']):,} elements")
                    if component.get("peak_stress_mpa") is not None:
                        detail.append(
                            f"peak {self._fmt(component['peak_stress_mpa'], 'MPa')}"
                        )
                    if component.get("yield_safety_factor") is not None:
                        detail.append(
                            "FoS "
                            + self._fmt(component["yield_safety_factor"])
                        )
                    component_rows.append((name, " · ".join(detail)))
                if component_rows:
                    self._add_section("Components", component_rows)

            connections = data.get("connections")
            if isinstance(connections, list) and connections:
                connection_rows = []
                for connection in connections:
                    if not isinstance(connection, dict):
                        continue
                    pair = (
                        f"{connection.get('component_a', '?')} ↔ "
                        f"{connection.get('component_b', '?')}"
                    )
                    connection_rows.append(
                        (
                            pair,
                            "Bonded tie · review · "
                            f"{self._fmt(connection.get('search_tolerance_mm'), 'mm')} search",
                        )
                    )
                if connection_rows:
                    self._add_section("Assembly interfaces", connection_rows)

        if rtype == "crash":
            crash_rows = []
            if "peak_displacement" in data:
                crash_rows.append(
                    (
                        "Event peak displacement",
                        self._fmt(data["peak_displacement"], "mm"),
                    )
                )
            if "final_frame_displacement" in data:
                crash_rows.append(
                    (
                        "Final-frame displacement",
                        self._fmt(data["final_frame_displacement"], "mm"),
                    )
                )
            if "peak_stress" in data:
                crash_rows.append(
                    ("Event peak Von Mises", self._fmt(data["peak_stress"], "MPa"))
                )
            if "final_frame_stress" in data:
                crash_rows.append(
                    (
                        "Final-frame Von Mises",
                        self._fmt(data["final_frame_stress"], "MPa"),
                    )
                )
            if "absorbed_energy_kj" in data:
                crash_rows.append(
                    (
                        "Absorbed / internal energy",
                        self._fmt(data["absorbed_energy_kj"], "kJ"),
                    )
                )
            elif "absorbed_energy" in data:
                crash_rows.append(
                    (
                        "Absorbed / internal energy",
                        self._fmt(float(data["absorbed_energy"]) / 1.0e6, "kJ"),
                    )
                )
            if "n_failed" in data:
                crash_rows.append(("Failed elements", str(data["n_failed"])))
            for key, label, unit in (
                ("peak_force", "Peak crushing force", "kN"),
                ("mean_force", "Mean crushing force", "kN"),
                ("crush_force_efficiency", "Crush-force efficiency", ""),
                ("specific_energy_absorption", "Specific energy absorption", "kJ/kg"),
                ("structural_mass_kg", "Deformable structure mass", "kg"),
                ("crush_distance", "Useful crush distance", "mm"),
                ("peak_acceleration_g", "Peak impact-pulse acceleration", "g"),
                ("delta_v", "Impact-pulse velocity change", "m/s"),
            ):
                if data.get(key) is not None:
                    crash_rows.append((label, self._fmt(data[key], unit)))
            if data.get("quality_status"):
                crash_rows.append(
                    ("Overall qualification", str(data["quality_status"]).title())
                )
            quality = data.get("quality")
            if isinstance(quality, dict):
                if quality.get("numerical_status"):
                    crash_rows.append(
                        (
                            "Numerical quality",
                            str(quality["numerical_status"]).title(),
                        )
                    )
                if quality.get("physical_validation_status"):
                    crash_rows.append(
                        (
                            "Physical validation",
                            str(quality["physical_validation_status"]).title(),
                        )
                    )
                failed_checks = list(quality.get("failed_checks") or [])
                if failed_checks:
                    crash_rows.append(
                        ("Failed qualification checks", ", ".join(failed_checks))
                    )
            if "ml_eligible" in data:
                crash_rows.append(
                    ("Eligible for ML dataset", "Yes" if data["ml_eligible"] else "No")
                )
            if "frames" in data and data["frames"]:
                crash_rows.append(("Animation frames", str(len(data["frames"]))))
            if "energy_balance_max_error" in data:
                crash_rows.append(
                    (
                        "Max. |energy error|",
                        f"{float(data['energy_balance_max_error']) * 100:.1f}%",
                    )
                )
            if "mass_balance_max_error" in data:
                crash_rows.append(
                    (
                        "Max. |mass change|",
                        f"{float(data['mass_balance_max_error']) * 100:.2f}%",
                    )
                )
            if crash_rows:
                self._add_section("Impact result", crash_rows)
            self._add_crash_history_plots(data)

        if rtype == "topopt_voxel":
            topo_rows = []
            if data.get("design_goal"):
                topo_rows.append(("Goal", str(data["design_goal"])))
            if data.get("physics_mode"):
                physics_label = (
                    "Coupled Structural + Thermal"
                    if str(data["physics_mode"]).lower() == "thermo_mechanical"
                    else str(data["physics_mode"]).replace("_", " ").title()
                )
                topo_rows.append(
                    ("Physics", physics_label)
                )
            if data.get("physics_model"):
                topo_rows.append(("Physics model", str(data["physics_model"])))
            grid_shape = data.get("grid_shape")
            bounds = data.get("bounds")
            voxel_dimensions = None
            try:
                grid = np.asarray(grid_shape, dtype=int)
                lower = np.asarray(bounds["min"], dtype=float)
                upper = np.asarray(bounds["max"], dtype=float)
                if (
                    grid.shape == (3,)
                    and np.all(grid > 0)
                    and lower.shape == (3,)
                    and upper.shape == (3,)
                ):
                    voxel_dimensions = (upper - lower) / grid
                    topo_rows.append(
                        (
                            "Analysis grid",
                            f"{grid[0]} × {grid[1]} × {grid[2]} voxels",
                        )
                    )
                    topo_rows.append(
                        (
                            "Analysis voxel dimensions",
                            " × ".join(f"{value:.3g}" for value in voxel_dimensions)
                            + " mm",
                        )
                    )
            except (TypeError, ValueError, KeyError, IndexError):
                voxel_dimensions = None
            if data.get("target_vol_frac") is not None:
                topo_rows.append(
                    ("Material budget", f"{float(data['target_vol_frac']) * 100:.1f}%")
                )
            if data.get("final_vol_frac") is not None:
                topo_rows.append(
                    ("Final material", f"{float(data['final_vol_frac']) * 100:.1f}%")
                )
            if data.get("intermediate_density_fraction") is not None:
                topo_rows.append(
                    (
                        "Intermediate density (0.1-0.9)",
                        f"{float(data['intermediate_density_fraction']) * 100:.1f}%",
                    )
                )
            if data.get("compliance") is not None:
                topo_rows.append(("Compliance", self._fmt(data["compliance"], "N mm")))
            if data.get("thermal_compliance") is not None:
                topo_rows.append(
                    (
                        "Thermal compliance",
                        self._fmt(data["thermal_compliance"], "W K"),
                    )
                )
            if data.get("stress_pnorm") is not None:
                topo_rows.append(
                    ("Stress P-norm proxy", self._fmt(data["stress_pnorm"], "MPa"))
                )
            if data.get("objective_reduction_pct") is not None:
                topo_rows.append(
                    (
                        "Objective improvement",
                        f"{float(data['objective_reduction_pct']):+.1f}%",
                    )
                )
            if data.get("volume") is not None:
                topo_rows.append(
                    (
                        "Density-equivalent volume",
                        self._fmt(data["volume"], "mm^3"),
                    )
                )
            if data.get("mass") is not None:
                topo_rows.append(
                    (
                        "Density-equivalent mass",
                        self._fmt(float(data["mass"]) * 1000.0, "kg"),
                    )
                )
            if data.get("recovered_design_volume") is not None:
                topo_rows.append(
                    (
                        "Recovered CAD design volume",
                        self._fmt(data["recovered_design_volume"], "mm^3"),
                    )
                )
            if data.get("recovered_design_mass") is not None:
                topo_rows.append(
                    (
                        "Recovered CAD design mass",
                        self._fmt(
                            float(data["recovered_design_mass"]) * 1000.0,
                            "kg",
                        ),
                    )
                )
            if data.get("recovery_volume_delta_pct") is not None:
                recovery_reference = str(
                    data.get("recovery_volume_reference")
                    or "density-equivalent design"
                )
                topo_rows.append(
                    (
                        f"CAD vs {recovery_reference} volume",
                        f"{float(data['recovery_volume_delta_pct']):+.1f}%",
                    )
                )
            if (
                data.get("assembly_hardware_volume") is not None
                and float(data.get("assembly_hardware_volume") or 0.0) > 0.0
            ):
                topo_rows.append(
                    (
                        "Separate joint hardware volume",
                        self._fmt(data["assembly_hardware_volume"], "mm^3"),
                    )
                )
            used_iterations = int(data.get("iterations") or 0)
            max_iterations = int(data.get("max_iterations") or 0)
            iteration_text = (
                f"{used_iterations} of {max_iterations} maximum"
                if max_iterations > 0
                else str(used_iterations)
            )
            topo_rows.append(("Optimizer iterations used", iteration_text))
            progressive = data.get("progressive_resolution")
            if isinstance(progressive, dict):
                levels = [
                    level
                    for level in progressive.get("levels", [])
                    if isinstance(level, dict)
                ]
                if levels:
                    grids = []
                    total_iterations = 0
                    for level in levels:
                        grid = level.get("grid")
                        if isinstance(grid, (list, tuple)) and len(grid) >= 3:
                            grids.append("×".join(str(int(v)) for v in grid[:3]))
                        total_iterations += int(level.get("iterations") or 0)
                    topo_rows.append(
                        (
                            "Program-controlled resolution",
                            f"{' → '.join(grids)} ({total_iterations} total iterations)",
                        )
                    )
            topo_rows.append(("Converged", "Yes" if data.get("converged") else "No"))
            if data.get("message"):
                topo_rows.append(("Stop reason", str(data["message"])))
            quality_gate = data.get("quality_gate")
            if isinstance(quality_gate, dict):
                topo_rows.append(
                    (
                        "Engineering gate",
                        str(quality_gate.get("status") or "concept only").title(),
                    )
                )
                topo_rows.append(
                    (
                        "Load-bearing components",
                        (
                            f"{int(quality_gate.get('manufactured_component_count') or 0)} "
                            f"(expected ≤ "
                            f"{int(quality_gate.get('expected_component_count') or 1)})"
                        ),
                    )
                )
                failed_checks = list(quality_gate.get("failed_checks") or [])
                if failed_checks:
                    topo_rows.append(
                        ("Blocking checks", ", ".join(failed_checks))
                    )
            manufacturing = data.get("manufacturing")
            if isinstance(manufacturing, dict):
                if manufacturing.get("structure"):
                    topo_rows.append(
                        ("Manufactured interpretation", str(manufacturing["structure"]))
                    )
                if manufacturing.get("surface_backend"):
                    topo_rows.append(
                        ("Surface recovery", str(manufacturing["surface_backend"]))
                    )
                surface_quality = manufacturing.get("surface_quality")
                if manufacturing.get("surface_quality_preset"):
                    topo_rows.append(
                        (
                            "Surface quality preset",
                            str(manufacturing["surface_quality_preset"]),
                        )
                    )
                if isinstance(surface_quality, dict):
                    watertight = surface_quality.get("watertight")
                    if watertight is not None:
                        topo_rows.append(
                            (
                                "Recovered surface watertight",
                                "Yes" if watertight else "No",
                            )
                        )
                    if surface_quality.get("open_boundary_edges") is not None:
                        topo_rows.append(
                            (
                                "Open surface edges",
                                str(surface_quality["open_boundary_edges"]),
                            )
                        )
                    topo_rows.extend(_surface_handoff_rows(surface_quality))
            self._add_section("Topology result", topo_rows)

            lattice = data.get("lattice_optimization")
            if isinstance(lattice, dict):
                lattice_rows = []
                for key, label in (
                    ("phase", "Workflow phase"),
                    ("cell_family", "Cell family"),
                    ("stiffness_model", "Part-scale stiffness surrogate"),
                    ("stiffness_interpolation", "Stiffness interpolation"),
                ):
                    value = lattice.get(key)
                    if key == "stiffness_model" and not value:
                        value = lattice.get("continuum_assumption")
                    if value:
                        lattice_rows.append((label, str(value)))
                minimum_density = lattice.get("minimum_relative_density")
                maximum_density = lattice.get("maximum_relative_density")
                variable_density = bool(lattice.get("variable_density"))
                lattice_rows.append(
                    (
                        "Explicit cell grading",
                        "Field graded" if variable_density else "Uniform / fixed cell",
                    )
                )
                if (
                    variable_density
                    and minimum_density is not None
                    and maximum_density is not None
                ):
                    lattice_rows.append(
                        (
                            "Local relative-density range",
                            (
                                f"{float(minimum_density):.3f} – "
                                f"{float(maximum_density):.3f}"
                            ),
                        )
                    )
                manufacturing = data.get("manufacturing")
                pitch_voxels = (
                    manufacturing.get("cell_size_voxels")
                    if isinstance(manufacturing, dict)
                    else None
                )
                if pitch_voxels is not None and voxel_dimensions is not None:
                    pitch = float(pitch_voxels) * voxel_dimensions
                    lattice_rows.append(
                        (
                            "Physical cell pitch",
                            " × ".join(f"{value:.3g}" for value in pitch) + " mm",
                        )
                    )
                lattice_rows.append(
                    (
                        "Independent explicit-lattice validation",
                        (
                            "Required"
                            if lattice.get("independent_validation_required")
                            else "Not requested"
                        ),
                    )
                )
                if lattice.get("limitation"):
                    lattice_rows.append(
                        ("Model limitation", str(lattice["limitation"]))
                    )
                member_sizing = lattice.get("member_sizing")
                if isinstance(member_sizing, dict):
                    completed = member_sizing.get("completed")
                    if completed is None:
                        completed = lattice.get("member_size_optimization")
                    lattice_rows.append(
                        (
                            "Phase 2 member sizing",
                            "Completed" if completed else "Not completed",
                        )
                    )
                    if member_sizing.get("reason"):
                        lattice_rows.append(
                            ("Member-sizing status", str(member_sizing["reason"]))
                        )
                    for key, label in (
                        ("member_count", "Sized members"),
                        (
                            "boundary_stabilizing_member_count",
                            "Boundary stabilizing members",
                        ),
                        ("iterations", "Sizing iterations"),
                    ):
                        if member_sizing.get(key) is not None:
                            lattice_rows.append(
                                (label, str(member_sizing[key]))
                            )
                    if member_sizing.get("converged") is not None:
                        lattice_rows.append(
                            (
                                "Member sizing converged",
                                "Yes" if member_sizing["converged"] else "No",
                            )
                        )
                    for key, label in (
                        ("maximum_stress_utilization", "Peak stress utilization"),
                        (
                            "maximum_buckling_utilization",
                            "Peak Euler-buckling utilization",
                        ),
                    ):
                        if member_sizing.get(key) is not None:
                            lattice_rows.append(
                                (label, f"{float(member_sizing[key]):.3f}")
                            )
                    if member_sizing.get("maximum_displacement") is not None:
                        lattice_rows.append(
                            (
                                "Peak truss displacement",
                                self._fmt(
                                    member_sizing["maximum_displacement"],
                                    "mm",
                                ),
                            )
                        )
                self._add_section("Lattice engineering status", lattice_rows)

            structural_cases = data.get("structural_cases")
            if isinstance(structural_cases, list) and structural_cases:
                case_rows = []
                for case in structural_cases:
                    if not isinstance(case, dict):
                        continue
                    details = []
                    if case.get("compliance") is not None:
                        details.append(
                            self._fmt(case["compliance"], "N mm")
                            + " compliance"
                        )
                    if case.get("maximum_displacement") is not None:
                        details.append(
                            self._fmt(case["maximum_displacement"], "mm")
                            + " max displacement"
                        )
                    case_rows.append(
                        (str(case.get("name") or "Load case"), " · ".join(details))
                    )
                if case_rows:
                    self._add_section("Structural load cases", case_rows)

            thermal_cases = data.get("thermal_cases")
            if isinstance(thermal_cases, list) and thermal_cases:
                thermal_rows = []
                for case in thermal_cases:
                    if not isinstance(case, dict):
                        continue
                    details = []
                    if case.get("thermal_compliance") is not None:
                        details.append(
                            self._fmt(case["thermal_compliance"], "W K")
                            + " thermal compliance"
                        )
                    if case.get("maximum_temperature_rise") is not None:
                        details.append(
                            self._fmt(case["maximum_temperature_rise"], "K")
                            + " max rise"
                        )
                    thermal_rows.append(
                        (str(case.get("name") or "Thermal case"), " · ".join(details))
                    )
                if thermal_rows:
                    self._add_section("Thermal load cases", thermal_rows)

            timing = data.get("timing")
            if isinstance(timing, dict):
                timing_rows = []
                for key, label in (
                    ("optimization_s", "Optimization"),
                    ("recovery_s", "Surface recovery"),
                    ("validation_and_cad_s", "Validation / CAD"),
                ):
                    if timing.get(key) is not None:
                        timing_rows.append(
                            (label, self._fmt(timing[key], "s"))
                        )
                if timing_rows:
                    self._add_section("Timing", timing_rows)

            validation = data.get("validation_summary")
            if isinstance(validation, dict):
                validation_rows = []
                if validation.get("max_stress") is not None:
                    validation_rows.append(
                        (
                            "Validated peak stress",
                            self._fmt(validation["max_stress"], "MPa"),
                        )
                    )
                if validation.get("compliance") is not None:
                    validation_rows.append(
                        (
                            "Validated compliance",
                            self._fmt(validation["compliance"], "N mm"),
                        )
                    )
                if validation.get("max_displacement") is not None:
                    validation_rows.append(
                        (
                            "Validated peak displacement",
                            self._fmt(validation["max_displacement"], "mm"),
                        )
                    )
                if validation_rows:
                    self._add_section("CalculiX validation", validation_rows)
            self._add_topology_history_plot(data)

        # Warnings from the external backends
        warnings = data.get("warnings") or []
        if warnings:
            self._add_warnings(list(warnings))


class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""

    _CATEGORY_ICONS = {
        "Geometry": ("fa5s.cube", "#81C784"),
        "Selection": ("fa5s.mouse-pointer", "#80CBC4"),
        "Analysis Setup": ("fa5s.sliders-h", "#64B5F6"),
        "Solvers": ("fa5s.play-circle", "#9CCC65"),
        "Measure": ("fa5s.ruler-combined", "#B39DDB"),
        "Data": ("fa5s.file-export", "#90A4AE"),
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
        ("com.cad.geometry.tube", "fa5s.circle"),
        ("com.cad.geometry.cylindrical_shell", "fa5s.circle-notch"),
        ("com.cad.geometry.boolean", "fa5s.object-group"),
        ("com.cad.geometry.through_hole", "fa5s.dot-circle"),
        ("com.cad.geometry.fillet", "fa5s.bezier-curve"),
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
        ("com.cad.topopt.support", "fa5s.anchor"),
        ("com.cad.topopt.load", "fa5s.arrow-down"),
        ("com.cad.sim.topopt_voxel", "fa5s.project-diagram"),
        ("com.cad.sim.lattice_voxel", "fa5s.th"),
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
    _DEFAULT_CATEGORY_COLOR = "#90A4AE"

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
            from pylcss.solver_backends.calculix import resolve_calculix_executable

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
    def _openradioss_status():
        """Return a concise, non-launching availability check for the palette."""
        try:
            from pylcss.solver_backends.execution import resolve_executable
            from pylcss.solver_backends.radioss_reader import resolve_anim_to_vtk

            starter = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
                ("starter_win64.exe", "starter_win64", "starter_linux64_gf"),
            )
            engine = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
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
                "OpenRadioss unavailable: missing " + ", ".join(missing) + ".\n"
                "Deck-only generation remains available; install with "
                "`python scripts/install_solvers.py --only radioss`."
            )
        return True, "OpenRadioss ready: Starter, Engine, and anim_to_vtk detected."

    def __init__(self, spawn_callback):
        super().__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)
        radioss_ready, radioss_status = self._openradioss_status()
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
                background: rgba(33, 150, 243, 60);
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
                    "Tube",
                    "com.cad.geometry.tube",
                    "Create a tube.",
                ),
                (
                    "Cylindrical Shell",
                    "com.cad.geometry.cylindrical_shell",
                    "Create a shell tube.",
                ),
                (
                    "Boolean",
                    "com.cad.geometry.boolean",
                    "Join, cut, or intersect solids.",
                ),
                (
                    "Through Hole",
                    "com.cad.geometry.through_hole",
                    "Cut a through-hole.",
                ),
                (
                    "Fillet",
                    "com.cad.geometry.fillet",
                    "Round selected edges.",
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
                    "Body",
                    "com.cad.sim.component",
                    "Pair a mesh and material for a multibody study.",
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
                    "Support",
                    "com.cad.sim.constraint",
                    "Shared support for static FEA and topology optimization. "
                    "Connect FE Mesh for FEA; Topology maps the selection internally.",
                ),
                (
                    "Force",
                    "com.cad.sim.load",
                    "Shared force for static FEA and topology optimization. "
                    "Connect FE Mesh for FEA; Topology maps the selection internally.",
                ),
                (
                    "Gravity",
                    "com.cad.sim.gravity",
                    "Add an FEA body acceleration.",
                ),
                (
                    "Pressure",
                    "com.cad.sim.pressure_load",
                    "Shared face pressure for static FEA and topology optimization.",
                ),
                (
                    "Impact Setup",
                    "com.cad.sim.impact",
                    "Set impact motion and contact.",
                ),
            ],
            "Solvers": [
                (
                    "Static FEA",
                    "com.cad.sim.solver",
                    "Run CalculiX.",
                ),
                (
                    "Topology Optimization",
                    "com.cad.sim.topopt_voxel",
                    "Optimize material for the selected objective and constraints. "
                    "Produces a solid envelope.",
                ),
                (
                    "Lattice Optimization",
                    "com.cad.sim.lattice_voxel",
                    "Optimize a graded cellular structure — gyroid, Schwarz "
                    "primitive, BCC, octet, or honeycomb — for the selected "
                    "objective and constraints.",
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
                    "Run OpenRadioss.",
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
                if node_id == "com.cad.sim.solver":
                    tooltip += "\n\n" + calculix_status
                    item.setData(0, QtCore.Qt.UserRole + 1, calculix_ready)
                    if not calculix_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
                elif node_id in (
                    "com.cad.sim.crash_solver",
                ):
                    tooltip += "\n\n" + radioss_status
                    item.setData(0, QtCore.Qt.UserRole + 1, radioss_ready)
                    if not radioss_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
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
