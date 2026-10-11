# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Interactive time-history and crashworthiness curves dialog."""

from __future__ import annotations

import csv
import logging
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = ["CrashPlotDialog"]

_DARK_BG = "#161922"
_AXES_BG = "#1e222e"
_TEXT_COLOR = "#eceff4"
_GRID_COLOR = "#2e3440"
_BORDER_COLOR = "#3b4252"


class CrashPlotDialog(QtWidgets.QDialog):
    """Commercial-grade explicit dynamics curve viewer.

    Provides:
    1. Force vs. Displacement (F-d) with Mean Force, Peak Force, and Energy area.
    2. Energy Balance (Kinetic, Internal, Total, Hourglass, Contact) vs. Time.
    3. Vehicle / Impactor Deceleration Pulse (g) vs. Time.
    4. CSV and PNG export capabilities.
    """

    def __init__(self, crash_result: dict[str, Any], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Crash Time-History & Crashworthiness Curves")
        self.resize(920, 640)
        self.setMinimumSize(750, 500)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {_DARK_BG}; color: {_TEXT_COLOR}; }}
            QTabWidget::pane {{ border: 1px solid {_BORDER_COLOR}; background-color: {_DARK_BG}; }}
            QTabBar::tab {{
                background: #232834; color: #9aa0a6; padding: 8px 18px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                font-weight: bold; font-size: 11px; margin-right: 2px;
            }}
            QTabBar::tab:selected {{ background: #1e5aab; color: #ffffff; }}
            QTabBar::tab:hover:!selected {{ background: #2c3240; color: #ecf0f1; }}
            QPushButton {{
                background: #2c3e50; color: white; border: 1px solid {_BORDER_COLOR};
                border-radius: 4px; padding: 6px 14px; font-weight: bold; font-size: 11px;
            }}
            QPushButton:hover {{ background: #1e5aab; }}
            QLabel {{ color: {_TEXT_COLOR}; font-size: 11px; }}
            """
        )

        self._result = crash_result or {}
        self._extract_data()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header with summary KPIs
        header_widget = self._build_header_widget()
        layout.addWidget(header_widget)

        # Tabs for curves
        self._tabs = QtWidgets.QTabWidget()
        self._tab_fd = QtWidgets.QWidget()
        self._tab_energy = QtWidgets.QWidget()
        self._tab_accel = QtWidgets.QWidget()

        self._init_fd_tab()
        self._init_energy_tab()
        self._init_accel_tab()

        self._tabs.addTab(self._tab_fd, "Force vs. Displacement (F-d)")
        self._tabs.addTab(self._tab_energy, "Energy Balance vs. Time")
        self._tabs.addTab(self._tab_accel, "Deceleration Pulse (a-t)")
        layout.addWidget(self._tabs, stretch=1)

        # Bottom toolbar with Exports and Close
        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.setSpacing(8)

        lbl_hint = QtWidgets.QLabel("Standard: SAE J211 / ISO 6487 zero-phase Butterworth filtering")
        lbl_hint.setStyleSheet("color: #718096; font-size: 10px;")
        btn_bar.addWidget(lbl_hint)
        btn_bar.addStretch(1)

        btn_export_csv = QtWidgets.QPushButton("Export CSV Data…")
        btn_export_csv.setToolTip("Export time history arrays to a spreadsheet CSV file")
        btn_export_csv.clicked.connect(self._export_csv)
        btn_bar.addWidget(btn_export_csv)

        btn_export_png = QtWidgets.QPushButton("Save Plot Image…")
        btn_export_png.setToolTip("Save the current curve view to high-resolution PNG")
        btn_export_png.clicked.connect(self._save_image)
        btn_bar.addWidget(btn_export_png)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(btn_close)

        layout.addLayout(btn_bar)

    def _extract_data(self):
        """Extract and align arrays from result dictionary."""
        histories = self._result.get("histories") or {}
        raw = histories.get("raw") or {}
        proc = histories.get("processed") or {}
        th = self._result.get("time_history") or {}
        metrics = self._result.get("crash_metrics") or {}

        # Time
        self._t = np.asarray(raw.get("time_ms") or th.get("t_ms") or th.get("time_ms") or [], dtype=float)

        # Stroke / Displacement
        self._d = np.asarray(proc.get("crush_displacement_mm") or raw.get("crush_displacement_mm") or [], dtype=float)

        # Force
        self._f_filt = np.asarray(proc.get("rigid_wall_force_kN") or raw.get("rigid_wall_force_kN") or [], dtype=float)
        self._f_raw = np.asarray(raw.get("rigid_wall_force_kN") or [], dtype=float)

        # Energies
        self._ke = np.asarray(raw.get("kinetic_energy_kj") or th.get("ke_kj") or th.get("kinetic_energy_kj") or [], dtype=float)
        self._ie = np.asarray(raw.get("internal_energy_kj") or th.get("ie_kj") or th.get("internal_energy_kj") or [], dtype=float)
        self._tot = np.asarray(raw.get("total_energy_kj") or th.get("tot_kj") or th.get("total_energy_kj") or [], dtype=float)
        self._hg = np.asarray(raw.get("hourglass_energy_kj") or th.get("he_kj") or th.get("hourglass_energy_kj") or [], dtype=float)
        self._ce = np.asarray(raw.get("contact_energy_kj") or th.get("ce_kj") or th.get("contact_energy_kj") or [], dtype=float)

        # Acceleration
        self._a_filt = np.asarray(proc.get("acceleration_g") or raw.get("acceleration_g") or [], dtype=float)
        self._a_raw = np.asarray(raw.get("acceleration_g") or [], dtype=float)

        # Metrics
        self._peak_force = float(self._result.get("peak_force") or metrics.get("peak_crushing_force_kN") or (np.max(self._f_filt) if self._f_filt.size else 0.0))
        self._mean_force = float(self._result.get("mean_force") or metrics.get("mean_crushing_force_kN") or 0.0)
        self._cfe = float(self._result.get("crush_force_efficiency") or metrics.get("crush_force_efficiency") or (self._mean_force / self._peak_force if self._peak_force > 0 else 0.0))
        self._sea = float(self._result.get("specific_energy_absorption") or metrics.get("specific_energy_absorption_kJ_kg") or 0.0)
        self._absorbed_energy = float(self._result.get("absorbed_energy_kj") or metrics.get("absorbed_energy_kJ") or 0.0)
        self._stroke = float(self._result.get("crush_distance") or metrics.get("useful_crush_stroke_mm") or (self._d[-1] if self._d.size else 0.0))
        self._peak_accel = float(self._result.get("peak_acceleration_g") or metrics.get("peak_acceleration_g") or (np.max(self._a_filt) if self._a_filt.size else 0.0))
        self._delta_v = float(self._result.get("delta_v") or metrics.get("delta_v_m_s") or 0.0)

    def _build_header_widget(self) -> QtWidgets.QWidget:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"background: #1e222e; border: 1px solid {_BORDER_COLOR}; border-radius: 6px; padding: 6px;"
        )
        hbox = QtWidgets.QHBoxLayout(frame)
        hbox.setContentsMargins(10, 4, 10, 4)
        hbox.setSpacing(20)

        # CFE
        cfe_pct = self._cfe * 100.0
        cfe_col = "#66d17a" if self._cfe >= 0.70 else "#4fc3f7" if self._cfe >= 0.50 else "#ffb74d"
        lbl_cfe = QtWidgets.QLabel(f"<b>CFE:</b> <span style='color: {cfe_col}; font-size: 13px;'>{cfe_pct:.1f}%</span>")
        lbl_cfe.setToolTip("Crash Force Efficiency (F_mean / F_peak). Ideal >= 70%")
        hbox.addWidget(lbl_cfe)

        # SEA
        lbl_sea = QtWidgets.QLabel(f"<b>SEA:</b> <span style='color: #4fc3f7; font-size: 13px;'>{self._sea:.2f} kJ/kg</span>")
        lbl_sea.setToolTip("Specific Energy Absorption per unit structural mass")
        hbox.addWidget(lbl_sea)

        # F_peak & F_mean
        lbl_f = QtWidgets.QLabel(f"<b>F<sub>peak</sub>:</b> {self._peak_force:.2f} kN &nbsp;|&nbsp; <b>F<sub>mean</sub>:</b> {self._mean_force:.2f} kN")
        hbox.addWidget(lbl_f)

        # Absorbed Energy
        lbl_e = QtWidgets.QLabel(f"<b>E<sub>abs</sub>:</b> {self._absorbed_energy:.2f} kJ &nbsp;|&nbsp; <b>Stroke:</b> {self._stroke:.1f} mm")
        hbox.addWidget(lbl_e)

        hbox.addStretch(1)
        return frame

    def _style_axes(self, ax):
        ax.set_facecolor(_AXES_BG)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(_BORDER_COLOR)
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.35, color=_GRID_COLOR)
        ax.title.set_color(_TEXT_COLOR)
        ax.title.set_fontsize(11)
        ax.title.set_weight("bold")
        ax.xaxis.label.set_color(_TEXT_COLOR)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_color(_TEXT_COLOR)
        ax.yaxis.label.set_fontsize(10)

    def _init_fd_tab(self):
        vbox = QtWidgets.QVBoxLayout(self._tab_fd)
        vbox.setContentsMargins(6, 6, 6, 6)

        fig = Figure(facecolor=_DARK_BG, tight_layout=True)
        canvas = FigureCanvas(fig)
        self._canvas_fd = canvas
        vbox.addWidget(canvas)

        ax = fig.add_subplot(111)
        self._style_axes(ax)

        ax.set_title("Crushing Force vs. Displacement (F-d)")
        ax.set_xlabel("Crush Displacement d [mm]")
        ax.set_ylabel("Crushing Force F [kN]")

        if self._d.size >= 2 and self._f_filt.size >= 2:
            n = min(self._d.size, self._f_filt.size)
            d = self._d[:n]
            f_filt = self._f_filt[:n]

            # Raw curve if distinct
            if self._f_raw.size >= n:
                f_raw = self._f_raw[:n]
                ax.plot(d, f_raw, label="Raw Force", color="#64748b", linestyle=":", linewidth=1.0, alpha=0.55)

            # Filtered solid curve
            ax.plot(d, f_filt, label="Filtered Force (CFC 600)", color="#00d2ff", linewidth=2.0)
            ax.fill_between(d, 0, f_filt, color="#00d2ff", alpha=0.15, label=f"Absorbed Energy ({self._absorbed_energy:.2f} kJ)")

            # Mean force horizontal line
            if self._mean_force > 0:
                ax.axhline(self._mean_force, color="#ffd600", linestyle="--", linewidth=1.5, label=f"Mean Force ({self._mean_force:.2f} kN)")

            # Peak force marker
            if self._peak_force > 0 and d.size > 0:
                idx_peak = int(np.argmax(f_filt))
                ax.scatter([d[idx_peak]], [self._peak_force], color="#ff5252", s=50, zorder=5, label=f"Peak Force ({self._peak_force:.2f} kN)")

            ax.set_xlim(left=0, right=max(float(d[-1]) * 1.05, 1.0))
            ax.set_ylim(bottom=0, top=max(self._peak_force * 1.15, 1.0))
            leg = ax.legend(facecolor="#232834", edgecolor=_BORDER_COLOR, labelcolor=_TEXT_COLOR, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No Force-Displacement channel available\n(Check simulation termination or rigid wall configuration)",
                    horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
                    color="#9aa0a6", fontsize=11)

    def _init_energy_tab(self):
        vbox = QtWidgets.QVBoxLayout(self._tab_energy)
        vbox.setContentsMargins(6, 6, 6, 6)

        fig = Figure(facecolor=_DARK_BG, tight_layout=True)
        canvas = FigureCanvas(fig)
        self._canvas_energy = canvas
        vbox.addWidget(canvas)

        ax = fig.add_subplot(111)
        self._style_axes(ax)

        ax.set_title("System Energy Conservation & Balance vs. Time")
        ax.set_xlabel("Time t [ms]")
        ax.set_ylabel("Energy E [kJ]")

        if self._t.size >= 2 and (self._ke.size >= 2 or self._ie.size >= 2):
            n = min(self._t.size, *(arr.size for arr in (self._ke, self._ie, self._tot, self._hg, self._ce) if arr.size >= 2))
            t = self._t[:n]

            if self._tot.size >= n:
                ax.plot(t, self._tot[:n], label="Total Energy (E_tot)", color="#00e676", linewidth=2.0)
            if self._ke.size >= n:
                ax.plot(t, self._ke[:n], label="Kinetic Energy (E_kin)", color="#00e5ff", linewidth=1.8)
            if self._ie.size >= n:
                ax.plot(t, self._ie[:n], label="Internal Energy (E_int)", color="#ff9100", linewidth=1.8)
            if self._hg.size >= n:
                ax.plot(t, self._hg[:n], label="Hourglass Energy (E_hg)", color="#f50057", linestyle="--", linewidth=1.4)
            if self._ce.size >= n:
                ax.plot(t, self._ce[:n], label="Contact Energy (E_cont)", color="#b388ff", linestyle=":", linewidth=1.4)

            ax.set_xlim(left=0, right=max(float(t[-1]), 0.1))
            ax.legend(facecolor="#232834", edgecolor=_BORDER_COLOR, labelcolor=_TEXT_COLOR, fontsize=9, loc="best")
        else:
            ax.text(0.5, 0.5, "No Energy channels available in time history",
                    horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
                    color="#9aa0a6", fontsize=11)

    def _init_accel_tab(self):
        vbox = QtWidgets.QVBoxLayout(self._tab_accel)
        vbox.setContentsMargins(6, 6, 6, 6)

        fig = Figure(facecolor=_DARK_BG, tight_layout=True)
        canvas = FigureCanvas(fig)
        self._canvas_accel = canvas
        vbox.addWidget(canvas)

        ax = fig.add_subplot(111)
        self._style_axes(ax)

        ax.set_title("Crash Deceleration Pulse vs. Time")
        ax.set_xlabel("Time t [ms]")
        ax.set_ylabel("Deceleration [g]")

        if self._t.size >= 2 and self._a_filt.size >= 2:
            n = min(self._t.size, self._a_filt.size)
            t = self._t[:n]
            a_filt = self._a_filt[:n]

            if self._a_raw.size >= n:
                ax.plot(t, self._a_raw[:n], label="Raw Deceleration", color="#64748b", linestyle=":", linewidth=1.0, alpha=0.55)

            ax.plot(t, a_filt, label=f"Filtered Pulse (Peak = {self._peak_accel:.1f} g)", color="#ff5252", linewidth=2.0)
            ax.axhline(0, color=_BORDER_COLOR, linewidth=0.8)

            ax.set_xlim(left=0, right=max(float(t[-1]), 0.1))
            ax.legend(facecolor="#232834", edgecolor=_BORDER_COLOR, labelcolor=_TEXT_COLOR, fontsize=9, loc="best")
        else:
            ax.text(0.5, 0.5, "No Acceleration Pulse available\n(Prescribed-speed platen or missing accelerometer channel)",
                    horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
                    color="#9aa0a6", fontsize=11)

    def _export_csv(self):
        """Export all time-history data to CSV."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Crash Time History CSV", "crash_time_history.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            n_samples = max(self._t.size, self._d.size, self._f_filt.size)
            if n_samples == 0:
                QtWidgets.QMessageBox.warning(self, "Export", "No time history samples to export.")
                return

            def _get(arr, idx):
                return f"{float(arr[idx]):.6g}" if idx < arr.size else ""

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time [ms]", "Crush Displacement [mm]", "Filtered Force [kN]", "Raw Force [kN]",
                    "Filtered Accel [g]", "Raw Accel [g]", "Kinetic Energy [kJ]",
                    "Internal Energy [kJ]", "Total Energy [kJ]", "Hourglass Energy [kJ]", "Contact Energy [kJ]"
                ])
                for i in range(n_samples):
                    writer.writerow([
                        _get(self._t, i),
                        _get(self._d, i),
                        _get(self._f_filt, i),
                        _get(self._f_raw, i),
                        _get(self._a_filt, i),
                        _get(self._a_raw, i),
                        _get(self._ke, i),
                        _get(self._ie, i),
                        _get(self._tot, i),
                        _get(self._hg, i),
                        _get(self._ce, i),
                    ])
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Successfully exported {n_samples} samples to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {exc}")

    def _save_image(self):
        """Save current active plot to PNG."""
        idx = self._tabs.currentIndex()
        canvas = self._canvas_fd if idx == 0 else self._canvas_energy if idx == 1 else self._canvas_accel
        name_hint = "crash_force_displacement.png" if idx == 0 else "crash_energy_balance.png" if idx == 1 else "crash_deceleration_pulse.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot Image", name_hint, "PNG Images (*.png);;PDF Files (*.pdf)"
        )
        if not path:
            return

        try:
            canvas.figure.savefig(path, dpi=300, facecolor=canvas.figure.get_facecolor(), edgecolor="none")
            QtWidgets.QMessageBox.information(self, "Saved", f"Plot image saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save image: {exc}")
