# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Detailed Simulation Results Viewer Dialog.

Provides an interactive popup window for comprehensive FEA, TopOpt, and Crash
results with:
1. Live 3D Model Display selector (switches VTK viewer in real time).
2. Data Tables (Components, Assembly Ties, Reactions, Load Cases, Mesh Quality, KPIs).
3. Graph Results (Convergence History, SAE J211 Crash Curves, Energy Balance).
4. Run summary and timing.
5. CSV Export.
"""

from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from .result_presentation import (
    analysis_label, crash_check_percent, fea_result, result_metric_sections,
    result_notices, result_state_rows, run_status,
)

try:
    import pyqtgraph as pg
    _HAVE_PYQTGRAPH = True
except Exception:
    _HAVE_PYQTGRAPH = False

logger = logging.getLogger(__name__)

__all__ = ["DetailedResultsDialog", "DetailedResultsWidget"]

_DIALOG_DARK_QSS = """
    QDialog, DetailedResultsWidget {
        background-color: #1e1f22;
        color: #e2e8f0;
        font-family: "Segoe UI";
    }
    #ResultsTopBar {
        background-color: #1e1f22;
        border: 1px solid #282a2e;
        border-radius: 6px;
    }
    QTabWidget::pane {
        border: 1px solid #282a2e;
        background-color: #1e1f22;
        border-radius: 6px;
    }
    QTabBar::tab {
        background: #181b22;
        color: #94a3b8;
        padding: 8px 22px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        font-size: 11px;
        margin-right: 4px;
        border: 1px solid #282a2e;
        border-bottom: none;
    }
    QTabBar::tab:selected {
        background: #1e1f22;
        color: #ffffff;
        border-color: #282a2e;
        border-bottom: 2px solid #d29922;
    }
    QTabBar::tab:hover:!selected {
        background: #282a2e;
        color: #e2e8f0;
    }
    #SubTabs QTabBar::tab {
        background: #282a2e;
        color: #94a3b8;
        padding: 5px 14px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        margin-right: 4px;
        border: 1px solid #383c44;
    }
    #SubTabs QTabBar::tab:selected {
        background: #383c44;
        color: #ffffff;
        border-color: #d29922;
        font-weight: 600;
    }
    #SubTabs QTabBar::tab:hover:!selected {
        background: #333740;
        color: #f1f5f9;
    }
    QTableWidget, QTableView {
        background-color: #16181d;
        alternate-background-color: #1e2127;
        gridline-color: #282c35;
        border: 1px solid #282c35;
        border-radius: 6px;
        color: #e2e8f0;
        font-size: 11px;
        selection-background-color: #2e3846;
        selection-color: #ffffff;
    }
    QTableWidget::item {
        color: #e2e8f0;
        padding: 6px 10px;
        border: none;
    }
    QTableWidget::item:selected {
        background-color: #2e3846;
        color: #ffffff;
    }
    QHeaderView::section {
        background-color: #21252d;
        color: #8da4c4;
        padding: 7px 10px;
        border: none;
        border-right: 1px solid #282c35;
        border-bottom: 1px solid #282c35;
        font-weight: 600;
        font-size: 11px;
    }
    QGroupBox {
        background: #24272d;
        border: 1px solid #2f333a;
        border-radius: 6px;
        margin-top: 13px;
        padding: 10px 10px 10px 10px;
        font-size: 11px;
        font-weight: 600;
        color: #e2e8f0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 4px;
        color: #d1d5db;
        font-weight: 700;
    }
    QPushButton {
        background: #282a2e;
        color: #e2e8f0;
        border: 1px solid #383c44;
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 11px;
    }
    QPushButton:hover {
        background: #333740;
        border-color: #4b515d;
        color: #ffffff;
    }
    QPushButton:pressed {
        background: #1e2124;
    }
    QPushButton:checked {
        background: #383c44;
        color: #ffffff;
        border-color: #d29922;
    }
    QComboBox {
        background: #282a2e;
        color: #e2e8f0;
        border: 1px solid #383c44;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        min-width: 140px;
    }
    QComboBox QAbstractItemView {
        background: #24272d;
        color: #e2e8f0;
        selection-background-color: #383c44;
        border: 1px solid #383c44;
    }
    QLabel {
        color: #94a3b8;
        font-size: 11px;
    }
    QCheckBox {
        color: #cbd5e1;
        font-size: 11px;
    }
    QScrollBar:vertical {
        background: #181b22;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #333a46;
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #475161;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
"""

_DIALOG_LIGHT_QSS = """
    QDialog, DetailedResultsWidget {
        background-color: #f4f6f8;
        color: #344054;
        font-family: "Segoe UI";
    }
    #ResultsTopBar {
        background-color: #ffffff;
        border: 1px solid #d5dce5;
        border-radius: 6px;
    }
    QTabWidget::pane {
        border: 1px solid #d5dce5;
        background-color: #ffffff;
        border-radius: 6px;
    }
    QTabBar::tab {
        background: #f4f6f8;
        color: #64748b;
        padding: 8px 22px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        font-size: 11px;
        margin-right: 4px;
        border: 1px solid #d5dce5;
        border-bottom: none;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #344054;
        border-color: #d5dce5;
        border-bottom: 2px solid #4f6f91;
    }
    QTabBar::tab:hover:!selected {
        background: #eef2f6;
        color: #1e293b;
    }
    #SubTabs QTabBar::tab {
        background: #f1f5f9;
        color: #64748b;
        padding: 5px 14px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 11px;
        margin-right: 4px;
        border: 1px solid #e2e8f0;
    }
    #SubTabs QTabBar::tab:selected {
        background: #ffffff;
        color: #4f6f91;
        border-color: #4f6f91;
        font-weight: 600;
    }
    #SubTabs QTabBar::tab:hover:!selected {
        background: #e2e8f0;
        color: #0f172a;
    }
    QTableWidget, QTableView {
        background-color: #ffffff;
        alternate-background-color: #f8fafc;
        gridline-color: #e2e8f0;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        color: #344054;
        font-size: 11px;
        selection-background-color: #e0e7f1;
        selection-color: #1e293b;
    }
    QTableWidget::item {
        color: #344054;
        padding: 6px 10px;
        border: none;
    }
    QTableWidget::item:selected {
        background-color: #e0e7f1;
        color: #1e293b;
    }
    QHeaderView::section {
        background-color: #f1f5f9;
        color: #475569;
        padding: 7px 10px;
        border: none;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 600;
        font-size: 11px;
    }
    QGroupBox {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin-top: 13px;
        padding: 10px 10px 10px 10px;
        font-size: 11px;
        font-weight: 600;
        color: #344054;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 4px;
        color: #344054;
        font-weight: 700;
    }
    QPushButton {
        background: #ffffff;
        color: #344054;
        border: 1px solid #d5dce5;
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 11px;
    }
    QPushButton:hover {
        background: #f1f5f9;
        border-color: #94a3b8;
        color: #0f172a;
    }
    QPushButton:pressed {
        background: #cbd5e1;
    }
    QPushButton:checked {
        background: #eef2f6;
        color: #4f6f91;
        border-color: #4f6f91;
    }
    QComboBox {
        background: #ffffff;
        color: #344054;
        border: 1px solid #d5dce5;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        min-width: 140px;
    }
    QComboBox QAbstractItemView {
        background: #ffffff;
        color: #344054;
        selection-background-color: #e0e7f1;
        selection-color: #1e293b;
        border: 1px solid #d5dce5;
    }
    QLabel {
        color: #64748b;
        font-size: 11px;
    }
    QCheckBox {
        color: #334155;
        font-size: 11px;
    }
    QScrollBar:vertical {
        background: #f8fafc;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #cbd5e1;
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #94a3b8;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
"""

_DIALOG_QSS = _DIALOG_DARK_QSS


def _fmt(value: Any, unit: str = "") -> str:
    """Format numeric or string metrics consistently."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "—"
    if v == 0.0:
        return f"0 {unit}".strip()
    if abs(v) >= 1e3 or abs(v) < 1e-2:
        return f"{v:.3e} {unit}".strip()
    return f"{v:.4g} {unit}".strip()


class DetailedResultsWidget(QtWidgets.QWidget):
    """Widget providing data tables, graphs, run summary, and 3D mode controls."""

    view_mode_requested = QtCore.Signal(str)
    undeformed_requested = QtCore.Signal(bool)
    extrema_requested = QtCore.Signal(bool)
    deform_scale_requested = QtCore.Signal(str)
    switch_to_setup_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._theme_name = "dark"
        self.setStyleSheet(_DIALOG_DARK_QSS)

        self._data: Dict[str, Any] = {}
        self._measurements: List[Tuple[str, str]] = []
        self._current_mode: str = ""
        self._updating_controls: bool = False

        self._init_ui()
        from pylcss.user_interface.common.theme_manager import current_theme
        self.apply_theme(current_theme())

    def _is_dark(self) -> bool:
        return getattr(self, "_theme_name", "dark") != "light"

    def apply_theme(self, theme_name: str):
        """Update widget stylesheets and plots when theme changes."""
        self._theme_name = str(theme_name).lower()
        is_dark = self._theme_name == "dark"
        self.setStyleSheet(_DIALOG_DARK_QSS if is_dark else _DIALOG_LIGHT_QSS)
        background = "#1e1f22" if is_dark else "#f4f6f8"
        foreground = "#e2e8f0" if is_dark else "#344054"
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(background))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(foreground))
        self.setPalette(palette)
        self._btn_go_setup.setStyleSheet(
            f"background: {'#d29922' if is_dark else '#4f6f91'}; color: {'#1e1f22' if is_dark else '#ffffff'}; padding: 6px 16px; border-radius: 4px; font-weight: bold; font-size: 11px;"
        )
        for widget in (self._tab_tables, self._tab_graphs, self._tab_summary, self._summary_content):
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)
        if hasattr(self, "_badge_type"):
            self._style_type_badge()
        if hasattr(self, "_results_container") and not self._results_container.isHidden():
            selected_tables = self._subtabs_tables.tabText(self._subtabs_tables.currentIndex())
            selected_graphs = self._subtabs_graphs.tabText(self._subtabs_graphs.currentIndex())
            self._populate_tables()
            self._populate_graphs()
            self._populate_summary()
            for tabs, title in ((self._subtabs_tables, selected_tables), (self._subtabs_graphs, selected_graphs)):
                for index in range(tabs.count()):
                    if tabs.tabText(index) == title:
                        tabs.setCurrentIndex(index)
                        break

    def _style_type_badge(self):
        tones = {
            "fea": ("#7a9bbf", "#3d5c7a"),
            "modal": ("#b3a1cb", "#70548f"),
            "topopt_voxel": ("#8bb699", "#3f7d5a"),
            "crash": ("#d7a08d", "#a05c48"),
        }
        pair = tones.get(self._data.get("type"), ("#d29922", "#4f6f91"))
        accent = pair[0] if self._is_dark() else pair[1]
        background = "#282a2e" if self._is_dark() else "#eef2f6"
        self._badge_type.setStyleSheet(f"background: {background}; color: {accent}; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;")

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # Empty state container
        self._empty_container = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self._empty_container)
        empty_layout.setAlignment(QtCore.Qt.AlignCenter)
        empty_layout.setSpacing(12)

        lbl_empty_title = QtWidgets.QLabel("No Results Yet")
        lbl_empty_title.setStyleSheet("font-weight: bold; font-size: 15px;")
        lbl_empty_title.setAlignment(QtCore.Qt.AlignCenter)

        lbl_empty_desc = QtWidgets.QLabel(
            "Run an FEA, Modal, Impact, or Topology Optimization study in Setup,\n"
            "or generate a lattice to inspect its geometry and available analysis here."
        )
        lbl_empty_desc.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_empty_desc.setAlignment(QtCore.Qt.AlignCenter)

        btn_go_setup = QtWidgets.QPushButton("Switch to Setup")
        self._btn_go_setup = btn_go_setup
        btn_go_setup.setStyleSheet(
            "background: #d29922; color: #ffffff; padding: 6px 16px; border-radius: 4px; font-weight: bold; font-size: 11px;"
        )
        btn_go_setup.clicked.connect(self.switch_to_setup_requested.emit)

        empty_layout.addStretch(1)
        empty_layout.addWidget(lbl_empty_title)
        empty_layout.addWidget(lbl_empty_desc)
        empty_layout.addWidget(btn_go_setup, alignment=QtCore.Qt.AlignCenter)
        empty_layout.addStretch(1)

        # Populated results container
        self._results_container = QtWidgets.QWidget()
        res_layout = QtWidgets.QVBoxLayout(self._results_container)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(10)

        # ── 1. Top 3D Control Bar ──────────────────────────────────────────
        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("ResultsTopBar")
        bar_layout = QtWidgets.QVBoxLayout(top_bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)
        title_layout = QtWidgets.QHBoxLayout()
        bar_layout.addLayout(title_layout)
        top_layout = QtWidgets.QHBoxLayout()
        bar_layout.addLayout(top_layout)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # Title & Badge
        self._lbl_title = QtWidgets.QLabel("Simulation Results")
        self._lbl_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        title_layout.addWidget(self._lbl_title)

        self._badge_type = QtWidgets.QLabel("FEA")
        self._badge_type.setStyleSheet(
            "background: #2e2710; color: #d29922; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;"
        )
        title_layout.addWidget(self._badge_type)
        title_layout.addStretch(1)

        # 3D View Selector
        lbl_3d = QtWidgets.QLabel("3D Field:")
        lbl_3d.setStyleSheet("font-weight: 600; font-size: 11px;")
        top_layout.addWidget(lbl_3d)

        self._3d_mode_combo = QtWidgets.QComboBox()
        self._3d_mode_combo.setObjectName("Results3DModeCombo")
        self._3d_mode_combo.setMinimumWidth(180)
        self._3d_mode_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._3d_mode_combo.currentTextChanged.connect(self._on_3d_combo_changed)
        top_layout.addWidget(self._3d_mode_combo)

        # Hidden button group for compatibility with existing tests
        self._btn_group_3d = QtWidgets.QButtonGroup(self)
        self._btn_group_3d.setExclusive(True)
        self._btn_group_3d.buttonClicked.connect(self._on_3d_btn_clicked)

        self._3d_buttons_widget = QtWidgets.QWidget()
        self._3d_buttons_widget.setVisible(False)
        self._3d_buttons_layout = QtWidgets.QHBoxLayout(self._3d_buttons_widget)
        self._3d_buttons_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self._3d_buttons_widget)

        self._field_controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(self._field_controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        bar_layout.addWidget(self._field_controls_widget)

        # Undeformed overlay toggle
        self._chk_undeformed = QtWidgets.QCheckBox("Undeformed")
        self._chk_undeformed.setToolTip("Show the undeformed analysis mesh as a wireframe")
        self._chk_undeformed.toggled.connect(self.undeformed_requested.emit)
        controls_layout.addWidget(self._chk_undeformed)

        # Deformation scale combo
        self._lbl_scale = QtWidgets.QLabel("Scale:")
        self._scale_combo = QtWidgets.QComboBox()
        self._scale_combo.setMinimumWidth(75)
        self._scale_combo.addItems(["Auto", "True (1x)", "2x", "5x", "10x", "25x", "50x", "100x"])
        self._scale_combo.currentTextChanged.connect(self.deform_scale_requested.emit)
        controls_layout.addWidget(self._lbl_scale)
        controls_layout.addWidget(self._scale_combo)
        self._chk_extrema = QtWidgets.QCheckBox("Min / Max")
        self._chk_extrema.toggled.connect(self.extrema_requested.emit)
        controls_layout.addWidget(self._chk_extrema)
        controls_layout.addStretch(1)

        res_layout.addWidget(top_bar)

        # ── 2. Tab Widget (Tables, Graphs, Summary) ────────────────────────
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setObjectName("MainResultsTabs")
        self._tab_tables = QtWidgets.QWidget()
        self._tab_graphs = QtWidgets.QWidget()
        self._tab_summary = QtWidgets.QWidget()

        self._init_tables_tab()
        self._init_graphs_tab()
        self._init_summary_tab()

        self._tabs.addTab(self._tab_summary, "Overview")
        self._tabs.addTab(self._tab_tables, "Details")
        self._tabs.addTab(self._tab_graphs, "Trends")
        res_layout.addWidget(self._tabs, stretch=1)

        # ── 3. Bottom Action Bar ───────────────────────────────────────────
        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.setContentsMargins(0, 4, 0, 0)
        bottom_bar.setSpacing(10)

        self._btn_export_csv = QtWidgets.QPushButton("Export Tables to CSV...")
        self._btn_export_csv.setToolTip("Export all result tables to CSV spreadsheet files")
        self._btn_export_csv.clicked.connect(self._export_csv)
        try:
            import qtawesome as qta
            self._btn_export_csv.setIcon(qta.icon("fa5s.file-csv", color="#8b949e"))
        except Exception:
            pass
        bottom_bar.addWidget(self._btn_export_csv)

        self._btn_copy_summary = QtWidgets.QPushButton("Copy Summary")
        self._btn_copy_summary.setToolTip("Copy run metrics summary to clipboard")
        self._btn_copy_summary.clicked.connect(self._copy_summary)
        try:
            import qtawesome as qta
            self._btn_copy_summary.setIcon(qta.icon("fa5s.copy", color="#94a3b8"))
        except Exception:
            pass
        bottom_bar.addWidget(self._btn_copy_summary)

        bottom_bar.addStretch(1)

        # Retained hidden for backward compatibility
        self._btn_return_setup = QtWidgets.QPushButton("Return to Setup", self)
        self._btn_return_setup.setVisible(False)
        self._btn_return_setup.clicked.connect(self.switch_to_setup_requested.emit)

        res_layout.addLayout(bottom_bar)

        main_layout.addWidget(self._empty_container)
        main_layout.addWidget(self._results_container)

        self._empty_container.setVisible(True)
        self._results_container.setVisible(False)

    # ──────────────────────────────────────────────────────────────────
    # Tables Tab
    # ──────────────────────────────────────────────────────────────────
    def _init_tables_tab(self):
        layout = QtWidgets.QVBoxLayout(self._tab_tables)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._subtabs_tables = QtWidgets.QTabWidget()
        self._subtabs_tables.setObjectName("SubTabs")
        layout.addWidget(self._subtabs_tables)

    def _init_graphs_tab(self):
        layout = QtWidgets.QVBoxLayout(self._tab_graphs)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._subtabs_graphs = QtWidgets.QTabWidget()
        self._subtabs_graphs.setObjectName("SubTabs")
        layout.addWidget(self._subtabs_graphs)

        self._no_graphs_label = QtWidgets.QLabel("No graph histories available for this result type.")
        self._no_graphs_label.setAlignment(QtCore.Qt.AlignCenter)
        self._no_graphs_label.setStyleSheet("color: #8f98a5; font-style: italic; font-size: 12px;")
        layout.addWidget(self._no_graphs_label)

    def _init_summary_tab(self):
        layout = QtWidgets.QVBoxLayout(self._tab_summary)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._summary_content = QtWidgets.QWidget()
        self._summary_content.setMaximumWidth(1120)
        self._summary_layout = QtWidgets.QVBoxLayout(self._summary_content)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(8)
        scroll.setWidget(self._summary_content)
        scroll.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(scroll)

    # ──────────────────────────────────────────────────────────────────
    # Helper to Build Table Widgets
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def _create_table(
        cls,
        columns: List[str],
        rows: List[List[Any]],
        is_dark: bool = True,
    ) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        table.setAlternatingRowColors(True)
        table.setShowGrid(True)
        table.setFrameShape(QtWidgets.QFrame.NoFrame)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(30)

        # Force dark or light palette on table and its viewport to prevent Windows system light fallbacks
        bg_base = QtGui.QColor("#16181d" if is_dark else "#ffffff")
        bg_alt = QtGui.QColor("#1e2127" if is_dark else "#f8fafc")
        text_color = QtGui.QColor("#e2e8f0" if is_dark else "#1e293b")
        grid_color = QtGui.QColor("#282c35" if is_dark else "#e2e8f0")
        hdr_bg = "#21252d" if is_dark else "#f1f5f9"
        hdr_text = "#8da4c4" if is_dark else "#475569"

        pal = table.palette()
        pal.setColor(QtGui.QPalette.Base, bg_base)
        pal.setColor(QtGui.QPalette.AlternateBase, bg_alt)
        pal.setColor(QtGui.QPalette.Window, bg_base)
        pal.setColor(QtGui.QPalette.Text, text_color)
        pal.setColor(QtGui.QPalette.WindowText, text_color)
        selection = "#2e3846" if is_dark else "#dde7f0"
        selection_text = "#ffffff" if is_dark else "#344054"
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(selection))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(selection_text))
        table.setPalette(pal)
        table.viewport().setPalette(pal)

        table.setStyleSheet(
            f"QTableWidget {{ background-color: {bg_base.name()}; alternate-background-color: {bg_alt.name()}; "
            f"color: {text_color.name()}; gridline-color: {grid_color.name()}; border: 1px solid {grid_color.name()}; border-radius: 6px; }} "
            f"QTableWidget::item {{ color: {text_color.name()}; padding: 6px 10px; border: none; }} "
            f"QTableWidget::item:selected {{ background-color: {selection}; color: {selection_text}; }} "
            f"QHeaderView::section {{ background-color: {hdr_bg}; color: {hdr_text}; padding: 7px 10px; "
            f"border: none; border-bottom: 1px solid {grid_color.name()}; border-right: 1px solid {grid_color.name()}; font-weight: 600; font-size: 11px; }} "
            f"QScrollBar:vertical {{ background: {bg_base.name()}; width: 8px; margin: 0; }} "
            f"QScrollBar::handle:vertical {{ background: #333a46; border-radius: 4px; min-height: 20px; }} "
            f"QScrollBar::handle:vertical:hover {{ background: #475161; }} "
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
            f"QScrollBar:horizontal {{ background: {bg_base.name()}; height: 8px; margin: 0; }}"
            f"QScrollBar::handle:horizontal {{ background: {hdr_text}; border-radius: 4px; min-width: 20px; }}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        )
        table.viewport().setStyleSheet(f"background-color: {bg_base.name()};")

        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setToolTip(str(val))
                item.setForeground(QtGui.QBrush(text_color))
                if c_idx > 0 and not (columns[-1] == "Meaning" and c_idx == len(columns) - 1) and any(char.isdigit() for char in str(val)):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                table.setItem(r_idx, c_idx, item)

        table.resizeColumnsToContents()
        for c in range(len(columns)):
            w = table.columnWidth(c)
            table.setColumnWidth(c, max(w + 24, 120))
        table.horizontalHeader().setStretchLastSection(True)
        if columns[-1] == "Meaning":
            table.setColumnWidth(0, 220)
            table.setColumnWidth(1, 140)
            table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        return table

    # ──────────────────────────────────────────────────────────────────
    # Data Population
    # ──────────────────────────────────────────────────────────────────
    def update_result(self, data: Optional[Dict[str, Any]], measurements: Optional[List[Tuple[str, str]]] = None):
        """Update tables, charts, and 3D mode selectors from result dict."""
        self._data = dict(data) if isinstance(data, dict) else {}
        self._measurements = list(measurements or [])

        if not self._data:
            self._empty_container.setVisible(True)
            self._results_container.setVisible(False)
            return

        self._empty_container.setVisible(False)
        self._results_container.setVisible(True)

        rtype = self._data.get("type", "unknown").lower()
        title = analysis_label(self._data)
        self._lbl_title.setText(f"{title} Results")
        badge = ("Lattice" if "Lattice" in title else "Topology") if rtype == "topopt_voxel" else {"external_solver": "FEA"}.get(rtype, rtype.upper())
        if rtype == "topopt_voxel" and fea_result(self._data):
            badge += " + FEA"
        self._badge_type.setText(badge)
        self._style_type_badge()

        self._populate_3d_controls()
        self._populate_tables()
        self._populate_graphs()
        self._populate_summary()

    def set_active_3d_mode(self, mode_name: str):
        """Update active 3D selector and button state without re-triggering signal."""
        self._current_mode = str(mode_name)
        self._data["visualization_mode"] = str(mode_name)
        self._updating_controls = True
        try:
            if hasattr(self, "_3d_mode_combo"):
                idx = self._3d_mode_combo.findText(mode_name)
                if idx >= 0:
                    self._3d_mode_combo.setCurrentIndex(idx)
            for btn in self._btn_group_3d.buttons():
                if btn.text() == mode_name:
                    btn.setChecked(True)
                    break
        finally:
            self._updating_controls = False

        self._update_field_controls(str(mode_name))

    def select_tab(self, tab_name: str):
        """Accept both the current tab names and legacy action names."""
        widgets = {"tables": self._tab_tables, "details": self._tab_tables,
                   "graphs": self._tab_graphs, "trends": self._tab_graphs}
        widget = widgets.get(str(tab_name).lower(), self._tab_summary)
        index = self._tabs.indexOf(widget)
        self._tabs.setCurrentWidget(widget if self._tabs.isTabVisible(index) else self._tab_summary)

    # ── 3D Controls ───────────────────────────────────────────────────
    def _get_available_3d_modes(self) -> List[str]:
        from .result_fields import available_result_fields
        return available_result_fields(self._data)

    def _populate_3d_controls(self):
        # Clear existing buttons
        while self._3d_buttons_layout.count():
            item = self._3d_buttons_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                self._btn_group_3d.removeButton(w)
                w.deleteLater()

        modes = self._get_available_3d_modes()
        current = str(self._data.get("visualization_mode") or modes[0])
        if current not in modes:
            current = modes[0]
        self._current_mode = current

        self._updating_controls = True
        try:
            self._3d_mode_combo.blockSignals(True)
            self._3d_mode_combo.clear()
            self._3d_mode_combo.addItems(modes)
            idx = self._3d_mode_combo.findText(current)
            if idx >= 0:
                self._3d_mode_combo.setCurrentIndex(idx)
            self._3d_mode_combo.blockSignals(False)

            for mode in modes:
                btn = QtWidgets.QPushButton(mode)
                btn.setCheckable(True)
                if mode == current:
                    btn.setChecked(True)
                self._btn_group_3d.addButton(btn)
                self._3d_buttons_layout.addWidget(btn)
        finally:
            self._updating_controls = False

        self._update_field_controls(current)

    def _update_field_controls(self, mode):
        display_data = fea_result(self._data) if mode.startswith("FEA ") else self._data
        has_disp = display_data.get("displacement") is not None
        self._lbl_scale.setVisible(has_disp)
        self._scale_combo.setVisible(has_disp)
        self._chk_undeformed.setVisible(has_disp)
        self._chk_extrema.setVisible(mode not in {"Default View", "CAD", "Manufactured Mesh", "Deformed Animation"})
        self._field_controls_widget.setVisible(has_disp or not self._chk_extrema.isHidden())

    def _on_3d_combo_changed(self, mode: str):
        if self._updating_controls or not mode:
            return
        self._current_mode = mode
        self._data["visualization_mode"] = mode
        self._update_field_controls(mode)
        self._updating_controls = True
        try:
            for btn in self._btn_group_3d.buttons():
                if btn.text() == mode:
                    btn.setChecked(True)
                    break
        finally:
            self._updating_controls = False
        self.view_mode_requested.emit(mode)

    def _on_3d_btn_clicked(self, button: QtWidgets.QAbstractButton):
        if self._updating_controls:
            return
        mode = button.text()
        self._current_mode = mode
        self._data["visualization_mode"] = mode
        self._update_field_controls(mode)
        self._updating_controls = True
        try:
            idx = self._3d_mode_combo.findText(mode)
            if idx >= 0:
                self._3d_mode_combo.setCurrentIndex(idx)
        finally:
            self._updating_controls = False
        self.view_mode_requested.emit(mode)

    # ── Tables Tab Population ─────────────────────────────────────────
    def _populate_tables(self):
        while self._subtabs_tables.count():
            w = self._subtabs_tables.widget(0)
            self._subtabs_tables.removeTab(0)
            if w is not None:
                w.deleteLater()

        data = self._data
        rtype = data.get("type", "")

        if rtype == "modal" and data.get("modes"):
            rows = [[m["mode"], _fmt(m.get("frequency_hz")), _fmt(m.get("frequency_rad_s")), _fmt(m.get("period_s"))] for m in data["modes"]]
            self._subtabs_tables.addTab(self._create_table(["Mode", "Frequency (Hz)", "Angular Frequency (rad/s)", "Period (s)"], rows, is_dark=self._is_dark()), "Natural Frequencies")

        for title, rows in result_metric_sections(data):
            self._subtabs_tables.addTab(
                self._create_table(["Result", "Value", "Meaning"], rows, is_dark=self._is_dark()), title
            )

        analysis = fea_result(data)
        detail_data = analysis or data
        levels = (analysis.get("convergence_study") or {}).get("levels") or []
        if levels:
            rows = [[level.get("refine", "—"), level.get("n_nodes", "—"), level.get("n_elements", "—"),
                     _fmt(level.get("compliance"), "N mm"), _fmt(level.get("peak_von_mises"), "MPa")]
                    for level in levels if isinstance(level, dict)]
            self._subtabs_tables.addTab(self._create_table(
                ["Refinement", "Nodes", "Elements", "Compliance", "Peak stress"], rows, is_dark=self._is_dark()
            ), "FEA mesh refinement")

        # 1. Components Table (FEA multi-body)
        components = detail_data.get("components")
        if isinstance(components, list) and components:
            cols = ["Component Name", "Material", "Elements", "Peak Stress (MPa)", "Yield FoS"]
            rows = []
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                name = str(comp.get("name", "Part"))
                mat = str(comp.get("material", "Steel"))
                elem = f"{int(comp.get('elements', 0)):,}" if comp.get("elements") is not None else "—"
                stress = _fmt(comp.get("peak_stress_mpa"), "MPa")
                fos_val = comp.get("yield_safety_factor")
                fos_str = _fmt(fos_val)
                rows.append([name, mat, elem, stress, fos_str])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Components")

        # 2. Assembly Interfaces Table (Tied contact)
        connections = detail_data.get("connections")
        if isinstance(connections, list) and connections:
            cols = ["Component Pair", "Connection Type", "Search Tolerance", "Interface Status"]
            rows = []
            for conn in connections:
                if not isinstance(conn, dict):
                    continue
                pair = f"{conn.get('component_a', '?')} ↔ {conn.get('component_b', '?')}"
                tol = _fmt(conn.get("search_tolerance_mm"), "mm")
                rows.append([pair, "Bonded", tol, "Active"])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Assembly Ties")

        # 3. Load Cases Table (TopOpt Multi-load & Cases)
        cases = [c for c in data.get("structural_cases") or [] if isinstance(c, dict)]
        if len(cases) > 1:
            cols = ["Case", "Discipline", "Compliance", "Peak displacement"]
            rows = []
            for c in (cases or []):
                if isinstance(c, dict):
                    name = str(c.get("name", "Structural Case"))
                    comp = _fmt(c.get("compliance"), "N mm")
                    disp = _fmt(c.get("maximum_displacement"), "mm")
                    rows.append([name, "Structural", comp, disp])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Case comparison")

        # 4. Reaction Forces Table
        rx = detail_data.get("reaction_force")
        if rx is not None:
            try:
                rx_arr = np.asarray(rx, dtype=float).reshape(-1)
                if rx_arr.size >= 3:
                    cols = ["Support Component / Axis", "Reaction Force (N)", "Direction"]
                    total = float(np.linalg.norm(rx_arr[:3]))
                    rows = [
                        ["Support Reaction FX", _fmt(rx_arr[0], "N"), "Global X"],
                        ["Support Reaction FY", _fmt(rx_arr[1], "N"), "Global Y"],
                        ["Support Reaction FZ", _fmt(rx_arr[2], "N"), "Global Z"],
                        ["Resultant Magnitude", _fmt(total, "N"), "Resultant Total"],
                    ]
                    self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Reaction Forces")
            except Exception:
                pass

        # 5. Mesh Quality Metrics Table
        quality = detail_data.get("mesh_quality")
        mesh_obj = detail_data.get("mesh")
        if quality is None and mesh_obj is not None:
            quality = getattr(mesh_obj, "quality_report", None)
        if isinstance(quality, dict):
            cols = ["Mesh Metric", "Value"]
            rows = [
                ["Minimum Mean-Ratio", _fmt(quality.get("min_mean_ratio"))],
                ["5th Percentile Mean-Ratio", _fmt(quality.get("p05_mean_ratio"))],
                ["Average Mean-Ratio", _fmt(quality.get("mean_mean_ratio"))],
                ["Maximum Edge Ratio", _fmt(quality.get("max_edge_ratio"))],
                ["Collapsed Elements", _fmt(quality.get("degenerate_elements"))],
            ]
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Mesh Quality")

        # 6. Crash SAE J211 KPIs Table
        if rtype == "crash":
            cols = ["Crash Metric", "Value", "Unit"]
            rows = []
            for k, lbl, u in [
                ("peak_force", "Peak Crushing Force", "kN"),
                ("mean_force", "Mean Crushing Force", "kN"),
                ("crush_force_efficiency", "Crush-Force Efficiency (CFE)", ""),
                ("specific_energy_absorption", "Specific Energy Absorption (SEA)", "kJ/kg"),
                ("structural_mass_kg", "Deformable Structure Mass", "kg"),
                ("crush_distance", "Useful Crush Distance", "mm"),
                ("peak_acceleration_g", "Peak Deceleration Pulse", "g"),
                ("delta_v", "Impact Velocity Change (Δv)", "m/s"),
                ("absorbed_energy_kj", "Total Absorbed Energy", "kJ"),
                ("n_failed", "Failed Elements Count", "elements"),
            ]:
                if data.get(k) is not None:
                    rows.append([lbl, _fmt(data[k]), u])
            if rows:
                self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Crash KPIs")

        # 6b. Crash Solution Quality & Energy Gates Table
        if rtype == "crash":
            quality = data.get("quality") or {}
            checks = {c.get("id"): c for c in quality.get("checks", []) if isinstance(c, dict)}
            q_cols = ["Numerical Check", "Value", "Status"]
            q_rows = []

            eb_chk = checks.get("energy_balance", {})
            eb_val = crash_check_percent(data, "energy_balance", "energy_balance_max_error")
            if eb_val is not None:
                val_num = float(eb_val)
                q_rows.append(["Energy Balance Error", f"{val_num:.2f}%", str(eb_chk.get("status") or "—").title()])

            hg_chk = checks.get("hourglass_energy", {})
            hg_val = hg_chk.get("value")
            if hg_val is not None:
                val_num = float(hg_val)
                q_rows.append(["Hourglass Energy Ratio", f"{val_num:.2f}%", str(hg_chk.get("status") or "—").title()])

            art_chk = checks.get("hourglass_plus_contact_energy", {})
            art_val = art_chk.get("value")
            if art_val is not None:
                val_num = float(art_val)
                q_rows.append(["Hourglass + Contact Energy", f"{val_num:.2f}%", str(art_chk.get("status") or "—").title()])

            am_chk = checks.get("added_mass", {})
            am_val = crash_check_percent(data, "added_mass", "mass_balance_max_error")
            if am_val is not None:
                val_num = float(am_val)
                q_rows.append(["Added Mass", f"{val_num:.2f}%", str(am_chk.get("status") or "—").title()])

            if q_rows:
                self._subtabs_tables.addTab(self._create_table(q_cols, q_rows, is_dark=self._is_dark()), "Numerical Checks")

        # 7. Measurements Table
        if self._measurements:
            cols = ["Measurement Entity / Property", "Measured Output"]
            rows = [[str(k), str(v)] for k, v in self._measurements]
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Measurements")

        # Default fallback table if no specialized tables were populated
        if self._subtabs_tables.count() == 0:
            cols = ["Metric Name", "Value"]
            rows = []
            for k in ("peak_stress_nodal", "peak_displacement", "compliance", "volume", "mass"):
                if data.get(k) is not None:
                    rows.append([k.replace("_", " ").title(), _fmt(data[k])])
            if not rows:
                rows = [["Result Status", run_status(data)]]
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "General Metrics")

    # ── Graphs Tab Population ─────────────────────────────────────────
    def _populate_graphs(self):
        while self._subtabs_graphs.count():
            w = self._subtabs_graphs.widget(0)
            self._subtabs_graphs.removeTab(0)
            if w is not None:
                w.deleteLater()

        if not _HAVE_PYQTGRAPH:
            self._no_graphs_label.setVisible(True)
            self._no_graphs_label.setText("Charts unavailable.")
            self._tabs.setTabVisible(self._tabs.indexOf(self._tab_graphs), False)
            return

        data = self._data
        rtype = data.get("type", "")
        has_any_plot = False
        is_dark = self._is_dark()
        plot_bg = "#181b22" if is_dark else "#ffffff"

        # Each optimization response has its own units and interpretation.
        if rtype == "topopt_voxel" and data.get("study_kind") != "lattice_infill" and not data.get("generated_lattice"):
            histories = (
                ("objective_history", "Objective", "Objective value", "#58a6ff"),
                ("compliance_history", "Stiffness", "Structural compliance (N mm)", "#3fb950"),
                ("change_history", "Design change", "Mean density change", "#d29922"),
                ("stress_history", "Stress constraint", "Aggregated stress (MPa)", "#f85149"),
            )
            for key, title, axis, color in histories:
                raw_values = data.get(key)
                if raw_values is None:
                    continue
                values = np.asarray(raw_values, dtype=float).reshape(-1)
                if values.size < 2 or np.count_nonzero(np.isfinite(values)) < 2:
                    continue
                plot = pg.PlotWidget(background=plot_bg)
                self._configure_plot(plot, "Optimization iteration", axis)
                plot.plot(np.arange(1, values.size + 1), values, name=title,
                          pen=pg.mkPen(color if is_dark else "#0969da", width=2), connect="finite")
                if key == "stress_history":
                    allowable = float((data.get("stress_aggregation") or {}).get("allowable_stress") or 0)
                    if allowable > 0:
                        plot.addItem(pg.InfiniteLine(pos=allowable, angle=0, label="Allowable stress",
                            pen=pg.mkPen("#f85149", width=1, style=QtCore.Qt.DashLine)))
                self._subtabs_graphs.addTab(plot, title)
                has_any_plot = True

        analysis = fea_result(data)
        levels = (analysis.get("convergence_study") or {}).get("levels") or []
        solved_levels = [level for level in levels if isinstance(level, dict)
                         and level.get("solved") and level.get("n_elements") is not None
                         and level.get("compliance") is not None]
        if len(solved_levels) >= 2:
            plot = pg.PlotWidget(background=plot_bg)
            self._configure_plot(plot, "FEA elements", "Structural compliance (N mm)")
            plot.plot([level["n_elements"] for level in solved_levels],
                      [level["compliance"] for level in solved_levels], name="FEA compliance",
                      symbol="o", pen=pg.mkPen("#58a6ff" if is_dark else "#0969da", width=2))
            self._subtabs_graphs.addTab(plot, "FEA mesh convergence")
            has_any_plot = True

        # 2. Crash SAE J211 Plots
        if rtype == "crash":
            histories = data.get("histories") or {}
            processed = histories.get("processed") or {}
            raw = histories.get("raw") or {}

            time_ms = processed.get("time_ms")
            crush_mm = processed.get("crush_displacement_mm")
            force_kn = processed.get("rigid_wall_force_kN")

            # 2a. Crush curve (Force vs. Displacement)
            if crush_mm is not None and force_kn is not None:
                c_arr = np.asarray(crush_mm, dtype=float)
                f_arr = np.asarray(force_kn, dtype=float)
                cnt = min(c_arr.size, f_arr.size)
                if cnt >= 2:
                    p_fd = pg.PlotWidget(background=plot_bg)
                    self._configure_plot(p_fd, "Crush Displacement (mm)", "Rigid Wall Force (kN)")

                    raw_f = raw.get("rigid_wall_force_kN")
                    if raw_f is not None:
                        rf_arr = np.asarray(raw_f, dtype=float)
                        if rf_arr.size >= cnt:
                            p_fd.plot(
                                c_arr[:cnt], rf_arr[:cnt], name="Raw Unfiltered",
                                pen=pg.mkPen("#64748b" if is_dark else "#94a3b8", width=1)
                            )

                    p_fd.plot(
                        c_arr[:cnt], f_arr[:cnt], name="CFC Filtered",
                        pen=pg.mkPen("#58a6ff" if is_dark else "#0969da", width=2)
                    )
                    mean_f = histories.get("metrics", {}).get("mean_crushing_force_kN")
                    if mean_f:
                        p_fd.addLine(
                            y=float(mean_f),
                            pen=pg.mkPen("#d29922" if is_dark else "#9a6700", width=1, style=QtCore.Qt.DashLine)
                        )
                    self._subtabs_graphs.addTab(p_fd, "Force vs. Crush (F-d)")
                    has_any_plot = True

            # 2b. Energy Balance vs. Time
            if time_ms is not None:
                t_arr = np.asarray(time_ms, dtype=float)
                channels = [
                    ("kinetic_energy_kj", "Kinetic Energy", "#58a6ff" if is_dark else "#0969da"),
                    ("internal_energy_kj", "Internal (Absorbed)", "#3fb950" if is_dark else "#1a7f37"),
                    ("total_energy_kj", "Total Energy", "#c9d1d9" if is_dark else "#24292f"),
                    ("contact_energy_kj", "Contact Energy", "#d29922" if is_dark else "#9a6700"),
                    ("hourglass_energy_kj", "Hourglass Energy", "#f85149" if is_dark else "#cf222e"),
                ]
                p_e = pg.PlotWidget(background=plot_bg)
                self._configure_plot(p_e, "Time (ms)", "Energy (kJ)")
                drawn = 0
                for ch_key, ch_name, color in channels:
                    ch_val = processed.get(ch_key)
                    if ch_val is not None:
                        v_arr = np.asarray(ch_val, dtype=float)
                        k_cnt = min(t_arr.size, v_arr.size)
                        if k_cnt >= 2:
                            p_e.plot(t_arr[:k_cnt], v_arr[:k_cnt], name=ch_name, pen=pg.mkPen(color, width=2))
                            drawn += 1
                if drawn:
                    self._subtabs_graphs.addTab(p_e, "Energy Balance vs. Time")
                    has_any_plot = True

        # 3. FEA Component Comparison Bar Chart
        components = (analysis or data).get("components")
        if isinstance(components, list) and len(components) >= 2:
            names = [str(c.get("name", f"P{idx+1}")) for idx, c in enumerate(components)]
            stresses = [float(c.get("peak_stress_mpa") or 0.0) for c in components]
            if any(s > 0 for s in stresses):
                p_bar = pg.PlotWidget(background=plot_bg)
                self._configure_plot(p_bar, "Assembly Component", "Peak Von Mises Stress (MPa)")
                p_bar.showGrid(x=False, y=True, alpha=0.25)
                x = np.arange(len(stresses))
                bar_item = pg.BarGraphItem(
                    x=x, height=stresses, width=0.6,
                    brush="#58a6ff" if is_dark else "#0969da"
                )
                p_bar.addItem(bar_item)
                ax = p_bar.getAxis("bottom")
                ax.setTicks([list(enumerate(names))])
                self._subtabs_graphs.addTab(p_bar, "Component Stresses")
                has_any_plot = True

        if data.get("type") == "modal" and data.get("modes"):
            plot = pg.PlotWidget(background=plot_bg)
            self._configure_plot(plot, "Mode", "Natural Frequency (Hz)")
            modes = data["modes"]
            plot.addItem(pg.BarGraphItem(x=[m["mode"] for m in modes], height=[m["frequency_hz"] for m in modes], width=0.6, brush="#58a6ff" if is_dark else "#0969da"))
            self._subtabs_graphs.addTab(plot, "Frequency Spectrum")
            has_any_plot = True
        self._no_graphs_label.setVisible(not has_any_plot)
        self._tabs.setTabVisible(self._tabs.indexOf(self._tab_graphs), has_any_plot)

    def _configure_plot(self, plot, xlabel: str = "", ylabel: str = ""):
        """Apply MDAO-aligned axis styles and grid visibility to pyqtgraph plot."""
        is_dark = self._is_dark()
        fg_col = "#8b949e" if is_dark else "#475569"
        plot.showGrid(x=True, y=True, alpha=0.25 if is_dark else 0.3)
        plot.setLabel("bottom", xlabel, color=fg_col)
        plot.setLabel("left", ylabel, color=fg_col)
        for ax_name in ("left", "bottom"):
            ax = plot.getAxis(ax_name)
            ax.setPen(QtGui.QColor(fg_col))
            ax.setTextPen(QtGui.QColor(fg_col))
        plot.addLegend(offset=(10, 10), labelTextColor=fg_col)

    # ── Summary Tab Population ────────────────────────────────────────
    def _add_overview_section(self, title, rows):
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        for index, row in enumerate(rows):
            label, value = row[:2]
            meaning = row[2] if len(row) > 2 else ""
            key_widget = QtWidgets.QLabel(label)
            key_widget.setWordWrap(True)
            key_widget.setToolTip(meaning)
            value_widget = QtWidgets.QLabel(value)
            value_widget.setWordWrap(True)
            value_widget.setToolTip(meaning)
            value_widget.setStyleSheet("font-weight: 600;")
            value_widget.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            value_widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(key_widget, index, 0)
            grid.addWidget(value_widget, index, 1)
        self._summary_layout.addWidget(group)

    def _populate_performance_overview(self):
        data = self._data
        sections = result_metric_sections(data)
        metrics = {label: (value, meaning) for _, rows in sections for label, value, meaning in rows}
        preferred = ("Lattice mass", "Simulated material mass", "Mass", "Material retained",
                     "Peak von Mises stress", "Peak displacement", "Minimum yield safety factor",
                     "Cell size", "Structural compliance")
        highlights = [label for label in preferred if label in metrics][:4]
        if highlights:
            cards = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(cards)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(8)
            for index, label in enumerate(highlights):
                value, meaning = metrics[label]
                card_title = f"FEA {label.lower()}" if data.get("type") == "topopt_voxel" and fea_result(data) and label in {"Peak von Mises stress", "Peak displacement", "Minimum yield safety factor"} else label
                card = QtWidgets.QGroupBox(card_title)
                layout = QtWidgets.QVBoxLayout(card)
                value_widget = QtWidgets.QLabel(value)
                value_widget.setObjectName("ResultHighlightValue")
                value_widget.setWordWrap(True)
                value_widget.setStyleSheet("font-size: 22px; font-weight: 600;")
                value_widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                layout.addWidget(value_widget)
                card.setToolTip(meaning)
                grid.addWidget(card, index // 2, index % 2)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            self._summary_layout.addWidget(cards)

        self._add_overview_section("Run status", result_state_rows(data))
        sections.sort(key=lambda section: section[0] != "FEA analysis")
        for title, rows in sections:
            self._add_overview_section(title, rows)
        for notice in result_notices(data):
            label = QtWidgets.QLabel(notice)
            label.setTextFormat(QtCore.Qt.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            label.setStyleSheet("padding: 8px; border-left: 2px solid #d29922;")
            self._summary_layout.addWidget(label)
        elapsed = data.get("solver_elapsed_s")
        if elapsed is None and isinstance(data.get("timing"), dict):
            elapsed = sum(float(value or 0) for value in data["timing"].values())
        if elapsed is not None:
            self._add_overview_section("Run time", [("Elapsed time", _fmt(elapsed, "s"))])
        self._summary_layout.addStretch(1)

    def _populate_summary(self):
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

        data = self._data
        if data.get("type") in {"topopt_voxel", "fea", "external_solver"}:
            self._populate_performance_overview()
            return
        is_dark = self._is_dark()
        key_color = "#aebbd0" if is_dark else "#64748b"
        value_color = "#f1f5f9" if is_dark else "#1e293b"
        key_style = f"color: {key_color};"
        value_style = f"color: {value_color}; font-weight: 600;"
        rtype = data.get("type", "—").upper()

        # Meta Group
        meta_group = QtWidgets.QGroupBox("Run")
        meta_grid = QtWidgets.QGridLayout(meta_group)
        meta_grid.setContentsMargins(10, 8, 10, 8)
        meta_rows = [
            ("Analysis", str(data.get("analysis_type") or analysis_label(data))),
            *result_state_rows(data),
        ]
        elapsed = data.get("solver_elapsed_s")
        if elapsed is None and isinstance(data.get("timing"), dict):
            elapsed = sum(float(value or 0.0) for value in data["timing"].values())
        if elapsed is not None:
            meta_rows.append(("Runtime", _fmt(elapsed, "s")))
        for idx, (lbl, val) in enumerate(meta_rows):
            k = QtWidgets.QLabel(lbl)
            k.setStyleSheet(key_style)
            v = QtWidgets.QLabel(val)
            v.setStyleSheet(value_style)
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            meta_grid.addWidget(k, idx, 0)
            meta_grid.addWidget(v, idx, 1)
        self._summary_layout.addWidget(meta_group)

        if rtype.lower() == "crash":
            metrics = data.get("crash_metrics") or {}
            crash_rows = []
            specifications = (
                ("crush_force_efficiency", "crush_force_efficiency", "Crush Force Efficiency", "%", 100.0),
                ("specific_energy_absorption", "specific_energy_absorption_kJ_kg", "Specific Energy Absorption", "kJ/kg", 1.0),
                ("peak_force", "peak_crushing_force_kN", "Peak Crushing Force", "kN", 1.0),
                ("mean_force", "mean_crushing_force_kN", "Mean Crushing Force", "kN", 1.0),
                ("absorbed_energy_kj", "absorbed_energy_kJ", "Absorbed Energy", "kJ", 1.0),
                ("crush_distance", "useful_crush_stroke_mm", "Crush Stroke", "mm", 1.0),
                ("peak_acceleration_g", "peak_acceleration_g", "Peak Deceleration", "g", 1.0),
                ("delta_v", "delta_v_m_s", "Velocity Change", "m/s", 1.0),
                ("peak_displacement", "peak_displacement_mm", "Peak Displacement", "mm", 1.0),
                ("peak_stress", "peak_stress_mpa", "Peak Stress", "MPa", 1.0),
                ("n_failed", "n_failed", "Eroded Elements", "", 1.0),
            )
            for key, metric_key, label, unit, scale in specifications:
                value = data.get(key)
                if value is None:
                    value = metrics.get(metric_key)
                if value is not None:
                    crash_rows.append((label, _fmt(float(value) * scale, unit)))
            if data.get("absorbed_energy_kj") is None and metrics.get("absorbed_energy_kJ") is None and data.get("absorbed_energy") is not None:
                crash_rows.append(("Absorbed Energy", _fmt(float(data["absorbed_energy"]) / 1e6, "kJ")))
            numerical_rows = []
            for check_id, fallback, label in (
                ("energy_balance", "energy_balance_max_error", "Energy Balance Error"),
                ("hourglass_energy", None, "Hourglass Energy"),
                ("hourglass_plus_contact_energy", None, "Hourglass + Contact Energy"),
                ("added_mass", "mass_balance_max_error", "Added Mass"),
            ):
                value = crash_check_percent(data, check_id, fallback)
                if value is not None:
                    numerical_rows.append((label, _fmt(value, "%")))
            for title, rows in (("Impact Results", crash_rows), ("Numerical Checks", numerical_rows)):
                if not rows:
                    continue
                group = QtWidgets.QGroupBox(title)
                grid = QtWidgets.QGridLayout(group)
                grid.setContentsMargins(10, 8, 10, 8)
                for index, (label, value) in enumerate(rows):
                    key_widget = QtWidgets.QLabel(label)
                    key_widget.setStyleSheet(key_style)
                    value_widget = QtWidgets.QLabel(value)
                    value_widget.setStyleSheet(value_style)
                    value_widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    grid.addWidget(key_widget, index, 0)
                    grid.addWidget(value_widget, index, 1)
                self._summary_layout.addWidget(group)
        else:
            # Performance / Key Results Group
            section_title = "Natural Frequencies" if rtype == "MODAL" else "Structural Results"
            res_group = QtWidgets.QGroupBox(section_title)
            res_grid = QtWidgets.QGridLayout(res_group)
            res_grid.setContentsMargins(10, 8, 10, 8)
            res_rows = []
            if data.get("peak_stress_nodal") is not None:
                res_rows.append(("Peak Stress (Nodal)", _fmt(data["peak_stress_nodal"], "MPa")))
            if rtype.lower() == "modal":
                res_rows.extend([
                    ("Extracted Modes", str(len(data.get("modes") or []))),
                    ("Active Mode", str(data.get("active_mode", "—"))),
                    ("Active Natural Frequency", _fmt(data.get("frequency_hz"), "Hz")),
                ])
            if data.get("peak_displacement") is not None and rtype.lower() != "modal":
                res_rows.append(("Peak Displacement", _fmt(data["peak_displacement"], "mm")))
            min_safety = data.get("minimum_yield_safety_factor", data.get("min_safety_factor"))
            if min_safety is not None:
                res_rows.append(("Minimum Yield Safety Factor", _fmt(min_safety)))
            if data.get("compliance") is not None:
                res_rows.append(("Structural Compliance", _fmt(data["compliance"], "N mm")))
            if data.get("strain_energy") is not None:
                res_rows.append(("Strain Energy", _fmt(data["strain_energy"], "N mm")))
            if data.get("volume") is not None:
                res_rows.append(("Volume", _fmt(data["volume"], "mm³")))
            if data.get("mass") is not None:
                res_rows.append(("Mass", _fmt(float(data["mass"]) * 1000.0, "kg")))

            for idx, (lbl, val) in enumerate(res_rows):
                k = QtWidgets.QLabel(lbl)
                k.setStyleSheet(key_style)
                v = QtWidgets.QLabel(val)
                v.setStyleSheet(value_style)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                res_grid.addWidget(k, idx, 0)
                res_grid.addWidget(v, idx, 1)
            if res_rows:
                self._summary_layout.addWidget(res_group)

        self._summary_layout.addStretch(1)

    # ── Export & Clipboard ─────────────────────────────────────────────
    def _export_csv(self):
        """Export all populated data tables to CSV files."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results CSVs"
        )
        if not folder:
            return

        exported_count = 0
        for tab_idx in range(self._subtabs_tables.count()):
            tab_title = self._subtabs_tables.tabText(tab_idx).replace(" ", "_").lower()
            widget = self._subtabs_tables.widget(tab_idx)
            if not isinstance(widget, QtWidgets.QTableWidget):
                continue

            file_path = f"{folder}/result_{tab_title}.csv"
            try:
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Header
                    headers = [widget.horizontalHeaderItem(col).text() for col in range(widget.columnCount())]
                    writer.writerow(headers)
                    # Rows
                    for row in range(widget.rowCount()):
                        row_data = [
                            widget.item(row, col).text() if widget.item(row, col) else ""
                            for col in range(widget.columnCount())
                        ]
                        writer.writerow(row_data)
                exported_count += 1
            except Exception as e:
                logger.exception("Could not export table to CSV: %s", e)

        QtWidgets.QMessageBox.information(
            self, "CSV Export Complete", f"Successfully exported {exported_count} result tables to:\n{folder}"
        )

    def _copy_summary(self):
        """Copy key metrics summary to system clipboard."""
        data = self._data
        lines = [
            f"PyLCSS {data.get('type', 'CAE').upper()} Simulation Results",
            "=" * 40,
            *(f"{label}: {value}" for label, value in result_state_rows(data)),
        ]
        rtype = str(data.get("type", "")).lower()
        if rtype == "crash":
            metrics = data.get("crash_metrics") or {}
            for k, lbl, u in [
                ("peak_force", "Peak Crushing Force", "kN"),
                ("mean_force", "Mean Crushing Force", "kN"),
                ("crush_force_efficiency", "Crush-Force Efficiency (CFE)", ""),
                ("specific_energy_absorption", "Specific Energy Absorption (SEA)", "kJ/kg"),
                ("absorbed_energy_kj", "Total Absorbed Energy", "kJ"),
                ("crush_distance", "Useful Crush Stroke", "mm"),
                ("peak_acceleration_g", "Peak Deceleration", "g"),
                ("delta_v", "Velocity Change", "m/s"),
                ("n_failed", "Eroded Elements", ""),
            ]:
                val = data.get(k) or metrics.get(k)
                if val is not None:
                    lines.append(f"{lbl}: {_fmt(val, u)}")
        else:
            for title, rows in result_metric_sections(data):
                lines.extend(["", title, *(f"{label}: {value}" for label, value, _ in rows)])
            lines.extend(result_notices(data))
        text = "\n".join(lines)
        QtGui.QGuiApplication.clipboard().setText(text)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Summary copied to clipboard!")


class DetailedResultsDialog(QtWidgets.QDialog):
    """Modeless dialog providing tables, graphs, and live 3D viewer switching."""

    view_mode_requested = QtCore.Signal(str)
    undeformed_requested = QtCore.Signal(bool)
    extrema_requested = QtCore.Signal(bool)
    deform_scale_requested = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Results Hub — Tables, Graphs & 3D Views")
        self.resize(960, 680)
        self.setMinimumSize(780, 520)
        self.setStyleSheet(_DIALOG_QSS)
        self.setModal(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = DetailedResultsWidget(self)
        layout.addWidget(self.view)

        # Forward signals
        self.view.view_mode_requested.connect(self.view_mode_requested.emit)
        self.view.undeformed_requested.connect(self.undeformed_requested.emit)
        self.view.extrema_requested.connect(self.extrema_requested.emit)
        self.view.deform_scale_requested.connect(self.deform_scale_requested.emit)
        self.view.switch_to_setup_requested.connect(self.hide)

        # Direct property forwarding for tests and external callers
        self._subtabs_tables = self.view._subtabs_tables
        self._subtabs_graphs = self.view._subtabs_graphs
        self._tabs = self.view._tabs
        self._btn_group_3d = self.view._btn_group_3d
        self._3d_mode_combo = self.view._3d_mode_combo
        self._chk_undeformed = self.view._chk_undeformed
        self._scale_combo = self.view._scale_combo
        from pylcss.user_interface.common.theme_manager import current_theme
        self.apply_theme(current_theme())

    def apply_theme(self, theme_name: str):
        self._theme_name = str(theme_name).lower()
        is_dark = self._theme_name == "dark"
        self.setStyleSheet(_DIALOG_DARK_QSS if is_dark else _DIALOG_LIGHT_QSS)
        self.view.apply_theme(theme_name)

    def update_result(self, data: Optional[Dict[str, Any]], measurements: Optional[List[Tuple[str, str]]] = None):
        self.view.update_result(data, measurements)

    def set_active_3d_mode(self, mode_name: str):
        self.view.set_active_3d_mode(mode_name)

    def select_tab(self, tab_name: str):
        self.view.select_tab(tab_name)

    def __getattr__(self, name: str):
        if hasattr(self, "view") and hasattr(self.view, name):
            return getattr(self.view, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
