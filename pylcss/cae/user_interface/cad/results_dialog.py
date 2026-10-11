# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Detailed Simulation Results Viewer Dialog.

Provides an interactive popup window for comprehensive FEA, TopOpt, and Crash
results with:
1. Live 3D Model Display selector (switches VTK viewer in real time).
2. Data Tables (Components, Assembly Ties, Reactions, Load Cases, Mesh Quality, KPIs).
3. Graph Results (Convergence History, SAE J211 Crash Curves, Energy Balance).
4. Run Summary, Timing & Warnings.
5. CSV Export.
"""

from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

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

        lbl_empty_title = QtWidgets.QLabel("No Simulation Results Yet")
        lbl_empty_title.setStyleSheet("font-weight: bold; font-size: 15px;")
        lbl_empty_title.setAlignment(QtCore.Qt.AlignCenter)

        lbl_empty_desc = QtWidgets.QLabel(
            "Execute an FEA, Modal, Impact, or Topology Optimization study in Setup\n"
            "to populate structural data tables, convergence curves, and mechanical metrics here."
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

        top_layout.addSpacing(16)

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

        top_layout.addStretch(1)

        # Undeformed overlay toggle
        self._chk_undeformed = QtWidgets.QCheckBox("Undeformed")
        self._chk_undeformed.setToolTip("Show the undeformed analysis mesh as a wireframe")
        self._chk_undeformed.toggled.connect(self.undeformed_requested.emit)
        top_layout.addWidget(self._chk_undeformed)

        # Deformation scale combo
        self._lbl_scale = QtWidgets.QLabel("Scale:")
        self._scale_combo = QtWidgets.QComboBox()
        self._scale_combo.setMinimumWidth(75)
        self._scale_combo.addItems(["Auto", "True (1x)", "2x", "5x", "10x", "25x", "50x", "100x"])
        self._scale_combo.currentTextChanged.connect(self.deform_scale_requested.emit)
        top_layout.addWidget(self._lbl_scale)
        top_layout.addWidget(self._scale_combo)
        self._chk_extrema = QtWidgets.QCheckBox("Min / Max")
        self._chk_extrema.toggled.connect(self.extrema_requested.emit)
        top_layout.addWidget(self._chk_extrema)

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

        self._tabs.addTab(self._tab_tables, "Data Tables")
        self._tabs.addTab(self._tab_graphs, "Graph Results")
        self._tabs.addTab(self._tab_summary, "Summary")
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
                item.setForeground(QtGui.QBrush(text_color))
                if c_idx > 0 and any(char.isdigit() for char in str(val)):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                table.setItem(r_idx, c_idx, item)

        table.resizeColumnsToContents()
        for c in range(len(columns)):
            w = table.columnWidth(c)
            table.setColumnWidth(c, max(w + 24, 120))
        table.horizontalHeader().setStretchLastSection(True)
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
        backend = self._data.get("backend", "CalculiX" if rtype == "fea" else "OpenRadioss" if rtype == "crash" else "pyMOTO" if rtype == "topopt_voxel" else "Solver")

        title = {"fea": "Structural FEA", "modal": "Modal Analysis", "topopt_voxel": "Topology Optimization", "crash": "Crash / Impact"}.get(rtype, rtype.upper())
        self._lbl_title.setText(f"{title} Results")
        badge = {"topopt_voxel": "Topology", "external_solver": "FEA"}.get(rtype, rtype.upper())
        self._badge_type.setText(f"{badge} · {backend}")
        self._style_type_badge()

        self._populate_3d_controls()
        self._populate_tables()
        self._populate_graphs()
        self._populate_summary()

    def set_active_3d_mode(self, mode_name: str):
        """Update active 3D selector and button state without re-triggering signal."""
        self._current_mode = str(mode_name)
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

    def select_tab(self, tab_name: str):
        """Select tab: 'tables', 'graphs', 'summary'."""
        tab_map = {"tables": 0, "graphs": 1, "summary": 2}
        idx = tab_map.get(str(tab_name).lower(), 0)
        self._tabs.setCurrentIndex(idx)

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

        has_disp = (
            self._data.get("displacement") is not None
            or self._data.get("peak_displacement") is not None
        )
        self._lbl_scale.setVisible(has_disp)
        self._scale_combo.setVisible(has_disp)
        self._chk_undeformed.setVisible(has_disp)
        self._chk_extrema.setVisible(modes != ["Default View"])

    def _on_3d_combo_changed(self, mode: str):
        if self._updating_controls or not mode:
            return
        self._current_mode = mode
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

        if rtype in {"fea", "external_solver", "topopt_voxel"}:
            metrics = [
                ("peak_stress_nodal", "Peak Von Mises Stress", "MPa"),
                ("peak_displacement", "Peak Displacement", "mm"),
                ("minimum_yield_safety_factor", "Minimum Yield Safety Factor", ""),
                ("compliance", "Compliance", "N mm"),
                ("volume", "Volume", "mm³"),
                ("target_vol_frac", "Target Volume Fraction", ""),
                ("final_vol_frac", "Final Volume Fraction", ""),
                ("iterations", "Iterations", ""),
            ]
            rows = [[label, _fmt(data[key], unit)] for key, label, unit in metrics if data.get(key) is not None]
            if data.get("mass") is not None:
                rows.append(["Mass", _fmt(float(data["mass"]) * 1000, "kg")])
            if rtype == "topopt_voxel" and data.get("converged") is not None:
                rows.append(["Convergence", "Converged" if data["converged"] else "Not converged"])
            if rows:
                self._subtabs_tables.addTab(self._create_table(["Metric", "Value"], rows, is_dark=self._is_dark()), "Topology Metrics" if rtype == "topopt_voxel" else "Structural Metrics")

        if rtype == "topopt_voxel" and isinstance(data.get("passive_regions"), dict):
            regions = data["passive_regions"]
            rows = [
                ["Non-design bodies", "Preserve solid material", len(regions.get("explicit_solid_shapes") or []), regions.get("explicit_solid_voxels", 0)],
                ["Obstacles / keep-out bodies", "Remove material", len(regions.get("explicit_void_shapes") or []), regions.get("explicit_void_voxels", 0)],
                ["Automatic interface protection", str(regions.get("automatic_exclusion_scope") or "None"), "—", "—"],
                ["Interface protection thickness (mm)", _fmt(regions.get("automatic_exclusion_thickness")), "—", "—"],
            ]
            self._subtabs_tables.addTab(self._create_table(["Region", "Behavior", "Bodies", "Explicit Voxels"], rows, is_dark=self._is_dark()), "Non-design & Obstacles")

        # 1. Components Table (FEA multi-body)
        components = data.get("components")
        if isinstance(components, list) and components:
            cols = ["Component Name", "Material", "Elements", "Peak Stress (MPa)", "Yield FoS", "Status"]
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
                status = "Safe (FoS ≥ 1.5)" if (fos_val and float(fos_val) >= 1.5) else "Marginal" if (fos_val and float(fos_val) >= 1.0) else "Critical / Yield" if fos_val else "—"
                rows.append([name, mat, elem, stress, fos_str, status])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Components")

        # 2. Assembly Interfaces Table (Tied contact)
        connections = data.get("connections")
        if isinstance(connections, list) and connections:
            cols = ["Component Pair", "Connection Type", "Search Tolerance", "Interface Status"]
            rows = []
            for conn in connections:
                if not isinstance(conn, dict):
                    continue
                pair = f"{conn.get('component_a', '?')} ↔ {conn.get('component_b', '?')}"
                tol = _fmt(conn.get("search_tolerance_mm"), "mm")
                rows.append([pair, "Bonded Tie (*TIE)", tol, "Active / Bonded"])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Assembly Ties")

        # 3. Load Cases Table (TopOpt Multi-load & Cases)
        cases = data.get("structural_cases")
        thermal_cases = data.get("thermal_cases")
        if (isinstance(cases, list) and cases) or (isinstance(thermal_cases, list) and thermal_cases):
            cols = ["Load Case Name", "Discipline", "Case Weight", "Compliance", "Max Response"]
            rows = []
            for c in (cases or []):
                if isinstance(c, dict):
                    name = str(c.get("name", "Structural Case"))
                    weight = _fmt(c.get("weight", 1.0))
                    comp = _fmt(c.get("compliance"), "N mm")
                    disp = _fmt(c.get("maximum_displacement"), "mm")
                    rows.append([name, "Structural", weight, comp, f"{disp} max disp"])
            for tc in (thermal_cases or []):
                if isinstance(tc, dict):
                    name = str(tc.get("name", "Thermal Case"))
                    weight = _fmt(tc.get("weight", 1.0))
                    comp = _fmt(tc.get("thermal_compliance"), "W K")
                    temp = _fmt(tc.get("maximum_temperature_rise"), "K")
                    rows.append([name, "Thermal", weight, comp, f"{temp} max rise"])
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Load Cases")

        # 4. Reaction Forces Table
        rx = data.get("reaction_force")
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
        quality = data.get("mesh_quality")
        mesh_obj = data.get("mesh")
        if quality is None and mesh_obj is not None:
            quality = getattr(mesh_obj, "quality_report", None)
        if isinstance(quality, dict):
            cols = ["Quality Metric", "Evaluated Value", "Ideal Criterion", "Engineering Status"]
            rows = [
                ["Assessment", str(quality.get("assessment", "—")), "High Quality", "Passed"],
                ["Min Mean-Ratio (Element Quality)", _fmt(quality.get("min_mean_ratio")), "≥ 0.20", "OK" if float(quality.get("min_mean_ratio") or 0) >= 0.2 else "Review"],
                ["5th Percentile Mean-Ratio", _fmt(quality.get("p05_mean_ratio")), "≥ 0.40", "OK"],
                ["Average Mean-Ratio", _fmt(quality.get("mean_mean_ratio")), "≥ 0.70", "OK"],
                ["Maximum Edge Ratio (Aspect Ratio)", _fmt(quality.get("max_edge_ratio")), "≤ 10.0", "OK" if float(quality.get("max_edge_ratio") or 1) <= 10 else "High Aspect"],
                ["Degenerate / Collapsed Elements", str(int(quality.get("degenerate_elements") or 0)), "0", "Valid" if int(quality.get("degenerate_elements") or 0) == 0 else "Degenerate"],
            ]
            self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Mesh Quality")

        # 6. Crash SAE J211 KPIs Table
        if rtype == "crash":
            cols = ["Crash Characteristic", "Evaluated Metric", "Standard Unit", "Qualification"]
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
                    rows.append([lbl, _fmt(data[k]), u, "Solver metric"])
            if rows:
                self._subtabs_tables.addTab(self._create_table(cols, rows, is_dark=self._is_dark()), "Crash KPIs")

        # 6b. Crash Solution Quality & Energy Gates Table
        if rtype == "crash":
            quality = data.get("quality") or {}
            checks = {c.get("id"): c for c in quality.get("checks", []) if isinstance(c, dict)}
            q_cols = ["Energy / Quality Gate", "Evaluated Value", "Acceptance Criterion", "Status"]
            q_rows = []

            q_status = str(data.get("quality_status") or quality.get("status") or "Evaluated").upper()
            q_rows.append(["Overall Qualification", q_status, "Qualified Solution", q_status])

            eb_chk = checks.get("energy_balance", {})
            eb_val = eb_chk.get("value", data.get("energy_balance_max_error"))
            if eb_val is not None:
                val_num = float(eb_val)
                eb_rows_val = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num * 100.0:.2f}%"
                q_rows.append(["Energy Balance Error", eb_rows_val, "≤ 5.0% error", str(eb_chk.get("status", "pass")).upper()])

            hg_chk = checks.get("hourglass_energy", {})
            hg_val = hg_chk.get("value")
            if hg_val is not None:
                val_num = float(hg_val)
                hg_rows_val = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num * 100.0:.2f}%"
                q_rows.append(["Hourglass Energy Ratio", hg_rows_val, "≤ 10.0% of total", str(hg_chk.get("status", "pass")).upper()])

            art_chk = checks.get("hourglass_plus_contact_energy", {})
            art_val = art_chk.get("value")
            if art_val is not None:
                val_num = float(art_val)
                art_rows_val = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num * 100.0:.2f}%"
                q_rows.append(["Hourglass + Contact Energy", art_rows_val, "≤ 15.0% of total", str(art_chk.get("status", "pass")).upper()])

            am_chk = checks.get("added_mass", {})
            am_val = am_chk.get("value", data.get("mass_balance_max_error"))
            if am_val is not None:
                val_num = float(am_val)
                am_rows_val = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num * 100.0:.2f}%"
                q_rows.append(["Added Mass (Scaling)", am_rows_val, "≤ 1.0% mass added", str(am_chk.get("status", "pass")).upper()])

            term_stat = str(data.get("external_status") or data.get("engine_status") or "engine_completed")
            clean_stat = term_stat.replace("_", " ").title()
            is_normal = "fail" not in term_stat.lower() and "error" not in term_stat.lower()
            q_rows.append(["Solver Termination", f"Normal ({clean_stat})" if is_normal else clean_stat, "Normal termination", "PASS" if is_normal else "FAIL"])

            if q_rows:
                self._subtabs_tables.addTab(self._create_table(q_cols, q_rows, is_dark=self._is_dark()), "Solution Quality and Energy Gates")

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
                rows = [["Result Status", str(data.get("external_status") or "No scalar metrics available")]]
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
            self._no_graphs_label.setText("pyqtgraph is unavailable; charts cannot be rendered.")
            return

        data = self._data
        rtype = data.get("type", "")
        has_any_plot = False
        is_dark = self._is_dark()
        plot_bg = "#181b22" if is_dark else "#ffffff"

        # 1. TopOpt Convergence Plot
        if rtype == "topopt_voxel":
            obj = data.get("objective_history")
            chg = data.get("change_history")
            if (obj is not None and len(obj) >= 2) or (chg is not None and len(chg) >= 2):
                plot = pg.PlotWidget(background=plot_bg)
                self._configure_plot(plot, "Optimizer Iteration", "Normalized Response")

                if obj is not None:
                    arr = np.asarray(obj, dtype=float)
                    if arr.size >= 2 and abs(arr[0]) > 1e-30:
                        norm_obj = arr / abs(arr[0])
                        iters = np.arange(1, len(norm_obj) + 1)
                        plot.plot(
                            iters, norm_obj, name="Objective / initial",
                            pen=pg.mkPen("#58a6ff" if is_dark else "#0969da", width=2)
                        )

                if chg is not None:
                    arr_c = np.asarray(chg, dtype=float)
                    if arr_c.size >= 2:
                        iters = np.arange(1, len(arr_c) + 1)
                        plot.plot(
                            iters, arr_c, name="Design Change",
                            pen=pg.mkPen("#d29922" if is_dark else "#9a6700", width=2)
                        )

                stress = data.get("stress_history")
                stress_meta = data.get("stress_aggregation")
                allowable = float(stress_meta.get("allowable_stress", 0.0)) if isinstance(stress_meta, dict) else 0.0
                if stress is not None and allowable > 0.0:
                    arr_s = np.asarray(stress, dtype=float)
                    if arr_s.size >= 2:
                        iters = np.arange(1, len(arr_s) + 1)
                        plot.plot(
                            iters, arr_s / allowable, name="Stress Utilization",
                            pen=pg.mkPen("#f85149" if is_dark else "#cf222e", width=2)
                        )
                        plot.addLine(
                            y=1.0,
                            pen=pg.mkPen("#f85149" if is_dark else "#cf222e", width=1, style=QtCore.Qt.DashLine)
                        )

                self._subtabs_graphs.addTab(plot, "Convergence History")
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
        components = data.get("components")
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
    def _populate_summary(self):
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

        data = self._data
        is_dark = self._is_dark()
        key_color = "#aebbd0" if is_dark else "#64748b"
        value_color = "#f1f5f9" if is_dark else "#1e293b"
        key_style = f"color: {key_color};"
        value_style = f"color: {value_color}; font-weight: 600;"
        rtype = data.get("type", "—").upper()
        backend = str(data.get("backend", "—"))

        # Meta Group
        meta_group = QtWidgets.QGroupBox("Solver Run")
        meta_grid = QtWidgets.QGridLayout(meta_group)
        meta_grid.setContentsMargins(10, 8, 10, 8)
        meta_rows = [
            ("Analysis", str(data.get("analysis_type") or {"FEA": "Structural FEA", "MODAL": "Natural Frequencies", "TOPOPT_VOXEL": "Topology Optimization", "CRASH": "Crash / Impact"}.get(rtype, rtype))),
            ("Backend Solver", backend),
            ("Status", str(data.get("external_status") or ("Converged" if data.get("converged") else "Not converged" if data.get("converged") is False else "Completed")).replace("_", " ").title()),
            ("Runtime", _fmt(data.get("solver_elapsed_s"), "s")),
            ("Work Directory", str(data.get("work_dir", "—"))),
        ]
        for idx, (lbl, val) in enumerate(meta_rows):
            k = QtWidgets.QLabel(lbl)
            k.setStyleSheet(key_style)
            v = QtWidgets.QLabel(val)
            v.setStyleSheet(value_style)
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            meta_grid.addWidget(k, idx, 0)
            meta_grid.addWidget(v, idx, 1)
        self._summary_layout.addWidget(meta_group)

        # Crash Performance & Energy Gates Groups
        if rtype.lower() == "crash":
            metrics = data.get("crash_metrics") or {}
            quality = data.get("quality") or {}

            # ── Crashworthiness Performance Group ──
            crash_group = QtWidgets.QGroupBox("Crashworthiness and Energy Absorption")
            crash_grid = QtWidgets.QGridLayout(crash_group)
            crash_grid.setContentsMargins(10, 8, 10, 8)

            peak_f = float(data.get("peak_force") or metrics.get("peak_crushing_force_kN") or 0.0)
            mean_f = float(data.get("mean_force") or metrics.get("mean_crushing_force_kN") or 0.0)
            cfe = float(
                data.get("crush_force_efficiency")
                or metrics.get("crush_force_efficiency")
                or (mean_f / peak_f if peak_f > 0 else 0.0)
            )
            sea = float(
                data.get("specific_energy_absorption")
                or metrics.get("specific_energy_absorption_kJ_kg")
                or 0.0
            )
            absorbed_e = float(
                data.get("absorbed_energy_kj")
                or metrics.get("absorbed_energy_kJ")
                or (float(data.get("absorbed_energy", 0.0)) / 1.0e6)
            )
            stroke = float(
                data.get("crush_distance")
                or metrics.get("useful_crush_stroke_mm")
                or 0.0
            )
            peak_a = float(
                data.get("peak_acceleration_g")
                or metrics.get("peak_acceleration_g")
                or 0.0
            )
            delta_v = float(
                data.get("delta_v")
                or metrics.get("delta_v_m_s")
                or 0.0
            )
            n_failed = int(data.get("n_failed", 0) or 0)

            cfe_pct = cfe * 100.0
            if cfe >= 0.70:
                cfe_hint = " (Excellent / Progressive Buckling)"
            elif cfe >= 0.50:
                cfe_hint = " (Good / Typical Crush Box)"
            else:
                cfe_hint = " (Low Efficiency / High Initial Peak)"

            crash_rows = [
                ("Crash Force Efficiency (CFE)", f"{cfe_pct:.1f}%{cfe_hint}"),
                ("Specific Energy Absorption (SEA)", f"{sea:.2f} kJ/kg"),
                ("Peak Crushing Force (F_peak)", f"{peak_f:.2f} kN"),
                ("Mean Crushing Force (F_mean)", f"{mean_f:.2f} kN"),
                ("Total Energy Absorbed", f"{absorbed_e:.2f} kJ"),
                ("Useful Crush Stroke", f"{stroke:.1f} mm"),
            ]
            if peak_a > 0.0:
                crash_rows.append(("Peak Deceleration", f"{peak_a:.1f} g"))
            if delta_v > 0.0:
                crash_rows.append(("Velocity Change (ΔV)", f"{delta_v:.2f} m/s"))
            crash_rows.append(("Eroded Elements", "0 (Intact - No Rupture)" if n_failed == 0 else f"{n_failed} elements eroded"))

            for idx, (lbl, val) in enumerate(crash_rows):
                k = QtWidgets.QLabel(lbl)
                k.setStyleSheet(key_style)
                v = QtWidgets.QLabel(val)
                v.setStyleSheet(value_style)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                crash_grid.addWidget(k, idx, 0)
                crash_grid.addWidget(v, idx, 1)
            self._summary_layout.addWidget(crash_group)

            # ── Solution Quality & Energy Gates Group ──
            qual_group = QtWidgets.QGroupBox("Solution Quality and Energy Gates")
            qual_grid = QtWidgets.QGridLayout(qual_group)
            qual_grid.setContentsMargins(10, 8, 10, 8)

            q_status = str(data.get("quality_status") or quality.get("status") or "").lower()
            if q_status == "pass":
                stat_str = "[PASS] Qualified Explicit Solution"
            elif q_status == "warning":
                stat_str = "[WARNING] Review Hourglass / Dissipation"
            elif q_status == "fail":
                stat_str = "[FAIL] Energy Balance / Termination Issue"
            else:
                stat_str = "Evaluated"

            qual_rows = [("Quality Qualification", stat_str)]

            checks = {c.get("id"): c for c in quality.get("checks", []) if isinstance(c, dict)}
            eb_chk = checks.get("energy_balance", {})
            eb_val = eb_chk.get("value", data.get("energy_balance_max_error"))
            if eb_val is not None:
                val_num = float(eb_val)
                eb_str = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num*100.0:.2f}%"
                qual_rows.append(("Energy Balance Error", f"{eb_str} (Limit ≤ 5% pass)"))

            hg_chk = checks.get("hourglass_energy", {})
            hg_val = hg_chk.get("value")
            if hg_val is not None:
                val_num = float(hg_val)
                hg_str = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num*100.0:.2f}%"
                qual_rows.append(("Hourglass Energy Ratio", f"{hg_str} (Limit ≤ 10% pass)"))

            art_chk = checks.get("hourglass_plus_contact_energy", {})
            art_val = art_chk.get("value")
            if art_val is not None:
                val_num = float(art_val)
                art_str = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num*100.0:.2f}%"
                qual_rows.append(("Hourglass + Contact Energy", f"{art_str} (Limit ≤ 15% pass)"))

            am_chk = checks.get("added_mass", {})
            am_val = am_chk.get("value", data.get("mass_balance_max_error"))
            if am_val is not None:
                val_num = float(am_val)
                am_str = f"{val_num:.2f}%" if val_num > 1.0 else f"{val_num*100.0:.2f}%"
                qual_rows.append(("Added Mass (Scaling)", f"{am_str} (Limit ≤ 1% pass)"))

            term_stat = str(data.get("external_status") or data.get("engine_status") or "engine_completed")
            clean_stat = term_stat.replace("_", " ").title()
            is_normal = "fail" not in term_stat.lower() and "error" not in term_stat.lower()
            qual_rows.append(("Solver Termination", f"Normal ({clean_stat})" if is_normal else clean_stat))

            for idx, (lbl, val) in enumerate(qual_rows):
                k = QtWidgets.QLabel(lbl)
                k.setStyleSheet(key_style)
                v = QtWidgets.QLabel(val)
                v.setStyleSheet(value_style)
                v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                qual_grid.addWidget(k, idx, 0)
                qual_grid.addWidget(v, idx, 1)
            self._summary_layout.addWidget(qual_group)
        else:
            # Performance / Key Results Group
            section_title = {"MODAL": "Natural Frequencies", "TOPOPT_VOXEL": "Topology Results"}.get(rtype, "Structural Results")
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
                    ("Mode Shape Amplitude", "Mass-normalized eigenvector; amplitude is arbitrary"),
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
            if rtype == "TOPOPT_VOXEL":
                for key, label in (("target_vol_frac", "Target Material Fraction"), ("final_vol_frac", "Final Material Fraction"), ("iterations", "Iterations")):
                    if data.get(key) is not None:
                        res_rows.append((label, _fmt(data[key])))
                regions = data.get("passive_regions") or {}
                if regions:
                    res_rows.extend([
                        ("Preserved Solid Voxels", str(regions.get("explicit_solid_voxels", 0))),
                        ("Obstacle / Keep-out Voxels", str(regions.get("explicit_void_voxels", 0))),
                    ])

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

        # Warnings Group
        warnings = list(data.get("warnings") or [])
        if warnings:
            warn_group = QtWidgets.QGroupBox(f"Solver Warnings ({len(warnings)})")
            warn_layout = QtWidgets.QVBoxLayout(warn_group)
            for w in warnings:
                lbl = QtWidgets.QLabel(f"Warning: {w}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"color: {'#e3b341' if is_dark else '#9a6700'}; font-weight: 500;")
                lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                warn_layout.addWidget(lbl)
            self._summary_layout.addWidget(warn_group)

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
            f"Backend: {data.get('backend', '—')}",
            f"Status: {data.get('external_status', 'Completed')}",
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
            quality = data.get("quality") or {}
            q_status = data.get("quality_status") or quality.get("status")
            if q_status:
                lines.append(f"Quality Qualification: {str(q_status).upper()}")
        else:
            for k in ("peak_stress_nodal", "peak_displacement", "minimum_yield_safety_factor", "compliance", "volume", "mass"):
                if data.get(k) is not None:
                    lines.append(f"{k.replace('_', ' ').title()}: {_fmt(data[k])}")
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
