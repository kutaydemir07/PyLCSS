# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Standalone CAE workspace for CAD, meshing, simulation and optimization."""

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.cae.user_interface.cad import ProfessionalCadApp
from pylcss.user_interface.common.theme_manager import THEMES, apply_theme, current_theme


_MENUBAR_DARK_QSS = """
    QMenuBar {
        background-color: #181a1f;
        color: #d1d5db;
        border-bottom: 1px solid #282c34;
        padding: 2px 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 12px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 5px 12px;
        border-radius: 4px;
        color: #d1d5db;
    }
    QMenuBar::item:selected {
        background-color: #242933;
        color: #ffffff;
    }
    QMenuBar::item:pressed {
        background-color: #1c2027;
        color: #d29922;
    }
    QMenu {
        background-color: #21252b;
        border: 1px solid #323844;
        border-radius: 6px;
        padding: 4px;
        color: #e0e4ea;
    }
    QMenu::item {
        padding: 6px 24px 6px 16px;
        border-radius: 4px;
        font-size: 11px;
    }
    QMenu::item:selected {
        background-color: #2b3240;
        color: #d29922;
    }
    QMenu::separator {
        height: 1px;
        background: #2d333f;
        margin: 4px 6px;
    }
"""

_MENUBAR_LIGHT_QSS = """
    QMenuBar {
        background-color: #ffffff;
        color: #344054;
        border-bottom: 1px solid #e2e8f0;
        padding: 2px 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 12px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 5px 12px;
        border-radius: 4px;
        color: #344054;
    }
    QMenuBar::item:selected {
        background-color: #f1f5f9;
        color: #0f172a;
    }
    QMenuBar::item:pressed {
        background-color: #e2e8f0;
        color: #4f6f91;
    }
    QMenu {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 4px;
        color: #1e293b;
    }
    QMenu::item {
        padding: 6px 24px 6px 16px;
        border-radius: 4px;
        font-size: 11px;
    }
    QMenu::item:selected {
        background-color: #f1f5f9;
        color: #4f6f91;
    }
    QMenu::separator {
        height: 1px;
        background: #e2e8f0;
        margin: 4px 6px;
    }
"""


class _PageMenuBar(QtWidgets.QMenuBar):
    """Menu bar that underlines the checked stage action and draws separators."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        colors = THEMES[current_theme()]
        painter = QtGui.QPainter(self)
        for action in self.actions():
            rect = self.actionGeometry(action)
            if action.isSeparator():
                painter.setPen(QtGui.QColor(colors.get("border", "#282c34")))
                x = rect.center().x()
                painter.drawLine(x, rect.top() + 6, x, rect.bottom() - 6)
            elif action.isCheckable() and action.isChecked():
                painter.fillRect(
                    rect.left() + 4,
                    rect.bottom() - 2,
                    rect.width() - 8,
                    3,
                    QtGui.QColor(colors["primary"]),
                )
        painter.end()


class CAEMainWindow(QtWidgets.QMainWindow):
    """Main application window for the PyLCSS CAE workspace."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyLCSS CAE")
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parents[2] / "user_interface/icon.png")))
        self.resize(1600, 900)

        # Central stacked widget for Setup and Results pages
        self.stack = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.cad_widget = ProfessionalCadApp()
        self.stack.addWidget(self.cad_widget)  # Index 0: Setup

        from pylcss.cae.user_interface.cad.results_dialog import DetailedResultsWidget
        from pylcss.cae.user_interface.cad.cad_viewer import CQ3DViewer

        self.results_page = DetailedResultsWidget()
        self.results_workspace = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.results_viewer = CQ3DViewer(self.results_workspace)
        self.results_viewer.setObjectName("results_viewer")
        self.results_viewer.setMinimumWidth(380)
        self.results_viewer.setToolTip(
            "Interactive 3D results viewer. Use the view controls above the model to adjust the camera."
        )
        self.results_workspace.setObjectName("results_workspace")
        self.results_workspace.setChildrenCollapsible(False)
        self.results_workspace.addWidget(self.results_viewer)
        self.results_workspace.addWidget(self.results_page)
        self.results_workspace.setStretchFactor(0, 2)
        self.results_workspace.setStretchFactor(1, 3)
        self.results_workspace.setSizes([640, 960])
        self.stack.addWidget(self.results_workspace)  # Index 1: Results

        # Connect results page back to cad_widget
        self.results_page.switch_to_setup_requested.connect(
            lambda: self.set_page("setup")
        )
        self.results_page.view_mode_requested.connect(
            self.cad_widget._on_results_view_mode_requested
        )
        self.results_page.undeformed_requested.connect(
            self.cad_widget._on_results_undeformed_requested
        )
        self.results_page.extrema_requested.connect(
            self.cad_widget._on_results_extrema_requested
        )
        self.results_page.deform_scale_requested.connect(
            self.cad_widget._on_results_deform_scale_requested
        )
        # Keep the Setup viewer and the dedicated Results viewer in sync. The
        # Results viewer exposes the same camera toolbar and result controls.
        self.results_page.view_mode_requested.connect(self.results_viewer.set_result_field)
        self.results_page.undeformed_requested.connect(self.results_viewer.set_undeformed_overlay)
        self.results_page.extrema_requested.connect(self.results_viewer.set_result_extrema)
        self.results_page.deform_scale_requested.connect(self.results_viewer.set_deformation_scale)
        self.results_viewer.result_field_changed.connect(
            self.cad_widget._on_viewer_result_field_changed
        )

        # Connect cad_widget results to results_page
        self.cad_widget.results.result_received.connect(
            self._on_result_received
        )
        self.cad_widget.results.open_results_page_requested.connect(
            self._on_open_results_page_requested
        )

        self.menu_bar = _PageMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # File Menu
        self.file_menu = self.menu_bar.addMenu("File")

        act_new = self.file_menu.addAction("New Project")
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self.cad_widget._new_project)

        act_open = self.file_menu.addAction("Open Project…")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.cad_widget._open_project)

        act_examples = self.file_menu.addAction("Examples…")
        act_examples.triggered.connect(self.cad_widget._open_example)

        self.file_menu.addSeparator()

        act_save = self.file_menu.addAction("Save Project")
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.cad_widget._save_project)

        act_import = self.file_menu.addAction("Import CAD…")
        act_import.triggered.connect(self.cad_widget._import_cad)

        act_export = self.file_menu.addAction("Export Results…")
        act_export.triggered.connect(self.cad_widget._export_simulation_results)

        self.file_menu.addSeparator()

        act_exit = self.file_menu.addAction("Exit")
        act_exit.triggered.connect(self.close)

        # View Menu
        self.view_menu = self.menu_bar.addMenu("View")

        layout_menu = self.view_menu.addMenu("Layout Mode")
        layout_menu.addAction("3D Focus", lambda: self._set_layout("3d"))
        layout_menu.addAction("Split View", lambda: self._set_layout("split"))
        layout_menu.addAction("Graph Focus", lambda: self._set_layout("graph"))
        layout_menu.addSeparator()
        layout_menu.addAction("Toggle Split Orientation", self.cad_widget.toggle_split_orientation)

        theme_menu = self.view_menu.addMenu("Theme")
        self._theme_action_group = QtGui.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        self._theme_actions = {}
        for theme_name, label in (("dark", "Dark"), ("light", "Light")):
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setData(theme_name)
            action.triggered.connect(
                lambda checked=False, name=theme_name: self._set_theme(name)
            )
            self._theme_action_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme_name] = action

        # Help Menu
        self._help_dialog = None
        self.help_menu = self.menu_bar.addMenu("Help")
        help_action = self.help_menu.addAction("User Guide")
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)

        self.menu_bar.addSeparator()

        # Navigation Actions: Setup & Results
        self._page_action_group = QtGui.QActionGroup(self)
        self._page_action_group.setExclusive(True)
        self._page_actions = {}

        pages = (
            ("setup", "Setup", "Parametric CAD modeling, node graph, boundary conditions, and solvers"),
            ("results", "Results", "Simulation data tables, convergence graphs, and mechanical KPIs"),
        )
        for page_id, label, tip in pages:
            action = QtGui.QAction(f"  {label}  ", self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.setData(page_id)
            action.triggered.connect(
                lambda checked=False, p=page_id: self.set_page(p)
            )
            self._page_action_group.addAction(action)
            self.menu_bar.addAction(action)
            self._page_actions[page_id] = action

        # Default page: Setup
        self._page_actions["setup"].setChecked(True)

        self._set_theme(current_theme(), persist=False)

    def set_page(self, page_id: str) -> None:
        """Switch active page between 'setup' and 'results'."""
        page_id = page_id.lower()
        action = self._page_actions.get(page_id)
        if action:
            action.setChecked(True)

        if page_id == "results":
            self.stack.setCurrentIndex(1)
            last_data = getattr(self.cad_widget.results, "_last_data", None)
            measurements = getattr(self.cad_widget.results, "_measurements", [])
            if last_data:
                self._on_result_received(last_data, measurements)
        else:  # "setup"
            self.stack.setCurrentIndex(0)
        self.menu_bar.update()

    def set_stage(self, stage_id: str) -> None:
        """Compatibility method for stage switching."""
        stage_id = stage_id.lower()
        if stage_id in ("results", "insights"):
            self.set_page("results")
            self.cad_widget.set_stage(stage_id)
        else:
            self.set_page("setup")
            self.cad_widget.set_stage(stage_id)

    def _on_open_results_page_requested(self, initial_tab: str = "tables") -> None:
        """Handle Tables & Graphs button click from Setup inspector."""
        last_data = getattr(self.cad_widget.results, "_last_data", None)
        measurements = getattr(self.cad_widget.results, "_measurements", [])
        if last_data:
            self._on_result_received(last_data, measurements)
        self.results_page.select_tab(initial_tab)
        self.set_page("results")

    def _on_result_received(self, data, measurements=None) -> None:
        """Refresh the result panels and mirror the payload in the VTK pane."""
        self.results_page.update_result(data, measurements)
        if self.stack.currentWidget() is self.results_workspace and isinstance(data, dict) and data:
            self.results_viewer.render_simulation(data)

    def _set_layout(self, mode: str) -> None:
        self.set_page("setup")
        if hasattr(self, "cad_widget") and self.cad_widget is not None:
            self.cad_widget.set_layout_mode(mode)

    def _show_help(self) -> None:
        if self._help_dialog is None:
            from pylcss.user_interface.help import HelpWidget
            self._help_dialog = QtWidgets.QDialog(self)
            self._help_dialog.setWindowTitle("PyLCSS User Guide")
            self._help_dialog.resize(1050, 750)
            layout = QtWidgets.QVBoxLayout(self._help_dialog)
            self._help_widget = HelpWidget()
            layout.addWidget(self._help_widget)
        self._help_widget.apply_theme(current_theme())
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def _set_theme(self, name: str, *, persist: bool = True) -> None:
        theme_name = apply_theme(name, persist=persist)
        self.apply_theme(theme_name)

    def apply_theme(self, theme_name: str) -> None:
        """Apply theme to CAE window, menu bar, results page and children."""
        theme_name = str(theme_name).lower()
        for theme_key, action in self._theme_actions.items():
            action.setChecked(theme_key == theme_name)
        if hasattr(self, "cad_widget") and self.cad_widget is not None:
            self.cad_widget.apply_theme(theme_name)
        if hasattr(self, "results_page") and self.results_page is not None:
            if hasattr(self.results_page, "apply_theme"):
                self.results_page.apply_theme(theme_name)
        if hasattr(self, "results_viewer") and self.results_viewer is not None:
            self.results_viewer.apply_theme(theme_name)
        if self._help_dialog is not None:
            self._help_widget.apply_theme(theme_name)

        # Style the menu bar to match active theme
        self.menu_bar.setStyleSheet(
            _MENUBAR_DARK_QSS if theme_name == "dark" else _MENUBAR_LIGHT_QSS
        )
        self.setStyleSheet(
            f"QMainWindow {{ background-color: {'#181a1f' if theme_name == 'dark' else '#f8fafc'}; }}"
        )
        self.menu_bar.update()

    def save_project(self) -> None:
        self.cad_widget._save_project()

    def load_project(self) -> None:
        self.cad_widget._open_project()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self.cad_widget._ensure_idle_for_io("closing PyLCSS CAE"):
            event.ignore()
            return
        super().closeEvent(event)
        if event.isAccepted():
            self.cad_widget.viewer.shutdown()
            self.results_viewer.shutdown()
