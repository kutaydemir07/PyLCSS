# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Main application window and cross-feature UI orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import qtawesome as qta
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.mdao.system_modeling.model import SystemModel
from pylcss.user_interface.help import HelpWidget
from pylcss.mdao.user_interface.optimization import OptimizationWidget
from pylcss.user_interface.project_files import (
    save_project_manifest,
    sanitize_project_name,
    validate_project_folder,
)
from pylcss.mdao.user_interface.sensitivity import SensitivityAnalysisWidget
from pylcss.mdao.user_interface.solution_space import SolutionSpaceWidget
from pylcss.mdao.user_interface.surrogate import SurrogateTrainingWidget
from pylcss.mdao.user_interface.system_modeling import ModelingWidget
from pylcss.mdao.user_interface.system_modeling.merge_dialog import (
    validate_merge_connections,
)
from pylcss.user_interface.common import (
    LIGHT_WORKSPACE_PALETTES as ALL_WORKSPACE_PALETTES,
    THEMES,
    apply_theme,
    current_theme,
    retheme_node_graph,
    retheme_widget_styles,
)

LIGHT_WORKSPACE_PALETTES = (ALL_WORKSPACE_PALETTES[0], *ALL_WORKSPACE_PALETTES[2:])

logger = logging.getLogger(__name__)


class ModelBuildWorker(QtCore.QThread):
    """Import and merge generated model source outside the GUI thread."""

    result_signal = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def __init__(self, models, product_name):
        super().__init__()
        self.models = list(models)
        self.product_name = product_name

    def run(self):
        try:
            model = SystemModel.from_models(self.models, self.product_name)
        except Exception as exc:
            logger.exception("Could not build the generated system model")
            self.error_signal.emit(str(exc))
        else:
            self.result_signal.emit(model)


class _PageMenuBar(QtWidgets.QMenuBar):
    """Menu bar that underlines the checked page action and draws separators."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        colors = THEMES[current_theme()]
        painter = QtGui.QPainter(self)
        for action in self.actions():
            rect = self.actionGeometry(action)
            if action.isSeparator():
                painter.setPen(QtGui.QColor(colors["border"]))
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


class MainWindow(QtWidgets.QMainWindow):
    """
    MDAO workspace for system modeling and multidisciplinary optimization.

    This window provides menu-bar navigation to six main sections:
    - Modeling Environment: Visual node-based system modeling
    - Surrogate Training: Machine learning surrogate model training
    - Solution Space Analysis: Monte Carlo sampling and visualization
    - Optimization: Multi-objective optimization tools
    - Sensitivity Analysis: Global sensitivity analysis
    - Help: Comprehensive documentation and about information
    """

    def __init__(self) -> None:
        """
        Initialize the main application window with all components.

        Sets up the window title, size, theme, and creates all tab widgets
        for the different application modules.
        """
        super().__init__()

        # Window setup
        self.setWindowTitle("PyLCSS MDAO")

        # Use absolute path for icon to support running from any directory
        icon_path = Path(__file__).parents[2] / "user_interface" / "icon.png"
        self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.resize(1600, 900)
        self.setMinimumSize(
            1024, 768
        )  # Ensure window can be resized smaller than default

        # Menu Bar
        self.menu_bar = _PageMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_menu = self.menu_bar.addMenu("File")

        self.action_save_project = QtGui.QAction("Save Project", self)
        self.action_save_project.setShortcut("Ctrl+S")
        self.action_save_project.setToolTip(
            "Save the MDAO project and cached engineering results."
        )
        self.action_save_project.triggered.connect(self.save_project)
        self.file_menu.addAction(self.action_save_project)

        self.action_load_project = QtGui.QAction("Load Project", self)
        self.action_load_project.setShortcut("Ctrl+O")
        self.action_load_project.setToolTip("Load a complete PyLCSS project folder.")
        self.action_load_project.triggered.connect(self.load_project)
        self.file_menu.addAction(self.action_load_project)

        self.view_menu = self.menu_bar.addMenu("View")
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

        self._project_io_busy = False
        self._project_io_dialog = None
        self._model_build_worker = None

        # Central Widget setup
        self.central_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Tabs setup
        self.tabs: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self.tabs.setMovable(False)  # Prevent tab reordering
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.tabs.setStyleSheet("QTabWidget::pane { border: none; background: transparent; }")

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.content_layout.addWidget(self.tabs, 1)
        self.layout.addWidget(self.content_widget, 1)

        # 1. Modeling Tab
        self.modeling_widget: ModelingWidget = ModelingWidget()
        self.modeling_widget.build_requested.connect(self.transfer_model)
        tab_index = self.tabs.addTab(
            self.modeling_widget,
            qta.icon("fa5s.project-diagram"),
            "  Modeling",
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Visual node-based system modeling environment. Create and connect computational nodes to define mathematical relationships between design variables and system outputs.",
        )

        self.cad_widget = None

        # 3. Surrogate Training Tab (NEW)
        # Pass modeling_widget to it so it can access the graph nodes
        self.surrogate_widget: SurrogateTrainingWidget = SurrogateTrainingWidget(
            modeling_widget=self.modeling_widget
        )
        tab_index = self.tabs.addTab(
            self.surrogate_widget, qta.icon("fa5s.brain"), "  Surrogates"
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Train machine learning surrogate models to replace expensive computational models. Supports MLP, Random Forest, Gradient Boosting, Gaussian Process, and deep neural networks.",
        )

        # 4. Solution Space Tab
        self.sol_space_widget: SolutionSpaceWidget = SolutionSpaceWidget()
        tab_index = self.tabs.addTab(
            self.sol_space_widget, qta.icon("fa5s.chart-area"), "  Solution Space"
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Explore and visualize the design space through Monte Carlo sampling. Analyze feasibility regions, constraint boundaries, and solution distributions.",
        )

        # 5. Optimization Tab
        self.optimization_widget: OptimizationWidget = OptimizationWidget()
        tab_index = self.tabs.addTab(
            self.optimization_widget, qta.icon("fa5s.rocket"), "  Optimization"
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Perform single and multi-objective optimization using various algorithms (SLSQP, NSGA-II, etc.). Includes real-time convergence plotting and constraint analysis.",
        )

        # 6. Sensitivity Analysis Tab (NEW)
        self.sensitivity_widget: SensitivityAnalysisWidget = SensitivityAnalysisWidget(
            optimization_widget=self.optimization_widget
        )
        tab_index = self.tabs.addTab(
            self.sensitivity_widget,
            qta.icon("fa5s.chart-bar"),
            "  Sensitivity",
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Conduct global sensitivity analysis using Sobol indices to identify which design variables have the most influence on system outputs.",
        )

        # 7. Help Tab (NEW)
        self.help_widget: HelpWidget = HelpWidget()
        tab_index = self.tabs.addTab(
            self.help_widget, qta.icon("fa5s.question-circle"), "  Help"
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Documentation, tutorials, and information about PyLCSS features, system requirements, and usage guidelines.",
        )
        self._tab_icon_names = (
            "fa5s.project-diagram",
            "fa5s.brain",
            "fa5s.chart-area",
            "fa5s.rocket",
            "fa5s.chart-bar",
            "fa5s.question-circle",
        )

        # Connect tab change to refresh nodes automatically when switching to this tab
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Keep page state and refresh signals while navigating from the menu row.
        self.tabs.tabBar().hide()
        self._page_actions = []
        self._page_action_group = QtGui.QActionGroup(self)
        self._page_action_group.setExclusive(True)
        self.menu_bar.addSeparator()
        for index in range(self.tabs.count()):
            action = QtGui.QAction(self.tabs.tabText(index).strip(), self)
            action.setCheckable(True)
            action.setToolTip(self.tabs.tabToolTip(index))
            action.triggered.connect(
                lambda checked=False, page=index: self.tabs.setCurrentIndex(page)
            )
            if self.tabs.widget(index) is self.help_widget:
                action.setShortcut("F1")
            self._page_action_group.addAction(action)
            self.menu_bar.addAction(action)
            self._page_actions.append(action)

        self._set_theme(current_theme(), persist=False)

    def _set_theme(self, theme_name: str, *, persist: bool = True) -> None:
        """Apply a complete application theme, including non-Qt canvases."""
        theme_name = apply_theme(theme_name, persist=persist)
        self.apply_theme(theme_name)

    def apply_theme(self, theme_name: str) -> None:
        """Apply theme to MDAO window components."""
        for name, action in self._theme_actions.items():
            action.setChecked(name == theme_name)

        palette = THEMES[theme_name]
        self._apply_workspace_palettes(theme_name)
        self._refresh_main_tab_chrome(theme_name)
        graphs = [getattr(self.cad_widget, "graph", None)]
        current_system_graph = getattr(self.modeling_widget, "current_graph", None)
        if current_system_graph is not None:
            graphs.append(current_system_graph)
        manager = getattr(self.modeling_widget, "system_manager", None)
        for system in getattr(manager, "systems", []) if manager else []:
            graph = (
                system.get("graph")
                if isinstance(system, dict)
                else getattr(system, "graph", None)
            )
            if graph is not None:
                graphs.append(graph)
        for graph in graphs:
            if graph is None:
                continue
            retheme_node_graph(graph, theme_name)

        if self.cad_widget is not None and hasattr(self.cad_widget, "apply_theme"):
            try:
                self.cad_widget.apply_theme(theme_name)
            except Exception:
                pass

        themed_components = (
            ("system modeling", self.modeling_widget),
            ("help", self.help_widget),
            ("optimization plots", self.optimization_widget.plots_widget),
            ("sensitivity", self.sensitivity_widget),
            ("surrogate training", self.surrogate_widget),
        )
        for label, component in themed_components:
            try:
                component.apply_theme(theme_name)
            except Exception:
                logger.warning(
                    "Could not apply the %s theme to %s.",
                    theme_name,
                    label,
                    exc_info=True,
                )
        retheme_widget_styles(self, theme_name)

    def _apply_workspace_palettes(self, theme_name: str) -> None:
        """Apply a harmonious workspace palette across all tabs for the active theme."""
        light = str(theme_name).lower() == "light"
        if light:
            border = THEMES["light"]["border"]
            text = THEMES["light"]["text_main"]
            for index, workspace_palette in enumerate(LIGHT_WORKSPACE_PALETTES):
                widget = self.tabs.widget(index)
                if widget is None:
                    continue
                object_name = workspace_palette["object_name"]
                widget.setObjectName(object_name)
                background = workspace_palette["background"]
                accent = workspace_palette["accent"]
                accent_soft = workspace_palette["accent_soft"]
                widget.setStyleSheet(
                    f"""
                    QWidget#{object_name} {{
                        background: {background};
                        color: {text};
                    }}
                    QWidget#{object_name} QGroupBox {{
                        background: #ffffff;
                        border-color: {border};
                        color: {text};
                    }}
                    QWidget#{object_name} QGroupBox::title {{
                        color: {accent};
                    }}
                    QWidget#{object_name} QTabWidget::pane {{
                        background: #ffffff;
                        border-color: {border};
                    }}
                    QWidget#{object_name} QTabBar::tab:selected {{
                        color: {text};
                        border-bottom-color: {accent};
                    }}
                    QWidget#{object_name} QHeaderView::section {{
                        background: {accent_soft};
                        color: {text};
                        border-color: {border};
                    }}
                    """
                )
        else:
            c = THEMES["dark"]
            for index, workspace_palette in enumerate(LIGHT_WORKSPACE_PALETTES):
                widget = self.tabs.widget(index)
                if widget is None:
                    continue
                object_name = workspace_palette["object_name"]
                widget.setObjectName(object_name)
                widget.setStyleSheet(
                    f"""
                    QWidget#{object_name} {{
                        background: {c["bg_dark"]};
                        color: {c["text_main"]};
                    }}
                    QWidget#{object_name} QGroupBox {{
                        background: {c["bg_dark"]};
                        border: 1px solid {c["border"]};
                        color: {c["text_main"]};
                    }}
                    QWidget#{object_name} QGroupBox::title {{
                        color: {c["primary"]};
                    }}
                    QWidget#{object_name} QTabWidget::pane {{
                        background: {c["bg_dark"]};
                        border: 1px solid {c["border"]};
                    }}
                    QWidget#{object_name} QTabBar::tab {{
                        background: {c["bg_dark"]};
                        color: {c["text_dim"]};
                    }}
                    QWidget#{object_name} QTabBar::tab:selected {{
                        background: {c["bg_dark"]};
                        color: {c["text_main"]};
                        border-bottom: 2px solid {c["primary"]};
                    }}
                    QWidget#{object_name} QHeaderView::section {{
                        background: {c["bg_dark"]};
                        color: {c["text_main"]};
                        border: 1px solid {c["border"]};
                    }}
                    """
                )

    def _refresh_main_tab_chrome(
        self, theme_name: str, active_index: int | None = None
    ) -> None:
        """Keep the active page highlighted in the menu row."""
        if active_index is None:
            active_index = self.tabs.currentIndex()
        for index, action in enumerate(self._page_actions):
            action.setChecked(index == active_index)
            font = action.font()
            font.setBold(index == active_index)
            action.setFont(font)
        self.menu_bar.update()

    def _set_project_io_enabled(self, enabled: bool) -> None:
        self.action_save_project.setEnabled(enabled)
        self.action_load_project.setEnabled(enabled)

    def _run_project_steps(
        self,
        title: str,
        steps: Sequence[tuple[str, Callable[[], Any]]],
        success_message: str,
        error_title: str,
    ) -> None:
        if self._project_io_busy:
            QtWidgets.QMessageBox.information(
                self,
                "Project Operation In Progress",
                "Wait for the current project save/load operation to finish first.",
            )
            return

        self._project_io_busy = True
        self._set_project_io_enabled(False)

        dialog = QtWidgets.QProgressDialog(title, "Cancel", 0, len(steps), self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        self._project_io_dialog = dialog
        self.statusBar().showMessage(title)

        state = {"index": 0, "cancelled": False}

        def finish(success: bool, message: str = "") -> None:
            self._project_io_busy = False
            self._set_project_io_enabled(True)
            if self._project_io_dialog is not None:
                self._project_io_dialog.close()
                self._project_io_dialog.deleteLater()
                self._project_io_dialog = None

            if success:
                self.statusBar().showMessage(success_message, 4000)
                QtWidgets.QMessageBox.information(self, "Success", success_message)
            elif message:
                self.statusBar().showMessage(message, 5000)
                QtWidgets.QMessageBox.critical(self, error_title, message)

        def run_next_step() -> None:
            if state["cancelled"]:
                finish(False, f"{title} cancelled.")
                return

            index = state["index"]
            if index >= len(steps):
                finish(True)
                return

            label, func = steps[index]
            if self._project_io_dialog is not None:
                self._project_io_dialog.setLabelText(label)
                self._project_io_dialog.setValue(index)

            try:
                func()
            except Exception as exc:
                logger.exception("Project step failed: %s", label)
                finish(False, str(exc))
                return

            state["index"] += 1
            if self._project_io_dialog is not None:
                self._project_io_dialog.setValue(state["index"])
            QtCore.QTimer.singleShot(0, run_next_step)

        dialog.canceled.connect(lambda: state.__setitem__("cancelled", True))
        QtCore.QTimer.singleShot(0, run_next_step)

    def _collect_active_tasks(self) -> list[str]:
        tasks: list[str] = []

        if (
            hasattr(self.cad_widget, "_execution_is_active")
            and self.cad_widget._execution_is_active()
        ):
            tasks.append("CAD computation")

        optimization_worker = getattr(self.optimization_widget, "worker", None)
        if optimization_worker is not None and optimization_worker.isRunning():
            tasks.append("optimization")

        if (
            hasattr(self.sol_space_widget, "has_active_background_tasks")
            and self.sol_space_widget.has_active_background_tasks()
        ):
            tasks.append("solution-space analysis")

        sensitivity_worker = getattr(self.sensitivity_widget, "worker", None)
        if sensitivity_worker is not None and sensitivity_worker.isRunning():
            tasks.append("sensitivity analysis")

        refresh_worker = getattr(self.sensitivity_widget, "refresh_worker", None)
        if refresh_worker is not None and refresh_worker.isRunning():
            tasks.append("sensitivity refresh")

        model_build_worker = self._model_build_worker
        if model_build_worker is not None and model_build_worker.isRunning():
            tasks.append("model build")

        for attr_name, label in (
            ("gen_worker", "surrogate data generation"),
            ("worker", "surrogate training"),
            ("adaptive_worker", "adaptive surrogate training"),
            ("evaluation_worker", "surrogate evaluation"),
        ):
            thread = getattr(self.surrogate_widget, attr_name, None)
            if thread is not None and thread.isRunning():
                tasks.append(label)

        return tasks

    def _require_idle_workflows(self, action: str) -> bool:
        tasks = self._collect_active_tasks()
        if not tasks:
            return True
        QtWidgets.QMessageBox.information(
            self,
            "Background Tasks Running",
            f"Wait for {', '.join(tasks)} before {action}.",
        )
        return False

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        from pylcss.user_interface.background_shutdown import prepare_window_close

        prepare_window_close(self)
        super().closeEvent(event)

    def on_tab_changed(self, index: int) -> None:
        """Refresh node list when switching to Surrogate Tab and outputs for Sensitivity Tab."""
        if hasattr(self, "_tab_icon_names"):
            self._refresh_main_tab_chrome(current_theme(), index)
        current_widget = self.tabs.widget(index)
        if current_widget == self.cad_widget:
            QtCore.QTimer.singleShot(
                0, self.cad_widget.viewer.ensure_navigation_cube_visible
            )
        elif current_widget == self.surrogate_widget:
            self.surrogate_widget.refresh_nodes()
        elif current_widget == self.sensitivity_widget:
            self.sensitivity_widget.refresh_outputs()

    def transfer_model(self) -> None:
        """Build one executable model and forward it to the analysis tabs."""

        active_worker = self._model_build_worker
        if active_worker is not None and active_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Build Model", "A model build is already running."
            )
            return
        if not self._require_idle_workflows("building a new model"):
            return

        if not self.modeling_widget.validate_graph():
            return

        models = self.modeling_widget.get_compiled_code()
        if not models:
            return
        if len(models) > 1 and not validate_merge_connections(models, self):
            return

        product_name = (
            self.modeling_widget.system_manager.product_name.text().strip() or "Product"
        )
        worker = ModelBuildWorker(models, product_name)
        self._model_build_worker = worker
        self.modeling_widget.action_build.setEnabled(False)
        self.statusBar().showMessage("Building system model...")
        worker.result_signal.connect(self._on_model_built)
        worker.error_signal.connect(self._on_model_build_error)
        worker.finished.connect(self._on_model_build_stopped)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_model_built(self, system_model) -> None:
        try:
            self.sol_space_widget.load_models([system_model])
            self.optimization_widget.load_models([system_model])
            if self.sol_space_widget.problem is None:
                raise RuntimeError("Solution Space rejected the built model.")
            if self.optimization_widget.problem is None:
                raise RuntimeError("Optimization rejected the built model.")
        except Exception as exc:
            logger.exception("Could not transport the built system model")
            QtWidgets.QMessageBox.critical(
                self,
                "Build Error",
                f"Could not transport the built model:\n{exc}",
            )
            return

        self.tabs.setCurrentWidget(self.sol_space_widget)
        self.statusBar().showMessage("System model built successfully.", 4000)

    def _on_model_build_error(self, message) -> None:
        QtWidgets.QMessageBox.critical(
            self,
            "Build Error",
            f"Could not build the generated model:\n{message}",
        )

    def _on_model_build_stopped(self) -> None:
        self._model_build_worker = None
        self.modeling_widget.action_build.setEnabled(True)

    def save_project(self) -> None:
        """Save the entire project to a folder."""
        if self._project_io_busy:
            QtWidgets.QMessageBox.information(
                self,
                "Project Operation In Progress",
                "Wait for the current project operation to finish first.",
            )
            return
        if not self._require_idle_workflows("saving the project"):
            return

        parent_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Parent Folder for Project"
        )
        if not parent_folder:
            return

        product_name = self.modeling_widget.system_manager.product_name.text().strip()
        safe_name = sanitize_project_name(product_name)
        folder_path = str(Path(parent_folder) / safe_name)

        steps = [
            (
                "Saving modeling graph...",
                lambda: self.modeling_widget.save_graph_to_file(folder_path),
            ),
            (
                "Saving surrogate settings...",
                lambda: self.surrogate_widget.save_to_folder(folder_path),
            ),
            (
                "Saving solution-space data...",
                lambda: self.sol_space_widget.save_to_folder(folder_path),
            ),
            (
                "Saving optimization settings...",
                lambda: self.optimization_widget.save_to_folder(folder_path),
            ),
            (
                "Saving sensitivity settings...",
                lambda: self.sensitivity_widget.save_to_folder(folder_path),
            ),
            (
                "Writing project manifest...",
                lambda: self._save_project_manifest(folder_path),
            ),
        ]
        self._run_project_steps(
            "Saving project...",
            steps,
            f"Project saved successfully to:\n{folder_path}",
            "Save Error",
        )

    def load_project(self) -> None:
        """Load the entire project from a folder."""
        if self._project_io_busy:
            QtWidgets.QMessageBox.information(
                self,
                "Project Operation In Progress",
                "Wait for the current project operation to finish first.",
            )
            return
        if not self._require_idle_workflows("loading another project"):
            return

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Project Folder to Load"
        )
        if not folder_path:
            return
        try:
            self._validate_project_folder(folder_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Project Folder",
                str(exc),
            )
            return

        def transfer_solution_space_problem() -> None:
            if self.sol_space_widget.problem:
                if self.sol_space_widget.models:
                    self.optimization_widget.load_models(self.sol_space_widget.models)
                self.optimization_widget.set_problem(self.sol_space_widget.problem)
                self.optimization_widget.system_code = self.sol_space_widget.system_code

        steps = [
            (
                "Loading modeling graph...",
                lambda: self.modeling_widget.load_graph_from_file(folder_path),
            ),
            (
                "Loading surrogate settings...",
                lambda: self.surrogate_widget.load_from_folder(folder_path),
            ),
            (
                "Loading solution-space data...",
                lambda: self.sol_space_widget.load_from_folder(folder_path),
            ),
            ("Syncing optimization problem...", transfer_solution_space_problem),
            (
                "Loading optimization settings...",
                lambda: self.optimization_widget.load_from_folder(folder_path),
            ),
            (
                "Loading sensitivity settings...",
                lambda: self.sensitivity_widget.load_from_folder(folder_path),
            ),
        ]
        self._run_project_steps(
            "Loading project...",
            steps,
            "Project loaded successfully!",
            "Load Error",
        )

    @staticmethod
    def _save_project_manifest(folder_path: str | Path) -> None:
        """Compatibility wrapper for older callers of the window helper."""
        save_project_manifest(folder_path)

    @staticmethod
    def _validate_project_folder(folder_path: str | Path) -> None:
        """Compatibility wrapper for older callers of the window helper."""
        validate_project_folder(folder_path)
