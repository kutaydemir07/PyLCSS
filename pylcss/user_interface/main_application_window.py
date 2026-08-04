# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Main application window and cross-feature UI orchestration."""

from __future__ import annotations

import html
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import qtawesome as qta
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.assistant_systems import AssistantConfig, AssistantManager
from pylcss.system_modeling.model import SystemModel
from pylcss.user_interface.cad import ProfessionalCadApp
from pylcss.user_interface.help import HelpWidget
from pylcss.user_interface.optimization import OptimizationWidget
from pylcss.user_interface.project_files import (
    save_project_manifest,
    sanitize_project_name,
    validate_project_folder,
)
from pylcss.user_interface.sensitivity import SensitivityAnalysisWidget
from pylcss.user_interface.solution_space import SolutionSpaceWidget
from pylcss.user_interface.surrogate import SurrogateTrainingWidget
from pylcss.user_interface.system_modeling import ModelingWidget
from pylcss.user_interface.system_modeling.merge_dialog import (
    validate_merge_connections,
)
from pylcss.user_interface.common import (
    LIGHT_WORKSPACE_PALETTES,
    THEMES,
    apply_theme,
    current_theme,
    retheme_node_graph,
    retheme_widget_styles,
)

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


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window containing all major components.

    This window provides a tabbed interface with seven main sections:
    - Modeling Environment: Visual node-based system modeling
    - Design Studio: Parametric CAD modeling with 3D viewer
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
        self.setWindowTitle("PyLCSS")

        # Use absolute path for icon to support running from any directory
        icon_path = Path(__file__).with_name("icon.png")
        self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.resize(1600, 900)
        self.setMinimumSize(
            1024, 768
        )  # Ensure window can be resized smaller than default

        # Menu Bar
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("File")

        self.action_save_project = QtGui.QAction("Save Project", self)
        self.action_save_project.setShortcut("Ctrl+S")
        self.action_save_project.setToolTip(
            "Save the complete application project, including Design Studio "
            "graphs and cached engineering results."
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

        # Tabs setup
        self.tabs: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self.tabs.setMovable(False)  # Prevent tab reordering
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

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
            "  Modeling Environment",
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Visual node-based system modeling environment. Create and connect computational nodes to define mathematical relationships between design variables and system outputs.",
        )

        # --- ADD NEW TAB HERE ---
        # 2. Design Studio Tab
        self.cad_widget = ProfessionalCadApp()
        tab_index = self.tabs.addTab(
            self.cad_widget, qta.icon("fa5s.cube"), "  Design Studio"
        )
        self.tabs.setTabToolTip(
            tab_index,
            "Parametric CAD modeling, simulation setup, and 3D result visualization.",
        )

        # 3. Surrogate Training Tab (NEW)
        # Pass modeling_widget to it so it can access the graph nodes
        self.surrogate_widget: SurrogateTrainingWidget = SurrogateTrainingWidget(
            modeling_widget=self.modeling_widget
        )
        tab_index = self.tabs.addTab(
            self.surrogate_widget, qta.icon("fa5s.brain"), "  Surrogate Training"
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
            "  Sensitivity Analysis",
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
            "fa5s.cube",
            "fa5s.brain",
            "fa5s.chart-area",
            "fa5s.rocket",
            "fa5s.chart-bar",
            "fa5s.question-circle",
        )

        # Connect tab change to refresh nodes automatically when switching to this tab
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Style the TabWidget specifically for main navigation
        self.tabs.setIconSize(QtCore.QSize(20, 20))

        # --- ASSISTANT CONTROL SETUP ---
        self._setup_assistant_systems()
        self._set_theme(current_theme(), persist=False)

    def _set_theme(self, theme_name: str, *, persist: bool = True) -> None:
        """Apply a complete application theme, including non-Qt canvases."""
        theme_name = apply_theme(theme_name, persist=persist)
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

        themed_components = (
            ("system modeling", self.modeling_widget),
            ("Design Studio", self.cad_widget),
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
        try:
            self.assistant_toggle_btn.setIcon(
                qta.icon("fa5s.robot", color=palette["text_main"])
            )
            self.assistant_settings_btn.setIcon(
                qta.icon("fa5s.cog", color=palette["text_main"])
            )
            self._apply_assistant_document_theme(theme_name)
        except Exception:
            logger.warning("Could not retheme the assistant panel.", exc_info=True)
        retheme_widget_styles(self, theme_name)

    def _apply_workspace_palettes(self, theme_name: str) -> None:
        """Apply one harmonious light workspace system without touching dark mode."""
        light = str(theme_name).lower() == "light"
        border = THEMES["light"]["border"]
        text = THEMES["light"]["text_main"]
        for index, workspace_palette in enumerate(LIGHT_WORKSPACE_PALETTES):
            widget = self.tabs.widget(index)
            if widget is None:
                continue
            object_name = workspace_palette["object_name"]
            widget.setObjectName(object_name)
            if not light:
                widget.setStyleSheet("")
                continue
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

    def _refresh_main_tab_chrome(
        self, theme_name: str, active_index: int | None = None
    ) -> None:
        """Accent only the active light tab; retain the original dark tab chrome."""
        theme_name = str(theme_name).lower()
        if active_index is None:
            active_index = self.tabs.currentIndex()
        palette = THEMES[theme_name]
        for index, icon_name in enumerate(self._tab_icon_names):
            if theme_name == "light" and index == active_index:
                icon_color = LIGHT_WORKSPACE_PALETTES[index]["accent"]
            elif theme_name == "light":
                icon_color = "#667085"
            else:
                icon_color = palette["text_main"]
            self.tabs.setTabIcon(index, qta.icon(icon_name, color=icon_color))

        if theme_name == "light":
            accent = LIGHT_WORKSPACE_PALETTES[active_index]["accent"]
            self.tabs.tabBar().setStyleSheet(
                "QTabBar::tab:selected {"
                f"border-bottom: 2px solid {accent};"
                f"color: {palette['text_main']};"
                "}"
            )
        else:
            self.tabs.tabBar().setStyleSheet("")

    def _apply_assistant_document_theme(self, theme_name: str) -> None:
        if not hasattr(self, "assistant_log"):
            return
        palette = THEMES[str(theme_name).lower()]
        user_color = "#0969da" if theme_name == "light" else "#a8c7ff"
        self.assistant_log.document().setDefaultStyleSheet(
            f".message {{ color: {palette['text_main']}; }}"
            f".assistant {{ color: {palette['primary']}; }}"
            f".user {{ color: {user_color}; }}"
            f".error {{ color: {palette['danger']}; }}"
        )

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
        active_tasks = self._collect_active_tasks()
        if self._project_io_busy:
            active_tasks.append("project save/load")

        if active_tasks:
            task_text = ", ".join(active_tasks)
            QtWidgets.QMessageBox.information(
                self,
                "Background Tasks Running",
                f"Wait for these tasks to finish before closing the application: {task_text}.",
            )
            event.ignore()
            return

        super().closeEvent(event)

    def _setup_assistant_systems(self) -> None:
        """Initialize assistant systems behind the compact side panel."""
        self._assistant_use_side_panel = True

        # Create assistant manager
        self.assistant_manager = AssistantManager(main_window=self)

        self._setup_assistant_panel()
        self.assistant_manager.status_changed.connect(self._on_assistant_status)
        self.assistant_manager.error_occurred.connect(self._on_assistant_error)
        self.assistant_manager.agentic_progress.connect(self._on_assistant_progress)
        self.assistant_manager.agentic_result_received.connect(
            self._on_assistant_agentic_result
        )
        self.assistant_manager.agentic_error_received.connect(self._on_assistant_error)

        # Initialize in background to not block startup
        QtCore.QTimer.singleShot(1000, self.assistant_manager.initialize)

    def _setup_assistant_panel(self) -> None:
        """Create the floating assistant button and fixed side panel."""
        self.assistant_toggle_btn = QtWidgets.QToolButton(self.central_widget)
        self.assistant_toggle_btn.setIcon(qta.icon("fa5s.robot", color="#dce8ff"))
        self.assistant_toggle_btn.setToolTip("AI Assistant")
        self.assistant_toggle_btn.setFixedSize(38, 34)
        self.assistant_toggle_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.assistant_toggle_btn.setStyleSheet("""
            QToolButton {
                background: rgba(35, 43, 58, 235);
                border: 1px solid rgba(120, 155, 210, 180);
                border-radius: 8px;
            }
            QToolButton:hover {
                background: rgba(58, 78, 110, 245);
            }
        """)
        self.assistant_toggle_btn.clicked.connect(self._toggle_assistant_panel)

        self.assistant_panel = QtWidgets.QFrame()
        self.assistant_panel.setObjectName("assistant_panel")
        self.assistant_panel.setFixedWidth(320)
        self.assistant_panel.setStyleSheet("""
            #assistant_panel {
                background: rgba(28, 32, 40, 245);
                border: 1px solid rgba(120, 155, 210, 160);
                border-radius: 8px;
            }
            QLabel { color: #dce4f2; }
            QTextEdit, QLineEdit {
                background: #20242d;
                color: #eef3ff;
                border: 1px solid #3f4b61;
                border-radius: 5px;
                padding: 7px;
            }
            QPushButton {
                background: #34445d;
                color: #edf4ff;
                border: 1px solid #526783;
                border-radius: 5px;
                padding: 7px 10px;
            }
            QPushButton:hover { background: #405572; }
            QPushButton:checked {
                background: #2f6f55;
                border-color: #50b982;
            }
        """)

        panel_layout = QtWidgets.QVBoxLayout(self.assistant_panel)
        panel_layout.setContentsMargins(12, 10, 12, 12)
        panel_layout.setSpacing(8)

        header_widget = QtWidgets.QWidget(self.assistant_panel)
        header = QtWidgets.QHBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("AI Assistant")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self.assistant_settings_btn = QtWidgets.QToolButton()
        self.assistant_settings_btn.setIcon(qta.icon("fa5s.cog", color="#dce4f2"))
        self.assistant_settings_btn.setToolTip("Assistant settings")
        self.assistant_settings_btn.clicked.connect(self._open_llm_settings)
        header.addWidget(self.assistant_settings_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setText("X")
        close_btn.setToolTip("Close assistant")
        close_btn.clicked.connect(lambda: self._set_assistant_panel_visible(False))
        header.addWidget(close_btn)
        panel_layout.addWidget(header_widget)

        self.assistant_status_label = QtWidgets.QLabel("Ready")
        self.assistant_status_label.setStyleSheet("color: #8ea0ba; font-size: 11px;")
        panel_layout.addWidget(self.assistant_status_label)

        self.assistant_log = QtWidgets.QTextEdit()
        self.assistant_log.setReadOnly(True)
        self.assistant_log.setMinimumHeight(180)
        panel_layout.addWidget(self.assistant_log, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.assistant_input = QtWidgets.QLineEdit()
        self.assistant_input.setPlaceholderText("Ask the assistant...")
        self.assistant_input.returnPressed.connect(self._send_assistant_text)
        input_row.addWidget(self.assistant_input, 1)

        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self._send_assistant_text)
        input_row.addWidget(send_btn)
        panel_layout.addLayout(input_row)

        self.content_layout.addWidget(self.assistant_panel)
        self.assistant_panel.hide()
        self.assistant_toggle_btn.raise_()
        self._position_assistant_panel()

    def _set_assistant_panel_visible(self, visible: bool) -> None:
        if not hasattr(self, "assistant_panel"):
            return
        if visible:
            self.assistant_panel.setVisible(True)
            self.assistant_toggle_btn.hide()
            self._position_assistant_panel()
            self.assistant_input.setFocus()
            return

        self.assistant_panel.setVisible(False)
        self.assistant_toggle_btn.show()
        self.assistant_toggle_btn.raise_()
        self._position_assistant_panel()

    def _position_assistant_panel(self) -> None:
        if not hasattr(self, "assistant_toggle_btn"):
            return

        margin = 12
        self.assistant_toggle_btn.move(
            max(
                margin,
                self.central_widget.width()
                - self.assistant_toggle_btn.width()
                - margin,
            ),
            margin,
        )

        if not hasattr(self, "assistant_panel") or not self.assistant_panel.isVisible():
            self.assistant_toggle_btn.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_assistant_panel()

    def _toggle_assistant_panel(self) -> None:
        self._set_assistant_panel_visible(not self.assistant_panel.isVisible())

    def _append_assistant_message(
        self, speaker: str, message: str, error: bool = False
    ) -> None:
        if not hasattr(self, "assistant_log"):
            return
        css_class = "error" if error else ("user" if speaker == "You" else "assistant")
        safe_speaker = html.escape(speaker)
        safe_message = html.escape(str(message)).replace("\n", "<br>")
        self.assistant_log.append(
            f'<p><b class="{css_class}">{safe_speaker}:</b> '
            f'<span class="message">{safe_message}</span></p>'
        )

    def _send_assistant_text(self) -> None:
        message = self.assistant_input.text().strip()
        if not message:
            return
        if not self._confirm_cloud_assistant_data_use():
            return
        self.assistant_input.clear()
        self._append_assistant_message("You", message)
        self.assistant_status_label.setText("Thinking...")
        self._set_assistant_panel_visible(True)

        if not self.assistant_manager.initialize():
            self._on_assistant_error("Assistant could not initialize.")
            return
        self.assistant_manager.process_agentic_request(message)

    def _confirm_cloud_assistant_data_use(self) -> bool:
        """Obtain informed consent before the first cloud-provider request."""
        llm = self.assistant_manager.config.llm_control
        provider = str(llm.provider or "").lower()
        if provider == "local" or llm.cloud_data_notice_acknowledged:
            return True

        provider_name = provider.title() or "cloud provider"
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Send engineering data to a cloud provider?")
        box.setText(f"PyLCSS will send this request to {provider_name}.")
        box.setInformativeText(
            "The request may include recent conversation history (if memory is "
            "enabled), the active Modeling or CAD graph, node names and properties, "
            "connections, and tool results. The provider's terms and privacy policy "
            "apply. Do not send confidential or personal data unless authorized."
        )
        send_button = box.addButton(
            f"Send to {provider_name}",
            QtWidgets.QMessageBox.AcceptRole,
        )
        box.addButton(QtWidgets.QMessageBox.Cancel)
        remember = QtWidgets.QCheckBox("Do not show this warning again")
        box.setCheckBox(remember)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is not send_button:
            return False

        if remember.isChecked():
            llm.cloud_data_notice_acknowledged = True
            self.assistant_manager.config.save()
        return True

    def _on_assistant_status(self, status: str) -> None:
        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText(status)

    def _on_assistant_progress(self, message: str) -> None:
        self._on_assistant_status(message)

    def _on_assistant_agentic_result(self, result: dict, _original_text: str) -> None:
        message = result.get("message", "Completed.")
        self._append_assistant_message(
            "Assistant", message, error=not result.get("success", False)
        )
        self._on_assistant_status("Ready")

    def _on_assistant_error(self, message: str) -> None:
        self._append_assistant_message("Assistant", message, error=True)
        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText("Error")

    def _open_llm_settings(self) -> None:
        """Open the LLM configuration dialog."""
        from pylcss.user_interface.assistant import LLMConfigDialog

        dialog = LLMConfigDialog(self)
        if dialog.exec():
            # Reload manager's provider after settings change
            self.assistant_manager.update_config(AssistantConfig.load())
            self.statusBar().showMessage("LLM settings updated", 3000)

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
                "Saving Design Studio workflows and results...",
                lambda: self.cad_widget.save_to_folder(folder_path),
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
                "Loading Design Studio workflows and results...",
                lambda: self.cad_widget.load_from_folder(folder_path),
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
