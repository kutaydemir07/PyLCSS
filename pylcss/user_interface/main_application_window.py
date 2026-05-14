# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Main application window for PyLCSS.

This module contains the MainWindow class which serves as the primary
interface for the application, providing tabs for modeling, solution
space analysis, optimization, and surrogate training.
"""

from PySide6 import QtWidgets, QtCore, QtGui
import qtawesome as qta

from pylcss.user_interface.system_modeling import ModelingWidget
from pylcss.system_modeling.system_model import SystemModel
from pylcss.system_modeling.model_merge import validate_merge_connections
from pylcss.system_modeling.graph_validation import validate_graph
from pylcss.user_interface.solution_space import SolutionSpaceWidget
from pylcss.user_interface.optimization import OptimizationWidget
from pylcss.user_interface.common import apply_professional_theme

# --- NEW IMPORTS ---
from pylcss.user_interface.surrogate import SurrogateTrainingWidget
from pylcss.user_interface.sensitivity import SensitivityAnalysisWidget
from pylcss.user_interface.help import HelpWidget

# --- NEW IMPORT ---
from pylcss.user_interface.cad import ProfessionalCadApp  # Import the widget

# --- HANDS-FREE IMPORTS ---
from pylcss.assistant_systems import AssistantManager, AssistantConfig

# --- I/O & MATH IMPORTS ---
import os
import logging
import re
import html

logger = logging.getLogger(__name__)

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
        super(MainWindow, self).__init__()

        # Window setup
        self.setWindowTitle("PyLCSS")
        
        # Use absolute path for icon to support running from any directory
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "icon.png")
        self.setWindowIcon(QtGui.QIcon(icon_path))
        
        self.resize(1600, 900)
        self.setMinimumSize(1024, 768) # Ensure window can be resized smaller than default

        # Apply Modern Professional Theme
        apply_professional_theme()

        # Menu Bar
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("File")
        
        self.action_save_project = QtGui.QAction("Save Project", self)
        self.action_save_project.setShortcut("Ctrl+S")
        self.action_save_project.triggered.connect(self.save_project)
        self.file_menu.addAction(self.action_save_project)
        
        self.action_load_project = QtGui.QAction("Load Project", self)
        self.action_load_project.setShortcut("Ctrl+O")
        self.action_load_project.triggered.connect(self.load_project)
        self.file_menu.addAction(self.action_load_project)

        self._project_io_busy = False
        self._project_io_dialog = None

        # Central Widget setup
        self.central_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(self.central_widget)

        # Tabs setup
        self.tabs: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self.tabs.setMovable(False)  # Prevent tab reordering
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.content_layout.addWidget(self.tabs, 1)
        self.layout.addWidget(self.content_widget, 1)

        # 1. Modeling Tab
        self.modeling_widget: ModelingWidget = ModelingWidget()
        self.modeling_widget.build_requested.connect(self.transfer_model)
        tab_index = self.tabs.addTab(self.modeling_widget, qta.icon('fa5s.project-diagram'), "  Modeling Environment")
        self.tabs.setTabToolTip(tab_index, "Visual node-based system modeling environment. Create and connect computational nodes to define mathematical relationships between design variables and system outputs.")

        # --- ADD NEW TAB HERE ---
        # 2. Design Studio Tab
        self.cad_widget = ProfessionalCadApp()
        tab_index = self.tabs.addTab(self.cad_widget, qta.icon('fa5s.cube'), "  Design Studio")
        self.tabs.setTabToolTip(tab_index, "Parametric CAD modeling, simulation setup, and 3D result visualization.")

        # 3. Surrogate Training Tab (NEW)
        # Pass modeling_widget to it so it can access the graph nodes
        self.surrogate_widget: SurrogateTrainingWidget = SurrogateTrainingWidget(modeling_widget=self.modeling_widget)
        tab_index = self.tabs.addTab(self.surrogate_widget, qta.icon('fa5s.brain'), "  Surrogate Training")
        self.tabs.setTabToolTip(tab_index, "Train machine learning surrogate models to replace expensive computational models. Supports MLP, Random Forest, Gradient Boosting, Gaussian Process, and deep neural networks.")
        
        # 4. Solution Space Tab
        self.sol_space_widget: SolutionSpaceWidget = SolutionSpaceWidget()
        tab_index = self.tabs.addTab(self.sol_space_widget, qta.icon('fa5s.chart-area'), "  Solution Space")
        self.tabs.setTabToolTip(tab_index, "Explore and visualize the design space through Monte Carlo sampling. Analyze feasibility regions, constraint boundaries, and solution distributions.")
        
        # 5. Optimization Tab
        self.optimization_widget: OptimizationWidget = OptimizationWidget()
        tab_index = self.tabs.addTab(self.optimization_widget, qta.icon('fa5s.rocket'), "  Optimization")
        self.tabs.setTabToolTip(tab_index, "Perform single and multi-objective optimization using various algorithms (SLSQP, NSGA-II, etc.). Includes real-time convergence plotting and constraint analysis.")
        
        # 6. Sensitivity Analysis Tab (NEW)
        self.sensitivity_widget: SensitivityAnalysisWidget = SensitivityAnalysisWidget(optimization_widget=self.optimization_widget)
        tab_index = self.tabs.addTab(self.sensitivity_widget, qta.icon('fa5s.chart-bar'), "  Sensitivity Analysis")
        self.tabs.setTabToolTip(tab_index, "Conduct global sensitivity analysis using Sobol indices to identify which design variables have the most influence on system outputs.")
        
        # 7. Help Tab (NEW)
        self.help_widget: HelpWidget = HelpWidget()
        tab_index = self.tabs.addTab(self.help_widget, qta.icon('fa5s.question-circle'), "  Help")
        self.tabs.setTabToolTip(tab_index, "Documentation, tutorials, and information about PyLCSS features, system requirements, and usage guidelines.")
        
        # Connect tab change to refresh nodes automatically when switching to this tab
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Style the TabWidget specifically for main navigation
        self.tabs.setIconSize(QtCore.QSize(20, 20))
        
        # --- ASSISTANT CONTROL SETUP ---
        self._setup_assistant_systems()

    def _set_project_io_enabled(self, enabled: bool) -> None:
        self.action_save_project.setEnabled(enabled)
        self.action_load_project.setEnabled(enabled)

    def _run_project_steps(self, title: str, steps, success_message: str, error_title: str) -> None:
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

        state = {'index': 0, 'cancelled': False}

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
            if state['cancelled']:
                finish(False, f"{title} cancelled.")
                return

            index = state['index']
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
                finish(False, str(exc))
                return

            state['index'] += 1
            if self._project_io_dialog is not None:
                self._project_io_dialog.setValue(state['index'])
            QtCore.QTimer.singleShot(0, run_next_step)

        dialog.canceled.connect(lambda: state.__setitem__('cancelled', True))
        QtCore.QTimer.singleShot(0, run_next_step)

    def _collect_active_tasks(self):
        tasks = []

        if hasattr(self.cad_widget, '_execution_is_active') and self.cad_widget._execution_is_active():
            tasks.append("CAD computation")

        optimization_worker = getattr(self.optimization_widget, 'worker', None)
        if optimization_worker is not None and optimization_worker.isRunning():
            tasks.append("optimization")

        if hasattr(self.sol_space_widget, 'has_active_background_tasks') and self.sol_space_widget.has_active_background_tasks():
            tasks.append("solution-space analysis")

        sensitivity_worker = getattr(self.sensitivity_widget, 'worker', None)
        if sensitivity_worker is not None and sensitivity_worker.isRunning():
            tasks.append("sensitivity analysis")

        refresh_worker = getattr(self.sensitivity_widget, 'refresh_worker', None)
        if refresh_worker is not None and refresh_worker.isRunning():
            tasks.append("sensitivity refresh")

        for attr_name, label in (
            ('gen_worker', 'surrogate data generation'),
            ('worker', 'surrogate training'),
            ('adaptive_worker', 'adaptive surrogate training'),
        ):
            thread = getattr(self.surrogate_widget, attr_name, None)
            if thread is not None and thread.isRunning():
                tasks.append(label)

        return tasks

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
        self.assistant_manager.llm_request_received.connect(self._on_assistant_voice_request)
        self.assistant_manager.llm_response_received.connect(self._on_assistant_llm_response)
        self.assistant_manager.llm_error_occurred.connect(self._on_assistant_llm_error)
        self.assistant_manager.agentic_progress.connect(self._on_assistant_progress)
        self.assistant_manager.agentic_result_received.connect(self._on_assistant_agentic_result)
        self.assistant_manager.agentic_error_received.connect(self._on_assistant_error)

        # Initialize in background to not block startup
        QtCore.QTimer.singleShot(1000, self.assistant_manager.initialize)

    def _setup_assistant_panel(self) -> None:
        """Create the floating assistant button and fixed side panel."""
        self.assistant_toggle_btn = QtWidgets.QToolButton(self.central_widget)
        self.assistant_toggle_btn.setIcon(qta.icon('fa5s.robot', color='#dce8ff'))
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

        settings_btn = QtWidgets.QToolButton()
        settings_btn.setIcon(qta.icon('fa5s.cog', color='#dce4f2'))
        settings_btn.setToolTip("Assistant settings")
        settings_btn.clicked.connect(self._open_llm_settings)
        header.addWidget(settings_btn)

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

        voice_row = QtWidgets.QHBoxLayout()
        self.assistant_voice_btn = QtWidgets.QPushButton("Voice")
        self.assistant_voice_btn.setCheckable(True)
        self.assistant_voice_btn.setIcon(qta.icon('fa5s.microphone'))
        self.assistant_voice_btn.toggled.connect(self._toggle_assistant_voice)
        voice_row.addWidget(self.assistant_voice_btn)

        hint = QtWidgets.QLabel("Speech is sent to the assistant as natural language.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8ea0ba; font-size: 11px;")
        voice_row.addWidget(hint, 1)
        panel_layout.addLayout(voice_row)

        self.content_layout.addWidget(self.assistant_panel)
        self.assistant_panel.hide()
        self.assistant_toggle_btn.raise_()
        self._position_assistant_panel()

    def _set_assistant_panel_visible(self, visible: bool) -> None:
        if not hasattr(self, 'assistant_panel'):
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
        if not hasattr(self, 'assistant_toggle_btn'):
            return

        margin = 12
        self.assistant_toggle_btn.move(
            max(margin, self.central_widget.width() - self.assistant_toggle_btn.width() - margin),
            margin,
        )

        if not hasattr(self, 'assistant_panel') or not self.assistant_panel.isVisible():
            self.assistant_toggle_btn.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_assistant_panel()

    def _toggle_assistant_panel(self) -> None:
        self._set_assistant_panel_visible(not self.assistant_panel.isVisible())

    def _append_assistant_message(self, speaker: str, message: str, error: bool = False) -> None:
        if not hasattr(self, 'assistant_log'):
            return
        color = '#ff9d9d' if error else ('#a8c7ff' if speaker == 'You' else '#dce4f2')
        safe_speaker = html.escape(speaker)
        safe_message = html.escape(str(message)).replace('\n', '<br>')
        self.assistant_log.append(
            f'<p><b style="color:{color};">{safe_speaker}:</b> '
            f'<span style="color:#eef3ff;">{safe_message}</span></p>'
        )

    def _send_assistant_text(self) -> None:
        message = self.assistant_input.text().strip()
        if not message:
            return
        self.assistant_input.clear()
        self._append_assistant_message("You", message)
        self.assistant_status_label.setText("Thinking...")
        self._set_assistant_panel_visible(True)

        if not self.assistant_manager.initialize():
            self._on_assistant_error("Assistant could not initialize.")
            return
        self.assistant_manager.process_agentic_request(message)

    def _toggle_assistant_voice(self, checked: bool) -> None:
        if checked:
            self.assistant_manager.config.voice_control.enabled = True
            if (
                getattr(self.assistant_manager, '_initialized', False)
                and getattr(self.assistant_manager, '_voice_controller', None) is None
            ):
                self.assistant_manager._initialized = False
            if self.assistant_manager.start():
                self.assistant_voice_btn.setText("Voice On")
                self.assistant_status_label.setText("Listening...")
            else:
                self.assistant_voice_btn.blockSignals(True)
                self.assistant_voice_btn.setChecked(False)
                self.assistant_voice_btn.blockSignals(False)
                self.assistant_status_label.setText("Voice unavailable")
        else:
            self.assistant_manager.stop()
            self.assistant_voice_btn.setText("Voice")
            self.assistant_status_label.setText("Voice stopped")

    def _on_assistant_status(self, status: str) -> None:
        if hasattr(self, 'assistant_status_label'):
            self.assistant_status_label.setText(status)

    def _on_assistant_progress(self, message: str) -> None:
        self._on_assistant_status(message)

    def _on_assistant_voice_request(self, text: str) -> None:
        self._set_assistant_panel_visible(True)
        self._append_assistant_message("You", text)

    def _on_assistant_agentic_result(self, result: dict, _original_text: str) -> None:
        message = result.get("message", "Completed.")
        self._append_assistant_message("Assistant", message, error=not result.get("success", False))
        self._on_assistant_status("Ready")

    def _on_assistant_llm_response(self, completion) -> None:
        self._append_assistant_message("Assistant", getattr(completion, 'content', completion))
        self._on_assistant_status("Ready")

    def _on_assistant_llm_error(self, error: Exception) -> None:
        self._on_assistant_error(str(error))

    def _on_assistant_error(self, message: str) -> None:
        self._append_assistant_message("Assistant", message, error=True)
        if hasattr(self, 'assistant_status_label'):
            self.assistant_status_label.setText("Error")

    def _toggle_voice_control(self, checked: bool) -> None:
        """Backward-compatible voice toggle routed through the side panel."""
        if hasattr(self, 'assistant_voice_btn'):
            self.assistant_voice_btn.setChecked(checked)
            
    def _open_llm_chat(self) -> None:
        """Backward-compatible chat entry point routed through the side panel."""
        if hasattr(self, 'assistant_panel') and not self.assistant_panel.isVisible():
            self._toggle_assistant_panel()
    
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
        current_widget = self.tabs.widget(index)
        if current_widget == self.surrogate_widget:
            self.surrogate_widget.refresh_nodes()
        elif current_widget == self.sensitivity_widget:
            self.sensitivity_widget.refresh_outputs()

    def transfer_model(self) -> None:
        """
        Transfer compiled models from modeling widget to analysis widgets.

        This method retrieves compiled code from the modeling environment,
        handles model merging for multiple systems, and loads the resulting
        models into the solution space and optimization widgets.
        """
        # Validate the graph before building
        if not validate_graph(self.modeling_widget):
            return  # Validation failed, don't build
        
        models = self.modeling_widget.get_compiled_code()
        if models:
            if len(models) > 1:
                # Show merge validation dialog
                if not validate_merge_connections(models, self):
                    return  # User cancelled

                # Create merged SystemModel
                try:
                    product_name = self.modeling_widget.system_manager.product_name.text().strip()
                    if not product_name:
                        product_name = "Product"
                    merged = SystemModel.from_models(models, product_name)
                    # Transfer the merged SystemModel
                    self.sol_space_widget.load_models([merged])
                    self.optimization_widget.load_models([merged])
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Build Error", f"Could not create merged model:\n{e}")
                    # Fallback: create individual SystemModels
                    system_models = []
                    for model in models:
                        try:
                            system_models.append(SystemModel.from_code_string(
                                model['name'], model['code'], model['inputs'], model['outputs']
                            ))
                        except Exception as e2:
                            QtWidgets.QMessageBox.warning(self, "Build Warning", f"Could not create SystemModel for {model['name']}:\n{e2}")
                    if system_models:
                        self.sol_space_widget.load_models(system_models)
                        self.optimization_widget.load_models(system_models)
            else:
                # Single model - create SystemModel
                model = models[0]
                try:
                    system_model = SystemModel.from_code_string(
                        model['name'], model['code'], model['inputs'], model['outputs']
                    )
                    self.sol_space_widget.load_models([system_model])
                    self.optimization_widget.load_models([system_model])
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Build Error", f"Could not create SystemModel:\n{e}")
                    # Fallback to old method
                    self.sol_space_widget.load_models(models)
                    self.optimization_widget.load_models(models)
            # Switch to Solution Space tab (Index 3)
            self.tabs.setCurrentIndex(3)

    def save_project(self):
        """Save the entire project to a folder."""
        if self._project_io_busy:
            QtWidgets.QMessageBox.information(self, "Project Operation In Progress", "Wait for the current project operation to finish first.")
            return

        parent_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Parent Folder for Project")
        if not parent_folder:
            return

        product_name = self.modeling_widget.system_manager.product_name.text().strip() or "New_Project"
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', product_name)
        folder_path = os.path.join(parent_folder, safe_name)

        steps = [
            ("Saving modeling graph...", lambda: self.modeling_widget.save_graph_to_file(folder_path)),
            ("Saving surrogate settings...", lambda: self.surrogate_widget.save_to_folder(folder_path)),
            ("Saving solution-space data...", lambda: self.sol_space_widget.save_to_folder(folder_path)),
            ("Saving optimization settings...", lambda: self.optimization_widget.save_to_folder(folder_path)),
            ("Saving sensitivity settings...", lambda: self.sensitivity_widget.save_to_folder(folder_path)),
        ]
        self._run_project_steps(
            "Saving project...",
            steps,
            f"Project saved successfully to:\n{folder_path}",
            "Save Error",
        )

    def load_project(self):
        """Load the entire project from a folder."""
        if self._project_io_busy:
            QtWidgets.QMessageBox.information(self, "Project Operation In Progress", "Wait for the current project operation to finish first.")
            return

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder to Load")
        if not folder_path:
            return

        def transfer_solution_space_problem() -> None:
            if self.sol_space_widget.problem:
                if self.sol_space_widget.models:
                    self.optimization_widget.load_models(self.sol_space_widget.models)
                self.optimization_widget.set_problem(self.sol_space_widget.problem)
                self.optimization_widget.system_code = self.sol_space_widget.system_code

        steps = [
            ("Loading modeling graph...", lambda: self.modeling_widget.load_graph_from_file(folder_path)),
            ("Loading surrogate settings...", lambda: self.surrogate_widget.load_from_folder(folder_path)),
            ("Loading solution-space data...", lambda: self.sol_space_widget.load_from_folder(folder_path)),
            ("Syncing optimization problem...", transfer_solution_space_problem),
            ("Loading optimization settings...", lambda: self.optimization_widget.load_from_folder(folder_path)),
            ("Loading sensitivity settings...", lambda: self.sensitivity_widget.load_from_folder(folder_path)),
        ]
        self._run_project_steps(
            "Loading project...",
            steps,
            "Project loaded successfully!",
            "Load Error",
        )

