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
from pylcss.user_interface.assistant import OverlayWidget


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window containing all major components.

    This window provides a tabbed interface with seven main sections:
    - Modeling Environment: Visual node-based system modeling
    - CAD Environment: Parametric CAD modeling with 3D viewer
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

        # Central Widget setup
        self.central_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(self.central_widget)

        # Tabs setup
        self.tabs: QtWidgets.QTabWidget = QtWidgets.QTabWidget()
        self.tabs.setMovable(False)  # Prevent tab reordering
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.tabs)

        # 1. Modeling Tab
        self.modeling_widget: ModelingWidget = ModelingWidget()
        self.modeling_widget.build_requested.connect(self.transfer_model)
        tab_index = self.tabs.addTab(self.modeling_widget, qta.icon('fa5s.project-diagram'), "  Modeling Environment")
        self.tabs.setTabToolTip(tab_index, "Visual node-based system modeling environment. Create and connect computational nodes to define mathematical relationships between design variables and system outputs.")

        # --- ADD NEW TAB HERE ---
        # 2. CAD Environment Tab
        self.cad_widget = ProfessionalCadApp()
        tab_index = self.tabs.addTab(self.cad_widget, qta.icon('fa5s.cube'), "  CAD Environment")
        self.tabs.setTabToolTip(tab_index, "Parametric CAD modeling with 3D viewer.")

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

    def _setup_assistant_systems(self) -> None:
        """Initialize assistant systems (voice + LLM)."""
        # Create assistant manager
        self.assistant_manager = AssistantManager(main_window=self)
        
        # Add Voice Control option to File menu (simple, near Save/Load)
        self.file_menu.addSeparator()
        
        self.voice_control_action = QtGui.QAction(
            qta.icon('fa5s.microphone'), "Voice Control", self
        )
        self.voice_control_action.setCheckable(True)
        self.voice_control_action.setShortcut("Ctrl+Shift+V")
        self.voice_control_action.setToolTip(
            "Toggle voice control (Ctrl+Shift+V)\n"
            "Say commands like 'add input', 'run optimization', 'save project'"
        )
        self.voice_control_action.toggled.connect(self._toggle_voice_control)
        self.file_menu.addAction(self.voice_control_action)
        
        # Chat with AI Action (added per user request)
        self.chat_action = QtGui.QAction(
            qta.icon('fa5s.comments'), "Chat with AI", self
        )
        self.chat_action.setShortcut("Ctrl+Shift+C")
        self.chat_action.setToolTip("Open AI Assistant Chat (Ctrl+Shift+C)")
        self.chat_action.triggered.connect(self._open_llm_chat)
        self.file_menu.addAction(self.chat_action)
        
        # LLM Settings action
        self.llm_settings_action = QtGui.QAction(
            qta.icon('fa5s.robot'), "LLM Assistant Settings...", self
        )
        self.llm_settings_action.setShortcut("Ctrl+Shift+L")
        self.llm_settings_action.setToolTip(
            "Configure LLM providers, API keys, and models (Ctrl+Shift+L)"
        )
        self.llm_settings_action.triggered.connect(self._open_llm_settings)
        self.file_menu.addAction(self.llm_settings_action)
        
        # Create overlay widget for visual feedback
        self.hands_free_overlay = OverlayWidget(self)
        self.hands_free_overlay.hide()
        
        # Connect manager signals to overlay
        self.assistant_manager.status_changed.connect(
            lambda s: self.hands_free_overlay.set_active(self.assistant_manager.is_running())
        )
        self.assistant_manager.command_recognized.connect(
            self.hands_free_overlay.show_command
        )
        self.assistant_manager.partial_text.connect(
            self.hands_free_overlay.show_partial
        )
        
        # Initialize in background to not block startup
        QtCore.QTimer.singleShot(1000, self.assistant_manager.initialize)
    
    def _toggle_voice_control(self, checked: bool) -> None:
        """Toggle voice control on/off."""
        if checked:
            if self.assistant_manager.start():
                self.voice_control_action.setIcon(qta.icon('fa5s.microphone', color='green'))
                self.voice_control_action.setText("Voice Control (ON)")
                # Show overlay for feedback
                if self.assistant_manager.get_config().overlay_enabled:
                    self.hands_free_overlay.position_in_corner("top-right")
                    self.hands_free_overlay.show()
                self.statusBar().showMessage("Voice control enabled - say commands to control the app", 3000)
            else:
                self.voice_control_action.setChecked(False)
                self.statusBar().showMessage("Failed to start voice control - check microphone", 5000)
        else:
            self.assistant_manager.stop()
            self.voice_control_action.setIcon(qta.icon('fa5s.microphone'))
            self.voice_control_action.setText("Voice Control")
            self.hands_free_overlay.hide()
            self.statusBar().showMessage("Voice control disabled", 2000)
            
    def _open_llm_chat(self) -> None:
        """Open the LLM chat dialog."""
        from pylcss.user_interface.assistant import LLMChatDialog
        if not hasattr(self, '_llm_chat_dialog') or self._llm_chat_dialog is None:
            # Pass the manager's command dispatcher so the chat can execute actions
            dispatcher = self.assistant_manager.command_dispatcher
            self._llm_chat_dialog = LLMChatDialog(
                command_dispatcher=dispatcher,
                assistant_manager=self.assistant_manager,
                parent=self
            )
        self._llm_chat_dialog.show()
        self._llm_chat_dialog.raise_()
    
    def _open_llm_settings(self) -> None:
        """Open the LLM configuration dialog."""
        from pylcss.user_interface.assistant import LLMConfigDialog
        dialog = LLMConfigDialog(self)
        if dialog.exec():
            # Reload manager's provider after settings change
            self.assistant_manager.update_config(self.assistant_manager.get_config())
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
        parent_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Parent Folder for Project")
        if not parent_folder:
            return

        try:
            # Get product name for folder creation
            product_name = self.modeling_widget.system_manager.product_name.text().strip()
            if not product_name:
                product_name = "New_Project"
            
            # Sanitize folder name
            import re
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', product_name)
            
            # Create project folder
            import os
            folder_path = os.path.join(parent_folder, safe_name)
            os.makedirs(folder_path, exist_ok=True)

            # 1. Save Modeling Graph
            self.modeling_widget.save_graph_to_file(folder_path)
            
            # 2. Save Surrogate Settings
            self.surrogate_widget.save_to_folder(folder_path)
            
            # 3. Save Solution Space Data
            self.sol_space_widget.save_to_folder(folder_path)
            
            # 4. Save Optimization Settings
            self.optimization_widget.save_to_folder(folder_path)
            
            # 5. Save Sensitivity Settings
            self.sensitivity_widget.save_to_folder(folder_path)
            
            QtWidgets.QMessageBox.information(self, "Success", f"Project saved successfully to:\n{folder_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        """Load the entire project from a folder."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder to Load")
        if not folder_path:
            return

        try:
            # 1. Load Modeling Graph
            self.modeling_widget.load_graph_from_file(folder_path)
            
            # 2. Load Surrogate Settings
            self.surrogate_widget.load_from_folder(folder_path)
            
            # 3. Load Solution Space Data
            self.sol_space_widget.load_from_folder(folder_path)
            
            # Transfer loaded problem to Optimization Widget
            if self.sol_space_widget.problem:
                # Transfer models list so the dropdown is populated
                if self.sol_space_widget.models:
                    self.optimization_widget.load_models(self.sol_space_widget.models)
                
                self.optimization_widget.set_problem(self.sol_space_widget.problem)
                self.optimization_widget.system_code = self.sol_space_widget.system_code

            # 4. Load Optimization Settings
            self.optimization_widget.load_from_folder(folder_path)
            
            # 5. Load Sensitivity Settings
            self.sensitivity_widget.load_from_folder(folder_path)
            
            QtWidgets.QMessageBox.information(self, "Success", "Project loaded successfully!")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load project: {str(e)}")







