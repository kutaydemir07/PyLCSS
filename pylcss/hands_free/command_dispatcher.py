# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Command Dispatcher Module.

Maps voice commands to PyLCSS-specific actions and general system controls.
"""

import logging
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from pylcss.hands_free.mouse_controller import MouseController

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class CommandDispatcher:
    """
    Dispatches voice commands to appropriate actions.
    
    Handles mouse actions, keyboard shortcuts, tab navigation,
    and PyLCSS-specific functionality.
    """
    
    def __init__(
        self,
        main_window: Optional["QMainWindow"] = None,
        mouse_controller: Optional[MouseController] = None
    ):
        """
        Initialize the command dispatcher.
        
        Args:
            main_window: Reference to PyLCSS main window for app-specific actions
            mouse_controller: Controller for mouse/keyboard actions
        """
        self.main_window = main_window
        self.mouse = mouse_controller or MouseController()
        
        # Control callbacks
        self._on_pause: Optional[Callable[[], None]] = None
        self._on_resume: Optional[Callable[[], None]] = None
        self._on_calibrate: Optional[Callable[[], None]] = None
        self._on_start_dictation: Optional[Callable[[], None]] = None
        self._on_stop_dictation: Optional[Callable[[], None]] = None
        
        # Action handlers by action type
        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {
            "mouse_click": self._handle_mouse_click,
            "mouse_double_click": self._handle_double_click,
            "mouse_drag_toggle": self._handle_drag_toggle,
            "scroll": self._handle_scroll,
            "scroll_horizontal": self._handle_scroll_horizontal,
            "switch_tab": self._handle_switch_tab,
            "next_tab": self._handle_next_tab,
            "previous_tab": self._handle_previous_tab,
            "keyboard": self._handle_keyboard,
            "pylcss_action": self._handle_pylcss_action,
            "control": self._handle_control,
            "window": self._handle_window,
        }
        
    def set_control_callbacks(
        self,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        on_calibrate: Optional[Callable[[], None]] = None,
        on_start_dictation: Optional[Callable[[], None]] = None,
        on_stop_dictation: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for control commands."""
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_calibrate = on_calibrate
        self._on_start_dictation = on_start_dictation
        self._on_stop_dictation = on_stop_dictation
        
    def dispatch(self, command_name: str, command_data: Dict[str, Any]) -> bool:
        """
        Dispatch a command to the appropriate handler.
        
        Args:
            command_name: The recognized command text
            command_data: Command configuration from VOICE_COMMANDS
            
        Returns:
            True if command was handled, False otherwise
        """
        action = command_data.get("action")
        
        if not action:
            logger.warning(f"Command '{command_name}' has no action defined")
            return False
            
        handler = self._action_handlers.get(action)
        
        if handler:
            try:
                handler(command_data)
                logger.info(f"Executed command: {command_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to execute command '{command_name}': {e}")
                return False
        else:
            logger.warning(f"Unknown action type: {action}")
            return False
            
    # --- Mouse Action Handlers ---
    
    def _handle_mouse_click(self, data: Dict[str, Any]) -> None:
        """Handle mouse click command."""
        button = data.get("button", "left")
        self.mouse.click(button=button)
        
    def _handle_double_click(self, data: Dict[str, Any]) -> None:
        """Handle double click command."""
        self.mouse.double_click()
        
    def _handle_drag_toggle(self, data: Dict[str, Any]) -> None:
        """Handle drag/drop toggle."""
        self.mouse.toggle_drag()
        
    def _handle_scroll(self, data: Dict[str, Any]) -> None:
        """Handle scroll command."""
        direction = data.get("direction", 3)
        self.mouse.scroll(direction)
        
    def _handle_scroll_horizontal(self, data: Dict[str, Any]) -> None:
        """Handle horizontal scroll command."""
        direction = data.get("direction", 3)
        self.mouse.scroll_horizontal(direction)
        
    # --- Tab Navigation Handlers ---
    
    def _handle_switch_tab(self, data: Dict[str, Any]) -> None:
        """Handle switch to specific tab."""
        tab_index = data.get("tab", 0)
        
        if self.main_window:
            try:
                # Find the tab widget in main window
                tab_widget = self._get_tab_widget()
                if tab_widget and 0 <= tab_index < tab_widget.count():
                    tab_widget.setCurrentIndex(tab_index)
                    logger.info(f"Switched to tab {tab_index}")
            except Exception as e:
                logger.error(f"Failed to switch tab: {e}")
        else:
            # Use keyboard shortcut as fallback
            # Ctrl+1 through Ctrl+9 for tabs
            if 0 <= tab_index <= 8:
                self.mouse.hotkey('ctrl', str(tab_index + 1))
                
    def _handle_next_tab(self, data: Dict[str, Any]) -> None:
        """Handle next tab command."""
        if self.main_window:
            try:
                tab_widget = self._get_tab_widget()
                if tab_widget:
                    current = tab_widget.currentIndex()
                    next_idx = (current + 1) % tab_widget.count()
                    tab_widget.setCurrentIndex(next_idx)
            except Exception as e:
                logger.error(f"Failed to switch to next tab: {e}")
        else:
            self.mouse.hotkey('ctrl', 'tab')
            
    def _handle_previous_tab(self, data: Dict[str, Any]) -> None:
        """Handle previous tab command."""
        if self.main_window:
            try:
                tab_widget = self._get_tab_widget()
                if tab_widget:
                    current = tab_widget.currentIndex()
                    prev_idx = (current - 1) % tab_widget.count()
                    tab_widget.setCurrentIndex(prev_idx)
            except Exception as e:
                logger.error(f"Failed to switch to previous tab: {e}")
        else:
            self.mouse.hotkey('ctrl', 'shift', 'tab')
            
    def _get_tab_widget(self):
        """Get the main tab widget from PyLCSS window."""
        if not self.main_window:
            return None
        # MainWindow uses 'tabs', fallback to 'tab_widget'
        tab_widget = getattr(self.main_window, 'tabs', None)
        if tab_widget is None:
            tab_widget = getattr(self.main_window, 'tab_widget', None)
        return tab_widget
        
    # --- Keyboard Handlers ---
    
    def _handle_keyboard(self, data: Dict[str, Any]) -> None:
        """Handle keyboard shortcut command."""
        keys = data.get("keys", [])
        if keys:
            self.mouse.hotkey(*keys)
            
    # --- PyLCSS Specific Handlers ---
    
    def _handle_pylcss_action(self, data: Dict[str, Any]) -> None:
        """Handle PyLCSS-specific actions."""
        command = data.get("command", "")
        
        if not self.main_window:
            logger.warning("PyLCSS action requires main window reference")
            return
            
        action_map = {
            # Core actions
            "run_optimization": self._run_optimization,
            "stop_optimization": self._stop_optimization,
            "generate_samples": self._generate_samples,
            "train_surrogate": self._train_surrogate,
            "run_sensitivity": self._run_sensitivity,
            "new_project": self._new_project,
            "open_project": self._open_project,
            "export_results": self._export_results,
            "build_model": self._build_model,
            
            # Modeling environment - nodes
            "add_input": self._add_modeling_node,
            "add_output": self._add_modeling_node,
            "add_function": self._add_modeling_node,
            "add_intermediate": self._add_modeling_node,
            "validate_graph": self._validate_graph,
            
            # Modeling environment - system management
            "add_system": self._add_system,
            "remove_system": self._remove_system,
            "rename_system": self._rename_system,
            "next_system": self._next_system,
            "previous_system": self._previous_system,
            
            # Modeling environment - graph operations
            "auto_connect": self._auto_connect,
            "clear_graph": self._clear_graph,
            "select_all_nodes": self._select_all_nodes,
            "delete_selected": self._delete_selected,
            
            # CAD environment
            "cad_add_box": self._add_cad_node,
            "cad_add_cylinder": self._add_cad_node,
            "cad_add_sphere": self._add_cad_node,
            "cad_add_cone": self._add_cad_node,
            "cad_add_torus": self._add_cad_node,
            "cad_add_extrude": self._add_cad_node,
            "cad_add_fillet": self._add_cad_node,
            "cad_add_chamfer": self._add_cad_node,
            "cad_add_boolean": self._add_cad_node,
            "cad_add_union": self._add_cad_node,
            "cad_add_cut": self._add_cad_node,
            "cad_add_revolve": self._add_cad_node,
            "cad_execute": self._cad_execute,
            "cad_export": self._cad_export,
            
            # Solution space
            "resample": self._resample,
            "add_plot": self._add_plot,
            "clear_plots": self._clear_plots,
            "save_plots": self._save_plots,
            "configure_colors": self._configure_colors,
            "view_code": self._view_code,
            "compute_family": self._compute_family,
            "add_variant": self._add_variant,
            "remove_variant": self._remove_variant,
            "edit_variant": self._edit_variant,
            "compute_adg": self._compute_adg,
            
            # Surrogate training
            "refresh_nodes": self._refresh_nodes,
            "generate_training_data": self._generate_training_data,
            "browse_data_file": self._browse_data_file,
            "save_surrogate": self._save_surrogate,
            "stop_training": self._stop_training,
            "adaptive_training": self._adaptive_training,
            
            # Optimization
            "optimization_settings": self._optimization_settings,
            
            # Sensitivity
            "refresh_outputs": self._refresh_outputs,
            "export_sensitivity": self._export_sensitivity,
        }
        
        # Special handling for node creation commands that need the command name
        # Note: add_system, remove_system, rename_system are NOT node creation commands
        node_creation_commands = ["add_input", "add_output", "add_function", "add_intermediate"]
        cad_node_commands = [c for c in action_map.keys() if c.startswith("cad_add_")]
        
        if command in node_creation_commands:
            self._add_modeling_node(command)
            return
        elif command in cad_node_commands:
            self._add_cad_node(command)
            return
        
        handler = action_map.get(command)
        if handler:
            handler()
        else:
            logger.warning(f"Unknown PyLCSS action: {command}")
            
    def _run_optimization(self) -> None:
        """Trigger optimization run."""
        if not self.main_window or not hasattr(self.main_window, 'optimization_widget'):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, 'btn_run'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_run, "click", Qt.QueuedConnection)
            logger.info("Voice command: Running optimization")
        elif hasattr(widget, 'start_optimization'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "start_optimization", Qt.QueuedConnection)
            logger.info("Voice command: Starting optimization")
                
    def _stop_optimization(self) -> None:
        """Stop optimization."""
        if not self.main_window or not hasattr(self.main_window, 'optimization_widget'):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, 'btn_stop'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_stop, "click", Qt.QueuedConnection)
            logger.info("Voice command: Stopping optimization")
        elif hasattr(widget, 'stop_optimization'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "stop_optimization", Qt.QueuedConnection)
                
    def _generate_samples(self) -> None:
        """Generate samples in solution space."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_compute_feasible'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_compute_feasible, "click", Qt.QueuedConnection)
            logger.info("Voice command: Computing solution space")
        elif hasattr(widget, 'run_computation'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "run_computation", Qt.QueuedConnection)
                
    def _train_surrogate(self) -> None:
        """Train surrogate model."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_train'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_train, "click", Qt.QueuedConnection)
            logger.info("Voice command: Training surrogate model")
        elif hasattr(widget, 'start_training'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "start_training", Qt.QueuedConnection)
                
    def _run_sensitivity(self) -> None:
        """Run sensitivity analysis."""
        if not self.main_window or not hasattr(self.main_window, 'sensitivity_widget'):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, 'btn_analyze'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_analyze, "click", Qt.QueuedConnection)
            logger.info("Voice command: Running sensitivity analysis")
        elif hasattr(widget, 'run_analysis'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "run_analysis", Qt.QueuedConnection)
                
    def _new_project(self) -> None:
        """Create new project."""
        self.mouse.hotkey('ctrl', 'n')
        
    def _open_project(self) -> None:
        """Open project dialog."""
        self.mouse.hotkey('ctrl', 'o')
        
    def _export_results(self) -> None:
        """Export results."""
        self.mouse.hotkey('ctrl', 'e')
    
    def _build_model(self) -> None:
        """Build/transfer model from modeling environment."""
        if self.main_window:
            # Use QMetaObject.invokeMethod to call on main thread
            from PySide6.QtCore import QMetaObject, Qt, Q_ARG
            if hasattr(self.main_window, 'transfer_model'):
                QMetaObject.invokeMethod(
                    self.main_window, 
                    "transfer_model",
                    Qt.QueuedConnection
                )
    
    def _add_system(self) -> None:
        """Add a new system in modeling environment."""
        if not self.main_window:
            return
        if hasattr(self.main_window, 'modeling_widget'):
            widget = self.main_window.modeling_widget
            if hasattr(widget, 'system_manager') and hasattr(widget.system_manager, 'add_system'):
                from PySide6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(widget.system_manager, "add_system", Qt.QueuedConnection)
                logger.info("Voice command: Adding new system")
    
    def _add_modeling_node(self, command: str) -> None:
        """Add a node in the modeling environment."""
        if not self.main_window:
            return
        if not hasattr(self.main_window, 'modeling_widget'):
            logger.warning("Modeling widget not found")
            return
            
        widget = self.main_window.modeling_widget
        
        # Map command to method
        method_map = {
            "add_input": "add_input_node",
            "add_output": "add_output_node",
            "add_function": "add_function_node",
            "add_intermediate": "add_intermediate_node",
        }
        
        method_name = method_map.get(command)
        if method_name and hasattr(widget, method_name):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, method_name, Qt.QueuedConnection)
            logger.info(f"Voice command: {command.replace('_', ' ')}")
        else:
            logger.warning(f"Method {method_name} not found on modeling widget")
    
    def _validate_graph(self) -> None:
        """Validate the current graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'validate_graph'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "validate_graph", Qt.QueuedConnection)
            logger.info("Voice command: Validating graph")
    
    def _add_cad_node(self, command: str) -> None:
        """Add a node in the CAD environment."""
        if not self.main_window:
            return
        if not hasattr(self.main_window, 'cad_widget'):
            logger.warning("CAD widget not found")
            return
            
        widget = self.main_window.cad_widget
        
        # Map command to node type and display name
        node_map = {
            "cad_add_box": ("com.cad.box", "Box"),
            "cad_add_cylinder": ("com.cad.cylinder", "Cylinder"),
            "cad_add_sphere": ("com.cad.sphere", "Sphere"),
            "cad_add_cone": ("com.cad.cone", "Cone"),
            "cad_add_torus": ("com.cad.torus", "Torus"),
            "cad_add_extrude": ("com.cad.extrude", "Extrude"),
            "cad_add_fillet": ("com.cad.fillet", "Fillet"),
            "cad_add_chamfer": ("com.cad.chamfer", "Chamfer"),
            "cad_add_boolean": ("com.cad.boolean", "Boolean"),
            "cad_add_union": ("com.cad.boolean", "Boolean Union"),
            "cad_add_cut": ("com.cad.boolean", "Boolean Cut"),
            "cad_add_revolve": ("com.cad.revolve", "Revolve"),
        }
        
        node_info = node_map.get(command)
        if node_info:
            node_type, label = node_info
            # The CAD widget uses _spawn_node method
            if hasattr(widget, '_spawn_node'):
                try:
                    widget._spawn_node(node_type, label)
                    logger.info(f"Voice command: Added CAD node {label}")
                except Exception as e:
                    logger.error(f"Failed to create CAD node: {e}")
            else:
                logger.warning("CAD widget does not have _spawn_node method")
        else:
            logger.warning(f"Unknown CAD command: {command}")
    
    def _cad_execute(self) -> None:
        """Execute/run the CAD graph."""
        if not self.main_window or not hasattr(self.main_window, 'cad_widget'):
            logger.warning("CAD widget not found")
            return
        widget = self.main_window.cad_widget
        if hasattr(widget, 'execute_graph'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "execute_graph", Qt.QueuedConnection)
            logger.info("Voice command: Executing CAD graph")
        elif hasattr(widget, 'btn_execute'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_execute, "click", Qt.QueuedConnection)
            logger.info("Voice command: Running CAD")
    
    def _cad_export(self) -> None:
        """Export the CAD model to STL."""
        if not self.main_window or not hasattr(self.main_window, 'cad_widget'):
            return
        widget = self.main_window.cad_widget
        if hasattr(widget, 'btn_export'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_export, "click", Qt.QueuedConnection)
            logger.info("Voice command: Exporting CAD")
    
    # --- System Management Handlers ---
    
    def _remove_system(self) -> None:
        """Remove current system in modeling environment."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'system_manager') and hasattr(widget.system_manager, 'btn_remove_system'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.system_manager.btn_remove_system, "click", Qt.QueuedConnection)
            logger.info("Voice command: Removing system")
    
    def _rename_system(self) -> None:
        """Rename current system in modeling environment."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'system_manager') and hasattr(widget.system_manager, 'btn_rename_system'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.system_manager.btn_rename_system, "click", Qt.QueuedConnection)
            logger.info("Voice command: Renaming system")
    
    def _next_system(self) -> None:
        """Switch to next system."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'system_manager') and hasattr(widget.system_manager, 'system_list'):
            from PySide6.QtCore import QMetaObject, Qt
            lst = widget.system_manager.system_list
            current = lst.currentRow()
            if current < lst.count() - 1:
                lst.setCurrentRow(current + 1)
            logger.info("Voice command: Next system")
    
    def _previous_system(self) -> None:
        """Switch to previous system."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'system_manager') and hasattr(widget.system_manager, 'system_list'):
            lst = widget.system_manager.system_list
            current = lst.currentRow()
            if current > 0:
                lst.setCurrentRow(current - 1)
            logger.info("Voice command: Previous system")
    
    # --- Graph Operations Handlers ---
    
    def _auto_connect(self) -> None:
        """Auto-connect nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'auto_connect'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget, "auto_connect", Qt.QueuedConnection)
            logger.info("Voice command: Auto-connecting nodes")
    
    def _clear_graph(self) -> None:
        """Clear all nodes from the graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'current_graph') and hasattr(widget.current_graph, 'clear_session'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.current_graph, "clear_session", Qt.QueuedConnection)
            logger.info("Voice command: Clearing graph")
    
    def _select_all_nodes(self) -> None:
        """Select all nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'current_graph'):
            graph = widget.current_graph
            if hasattr(graph, 'select_all'):
                graph.select_all()
            logger.info("Voice command: Selecting all nodes")
    
    def _delete_selected(self) -> None:
        """Delete selected nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, 'current_graph') and hasattr(widget.current_graph, 'delete_selected'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.current_graph, "delete_selected", Qt.QueuedConnection)
            logger.info("Voice command: Deleting selected nodes")
    
    # --- Solution Space Handlers ---
    
    def _resample(self) -> None:
        """Resample the solution space."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_resample'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_resample, "click", Qt.QueuedConnection)
            logger.info("Voice command: Resampling")
    
    def _add_plot(self) -> None:
        """Add a new plot to the solution space."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_add_plot'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_add_plot, "click", Qt.QueuedConnection)
            logger.info("Voice command: Adding plot")
    
    def _clear_plots(self) -> None:
        """Clear all plots."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_clear_plots'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_clear_plots, "click", Qt.QueuedConnection)
            logger.info("Voice command: Clearing plots")
    
    def _save_plots(self) -> None:
        """Save all plots."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_save_all'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_save_all, "click", Qt.QueuedConnection)
            logger.info("Voice command: Saving plots")
    
    def _configure_colors(self) -> None:
        """Open color configuration dialog."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_colors'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_colors, "click", Qt.QueuedConnection)
            logger.info("Voice command: Configuring colors")
    
    def _view_code(self) -> None:
        """View the generated code."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_view_code'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_view_code, "click", Qt.QueuedConnection)
            logger.info("Voice command: Viewing code")
    
    def _compute_family(self) -> None:
        """Compute product family solution space."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_compute_family'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_compute_family, "click", Qt.QueuedConnection)
            logger.info("Voice command: Computing product family")
    
    def _add_variant(self) -> None:
        """Add a product variant."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_add_variant'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_add_variant, "click", Qt.QueuedConnection)
            logger.info("Voice command: Adding variant")
    
    def _remove_variant(self) -> None:
        """Remove a product variant."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_remove_variant'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_remove_variant, "click", Qt.QueuedConnection)
            logger.info("Voice command: Removing variant")
    
    def _edit_variant(self) -> None:
        """Edit variant requirements."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_edit_variant'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_edit_variant, "click", Qt.QueuedConnection)
            logger.info("Voice command: Editing variant")
    
    def _compute_adg(self) -> None:
        """Generate Attribute Dependency Graph."""
        if not self.main_window or not hasattr(self.main_window, 'sol_space_widget'):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, 'btn_compute_adg'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_compute_adg, "click", Qt.QueuedConnection)
            logger.info("Voice command: Computing ADG")
    
    # --- Surrogate Training Handlers ---
    
    def _refresh_nodes(self) -> None:
        """Refresh the node list in surrogate training."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_refresh'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_refresh, "click", Qt.QueuedConnection)
            logger.info("Voice command: Refreshing nodes")
    
    def _generate_training_data(self) -> None:
        """Generate training data for surrogate model."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_generate'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_generate, "click", Qt.QueuedConnection)
            logger.info("Voice command: Generating training data")
    
    def _browse_data_file(self) -> None:
        """Browse for data file."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_browse'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_browse, "click", Qt.QueuedConnection)
            logger.info("Voice command: Browsing for data file")
    
    def _save_surrogate(self) -> None:
        """Save and attach surrogate model to node."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_save'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_save, "click", Qt.QueuedConnection)
            logger.info("Voice command: Saving surrogate model")
    
    def _stop_training(self) -> None:
        """Stop surrogate model training."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_stop'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_stop, "click", Qt.QueuedConnection)
            logger.info("Voice command: Stopping training")
    
    def _adaptive_training(self) -> None:
        """Start adaptive/active learning training."""
        if not self.main_window or not hasattr(self.main_window, 'surrogate_widget'):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, 'btn_adaptive'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_adaptive, "click", Qt.QueuedConnection)
            logger.info("Voice command: Starting adaptive training")
    
    # --- Optimization Handlers ---
    
    def _optimization_settings(self) -> None:
        """Open optimization settings dialog."""
        if not self.main_window or not hasattr(self.main_window, 'optimization_widget'):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, 'settings_widget') and hasattr(widget.settings_widget, 'btn_settings'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.settings_widget.btn_settings, "click", Qt.QueuedConnection)
            logger.info("Voice command: Opening optimization settings")
    
    # --- Sensitivity Handlers ---
    
    def _refresh_outputs(self) -> None:
        """Refresh outputs in sensitivity analysis."""
        if not self.main_window or not hasattr(self.main_window, 'sensitivity_widget'):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, 'btn_refresh_outputs'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_refresh_outputs, "click", Qt.QueuedConnection)
            logger.info("Voice command: Refreshing outputs")
    
    def _export_sensitivity(self) -> None:
        """Export sensitivity analysis results."""
        if not self.main_window or not hasattr(self.main_window, 'sensitivity_widget'):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, 'btn_export'):
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(widget.btn_export, "click", Qt.QueuedConnection)
            logger.info("Voice command: Exporting sensitivity results")
        
    # --- Control Handlers ---
    
    def _handle_control(self, data: Dict[str, Any]) -> None:
        """Handle control commands (pause, resume, etc.)."""
        command = data.get("command", "")
        
        callbacks = {
            "pause_tracking": self._on_pause,
            "resume_tracking": self._on_resume,
            "calibrate": self._on_calibrate,
            "center_cursor": lambda: self.mouse.center_cursor(),
            "start_dictation": self._on_start_dictation,
            "stop_dictation": self._on_stop_dictation,
        }
        
        callback = callbacks.get(command)
        if callback:
            callback()
        else:
            logger.warning(f"Unknown control command: {command}")
            
    # --- Window Handlers ---
    
    def _handle_window(self, data: Dict[str, Any]) -> None:
        """Handle window control commands."""
        command = data.get("command", "")
        
        if self.main_window:
            if command == "minimize":
                self.main_window.showMinimized()
            elif command == "maximize":
                if self.main_window.isMaximized():
                    self.main_window.showNormal()
                else:
                    self.main_window.showMaximized()
            elif command == "close":
                self.main_window.close()
        else:
            # Use keyboard shortcuts as fallback
            if command == "minimize":
                self.mouse.hotkey('win', 'down')
            elif command == "maximize":
                self.mouse.hotkey('win', 'up')
            elif command == "close":
                self.mouse.hotkey('alt', 'f4')
