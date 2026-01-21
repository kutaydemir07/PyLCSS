# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Command Dispatcher Module.

Maps voice commands to PyLCSS-specific actions and general system controls.
"""

import logging
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from pylcss.hands_free.mouse_controller import MouseController
from pylcss.hands_free.mouse_controller import MouseController
from pylcss.cad.node_library import NODE_CLASS_MAPPING
from pylcss.system_modeling.node_registry import SYSTEM_NODE_CLASS_MAPPING

from PySide6.QtCore import QObject, Signal, Slot, Qt, QMetaObject

class MainThreadExecutor(QObject):
    """Helper to execute functions on the main thread safely."""
    
    # Define signal that will be emitted from background thread
    execute_signal = Signal(object, object, object)
    
    def __init__(self):
        super().__init__()
        # Connect signal to slot (BlockingQueuedConnection will be used at emission time)
        # Actually, if we connect with BlockingQueuedConnection, logic is:
        # emit -> wait -> slot executes -> return
        self.execute_signal.connect(self.run_func, Qt.BlockingQueuedConnection)
        
    @Slot(object, object, object)
    def run_func(self, func, args, result_container):
        """Run function and store result."""
        try:
            result_container['val'] = func(*args)
        except Exception as e:
            result_container['error'] = e



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
        self.main_window = main_window
        self.mouse = mouse_controller or MouseController()
        
        # Helper for thread safety
        self._executor = MainThreadExecutor()
        if main_window:
            self._executor.moveToThread(main_window.thread())
        
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
            "build_node_graph": self._build_node_graph,
            "build_system_graph": self._build_system_graph,  # NEW

            
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
            "get_sensitivity": self._get_sensitivity_results, # NEW
            
            # Surrogate (extended)
            "train_surrogate_node": self._train_surrogate_node, # NEW

            
            # LLM Assistant
            "open_llm_assistant": self._open_llm_assistant,
            "close_llm_assistant": self._close_llm_assistant,

            # Granular Control
            "connect_nodes": self._connect_nodes,
            "set_property": self._set_property,
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
            # Special case for commands needing data
            if command == "build_node_graph":
                handler(data)
            elif command == "build_system_graph":
                handler(data)
            elif command == "train_surrogate_node":
                handler(data)
            elif command == "connect_nodes":
                handler(data)
            elif command == "set_property":
                handler(data)
            else:

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
    
    def _run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function on the main thread and wait for result.
        Uses Qt.BlockingQueuedConnection which is the correct, safe way.
        """
        import threading
        if threading.current_thread() is threading.main_thread():
             return func(*args, **kwargs)

        # Container for result (mutable)
        result = {"val": None, "error": None}
        
        # If we have kwargs, we need a partial to bind them since invokeMethod is limited
        if kwargs:
            from functools import partial
            func = partial(func, **kwargs)
            
        # Emit the signal which uses BlockingQueuedConnection
        # This blocks only the calling thread (background) until the slot (main) returns
        self._executor.execute_signal.emit(func, args, result)
        
        if result["error"]:
            raise result["error"]
        return result["val"]

    def _build_node_graph(self, data: Dict[str, Any], sync: bool = False) -> None:
        """
        Build a complete CAD node graph from LLM specification.
        
        Args:
            data: Command data containing 'params' with 'nodes' and 'connections' list.
            sync: If True, blocks until completion and raises errors.
        """
        params = data.get("params", {})
        nodes_spec = params.get("nodes", [])
        conns_spec = params.get("connections", [])
        
        if not nodes_spec:
            logger.warning("Empty node spec for build_node_graph")
            return
            
        if not self.main_window:
            logger.warning("Main window required for build_node_graph")
            if sync: raise RuntimeError("Main window required")
            return
            
        # 1. Switch to CAD tab (Index 1) - Can fire and forget safely usually
        try:
            tab_widget = self._get_tab_widget()
            if tab_widget and tab_widget.currentIndex() != 1:
                from PySide6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(tab_widget, "setCurrentIndex", Qt.QueuedConnection, 1)
        except Exception:
            pass
            
        # 2. Get CAD Widget
        if not hasattr(self.main_window, 'cad_widget'):
            logger.error("CAD widget not found")
            if sync: raise RuntimeError("CAD widget not found")
            return
        cad_widget = self.main_window.cad_widget
        graph_controller = getattr(cad_widget, 'graph', None)
        
        if not graph_controller:
            logger.error("CAD Graph controller not found")
            if sync: raise RuntimeError("CAD Graph controller not found")
            return

        # 3. Execute Graph Building on GUI Thread (Safety)
        # We define the function here and invoke it safely
        def build_graph_safe():
            # Helper for property setting with auto-fixes
            def set_prop_safe(node, prop_name, prop_val, node_identifier):
                if not hasattr(node, 'set_property'):
                    return
                    
                try:
                    # 1. Check for Enum/Combo mismatch (Case Insensitive Fix)
                    ENUM_MAP = {
                        "operation": ["Union", "Cut", "Intersect"],
                        "selector_type": ["Direction", "NearestToPoint", "Index", "Largest Area", "Tag"],
                    }
                    
                    final_val = prop_val
                    if prop_name in ENUM_MAP and isinstance(prop_val, str):
                        options = ENUM_MAP[prop_name]
                        for opt in options:
                            if opt.lower() == prop_val.lower():
                                final_val = opt
                                break
                    
                    # 2. Try Setting Property
                    if prop_name in node.properties():
                        node.set_property(prop_name, final_val)
                    else:
                        # 3. Property Missing -> Check for Port Mismatch (Auto-fix)
                        input_port = node.get_input(prop_name)
                        if input_port and isinstance(prop_val, str):
                            logger.info(f"Auto-fixing LLM error: Converting property '{prop_name}'='{prop_val}' to connection")
                            src_port = "sketch" if prop_name == "sketch" else "shape"
                            conns_spec.append({
                                "from": f"{prop_val}.{src_port}",
                                "to": f"{node_identifier}.{prop_name}"
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to set property {prop_name} on {node_identifier}: {e}")

            # Map of user-defined ID to Node object
            id_to_node = {}
            
            # A. Map Existing Nodes first
            existing_nodes = graph_controller.all_nodes()
            existing_by_name = {n.name(): n for n in existing_nodes}
            id_to_node.update(existing_by_name)

            # Determine start position for NEW nodes
            start_x, start_y = 0, 0
            if existing_nodes:
                    start_y = max((n.pos()[1] for n in existing_nodes), default=0) + 200
            
            # B. Process Nodes (Create or Update)
            new_node_count = 0
            created_names = {} # Map requested ID -> Actual Name
            
            for i, node_def in enumerate(nodes_spec):
                node_id = node_def.get("id")
                node_type = node_def.get("type")
                props = node_def.get("properties", {})
                
                if not node_id:
                    continue
                    
                # CHECK IF EXISTS
                if node_id in existing_by_name:
                    # UPDATE EXISTING
                    node = existing_by_name[node_id]
                    logger.info(f"Updating existing node: {node_id}")
                    
                    # Update using safe helper
                    for prop_name, prop_val in props.items():
                        set_prop_safe(node, prop_name, prop_val, node_id)
                    
                    id_to_node[node_id] = node
                    
                else:
                    # CREATE NEW
                    if not node_type: 
                        continue
                        
                    node_class = NODE_CLASS_MAPPING.get(node_type)
                    if not node_class:
                        logger.warning(f"Unknown node type: {node_type}")
                        continue
                    
                    node = node_class()
                    # node.set_name(str(node_id)) # Don't set here, set after add
                    
                    # Position (staggered for new batch)
                    row = new_node_count // 4
                    col = new_node_count % 4
                    node.set_pos(start_x + (col * 250), start_y + (row * 150))
                    new_node_count += 1
                    
                    graph_controller.add_node(node)
                    
                    # Force name match *after* adding (NodeGraphQt often resets on add)
                    node.set_name(str(node_id))
                    
                    # Log actual name to debug renames
                    actual_name = node.name()
                    if actual_name != str(node_id):
                        logger.warning(f"Node '{node_id}' renamed to '{actual_name}' by graph")
                    else:
                        logger.info(f"Created node: {actual_name}")
                        
                    id_to_node[node_id] = node
                    
                    # Record name mapping for result
                    created_names[node_id] = actual_name
                    
                    # Set properties using safe helper
                    for prop_name, prop_val in props.items():
                        set_prop_safe(node, prop_name, prop_val, node_id)
                                
            # C. Process Connections (Update to use actual names if possible)
            for conn in conns_spec:
                try:
                    from_str = conn.get("from", "")
                    from_node_spec = conn.get("from_node", "")
                    
                    # Support both formats (node.port or separate fields)
                    if from_node_spec:
                         from_id = conn.get("from_node")
                         from_port = conn.get("from_port", "shape")
                         to_id = conn.get("to_node")
                         to_port = conn.get("to_port", "shape")
                    elif "." in from_str:
                         from_id, from_port = from_str.split(".", 1)
                         to_id, to_port = conn.get("to", "").split(".", 1)
                    else:
                         continue

                    # Look up using ID map first (most reliable for just-created nodes)
                    src_node = id_to_node.get(from_id)
                    dst_node = id_to_node.get(to_id)
                    
                    # If not found in current batch, look up by name (potentially remapped)
                    if not src_node and from_id in existing_by_name:
                         src_node = existing_by_name[from_id]
                    if not dst_node and to_id in existing_by_name:
                         dst_node = existing_by_name[to_id]
                    
                    if src_node and dst_node:
                        # Auto-fix: Port Aliases
                        # Use internal NodeGraphQt API or try common alternatives
                        
                        # 1. Output Mapping
                        output_aliases = {
                            "shape": ["result", "out", "entity", "workplane"],
                            "result": ["shape", "out"],
                        }
                        
                        out_port = src_node.get_output(from_port)
                        if not out_port and from_port in output_aliases:
                            for alias in output_aliases[from_port]:
                                out_port = src_node.get_output(alias)
                                if out_port:
                                    logger.info(f"Auto-fixed port alias: {from_port} -> {alias} on {src_node.name()}")
                                    break
                        
                        # 2. Input Mapping
                        input_aliases = {
                            "shape": ["input_shape", "target", "profile"],
                        }
                        
                        in_port = dst_node.get_input(to_port)
                        if not in_port and to_port in input_aliases:
                            for alias in input_aliases[to_port]:
                                in_port = dst_node.get_input(alias)
                                if in_port:
                                    logger.info(f"Auto-fixed port alias: {to_port} -> {alias} on {dst_node.name()}")
                                    break
                        
                        if out_port and in_port:
                            out_port.connect_to(in_port)
                        else:
                            logger.warning(f"Port not found: {from_port} -> {to_port} on {src_node.name()}->{dst_node.name()}")
                    else:
                        logger.warning(f"Connection nodes not found: {from_id} -> {to_id}")

                except Exception as e:
                    logger.warning(f"Connection failed: {e}")
                
            logger.info(f"Graph update complete. Processed {len(nodes_spec)} nodes.")
            
            # E. Force Re-computation
            if hasattr(cad_widget, 'execute_graph'):
                cad_widget.execute_graph()
                
            return f"Created nodes: {created_names}"

        if sync:
            self._run_sync(build_graph_safe)
        else:
            # Invoke on main thread async
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, build_graph_safe)

    def _build_system_graph(self, data: Dict[str, Any], sync: bool = False) -> None:
        """
        Build a complete System node graph from LLM specification.
        Parallel to _build_node_graph but for the Modeling environment.
        """
        params = data.get("params", {})
        nodes_spec = params.get("nodes", [])
        conns_spec = params.get("connections", [])
        
        if not nodes_spec:
            logger.warning("Empty node spec for build_system_graph")
            return
            
        if not self.main_window:
            logger.warning("Main window required for build_system_graph")
            if sync: raise RuntimeError("Main window required")
            return
            
        # 1. Switch to Modeling tab (Index 0)
        try:
            tab_widget = self._get_tab_widget()
            if tab_widget and tab_widget.currentIndex() != 0:
                from PySide6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(tab_widget, "setCurrentIndex", Qt.QueuedConnection, 0)
        except Exception:
            pass
            
        # 2. Get Modeling Widget
        if not hasattr(self.main_window, 'modeling_widget'):
            logger.error("Modeling widget not found")
            if sync: raise RuntimeError("Modeling widget not found")
            return
        modeling_widget = self.main_window.modeling_widget
        graph_controller = getattr(modeling_widget, 'current_graph', None)
        
        if not graph_controller:
            logger.error("System Graph controller not found")
            if sync: raise RuntimeError("System Graph controller not found")
            return

        # 3. Execute Graph Building on GUI Thread
        def build_graph_safe():
            try:
                # Helper for property setting
                def set_prop_safe(node, prop_name, prop_val, node_identifier):
                    if not hasattr(node, 'set_property'):
                        return
                    try:
                        # Handle surrogate controls special case
                        if prop_name == 'use_surrogate':
                             node.set_property('use_surrogate', bool(prop_val))
                             return

                        if prop_name in node.properties():
                             node.set_property(prop_name, prop_val)
                    except Exception as e:
                        logger.warning(f"Failed to set property {prop_name} on {node_identifier}: {e}")

                id_to_node = {}
                
                # A. Map Existing Nodes
                existing_nodes = graph_controller.all_nodes()
                existing_by_name = {n.name(): n for n in existing_nodes}
                id_to_node.update(existing_by_name)

                # Determine start position
                start_x, start_y = 0, 0
                if existing_nodes:
                     start_y = max((n.pos()[1] for n in existing_nodes), default=0) + 200
                
                # B. Process Nodes
                new_node_count = 0
                
                for node_def in nodes_spec:
                    node_id = node_def.get("id")
                    node_type = node_def.get("type")
                    props = node_def.get("properties", {})
                    
                    if not node_id: continue
                        
                    if node_id in existing_by_name:
                        # Update
                        node = existing_by_name[node_id]
                        logger.info(f"Updating existing system node: {node_id}")
                        for prop_name, prop_val in props.items():
                            set_prop_safe(node, prop_name, prop_val, node_id)
                        id_to_node[node_id] = node
                    else:
                        # Create
                        if not node_type: continue
                            
                        node_class = SYSTEM_NODE_CLASS_MAPPING.get(node_type)
                        if not node_class:
                            logger.warning(f"Unknown system node type: {node_type}")
                            continue
                        
                        node = node_class()
                        node.set_name(str(node_id))
                        
                        # Position
                        row = new_node_count // 3
                        col = new_node_count % 3
                        node.set_pos(start_x + (col * 300), start_y + (row * 150))
                        new_node_count += 1
                        
                        graph_controller.add_node(node)
                        id_to_node[node_id] = node
                        
                        for prop_name, prop_val in props.items():
                            set_prop_safe(node, prop_name, prop_val, node_id)
                                 
                # C. Process Connections
                for conn in conns_spec:
                    try:
                        from_str = conn.get("from", "")
                        to_str = conn.get("to", "")
                        
                        if "." not in from_str or "." not in to_str: continue
                            
                        from_id, from_port = from_str.split(".", 1)
                        to_id, to_port = to_str.split(".", 1)
                        
                        src_node = id_to_node.get(from_id)
                        dst_node = id_to_node.get(to_id)
                        
                        if src_node and dst_node:
                            out_port = src_node.get_output(from_port)
                            in_port = dst_node.get_input(to_port)
                            if out_port and in_port:
                                out_port.connect_to(in_port)
                    except Exception as e:
                        logger.warning(f"Connection failed: {e}")

                logger.info(f"System Graph update complete. Processed {len(nodes_spec)} nodes.")
                
            except Exception as e:
                logger.error(f"System Graph build failed: {e}")

        if sync:
            self._run_sync(build_graph_safe)
        else:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, build_graph_safe)


    
    def _train_surrogate_node(self, data: Dict[str, Any] = None) -> None:
        """Trigger surrogate training for a specific node."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
            
        params = data.get("params", {}) if data else {}
        node_name = params.get("node_name")
        
        if not node_name:
            logger.warning("No node name specified for surrogate training")
            return
            
        def trigger_training():
            try:
                widget = self.main_window.modeling_widget
                graph = widget.current_graph
                # Find node by name
                nodes = graph.all_nodes()
                node = next((n for n in nodes if n.name() == node_name), None)
                
                if node and hasattr(node, 'surrogate_widget'):
                    # Trigger the click programmatically
                    node.surrogate_widget.btn_train.click()
                    logger.info(f"Triggered training for node {node_name}")
                else:
                    logger.warning(f"Node {node_name} not found or has no surrogate widget")
            except Exception as e:
                logger.error(f"Failed to trigger surrogate training: {e}")

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, trigger_training)

    def _get_sensitivity_results(self) -> None:
        """
        Retrieve sensitivity analysis results and inject them into LLM context.
        """
        if not self.main_window or not hasattr(self.main_window, 'sensitivity_widget'):
            return
            
        def fetch_results():
            try:
                widget = self.main_window.sensitivity_widget
                results = getattr(widget, 'last_results', None)
                
                msg = ""
                if not results:
                    msg = "No sensitivity results available. Please run analysis first."
                else:
                    lines = ["**Sensitivity Analysis Results:**"]
                    vars_ = results.get('variable_names', [])
                    total = results.get('total_order', [])
                    
                    # Sort by importance
                    combined = sorted(zip(vars_, total), key=lambda x: x[1], reverse=True)
                    
                    for v, t in combined:
                        lines.append(f"- {v}: {t:.4f}")
                        
                    msg = "\n".join(lines)
                
                # Check if we can send this to the LLM dialog
                if hasattr(self.main_window, '_llm_dialog') and self.main_window._llm_dialog:
                    dialog = self.main_window._llm_dialog
                    if hasattr(dialog, 'add_system_message'):
                        dialog.add_system_message(msg)
                    elif hasattr(dialog, 'chat_widget') and hasattr(dialog.chat_widget, 'add_message'):
                        # Fallback attempt
                        dialog.chat_widget.add_message("System", msg)
                
                logger.info(f"Sensitivity Results Retrieved: {msg}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch sensitivity results: {e}")

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, fetch_results)

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
    
    def _validate_graph(self, sync: bool = False) -> None:
        """Validate the current graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        from PySide6.QtCore import QMetaObject, Qt
        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection

        if hasattr(widget, 'validate_graph'):
            QMetaObject.invokeMethod(widget, "validate_graph", conn_type)
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
    
    def _cad_execute(self, sync: bool = False) -> None:
        """Execute/run the CAD graph."""
        if not self.main_window or not hasattr(self.main_window, 'cad_widget'):
            logger.warning("CAD widget not found")
            if sync: raise RuntimeError("CAD widget not found")
            return
        widget = self.main_window.cad_widget
        from PySide6.QtCore import QMetaObject, Qt
        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection
        
        if hasattr(widget, 'execute_graph'):
            QMetaObject.invokeMethod(widget, "execute_graph", conn_type)
            logger.info("Voice command: Executing CAD graph")
        elif hasattr(widget, 'btn_execute'):
            QMetaObject.invokeMethod(widget.btn_execute, "click", conn_type)
            logger.info("Voice command: Running CAD")
    
    def _cad_export(self, sync: bool = False) -> None:
        """Export the CAD model to STL."""
        if not self.main_window or not hasattr(self.main_window, 'cad_widget'):
            return
        widget = self.main_window.cad_widget
        from PySide6.QtCore import QMetaObject, Qt
        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection

        if hasattr(widget, 'btn_export'):
            QMetaObject.invokeMethod(widget.btn_export, "click", conn_type)
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
    
    def _clear_graph(self, sync: bool = False) -> None:
        """Clear all nodes from the graph."""
        if not self.main_window or not hasattr(self.main_window, 'modeling_widget'):
            return
        widget = self.main_window.modeling_widget
        from PySide6.QtCore import QMetaObject, Qt
        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection
        if hasattr(widget, 'current_graph') and hasattr(widget.current_graph, 'clear_session'):
            QMetaObject.invokeMethod(widget.current_graph, "clear_session", conn_type)
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
    
    # --- LLM Assistant Handlers ---
    
    def _open_llm_assistant(self) -> None:
        """Open the LLM assistant dialog (thread-safe via custom event)."""
        logger.info("LLM Assistant: _open_llm_assistant called")
        if not self.main_window:
            logger.warning("Main window not available for LLM assistant")
            return
        
        # Post a custom event to the main window to create/show the dialog
        from PySide6.QtCore import QEvent, QCoreApplication
        from PySide6.QtWidgets import QApplication
        
        # Use a lambda with QApplication.instance().postEvent is not possible
        # Instead, create a callable wrapper class
        class ShowLLMDialogEvent(QEvent):
            EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
            def __init__(self, dispatcher):
                super().__init__(ShowLLMDialogEvent.EVENT_TYPE)
                self.dispatcher = dispatcher
        
        # Install event filter on main window if not already done
        if not hasattr(self.main_window, '_llm_event_filter_installed'):
            original_event = self.main_window.event
            dispatcher = self  # Capture reference
            
            def patched_event(event):
                if isinstance(event, ShowLLMDialogEvent):
                    dispatcher._do_create_llm_dialog()
                    return True
                return original_event(event)
            
            self.main_window.event = patched_event
            self.main_window._llm_event_filter_installed = True
            logger.info("LLM Assistant: Event filter installed")
        
        # Post the event
        QCoreApplication.postEvent(self.main_window, ShowLLMDialogEvent(self))
        logger.info("LLM Assistant: Event posted to main thread")
    
    def _do_create_llm_dialog(self) -> None:
        """Actually create and show the LLM dialog (runs on main thread via event)."""
        logger.info("LLM Assistant: _do_create_llm_dialog executing on main thread")
        if not self.main_window:
            return
            
        # Check if dialog already exists
        if hasattr(self.main_window, '_llm_dialog') and self.main_window._llm_dialog:
            logger.info("LLM Assistant: Showing existing dialog")
            self.main_window._llm_dialog.show()
            self.main_window._llm_dialog.raise_()
            self.main_window._llm_dialog.activateWindow()
            return
            
        try:
            logger.info("LLM Assistant: Creating new dialog...")
            from pylcss.hands_free.ui.llm_chat_dialog import LLMChatDialog
            from pylcss.hands_free.config import HandsFreeConfig
            
            # Create dialog with command dispatcher reference
            dialog = LLMChatDialog(command_dispatcher=self, parent=self.main_window)
            logger.info("LLM Assistant: Dialog created successfully")
            
            # Load token from config if available
            config = HandsFreeConfig.load()
            if config.llm_control.access_token:
                dialog.set_token(config.llm_control.access_token)
                
            # Store reference on main window
            self.main_window._llm_dialog = dialog
            
            # Show dialog
            dialog.show()
            logger.info("LLM Assistant: Dialog shown!")
            
        except ImportError as e:
            logger.error(f"LLM Assistant: Failed to import dialog: {e}")
        except Exception as e:
            import traceback
            logger.error(f"LLM Assistant: Failed to open: {e}\n{traceback.format_exc()}")
    
    def _close_llm_assistant(self) -> None:
        """Close the LLM assistant dialog."""
        if self.main_window and hasattr(self.main_window, '_llm_dialog'):
            if self.main_window._llm_dialog:
                self.main_window._llm_dialog.close()
                logger.info("Voice command: Closed LLM assistant")
        
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
    
    # --- New Granular Control Handlers ---

    def _connect_nodes(self, data: Dict[str, Any], sync: bool = False) -> None:
        """Connect two nodes explicitly."""
        params = data.get("params", {})
        from_node_name = params.get("from_node")
        from_port_name = params.get("from_port", "shape")
        to_node_name = params.get("to_node")
        to_port_name = params.get("to_port")
        
        if not (from_node_name and to_node_name and to_port_name):
            logger.warning("Missing parameters for connect_nodes")
            if sync: raise ValueError("Missing parameters for connect_nodes")
            return

        if not self.main_window:
            if sync: raise RuntimeError("Main window required")
            return

        # Determine which environment we are in
        # Try CAD first, then Modeling
        target_graph = None
        if hasattr(self.main_window, 'cad_widget') and self.main_window.cad_widget.isVisible():
             target_graph = getattr(self.main_window.cad_widget, 'graph', None)
        elif hasattr(self.main_window, 'modeling_widget'):
              target_graph = getattr(self.main_window.modeling_widget, 'current_graph', None)
        
        if not target_graph:
             logger.warning("No active graph found for connection")
             if sync: raise RuntimeError("No active graph found for connection")
             return

        def do_connect():
            try:
                nodes = target_graph.all_nodes()
                src_node = next((n for n in nodes if n.name() == from_node_name), None)
                dst_node = next((n for n in nodes if n.name() == to_node_name), None)
                
                if src_node and dst_node:
                    out = src_node.get_output(from_port_name)
                    inp = dst_node.get_input(to_port_name)
                    if out and inp:
                        out.connect_to(inp)
                        logger.info(f"Connected {from_node_name}.{from_port_name} -> {to_node_name}.{to_port_name}")
                    else:
                        msg = f"Ports not found: {from_port_name} -> {to_port_name}"
                        logger.warning(msg)
                        if sync: raise ValueError(msg)
                else:
                    msg = f"Nodes not found: {from_node_name}, {to_node_name}"
                    logger.warning(msg)
                    if sync: raise ValueError(msg)
            except Exception as e:
                logger.error(f"Connect failed: {e}")
                if sync: raise

        if sync:
            self._run_sync(do_connect)
        else:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, do_connect)

    def _set_property(self, data: Dict[str, Any]) -> None:
        """Set a property on a specific node."""
        params = data.get("params", {})
        node_name = params.get("node_name")
        prop_name = params.get("property")
        prop_value = params.get("value")
        
        if not (node_name and prop_name):
            logger.warning("Missing parameters for set_property")
            return

        if not self.main_window:
            return

        target_graph = None
        if hasattr(self.main_window, 'cad_widget') and self.main_window.cad_widget.isVisible():
             target_graph = getattr(self.main_window.cad_widget, 'graph', None)
        elif hasattr(self.main_window, 'modeling_widget'):
              target_graph = getattr(self.main_window.modeling_widget, 'current_graph', None)
        
        if not target_graph:
             return

        def do_set():
            try:
                nodes = target_graph.all_nodes()
                node = next((n for n in nodes if n.name() == node_name), None)
                
                if node:
                    # Try to cast value type if property exists
                    if prop_name in node.properties():
                        # Simple type inference could go here if needed, 
                        # but NodeGraphQt usually handles it or we rely on JSON types
                        node.set_property(prop_name, prop_value)
                        logger.info(f"Set {node_name}.{prop_name} = {prop_value}")
                    else:
                        logger.warning(f"Property {prop_name} not found on {node_name}")
                else:
                    logger.warning(f"Node {node_name} not found")
            except Exception as e:
                logger.error(f"Set property failed: {e}")

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, do_set)
