# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hands-Free Manager - Main orchestrator for the assistant system.

Provides voice recognition and command execution for hands-free control.
Note: Camera-based head tracking has been removed.
"""

import logging
import json
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, QTimer, Qt, Slot

from pylcss.assistant_systems.config import AssistantConfig
from pylcss.assistant_systems.services.voice import VoiceController, WHISPER_AVAILABLE
from pylcss.assistant_systems.services.input import MouseController
from pylcss.assistant_systems.api.dispatcher import CommandDispatcher
from pylcss.assistant_systems.services.llm import (
    LLMProvider, get_provider, ChatCompletion, LLMProviderError
)
from pylcss.assistant_systems.services.memory import LLMMemory, get_secure_storage
from pylcss.assistant_systems.core.interpreter import LLMInterpreter, ParsedResponse, ActionCommand
from pylcss.assistant_systems.services.tts import get_tts, TextToSpeech

# Agentic AI components
try:
    from pylcss.assistant_systems.core.orchestrator import AgentOrchestrator
    from pylcss.assistant_systems.tools.registry import create_pylcss_tools, ToolRegistry
    from pylcss.assistant_systems.core.workflows import WorkflowLibrary, WorkflowRecorder, WorkflowPlayer
    AGENTIC_AVAILABLE = True
except ImportError as e:
    AGENTIC_AVAILABLE = False
    AgentOrchestrator = None
    logging_msg = f"Agentic AI components not available: {e}"

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class AssistantManager(QObject):
    """
    Main orchestrator for the hands-free control system.
    
    Manages voice control and mouse control for hands-free operation.
    Note: Camera-based head tracking has been removed.
    
    Signals:
        status_changed: Emitted when system status changes (str)
        command_recognized: Emitted when a voice command is recognized (str)
        partial_text: Emitted for partial voice recognition results (str)
        error_occurred: Emitted when an error occurs (str)
    """
    
    status_changed = Signal(str)
    command_recognized = Signal(str)
    partial_text = Signal(str)
    error_occurred = Signal(str)
    llm_response_received = Signal(object)  # Emits ChatCompletion
    llm_error_occurred = Signal(Exception)  # Emits Exception
    llm_request_received = Signal(str)      # Emits prompt text (fix for threading)
    voice_command_received = Signal(str, object) # Emits (command_name, command_data)
    
    # Agentic system signals for thread-safe UI updates
    agentic_result_received = Signal(dict, str)  # (result_dict, original_text)
    agentic_error_received = Signal(str)          # error_message
    agentic_progress = Signal(str)                # progress_message
    
    def __init__(self, main_window: Optional["QMainWindow"] = None):
        """
        Initialize the hands-free manager.
        
        Args:
            main_window: Reference to the PyLCSS main window
        """
        super().__init__()
        
        self.main_window = main_window
        self.config = AssistantConfig.load()
        
        # Connect signals
        self.llm_response_received.connect(self._process_llm_response)
        self.llm_error_occurred.connect(self._handle_llm_error)
        self.llm_request_received.connect(self._process_llm_request, Qt.QueuedConnection)
        self.voice_command_received.connect(self._process_voice_command, Qt.QueuedConnection)
        
        # Connect agentic signals
        self.agentic_result_received.connect(self._handle_agentic_result, Qt.QueuedConnection)
        self.agentic_error_received.connect(self._handle_agentic_error, Qt.QueuedConnection)
        self.agentic_progress.connect(self._handle_agentic_progress, Qt.QueuedConnection)
        
        # Components (head tracking removed - voice only)
        self._voice_controller: Optional[VoiceController] = None
        self._mouse_controller: Optional[MouseController] = None
        self._command_dispatcher: Optional[CommandDispatcher] = None
        
        # LLM components (multi-provider)
        self._llm_provider: Optional[LLMProvider] = None
        self._llm_interpreter: Optional[LLMInterpreter] = None
        self._llm_memory: Optional[LLMMemory] = None
        self._llm_overlay = None  # Created lazily to avoid import issues
        self._pending_actions: List[ActionCommand] = []
        self._secure_storage = get_secure_storage()
        
        # Agentic AI components
        self._agent_orchestrator: Optional['AgentOrchestrator'] = None
        self._workflow_library: Optional['WorkflowLibrary'] = None
        self._workflow_recorder: Optional['WorkflowRecorder'] = None
        self._workflow_player: Optional['WorkflowPlayer'] = None
        self._use_agentic_mode: bool = self.config.llm_control.agentic_mode and AGENTIC_AVAILABLE
        
        # TTS state for unmute during speech
        self._is_speaking = False
        self._speech_text = ""  # Text being spoken (shown until speech ends)
        self._speech_timer: Optional[QTimer] = None
        
        # State
        self._initialized = False
        self._running = False
        
        # Availability flags
        self.voice_control_available = WHISPER_AVAILABLE
        
    @property
    def command_dispatcher(self) -> Optional[CommandDispatcher]:
        """Get the command dispatcher instance."""
        return self._command_dispatcher
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if at least one component initialized successfully.
        """
        if self._initialized:
            return True
            
        success = False
        
        # Initialize mouse controller (always available)
        try:
            self._mouse_controller = MouseController()
            success = True
        except Exception as e:
            logger.error(f"Failed to initialize mouse controller: {e}")
            self.error_occurred.emit(f"Mouse control unavailable: {e}")
            
        # Initialize command dispatcher
        self._command_dispatcher = CommandDispatcher(
            main_window=self.main_window,
            mouse_controller=self._mouse_controller
        )
        
        # Set control callbacks
        self._command_dispatcher.set_control_callbacks(
            on_pause=self.pause,
            on_resume=self.resume,
            on_calibrate=None,  # Head tracking removed
            on_start_dictation=self._start_dictation,
            on_stop_dictation=self._stop_dictation,
        )
        
        # Initialize LLM components (multi-provider)
        try:
            self._llm_interpreter = LLMInterpreter()
            
            # Initialize memory system
            if self.config.llm_control.memory_enabled:
                self._llm_memory = LLMMemory(
                    max_conversations=100
                )
                logger.info(f"LLM memory initialized with {self._llm_memory.get_conversation_count()} conversations")
            
            # Initialize provider based on config
            self._initialize_llm_provider()
            
            logger.info("LLM components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {e}")
        
        # Initialize agentic AI components
        if self._use_agentic_mode and self._command_dispatcher:
            self._initialize_agentic_system()
        
        # Initialize voice controller
        if self.voice_control_available and self.config.voice_control.enabled:
            try:
                self._voice_controller = VoiceController(self.config.voice_control)
                
                # Set callbacks including LLM request handler
                self._voice_controller.set_callbacks(
                    on_command=self._on_voice_command,
                    on_text=self._on_dictation_text,
                    on_partial=self._on_partial_text,
                    on_status=self._on_voice_status,
                    on_llm_request=self._on_llm_request,
                )
                success = True
            except Exception as e:
                logger.error(f"Failed to initialize voice controller: {e}")
                self.error_occurred.emit(f"Voice control unavailable: {e}")
                
        self._initialized = success
        return success
        
    def start(self) -> bool:
        """
        Start the hands-free control system.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            return True
            
        if not self._initialized:
            if not self.initialize():
                self.error_occurred.emit("Failed to initialize hands-free system")
                return False
                
        started = False
        
        # Start voice control
        if self._voice_controller and self.config.voice_control.enabled:
            if self._voice_controller.start():
                started = True
                self.status_changed.emit("Voice control active")
            else:
                # Check if model is missing
                if not self._voice_controller.is_model_available():
                    info = self._voice_controller.get_model_download_info()
                    self.error_occurred.emit(
                        f"Vosk model not found. Please download from:\n{info['url']}\n"
                        f"and extract to:\n{info['path']}"
                    )
                else:
                    self.error_occurred.emit("Failed to start voice control")
                    
        self._running = started
        
        if started:
            self.status_changed.emit("Voice control active")
        else:
            self.status_changed.emit("Voice control failed to start")
            
        return started
        
    def stop(self) -> None:
        """Stop the hands-free control system."""
        if self._voice_controller:
            self._voice_controller.stop()
            
        self._running = False
        self.status_changed.emit("Voice control stopped")
        
    def pause(self) -> None:
        """Pause voice control."""
        if self._voice_controller:
            self._voice_controller.pause()
            
        self.status_changed.emit("Voice control paused")
        
    def resume(self) -> None:
        """Resume voice control."""
        if self._voice_controller:
            self._voice_controller.resume()
            
        self.status_changed.emit("Voice control resumed")
        
    def is_running(self) -> bool:
        """Check if the system is running."""
        return self._running
        
    def is_voice_control_active(self) -> bool:
        """Check if voice control is active."""
        return self._voice_controller is not None and self._voice_controller.is_running()
            
    def set_voice_control_enabled(self, enabled: bool) -> None:
        """Enable or disable voice control."""
        self.config.voice_control.enabled = enabled
        
        if enabled and self._running:
            if self._voice_controller:
                self._voice_controller.start()
        elif not enabled and self._voice_controller:
            self._voice_controller.stop()
            
    def get_config(self) -> AssistantConfig:
        """Get the current configuration."""
        return self.config
        
    def update_config(self, config: AssistantConfig) -> None:
        """Update configuration (caller should save if needed)."""
        logger.info("Updating AssistantManager configuration...")
        self.config = config
            
        # Update voice controller config
        if self._voice_controller:
            self._voice_controller.config = config.voice_control
        
        # Re-initialize LLM provider if settings changed
        self._llm_provider = None  # Force clear
        self._initialize_llm_provider()
        
        # NOTE: We no longer save here - the caller is responsible for saving
        # This prevents double-saves that can corrupt encrypted data
    
    def _initialize_llm_provider(self) -> None:
        """Initialize or re-initialize the LLM provider based on config."""
        llm_config = self.config.llm_control
        provider_name = llm_config.provider
        encrypted_key = llm_config.get_api_key_for_provider(provider_name)
        
        logger.info(f"Initializing LLM provider: {provider_name}, Key length: {len(encrypted_key) if encrypted_key else 0}")
        
        if not encrypted_key:
            logger.warning(f"No API key set for provider {provider_name}")
            self._llm_provider = None
            return
        
        # Decrypt the key
        try:
            api_key = self._secure_storage.decrypt(encrypted_key)
            if not api_key:
                # Try as unencrypted (legacy)
                api_key = encrypted_key
        except Exception:
            api_key = encrypted_key
        
        if not api_key:
            logger.warning(f"Failed to decrypt API key for {provider_name}")
            self._llm_provider = None
            return
        
        try:
            kwargs = {}
            if provider_name == "local":
                kwargs["local_api_url"] = llm_config.local_api_url
                kwargs["selected_model"] = llm_config.selected_model or llm_config.model
            
            self._llm_provider = get_provider(provider_name, api_key, **kwargs)
            self._llm_provider.temperature = llm_config.temperature
            self._llm_provider.max_tokens = llm_config.max_tokens
            
            # Set system prompt
            if self._llm_interpreter:
                self._llm_provider.set_system_prompt(self._llm_interpreter.get_system_prompt())
            
            # Restore memory context if available
            if self._llm_memory:
                context = self._llm_memory.get_context_messages(llm_config.max_memory_messages)
                for msg in context:
                    if msg["role"] != "system":
                        self._llm_provider._messages.append(
                            type('Message', (), {'role': msg["role"], 'content': msg["content"]})()
                        )
            
            logger.info(f"LLM provider initialized: {provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {provider_name}: {e}")
            self._llm_provider = None
    
    def set_llm_provider(self, provider: Optional[LLMProvider]) -> None:
        """Set the LLM provider directly (from UI)."""
        self._llm_provider = provider
        if provider and self._llm_interpreter:
            provider.set_system_prompt(self._llm_interpreter.get_system_prompt())
        logger.info(f"LLM provider set: {provider.name if provider else 'None'}")
        
        # Initialize agentic system if not already done (lazy init on first provider)
        if self._use_agentic_mode and provider and not self._agent_orchestrator:
            self._initialize_agentic_system()
        elif self._agent_orchestrator and provider:
            self._agent_orchestrator.update_provider(provider)
    
    def _initialize_agentic_system(self) -> None:
        """Initialize the multi-agent system for agentic AI."""
        if not AGENTIC_AVAILABLE:
            logger.warning("Agentic AI components not available")
            return
            
        if not self._llm_provider:
            logger.warning("Cannot initialize agentic system without LLM provider")
            return
            
        try:
            # Create tool registry with all PyLCSS tools
            tool_registry = create_pylcss_tools(self._command_dispatcher)
            
            # Create agent orchestrator
            self._agent_orchestrator = AgentOrchestrator(
                llm_provider=self._llm_provider,
                tool_registry=tool_registry,
                use_critic=self.config.llm_control.use_critic_agent,
                validate_design_intent=self.config.llm_control.validate_design_intent,
                max_retries=self.config.llm_control.max_retries,
                on_step_complete=self._on_agent_step_complete,
            )
            
            # Initialize workflow system
            self._workflow_library = WorkflowLibrary()
            self._workflow_recorder = WorkflowRecorder(
                library=self._workflow_library,
                auto_save=self.config.llm_control.auto_save_workflows,
            )
            self._workflow_player = WorkflowPlayer(
                executor_agent=self._agent_orchestrator.executor,
                library=self._workflow_library,
            )
            
            logger.info(f"Agentic AI system initialized with {len(tool_registry.all_tools)} tools, "
                       f"{self._workflow_library.count} workflows")
            
        except Exception as e:
            logger.error(f"Failed to initialize agentic system: {e}")
            self._agent_orchestrator = None
            self._workflow_library = None
            self._workflow_recorder = None
            self._workflow_player = None
    
    def _on_agent_step_complete(self, step) -> None:
        """Callback for agent step completion (for UI updates)."""
        if step.result:
            if step.result.success:
                logger.info(f"Agent step completed: {step.action.tool_name if step.action else 'unknown'}")
            else:
                logger.warning(f"Agent step failed: {step.result.error}")
        
        
    # --- Callbacks from components ---
    def _on_voice_command(self, command_name: str, command_data: Dict[str, Any]) -> None:
        """Handle recognized voice command (called from background thread)."""
        # Route to main thread via signal
        self.voice_command_received.emit(command_name, command_data)

    def _process_voice_command(self, command_name: str, command_data: Dict[str, Any]) -> None:
        """Process voice command on main thread."""
        self.command_recognized.emit(command_name)
        
        # Check for LLM control commands
        action = command_data.get("action")
        command = command_data.get("command")
        
        if action == "control":
            if command == "llm_mode":
                self.start_llm_mode()
                return
            elif command == "llm_confirm":
                self.confirm_llm_actions()
                return
            elif command == "llm_cancel":
                self.cancel_llm_actions()
                return
            elif command == "llm_exit":
                self.exit_llm_mode()
                return
        
        # Standard command dispatch
        if self._command_dispatcher:
            self._command_dispatcher.dispatch(command_name, command_data)
            
    def _on_dictation_text(self, text: str) -> None:
        """Handle dictation text."""
        if self._mouse_controller:
            self._mouse_controller.write_text(text + " ")
            
    def _on_partial_text(self, text: str) -> None:
        """Handle partial recognition results."""
        self.partial_text.emit(text)
        
        # Also show in LLM overlay if active
        if self._voice_controller and self._voice_controller.is_llm_mode():
            overlay = self._get_or_create_llm_overlay()
            if overlay:
                overlay.show_partial(text)
        
    def _on_voice_status(self, status: str) -> None:
        """Handle voice controller status updates."""
        self.status_changed.emit(status)
    def _start_dictation(self) -> None:
        """Start dictation mode."""
        if self._voice_controller:
            self._voice_controller.start_dictation()
            
    def _stop_dictation(self) -> None:
        """Stop dictation mode."""
        if self._voice_controller:
            self._voice_controller.stop_dictation()
    
    # --- LLM Voice-First Mode Handlers ---
    
    def _get_or_create_llm_overlay(self):
        """Get the LLM overlay - use main window's overlay widget for corner display."""
        # Use the main window's hands_free_overlay which is in the top-right corner
        if self.main_window and hasattr(self.main_window, 'hands_free_overlay'):
            overlay = self.main_window.hands_free_overlay
            if overlay:
                overlay.show()  # Make sure it's visible
                return overlay
        
        # Fallback: create separate overlay (shouldn't normally happen)
        if self._llm_overlay is None:
            try:
                from pylcss.assistant_systems.ui.overlay_widget import OverlayWidget
                self._llm_overlay = OverlayWidget()
                logger.info("LLM Overlay created (standalone)")
            except Exception as e:
                logger.error(f"Failed to create LLM overlay: {e}")
        return self._llm_overlay
    
    def _on_llm_request(self, text: str) -> None:
        """Handle speech routed to LLM (from VoiceController)."""
        # This is called from a background thread, so we must use a signal
        # to pass execution to the main thread where UI lives.
        self.llm_request_received.emit(text)

    def _get_current_graph_context(self) -> str:
        """Serialize the current graph to JSON."""
        try:
            if not self.main_window:
                return ""
            
            # Determine active tab (0=Modeling, 1=CAD)
            tab_index = 0
            if hasattr(self.main_window, 'tabs'):
                tab_index = self.main_window.tabs.currentIndex()
            elif hasattr(self.main_window, 'tab_widget'):
                tab_index = self.main_window.tab_widget.currentIndex()
                
            graph = None
            context_type = ""
            
            if tab_index == 0: # Modeling
                if hasattr(self.main_window, 'modeling_widget'):
                    graph = self.main_window.modeling_widget.current_graph
                context_type = "Modeling"
            elif tab_index == 1: # CAD
                if hasattr(self.main_window, 'cad_widget'):
                    graph = getattr(self.main_window.cad_widget, 'graph', None)
                context_type = "CAD"
                
            if not graph:
                return ""
                
            # Serialize nodes
            nodes_data = []
            for node in graph.all_nodes():
                node_data = {
                    "id": node.name(),
                    "type": node.type_,
                    "properties": {}
                }
                # Strict property filtering to save tokens
                # We only want functional properties (dimensions, positions, etc.)
                # Exclude purely visual or internal state
                ignore_starts = ['_', 'error', 'port', 'selected', 'disabled']
                ignore_exact = ['pos', 'color', 'border_color', 'text_color', 'icon', 'width', 'height'] # width/height are often visual unless node is a Box
                
                for name, val in node.properties().items():
                    name_lower = name.lower()
                    
                    # 1. Skip ignored prefixes
                    if any(name_lower.startswith(p) for p in ignore_starts):
                        continue
                        
                    # 2. Skip ignored exact matches (visual props)
                    # NOTE: Be careful not to filter 'width' if it's a Box property!
                    # NodeGraphQt adds generic 'width'/'height' visual props to all nodes. 
                    # Our BoxNode uses 'box_width'/'box_length', so 'width' is safe to ignore.
                    if name_lower in ignore_exact:
                        continue
                        
                    node_data["properties"][name] = val
                    
                nodes_data.append(node_data)
                
            # Serialize connections
            connections_data = []
            for node in graph.all_nodes():
                for out_port in node.output_ports():
                    for cp in out_port.connected_ports():
                        conn = {
                            "from": f"{node.name()}.{out_port.name()}",
                            "to": f"{cp.node().name()}.{cp.name()}"
                        }
                        connections_data.append(conn)
            
            context = {
                "environment": context_type,
                "nodes": nodes_data,
                "connections": connections_data
            }
            # Use compact JSON separators (no spaces) and no indentation
            return json.dumps(context, separators=(',', ':'))
            
        except Exception as e:
            logger.error(f"Context error: {e}")
            return ""

    def _process_llm_request(self, text: str) -> None:
        """Process LLM request in main thread."""
        logger.info(f"LLM Request received: {text}")
        
        if not self._llm_provider:
            logger.error("LLM provider not initialized")
            self._initialize_llm_provider()
            if not self._llm_provider:
                self.error_occurred.emit("LLM not configured.")
                if self._voice_controller:
                    self._voice_controller.stop_llm_mode()
                return

        # Check for workflow triggers first
        if self._workflow_library:
            workflow = self._workflow_library.find_by_phrase(text)
            if workflow:
                logger.info(f"Matched workflow: {workflow.name}")
                self._play_workflow(workflow, text)
                return
        
        # Use agentic mode if enabled and available
        if self._use_agentic_mode and self._agent_orchestrator:
            self.process_agentic_request(text)
            return

        # === Legacy single-shot processing ===
        # 1. Get Graph Context
        context = self._get_current_graph_context()
        full_message = text
        if context:
            full_message += f"\n\n[Current Graph State]:\n{context}"
            
        # 2. Add to Memory (System sees full context, User sees clean text in history?)
        # Actually better to store what we send so LLM has history of state.
        if self._llm_memory:
            self._llm_memory.add_message(
                role="user",
                content=full_message,
                provider=self.config.llm_control.provider,
                model=self.config.llm_control.selected_model or self.config.llm_control.model
            )
            

            
        # Show thinking overlay with user's text
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_thinking(user_text=text)
        
        # Pause voice recognition while processing
        if self._voice_controller:
            self._voice_controller.pause()
            
        # Send to LLM asynchronously using multi-provider
        model = self.config.llm_control.selected_model or self.config.llm_control.model
        self._llm_provider.chat_async(
            user_message=full_message,
            on_complete=self._on_llm_response,
            on_error=self._on_llm_error,
            model=model,
        )
    
    def process_agentic_request(self, text: str) -> None:
        """Process request through the multi-agent system (in background thread)."""
        logger.info(f"Processing agentic request: {text[:100]}...")
        
        # Store for overlay updates
        self._current_request_text = text
        
        # Show thinking overlay
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_thinking(user_text=text)
            
        # Pause voice recognition while processing
        if self._voice_controller:
            self._voice_controller.pause()
        
        # Get current graph context (must be done on main thread)
        graph_state = self._get_current_graph_context()
        
        # Run in background thread to avoid UI freeze
        import threading
        
        def run_agentic():
            try:
                # Emit progress for planning phase
                self.agentic_progress.emit("Planning...")
                
                # Process through orchestrator
                result = self._agent_orchestrator.process_request(
                    user_request=text,
                    graph_state=graph_state,
                    get_graph_state=self._get_current_graph_context,
                )
                
                # Emit result signal (works with dict and str)
                self.agentic_result_received.emit(result, text)
                
            except Exception as e:
                logger.error(f"Agentic processing failed: {e}")
                self.agentic_error_received.emit(str(e))
        
        thread = threading.Thread(target=run_agentic, daemon=True)
        thread.start()
    
    @Slot(object, str)
    def _handle_agentic_result(self, result: dict, original_text: str) -> None:
        """Handle agentic result on main thread."""
        overlay = self._get_or_create_llm_overlay()
        
        # Handle result
        message = result.get("message", "Completed.")
        success = result.get("success", False)
        plan = result.get("plan")
        results = result.get("results", [])
        
        # Auto-save successful workflows
        if success and self._workflow_recorder and plan and len(plan.steps) >= 2:
            self._workflow_recorder.auto_record_from_plan(plan, results, original_text)
        
        # Add to memory
        if self._llm_memory:
            self._llm_memory.add_message(
                role="assistant",
                content=message,
                provider=self.config.llm_control.provider,
                model=self.config.llm_control.selected_model or "agentic"
            )
        
        # Update overlay
        if overlay:
            if success:
                overlay.show_response(message, has_actions=False)
            else:
                overlay.show_error(message)
        
        # Speak result
        self._do_speak(message, len(message.split()) * 500)
        
        # Resume voice
        if self._voice_controller:
            self._voice_controller.resume()
    
    @Slot(str)
    def _handle_agentic_error(self, error_msg: str) -> None:
        """Handle agentic error on main thread."""
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_error(error_msg)
        
        # Resume voice on error
        if self._voice_controller:
            self._voice_controller.resume()

    @Slot(str)
    def _handle_agentic_progress(self, message: str) -> None:
        """Handle progress updates on main thread."""
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            # Use stored request text if available
            user_text = getattr(self, '_current_request_text', "")
            overlay.show_thinking(user_text=user_text, detail_text=message)

    def _on_agent_step_complete(self, step) -> None:
        """Callback for agent step completion (for UI updates)."""
        # Emit progress signal instead of just logging
        if step.result:
            if step.result.success:
                msg = f"Completed: {step.action.tool_name if step.action else 'unknown'}"
                logger.info(f"Agent step completed: {step.action.tool_name if step.action else 'unknown'}")
            else:
                msg = f"Failed: {step.action.tool_name if step.action else 'unknown'} (Retrying...)"
                logger.warning(f"Agent step failed: {step.result.error}")
            
            # Emit safely from background thread
            self.agentic_progress.emit(msg)
        
    def _on_agent_plan_created(self, plan) -> None:
        """Callback for plan creation."""
        msg = f"Plan created: {len(plan.steps)} steps"
        self.agentic_progress.emit(msg)
    
    def _play_workflow(self, workflow, trigger_text: str) -> None:
        """Play a saved workflow."""
        if not self._workflow_player:
            logger.warning("Workflow player not initialized")
            return
            
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_thinking(user_text=f"Running workflow: {workflow.name}")
            
        # Pause voice
        if self._voice_controller:
            self._voice_controller.pause()
            
        try:
            context = {
                "graph_state": self._get_current_graph_context(),
                "get_graph_state": self._get_current_graph_context,
            }
            
            results = self._workflow_player.play(workflow, context)
            
            success = all(r.success for r in results)
            if success:
                message = f"✅ Workflow '{workflow.name}' completed ({len(results)} steps)"
            else:
                failed_count = sum(1 for r in results if not r.success)
                message = f"⚠️ Workflow '{workflow.name}' had {failed_count} failures"
                
            if overlay:
                overlay.show_response(message, has_actions=False)
                
            self._do_speak(message, len(message.split()) * 500)
            
        except Exception as e:
            logger.error(f"Workflow playback failed: {e}")
            if overlay:
                overlay.show_error(str(e))
            if self._voice_controller:
                self._voice_controller.resume()
        
        
    def _on_llm_response(self, completion: ChatCompletion) -> None:
        """Handle LLM response (called from background thread)."""
        logger.info(f"LLM Response: {completion.content[:100]}...")
        # Emit signal to process in main thread
        self.llm_response_received.emit(completion)
        
    def _on_llm_error(self, error: Exception) -> None:
        """Handle LLM error (called from background thread)."""
        logger.error(f"LLM Error: {error}")
        # Emit signal to handle in main thread
        self.llm_error_occurred.emit(error)
    
    def _process_llm_response(self, completion: ChatCompletion) -> None:
        """Process LLM response in main thread."""
        overlay = self._get_or_create_llm_overlay()
        
        # Parse response for actions
        parsed = self._llm_interpreter.parse_response(completion.content)
        logger.info(f"Parsed {len(parsed.actions)} actions from LLM response")
        
        # Save assistant response to memory
        if self._llm_memory:
            self._llm_memory.add_message(
                role="assistant",
                content=completion.content,
                provider=completion.provider,
                model=completion.model
            )
        
        # Store pending actions
        self._pending_actions = parsed.actions
        
        # Get the message to display/speak
        message = parsed.message or completion.content
        
        # Calculate speech duration for timeout
        speak_text = message[:500] if len(message) > 500 else message
        if parsed.actions:
            speak_text += f". Executing {len(parsed.actions)} actions."
        
        
        word_count = len(speak_text.split())
        # Assume slower speech (120 WPM) and add larger buffer (5s) to ensure text stays visible
        estimated_duration_ms = int((word_count / 120) * 60 * 1000) + 5000
        
        # Update overlay - text stays visible until speech ends
        if overlay:
            has_actions = len(parsed.actions) > 0
            # Pass calculated timeout to keep overlay visible during speech
            if hasattr(overlay, 'show_llm_response'):
                 # Check if show_llm_response accepts timeout_ms (it might not if not updated)
                 # We assume it is updated.
                 try:
                     overlay.show_response(message, has_actions=has_actions, timeout_ms=estimated_duration_ms + 2000)
                 except TypeError:
                     # Fallback for old signature
                     overlay.show_response(message, has_actions=has_actions)
        
        # Store speech text for display
        self._speech_text = message
        self._is_speaking = True
            
            
        # Execute actions immediately but DO NOT resume mic, speak, or show overlay (handled by LLM response flow)
        if parsed.actions:
            logger.info(f"Auto-executing {len(parsed.actions)} actions")
            # Call renamed method
            self.execute_actions_safe(resume_listening=False, speak_confirmation=False, show_overlay=False)
        
        # Speak the response using TTS with unmute
        # Delay speech slightly to allow UI (overlay) to update first
        QTimer.singleShot(50, lambda: self._do_speak(speak_text, estimated_duration_ms))
        
    def _do_speak(self, text: str, duration_ms: int) -> None:
        """Actual speech execution with exact callback sync."""
        try:
            tts = get_tts()
            if tts.is_available():
                
                # Ensure we are paused BEFORE speaking
                if self._voice_controller:
                    self._voice_controller.pause()
                    
                # Define callback to resume (called from background thread)
                def on_speech_done():
                    # Invoke on main thread to be safe with QT
                    from PySide6.QtCore import QMetaObject, Qt, QTimer
                    # Using a tiny delay before unmute helps avoid catching the very end of the audio buffer
                    # if there's any system latency.
                    QMetaObject.invokeMethod(self, "_on_speech_finished", Qt.QueuedConnection)

                # Start speaking with callback
                tts.speak(text, on_complete=on_speech_done)
                logger.info(f"TTS: Speaking LLM response (approx {duration_ms}ms)")
                
                # We NO LONGER use the timer for flow control, only as a fail-safe
                # backup in case the callback never fires (e.g. thread crash)
                if self._speech_timer:
                    self._speech_timer.stop()
                    
                self._speech_timer = QTimer(self)
                self._speech_timer.setSingleShot(True)
                # Fail-safe timeout: estimated duration + 5 seconds
                self._speech_timer.timeout.connect(self._on_speech_finished)
                self._speech_timer.start(duration_ms + 5000)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Resume immediately on error
            self._on_speech_finished()

    # Old method removed/replaced by _do_speak logic inside _process_llm_response
    def _speak_with_unmute(self, message: str, actions: List[ActionCommand]) -> None:
         # Backward compatibility stub if needed, but we replaced the call site
         pass
    
    # ... (other methods)
    
    def execute_actions_safe(self, resume_listening: bool = True, speak_confirmation: bool = True, show_overlay: bool = True) -> None:
        """Execute pending LLM actions (Renamed to bypass cache)."""
        logger.info(f"SAFE EXEC: Executing {len(self._pending_actions)} actions")
        
        if self._pending_actions and self._command_dispatcher:
            for action in self._pending_actions:
                result = self._llm_interpreter.action_to_dispatcher_format(action)
                if result:
                    command_name, command_data = result
                    try:
                        self._command_dispatcher.dispatch(command_name, command_data)
                        logger.info(f"Executed action: {action.command}")
                    except Exception as e:
                        logger.error(f"Failed to execute action {action.command}: {e}")
                        
        count = len(self._pending_actions)
        self._pending_actions = []
        
        if show_overlay:
            overlay = self._get_or_create_llm_overlay()
            if overlay:
                overlay.show_confirmed()
            
        # Speak confirmation
        if speak_confirmation:
            try:
                tts = get_tts()
                if tts.is_available():
                    tts.speak(f"Executed {count} actions.")
            except:
                pass
            
        # Do not exit LLM mode automatically, keep conversation open
        
        # Resume listening ONLY if requested (don't if about to speak long text)
        if resume_listening and self._voice_controller:
            self._voice_controller.resume()
            
    # Force recompile check
    _force_recompile = True
    
    def _on_speech_finished(self) -> None:
        """Called when TTS speech is estimated to have finished."""
        self._is_speaking = False
        self._speech_text = ""
        
        overlay = self._get_or_create_llm_overlay()
        
        # Resume voice recognition
        if self._voice_controller:
            self._voice_controller.resume()
        
        # If still in LLM mode (no actions were executed), show listening state
        if self._voice_controller and self._voice_controller.is_llm_mode():
            if overlay:
                overlay.show_listening()
            logger.info("Speech finished, resuming listening in LLM mode")
            
    def _handle_llm_error(self, error: Exception) -> None:
        """Handle LLM error in main thread."""
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_error(str(error))
        
        # Resume voice recognition
        if self._voice_controller:
            self._voice_controller.resume()
            
        # Speak error
        try:
            tts = get_tts()
            if tts.is_available():
                tts.speak("Sorry, I encountered an error.")
        except:
            pass
            
        # Exit LLM mode on error
        if self._voice_controller:
            self._voice_controller.stop_llm_mode()
            
    def start_llm_mode(self) -> None:
        """Start LLM voice-first mode."""
        # Reload config to pick up any changes (e.g. new API key)
        try:
            new_config = HandsFreeConfig.load()
            self.config = new_config
            
            # Update voice controller config
            if self._voice_controller:
                self._voice_controller.config = new_config.voice_control
                self._voice_controller.start_llm_mode()
            
            # Re-initialize LLM provider with new config
            self._initialize_llm_provider()
            
            if not self._llm_provider:
                self.error_occurred.emit("LLM not configured. Please set API key in LLM Settings.")
                if self._voice_controller:
                    self._voice_controller.stop_llm_mode()
                return
            
            # Create new conversation in memory
            if self._llm_memory:
                self._llm_memory.new_conversation(
                    provider=self.config.llm_control.provider,
                    model=self.config.llm_control.selected_model or self.config.llm_control.model
                )
                logger.info("New LLM conversation started")
                
        except Exception as e:
            logger.error(f"Failed to start LLM mode: {e}")
            self.error_occurred.emit(f"Failed to start LLM: {e}")
            if self._voice_controller:
                self._voice_controller.stop_llm_mode()
            return
            
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_listening()
            
        # Optional: Speak listening prompt
        # try:
        #     get_tts().speak("I'm listening", async_=True)
        # except:
        #     pass
            
    def confirm_llm_actions(self) -> None:
        """Execute pending LLM actions."""
        logger.info(f"Confirming {len(self._pending_actions)} LLM actions")
        
        if self._pending_actions and self._command_dispatcher:
            for action in self._pending_actions:
                result = self._llm_interpreter.action_to_dispatcher_format(action)
                if result:
                    command_name, command_data = result
                    try:
                        self._command_dispatcher.dispatch(command_name, command_data)
                        logger.info(f"Executed action: {action.command}")
                    except Exception as e:
                        logger.error(f"Failed to execute action {action.command}: {e}")
                        
        count = len(self._pending_actions)
        self._pending_actions = []
        
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_confirmed()
            
        # Speak confirmation
        try:
            tts = get_tts()
            if tts.is_available():
                tts.speak(f"Executed {count} actions.")
        except:
            pass
            
        # Do not exit LLM mode automatically, keep conversation open
        # if self._voice_controller:
        #     self._voice_controller.stop_llm_mode()
        
        # Resume listening
        if self._voice_controller:
            self._voice_controller.resume()
            
    def cancel_llm_actions(self) -> None:
        """Cancel pending LLM actions."""
        logger.info("Cancelling LLM actions")
        self._pending_actions = []
        
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_cancelled()
            
        # Speak cancellation
        try:
            tts = get_tts()
            if tts.is_available():
                tts.speak("Cancelled.")
        except:
            pass
            
        # Exit LLM mode
        if self._voice_controller:
            self._voice_controller.stop_llm_mode()
            
    def exit_llm_mode(self) -> None:
        """Exit LLM mode without executing."""
        logger.info("Exiting LLM mode")
        self._pending_actions = []
        
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.hide_overlay()
        
        if self._voice_controller:
            self._voice_controller.stop_llm_mode()
            
        if self._voice_controller:
            self._voice_controller.stop_llm_mode()
