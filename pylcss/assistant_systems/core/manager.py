# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hands-Free Manager - Main orchestrator for the assistant system.

Provides voice recognition and command execution for hands-free control.
Note: Camera-based head tracking has been removed.
"""

import logging
import json
import re
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
from pylcss.assistant_systems.services.tts import get_tts, TextToSpeech

# Agentic AI components -- PydanticAI native function-calling path replaces
# the legacy supervisor/specialist/executor JSON-plan loop. Workflow
# recording is paused: it relied on the legacy plan/step model and will be
# re-implemented on top of PydanticAgentRunner traces in a follow-up.
try:
    from pylcss.assistant_systems.tools.registry import create_pylcss_tools, ToolRegistry
    from pylcss.assistant_systems.services.pydantic_agent import (
        PydanticAgentRunner, PydanticAgentResult,
    )
    AGENTIC_AVAILABLE = True
except ImportError as e:
    AGENTIC_AVAILABLE = False
    PydanticAgentRunner = None
    PydanticAgentResult = None
    logging_msg = f"Agentic AI components not available: {e}"

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


# Sensible default models per provider for when the user picked a provider but
# never selected a specific model. Updated per the model-knowledge cutoff in
# this codebase.
_DEFAULT_MODEL_FOR_PROVIDER = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-2.5-flash",
    "gemini":    "gemini-2.5-flash",
    "local":     "qwen2.5-7b-instruct",
}

# Path suffixes that the OpenAI client appends automatically; if a user
# accidentally saves the full endpoint URL we strip them here so the client
# does not double-append them.
_OPENAI_CLIENT_SUFFIXES = (
    "/chat/completions",
    "/completions",
)


def _normalize_base_url(url: Optional[str]) -> Optional[str]:
    """Strip path segments the OpenAI client appends automatically.

    If a user configures ``http://host:1234/v1/chat/completions`` instead of
    ``http://host:1234/v1`` the client produces a doubled path.  This helper
    removes the superfluous suffix and any trailing slash so the caller always
    gets a clean base URL (or ``None`` if nothing was provided).
    """
    if not url:
        return None
    cleaned = url.rstrip("/")
    for suffix in _OPENAI_CLIENT_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned or None


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
        # llm_response_received used to drive the legacy interpreter; the
        # PydanticAI path emits agentic_result_received / agentic_error_received
        # instead. The signal object is kept so any external listener still
        # connected to it doesn't crash on disconnect.
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
        
        # LLM components (multi-provider) -- LLMProvider is the legacy
        # text-only client kept for backwards compat with anything that still
        # asks for ``manager.llm_provider``; PydanticAgentRunner now owns the
        # real conversation loop.
        self._llm_provider: Optional[LLMProvider] = None
        self._llm_memory: Optional[LLMMemory] = None
        self._llm_overlay = None  # Created lazily to avoid import issues
        self._secure_storage = get_secure_storage()
        
        # Agentic AI components -- single PydanticAgentRunner replaces the
        # old supervisor/specialist/executor stack and workflow library.
        self._agent_runner: Optional['PydanticAgentRunner'] = None
        self._workflow_library = None  # legacy attr kept None for back-compat
        self._workflow_recorder = None
        self._workflow_player = None
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
        
        # Initialize LLM components -- legacy LLMInterpreter is gone;
        # PydanticAgentRunner (built below) owns tool dispatch.
        try:
            if self.config.llm_control.memory_enabled:
                self._llm_memory = LLMMemory(max_conversations=100)
                logger.info(f"LLM memory initialized with {self._llm_memory.get_conversation_count()} conversations")

            # Provider object is kept around for any UI code still asking for
            # ``manager.llm_provider`` (config dialog model lists, etc.),
            # but it is not consulted for chat any more.
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
                # Surface the actual reason from voice.py instead of the
                # generic "Failed to start" string that used to send users
                # straight to "revert the whole stack".
                detail = (
                    getattr(self._voice_controller, "get_last_error", lambda: None)()
                    or "Unknown error -- check the terminal log."
                )
                if not self._voice_controller.is_model_available():
                    self.error_occurred.emit(
                        f"Voice control unavailable: speech model missing.\n{detail}"
                    )
                else:
                    self.error_occurred.emit(f"Failed to start voice control.\n{detail}")
                    
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

            # System prompt + memory restore moved into PydanticAgentRunner;
            # the legacy provider object below is only kept for the LLM
            # Settings dialog (which still uses provider.list_models()).

            logger.info(f"LLM provider initialized: {provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {provider_name}: {e}")
            self._llm_provider = None
    
    def set_llm_provider(self, provider: Optional[LLMProvider]) -> None:
        """Set the LLM provider directly (from UI)."""
        self._llm_provider = provider
        logger.info(f"LLM provider set: {provider.name if provider else 'None'}")
        
        # Lazy-init the PydanticAgentRunner on first provider set.  Subsequent
        # provider changes rebuild the runner so the new provider's native
        # function-calling client takes over immediately.
        if self._use_agentic_mode and provider:
            self._initialize_agentic_system()
    
    def _initialize_agentic_system(self) -> None:
        """Build the PydanticAgentRunner from the current LLM config + tool
        registry.  Replaces the old supervisor/executor stack.
        """
        if not AGENTIC_AVAILABLE:
            logger.warning("Agentic AI components not available")
            return

        if not self._command_dispatcher:
            logger.warning("Cannot initialize agentic system without command dispatcher")
            return

        try:
            tool_registry = create_pylcss_tools(self._command_dispatcher)

            # Pull provider + model + key from the same config UI everything
            # else uses.  Local servers (LM Studio / Ollama / vLLM) all speak
            # the OpenAI wire protocol so they share the "openai"/"local" code
            # path.
            cfg = self.config.llm_control
            provider = (cfg.provider or "openai").lower()
            model = cfg.selected_model or _DEFAULT_MODEL_FOR_PROVIDER.get(provider, "gpt-4o-mini")
            api_key = self._resolve_api_key(provider, cfg)
            base_url = _normalize_base_url(getattr(cfg, "local_api_url", None) or None)

            self._agent_runner = PydanticAgentRunner.from_legacy_registry(
                registry=tool_registry,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            logger.info(
                "PydanticAgentRunner initialized (provider=%s, model=%s, %d tools)",
                provider, model, len(self._agent_runner.tool_names),
            )

        except Exception as e:
            logger.error(f"Failed to initialize PydanticAgentRunner: {e}", exc_info=True)
            self._agent_runner = None

    def _resolve_api_key(self, provider: str, cfg) -> Optional[str]:
        """Decrypt and return the API key for the active provider."""
        try:
            key_attr = f"{provider}_api_key"
            encrypted = getattr(cfg, key_attr, None)
            if not encrypted:
                return None
            return self._secure_storage.decrypt(encrypted) if encrypted else None
        except Exception as exc:
            logger.warning("Could not decrypt API key for %s: %s", provider, exc)
            return None
    
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
            elif command in ("llm_confirm", "llm_cancel"):
                # Legacy "preview-then-confirm" flow is gone; PydanticAgentRunner
                # invokes tools directly. Voice "confirm"/"cancel" become no-ops.
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
        if self.main_window and getattr(self.main_window, '_assistant_use_side_panel', False):
            return None

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
        """All LLM requests now go through PydanticAgentRunner -- the legacy
        chat_async / JSON-action / preview-and-confirm path was deleted with
        the orchestrator. If the runner isn't initialised we surface a clear
        error instead of silently doing nothing."""
        logger.info(f"LLM Request received: {text}")

        if not self._agent_runner:
            # Lazy retry: maybe the provider just changed and the runner
            # hasn't been built yet.
            self._initialize_agentic_system()

        if not self._agent_runner:
            self.error_occurred.emit("LLM not configured. Open LLM Settings.")
            if self._voice_controller:
                self._voice_controller.stop_llm_mode()
            return

        self.process_agentic_request(text)
    
    def process_agentic_request(self, text: str) -> None:
        """Process request via PydanticAgentRunner (background thread)."""
        logger.info(f"Processing agentic request: {text[:100]}...")

        if not self._agent_runner:
            logger.warning("Agentic request received without initialized runner; falling back to legacy LLM path")
            self._process_llm_request(text)
            return

        # Store for overlay updates
        self._current_request_text = text

        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_thinking(user_text=text)
        if self._voice_controller:
            self._voice_controller.pause()

        # Pull graph context on the main thread so we hand the runner a
        # snapshot the LLM can reason about ("here's what already exists").
        graph_state = self._get_current_graph_context()
        prompt = self._compose_agent_prompt(text, graph_state)

        import threading

        def run_agentic():
            try:
                self.agentic_progress.emit("Calling tools...")
                result = self._agent_runner.run_sync(prompt)

                # Translate PydanticAgentResult -> the dict shape
                # _handle_agentic_result already understands.  No plan/steps
                # because pydantic-ai handles tool dispatch internally; we
                # surface the LLM's final text + the list of tools it called.
                ok = result.success
                payload = {
                    "success": ok,
                    "message": result.output if ok else (result.error or "Failed."),
                    "plan": None,
                    "results": result.tool_calls,
                    "telemetry_summary": {
                        "tool_call_count": len(result.tool_calls),
                    },
                    "telemetry": None,
                }
                self.agentic_result_received.emit(payload, text)

            except Exception as e:
                logger.error(f"Agentic processing failed: {e}", exc_info=True)
                self.agentic_error_received.emit(str(e))

        thread = threading.Thread(target=run_agentic, daemon=True)
        thread.start()

    def _compose_agent_prompt(self, user_text: str, graph_state: str) -> str:
        """Glue user request + current graph snapshot into one prompt string.

        PydanticAI's Agent.run_sync takes a single string; we let the system
        prompt do the role/tone work and put per-turn context (graph state)
        in the user message so it stays cache-friendly across turns of a
        conversation.
        """
        if not graph_state:
            return user_text
        return (
            f"Current graph context (read-only snapshot):\n{graph_state}\n\n"
            f"User request:\n{user_text}"
        )
    
    @Slot(object, str)
    def _handle_agentic_result(self, result: dict, original_text: str) -> None:
        """Handle agentic result on main thread."""
        overlay = self._get_or_create_llm_overlay()
        
        # Handle result
        message = result.get("message", "Completed.")
        success = result.get("success", False)
        plan = result.get("plan")
        results = result.get("results", [])
        telemetry_summary = result.get("telemetry_summary")
        telemetry = result.get("telemetry")

        if telemetry:
            logger.info(f"Agent telemetry snapshot: {telemetry}")
        if telemetry_summary:
            logger.info(f"Agent telemetry summary: {telemetry_summary}")
        
        # Workflow auto-save was tied to the legacy plan/step model and is
        # disabled during the PydanticAI migration; the runner exposes
        # tool_calls in `results` for future re-implementation.

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

    # Legacy step-complete / plan-created / workflow playback hooks were
    # removed with the PydanticAI migration -- pydantic_ai.Agent owns its own
    # tool dispatch loop and reports per-tool messages via streaming/trace
    # APIs that the manager will subscribe to in a follow-up.


    # Legacy LLM chat_async / JSON-action / preview-and-confirm path is
    # gone. The remaining signal handlers (_on_llm_response,
    # _on_llm_error, _process_llm_response) were the bridge between the
    # legacy provider client and the interpreter; with PydanticAgentRunner
    # owning the loop they had nothing to do, so they were deleted along
    # with execute_actions_safe / confirm_llm_actions / cancel_llm_actions.

    @staticmethod
    def _sanitize_tts_text(text: str) -> str:
        """Strip symbols and UI-only lines that sound bad in speech."""
        if not text:
            return ""

        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Session telemetry:"):
                continue
            lines.append(line)

        cleaned = " ".join(lines)
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
        
    def _do_speak(self, text: str, duration_ms: int) -> None:
        """Actual speech execution with exact callback sync."""
        try:
            text = self._sanitize_tts_text(text)
            if not text:
                self._on_speech_finished()
                return

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
        """Handle LLM error in main thread (now produced by PydanticAgentRunner
        via agentic_error_received). Kept for back-compat with anything still
        connected to llm_error_occurred."""
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.show_error(str(error))
        if self._voice_controller:
            self._voice_controller.resume()
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
            
    # confirm_llm_actions / cancel_llm_actions were tied to the legacy
    # preview-and-confirm flow.  PydanticAgentRunner runs tools as the LLM
    # decides, so there is nothing to confirm or cancel between turns --
    # individual destructive tools should do their own confirmation gate.

    def exit_llm_mode(self) -> None:
        """Exit LLM voice-first mode."""
        logger.info("Exiting LLM mode")
        overlay = self._get_or_create_llm_overlay()
        if overlay:
            overlay.hide_overlay()
        if self._voice_controller:
            self._voice_controller.stop_llm_mode()
