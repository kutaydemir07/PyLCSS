# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Assistant Manager - orchestrator for the text AI assistant.

Owns the PydanticAI agent (``PydanticAgentRunner``) that drives PyLCSS by
calling its tools.  Requests come in as natural-language text from the
assistant side panel; the agent runs on a background thread and reports back
through the ``agentic_*`` signals.  (The legacy voice / speech-to-text /
text-to-speech stack was removed.)
"""

import logging
import json
from typing import Optional, Dict, Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Qt, Slot

from pylcss.assistant_systems.config import AssistantConfig
from pylcss.assistant_systems.services.input import MouseController
from pylcss.assistant_systems.api.dispatcher import CommandDispatcher
from pylcss.assistant_systems.services.llm import (
    LLMProvider, get_provider,
)
from pylcss.assistant_systems.services.memory import LLMMemory, get_secure_storage

# Agentic AI components -- PydanticAI native function-calling owns tool dispatch.
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
    logging.getLogger(__name__).warning("Agentic AI components not available: %s", e)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


# Sensible default models per provider for when the user picked a provider but
# never selected a specific model.
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
    Orchestrator for the text AI assistant.

    Signals:
        status_changed: Emitted when assistant status changes (str)
        error_occurred: Emitted when an error occurs (str)
        agentic_result_received: (result_dict, original_text) on success
        agentic_error_received: error_message (str)
        agentic_progress: progress_message (str)
    """

    status_changed = Signal(str)
    error_occurred = Signal(str)

    # Agentic system signals for thread-safe UI updates
    agentic_result_received = Signal(dict, str)   # (result_dict, original_text)
    agentic_error_received = Signal(str)           # error_message
    agentic_progress = Signal(str)                 # progress_message

    def __init__(self, main_window: Optional["QMainWindow"] = None):
        """
        Initialize the assistant manager.

        Args:
            main_window: Reference to the PyLCSS main window
        """
        super().__init__()

        self.main_window = main_window
        self.config = AssistantConfig.load()

        # Agentic signals -> main thread handlers
        self.agentic_result_received.connect(self._handle_agentic_result, Qt.QueuedConnection)
        self.agentic_error_received.connect(self._handle_agentic_error, Qt.QueuedConnection)
        self.agentic_progress.connect(self._handle_agentic_progress, Qt.QueuedConnection)

        # Components
        self._mouse_controller: Optional[MouseController] = None
        self._command_dispatcher: Optional[CommandDispatcher] = None

        # LLM components -- LLMProvider is the legacy text-only client kept for
        # the settings dialog (provider.list_models()); PydanticAgentRunner
        # owns the real conversation loop.
        self._llm_provider: Optional[LLMProvider] = None
        self._llm_memory: Optional[LLMMemory] = None
        self._secure_storage = get_secure_storage()

        # Agentic AI components -- single PydanticAgentRunner.
        self._agent_runner: Optional['PydanticAgentRunner'] = None
        self._use_agentic_mode: bool = self.config.llm_control.agentic_mode and AGENTIC_AVAILABLE

        # State
        self._initialized = False
        self._current_request_text = ""

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

        # Initialize mouse controller (used by the command dispatcher).
        try:
            self._mouse_controller = MouseController()
            success = True
        except Exception as e:
            logger.error(f"Failed to initialize mouse controller: {e}")
            self.error_occurred.emit(f"Mouse control unavailable: {e}")

        # Initialize command dispatcher (the action layer the agent's tools use).
        self._command_dispatcher = CommandDispatcher(
            main_window=self.main_window,
            mouse_controller=self._mouse_controller,
        )

        # Initialize LLM components.
        try:
            if self.config.llm_control.memory_enabled:
                self._llm_memory = LLMMemory(max_conversations=100)
                logger.info(
                    "LLM memory initialized with %d conversations",
                    self._llm_memory.get_conversation_count(),
                )

            # Provider object is kept for the settings dialog model lists; it is
            # not consulted for chat any more (PydanticAgentRunner owns that).
            self._initialize_llm_provider()
            logger.info("LLM components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {e}")

        # Initialize agentic AI components.
        if self._use_agentic_mode and self._command_dispatcher:
            self._initialize_agentic_system()

        self._initialized = success or self._agent_runner is not None
        return self._initialized

    def get_config(self) -> AssistantConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, config: AssistantConfig) -> None:
        """Update configuration (caller should save if needed)."""
        logger.info("Updating AssistantManager configuration...")
        self.config = config
        self._use_agentic_mode = config.llm_control.agentic_mode and AGENTIC_AVAILABLE

        # Re-initialize LLM provider + agent runner so new keys/models apply.
        self._llm_provider = None
        self._initialize_llm_provider()
        if self._use_agentic_mode and self._command_dispatcher:
            self._initialize_agentic_system()

    def _initialize_llm_provider(self) -> None:
        """Initialize or re-initialize the LLM provider based on config."""
        llm_config = self.config.llm_control
        provider_name = llm_config.provider
        encrypted_key = llm_config.get_api_key_for_provider(provider_name)

        logger.info(
            "Initializing LLM provider: %s, Key length: %d",
            provider_name, len(encrypted_key) if encrypted_key else 0,
        )

        if not encrypted_key:
            logger.warning(f"No API key set for provider {provider_name}")
            self._llm_provider = None
            return

        # Decrypt the key (fall back to treating it as plaintext for legacy).
        try:
            api_key = self._secure_storage.decrypt(encrypted_key) or encrypted_key
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
            logger.info(f"LLM provider initialized: {provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {provider_name}: {e}")
            self._llm_provider = None

    def set_llm_provider(self, provider: Optional[LLMProvider]) -> None:
        """Set the LLM provider directly (from UI)."""
        self._llm_provider = provider
        logger.info(f"LLM provider set: {provider.name if provider else 'None'}")
        if self._use_agentic_mode and provider:
            self._initialize_agentic_system()

    def _initialize_agentic_system(self) -> None:
        """Build the PydanticAgentRunner from the current LLM config + tool registry."""
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
            # the OpenAI wire protocol so they share the "openai"/"local" path.
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
            encrypted = getattr(cfg, f"{provider}_api_key", None)
            if not encrypted:
                return None
            return self._secure_storage.decrypt(encrypted)
        except Exception as exc:
            logger.warning("Could not decrypt API key for %s: %s", provider, exc)
            return None

    # --- Graph context ---

    def _get_current_graph_context(self) -> str:
        """Serialize the current graph to compact JSON for the agent prompt."""
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
            if tab_index == 0:  # Modeling
                if hasattr(self.main_window, 'modeling_widget'):
                    graph = self.main_window.modeling_widget.current_graph
                context_type = "Modeling"
            elif tab_index == 1:  # CAD
                if hasattr(self.main_window, 'cad_widget'):
                    graph = getattr(self.main_window.cad_widget, 'graph', None)
                context_type = "CAD"

            if not graph:
                return ""

            # Serialize nodes (functional properties only, to save tokens).
            ignore_starts = ['_', 'error', 'port', 'selected', 'disabled']
            ignore_exact = ['pos', 'color', 'border_color', 'text_color', 'icon', 'width', 'height']
            nodes_data = []
            for node in graph.all_nodes():
                node_data = {"id": node.name(), "type": node.type_, "properties": {}}
                for name, val in node.properties().items():
                    name_lower = name.lower()
                    if any(name_lower.startswith(p) for p in ignore_starts):
                        continue
                    if name_lower in ignore_exact:
                        continue
                    node_data["properties"][name] = val
                nodes_data.append(node_data)

            # Serialize connections.
            connections_data = []
            for node in graph.all_nodes():
                for out_port in node.output_ports():
                    for cp in out_port.connected_ports():
                        connections_data.append({
                            "from": f"{node.name()}.{out_port.name()}",
                            "to": f"{cp.node().name()}.{cp.name()}",
                        })

            context = {
                "environment": context_type,
                "nodes": nodes_data,
                "connections": connections_data,
            }
            return json.dumps(context, separators=(',', ':'))
        except Exception as e:
            logger.error(f"Context error: {e}")
            return ""

    # --- Request handling ---

    def process_agentic_request(self, text: str) -> None:
        """Process a natural-language request via PydanticAgentRunner.

        Entry point from the assistant side panel.  Runs the agent on a
        background thread and reports back through the ``agentic_*`` signals.
        """
        logger.info(f"Assistant request: {text[:100]}...")

        if not self._agent_runner:
            # Lazy retry: maybe the provider just changed and the runner
            # hasn't been built yet.
            self._initialize_agentic_system()

        if not self._agent_runner:
            self.error_occurred.emit("LLM not configured. Open LLM Settings.")
            return

        self._current_request_text = text
        self.agentic_progress.emit("Calling tools...")

        # Pull graph context on the main thread so we hand the runner a
        # snapshot the LLM can reason about ("here's what already exists").
        graph_state = self._get_current_graph_context()
        prompt = self._compose_agent_prompt(text, graph_state)

        import threading

        def run_agentic():
            try:
                result = self._agent_runner.run_sync(prompt)
                ok = result.success
                payload = {
                    "success": ok,
                    "message": result.output if ok else (result.error or "Failed."),
                    "results": result.tool_calls,
                    "telemetry_summary": {"tool_call_count": len(result.tool_calls)},
                }
                self.agentic_result_received.emit(payload, text)
            except Exception as e:
                logger.error(f"Agentic processing failed: {e}", exc_info=True)
                self.agentic_error_received.emit(str(e))

        threading.Thread(target=run_agentic, daemon=True).start()

    def _compose_agent_prompt(self, user_text: str, graph_state: str) -> str:
        """Glue the user request + current graph snapshot into one prompt string."""
        if not graph_state:
            return user_text
        return (
            f"Current graph context (read-only snapshot):\n{graph_state}\n\n"
            f"User request:\n{user_text}"
        )

    @Slot(object, str)
    def _handle_agentic_result(self, result: dict, original_text: str) -> None:
        """Handle agentic result on the main thread."""
        message = result.get("message", "Completed.")
        telemetry_summary = result.get("telemetry_summary")
        if telemetry_summary:
            logger.info(f"Agent telemetry summary: {telemetry_summary}")

        if self._llm_memory:
            self._llm_memory.add_message(
                role="assistant",
                content=message,
                provider=self.config.llm_control.provider,
                model=self.config.llm_control.selected_model or "agentic",
            )

    @Slot(str)
    def _handle_agentic_error(self, error_msg: str) -> None:
        """Handle agentic error on the main thread."""
        logger.error(f"Agentic error: {error_msg}")

    @Slot(str)
    def _handle_agentic_progress(self, message: str) -> None:
        """Handle progress updates on the main thread."""
        logger.debug(f"Agentic progress: {message}")
