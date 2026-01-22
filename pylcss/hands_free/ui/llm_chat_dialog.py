# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Chat Dialog for Hands-Free Control.

Provides a modal dialog for interacting with the LLM assistant,
similar to the existing hands-free voice control popup.
"""

import logging
from typing import Optional, List, TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QScrollArea, QWidget, QFrame,
    QComboBox, QGroupBox, QSizePolicy, QSplitter, QToolButton
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon

from pylcss.hands_free.llm_providers import (
    LLMProvider, get_provider, get_available_providers,
    PROVIDER_DISPLAY_NAMES, ChatCompletion, LLMProviderError, ModelInfo
)
from pylcss.hands_free.llm_memory import LLMMemory, get_secure_storage
from pylcss.hands_free.llm_interpreter import LLMInterpreter, ParsedResponse, ActionCommand
from pylcss.hands_free.config import HandsFreeConfig

if TYPE_CHECKING:
    from pylcss.hands_free.command_dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


class ThinkingBubble(QFrame):
    """A collapsible bubble for agent thinking process."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header (Always visible)
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(10, 8, 10, 8)
        
        self.toggle_btn = QToolButton()
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.setAutoRaise(True)
        self.toggle_btn.clicked.connect(self._toggle_expanded)
        self.toggle_btn.setStyleSheet("QToolButton { color: #aaaaaa; border: none; }")
        self.header_layout.addWidget(self.toggle_btn)
        
        self.title_label = QLabel("Thinking Process...")
        self.title_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        
        layout.addWidget(self.header)
        
        # Content (Collapsible)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(25, 0, 10, 10)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #888888;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log_text.setMaximumHeight(200)
        self.content_layout.addWidget(self.log_text)
        
        self.content_area.hide()
        layout.addWidget(self.content_area)
        
        # Styling
        self.setStyleSheet("""
            ThinkingBubble {
                background-color: #252525;
                border: 1px solid #333333;
                border-radius: 6px;
                margin-left: 20px;
                margin-right: 20px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
        """)
        
    def _toggle_expanded(self):
        visible = not self.content_area.isVisible()
        self.content_area.setVisible(visible)
        self.toggle_btn.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)
        
    def add_log(self, text: str):
        self.log_text.append(f"• {text}")
        # Auto-expand on first log or error
        if "error" in text.lower() or "fail" in text.lower():
            if not self.content_area.isVisible():
                self._toggle_expanded()


class MessageBubble(QFrame):
    """A chat message bubble widget."""
    
    def __init__(self, text: str, is_user: bool, is_thought: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.is_user = is_user
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Role label
        role_text = "You" if is_user else "🤖 PyLCSS Assistant"
        role_label = QLabel(role_text)
        role_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #aaa;")
        layout.addWidget(role_label)
        
        # Message text
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.text_label)
        
        # Styling
        if is_user:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #2d5a7d;
                    border-radius: 10px;
                    margin-left: 50px;
                    margin-right: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #3a3a3a;
                    border-radius: 10px;
                    margin-left: 5px;
                    margin-right: 50px;
                }
            """)


class ActionPreviewWidget(QFrame):
    """Widget showing pending actions for confirmation."""
    
    confirm_clicked = Signal()
    cancel_clicked = Signal()
    
    def __init__(self, actions: List[ActionCommand], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.actions = actions
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ActionPreviewWidget {
                background-color: #2d4a2d;
                border: 1px solid #4a7a4a;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Title
        title = QLabel("📋 Pending Actions")
        title.setStyleSheet("font-weight: bold; color: #8fdf8f;")
        layout.addWidget(title)
        
        # Actions list
        for i, action in enumerate(actions, 1):
            desc = action.description or f"{action.action_type}: {action.command}"
            action_label = QLabel(f"  {i}. {desc}")
            action_label.setStyleSheet("color: #c0e0c0;")
            layout.addWidget(action_label)
            
        # Buttons
        button_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("✓ Execute")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9a4a;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5ab05a;
            }
        """)
        self.confirm_btn.clicked.connect(self.confirm_clicked.emit)
        button_layout.addWidget(self.confirm_btn)
        
        self.cancel_btn = QPushButton("✕ Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #7a4a4a;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #9a5a5a;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_clicked.emit)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)


class ThinkingIndicator(QLabel):
    """Animated thinking indicator."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("🧠 Thinking", parent)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        
        self.setStyleSheet("""
            ThinkingIndicator {
                color: #ffaa00;
                font-weight: bold;
                padding: 10px;
                background-color: #3a3a2a;
                border-radius: 8px;
            }
        """)
        self.hide()
        
    def start(self):
        """Start the animation."""
        self._dots = 0
        self._timer.start(500)
        self.show()
        
    def stop(self):
        """Stop the animation."""
        self._timer.stop()
        self.hide()
        
    def _animate(self):
        """Update animation."""
        self._dots = (self._dots + 1) % 4
        dots = "." * self._dots
        self.setText(f"🧠 Thinking{dots}")


class LLMChatDialog(QDialog):
    """
    Modal dialog for LLM-assisted hands-free control.
    
    Features:
    - Chat interface with message history
    - Thinking indicator while waiting for LLM
    - Action preview with confirmation
    - Multi-provider support (OpenAI, Claude, Gemini)
    - Model selection
    """
    
    # Signal when actions should be executed
    execute_actions = Signal(list)  # List[ActionCommand]
    
    # Internal signals for thread safety
    response_received = Signal(object)  # ChatCompletion
    error_occurred = Signal(Exception)
    connection_success = Signal(object)  # (LLMProvider, List[ModelInfo])
    connection_failed = Signal(str)  # Error message
    
    def __init__(
        self,
        command_dispatcher: Optional["CommandDispatcher"] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self.command_dispatcher = command_dispatcher
        self.interpreter = LLMInterpreter()
        
        # Load config
        self._config = HandsFreeConfig.load()
        self._secure_storage = get_secure_storage()
        
        # Multi-provider support
        self._provider: Optional[LLMProvider] = None
        self._memory: Optional[LLMMemory] = None
        
        # Initialize memory if enabled
        if self._config.llm_control.memory_enabled:
            self._memory = LLMMemory()
        
        # Pending actions waiting for confirmation
        self._pending_actions: List[ActionCommand] = []
        self._action_preview: Optional[ActionPreviewWidget] = None
        self._current_thinking_bubble: Optional[ThinkingBubble] = None
        
        self._setup_ui()
        self._connect_signals()
        
        # Load provider from config
        self._load_provider_from_config()
        
        
    def _setup_ui(self):
        """Create the dialog UI."""
        self.setWindowTitle("🤖 PyLCSS LLM Assistant")
        self.setMinimumSize(500, 600)
        self.resize(600, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QLineEdit, QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #5a9a5a;
            }
            QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header with settings
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()
        
        self.chat_scroll.setWidget(self.chat_container)
        main_layout.addWidget(self.chat_scroll, 1)
        
        # Thinking indicator
        self.thinking_indicator = ThinkingIndicator()
        main_layout.addWidget(self.thinking_indicator)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask me to create shapes, build graphs, or control PyLCSS...")
        self.input_field.setMinimumHeight(40)
        input_layout.addWidget(self.input_field, 1)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a7a4a;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a9a5a;
            }
        """)
        input_layout.addWidget(self.send_btn)
        
        main_layout.addLayout(input_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #808080; font-size: 11px;")
        main_layout.addWidget(self.status_label)
        
    def _create_header(self) -> QWidget:
        """Create the settings header with multi-provider support."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Provider selector
        layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(130)
        for provider_name in get_available_providers():
            display_name = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
            self.provider_combo.addItem(display_name, provider_name)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        layout.addWidget(self.provider_combo)
        
        # Model selector
        layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.addItem("Connect first...", "")
        layout.addWidget(self.model_combo)
        
        layout.addStretch()
        
        # API Key input
        layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API key...")
        self.api_key_input.setMaximumWidth(150)
        layout.addWidget(self.api_key_input)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear Chat")
        clear_btn.clicked.connect(self._clear_chat)
        layout.addWidget(clear_btn)
        
        # Set initial provider from config
        provider = self._config.llm_control.provider
        for i in range(self.provider_combo.count()):
            if self.provider_combo.itemData(i) == provider:
                self.provider_combo.setCurrentIndex(i)
                break
        
        return header
    
    def _load_provider_from_config(self) -> None:
        """Load provider from config and initialize if API key exists."""
        llm_config = self._config.llm_control
        provider_name = llm_config.provider
        encrypted_key = llm_config.get_api_key_for_provider(provider_name)
        
        if encrypted_key:
            # Decrypt and set in input
            try:
                api_key = self._secure_storage.decrypt(encrypted_key)
                if api_key:
                    self.api_key_input.setText(api_key)
                else:
                    self.api_key_input.setText(encrypted_key)
            except Exception:
                self.api_key_input.setText(encrypted_key)
    
    @Slot()
    def _on_provider_changed(self) -> None:
        """Handle provider selection change."""
        # Clear models
        self.model_combo.clear()
        self.model_combo.addItem("Connect to load models...", "")
        self._provider = None
        self.connect_btn.setText("Connect")
        self.connect_btn.setStyleSheet("")
        
        # Load API key for this provider
        provider_name = self.provider_combo.currentData()
        encrypted_key = self._config.llm_control.get_api_key_for_provider(provider_name)
        
        if encrypted_key:
            try:
                api_key = self._secure_storage.decrypt(encrypted_key)
                self.api_key_input.setText(api_key if api_key else encrypted_key)
            except Exception:
                self.api_key_input.setText(encrypted_key)
        else:
            self.api_key_input.clear()
        
    def _connect_signals(self):
        """Connect UI signals."""
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.input_field.returnPressed.connect(self._on_send_clicked)
        self.execute_actions.connect(self._execute_actions)
        
        # Check thread safety connections
        self.response_received.connect(self._process_response)
        self.error_occurred.connect(self._handle_error)
        self.connection_success.connect(self._on_connection_success)
        self.connection_failed.connect(self._on_connection_failed)
        
        
    def set_token(self, token: str):
        """Set the API token (legacy compatibility)."""
        self.api_key_input.setText(token)
        
    def get_token(self) -> str:
        """Get the current token (legacy compatibility)."""
        return self.api_key_input.text()
        
    @Slot()
    def _on_connect_clicked(self):
        """Handle connect button click - initialize multi-provider."""
        logger.info("LLM Dialog: Connect clicked")
        provider_name = self.provider_combo.currentData()
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            self._set_status("Please enter your API key", error=True)
            return
        
        # Disable UI during connection
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")
        self._set_status("Connecting...")
        logger.info(f"LLM Dialog: Testing connection to {provider_name}...")
        
        # Run connection in background thread
        import threading
        thread = threading.Thread(
            target=self._connect_worker,
            args=(provider_name, api_key),
            daemon=True
        )
        thread.start()
    
    def _connect_worker(self, provider_name: str, api_key: str):
        """Background worker for connection (runs in separate thread)."""
        try:
            kwargs = {}            
            provider = get_provider(provider_name, api_key, **kwargs)
            
            # Get models (this is the slow network call)
            models = provider.get_models()
            logger.info(f"LLM Dialog: Received {len(models)} models")
            
            # Emit success signal back to main thread
            self.connection_success.emit((provider, models, provider_name, api_key))
            
        except LLMProviderError as e:
            logger.error(f"LLM Dialog: Connection failed: {e}")
            self.connection_failed.emit(str(e))
        except Exception as e:
            logger.error(f"LLM Dialog: Connection failed: {e}")
            self.connection_failed.emit(f"Error: {str(e)[:50]}")
    
    @Slot(object)
    def _on_connection_success(self, result: tuple):
        """Handle successful connection (called in main thread)."""
        provider, models, provider_name, api_key = result
        
        self._provider = provider
        
        self.model_combo.clear()
        for model in models:
            display_name = model.name
            self.model_combo.addItem(display_name, model.id)
        
        # Select first model or previously selected
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
            
        # Set system prompt
        self._provider.set_system_prompt(self.interpreter.get_system_prompt())
        
        # Save encrypted API key to config
        encrypted_key = self._secure_storage.encrypt(api_key)
        self._config.llm_control.set_api_key_for_provider(provider_name, encrypted_key)
        self._config.llm_control.provider = provider_name
        self._config.save()
                
        self._set_status(f"Connected! {len(models)} models available")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("✓ Connected")
        self.connect_btn.setStyleSheet("background-color: #4a7a4a;")
        logger.info("LLM Dialog: Connection successful")
    
    @Slot(str)
    def _on_connection_failed(self, error_message: str):
        """Handle failed connection (called in main thread)."""
        self._set_status(error_message, error=True)
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self.connect_btn.setStyleSheet("")
            
    @Slot()
    def _on_send_clicked(self):
        """Handle send button click."""
        message = self.input_field.text().strip()
        if not message:
            return
        
        if not self._provider:
            self._set_status("Please connect first", error=True)
            return
        
        logger.info(f"LLM Dialog: Sending message: {message[:50]}...")
            
        # Add user message to memory
        if self._memory:
            self._memory.add_message("user", message)
            
        # Add user message to UI
        self._add_message(message, is_user=True)
        self.input_field.clear()
        
        # Reset thinking bubble
        self._current_thinking_bubble = None
        
        # Start thinking
        self.thinking_indicator.start()
        self.send_btn.setEnabled(False)
        self._set_status("Waiting for response...")
        
        # Get selected model
        model = self.model_combo.currentData() or ""
        logger.info(f"LLM Dialog: Using model: {model}")
        
        # Send async request using provider
        self._provider.chat_async(
            user_message=message,
            on_complete=self._on_llm_response,
            on_error=self._on_llm_error,
            model=model,
        )
        
    def _on_llm_response(self, completion: ChatCompletion):
        """Handle LLM response (called from background thread)."""
        logger.info(f"LLM Dialog: Received response ({len(completion.content)} chars)")
        # Emit signal to handle in main thread
        self.response_received.emit(completion)
        
    def _on_llm_error(self, error: Exception):
        """Handle LLM error (called from background thread)."""
        logger.error(f"LLM Dialog: Error received: {error}")
        # Emit signal to handle in main thread
        self.error_occurred.emit(error)
        
    @Slot(object)
    def _process_response(self, completion: ChatCompletion):
        """Process LLM response in main thread."""
        logger.info("LLM Dialog: Processing response in main thread")
        self.thinking_indicator.stop()
        self.send_btn.setEnabled(True)
        
        # Save assistant response to memory
        if self._memory:
            self._memory.add_message("assistant", completion.content)
        
        # Parse response
        parsed = self.interpreter.parse_response(completion.content)
        logger.info(f"LLM Dialog: Parsed {len(parsed.actions)} actions")
        
        # Show thinking if available
        if parsed.thinking:
            self._set_status(f"💭 {parsed.thinking}")
            
        # Add assistant message
        self._add_message(parsed.message or completion.content, is_user=False)
        
        # Show action preview if actions present
        if parsed.actions:
            self._show_action_preview(parsed.actions)
        else:
            self._set_status("Ready")
            
    @Slot(Exception)
    def _handle_error(self, error: Exception):
        """Handle error in main thread."""
        logger.error(f"LLM Dialog: Handling error in main thread: {error}")
        self.thinking_indicator.stop()
        self.send_btn.setEnabled(True)
        self._set_status(str(error), error=True)
        self._add_message(f"❌ Error: {error}", is_user=False)
        
    def _add_message(self, text: str, is_user: bool):
        """Add a message bubble to the chat."""
        bubble = MessageBubble(text, is_user)
        
        # Insert before stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))
        
    def add_agent_thought(self, text: str):
        """Add an agent thought to the current thinking bubble."""
        if not self._current_thinking_bubble:
            self._current_thinking_bubble = ThinkingBubble()
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, self._current_thinking_bubble)
            
        self._current_thinking_bubble.add_log(text)
        
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))
        
    def _show_action_preview(self, actions: List[ActionCommand]):
        """Show action preview for confirmation."""
        # Remove existing preview
        if self._action_preview:
            self._action_preview.deleteLater()
            
        self._pending_actions = actions
        self._action_preview = ActionPreviewWidget(actions)
        self._action_preview.confirm_clicked.connect(self._on_confirm_actions)
        self._action_preview.cancel_clicked.connect(self._on_cancel_actions)
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self._action_preview)
        self._set_status(f"{len(actions)} actions pending confirmation")
        
    @Slot()
    def _on_confirm_actions(self):
        """Execute pending actions."""
        if self._pending_actions:
            self.execute_actions.emit(self._pending_actions)
            self._add_message(f"✓ Executed {len(self._pending_actions)} actions", is_user=False)
            
        self._clear_action_preview()
        self._set_status("Actions executed")
        
    @Slot()
    def _on_cancel_actions(self):
        """Cancel pending actions."""
        self._add_message("❌ Actions cancelled", is_user=False)
        self._clear_action_preview()
        self._set_status("Actions cancelled")
        
    def _clear_action_preview(self):
        """Remove action preview widget."""
        if self._action_preview:
            self._action_preview.deleteLater()
            self._action_preview = None
        self._pending_actions = []
        
    @Slot(list)
    def _execute_actions(self, actions: List[ActionCommand]):
        """Execute a list of actions via CommandDispatcher."""
        if not self.command_dispatcher:
            logger.warning("No command dispatcher available")
            return
            
        for action in actions:
            result = self.interpreter.action_to_dispatcher_format(action)
            if result:
                command_name, command_data = result
                try:
                    self.command_dispatcher.dispatch(command_name, command_data)
                    logger.info(f"Executed action: {action.command}")
                except Exception as e:
                    logger.error(f"Failed to execute action {action.command}: {e}")
                    
    @Slot()
    def _clear_chat(self):
        """Clear chat history."""
        # Remove all message bubbles
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), (MessageBubble, ActionPreviewWidget)):
                item.widget().deleteLater()
                
        self._clear_action_preview()
        
        # Clear provider history
        if self._provider:
            self._provider.clear_history()
            self._provider.set_system_prompt(self.interpreter.get_system_prompt())
        
        # Start new conversation in memory
        if self._memory:
            provider = self.provider_combo.currentData() if hasattr(self, 'provider_combo') else ""
            model = self.model_combo.currentData() if hasattr(self, 'model_combo') else ""
            self._memory.new_conversation(provider=provider, model=model)
            
        self._set_status("Chat cleared")
        
    def _set_status(self, text: str, error: bool = False):
        """Update status label."""
        self.status_label.setText(text)
        if error:
            self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        else:
            self.status_label.setStyleSheet("color: #808080; font-size: 11px;")

