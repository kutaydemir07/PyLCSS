# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Settings Widget for Voice Control Popup.

Provides a collapsible settings panel for configuring LLM providers,
API keys (encrypted), model selection, and generation parameters.
"""

import logging
from typing import Optional, List, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QLineEdit, QSlider, QSpinBox, QCheckBox,
    QFrame, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QObject

from pylcss.hands_free.config import LLMControlConfig
from pylcss.hands_free.llm_providers import (
    LLMProvider, get_provider, get_available_providers,
    PROVIDER_DISPLAY_NAMES, LLMProviderError, ModelInfo
)
from pylcss.hands_free.llm_memory import get_secure_storage

logger = logging.getLogger(__name__)


class ConnectionWorker(QThread):
    """Async worker to test connection and fetch models."""
    success = Signal(object)  # List[ModelInfo]
    error = Signal(str)
    
    def __init__(self, provider: LLMProvider):
        super().__init__()
        self.provider = provider
        
    def run(self):
        try:
            models = self.provider.get_models()
            self.success.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class LLMSettingsWidget(QWidget):
    """
    LLM Settings panel for the voice control popup.
    
    Features:
    - Provider dropdown (OpenAI, Claude, Gemini, GPT@RUB)
    - API key input with encryption
    - Model dropdown (dynamically populated)
    - Temperature and max tokens controls
    - Connection test button
    """
    
    settings_changed = Signal()
    provider_connected = Signal(str)  # Emits provider name when connected
    
    def __init__(
        self,
        config: LLMControlConfig,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.config = config
        self._secure_storage = get_secure_storage()
        self._current_provider: Optional[LLMProvider] = None
        self._models_cache: Dict[str, List[ModelInfo]] = {}
        self._worker: Optional[ConnectionWorker] = None
        
        # Debounce timer for config saves
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._emit_settings_changed)
        
        self._setup_ui()
        self._load_current_config()
    
    def _setup_ui(self) -> None:
        """Create the settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(150)
        for provider_name in get_available_providers():
            display_name = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
            self.provider_combo.addItem(display_name, provider_name)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self.provider_combo, 1)
        
        layout.addLayout(provider_layout)
        
        # API Key input
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter your API key...")
        self.api_key_input.textChanged.connect(self._on_api_key_changed)
        api_key_layout.addWidget(self.api_key_input, 1)
        
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setMaximumWidth(30)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visibility)
        self.show_key_btn.setToolTip("Show/hide API key")
        api_key_layout.addWidget(self.show_key_btn)
        
        layout.addLayout(api_key_layout)
        
        # Connection test
        test_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("🔌 Test Connection")
        self.test_btn.clicked.connect(self._test_connection)
        test_layout.addWidget(self.test_btn)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 11px;")
        test_layout.addWidget(self.status_label, 1)
        
        layout.addLayout(test_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #3a3a3a;")
        layout.addWidget(line)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.addItem("Connect to load models...", "")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo, 1)
        
        layout.addLayout(model_layout)
        
        # Temperature slider
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(200)  # 0.0 to 2.0
        self.temp_slider.setValue(int(self.config.temperature * 100))
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        temp_layout.addWidget(self.temp_slider, 1)
        
        self.temp_label = QLabel(f"{self.config.temperature:.1f}")
        self.temp_label.setMinimumWidth(30)
        temp_layout.addWidget(self.temp_label)
        
        layout.addLayout(temp_layout)
        
        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setMinimum(100)
        self.tokens_spin.setMaximum(8000)
        self.tokens_spin.setSingleStep(100)
        self.tokens_spin.setValue(self.config.max_tokens)
        self.tokens_spin.valueChanged.connect(self._on_tokens_changed)
        tokens_layout.addWidget(self.tokens_spin, 1)
        
        layout.addLayout(tokens_layout)
        
        # Memory toggle
        self.memory_cb = QCheckBox("Enable conversation memory")
        self.memory_cb.setChecked(self.config.memory_enabled)
        self.memory_cb.setToolTip("LLM remembers previous conversations for context")
        self.memory_cb.toggled.connect(self._on_memory_toggled)
        layout.addWidget(self.memory_cb)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
            }
            QComboBox, QLineEdit, QSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus {
                border-color: #5a9a5a;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:checked {
                background-color: #4a7a4a;
            }
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9a5a;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QCheckBox {
                color: #e0e0e0;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
    
    def _load_current_config(self) -> None:
        """Load the current configuration into the UI."""
        # Block signals to prevent textChanged from triggering saves
        self.provider_combo.blockSignals(True)
        self.api_key_input.blockSignals(True)
        
        try:
            # Set provider
            for i in range(self.provider_combo.count()):
                if self.provider_combo.itemData(i) == self.config.provider:
                    self.provider_combo.setCurrentIndex(i)
                    break
            
            # Load API key (decrypt if encrypted)
            self._load_api_key_for_current_provider()
        finally:
            # Restore signals
            self.provider_combo.blockSignals(False)
            self.api_key_input.blockSignals(False)
    
    def _load_api_key_for_current_provider(self) -> None:
        """Load the API key for the current provider."""
        provider = self.provider_combo.currentData()
        encrypted_key = self.config.get_api_key_for_provider(provider)
        
        # Block signals to prevent textChanged from triggering re-encryption
        was_blocked = self.api_key_input.signalsBlocked()
        self.api_key_input.blockSignals(True)
        
        try:
            if encrypted_key:
                # Try to decrypt
                try:
                    decrypted = self._secure_storage.decrypt(encrypted_key)
                    if decrypted:
                        self.api_key_input.setText(decrypted)
                    else:
                        # Key might not be encrypted (legacy)
                        self.api_key_input.setText(encrypted_key)
                except Exception:
                    # Fallback to raw key (legacy unencrypted)
                    self.api_key_input.setText(encrypted_key)
            else:
                self.api_key_input.clear()
        finally:
            self.api_key_input.blockSignals(was_blocked)
    
    def _save_api_key_for_current_provider(self) -> None:
        """Save and encrypt the API key for the current provider."""
        provider = self.provider_combo.currentData()
        plaintext_key = self.api_key_input.text().strip()
        
        if plaintext_key:
            # Encrypt the key
            encrypted_key = self._secure_storage.encrypt(plaintext_key)
            self.config.set_api_key_for_provider(provider, encrypted_key)
        else:
            self.config.set_api_key_for_provider(provider, "")
        
        # Verify it was set
        stored_key = self.config.get_api_key_for_provider(provider)
    
    @Slot()
    def _on_provider_changed(self) -> None:
        """Handle provider selection change."""
        provider = self.provider_combo.currentData()
        self.config.provider = provider
        
        # Load API key for this provider
        self._load_api_key_for_current_provider()
        
        # Clear models and status
        self.model_combo.clear()
        self.model_combo.addItem("Connect to load models...", "")
        self.status_label.setText("")
        self.status_label.setStyleSheet("color: #808080;")
        
        self._schedule_save()
        
        # Auto-test if this provider has an API key
        if self.api_key_input.text().strip():
            self.status_label.setText("⏳ Loading models...")
            # Create or restart debounce timer for auto-test
            if not hasattr(self, '_auto_test_timer'):
                self._auto_test_timer = QTimer(self)
                self._auto_test_timer.setSingleShot(True)
                self._auto_test_timer.setInterval(500)  # 0.5 second delay after provider switch
                self._auto_test_timer.timeout.connect(self._test_connection)
            self._auto_test_timer.start()
    
    @Slot()
    def _on_api_key_changed(self) -> None:
        """Handle API key text change."""
        self._save_api_key_for_current_provider()
        self._schedule_save()
        
        # Auto-test connection after a debounce delay
        api_key = self.api_key_input.text().strip()
        if api_key:
            # Show status
            self.status_label.setText("⏳ Will test connection...")
            self.status_label.setStyleSheet("color: #808080;")
            
            # Create or restart debounce timer for auto-test
            if not hasattr(self, '_auto_test_timer'):
                self._auto_test_timer = QTimer(self)
                self._auto_test_timer.setSingleShot(True)
                self._auto_test_timer.setInterval(1500)  # 1.5 second debounce
                self._auto_test_timer.timeout.connect(self._test_connection)
            self._auto_test_timer.start()
        else:
            self.status_label.setText("")
    
    @Slot(bool)
    def _toggle_key_visibility(self, show: bool) -> None:
        """Toggle API key visibility."""
        if show:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("🔒")
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("👁")
    
    @Slot()
    def _test_connection(self) -> None:
        """Test the connection to the selected provider."""
        provider_name = self.provider_combo.currentData()
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            self.status_label.setText("❌ Enter API key first")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            return
        
        self.status_label.setText("⏳ Connecting...")
        self.status_label.setStyleSheet("color: #ffaa00;")
        self.test_btn.setEnabled(False)
        self.provider_combo.setEnabled(False)
        
        # Create provider
        try:
            kwargs = {}
            if provider_name == "gptrub":
                kwargs["api_url"] = self.config.gptrub_api_url
            
            self._current_provider = get_provider(provider_name, api_key, **kwargs)
            
            # Start async worker
            self._worker = ConnectionWorker(self._current_provider)
            self._worker.success.connect(self._on_connection_success)
            self._worker.error.connect(self._on_connection_error)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)[:40]}")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            self.test_btn.setEnabled(True)
            self.provider_combo.setEnabled(True)
            logger.error(f"LLM connection setup failed: {e}")

    @Slot(object)
    def _on_connection_success(self, models: List[ModelInfo]) -> None:
        """Handle successful connection."""
        provider_name = self.provider_combo.currentData()
        self._models_cache[provider_name] = models
        
        # Update model dropdown
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model.name, model.id)
        
        # Select previously selected model if available
        if self.config.selected_model:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == self.config.selected_model:
                    self.model_combo.setCurrentIndex(i)
                    break
        
        self.status_label.setText(f"✓ Connected ({len(models)} models)")
        self.status_label.setStyleSheet("color: #6bff6b;")
        self.provider_connected.emit(provider_name)
        
    @Slot(str)
    def _on_connection_error(self, error_msg: str) -> None:
        """Handle connection error."""
        self.status_label.setText(f"❌ {error_msg[:50]}")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        logger.error(f"LLM connection test failed: {error_msg}")
        
    @Slot()
    def _on_worker_finished(self) -> None:
        """Cleanup worker."""
        self.test_btn.setEnabled(True)
        self.provider_combo.setEnabled(True)
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
    
    @Slot()
    def _on_model_changed(self) -> None:
        """Handle model selection change."""
        model_id = self.model_combo.currentData()
        if model_id:
            self.config.selected_model = model_id
            self.config.model = model_id  # Legacy field
            self._schedule_save()
    
    @Slot(int)
    def _on_temp_changed(self, value: int) -> None:
        """Handle temperature slider change."""
        temp = value / 100.0
        self.config.temperature = temp
        self.temp_label.setText(f"{temp:.1f}")
        self._schedule_save()
    
    @Slot(int)
    def _on_tokens_changed(self, value: int) -> None:
        """Handle max tokens change."""
        self.config.max_tokens = value
        self._schedule_save()
    
    @Slot(bool)
    def _on_memory_toggled(self, enabled: bool) -> None:
        """Handle memory toggle."""
        self.config.memory_enabled = enabled
        self._schedule_save()
    
    def _schedule_save(self) -> None:
        """Schedule a debounced save."""
        self._save_timer.start()
    
    @Slot()
    def _emit_settings_changed(self) -> None:
        """Emit settings changed signal."""
        self.settings_changed.emit()
    
    def get_current_provider(self) -> Optional[LLMProvider]:
        """Get the currently connected provider instance."""
        return self._current_provider
    
    def get_config(self) -> LLMControlConfig:
        """Get the current configuration."""
        return self.config
    
    def refresh_provider(self) -> Optional[LLMProvider]:
        """
        Create a fresh provider instance with current settings.
        
        Returns:
            LLMProvider instance if API key is set, None otherwise.
        """
        provider_name = self.config.provider
        encrypted_key = self.config.get_api_key_for_provider(provider_name)
        
        if not encrypted_key:
            return None
        
        # Decrypt the key
        try:
            api_key = self._secure_storage.decrypt(encrypted_key)
            if not api_key:
                # Try as unencrypted (legacy)
                api_key = encrypted_key
        except Exception:
            api_key = encrypted_key
        
        if not api_key:
            return None
        
        try:
            kwargs = {}
            if provider_name == "gptrub":
                kwargs["api_url"] = self.config.gptrub_api_url
            
            self._current_provider = get_provider(provider_name, api_key, **kwargs)
            self._current_provider.temperature = self.config.temperature
            self._current_provider.max_tokens = self.config.max_tokens
            
            return self._current_provider
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            return None
