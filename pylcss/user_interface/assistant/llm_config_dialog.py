# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Configuration Dialog.

A comprehensive settings dialog for configuring LLM providers, API keys,
model selection, and generation parameters.
"""

import logging
from typing import Optional, List, Dict

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QLineEdit, QSlider, QSpinBox, QCheckBox,
    QFrame, QSizePolicy, QMessageBox, QTabWidget, QWidget, QTextEdit,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont

from pylcss.assistant_systems.config import AssistantConfig, LLMControlConfig
from pylcss.assistant_systems.services.llm import (
    LLMProvider, get_provider, get_available_providers,
    PROVIDER_DISPLAY_NAMES, LLMProviderError, ModelInfo
)
from pylcss.assistant_systems.services.memory import get_secure_storage, LLMMemory

logger = logging.getLogger(__name__)


class LLMConfigDialog(QDialog):
    """
    Comprehensive LLM configuration dialog.
    
    Features:
    - Provider selection with visual feedback
    - API key management with encryption
    - Model selection per provider
    - Temperature and generation controls
    - Connection testing
    - Memory management
    """
    
    # Signal for thread-safe UI updates: (provider_name, provider_obj, models_list, error_obj)
    models_loaded_signal = Signal(str, object, object, object)
    settings_saved = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._config = AssistantConfig.load()
        self._secure_storage = get_secure_storage()
        self._providers: Dict[str, LLMProvider] = {}
        self._models_cache: Dict[str, List[ModelInfo]] = {}
        
        # Connect the thread-safe signal
        self.models_loaded_signal.connect(self._on_models_loaded)
        
        self.setWindowTitle("LLM Assistant Settings")
        self.setMinimumSize(450, 350)  # Smaller for laptops
        self.resize(550, 450)
        
        self._setup_ui()
        self._apply_dark_theme()
        self._load_config()
    
    # ... (skipping unchanged methods) ...

    def _load_models(self, provider_name: str) -> None:
        """Load models for a specific provider (async to prevent UI freeze)."""
        import threading
        
        logger.info(f"Starting model load for provider: {provider_name}")
        
        group = self.provider_configs[provider_name]
        key_input = group.findChild(QLineEdit, f"{provider_name}_key")
        model_combo = group.findChild(QComboBox, f"{provider_name}_model")
        status_label = group.findChild(QLabel, f"{provider_name}_status")
        load_btn = group.findChild(QPushButton, f"{provider_name}_load")
        
        api_key = key_input.text().strip() if key_input else ""
        
        if not api_key:
            logger.warning(f"No API key entered for {provider_name}")
            if status_label:
                status_label.setText("Enter API key first")
                status_label.setStyleSheet("color: #ff6b6b;")
            return
        
        if status_label:
            status_label.setText("Loading models...")
            status_label.setStyleSheet("color: #ffaa00;")
        
        if load_btn:
            load_btn.setEnabled(False)
        

        
        # Prepare kwargs for provider
        kwargs = {}
        if provider_name == "local":
            url_input = group.findChild(QLineEdit, f"{provider_name}_url")
            if url_input:
                kwargs["local_api_url"] = url_input.text().strip()
        
        def load_in_thread():
            try:
                # pass kwargs for local provider
                provider = get_provider(provider_name, api_key, **kwargs)
                models = provider.get_models()
                
                # Update UI from main thread using Signal (thread-safe)
                self.models_loaded_signal.emit(provider_name, provider, models, None)
                
            except Exception as e:
                # Signal error too
                self.models_loaded_signal.emit(provider_name, None, [], e)
        
        thread = threading.Thread(target=load_in_thread, daemon=True)
        thread.start()
    
    def _setup_ui(self) -> None:
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Configure your LLM Assistant")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(header)
        
        # Tab widget for organization
        tabs = QTabWidget()
        tabs.addTab(self._create_provider_tab(), "Providers")
        tabs.addTab(self._create_generation_tab(), "Generation")
        tabs.addTab(self._create_memory_tab(), "Memory")
        layout.addWidget(tabs)
        
        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #808080; font-size: 11px; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_connection)
        btn_layout.addWidget(self.test_btn)
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_and_close)
        save_btn.setStyleSheet("background-color: #4a7a4a; font-weight: bold;")
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _create_provider_tab(self) -> QWidget:
        """Create the providers configuration tab with scroll area."""
        # Create scroll area for small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)  # Compact spacing
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Active provider selection
        active_group = QGroupBox("Active Provider")
        active_layout = QHBoxLayout(active_group)
        active_layout.setContentsMargins(8, 8, 8, 8)
        
        active_layout.addWidget(QLabel("Use:"))
        self.active_provider_combo = QComboBox()
        self.active_provider_combo.setMinimumWidth(150)
        for provider_name in get_available_providers():
            display_name = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
            icon = self._get_provider_icon(provider_name)
            self.active_provider_combo.addItem(f"{display_name}", provider_name)
        self.active_provider_combo.currentIndexChanged.connect(self._on_active_provider_changed)
        active_layout.addWidget(self.active_provider_combo)
        active_layout.addStretch()
        
        layout.addWidget(active_group)
        
        # Provider-specific configuration
        self.provider_configs = {}
        
        for provider_name in get_available_providers():
            group = self._create_provider_config_group(provider_name)
            self.provider_configs[provider_name] = group
            layout.addWidget(group)
        
        layout.addStretch()
        scroll.setWidget(widget)
        return scroll
    
    def _create_provider_config_group(self, provider_name: str) -> QGroupBox:
        """Create configuration group for a specific provider."""
        display_name = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)
        icon = self._get_provider_icon(provider_name)
        group = QGroupBox(f"{display_name}")
        
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)
        
        # API Key
        layout.addWidget(QLabel("API Key:"), 0, 0)
        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.Password)
        key_input.setPlaceholderText(f"Enter your {display_name} API key...")
        key_input.setObjectName(f"{provider_name}_key")
        layout.addWidget(key_input, 0, 1)
        
        # Connect API key changes to auto-load models after debounce
        # Create a debounce timer for this provider
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(1500)  # 1.5 second debounce after typing stops
        timer.timeout.connect(lambda pn=provider_name: self._load_models(pn))
        
        # Store timer reference to avoid garbage collection
        if not hasattr(self, '_key_timers'):
            self._key_timers = {}
        self._key_timers[provider_name] = timer
        
        key_input.textChanged.connect(lambda text, t=timer, pn=provider_name: self._on_api_key_changed(pn, text, t))
        
        show_btn = QPushButton("*")
        show_btn.setMaximumWidth(35)
        show_btn.setCheckable(True)
        show_btn.toggled.connect(lambda checked, ki=key_input: 
            ki.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        layout.addWidget(show_btn, 0, 2)
        

        
        # Model selection
        row = 1
        layout.addWidget(QLabel("Model:"), row, 0)
        model_combo = QComboBox()
        model_combo.setObjectName(f"{provider_name}_model")
        model_combo.addItem("Connect to load models...", "")
        layout.addWidget(model_combo, row, 1)
        
        load_btn = QPushButton("Load Models")
        load_btn.setObjectName(f"{provider_name}_load")
        load_btn.clicked.connect(lambda checked, pn=provider_name: self._load_models(pn))
        layout.addWidget(load_btn, row, 2)
        
        # Status indicator
        status_label = QLabel("")
        status_label.setObjectName(f"{provider_name}_status")
        status_label.setStyleSheet("color: #808080; font-size: 10px;")
        layout.addWidget(status_label, row + 1, 0, 1, 3)
        
        # Local URL (only for local provider)
        if provider_name == "local":
            row += 2
            layout.addWidget(QLabel("Local URL:"), row, 0)
            url_input = QLineEdit()
            url_input.setPlaceholderText("http://localhost:1234/v1")
            url_input.setObjectName(f"{provider_name}_url")
            layout.addWidget(url_input, row, 1, 1, 2)
        
        return group
    
    def _on_api_key_changed(self, provider_name: str, text: str, timer: QTimer) -> None:
        """Handle API key text change with debounced auto-load."""
        # Update status to show key was changed
        group = self.provider_configs.get(provider_name)
        if group:
            status_label = group.findChild(QLabel, f"{provider_name}_status")
            if status_label and text.strip():
                status_label.setText("Will load models shortly...")
                status_label.setStyleSheet("color: #808080; font-size: 10px;")
        
        # Restart the debounce timer - models will load after typing stops
        if text.strip():
            timer.start()  # Restart the timer
        else:
            timer.stop()  # Don't try to load with empty key
    
    def _create_generation_tab(self) -> QWidget:
        """Create the generation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        gen_group = QGroupBox("Generation Parameters")
        gen_layout = QGridLayout(gen_group)
        gen_layout.setColumnStretch(1, 1)
        
        # Temperature
        gen_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(200)
        self.temp_slider.setValue(70)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        gen_layout.addWidget(self.temp_slider, 0, 1)
        
        self.temp_label = QLabel("0.7")
        self.temp_label.setMinimumWidth(40)
        gen_layout.addWidget(self.temp_label, 0, 2)
        
        # Temperature description
        temp_desc = QLabel("Lower = more focused, Higher = more creative")
        temp_desc.setStyleSheet("color: #808080; font-size: 10px;")
        gen_layout.addWidget(temp_desc, 1, 1, 1, 2)
        
        # Max Tokens
        gen_layout.addWidget(QLabel("Max Tokens:"), 2, 0)
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setMinimum(100)
        self.tokens_spin.setMaximum(16000)
        self.tokens_spin.setSingleStep(100)
        self.tokens_spin.setValue(1000)
        gen_layout.addWidget(self.tokens_spin, 2, 1)
        
        tokens_desc = QLabel("Maximum response length")
        tokens_desc.setStyleSheet("color: #808080; font-size: 10px;")
        gen_layout.addWidget(tokens_desc, 3, 1, 1, 2)
        
        # Auto-execute
        self.auto_execute_cb = QCheckBox("Auto-execute actions (no confirmation)")
        self.auto_execute_cb.setToolTip("Execute LLM-suggested actions immediately without asking")
        gen_layout.addWidget(self.auto_execute_cb, 4, 0, 1, 3)
        
        layout.addWidget(gen_group)
        layout.addStretch()
        return widget
    
    def _create_memory_tab(self) -> QWidget:
        """Create the memory settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        mem_group = QGroupBox("Conversation Memory")
        mem_layout = QVBoxLayout(mem_group)
        
        # Enable memory
        self.memory_cb = QCheckBox("Enable conversation memory")
        self.memory_cb.setToolTip("LLM remembers previous conversations for context")
        mem_layout.addWidget(self.memory_cb)
        
        # Memory size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Context size:"))
        self.memory_size_spin = QSpinBox()
        self.memory_size_spin.setMinimum(5)
        self.memory_size_spin.setMaximum(100)
        self.memory_size_spin.setValue(20)
        self.memory_size_spin.setSuffix(" messages")
        size_layout.addWidget(self.memory_size_spin)
        size_layout.addStretch()
        mem_layout.addLayout(size_layout)
        
        # Memory stats
        self.memory_stats = QLabel("Loading memory stats...")
        self.memory_stats.setStyleSheet("color: #808080; padding-top: 10px;")
        mem_layout.addWidget(self.memory_stats)
        
        # Clear memory button
        clear_btn = QPushButton("Clear All Memory")
        clear_btn.clicked.connect(self._clear_memory)
        mem_layout.addWidget(clear_btn)
        
        layout.addWidget(mem_group)
        layout.addStretch()
        
        # Load memory stats
        self._update_memory_stats()
        
        return widget
    
    def _get_provider_icon(self, provider_name: str) -> str:
        """Get icon for provider."""
        return ""
    
    def _load_config(self) -> None:
        """Load current configuration into UI."""
        llm = self._config.llm_control
        
        # Set active provider
        for i in range(self.active_provider_combo.count()):
            if self.active_provider_combo.itemData(i) == llm.provider:
                self.active_provider_combo.setCurrentIndex(i)
                break
        
        # Load API keys for each provider
        for provider_name in get_available_providers():
            group = self.provider_configs[provider_name]
            key_input = group.findChild(QLineEdit, f"{provider_name}_key")
            
            if key_input:
                encrypted_key = llm.get_api_key_for_provider(provider_name)
                if encrypted_key:
                    # Try to decrypt, fall back to raw
                    try:
                        decrypted = self._secure_storage.decrypt(encrypted_key)
                        key_input.setText(decrypted if decrypted else encrypted_key)
                    except:
                        key_input.setText(encrypted_key)
            
            # Load Local URL
            if provider_name == "local":
                url_input = group.findChild(QLineEdit, f"{provider_name}_url")
                if url_input:
                    url_input.setText(llm.local_api_url)
            

        
        # Generation settings
        self.temp_slider.setValue(int(llm.temperature * 100))
        self.tokens_spin.setValue(llm.max_tokens)
        self.auto_execute_cb.setChecked(llm.auto_execute)
        
        # Memory settings
        self.memory_cb.setChecked(llm.memory_enabled)
        self.memory_size_spin.setValue(llm.max_memory_messages)
        
        self._update_provider_visibility()
    
    def _update_provider_visibility(self) -> None:
        """Highlight the active provider."""
        active = self.active_provider_combo.currentData()
        for provider_name, group in self.provider_configs.items():
            if provider_name == active:
                group.setStyleSheet("QGroupBox { border: 2px solid #5a9a5a; }")
            else:
                group.setStyleSheet("")
    
    @Slot()
    def _on_active_provider_changed(self) -> None:
        """Handle active provider change."""
        self._update_provider_visibility()
    
    @Slot(int)
    def _on_temp_changed(self, value: int) -> None:
        """Update temperature label."""
        self.temp_label.setText(f"{value / 100:.1f}")
    

    def _on_models_loaded(self, provider_name: str, provider, models, error) -> None:
        """Handle models loaded callback (runs in main thread)."""
        group = self.provider_configs.get(provider_name)
        if not group:
            return

        model_combo = group.findChild(QComboBox, f"{provider_name}_model")
        status_label = group.findChild(QLabel, f"{provider_name}_status")
        load_btn = group.findChild(QPushButton, f"{provider_name}_load")
        
        if load_btn:
            load_btn.setEnabled(True)
        
        if error:
            if status_label:
                error_msg = str(error)[:60]
                status_label.setText(f"Error: {error_msg}")
                status_label.setStyleSheet("color: #ff6b6b;")
            return
        
        if provider:
            self._providers[provider_name] = provider
            self._models_cache[provider_name] = models
            
            if model_combo:
                model_combo.clear()
                for model in models:
                    model_combo.addItem(model.name, model.id)
            
            if status_label:
                status_label.setText(f"Loaded {len(models)} models")
                status_label.setStyleSheet("color: #6bff6b;")
    
    @Slot()
    def _test_connection(self) -> None:
        """Test connection with active provider."""
        active = self.active_provider_combo.currentData()
        self._load_models(active)
    
    def _update_memory_stats(self) -> None:
        """Update memory statistics display."""
        try:
            memory = LLMMemory()
            count = memory.get_conversation_count()
            self.memory_stats.setText(f"Conversations stored: {count}")
        except Exception as e:
            self.memory_stats.setText(f"Error loading memory: {e}")
    
    @Slot()
    def _clear_memory(self) -> None:
        """Clear all stored conversations."""
        reply = QMessageBox.question(
            self, "Clear Memory",
            "Are you sure you want to delete all conversation history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                memory = LLMMemory()
                memory.clear_all()
                self.memory_stats.setText("Memory cleared")
                self._set_status("Conversation history cleared")
            except Exception as e:
                self._set_status(f"Error clearing memory: {e}", error=True)
    
    @Slot()
    def _save_and_close(self) -> None:
        """Save settings and close dialog."""
        llm = self._config.llm_control
        
        # Save active provider
        llm.provider = self.active_provider_combo.currentData()
        
        # Save API keys (encrypted)
        for provider_name in get_available_providers():
            group = self.provider_configs[provider_name]
            key_input = group.findChild(QLineEdit, f"{provider_name}_key")
            
            if key_input:
                plaintext = key_input.text().strip()
                if plaintext:
                    encrypted = self._secure_storage.encrypt(plaintext)
                    llm.set_api_key_for_provider(provider_name, encrypted)
                else:
                    llm.set_api_key_for_provider(provider_name, "")
            
            # Save Local URL
            if provider_name == "local":
                url_input = group.findChild(QLineEdit, f"{provider_name}_url")
                if url_input:
                    llm.local_api_url = url_input.text().strip()
            # Save selected model
            model_combo = group.findChild(QComboBox, f"{provider_name}_model")
            if model_combo and provider_name == llm.provider:
                model_id = model_combo.currentData()
                if model_id:
                    llm.selected_model = model_id
                    llm.model = model_id
        
        # Save generation settings
        llm.temperature = self.temp_slider.value() / 100.0
        llm.max_tokens = self.tokens_spin.value()
        llm.auto_execute = self.auto_execute_cb.isChecked()
        
        # Save memory settings
        llm.memory_enabled = self.memory_cb.isChecked()
        llm.max_memory_messages = self.memory_size_spin.value()
        
        # Persist to file
        self._config.save()
        
        self.settings_saved.emit()
        self._set_status("Settings saved!")
        
        QTimer.singleShot(500, self.accept)
    
    def _set_status(self, text: str, error: bool = False) -> None:
        """Update status bar."""
        self.status_label.setText(text)
        color = "#ff6b6b" if error else "#6bff6b"
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px; padding: 5px;")
    
    def _apply_dark_theme(self) -> None:
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                padding-top: 25px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QComboBox, QLineEdit, QSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
                min-height: 20px;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus {
                border-color: #5a9a5a;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #5a5a5a;
            }
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9a5a;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                border-bottom-color: #3a3a3a;
            }
        """)
