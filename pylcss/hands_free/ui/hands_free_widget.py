# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hands-Free Control Widget.

Provides settings and status display for voice-based hands-free control.
Note: Camera-based head tracking has been removed.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QCheckBox, QFrame, QScrollArea,
    QTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Slot

from pylcss.hands_free.hands_free_manager import HandsFreeManager
from pylcss.hands_free.config import HandsFreeConfig
from pylcss.hands_free.ui.llm_settings_widget import LLMSettingsWidget

logger = logging.getLogger(__name__)


class HandsFreeWidget(QWidget):
    """
    Main widget for hands-free control settings and monitoring.
    
    Features:
    - Voice control toggle
    - Voice command log
    - Dictation mode
    
    Note: Camera preview and head tracking have been removed.
    """
    
    def __init__(self, manager: HandsFreeManager, parent: Optional[QWidget] = None):
        """
        Initialize the hands-free widget.
        
        Args:
            manager: The hands-free manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.manager = manager
        self.config = manager.get_config()
        
        # Debounce timer for config saves
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)  # Save after 500ms of no changes
        self._save_timer.timeout.connect(self._do_save_config)
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self) -> None:
        """Create the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Controls scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Control groups
        controls_layout.addWidget(self._create_main_controls())
        controls_layout.addWidget(self._create_status_group())
        controls_layout.addWidget(self._create_voice_control_group())
        controls_layout.addWidget(self._create_voice_control_group())
        controls_layout.addWidget(self._create_llm_settings_group())  # LLM Settings
        controls_layout.addWidget(self._create_command_log_group())
        controls_layout.addStretch()
        
        scroll.setWidget(controls_widget)
        main_layout.addWidget(scroll)

    def _create_status_group(self) -> QGroupBox:
        """Create the status display group."""
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        
        # Status label
        self.status_label = QLabel("Status: Not started")
        self.status_label.setStyleSheet("color: #808080;")
        layout.addWidget(self.status_label)
        
        return group
        
    def _create_main_controls(self) -> QWidget:
        """Create main start/stop controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 10)
        
        # Start button
        self.start_btn = QPushButton("▶ Start Voice Control")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d7d46;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a9d5a;
            }
            QPushButton:pressed {
                background-color: #1e6030;
            }
        """)
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #7d2d2d;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #9d3a3a;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #808080;
            }
        """)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_btn)
        
        # Pause button
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause_toggled)
        layout.addWidget(self.pause_btn)
        
        return widget
        
    def _create_voice_control_group(self) -> QGroupBox:
        """Create voice control settings group."""
        group = QGroupBox("Voice Control")
        layout = QVBoxLayout(group)
        
        # Enable checkbox
        self.voice_enabled_cb = QCheckBox("Enable Voice Control")
        self.voice_enabled_cb.setChecked(self.config.voice_control.enabled)
        self.voice_enabled_cb.toggled.connect(self._on_voice_control_toggled)
        layout.addWidget(self.voice_enabled_cb)
        
        # Model status
        self.model_status = QLabel("Model: Checking...")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        
        # Check model availability
        QTimer.singleShot(100, self._check_voice_model)
        
        # Hotword settings
        hotword_layout = QHBoxLayout()
        self.hotword_cb = QCheckBox("Use Hotword:")
        self.hotword_cb.setChecked(self.config.voice_control.hotword_enabled)
        hotword_layout.addWidget(self.hotword_cb)
        
        self.hotword_edit = QLabel(f'"{self.config.voice_control.hotword}"')
        hotword_layout.addWidget(self.hotword_edit)
        layout.addLayout(hotword_layout)
        
        # Dictation mode button
        self.dictation_btn = QPushButton("📝 Start Dictation Mode")
        self.dictation_btn.setCheckable(True)
        self.dictation_btn.toggled.connect(self._on_dictation_toggled)
        layout.addWidget(self.dictation_btn)
        
        # Current recognition
        self.partial_label = QLabel("")
        self.partial_label.setStyleSheet("color: #808080; font-style: italic;")
        self.partial_label.setWordWrap(True)
        layout.addWidget(self.partial_label)
        
        return group
    
    def _create_llm_settings_group(self) -> QGroupBox:
        """Create LLM settings group with quick settings and full config button."""
        group = QGroupBox("🤖 LLM Assistant Settings")
        layout = QVBoxLayout(group)
        
        # Quick settings widget (collapsed by default)
        self.llm_settings = LLMSettingsWidget(self.config.llm_control)
        self.llm_settings.settings_changed.connect(self._on_llm_settings_changed)
        self.llm_settings.provider_connected.connect(self._on_provider_connected)
        layout.addWidget(self.llm_settings)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        # Full settings button
        full_settings_btn = QPushButton("⚙️ Full Settings...")
        full_settings_btn.setToolTip("Open comprehensive LLM configuration dialog")
        full_settings_btn.clicked.connect(self._open_llm_config_dialog)
        full_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a5a7a;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a6a8a;
            }
        """)
        btn_layout.addWidget(full_settings_btn)
        
        layout.addLayout(btn_layout)
        
        return group
    
    @Slot()
    def _open_llm_config_dialog(self) -> None:
        """Open the full LLM configuration dialog."""
        from pylcss.hands_free.ui.llm_config_dialog import LLMConfigDialog
        dialog = LLMConfigDialog(self)
        dialog.settings_saved.connect(self._on_llm_config_saved)
        dialog.exec()
    
    @Slot()
    def _on_llm_config_saved(self) -> None:
        """Handle LLM config dialog save."""
        # Reload config from file (config dialog saved to file)
        from pylcss.hands_free.config import HandsFreeConfig
        new_config = HandsFreeConfig.load()
        self.config = new_config
        
        # Update the inline LLM settings widget to reflect new config
        self.llm_settings.config = new_config.llm_control
        self.llm_settings._load_current_config()
        
        self._log_command("✓ LLM settings updated")
        
        # Re-initialize manager's provider with new config
        self.manager.update_config(new_config)
    
    @Slot()
    def _open_llm_chat_dialog(self) -> None:
        """Open the LLM chat dialog."""
        from pylcss.hands_free.ui.llm_chat_dialog import LLMChatDialog
        if not hasattr(self, '_chat_dialog') or self._chat_dialog is None:
            # Pass the manager's command dispatcher
            dispatcher = None
            if hasattr(self.manager, '_command_dispatcher'):
                dispatcher = self.manager._command_dispatcher
            self._chat_dialog = LLMChatDialog(command_dispatcher=dispatcher, parent=self)
        self._chat_dialog.show()
        self._chat_dialog.raise_()
    
    @Slot()
    def _on_llm_settings_changed(self) -> None:
        """Handle LLM settings change."""
        # Update the manager's config
        self.config.llm_control = self.llm_settings.get_config()
        self._save_timer.start()
    
    @Slot(str)
    def _on_provider_connected(self, provider: str) -> None:
        """Handle successful provider connection."""
        self._log_command(f"✓ LLM connected: {provider}")
        # Notify manager of new provider
        if hasattr(self.manager, 'set_llm_provider'):
            self.manager.set_llm_provider(self.llm_settings.get_current_provider())
        
    def _create_command_log_group(self) -> QGroupBox:
        """Create command log group."""
        group = QGroupBox("Command Log")
        layout = QVBoxLayout(group)
        
        self.command_log = QTextEdit()
        self.command_log.setReadOnly(True)
        self.command_log.setMaximumHeight(150)
        self.command_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                font-family: Consolas, monospace;
            }
        """)
        layout.addWidget(self.command_log)
        
        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.command_log.clear)
        layout.addWidget(clear_btn)
        
        return group
        
    def _connect_signals(self) -> None:
        """Connect manager signals to UI updates."""
        self.manager.status_changed.connect(self._on_status_changed)
        self.manager.command_recognized.connect(self._on_command_recognized)
        self.manager.partial_text.connect(self._on_partial_text)
        self.manager.error_occurred.connect(self._on_error)
        # Connect agentic progress
        if hasattr(self.manager, 'agentic_progress'):
            self.manager.agentic_progress.connect(self._on_agentic_progress)
        
    # --- Slots ---
    
    @Slot()
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if self.manager.start():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            
    @Slot()
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.manager.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setChecked(False)
        
    @Slot(bool)
    def _on_pause_toggled(self, paused: bool) -> None:
        """Handle pause button toggle."""
        if paused:
            self.manager.pause()
            self.pause_btn.setText("▶ Resume")
        else:
            self.manager.resume()
            self.pause_btn.setText("⏸ Pause")
            
    @Slot(bool)
    def _on_voice_control_toggled(self, enabled: bool) -> None:
        """Handle voice control enable toggle."""
        self.manager.set_voice_control_enabled(enabled)
        
    @Slot()
    def _do_save_config(self) -> None:
        """Actually save the configuration to disk (debounced)."""
        try:
            self.config.save()
            self._log_command("✓ Settings saved")
            logger.info("Hands-free config saved")
            # Also update the manager with the new config
            self.manager.update_config(self.config)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            self._log_command(f"❌ Save failed: {e}")
            
    @Slot(bool)
    def _on_dictation_toggled(self, enabled: bool) -> None:
        """Handle dictation mode toggle."""
        if enabled:
            self.manager._start_dictation()
            self.dictation_btn.setText("📝 Stop Dictation Mode")
        else:
            self.manager._stop_dictation()
            self.dictation_btn.setText("📝 Start Dictation Mode")
            
    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        """Handle status change from manager."""
        self.status_label.setText(f"Status: {status}")
        
    @Slot(str)
    def _on_command_recognized(self, command: str) -> None:
        """Handle recognized command."""
        self._log_command(f"✓ {command}")
        
    @Slot(str)
    def _on_partial_text(self, text: str) -> None:
        """Handle partial recognition."""
        self.partial_label.setText(f"Hearing: {text}...")
            
    @Slot(str)
    def _on_error(self, error: str) -> None:
        """Handle error from manager."""
        self._log_command(f"❌ Error: {error}")
        self.status_label.setText(f"Error: {error}")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        
    @Slot(str)
    def _on_agentic_progress(self, message: str) -> None:
        """Handle agentic progress update."""
        # Update chat dialog if visible
        if hasattr(self, '_chat_dialog') and self._chat_dialog and self._chat_dialog.isVisible():
            self._chat_dialog.add_agent_thought(message)
        
    def _check_voice_model(self) -> None:
        """Check if voice model is available."""
        from pylcss.hands_free.voice_controller import VoiceController, VOSK_AVAILABLE
        
        if not VOSK_AVAILABLE:
            self.model_status.setText("❌ Vosk not installed")
            self.model_status.setStyleSheet("color: #ff6b6b;")
            return
            
        vc = VoiceController(self.config.voice_control)
        if vc.is_model_available():
            self.model_status.setText("✓ Model ready")
            self.model_status.setStyleSheet("color: #6bff6b;")
        else:
            info = vc.get_model_download_info()
            self.model_status.setText(
                f"❌ Model not found.\nDownload: {info['url']}\n"
                f"Extract to: {info['path']}"
            )
            self.model_status.setStyleSheet("color: #ff6b6b;")
            
    def _log_command(self, text: str) -> None:
        """Add text to command log."""
        self.command_log.append(text)
        # Scroll to bottom
        scrollbar = self.command_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
