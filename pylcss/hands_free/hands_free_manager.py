# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Hands-Free Manager - Main orchestrator for the hands-free control system.

Provides voice recognition and command execution for hands-free control.
Note: Camera-based head tracking has been removed.
"""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from pylcss.hands_free.config import HandsFreeConfig
from pylcss.hands_free.voice_controller import VoiceController, VOSK_AVAILABLE
from pylcss.hands_free.mouse_controller import MouseController
from pylcss.hands_free.command_dispatcher import CommandDispatcher

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class HandsFreeManager(QObject):
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
    
    def __init__(self, main_window: Optional["QMainWindow"] = None):
        """
        Initialize the hands-free manager.
        
        Args:
            main_window: Reference to the PyLCSS main window
        """
        super().__init__()
        
        self.main_window = main_window
        self.config = HandsFreeConfig.load()
        
        # Components (head tracking removed - voice only)
        self._voice_controller: Optional[VoiceController] = None
        self._mouse_controller: Optional[MouseController] = None
        self._command_dispatcher: Optional[CommandDispatcher] = None
        
        # State
        self._initialized = False
        self._running = False
        
        # Availability flags
        self.voice_control_available = VOSK_AVAILABLE
        
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
        
        # Initialize voice controller
        if self.voice_control_available and self.config.voice_control.enabled:
            try:
                self._voice_controller = VoiceController(self.config.voice_control)
                
                # Set callbacks
                self._voice_controller.set_callbacks(
                    on_command=self._on_voice_command,
                    on_text=self._on_dictation_text,
                    on_partial=self._on_partial_text,
                    on_status=self._on_voice_status,
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
            
    def get_config(self) -> HandsFreeConfig:
        """Get the current configuration."""
        return self.config
        
    def update_config(self, config: HandsFreeConfig) -> None:
        """Update configuration and save."""
        self.config = config
            
        # Update voice controller config
        if self._voice_controller:
            self._voice_controller.config = config.voice_control
            
        # Save to file
        config.save()
        
    # --- Callbacks from components ---
    def _on_voice_command(self, command_name: str, command_data: Dict[str, Any]) -> None:
        """Handle recognized voice command."""
        self.command_recognized.emit(command_name)
        
        if self._command_dispatcher:
            self._command_dispatcher.dispatch(command_name, command_data)
            
    def _on_dictation_text(self, text: str) -> None:
        """Handle dictation text."""
        if self._mouse_controller:
            self._mouse_controller.write_text(text + " ")
            
    def _on_partial_text(self, text: str) -> None:
        """Handle partial recognition results."""
        self.partial_text.emit(text)
        
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
