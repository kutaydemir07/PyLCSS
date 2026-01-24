# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Overlay Widget for Hands-Free Status Display.

Provides a floating overlay showing current hands-free status,
recognized commands, visual feedback, and LLM conversations.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QPen

logger = logging.getLogger(__name__)


class OverlayWidget(QWidget):
    """
    Floating overlay widget showing hands-free status and LLM conversations.
    
    Displays:
    - Active/paused indicator
    - Last recognized command
    - Partial voice recognition
    - LLM conversations (what you said + response)
    """
    
    hidden_signal = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Window flags for overlay
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Size - expandable for LLM mode
        self._base_height = 80
        self._llm_height = 250
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)
        
        # State
        self._active = False
        self._paused = False
        self._llm_mode = False
        self._last_command = ""
        self._partial_text = ""
        
        # Timers
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._clear_command)
        self._fade_timer.setSingleShot(True)
        
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide)
        self._auto_hide_timer.setSingleShot(True)
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Create the overlay UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # Status row
        status_layout = QHBoxLayout()
        
        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #808080; font-size: 16px;")
        status_layout.addWidget(self.status_indicator)
        
        # Status text
        self.status_label = QLabel("Hands-Free: Off")
        self.status_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # Command display (for regular commands)
        self.command_label = QLabel("")
        self.command_label.setStyleSheet("color: #6bff6b; font-size: 12px;")
        self.command_label.setWordWrap(True)
        layout.addWidget(self.command_label)
        
        # Partial text / what you're saying
        self.partial_label = QLabel("")
        self.partial_label.setStyleSheet("color: #60a5fa; font-size: 12px; font-style: italic;")
        self.partial_label.setWordWrap(True)
        layout.addWidget(self.partial_label)
        
        # LLM Section (hidden by default)
        self.llm_frame = QFrame()
        self.llm_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(40, 40, 45, 180);
                border-radius: 6px;
                margin-top: 4px;
            }
        """)
        llm_layout = QVBoxLayout(self.llm_frame)
        llm_layout.setContentsMargins(8, 6, 8, 6)
        llm_layout.setSpacing(4)
        
        # User text
        self.user_header = QLabel("💬 You:")
        self.user_header.setStyleSheet("color: #60a5fa; font-size: 10px; font-weight: bold;")
        llm_layout.addWidget(self.user_header)
        
        self.user_label = QLabel("")
        self.user_label.setWordWrap(True)
        self.user_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        llm_layout.addWidget(self.user_label)
        
        # Response text  
        self.response_header = QLabel("🤖 Assistant:")
        self.response_header.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold;")
        llm_layout.addWidget(self.response_header)
        
        self.response_label = QLabel("")
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        llm_layout.addWidget(self.response_label)
        
        # Hint
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("color: #fbbf24; font-size: 10px; font-style: italic;")
        llm_layout.addWidget(self.hint_label)
        
        self.llm_frame.hide()
        layout.addWidget(self.llm_frame)
        
        layout.addStretch()
        
    def paintEvent(self, event) -> None:
        """Paint the semi-transparent background."""
        painter = QPainter(self)
        if not painter.isActive():
            return

        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw rounded rectangle background
        painter.setBrush(QBrush(QColor(25, 25, 30, 220)))
        painter.setPen(QPen(QColor(80, 80, 80, 200), 1))
        painter.drawRoundedRect(self.rect(), 10, 10)
        painter.end()
        
    def set_active(self, active: bool) -> None:
        """Set the active state."""
        self._active = active
        self._update_status_display()
        
    def set_paused(self, paused: bool) -> None:
        """Set the paused state."""
        self._paused = paused
        self._update_status_display()
        
    def show_command(self, command: str, duration_ms: int = 3000) -> None:
        """Show a recognized command."""
        self._last_command = command
        self.command_label.setText(f"✓ {command}")
        self._fade_timer.stop()
        self._fade_timer.start(duration_ms)
        
    def show_partial(self, text: str) -> None:
        """Show partial voice recognition."""
        self._partial_text = text
        if text:
            self.partial_label.setText(f"🎤 {text}...")
            # Also show in user label for LLM mode
            if self._llm_mode:
                self.user_label.setText(f"{text}...")
        else:
            self.partial_label.setText("")
            
    # --- LLM Mode Methods ---
    
    def show_llm_listening(self) -> None:
        """Enter LLM listening mode."""
        self._llm_mode = True
        self.status_indicator.setStyleSheet("color: #60a5fa; font-size: 16px;")
        self.status_label.setText("🎤 LLM: Listening...")
        
        # Show LLM section
        self.llm_frame.show()
        self.user_label.setText("")
        self.response_label.setText("")
        self.response_header.hide()
        self.hint_label.setText("Speak to the AI assistant")
        self.hint_label.show()
        
        self._resize_for_llm()
        self._reposition()
        
    def show_llm_thinking(self, user_text: str = "", detail_text: str = "") -> None:
        """Show LLM is thinking."""
        self._llm_mode = True
        self.status_indicator.setStyleSheet("color: #fbbf24; font-size: 16px;")
        
        status_text = "🧠 LLM: Thinking..."
        if detail_text:
            status_text = f"🧠 {detail_text}"
        self.status_label.setText(status_text)
        
        if user_text:
            self.user_label.setText(user_text)
            
        self.hint_label.hide()
        self.response_header.hide()
        self.response_label.setText("")
        self.llm_frame.show()
        
        self._resize_for_llm()
        self._reposition()
        
    def show_llm_response(self, response: str, has_actions: bool = False, timeout_ms: int = 8000) -> None:
        """Show LLM response."""
        self._llm_mode = True
        self.status_indicator.setStyleSheet("color: #4ade80; font-size: 16px;")
        self.status_label.setText("🤖 LLM: Response")
        
        # Truncate long responses
        display_text = response[:350] + "..." if len(response) > 350 else response
        self.response_label.setText(display_text)
        self.response_header.show()
        
        if has_actions:
            self.hint_label.setText("📋 Executing actions...")
            self.hint_label.show()
        else:
            self.hint_label.hide()
            # Auto-hide after timeout
            if timeout_ms > 0:
                self._auto_hide_timer.start(timeout_ms)
            else:
                self._auto_hide_timer.stop()
        
        self.llm_frame.show()
        self._resize_for_llm()
        self._reposition()
        
    def show_llm_error(self, error: str) -> None:
        """Show LLM error."""
        self.status_indicator.setStyleSheet("color: #ef4444; font-size: 16px;")
        self.status_label.setText("❌ LLM: Error")
        self.response_label.setText(error[:200])
        self.response_header.show()
        self.hint_label.hide()
        self._auto_hide_timer.start(5000)
        
    def show_llm_confirmed(self) -> None:
        """Show action confirmed."""
        self.status_indicator.setStyleSheet("color: #4ade80; font-size: 16px;")  
        self.status_label.setText("✅ Actions Executed!")
        self.hint_label.hide()
        self._auto_hide_timer.start(3000)
        
    def show_llm_cancelled(self) -> None:
        """Show action cancelled."""
        self.status_indicator.setStyleSheet("color: #ef4444; font-size: 16px;")
        self.status_label.setText("❌ Cancelled")
        self.hint_label.hide()
        self._auto_hide_timer.start(2000)
        
    def hide_llm_mode(self) -> None:
        """Exit LLM mode and collapse."""
        self._llm_mode = False
        self._auto_hide_timer.stop()
        self.llm_frame.hide()
        self._resize_for_normal()
        self._update_status_display()
        self.hidden_signal.emit()
        
    def start_auto_hide(self, delay_ms: int = 5000) -> None:
        """Start auto-hide timer."""
        self._auto_hide_timer.start(delay_ms)
            
    def _update_status_display(self) -> None:
        """Update the status indicator and text."""
        if self._llm_mode:
            return  # Don't override LLM mode display
            
        if not self._active:
            self.status_indicator.setStyleSheet("color: #808080; font-size: 16px;")
            self.status_label.setText("Hands-Free: Off")
        elif self._paused:
            self.status_indicator.setStyleSheet("color: #ffaa00; font-size: 16px;")
            self.status_label.setText("Hands-Free: Paused")
        else:
            self.status_indicator.setStyleSheet("color: #6bff6b; font-size: 16px;")
            self.status_label.setText("Hands-Free: Active")
            
    def _clear_command(self) -> None:
        """Clear the command display."""
        self.command_label.setText("")
        
    def _on_auto_hide(self) -> None:
        """Auto-hide LLM mode after timeout."""
        self.hide_llm_mode()
        
    def _resize_for_llm(self) -> None:
        """Expand overlay for LLM conversation."""
        self.setMinimumHeight(self._llm_height)
        self.setMaximumHeight(600)  # Allow growth up to 600px
        self.adjustSize()  # Resize to fit content
        
    def _resize_for_normal(self) -> None:
        """Collapse overlay to normal size."""
        self.setMaximumHeight(self._base_height)
        self.setFixedHeight(self._base_height)
        self.adjustSize()
        
    def _reposition(self) -> None:
        """Reposition in top-right corner."""
        if self.parent():
            self.position_in_corner("top-right")
        
    def position_in_corner(self, corner: str = "top-right") -> None:
        """Position the overlay in a screen corner."""
        if not self.parent():
            return
            
        parent_rect = self.parent().rect()
        margin = 10
        
        if corner == "top-right":
            x = parent_rect.width() - self.width() - margin
            y = margin
        elif corner == "top-left":
            x = margin
            y = margin
        elif corner == "bottom-right":
            x = parent_rect.width() - self.width() - margin
            y = parent_rect.height() - self.height() - margin
        elif corner == "bottom-left":
            x = margin
            y = parent_rect.height() - self.height() - margin
        else:
            x = parent_rect.width() - self.width() - margin
            y = margin
            
        self.move(x, y)
        
    # Alias methods for compatibility with LLMOverlayWidget interface
    def show_listening(self):
        self.show_llm_listening()
        
    def show_thinking(self, user_text: str = "", detail_text: str = ""):
        self.show_llm_thinking(user_text, detail_text)
        
    def show_response(self, message: str, has_actions: bool = False, timeout_ms: int = 8000):
        self.show_llm_response(message, has_actions, timeout_ms)
        
    def show_error(self, error: str):
        self.show_llm_error(error)
        
    def show_confirmed(self):
        self.show_llm_confirmed()
        
    def show_cancelled(self):
        self.show_llm_cancelled()
        
    def hide_overlay(self):
        self.hide_llm_mode()
        
    def show_user_text(self, text: str):
        self.user_label.setText(text)
        
    def get_state(self) -> str:
        if self._llm_mode:
            return "llm_active"
        return "normal"
