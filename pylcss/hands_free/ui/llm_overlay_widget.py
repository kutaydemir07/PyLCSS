# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
LLM Status Overlay Widget.

Provides a floating overlay for LLM voice interaction feedback.
Shows what user said, thinking animation, LLM responses, and action confirmations.
"""

import logging
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QApplication, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


class LLMOverlayWidget(QFrame):
    """
    Floating overlay for LLM voice interaction.
    
    Shows:
    - What user said (your question)
    - LLM thinking state
    - LLM response
    - Action confirmations
    """
    
    # Signals
    hidden_signal = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._state = "idle"
        self._dots = 0
        self._user_text = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.timeout.connect(self._auto_hide)
        self._auto_hide_timer.setSingleShot(True)
        
        self._setup_ui()
        self.hide()
        
    def _setup_ui(self):
        """Create the overlay UI with conversation display."""
        # Frameless, always on top
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint | 
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Styling - darker, more visible
        self.setStyleSheet("""
            LLMOverlayWidget {
                background-color: rgba(20, 20, 25, 245);
                border: 2px solid #4a7a4a;
                border-radius: 12px;
            }
        """)
        
        self.setMinimumWidth(400)
        self.setMaximumWidth(600)
        self.setMinimumHeight(100)
        self.setMaximumHeight(350)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(10)
        
        # Header with status indicator
        header = QHBoxLayout()
        
        # Status indicator (green dot)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("QLabel { color: #4ade80; font-size: 14px; }")
        header.addWidget(self.status_dot)
        
        # Status text
        self.status_label = QLabel("🤖 Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4ade80;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        header.addWidget(self.status_label)
        header.addStretch()
        layout.addLayout(header)
        
        # User message (what you said)
        self.user_frame = QFrame()
        self.user_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(60, 80, 100, 150);
                border-radius: 8px;
                padding: 5px;
            }
        """)
        user_layout = QVBoxLayout(self.user_frame)
        user_layout.setContentsMargins(10, 8, 10, 8)
        user_layout.setSpacing(3)
        
        user_header = QLabel("💬 You said:")
        user_header.setStyleSheet("QLabel { color: #60a5fa; font-size: 11px; font-weight: bold; }")
        user_layout.addWidget(user_header)
        
        self.user_text_label = QLabel("")
        self.user_text_label.setWordWrap(True)
        self.user_text_label.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; }")
        user_layout.addWidget(self.user_text_label)
        
        self.user_frame.hide()
        layout.addWidget(self.user_frame)
        
        # LLM Response
        self.response_frame = QFrame()
        self.response_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(50, 80, 50, 150);
                border-radius: 8px;
                padding: 5px;
            }
        """)
        response_layout = QVBoxLayout(self.response_frame)
        response_layout.setContentsMargins(10, 8, 10, 8)
        response_layout.setSpacing(3)
        
        response_header = QLabel("🤖 Assistant:")
        response_header.setStyleSheet("QLabel { color: #4ade80; font-size: 11px; font-weight: bold; }")
        response_layout.addWidget(response_header)
        
        self.response_label = QLabel("")
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; }")
        response_layout.addWidget(self.response_label)
        
        self.response_frame.hide()
        layout.addWidget(self.response_frame)
        
        # Actions hint
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("""
            QLabel {
                color: #fbbf24;
                font-size: 11px;
                font-style: italic;
                padding-top: 5px;
            }
        """)
        self.hint_label.hide()
        layout.addWidget(self.hint_label)
    
    def show_partial(self, text: str):
        """Show partial/live recognition while user is speaking."""
        if text:
            self.user_text_label.setText(f"{text}...")
            self.user_frame.show()
            if not self.isVisible():
                self._position_overlay()
                self.show()
                self.raise_()
        
    def show_user_text(self, text: str):
        """Show what the user said (final text)."""
        self._user_text = text
        self.user_text_label.setText(text)
        self.user_frame.show()
        self._position_overlay()
        
    def show_listening(self):
        """Show listening state."""
        logger.info("LLM Overlay: Listening state")
        self._state = "listening"
        self.status_dot.setStyleSheet("QLabel { color: #60a5fa; font-size: 14px; }")
        self.status_label.setText("🎤 Listening...")
        self.status_label.setStyleSheet("QLabel { color: #60a5fa; font-size: 14px; font-weight: bold; }")
        
        # Clear previous conversation
        self.user_frame.hide()
        self.response_frame.hide()
        
        self.hint_label.setText("Speak your request to the AI assistant")
        self.hint_label.show()
        self._position_overlay()
        self.show()
        self.raise_()
        
    def show_thinking(self, user_text: str = "", detail_text: str = ""):
        """Show thinking state with animation and optional detail."""
        logger.info(f"LLM Overlay: Thinking state ({detail_text})")
        self._state = "thinking"
        self._dots = 0
        self._thinking_detail = detail_text
        
        # Show what user said
        if user_text:
            self.show_user_text(user_text)
        
        self.status_dot.setStyleSheet("QLabel { color: #fbbf24; font-size: 14px; }")
        self._update_thinking_label()
        self.status_label.setStyleSheet("QLabel { color: #fbbf24; font-size: 14px; font-weight: bold; }")
        
        self.response_frame.hide()
        self.hint_label.hide()
        
        self._timer.start(400)
        self._position_overlay()
        self.show()
        self.raise_()
        
    def show_response(self, message: str, has_actions: bool = False):
        """Show LLM response alongside user's question."""
        logger.info(f"LLM Overlay: Response - {message[:50]}...")
        self._state = "response"
        self._timer.stop()
        
        self.status_dot.setStyleSheet("QLabel { color: #4ade80; font-size: 14px; }")
        self.status_label.setText("🤖 Response")
        self.status_label.setStyleSheet("QLabel { color: #4ade80; font-size: 14px; font-weight: bold; }")
        
        # Truncate long messages for display
        display_msg = message[:400] + "..." if len(message) > 400 else message
        self.response_label.setText(display_msg)
        self.response_frame.show()
        
        if has_actions:
            self.hint_label.setText("📋 Say 'confirm' to execute or 'cancel' to abort")
            self.hint_label.show()
            self._state = "confirmation"
        else:
            self.hint_label.hide()
            # Don't auto-hide while speaking - let manager control this
            
        self._position_overlay()
        self.show()
        self.raise_()
        
    def show_actions_pending(self, action_count: int):
        """Show pending actions state."""
        logger.info(f"LLM Overlay: {action_count} actions pending")
        self._state = "confirmation"
        self.status_dot.setStyleSheet("QLabel { color: #fbbf24; font-size: 14px; }")
        self.status_label.setText(f"📋 {action_count} Action{'s' if action_count > 1 else ''} Pending")
        self.status_label.setStyleSheet("QLabel { color: #fbbf24; font-size: 14px; font-weight: bold; }")
        self.hint_label.setText("Say 'confirm' to execute or 'cancel' to abort")
        self.hint_label.show()
        self._position_overlay()
        self.show()
        self.raise_()
        
    def show_confirmed(self):
        """Show confirmation success."""
        logger.info("LLM Overlay: Confirmed")
        self._state = "idle"
        self._timer.stop()
        self.status_dot.setStyleSheet("QLabel { color: #4ade80; font-size: 14px; }")
        self.status_label.setText("✅ Actions Executed!")
        self.status_label.setStyleSheet("QLabel { color: #4ade80; font-size: 14px; font-weight: bold; }")
        self.hint_label.hide()
        self._auto_hide_timer.start(3000)
        
    def show_cancelled(self):
        """Show cancellation."""
        logger.info("LLM Overlay: Cancelled")
        self._state = "idle"
        self._timer.stop()
        self.status_dot.setStyleSheet("QLabel { color: #ef4444; font-size: 14px; }")
        self.status_label.setText("❌ Cancelled")
        self.status_label.setStyleSheet("QLabel { color: #ef4444; font-size: 14px; font-weight: bold; }")
        self.hint_label.hide()
        self._auto_hide_timer.start(2000)
        
    def show_error(self, error: str):
        """Show error state."""
        logger.info(f"LLM Overlay: Error - {error}")
        self._state = "error"
        self._timer.stop()
        self.status_dot.setStyleSheet("QLabel { color: #ef4444; font-size: 14px; }")
        self.status_label.setText("❌ Error")
        self.status_label.setStyleSheet("QLabel { color: #ef4444; font-size: 14px; font-weight: bold; }")
        
        # Show error in response area
        self.response_label.setText(error[:300])
        self.response_frame.show()
        self.hint_label.hide()
        
        self._auto_hide_timer.start(5000)
        self._position_overlay()
        self.show()
        self.raise_()
    
    def start_auto_hide(self, delay_ms: int = 5000):
        """Start auto-hide timer (called after speech finishes)."""
        self._auto_hide_timer.start(delay_ms)
        
    def hide_overlay(self):
        """Hide the overlay."""
        self._timer.stop()
        self._auto_hide_timer.stop()
        self._state = "idle"
        self.hide()
        self.hidden_signal.emit()
        
    def _animate(self):
        """Animate thinking dots."""
        self._dots = (self._dots + 1) % 4
        self._update_thinking_label()
        
    def _update_thinking_label(self):
        """Update thinking label text."""
        dots = "." * self._dots
        text = "🧠 Thinking"
        if hasattr(self, '_thinking_detail') and self._thinking_detail:
            text += f": {self._thinking_detail}"
        text += dots
        self.status_label.setText(text)
        
    def _auto_hide(self):
        """Auto-hide after timeout."""
        if self._state in ("response", "idle", "error"):
            self.hide_overlay()
            
    def _position_overlay(self):
        """Position overlay at top-center of screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            self.adjustSize()
            x = screen_geo.center().x() - self.width() // 2
            y = screen_geo.top() + 50
            self.move(x, y)
            
    def get_state(self) -> str:
        """Get current overlay state."""
        return self._state
