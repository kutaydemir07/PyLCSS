# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Overlay Widget for Hands-Free Status Display.

Provides a floating overlay showing current hands-free status,
recognized commands, and visual feedback.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QBrush, QPen

logger = logging.getLogger(__name__)


class OverlayWidget(QWidget):
    """
    Floating overlay widget showing hands-free status.
    
    Displays:
    - Active/paused indicator
    - Last recognized command
    - Partial voice recognition
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the overlay widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Window flags for overlay
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Size and position
        self.setFixedSize(300, 80)
        
        # State
        self._active = False
        self._paused = False
        self._last_command = ""
        self._partial_text = ""
        
        # Fade timer for command display
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._clear_command)
        self._fade_timer.setSingleShot(True)
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Create the overlay UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # Status row
        status_layout = QHBoxLayout()
        
        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #808080; font-size: 16px;")
        status_layout.addWidget(self.status_indicator)
        
        # Status text
        self.status_label = QLabel("Hands-Free: Off")
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # Command display
        self.command_label = QLabel("")
        self.command_label.setStyleSheet("color: #6bff6b; font-size: 12px;")
        self.command_label.setWordWrap(True)
        layout.addWidget(self.command_label)
        
        # Partial text display
        self.partial_label = QLabel("")
        self.partial_label.setStyleSheet("color: #808080; font-size: 11px; font-style: italic;")
        self.partial_label.setWordWrap(True)
        layout.addWidget(self.partial_label)
        
    def paintEvent(self, event) -> None:
        """Paint the semi-transparent background."""
        painter = QPainter(self)
        if not painter.isActive():
            return

        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw rounded rectangle background
        painter.setBrush(QBrush(QColor(30, 30, 30, 200)))
        painter.setPen(QPen(QColor(60, 60, 60, 200), 1))
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
        """
        Show a recognized command.
        
        Args:
            command: The command text to display
            duration_ms: How long to show it (milliseconds)
        """
        self._last_command = command
        self.command_label.setText(f"✓ {command}")
        
        # Start fade timer
        self._fade_timer.stop()
        self._fade_timer.start(duration_ms)
        
    def show_partial(self, text: str) -> None:
        """Show partial voice recognition."""
        self._partial_text = text
        if text:
            self.partial_label.setText(f"🎤 {text}...")
        else:
            self.partial_label.setText("")
            
    def _update_status_display(self) -> None:
        """Update the status indicator and text."""
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
        
    def position_in_corner(self, corner: str = "top-right") -> None:
        """
        Position the overlay in a screen corner.
        
        Args:
            corner: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
        """
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
