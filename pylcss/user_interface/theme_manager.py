# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Theme management for the PyLCSS application.

This module provides functions for applying visual themes to the Qt application,
including modern dark theme configuration with professional styling.
"""

from PySide6 import QtWidgets, QtGui, QtCore

# PROFESSIONAL COLOR PALETTE (Dark Modern)
COLORS = {
    "bg_dark": "#1e1f22",       # Main Window Background
    "bg_panel": "#2b2d31",      # Panels/Dock Widgets
    "bg_input": "#383a40",      # Text Inputs
    "primary": "#d29922",       # Main Action Color (blue)
    "primary_hover": "#e3b341",
    "text_main": "#ffffff",
    "text_dim": "#b5bac1",
    "border": "#1e1f22",
    "success": "#2ecc71",
    "danger": "#ed4245",
}

MODERN_STYLESHEET = f"""
/* --- MAIN WINDOW & CONTAINERS --- */
QMainWindow, QDialog {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_main"]};
}}

QWidget {{
    color: {COLORS["text_main"]};
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
    font-size: 9pt;
}}

/* --- TAB WIDGETS --- */
QTabWidget::pane {{
    border: 1px solid {COLORS["bg_panel"]};
    background-color: {COLORS["bg_panel"]};
    border-radius: 5px;
}}

QTabBar::tab {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_dim"]};
    padding: 8px 20px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_main"]};
    border-bottom: 2px solid {COLORS["primary"]};
    font-weight: bold;
}}

QTabBar::tab:hover {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text_main"]};
}}

/* --- BUTTONS --- */
QPushButton {{
    background-color: {COLORS["primary"]};
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-weight: bold;
    min-width: 60px;
    text-align: center;
}}

QPushButton:hover {{
    background-color: {COLORS["primary_hover"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["bg_input"]};
}}

QPushButton:disabled {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text_dim"]};
}}

/* Tool Buttons (Icons) */
QToolButton {{
    background-color: transparent;
    border-radius: 4px;
    padding: 4px;
}}

QToolButton:hover {{
    background-color: {COLORS["bg_input"]};
}}

/* --- LABELS & TEXT --- */
QLabel {{
    color: {COLORS["text_main"]};
    padding: 2px;
}}

QLabel:disabled {{
    color: {COLORS["text_dim"]};
}}

/* --- INPUTS & TABLES --- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS["bg_input"]};
    color: {COLORS["text_main"]};
    border: 1px solid {COLORS["bg_dark"]};
    border-radius: 3px;
    padding: 4px;
    selection-background-color: {COLORS["primary"]};
}}

QLineEdit:focus, QTextEdit:focus {{
    border: 1px solid {COLORS["primary"]};
}}

QTableWidget {{
    background-color: {COLORS["bg_panel"]};
    gridline-color: {COLORS["bg_input"]};
    border: none;
}}

QTableWidget::item {{
    padding: 5px;
}}

QTableWidget::item:selected {{
    background-color: {COLORS["primary"]};
    color: white;
}}

QHeaderView::section {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_main"]};
    padding: 5px;
    border: none;
    font-weight: bold;
}}

/* --- SCROLLBARS --- */
QScrollBar:vertical {{
    background: {COLORS["bg_dark"]};
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS["bg_input"]};
    min-height: 20px;
    border-radius: 5px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* --- GROUP BOX --- */
QGroupBox {{
    border: 1px solid {COLORS["bg_input"]};
    border-radius: 5px;
    margin-top: 20px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: {COLORS["primary"]};
}}
"""

def apply_professional_theme():
    """Apply the modern dark theme to the application."""
    app = QtWidgets.QApplication.instance()

    # 1. Set Style Strategy
    app.setStyle("Fusion")

    # 2. Apply QSS
    app.setStyleSheet(MODERN_STYLESHEET)

    # 3. Set Global Font (Professional Engineering look)
    font = QtGui.QFont("Segoe UI", 8)
    font.setStyleHint(QtGui.QFont.System)
    app.setFont(font)

    # 4. Set Palette (Fallback for some widgets that don't fully respect QSS)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(COLORS["bg_dark"]))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(COLORS["text_main"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(COLORS["bg_panel"]))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(COLORS["bg_dark"]))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(COLORS["text_main"]))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(COLORS["text_main"]))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(COLORS["text_main"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(COLORS["bg_panel"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(COLORS["text_main"]))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(COLORS["danger"]))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(COLORS["primary"]))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(COLORS["primary"]))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)

    # 5. Set up automatic text eliding for existing widgets
    # This will be called after the main window is created
    QtCore.QTimer.singleShot(100, lambda: _setup_app_text_eliding(app))


def _setup_app_text_eliding(app):
    """
    Set up automatic text eliding for all widgets in the application.
    This adds tooltips to buttons and labels with long text.
    """
    if not app:
        return

    try:
        # Find all top-level widgets (windows)
        for widget in app.topLevelWidgets():
            _setup_widget_text_eliding(widget)
    except RuntimeError:
        # App is being destroyed, skip
        pass


def _setup_widget_text_eliding(widget):
    """
    Recursively set up text eliding for a widget and all its children.
    """
    if not widget:
        return

    try:
        # Set up eliding for this widget
        if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QLabel, QtWidgets.QCheckBox, QtGui.QAction)):
            _add_tooltip_for_long_text(widget)

        # Recursively process children
        for child in widget.children():
            if isinstance(child, QtWidgets.QWidget):
                _setup_widget_text_eliding(child)
    except RuntimeError:
        # Widget has been deleted, skip it
        pass


def _add_tooltip_for_long_text(widget):
    """
    Add tooltip for widgets with long text that might get truncated.
    """
    try:
        if hasattr(widget, 'text') and callable(widget.text):
            text = widget.text()
            if text and len(text) > 15:  # Add tooltip for text longer than 15 chars
                widget.setToolTip(text)
        elif hasattr(widget, 'title') and callable(widget.title):
            # For actions
            title = widget.title()
            if title and len(title) > 15:
                widget.setToolTip(title)
    except:
        pass  # Ignore any errors in tooltip setup


def setup_text_eliding(widget):
    """
    Set up text eliding for widgets that might have long text.

    This function adds tooltips and eliding for buttons, labels, and other
    text widgets to prevent text truncation.

    Args:
        widget: The widget to set up text eliding for
    """
    if hasattr(widget, 'text') and callable(getattr(widget, 'text')):
        # For widgets with text() method (buttons, labels, etc.)
        def update_tooltip():
            text = widget.text()
            if len(text) > 20:  # If text is long, add tooltip
                widget.setToolTip(text)

        # Connect to text changes if possible
        if hasattr(widget, 'textChanged'):
            widget.textChanged.connect(update_tooltip)

        # Set initial tooltip
        update_tooltip()

    # Recursively apply to children
    for child in widget.findChildren(QtWidgets.QWidget):
        if isinstance(child, (QtWidgets.QPushButton, QtWidgets.QLabel, QtWidgets.QCheckBox)):
            setup_text_eliding(child)


def create_elided_button(text, max_width=120, parent=None):
    """
    Create a button with text eliding and tooltip for long text.

    Args:
        text: Button text
        max_width: Maximum width in pixels
        parent: Parent widget

    Returns:
        QPushButton with elided text and tooltip
    """
    button = QtWidgets.QPushButton(parent)
    button.setText(text)
    button.setMaximumWidth(max_width)

    # Add tooltip for long text
    if len(text) > 15:
        button.setToolTip(text)

    # Set size policy to allow shrinking
    button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    return button


def create_elided_label(text, max_width=None, parent=None):
    """
    Create a label with text eliding for long text.

    Args:
        text: Label text
        max_width: Maximum width in pixels (None for no limit)
        parent: Parent widget

    Returns:
        QLabel with elided text and tooltip
    """
    label = QtWidgets.QLabel(parent)
    label.setText(text)

    if max_width:
        label.setMaximumWidth(max_width)
        label.setWordWrap(False)

        # Add tooltip for long text
        if len(text) > 20:
            label.setToolTip(text)

    return label


def apply_dark_theme():
    """
    Legacy function for backward compatibility.
    Now redirects to the modern professional theme.
    """
    apply_professional_theme()






