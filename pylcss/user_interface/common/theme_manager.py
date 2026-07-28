# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Application-wide dark/light theming.

The palette and global QSS are the source of truth.  A small event filter also
adapts older widget-local dark QSS when a widget/dialog is shown, which keeps
the light theme complete while those panels are progressively converted to
palette-native styling.
"""

from __future__ import annotations

import logging

import re

from PySide6 import QtCore, QtGui, QtWidgets


THEMES = {
    "dark": {
        "bg_dark": "#1e1f22",
        "bg_panel": "#2b2d31",
        "bg_input": "#383a40",
        "primary": "#d29922",
        "primary_hover": "#e3b341",
        "text_main": "#ffffff",
        "text_dim": "#b5bac1",
        "border": "#1e1f22",
        "success": "#2ecc71",
        "danger": "#ed4245",
        "plot_bg": "#ffffff",
        "plot_fg": "#24292f",
        # Engineering charts deliberately use a paper-white canvas in both
        # application themes.  It gives exported figures and on-screen plots
        # the same contrast and avoids light curves disappearing on dark
        # charcoal.
        "chart_bg": "#ffffff",
        "chart_fg": "#24292f",
        "graph_bg": "#232323",
        "graph_grid": "#2d2d2d",
        "viewer_bg": "#333333",
    },
    "light": {
        "bg_dark": "#f3f4f6",
        "bg_panel": "#ffffff",
        "bg_input": "#eef0f3",
        "primary": "#8a5b00",
        "primary_hover": "#6f4900",
        "text_main": "#1f2933",
        # Light mode keeps secondary/disabled copy dark as well.  Muted gray
        # text was repeatedly ending up on paper-white panels, especially in
        # dynamically-created inspectors, and was needlessly hard to scan.
        "text_dim": "#1f2933",
        "border": "#d0d5dd",
        "success": "#1a7f37",
        "danger": "#cf222e",
        "plot_bg": "#ffffff",
        "plot_fg": "#24292f",
        "chart_bg": "#ffffff",
        "chart_fg": "#24292f",
        "graph_bg": "#f3f4f6",
        "graph_grid": "#d0d5dd",
        "viewer_bg": "#e9edf2",
    },
}

# Kept as one mutable object because several modules import COLORS directly.
COLORS = dict(THEMES["dark"])
_current_theme = "dark"
_event_filter = None
_native_set_widget_stylesheet = QtWidgets.QWidget.setStyleSheet
_stylesheet_hook_installed = False


def current_theme() -> str:
    return _current_theme


def is_dark_theme() -> bool:
    return _current_theme == "dark"


def _build_legacy_dark_stylesheet(c):
    """Return the original PyLCSS dark stylesheet unchanged."""
    return f"""
QMainWindow, QDialog {{
    background-color: {c["bg_dark"]};
    color: {c["text_main"]};
}}
QWidget {{
    color: {c["text_main"]};
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
    font-size: 9pt;
}}
QTabWidget::pane {{
    border: 1px solid {c["bg_panel"]};
    background-color: {c["bg_panel"]};
    border-radius: 5px;
}}
QTabBar::tab {{
    background-color: {c["bg_dark"]};
    color: {c["text_dim"]};
    padding: 8px 20px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {c["bg_panel"]};
    color: {c["text_main"]};
    border-bottom: 2px solid {c["primary"]};
    font-weight: bold;
}}
QTabBar::tab:hover {{
    background-color: {c["bg_input"]};
    color: {c["text_main"]};
}}
QPushButton {{
    background-color: {c["primary"]};
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-weight: bold;
    min-width: 60px;
    text-align: center;
}}
QPushButton:hover {{ background-color: {c["primary_hover"]}; }}
QPushButton:pressed {{ background-color: {c["bg_input"]}; }}
QPushButton:disabled {{
    background-color: {c["bg_input"]};
    color: {c["text_dim"]};
}}
QToolButton {{
    background-color: transparent;
    border-radius: 4px;
    padding: 4px;
}}
QToolButton:hover {{ background-color: {c["bg_input"]}; }}
QLabel {{
    color: {c["text_main"]};
    padding: 2px;
}}
QLabel:disabled {{ color: {c["text_dim"]}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {c["bg_input"]};
    color: {c["text_main"]};
    border: 1px solid {c["bg_dark"]};
    border-radius: 3px;
    padding: 4px;
    selection-background-color: {c["primary"]};
}}
QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {c["primary"]}; }}
QTableWidget {{
    background-color: {c["bg_panel"]};
    gridline-color: {c["bg_input"]};
    border: none;
}}
QTableWidget::item {{ padding: 5px; }}
QTableWidget::item:selected {{
    background-color: {c["primary"]};
    color: white;
}}
QHeaderView::section {{
    background-color: {c["bg_dark"]};
    color: {c["text_main"]};
    padding: 5px;
    border: none;
    font-weight: bold;
}}
QScrollBar:vertical {{
    background: {c["bg_dark"]};
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {c["bg_input"]};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QGroupBox {{
    border: 1px solid {c["bg_input"]};
    border-radius: 5px;
    margin-top: 20px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: {c["primary"]};
}}
"""


def _build_stylesheet(c, theme="light"):
    # Light mode deliberately uses the exact same selectors, dimensions and
    # widget hierarchy as the original dark theme.  Only semantic colour
    # roles change, so switching themes cannot alter spacing or invent a
    # different collection of surfaces.
    return _build_legacy_dark_stylesheet(c)


# Common colors in legacy widget-local styles. Mapping is intentionally
# conservative: semantic red/green/blue accents remain unchanged.
_LIGHT_QSS_MAP = {
    "#0d1117": "#ffffff",
    "#121417": "#ffffff",
    "#14161a": "#ffffff",
    "#15171b": "#eef0f3",
    "#16181c": "#ffffff",
    "#1a1c20": "#eef0f3",
    "#181b20": "#ffffff",
    "#1c1c1c": "#f3f4f6",
    "#1c1e22": "#f3f4f6",
    "#1e1e1e": "#f3f4f6",
    "#1e1f22": "#f3f4f6",
    "#202329": "#ffffff",
    "#20242d": "#ffffff",
    "#21262d": "#eef0f3",
    "#24272d": "#eef0f3",
    "#252525": "#ffffff",
    "#25272b": "#ffffff",
    "#282c33": "#ffffff",
    "#2a2a2a": "#ffffff",
    "#2b2b2b": "#ffffff",
    "#2a2d33": "#d0d5dd",
    "#2a2e35": "#ffffff",
    "#2b2d31": "#ffffff",
    "#2c3546": "#0969da",
    "#2c3e50": "#ffffff",
    "#2f333a": "#d0d5dd",
    "#30363d": "#d0d5dd",
    "#313641": "#d0d5dd",
    "#323843": "#eef0f3",
    "#34445d": "#eef0f3",
    "#343943": "#d0d5dd",
    "#363b44": "#d0d5dd",
    "#383a40": "#eef0f3",
    "#383d46": "#d0d5dd",
    "#3a3d43": "#d0d5dd",
    "#3a3a3a": "#d0d5dd",
    "#3a3f48": "#d0d5dd",
    "#3b4658": "#0969da",
    "#3f4b61": "#d0d5dd",
    "#405572": "#d0d5dd",
    "#526783": "#d0d5dd",
    "#4a4a4a": "#d0d5dd",
    "#4a4a6a": "#d0d5dd",
    "#5a5a5a": "#d0d5dd",
    "#333": "#eef0f3",
    "#444": "#d0d5dd",
    "#555": "#d0d5dd",
    "#666": "#1f2933",
    "#777": "#1f2933",
    "#888": "#1f2933",
    "#444444": "#d0d5dd",
    "#6b7178": "#1f2933",
    "#7f8c8d": "#1f2933",
    "#808080": "#1f2933",
    "#90a4ae": "#1f2933",
    "#9aa3b2": "#1f2933",
    "#a9a9a9": "#1f2933",
    "#b0bec5": "#1f2933",
    "#b5bac1": "#1f2933",
    "#aab0b8": "#1f2933",
    "#8ea0ba": "#1f2933",
    "#8f98a5": "#1f2933",
    "#9aa0a6": "#1f2933",
    "#7f8792": "#1f2933",
    "#b9c0ca": "#1f2933",
    "#ccc": "#1f2933",
    "#cccccc": "#1f2933",
    "#c9d1d9": "#1f2933",
    "#dce2eb": "#1f2933",
    "#dce4f2": "#1f2933",
    "#d6dae0": "#1f2933",
    "#e0e0e0": "#1f2933",
    "#e1e6ed": "#1f2933",
    "#ecf0f1": "#1f2933",
    "#eef1f5": "#1f2933",
    "#edf4ff": "#1f2933",
    "#eef3ff": "#1f2933",
    "#eef2f7": "#1f2933",
    "#f2f4f7": "#1f2933",
    "#f2f5f9": "#1f2933",
    "#f8f8f2": "#1f2933",
    "#fafafa": "#1f2933",
    # Legacy dark-theme accent foregrounds are too pale on a white panel.
    "#72d38a": "#1a7f37",
    "#6dde8d": "#1a7f37",
    "#81c784": "#1a7f37",
    "#9ccc65": "#1a7f37",
    "#ffb35c": "#9a6700",
    "#f0b040": "#9a6700",
    "#ffb74d": "#9a6700",
    "#ffcc80": "#9a6700",
    "#80cbc4": "#087f8c",
    "#64b5f6": "#0969da",
    "#6fb3ff": "#0969da",
    "#82b8ef": "#0969da",
    "#aad4ff": "#0969da",
    "#bbdefb": "#0969da",
}


def _lighten_local_qss(qss: str) -> str:
    # Translate against the original text in one pass. Sequential replacement
    # avoids translating a replacement again if the mapping grows to contain
    # overlapping source and target colours.
    color_pattern = re.compile(
        "(?:"
        + "|".join(
            re.escape(color) for color in sorted(_LIGHT_QSS_MAP, key=len, reverse=True)
        )
        + r")(?![0-9a-f])",
        flags=re.IGNORECASE,
    )
    output = color_pattern.sub(
        lambda match: _LIGHT_QSS_MAP[match.group(0).lower()],
        str(qss),
    )
    rgba_map = {
        "rgba(15, 20, 35, 230)": "rgba(255, 255, 255, 245)",
        "rgba(28, 32, 40, 245)": "rgba(255, 255, 255, 245)",
        "rgba(30, 30, 30, 180)": "rgba(255, 255, 255, 220)",
        "rgba(35, 43, 58, 235)": "rgba(243, 244, 246, 245)",
        "rgba(58, 78, 110, 245)": "rgba(238, 240, 243, 245)",
        "rgba(80, 80, 80, 200)": "rgba(208, 213, 221, 230)",
        "rgba(255, 255, 255, 18)": "rgba(31, 35, 40, 18)",
        "rgba(255, 255, 255, 28)": "rgba(31, 35, 40, 28)",
    }
    rgba_pattern = re.compile(
        "|".join(re.escape(value) for value in rgba_map),
        flags=re.IGNORECASE,
    )
    output = rgba_pattern.sub(
        lambda match: rgba_map[match.group(0).lower()],
        output,
    )

    # A local dark panel often declares both a dark background and white text.
    # Once its background is translated to light, translate only that block's
    # neutral white foreground. White text on coloured action buttons remains
    # white and readable.
    def _fix_light_block(match):
        selector, body = match.group(1), match.group(2)
        has_light_background = re.search(
            r"background(?:-color)?\s*:\s*(?:"
            r"#(?:fff(?:fff)?|f3f4f6|eef0f3|ffffff)"
            r"|rgba\(\s*2(?:[0-4]\d|5[0-5])\s*,|transparent)",
            body,
            flags=re.IGNORECASE,
        )
        has_background = re.search(
            r"background(?:-color)?\s*:", body, flags=re.IGNORECASE
        )
        # Text-only local styles inherit the light parent background.  Keep
        # white text only when the same block deliberately supplies a
        # coloured (for example red/green action) background.
        if has_light_background or not has_background:
            body = re.sub(
                r"(?<![-\w])color\s*:\s*(?:white|#fff(?:fff)?)",
                "color: #1f2933",
                body,
                flags=re.IGNORECASE,
            )
        return f"{selector}{{{body}}}"

    output = re.sub(r"([^{}]+)\{([^{}]*)\}", _fix_light_block, output)
    # QWidget.setStyleSheet also accepts declaration-only strings such as
    # ``background:#333; color:white``. They have no selector block for the
    # callback above, and were the source of white status text on white panels.
    if "{" not in output and re.search(
        r"background(?:-color)?\s*:\s*#(?:fff(?:fff)?|f3f4f6|eef0f3|ffffff)",
        output,
        flags=re.IGNORECASE,
    ):
        output = re.sub(
            r"(?<![-\w])color\s*:\s*(?:white|#fff(?:fff)?)",
            "color: #1f2933",
            output,
            flags=re.IGNORECASE,
        )
    return output


def _install_stylesheet_hook() -> None:
    """Translate widget-local QSS at assignment time while light mode is active."""
    global _stylesheet_hook_installed
    if _stylesheet_hook_installed:
        return

    def _set_themed_stylesheet(widget, qss):
        source = str(qss or "")
        if not source:
            widget.setProperty("_pylcss_original_qss", None)
            widget.setProperty("_pylcss_applied_qss", None)
            return _native_set_widget_stylesheet(widget, source)

        transformed = (
            _lighten_local_qss(source) if _current_theme == "light" else source
        )
        widget.setProperty("_pylcss_original_qss", source)
        widget.setProperty("_pylcss_applied_qss", transformed)
        return _native_set_widget_stylesheet(widget, transformed)

    QtWidgets.QWidget.setStyleSheet = _set_themed_stylesheet
    _stylesheet_hook_installed = True


def retheme_widget_styles(root, theme: str | None = None) -> None:
    """Adapt legacy widget-local styles below ``root``."""
    chosen = (theme or _current_theme).lower()
    widgets = [root]
    if isinstance(root, QtWidgets.QWidget):
        widgets.extend(root.findChildren(QtWidgets.QWidget))

    for widget in widgets:
        try:
            widget.setProperty("_pylcss_theme_ready", True)
            qss = widget.styleSheet()
        except (AttributeError, RuntimeError):
            continue
        original = widget.property("_pylcss_original_qss")
        last_applied = widget.property("_pylcss_applied_qss")
        # Widgets update status/validation colours dynamically.  If the
        # current stylesheet did not come from this theme manager, treat it
        # as the new source stylesheet instead of restoring an obsolete style
        # captured the first time the widget was shown.
        if qss and (original is None or qss != last_applied):
            original = qss
            widget.setProperty("_pylcss_original_qss", original)
        if original:
            transformed = (
                _lighten_local_qss(str(original))
                if chosen == "light"
                else str(original)
            )
            if qss != transformed:
                # Record the translated value before the native write so a
                # later tree pass does not mistake it for new source QSS.
                widget.setProperty("_pylcss_applied_qss", transformed)
                _native_set_widget_stylesheet(widget, transformed)
            else:
                widget.setProperty("_pylcss_applied_qss", transformed)


def retheme_node_graph(graph, theme: str | None = None) -> None:
    """Theme NodeGraphQt nodes as well as its canvas.

    NodeGraphQt does not follow QApplication palettes.  Its default body and
    title text therefore stayed nearly black after switching to light mode.
    Preserve each node's dark semantic border/body palette and substitute a
    high-contrast paper body only while light mode is active.
    """
    chosen = (theme or _current_theme).lower()
    c = THEMES[chosen]
    try:
        dark_canvas = getattr(graph, "_pylcss_dark_canvas_palette", None)
        if dark_canvas is None:
            dark_canvas = (tuple(graph.background_color()), tuple(graph.grid_color()))
            graph._pylcss_dark_canvas_palette = dark_canvas
        if chosen == "dark":
            graph_background, graph_grid = dark_canvas
        else:
            graph_background = QtGui.QColor(c["graph_bg"]).getRgb()[:3]
            graph_grid = QtGui.QColor(c["graph_grid"]).getRgb()[:3]
        graph.set_background_color(*graph_background)
        graph.set_grid_color(*graph_grid)
        nodes = graph.all_nodes()
    except Exception:
        return

    for node in nodes:
        dark_palette = getattr(node, "_pylcss_dark_node_palette", None)
        if dark_palette is None:
            try:
                model = node.model
                candidate = (
                    tuple(model.color),
                    tuple(model.border_color),
                    tuple(model.text_color),
                )
                # If the first encounter is already in light mode, a freshly
                # deserialized/default NodeGraph node can still be black. Save
                # that as its dark palette before replacing it.
                dark_palette = candidate
                node._pylcss_dark_node_palette = candidate
            except Exception:
                continue

        if chosen == "light":
            body = (248, 250, 252, 255)
            border = dark_palette[1]
            text = (31, 35, 40, 255)
        else:
            body, border, text = dark_palette
        try:
            node.set_property("color", body, push_undo=False)
            node.set_property("border_color", border, push_undo=False)
            node.set_property("text_color", text, push_undo=False)
            node.view.draw_node()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )


class _ThemeEventFilter(QtCore.QObject):
    def eventFilter(self, watched, event):
        # Do not inspect widgets on ChildAdded/Polish. Qt can deliver those
        # events while a C++ subclass (for example QMenuBar) is still under
        # construction; asking PySide for the object at that point can
        # permanently cache only its QWidget base wrapper. QShow arrives after
        # construction and safely marks newly-created dialogs/panels.
        if event.type() == QtCore.QEvent.Show:
            if isinstance(watched, QtWidgets.QWidget):
                watched.setProperty("_pylcss_theme_ready", True)
                QtCore.QTimer.singleShot(
                    0, lambda w=watched: _retheme_visible_widget(w)
                )
        return False


def _retheme_visible_widget(widget) -> None:
    """Retheme a fully-constructed visible widget, if it still exists."""
    try:
        if isinstance(widget, QtWidgets.QWidget) and widget.isVisible():
            retheme_widget_styles(widget)
    except RuntimeError:
        pass


def apply_theme(theme: str, *, persist: bool = True) -> str:
    """Apply ``dark`` or ``light`` consistently to the QApplication."""
    global _current_theme, _event_filter
    chosen = str(theme or "").strip().lower()
    if chosen not in THEMES:
        raise ValueError("Theme must be 'dark' or 'light'.")
    app = QtWidgets.QApplication.instance()
    if app is None:
        return chosen

    _current_theme = chosen
    COLORS.clear()
    COLORS.update(THEMES[chosen])
    _install_stylesheet_hook()
    c = COLORS
    app.setStyle("Fusion")
    # Clear the previous application QSS before changing the palette.  Qt
    # otherwise resolves unstyled palette roles (menus, list viewports, combo
    # popups) against the *previous* theme and those widgets remain one switch
    # behind: dark panels in light mode, then white panels in dark mode.
    app.setStyleSheet("")

    # Keep typography and geometry identical across themes.  The original
    # application used 8 pt; changing it during a colour switch caused text
    # clipping and made the light UI feel like a different application.
    font = QtGui.QFont("Segoe UI", 8)
    font.setStyleHint(QtGui.QFont.System)
    app.setFont(font)

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(c["bg_dark"]))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(c["text_main"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(c["bg_panel"]))
    palette.setColor(
        QtGui.QPalette.AlternateBase,
        QtGui.QColor(c["bg_dark"] if chosen == "dark" else c["bg_input"]),
    )
    palette.setColor(
        QtGui.QPalette.ToolTipBase,
        QtGui.QColor(c["text_main"] if chosen == "dark" else c["bg_panel"]),
    )
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(c["text_main"]))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(c["text_main"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(c["bg_panel"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(c["text_main"]))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(c["danger"]))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(c["primary"]))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(c["primary"]))
    palette.setColor(
        QtGui.QPalette.HighlightedText,
        QtGui.QColor("#ffffff" if chosen == "dark" else "#ffffff"),
    )
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.Text,
        QtGui.QColor(c["text_dim"]),
    )
    app.setPalette(palette)
    app.setStyleSheet(_build_stylesheet(c, chosen))

    # Install a fresh filter on each explicit theme change. Embedded graphics
    # libraries can install application filters that consume widget events;
    # merely reinstalling the same QObject does not reliably restore ordering
    # on every Qt binding/platform combination.
    if _event_filter is not None:
        try:
            if _event_filter.parent() is app:
                app.removeEventFilter(_event_filter)
        except RuntimeError:
            pass
    _event_filter = _ThemeEventFilter(app)
    app.installEventFilter(_event_filter)
    if persist:
        QtCore.QSettings("PyLCSS", "PyLCSS").setValue("ui/theme", chosen)

    for widget in app.topLevelWidgets():
        retheme_widget_styles(widget, chosen)
    QtCore.QTimer.singleShot(100, lambda: _setup_app_text_eliding(app))
    return chosen


def apply_professional_theme(theme: str | None = None):
    """Apply the requested or last-used professional application theme."""
    if theme is None:
        theme = str(QtCore.QSettings("PyLCSS", "PyLCSS").value("ui/theme", "dark"))
    if theme.lower() not in THEMES:
        theme = "dark"
    return apply_theme(theme, persist=False)


def _setup_app_text_eliding(app):
    if not app:
        return
    try:
        for widget in app.topLevelWidgets():
            _setup_widget_text_eliding(widget)
    except RuntimeError:
        pass


def _setup_widget_text_eliding(widget):
    if not widget:
        return
    try:
        if isinstance(
            widget,
            (QtWidgets.QPushButton, QtWidgets.QLabel, QtWidgets.QCheckBox),
        ):
            _add_tooltip_for_long_text(widget)
        for child in widget.children():
            if isinstance(child, QtWidgets.QWidget):
                _setup_widget_text_eliding(child)
    except RuntimeError:
        pass


def _add_tooltip_for_long_text(widget):
    try:
        text = widget.text() if callable(getattr(widget, "text", None)) else ""
        if text and len(text) > 22 and not widget.toolTip():
            widget.setToolTip(text)
    except Exception:
        logging.getLogger(__name__).debug(
            "Optional UI operation failed.", exc_info=True
        )
