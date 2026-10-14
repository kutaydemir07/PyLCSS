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
        "bg_panel": "#1e1f22",
        "bg_input": "#282a2e",
        "primary": "#d29922",
        "primary_hover": "#e3b341",
        "text_main": "#ffffff",
        "text_dim": "#b5bac1",
        "border": "#282a2e",
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
        "graph_bg": "#181b22",
        "graph_grid": "#334155",
        "viewer_bg": "#222732",
    },
    "light": {
        "bg_dark": "#f4f6f8",
        "bg_panel": "#ffffff",
        "bg_input": "#eef2f6",
        "primary": "#4f6f91",
        "primary_hover": "#3d5c7a",
        # Blue-gray copy is easier on the eye than near-black while retaining
        # strong contrast on white and off-white application surfaces.
        "text_main": "#344054",
        "text_dim": "#667085",
        "border": "#d5dce5",
        "success": "#3f7d5a",
        "danger": "#b54747",
        "plot_bg": "#ffffff",
        "plot_fg": "#344054",
        "chart_bg": "#ffffff",
        "chart_fg": "#344054",
        "graph_bg": "#ffffff",
        "graph_grid": "#edf1f5",
        "viewer_bg": "#f8fafc",
    },
}

# Light-mode system nodes use restrained role tints instead of one undifferentiated
# white card.  The pale surfaces keep controls readable while the border/title
# pair lets each node type remain recognizable at a glance.  Dark-mode node
# palettes continue to come from the node classes unchanged.
LIGHT_SYSTEM_NODE_PALETTES = {
    # MDAO nodes
    "com.pfd.input": (
        (244, 247, 250, 255),  # #f4f7fa
        (91, 127, 163, 255),  # #5b7fa3
        (52, 64, 84, 255),  # #344054
    ),
    "com.pfd.custom_block": (
        (247, 245, 250, 255),  # #f7f5fa
        (126, 111, 153, 255),  # #7e6f99
        (63, 58, 74, 255),  # #3f3a4a
    ),
    "com.pfd.output": (
        (244, 248, 245, 255),  # #f4f8f5
        (92, 133, 105, 255),  # #5c8569
        (52, 72, 60, 255),  # #34483c
    ),
    "com.pfd.intermediate": (
        (246, 247, 249, 255),  # #f6f7f9
        (122, 134, 153, 255),  # #7a8699
        (52, 64, 84, 255),  # #344054
    ),
    # CAE Parametric CAD & Geometry nodes (calm steel blue / slate)
    "com.cad": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.code_part": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.freecad_part": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.geometry": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.assembly": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.import_step": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    "com.cad.import_stl": (
        (245, 247, 250, 255),
        (85, 115, 150, 255),
        (50, 65, 85, 255),
    ),
    # CAE FEA & Mesh nodes (disciplined cobalt / navy blue)
    "com.cad.sim.mesh": (
        (244, 246, 252, 255),
        (75, 110, 175, 255),
        (45, 60, 90, 255),
    ),
    "com.cad.sim.material": (
        (244, 246, 252, 255),
        (75, 110, 175, 255),
        (45, 60, 90, 255),
    ),
    "com.cad.sim.component": (
        (244, 246, 252, 255),
        (75, 110, 175, 255),
        (45, 60, 90, 255),
    ),
    "com.cad.sim.solver": (
        (244, 246, 252, 255),
        (75, 110, 175, 255),
        (45, 60, 90, 255),
    ),
    # CAE Boundary Conditions & Loads (disciplined warm terracotta)
    "com.cad.sim.constraint": (
        (253, 247, 245, 255),
        (195, 95, 80, 255),
        (90, 48, 42, 255),
    ),
    "com.cad.sim.load": (
        (253, 247, 245, 255),
        (195, 95, 80, 255),
        (90, 48, 42, 255),
    ),
    "com.cad.sim.pressure_load": (
        (253, 247, 245, 255),
        (195, 95, 80, 255),
        (90, 48, 42, 255),
    ),
    "com.cad.topopt.support": (
        (253, 247, 245, 255),
        (195, 95, 80, 255),
        (90, 48, 42, 255),
    ),
    "com.cad.topopt.load": (
        (253, 247, 245, 255),
        (195, 95, 80, 255),
        (90, 48, 42, 255),
    ),
    # CAE Topology and lattice geometry (warm amber gold matching MDAO primary)
    "com.cad.sim.topopt_voxel": (
        (253, 250, 242, 255),
        (190, 140, 30, 255),
        (95, 70, 25, 255),
    ),
    "com.cad.sim.lattice_infill": (
        (253, 250, 242, 255),
        (190, 140, 30, 255),
        (95, 70, 25, 255),
    ),
    # CAE Impact & Crash Dynamics (muted crimson)
    "com.cad.sim.impact": (
        (252, 245, 245, 255),
        (180, 75, 65, 255),
        (95, 42, 38, 255),
    ),
    "com.cad.sim.crash_solver": (
        (252, 245, 245, 255),
        (180, 75, 65, 255),
        (95, 42, 38, 255),
    ),
    # CAE Parameters & Inputs (steel blue)
    "com.cad.number": (
        (244, 247, 250, 255),
        (91, 127, 163, 255),
        (52, 64, 84, 255),
    ),
    "com.cad.variable": (
        (244, 247, 250, 255),
        (91, 127, 163, 255),
        (52, 64, 84, 255),
    ),
    "com.cad.select_face": (
        (244, 247, 250, 255),
        (91, 127, 163, 255),
        (52, 64, 84, 255),
    ),
    # CAE Outputs & Exports (muted sage)
    "com.cad.export_step": (
        (244, 248, 245, 255),
        (92, 133, 105, 255),
        (52, 72, 60, 255),
    ),
    "com.cad.export_stl": (
        (244, 248, 245, 255),
        (92, 133, 105, 255),
        (52, 72, 60, 255),
    ),
    "com.cad.analysis": (
        (244, 248, 245, 255),
        (92, 133, 105, 255),
        (52, 72, 60, 255),
    ),
}

DARK_SYSTEM_NODE_PALETTES = {
    # MDAO nodes
    "com.pfd.input": ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    "com.pfd.custom_block": ((35, 35, 35, 255), (148, 107, 220, 255), (242, 238, 250, 255)),
    "com.pfd.output": ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
    "com.pfd.intermediate": ((35, 35, 35, 255), (118, 130, 150, 255), (238, 241, 246, 255)),
    # CAE Parametric CAD & Geometry nodes
    "com.cad": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.code_part": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.freecad_part": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.geometry": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.assembly": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.import_step": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    "com.cad.import_stl": ((35, 35, 35, 255), (90, 120, 160, 255), (235, 243, 250, 255)),
    # CAE FEA & Mesh nodes
    "com.cad.sim.mesh": ((35, 35, 35, 255), (88, 125, 170, 255), (238, 242, 252, 255)),
    "com.cad.sim.material": ((35, 35, 35, 255), (88, 125, 170, 255), (238, 242, 252, 255)),
    "com.cad.sim.component": ((35, 35, 35, 255), (88, 125, 170, 255), (238, 242, 252, 255)),
    "com.cad.sim.solver": ((35, 35, 35, 255), (88, 125, 170, 255), (238, 242, 252, 255)),
    # CAE Boundary Conditions & Loads
    "com.cad.sim.constraint": ((35, 35, 35, 255), (215, 110, 95, 255), (252, 244, 242, 255)),
    "com.cad.sim.load": ((35, 35, 35, 255), (215, 110, 95, 255), (252, 244, 242, 255)),
    "com.cad.sim.pressure_load": ((35, 35, 35, 255), (215, 110, 95, 255), (252, 244, 242, 255)),
    "com.cad.topopt.support": ((35, 35, 35, 255), (215, 110, 95, 255), (252, 244, 242, 255)),
    "com.cad.topopt.load": ((35, 35, 35, 255), (215, 110, 95, 255), (252, 244, 242, 255)),
    # CAE Topology and lattice geometry
    "com.cad.sim.topopt_voxel": ((35, 35, 35, 255), (210, 153, 34, 255), (252, 248, 238, 255)),
    "com.cad.sim.lattice_infill": ((35, 35, 35, 255), (210, 153, 34, 255), (252, 248, 238, 255)),
    # CAE Impact & Crash Dynamics
    "com.cad.sim.impact": ((35, 35, 35, 255), (195, 80, 75, 255), (252, 242, 242, 255)),
    "com.cad.sim.crash_solver": ((35, 35, 35, 255), (195, 80, 75, 255), (252, 242, 242, 255)),
    # CAE Parameters & Inputs
    "com.cad.number": ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    "com.cad.variable": ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    "com.cad.select_face": ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    # CAE Outputs & Exports
    "com.cad.export_step": ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
    "com.cad.export_stl": ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
    "com.cad.analysis": ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
}


def _lookup_node_palette(identifier: str, palettes_dict: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Find exact or longest-prefix match for a node identifier in a palette dictionary."""
    if not identifier:
        return None
    if identifier in palettes_dict:
        return palettes_dict[identifier]
    for key in sorted(palettes_dict.keys(), key=len, reverse=True):
        if identifier.startswith(key):
            return palettes_dict[key]
    return None

# Every main workspace uses the same quiet light palette.  Keeping the entries
# explicit preserves per-workspace object names while ensuring backgrounds,
# borders, headings, active tabs, and icons stay visually harmonious.  Dark
# mode does not use this table.
LIGHT_WORKSPACE_PALETTES = (
    {
        "object_name": "modelingWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "designStudioWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "surrogateWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "solutionSpaceWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "optimizationWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "sensitivityWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
    {
        "object_name": "helpWorkspace",
        "background": "#f4f6f8",
        "accent": "#4f6f91",
        "accent_soft": "#e8edf3",
    },
)

# Kept as one mutable object because several modules import COLORS directly.
COLORS = dict(THEMES["dark"])
_current_theme = "dark"
_event_filter = None
_native_set_widget_stylesheet = QtWidgets.QWidget.setStyleSheet
_stylesheet_hook_installed = False
_node_groupbox_hook_installed = False


def current_theme() -> str:
    return _current_theme


def is_dark_theme() -> bool:
    return _current_theme == "dark"


def set_filled_button_icon(
    button: QtWidgets.QPushButton,
    icon_name: str,
) -> None:
    """Give a filled action button an icon with surface-aware contrast.

    QtAwesome icons are rasterized with their requested foreground colour, so
    the application palette cannot turn a default dark icon white when it is
    placed on a filled button.  Remember the icon name on the button and
    rebuild it whenever the theme changes.
    """
    button.setProperty("_pylcss_filled_icon_name", icon_name)
    _apply_filled_button_icon(button, _current_theme)


def _apply_filled_button_icon(
    button: QtWidgets.QPushButton,
    theme: str,
) -> None:
    icon_name = button.property("_pylcss_filled_icon_name")
    if not icon_name:
        return
    try:
        import qtawesome as qta

        if theme == "light":
            button.setIcon(
                qta.icon(
                    str(icon_name),
                    color="#ffffff",
                    color_disabled=THEMES["light"]["text_dim"],
                )
            )
        else:
            # This is the icon treatment used by the original dark theme:
            # the default dark glyph reads clearly on its amber buttons.
            button.setIcon(qta.icon(str(icon_name)))
    except (ImportError, RuntimeError):
        logging.getLogger(__name__).debug(
            "Could not apply a themed filled-button icon.", exc_info=True
        )


def _build_legacy_dark_stylesheet(c):
    """Return the original PyLCSS dark stylesheet unchanged."""
    return f"""
QMainWindow, QDialog {{
    background-color: {c["bg_dark"]};
    color: {c["text_main"]};
}}
QWidget {{
    color: {c["text_main"]};
    font-family: "Segoe UI", "Segoe UI Variable Text", "Inter", "Roboto", -apple-system, sans-serif;
    font-size: 9.5pt;
}}

QTabWidget::pane {{
    border: 1px solid {c["border"]};
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
    border: 1px solid {c["border"]};
    border-radius: 3px;
    padding: 4px;
    selection-background-color: {c["primary"]};
}}
QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {c["primary"]}; }}
QTableWidget {{
    background-color: {c["bg_panel"]};
    gridline-color: {c["border"]};
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
    border: 1px solid {c["border"]};
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
    "#282a2e": "#eef0f3",
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
    # Inspector body text. Without these the panel kept its dark-theme pale
    # greys on a white background and was close to unreadable.
    "#cdd2d9": "#1f2933",
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
    "#4a9eff": "#0969da",
    "#64b5f6": "#0969da",
    "#6fb3ff": "#0969da",
    # Pale coral error text needs a readable red on white.
    "#f08080": "#b42318",
    "#82b8ef": "#0969da",
    "#aad4ff": "#0969da",
    "#bbdefb": "#0969da",
}

# Older panels still describe the previous light neutrals as translation
# targets.  Normalize those targets to the current harmonious palette in one
# place so every legacy inspector, dialog, and workbench receives the same
# surface, text, border, and action colors.
_LIGHT_SEMANTIC_TARGETS = {
    "#f3f4f6": THEMES["light"]["bg_dark"],
    "#eef0f3": THEMES["light"]["bg_input"],
    "#0969da": THEMES["light"]["primary"],
    "#0550ae": THEMES["light"]["primary_hover"],
    "#1f2933": THEMES["light"]["text_main"],
    "#d0d5dd": THEMES["light"]["border"],
    "#1a7f37": THEMES["light"]["success"],
    "#cf222e": THEMES["light"]["danger"],
}
_LIGHT_QSS_MAP = {
    source: _LIGHT_SEMANTIC_TARGETS.get(target.lower(), target)
    for source, target in _LIGHT_QSS_MAP.items()
}
_LIGHT_QSS_MAP.update(
    {
        # Common vivid legacy action colors are intentionally muted in light
        # mode; their semantic meaning remains available through icon/label
        # text and button placement.
        "#1e5aab": THEMES["light"]["primary"],
        "#2673cc": THEMES["light"]["primary_hover"],
        "#1e5ab4": THEMES["light"]["primary"],
        "#2470d8": THEMES["light"]["primary_hover"],
        "#5865f2": THEMES["light"]["primary"],
        "#4752c4": THEMES["light"]["primary_hover"],
        "#27ae60": THEMES["light"]["success"],
        "#2ecc71": THEMES["light"]["success"],
        "#4a7a4a": THEMES["light"]["success"],
        "#5a9a5a": THEMES["light"]["success"],
        "#c0392b": THEMES["light"]["danger"],
        "#e74c3c": THEMES["light"]["danger"],
        "#ff4444": THEMES["light"]["danger"],
    }
)


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
        "rgba(35, 43, 58, 235)": "rgba(244, 246, 248, 245)",
        "rgba(58, 78, 110, 245)": "rgba(238, 242, 246, 245)",
        "rgba(80, 80, 80, 200)": "rgba(213, 220, 229, 230)",
        "rgba(255, 255, 255, 18)": "rgba(52, 64, 84, 18)",
        "rgba(255, 255, 255, 28)": "rgba(52, 64, 84, 28)",
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
            r"#(?:fff(?:fff)?|f3f4f6|eef0f3|f4f6f8|eef2f6|ffffff)"
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
                f"color: {THEMES['light']['text_main']}",
                body,
                flags=re.IGNORECASE,
            )
        return f"{selector}{{{body}}}"

    output = re.sub(r"([^{}]+)\{([^{}]*)\}", _fix_light_block, output)
    # QWidget.setStyleSheet also accepts declaration-only strings such as
    # ``background:#333; color:white``. They have no selector block for the
    # callback above, and were the source of white status text on white panels.
    if "{" not in output and re.search(
        r"background(?:-color)?\s*:\s*#(?:"
        r"fff(?:fff)?|f3f4f6|eef0f3|f4f6f8|eef2f6|ffffff)",
        output,
        flags=re.IGNORECASE,
    ):
        output = re.sub(
            r"(?<![-\w])color\s*:\s*(?:white|#fff(?:fff)?)",
            f"color: {THEMES['light']['text_main']}",
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
        if isinstance(widget, QtWidgets.QPushButton):
            _apply_filled_button_icon(widget, chosen)


# NodeGraphQt derives every embedded control from its process-wide dark viewer
# palette and writes those colours into widget-local stylesheets.  Appending
# only a foreground override is not enough: it produces dark text on the
# library's original charcoal buttons in light mode.  This complete light
# override is appended after NodeGraphQt's generated rules, using the same
# selector specificity so it wins the cascade without changing node geometry.
def _node_widget_style_override(theme: str) -> str:
    if theme == "light":
        return """
QGroupBox, _NodeGroupBox {
  background: transparent;
  background-color: transparent;
  border: none;
  margin-top: 15px;
  margin-bottom: 2px;
  padding: 0px;
  color: #0f172a;
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
}
QGroupBox::title, _NodeGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top center;
  background: transparent;
  color: #475569;
  font-size: 8pt;
  font-weight: 600;
  padding: 0px;
}
QLabel {
  background: transparent;
  color: #475569;
  font-size: 8.5pt;
  font-weight: 500;
}
QCheckBox {
  background: transparent;
  color: #1e293b;
  font-size: 8.5pt;
  font-weight: 500;
  spacing: 6px;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractItemView {
  background-color: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 4px;
  color: #0f172a;
  font-size: 8.5pt;
  padding: 3px 6px;
  selection-background-color: #3b82f6;
  selection-color: #ffffff;
}
QComboBox {
  padding-right: 20px;
}
QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 18px;
  border: none;
}
QComboBox::down-arrow {
  image: none;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 5px solid #64748b;
  width: 0px;
  height: 0px;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
  border: 1.5px solid #2563eb;
  background-color: #ffffff;
}
QLineEdit:read-only {
  background-color: #f8fafc;
  color: #475569;
  border: 1px solid #e2e8f0;
}
QPushButton {
  background-color: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 4px;
  color: #1e293b;
  font-size: 8.5pt;
  font-weight: 500;
  padding: 4px 8px;
}
QPushButton:hover {
  background-color: #f1f5f9;
  border-color: #94a3b8;
  color: #0f172a;
}
QPushButton:pressed {
  background-color: #e2e8f0;
}
"""
    else:
        return """
QGroupBox, _NodeGroupBox {
  background: transparent;
  background-color: transparent;
  border: none;
  margin-top: 15px;
  margin-bottom: 2px;
  padding: 0px;
  color: #f0f6fc;
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
}
QGroupBox::title, _NodeGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top center;
  background: transparent;
  color: #94a3b8;
  font-size: 8pt;
  font-weight: 600;
  padding: 0px;
}
QLabel {
  background: transparent;
  color: #cbd5e1;
  font-size: 8.5pt;
  font-weight: 500;
}
QCheckBox {
  background: transparent;
  color: #f1f5f9;
  font-size: 8.5pt;
  font-weight: 500;
  spacing: 6px;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractItemView {
  background-color: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  color: #f0f6fc;
  font-size: 8.5pt;
  padding: 3px 6px;
  selection-background-color: #1f6feb;
  selection-color: #ffffff;
}
QComboBox {
  padding-right: 20px;
}
QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 18px;
  border: none;
}
QComboBox::down-arrow {
  image: none;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 5px solid #94a3b8;
  width: 0px;
  height: 0px;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
  border: 1.5px solid #58a6ff;
  background-color: #0d1117;
}
QLineEdit:read-only {
  background-color: #0d1117;
  color: #8b949e;
  border: 1px solid #21262d;
}
QPushButton {
  background-color: #21262d;
  border: 1px solid #30363d;
  border-radius: 4px;
  color: #c9d1d9;
  font-size: 8.5pt;
  font-weight: 500;
  padding: 4px 8px;
}
QPushButton:hover {
  background-color: #30363d;
  border-color: #8b949e;
  color: #ffffff;
}
QPushButton:pressed {
  background-color: #161b22;
}
"""



def _harmonized_light_node_border(border) -> tuple[int, int, int, int]:
    """Desaturate arbitrary Design Studio node borders into the light system."""
    source = tuple(border)
    base = (122, 134, 153)
    rgb = tuple(
        max(72, min(184, round(channel * 0.35 + neutral * 0.65)))
        for channel, neutral in zip(source[:3], base)
    )
    return (*rgb, 255)


def _install_node_groupbox_hook() -> None:
    """Keep the label above each embedded widget readable across redraws.

    ``NodeItem.align_widgets`` calls ``_NodeGroupBox.setTitleAlign`` on every
    redraw, and that method rebuilds the whole stylesheet from the dark
    ``ViewerEnum.BACKGROUND_COLOR``.  A one-shot restyle is therefore lost as
    soon as the node is moved, resized, or re-laid out, so the override has to
    be re-applied at the point NodeGraphQt regenerates the rules.
    """
    global _node_groupbox_hook_installed
    if _node_groupbox_hook_installed:
        return
    try:
        from NodeGraphQt.widgets.node_widgets import _NodeGroupBox
    except Exception:
        logging.getLogger(__name__).debug(
            "NodeGraphQt group-box styling hook unavailable.", exc_info=True
        )
        return

    native_set_title_align = _NodeGroupBox.setTitleAlign

    def _themed_set_title_align(box, align="center"):
        native_set_title_align(box, align)
        override = _node_widget_style_override(_current_theme)
        if not override:
            return
        try:
            _native_set_widget_stylesheet(
                box, str(box.styleSheet() or "") + override
            )
        except RuntimeError:
            pass

    _NodeGroupBox.setTitleAlign = _themed_set_title_align
    _node_groupbox_hook_installed = True


def _retheme_node_embedded_widgets(node, theme: str) -> None:
    """Restore readable text in a node's embedded editors for *theme*."""
    try:
        embedded = node.widgets()
    except Exception:
        return
    if not embedded:
        return
    override = _node_widget_style_override(theme)
    for proxy in list(embedded.values()):
        try:
            root = proxy.widget()
        except Exception:
            continue
        if root is None:
            continue
        try:
            targets = [root, *root.findChildren(QtWidgets.QWidget)]
        except RuntimeError:
            continue
        for widget in targets:
            try:
                # Capture NodeGraphQt's generated sheet once, before any
                # override is appended, so switching themes repeatedly cannot
                # stack overrides or lose the original rules.
                original = widget.property("_pylcss_node_widget_qss")
                if original is None:
                    original = widget.styleSheet() or ""
                    widget.setProperty("_pylcss_node_widget_qss", original)
                widget.setStyleSheet(str(original) + override)
            except RuntimeError:
                break
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )


def retheme_node_graph(graph, theme: str | None = None) -> None:
    """Theme NodeGraphQt nodes as well as its canvas.

    NodeGraphQt does not follow QApplication palettes.  Its default body and
    title text therefore stayed nearly black after switching to light mode.
    Preserve each node's dark semantic border/body palette and substitute a
    high-contrast paper body only while light mode is active.
    """
    chosen = (theme or _current_theme).lower()
    c = THEMES[chosen]
    _install_node_groupbox_hook()
    # NodeGraphQt paints selected-node outlines from a process-wide enum
    # constant instead of the node/canvas palette. Without updating it, light
    # mode retained the library's hard-coded amber outline even though every
    # other accent had changed to blue.
    try:
        from NodeGraphQt.constants import NodeEnum

        accent = QtGui.QColor(c["primary"]).getRgb()
        NodeEnum.SELECTED_BORDER_COLOR._value_ = accent
        NodeEnum.SELECTED_COLOR._value_ = (*accent[:3], 30)
    except Exception:
        logging.getLogger(__name__).debug(
            "NodeGraphQt selection accent is unavailable.", exc_info=True
        )
    try:
        dark_canvas = getattr(graph, "_pylcss_dark_canvas_palette", None)
        if dark_canvas is None:
            dark_canvas = (
                QtGui.QColor(THEMES["dark"]["graph_bg"]).getRgb()[:3],
                QtGui.QColor(THEMES["dark"]["graph_grid"]).getRgb()[:3],
            )
            graph._pylcss_dark_canvas_palette = dark_canvas
        if chosen == "dark":
            graph_background, graph_grid = dark_canvas
        else:
            graph_background = QtGui.QColor(c["graph_bg"]).getRgb()[:3]
            graph_grid = QtGui.QColor(c["graph_grid"]).getRgb()[:3]
        graph.set_background_color(*graph_background)
        graph.set_grid_color(*graph_grid)
        try:
            from NodeGraphQt.constants import ViewerEnum
            graph.set_grid_mode(ViewerEnum.GRID_DISPLAY_LINES.value)
        except Exception:
            pass
        nodes = graph.all_nodes()
    except Exception:
        return

    for node in nodes:
        identifier = (
            getattr(node, "__identifier__", "")
            or getattr(node, "type_", "")
            or (node.__class__.__name__ if hasattr(node, "__class__") else "")
        )
        system_dark = _lookup_node_palette(identifier, DARK_SYSTEM_NODE_PALETTES)
        dark_palette = getattr(node, "_pylcss_dark_node_palette", None)
        if dark_palette is None:
            if system_dark is not None:
                dark_palette = system_dark
            else:
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
                except Exception:
                    continue
            node._pylcss_dark_node_palette = dark_palette

        if chosen == "light":
            light_palette = _lookup_node_palette(identifier, LIGHT_SYSTEM_NODE_PALETTES)
            if light_palette is None:
                body = (247, 248, 250, 255)
                border = _harmonized_light_node_border(dark_palette[1])
                text = (52, 64, 84, 255)
            else:
                body, border, text = light_palette
        else:
            body, border, text = system_dark or dark_palette
        try:
            try:
                node.set_property("color", body, push_undo=False)
                node.set_property("border_color", border, push_undo=False)
                node.set_property("text_color", text, push_undo=False)
            except TypeError:
                node.set_property("color", body)
                node.set_property("border_color", border)
                node.set_property("text_color", text)
            _retheme_node_embedded_widgets(node, chosen)
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
            apply_fn = getattr(widget, "apply_theme", None)
            if callable(apply_fn):
                try:
                    apply_fn(_current_theme)
                except Exception:
                    pass
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
    font = QtGui.QFont("Segoe UI", 10)
    font.setStyleHint(QtGui.QFont.SansSerif)
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
        apply_fn = getattr(widget, "apply_theme", None)
        if callable(apply_fn):
            try:
                apply_fn(chosen)
            except Exception:
                pass
    QtCore.QTimer.singleShot(100, lambda: _setup_app_text_eliding(app))
    return chosen


def apply_professional_theme(theme: str | None = None):
    """Apply the requested or last-used application theme."""
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
