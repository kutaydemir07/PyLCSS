# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Shared, dependency-light UI helpers."""

from . import qt_patches as qt_patches
from .qt_patches import (
    NumericTableWidgetItem,
    install_node_removal_patch,
    install_nodegraph_patches,
)
from .text_utils import format_html
from .theme_manager import (
    COLORS,
    LIGHT_WORKSPACE_PALETTES,
    THEMES,
    apply_professional_theme,
    apply_theme,
    current_theme,
    is_dark_theme,
    retheme_node_graph,
    retheme_widget_styles,
    set_filled_button_icon,
)

__all__ = [
    "COLORS",
    "LIGHT_WORKSPACE_PALETTES",
    "THEMES",
    "NumericTableWidgetItem",
    "apply_professional_theme",
    "apply_theme",
    "current_theme",
    "format_html",
    "install_node_removal_patch",
    "install_nodegraph_patches",
    "is_dark_theme",
    "qt_patches",
    "retheme_node_graph",
    "retheme_widget_styles",
    "set_filled_button_icon",
]
