# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Qt actions around pure system-modeling persistence and validation."""

from __future__ import annotations

import logging
from copy import deepcopy
from os import PathLike
from typing import Any

from PySide6 import QtWidgets

from pylcss.mdao.system_modeling.persistence import load_systems, save_systems
from pylcss.mdao.system_modeling.validation import validate_systems

logger = logging.getLogger(__name__)


def save_graph(widget: Any) -> None:
    product_name = widget.system_manager.product_name.text().strip() or "Product"
    path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
        widget,
        "Save Systems",
        f"{product_name}.json",
        "JSON (*.json)",
    )
    if path:
        save_graph_to_file(widget, path)


def save_graph_to_file(widget: Any, path: str | PathLike[str]) -> None:
    try:
        manager = widget.system_manager
        records = [
            {"name": system["name"], "graph": system["graph"].serialize_session()}
            for system in manager.systems
        ]
        save_systems(records, path, product_name=manager.product_name.text())
    except Exception as exc:
        _show_error(widget, "Failed to save systems", exc)
        raise
    if _is_visible_widget(widget):
        QtWidgets.QMessageBox.information(
            widget,
            "Saved",
            "Systems saved successfully.",
        )


def load_graph(widget: Any) -> None:
    path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
        widget,
        "Load Systems",
        "",
        "JSON (*.json)",
    )
    if path:
        load_graph_from_file(widget, path)


def load_graph_from_file(widget: Any, path: str | PathLike[str]) -> None:
    try:
        product_name, records = load_systems(path)
        _replace_loaded_systems(widget.system_manager, product_name, records)
    except Exception as exc:
        _show_error(widget, "Failed to load systems", exc)
        raise


def _replace_loaded_systems(
    manager: Any, product_name: str, records: list[dict[str, Any]]
) -> None:
    """Prepare every graph before replacing the currently displayed project."""

    prepared: list[tuple[str, Any]] = []
    try:
        for record in records:
            graph = manager.create_graph()
            try:
                session = deepcopy(record["graph"])
                for node in session.get("nodes", {}).values():
                    if str(node.get("type_", "")).startswith("com.pfd.custom_block"):
                        node.get("custom", {}).pop("func_name", None)
                graph.deserialize_session(session)
                manager.prepare_loaded_graph(graph)
            except Exception:
                graph.widget.deleteLater()
                raise
            prepared.append((record["name"], graph))
    except Exception:
        for _name, graph in prepared:
            graph.widget.deleteLater()
        raise
    manager.replace_systems(prepared)
    manager.product_name.setText(product_name)


def validate_graph(widget: Any) -> bool:
    """Validate current systems and show one consolidated result dialog."""

    report = validate_systems(widget.system_manager.systems)
    if report.errors:
        message = "Validation errors:\n\n" + "\n".join(
            f"• {item}" for item in report.errors
        )
        if report.warnings:
            message += "\n\nWarnings:\n" + "\n".join(
                f"• {item}" for item in report.warnings
            )
        QtWidgets.QMessageBox.warning(
            widget,
            "Validation Failed",
            message,
        )
        return False

    message = "Graph validation passed."
    if report.warnings:
        message += "\n\nWarnings:\n" + "\n".join(
            f"• {item}" for item in report.warnings
        )
    QtWidgets.QMessageBox.information(
        widget,
        "Validation Successful",
        message,
    )
    return True


def _show_error(widget: Any, action: str, error: Exception) -> None:
    if _is_visible_widget(widget):
        QtWidgets.QMessageBox.critical(widget, "Error", f"{action}: {error}")
    else:
        logger.exception("%s", action, exc_info=error)


def _is_visible_widget(widget: Any) -> bool:
    return isinstance(widget, QtWidgets.QWidget) and widget.isVisible()


__all__ = [
    "load_graph",
    "load_graph_from_file",
    "save_graph",
    "save_graph_to_file",
    "validate_graph",
]
