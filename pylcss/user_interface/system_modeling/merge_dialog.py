# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Confirmation dialog for automatic subsystem connections."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtWidgets

from pylcss.system_modeling.merge import (
    ModelMergeError,
    VariableEndpoint,
    analyze_merge,
)
from pylcss.system_modeling.types import CompiledModel


def validate_merge_connections(
    models: Sequence[CompiledModel],
    parent: QtWidgets.QWidget | None = None,
) -> bool:
    """Show the merge interface and return whether the user accepts it."""

    try:
        summary = analyze_merge(models)
    except (ModelMergeError, TypeError) as exc:
        QtWidgets.QMessageBox.critical(parent, "Cannot Merge Systems", str(exc))
        return False

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Merge Validation")
    dialog.resize(640, 480)
    layout = QtWidgets.QVBoxLayout(dialog)

    introduction = QtWidgets.QLabel(
        "Outputs and inputs with the same variable name are connected "
        "automatically. Review the resulting interface before continuing."
    )
    introduction.setWordWrap(True)
    layout.addWidget(introduction)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QVBoxLayout(content)
    if not summary.connections:
        content_layout.addWidget(QtWidgets.QLabel("No subsystem connections found."))
    for connection in summary.connections:
        group = QtWidgets.QGroupBox(connection.name)
        group_layout = QtWidgets.QFormLayout(group)
        group_layout.addRow(
            "Provided by:",
            QtWidgets.QLabel(_endpoint_text(connection.providers)),
        )
        group_layout.addRow(
            "Consumed by:",
            QtWidgets.QLabel(_endpoint_text(connection.consumers)),
        )
        if connection.unit_issue:
            issue = QtWidgets.QLabel(connection.unit_issue)
            issue.setWordWrap(True)
            issue.setStyleSheet("color: #856404; background: #fff3cd; padding: 5px;")
            group_layout.addRow("Units:", issue)
        content_layout.addWidget(group)
    content_layout.addStretch()
    scroll.setWidget(content)
    layout.addWidget(scroll)

    interface = QtWidgets.QGroupBox("Global Interface")
    interface_layout = QtWidgets.QFormLayout(interface)
    interface_layout.addRow(
        "Inputs:",
        QtWidgets.QLabel(", ".join(summary.global_inputs) or "None"),
    )
    interface_layout.addRow(
        "Outputs:",
        QtWidgets.QLabel(", ".join(summary.global_outputs) or "None"),
    )
    layout.addWidget(interface)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted


def _endpoint_text(endpoints: Sequence[VariableEndpoint]) -> str:
    return ", ".join(
        f"{endpoint.model_name} [{endpoint.unit}]" for endpoint in endpoints
    )


__all__ = ["validate_merge_connections"]
