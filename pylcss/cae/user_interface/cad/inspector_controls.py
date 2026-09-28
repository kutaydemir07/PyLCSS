# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Small reusable controls used by the CAD property inspector."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None

__all__ = ["ExpressionEdit", "InspectorSection"]


class ExpressionEdit(QtWidgets.QLineEdit):
    """A text field that evaluates math expressions (e.g., '10/2 + 5')."""

    value_changed = QtCore.Signal(float)

    def __init__(
        self,
        value: float | str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(str(value), parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.editingFinished.connect(self._evaluate)

    def _evaluate(self) -> None:
        text = self.text()
        try:
            # Secure evaluation using simpleeval (safe math expressions)
            if simple_eval is not None:
                val = float(simple_eval(text))
            else:
                # Without the optional expression parser, accept literals only.
                val = float(text)
            self.setText(str(val))
            self.value_changed.emit(val)
        except Exception:
            # If invalid (e.g. text), keep it but don't emit
            pass


class InspectorSection(QtWidgets.QWidget):
    """Compact collapsible wrapper for one inspector property group."""

    def __init__(
        self,
        title: str,
        content: QtWidgets.QWidget,
        expanded: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._content = content
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("InspectorSectionHeader")
        self.toggle.setText(str(title))
        self.toggle.setCheckable(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.toggle.setStyleSheet(
            """
            QToolButton#InspectorSectionHeader {
                background: #24272d;
                border: 1px solid #313641;
                border-radius: 6px;
                color: #cdd2d9;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 7px;
                text-align: left;
            }
            QToolButton#InspectorSectionHeader:hover {
                background: #2a2e35;
                border-color: #4a9eff;
            }
            """
        )
        layout.addWidget(self.toggle)

        content.setTitle("")
        content.setStyleSheet(
            """
            QGroupBox {
                margin-top: 2px;
                padding: 6px;
                border: 1px solid #2f333a;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                background: #202329;
            }
            """
        )
        layout.addWidget(content)

        self.toggle.toggled.connect(self._set_expanded)
        self._set_expanded(bool(expanded))

    def _set_expanded(self, expanded):
        self._content.setVisible(bool(expanded))
        self.toggle.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )
