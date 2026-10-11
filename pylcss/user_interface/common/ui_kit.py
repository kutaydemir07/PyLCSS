# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Small shared widgets and styles for the workspace pages."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = [
    "CollapsibleSection",
    "SECONDARY_QSS",
    "SegmentedControl",
    "secondary_button",
    "segment_qss",
]

# Written for the dark theme; the theme manager translates it for light mode.
SECONDARY_QSS = (
    "QPushButton { background: transparent; color: white;"
    " border: 1px solid rgba(128, 128, 128, 140); border-radius: 4px;"
    " padding: 4px 8px; }"
    " QPushButton:hover { background: rgba(255, 255, 255, 18); }"
    " QPushButton:disabled { color: rgba(128, 128, 128, 160); }"
)


def secondary_button(button: QtWidgets.QPushButton) -> QtWidgets.QPushButton:
    """Give non-primary actions an outline look so the main action stands out."""
    button.setStyleSheet(SECONDARY_QSS)
    return button


def segment_qss(first: bool, last: bool) -> str:
    """Style for one button of a joined, exclusive button row."""
    left = "4px" if first else "0px"
    right = "4px" if last else "0px"
    return (
        "QPushButton { background: transparent; color: white;"
        " border: 1px solid rgba(128, 128, 128, 140);"
        f" border-top-left-radius: {left}; border-bottom-left-radius: {left};"
        f" border-top-right-radius: {right}; border-bottom-right-radius: {right};"
        + ("" if first else " border-left-width: 0px;")
        + " padding: 5px 8px; }"
        " QPushButton:hover { background: rgba(255, 255, 255, 18); }"
        " QPushButton:checked { background: #1e5aab; color: white;"
        " border-color: #1e5aab; }"
    )


class SegmentedControl(QtWidgets.QWidget):
    """Joined exclusive buttons; ``index_changed`` fires on user clicks only."""

    index_changed = QtCore.Signal(int)

    def __init__(self, labels) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        labels = list(labels)
        for index, label in enumerate(labels):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setStyleSheet(segment_qss(index == 0, index == len(labels) - 1))
            self._group.addButton(button, index)
            layout.addWidget(button)
        self._group.idClicked.connect(self.index_changed)

    def index(self) -> int:
        return self._group.checkedId()

    def set_index(self, index) -> None:
        """Select a segment, or pass None to clear the selection."""
        self._group.setExclusive(False)
        for button in self._group.buttons():
            button.setChecked(False)
        if index is not None:
            self._group.button(index).setChecked(True)
        self._group.setExclusive(True)


class CollapsibleSection(QtWidgets.QWidget):
    """Workflow step: state mark + title header, a one-line summary when collapsed, and a body."""

    _MARKS = {
        "done": ("\u2713", "#2e8b57"),
        "current": ("\u25cf", "#4f6f91"),
        "todo": ("\u25cb", "gray"),
    }

    def __init__(self, title: str) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(4)
        self._mark = QtWidgets.QLabel()
        self._mark.setFixedWidth(16)
        self._mark.setAlignment(QtCore.Qt.AlignCenter)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; text-align: left; border: none; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        header.addWidget(self._mark)
        header.addWidget(self._toggle, 1)
        root.addLayout(header)

        self._summary = QtWidgets.QLabel()
        self._summary.setWordWrap(True)
        self._summary.setContentsMargins(22, 0, 0, 4)
        self._summary.setStyleSheet("color: gray;")
        root.addWidget(self._summary)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(22, 2, 0, 6)
        self.body_layout.setSpacing(6)
        root.addWidget(self.body)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Plain)
        divider.setStyleSheet("color: rgba(128, 128, 128, 60);")
        root.addWidget(divider)

        self.set_state("todo")
        self._on_toggled(True)

    def _on_toggled(self, expanded: bool) -> None:
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.body.setVisible(expanded)
        self._summary.setVisible(not expanded and bool(self._summary.text()))

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)
        self._summary.setToolTip(text)
        self._summary.setVisible(not self._toggle.isChecked() and bool(text))

    def set_state(self, state: str) -> None:
        glyph, color = self._MARKS[state]
        self._mark.setText(glyph)
        self._mark.setStyleSheet(f"color: {color}; font-weight: bold;")
