# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""CadQuery code-editor dialog used by the Design Studio inspector."""

from __future__ import annotations

import logging

from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.system_modeling.system_node_types import (
    CodeEditor as _CodeEditor,
)

logger = logging.getLogger(__name__)

__all__ = ["CadCodeEditorDialog"]


class CadCodeEditorDialog(QtWidgets.QDialog):
    """Full-screen CadQuery script editor for ``CadQueryCodeNode``.

    Same UX as the function-block editor in the system-modeling tab:
    Python syntax highlighting, line numbers, double-click-to-insert sidebar.
    Sidebar lists the node's exposed parameter names (so the user can drop
    them into the script without retyping) plus a small CadQuery cheat-sheet
    of the most common building blocks.
    """

    # (display, snippet, tooltip).  Snippet is inserted at cursor on double-click.
    _CHEATSHEET = (
        ("— Primitives —", None, None),
        (
            "cq.Workplane('XY')",
            "cq.Workplane('XY')",
            "Start a workplane on the XY plane (X-right, Y-up, Z-out).",
        ),
        (
            ".box(L, W, H)",
            ".box(L, W, H)",
            "Centered rectangular block — L along X, W along Y, H along Z.",
        ),
        (
            ".circle(R).extrude(H)",
            ".circle(R).extrude(H)",
            "Circle of radius R, extruded by H along the workplane normal.",
        ),
        (
            ".sphere(R)",
            ".sphere(R)",
            "Sphere of radius R centred at the workplane origin.",
        ),
        (
            ".cylinder(H, R)",
            ".cylinder(H, R)",
            "Cylinder along Z with height H and radius R, centred at origin.",
        ),
        ("— Sketch & extrude —", None, None),
        (
            ".polyline([(x,y),…]).close().extrude(H)",
            ".polyline([(0,0),(1,0),(1,1),(0,1)]).close().extrude(H)",
            "Sketch a closed polygon from 2-D points, then extrude by H.",
        ),
        (
            ".workplane(offset=z)",
            ".workplane(offset=z)",
            "Move the workplane along its normal by `z` (useful for stacking layers).",
        ),
        ("— Modifications —", None, None),
        (
            ".faces('>Z').workplane().hole(d)",
            ".faces('>Z').workplane().hole(d)",
            "Drill a through-hole of diameter `d` from the top face.",
        ),
        (
            ".edges('|Z').fillet(r)",
            ".edges('|Z').fillet(r)",
            "Round vertical edges with radius `r`.",
        ),
        (
            ".edges('|Z').chamfer(c)",
            ".edges('|Z').chamfer(c)",
            "Chamfer vertical edges by `c`.",
        ),
        (
            ".shell(-t)",
            ".faces('>Z').shell(-t)",
            "Hollow the solid leaving wall thickness `t` (negative = inward).",
        ),
        (".translate((x,y,z))", ".translate((x, y, z))", "Move by (x, y, z)."),
        (
            ".rotate((0,0,0),(0,0,1), deg)",
            ".rotate((0,0,0), (0,0,1), deg)",
            "Rotate `deg` degrees about the Z-axis through the origin.",
        ),
        ("— Boolean / Compose —", None, None),
        (".union(other)", ".union(other)", "Boolean union with another shape."),
        (".cut(other)", ".cut(other)", "Boolean subtract."),
        (".intersect(other)", ".intersect(other)", "Boolean intersection."),
        (
            "cq.Assembly()",
            "asm = cq.Assembly()\nasm.add(part, name='name')\nresult = asm",
            "Build a multi-part assembly — no boolean unions, every child is\n"
            "addressable downstream.",
        ),
        ("— Result —", None, None),
        (
            "result = …",
            "result = ",
            "The node looks for `result`, then `shape`, then `assembly` in the\n"
            "evaluated namespace.  Assign a CadQuery Workplane, Shape, or Assembly.",
        ),
    )

    def __init__(self, code: str, node=None, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle("CAD Code Editor")
        self.resize(1200, 720)
        self.showMaximized()

        main_layout = QtWidgets.QHBoxLayout(self)

        # ── Left: editor + buttons ─────────────────────────────────────
        editor_panel = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(editor_panel)
        self.editor = _CodeEditor([])
        self.editor.setPlainText(code or "")
        ev.addWidget(self.editor)

        btn_row = QtWidgets.QHBoxLayout()
        help_btn = QtWidgets.QPushButton("?")
        help_btn.setFixedSize(30, 30)
        help_btn.setToolTip("Open the CadQuery cheat-sheet")
        help_btn.clicked.connect(self._show_help)
        btn_row.addWidget(help_btn)
        find_btn = QtWidgets.QPushButton("Find / Replace")
        find_btn.clicked.connect(self._show_find_replace)
        btn_row.addWidget(find_btn)
        btn_row.addStretch()
        ok_cancel = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        ok_cancel.accepted.connect(self.accept)
        ok_cancel.rejected.connect(self.reject)
        btn_row.addWidget(ok_cancel)
        ev.addLayout(btn_row)

        # ── Right: sidebar ─────────────────────────────────────────────
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(320)
        sv = QtWidgets.QVBoxLayout(sidebar)

        sv.addWidget(QtWidgets.QLabel("<b>Available Parameters:</b>"))
        self.params_list = QtWidgets.QListWidget()
        self.params_list.setToolTip("Double-click to insert a parameter name")
        sv.addWidget(self.params_list)
        self.params_list.itemDoubleClicked.connect(self._insert_param)

        sv.addWidget(QtWidgets.QLabel("<b>CadQuery cheat-sheet:</b>"))
        self.cheat_list = QtWidgets.QListWidget()
        self.cheat_list.setToolTip("Double-click to insert a snippet")
        self._populate_cheat_sheet(self.cheat_list)
        sv.addWidget(self.cheat_list)
        self.cheat_list.itemDoubleClicked.connect(self._insert_cheat)

        if node is not None:
            self._refresh_params()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(editor_panel)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        main_layout.addWidget(splitter)

    # ──────────────────────────────────────────────────────────────
    def _refresh_params(self):
        self.params_list.clear()
        names = []
        for i in range(1, 7):
            try:
                name = (self.node.get_property(f"param_{i}_name") or "").strip()
            except Exception:
                name = ""
            if name:
                names.append(name)
                self.params_list.addItem(name)
        self.editor.update_variables(names)

    def _populate_cheat_sheet(self, list_widget: QtWidgets.QListWidget) -> None:
        header_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags)
        header_font = list_widget.font()
        header_font.setBold(True)
        for display, snippet, tooltip in self._CHEATSHEET:
            item = QtWidgets.QListWidgetItem(display)
            if snippet is None:
                item.setFlags(header_flags)
                item.setFont(header_font)
            else:
                item.setData(QtCore.Qt.UserRole, snippet)
                if tooltip:
                    item.setToolTip(tooltip)
            list_widget.addItem(item)

    def _insert_param(self, item: QtWidgets.QListWidgetItem) -> None:
        self.editor.insertPlainText(item.text())
        self.editor.setFocus()

    def _insert_cheat(self, item: QtWidgets.QListWidgetItem) -> None:
        snippet = item.data(QtCore.Qt.UserRole)
        if not snippet:
            return
        self.editor.insertPlainText(snippet)
        self.editor.setFocus()

    # ──────────────────────────────────────────────────────────────
    def _show_help(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "CadQuery quick reference",
            "# CAD CODE EDITOR\n"
            "# ===============\n"
            "# \n"
            "# Set `result = <CadQuery shape or Assembly>` somewhere in the\n"
            "# script.  The node then exposes that on its 'shape' output.\n"
            "# \n"
            "# Available in the namespace:\n"
            "#   cq, math, np, params\n"
            "#   plus your 6 parameter names (e.g. L, W, H, …)\n"
            "# \n"
            "# Inside helper functions, capture parameters as default args:\n"
            "#     def make_part(L=L, W=W):   # <-- default-arg capture\n"
            "#         return cq.Workplane('XY').box(L, W, 5)\n"
            "# Top-level free-variable lookup is not visible to inner\n"
            "# functions because the node runs under exec() with separate\n"
            "# locals/globals dicts.\n"
            "# \n"
            "# Assembly approach (recommended for multi-part models):\n"
            "#     asm = cq.Assembly()\n"
            "#     asm.add(part_a, name='a')\n"
            "#     asm.add(part_b, name='b')\n"
            "#     result = asm\n"
            "# \n"
            "# Sidebar:\n"
            "#   - 'Available Parameters' — double-click to insert a param name.\n"
            "#   - 'CadQuery cheat-sheet' — double-click to insert a snippet.\n",
        )

    def _show_find_replace(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find & Replace")
        dlg.resize(400, 150)
        layout = QtWidgets.QVBoxLayout(dlg)

        find_row = QtWidgets.QHBoxLayout()
        find_row.addWidget(QtWidgets.QLabel("Find:"))
        find_edit = QtWidgets.QLineEdit()
        find_row.addWidget(find_edit)
        layout.addLayout(find_row)

        repl_row = QtWidgets.QHBoxLayout()
        repl_row.addWidget(QtWidgets.QLabel("Replace:"))
        repl_edit = QtWidgets.QLineEdit()
        repl_row.addWidget(repl_edit)
        layout.addLayout(repl_row)

        btns = QtWidgets.QHBoxLayout()
        find_btn = QtWidgets.QPushButton("Find")
        replace_btn = QtWidgets.QPushButton("Replace")
        replace_all_btn = QtWidgets.QPushButton("Replace All")
        btns.addWidget(find_btn)
        btns.addWidget(replace_btn)
        btns.addWidget(replace_all_btn)
        layout.addLayout(btns)

        def do_find():
            txt = find_edit.text()
            if not txt:
                return
            if not self.editor.find(txt):
                cursor = self.editor.textCursor()
                cursor.movePosition(QtGui.QTextCursor.Start)
                self.editor.setTextCursor(cursor)
                self.editor.find(txt)

        def do_replace():
            txt = find_edit.text()
            new = repl_edit.text()
            cursor = self.editor.textCursor()
            if txt and cursor.hasSelection():
                cursor.insertText(new)
                do_find()

        def do_replace_all():
            txt = find_edit.text()
            new = repl_edit.text()
            if not txt:
                return
            content = self.editor.toPlainText().replace(txt, new)
            self.editor.setPlainText(content)

        find_btn.clicked.connect(do_find)
        replace_btn.clicked.connect(do_replace)
        replace_all_btn.clicked.connect(do_replace_all)
        dlg.exec()

    def get_code(self) -> str:
        return self.editor.toPlainText()
