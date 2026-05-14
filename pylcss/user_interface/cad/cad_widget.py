# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Engineering design studio - full-featured Simulink-like interface.

Features:
- Advanced node library (Sketches, Constraints, Assembly, Analysis, Simulation)
- Multi-view workspace (3D view, properties, timeline, library)
- Undo/Redo system
- File management (Save/Load projects)
- Real-time property editing
- Export capabilities (STEP, STL, PDF)
"""
import sys
import os
import json
import tempfile
import time
from datetime import datetime
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import cadquery as cq
from .cad_viewer import CQ3DViewer
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag
from NodeGraphQt import NodeGraph
from pylcss.cad.engine import execute_graph
from pylcss.cad.node_library import NODE_CLASS_MAPPING, NODE_NAME_MAPPING

try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None  # Fallback if not installed

# Import all node types
from pylcss.cad.nodes import NODE_REGISTRY, NumberNode, ExportStepNode, ExportStlNode
from pylcss.cad.nodes.modeling import InteractiveSelectFaceNode

# Reuse the Python-aware editor widget from the system-modeling tab so the
# CAD code editor matches the look-and-feel users already know.
from pylcss.user_interface.system_modeling.system_node_types import CodeEditor as _CodeEditor


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
        ("cq.Workplane('XY')",
         "cq.Workplane('XY')",
         "Start a workplane on the XY plane (X-right, Y-up, Z-out)."),
        (".box(L, W, H)",
         ".box(L, W, H)",
         "Centered rectangular block — L along X, W along Y, H along Z."),
        (".circle(R).extrude(H)",
         ".circle(R).extrude(H)",
         "Circle of radius R, extruded by H along the workplane normal."),
        (".sphere(R)",
         ".sphere(R)",
         "Sphere of radius R centred at the workplane origin."),
        (".cylinder(H, R)",
         ".cylinder(H, R)",
         "Cylinder along Z with height H and radius R, centred at origin."),
        ("— Sketch & extrude —", None, None),
        (".polyline([(x,y),…]).close().extrude(H)",
         ".polyline([(0,0),(1,0),(1,1),(0,1)]).close().extrude(H)",
         "Sketch a closed polygon from 2-D points, then extrude by H."),
        (".workplane(offset=z)",
         ".workplane(offset=z)",
         "Move the workplane along its normal by `z` (useful for stacking layers)."),
        ("— Modifications —", None, None),
        (".faces('>Z').workplane().hole(d)",
         ".faces('>Z').workplane().hole(d)",
         "Drill a through-hole of diameter `d` from the top face."),
        (".edges('|Z').fillet(r)",
         ".edges('|Z').fillet(r)",
         "Round vertical edges with radius `r`."),
        (".edges('|Z').chamfer(c)",
         ".edges('|Z').chamfer(c)",
         "Chamfer vertical edges by `c`."),
        (".shell(-t)",
         ".faces('>Z').shell(-t)",
         "Hollow the solid leaving wall thickness `t` (negative = inward)."),
        (".translate((x,y,z))",
         ".translate((x, y, z))",
         "Move by (x, y, z)."),
        (".rotate((0,0,0),(0,0,1), deg)",
         ".rotate((0,0,0), (0,0,1), deg)",
         "Rotate `deg` degrees about the Z-axis through the origin."),
        ("— Boolean / Compose —", None, None),
        (".union(other)",
         ".union(other)",
         "Boolean union with another shape."),
        (".cut(other)",
         ".cut(other)",
         "Boolean subtract."),
        (".intersect(other)",
         ".intersect(other)",
         "Boolean intersection."),
        ("cq.Assembly()",
         "asm = cq.Assembly()\nasm.add(part, name='name')\nresult = asm",
         "Build a multi-part assembly — no boolean unions, every child is\n"
         "addressable downstream."),
        ("— Result —", None, None),
        ("result = …",
         "result = ",
         "The node looks for `result`, then `shape`, then `assembly` in the\n"
         "evaluated namespace.  Assign a CadQuery Workplane, Shape, or Assembly."),
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
        self.editor.setPlainText(code or '')
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
                name = (self.node.get_property(f'param_{i}_name') or '').strip()
            except Exception:
                name = ''
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
                item.setForeground(QtGui.QColor("#9aa0a6"))
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
            self, "CadQuery quick reference",
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
            "#   - 'CadQuery cheat-sheet' — double-click to insert a snippet.\n"
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


class GraphExecutionWorker(QtCore.QThread):
    """Background worker to run the node graph without freezing the UI."""
    computation_finished = QtCore.Signal(object)  # Emits results dict
    computation_error = QtCore.Signal(str)
    optimization_step = QtCore.Signal(object, object, int, int) # mesh, densities, step, total

    def __init__(self, nodes, skip_simulation=False, parent=None):
        super().__init__(parent)
        self.nodes = nodes
        self.skip_simulation = skip_simulation
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            from pylcss.cad.engine import execute_graph
            
            # Callback for real-time updates
            def progress_cb(mesh, densities, step, total):
                if self._is_running:
                    self.optimization_step.emit(mesh, densities, step, total)

            # Pass skip_simulation and callback to engine
            results = execute_graph(
                self.nodes, 
                skip_simulation=self.skip_simulation,
                progress_callback=progress_cb
            )

            self.computation_finished.emit(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.computation_error.emit(str(e))
        finally:
            self._is_running = False


class ExpressionEdit(QtWidgets.QLineEdit):

    """A text field that evaluates math expressions (e.g., '10/2 + 5')."""
    value_changed = QtCore.Signal(float)

    def __init__(self, value, parent=None):
        super().__init__(str(value), parent)
        self.editingFinished.connect(self._evaluate)

    def _evaluate(self):
        text = self.text()
        try:
            # Secure evaluation using simpleeval (safe math expressions)
            if simple_eval is not None:
                val = float(simple_eval(text))
            else:
                # Fallback to restricted eval if simpleeval not available
                val = float(eval(text, {"__builtins__": None}, {}))
            self.setText(str(val))
            self.value_changed.emit(val)
        except Exception:
            # If invalid (e.g. text), keep it but don't emit
            pass
        
class PropertiesPanel(QtWidgets.QWidget):
    """Inspector Panel: Specialized UI for editing node properties."""
    property_changed = QtCore.Signal(object, str, object, object)
    
    def __init__(self):
        super(PropertiesPanel, self).__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.current_node = None
        self.property_widgets = {}
        self._updating_property = False  # guard against feedback loop
        
        # Title
        title = QtWidgets.QLabel("INSPECTOR")
        title.setStyleSheet("font-weight: 900; font-size: 14px; letter-spacing: 1px; color: #E0E0E0;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(title)
        
        # Separator
        self._add_separator()
        
        # Properties area (scrollable)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.props_widget = QtWidgets.QWidget()
        self.props_layout = QtWidgets.QVBoxLayout(self.props_widget)
        self.props_layout.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(self.props_widget)
        self.layout.addWidget(scroll)
        
    def _add_separator(self):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        self.layout.addWidget(sep)
    
    def display_node(self, node):
        """Display specialized inspector for a selected node."""
        self.current_node = node
        
        # Clear previous UI
        while self.props_layout.count():
            item = self.props_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.property_widgets.clear()
        
        if node is None:
            lbl = QtWidgets.QLabel("No Selection")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("color: #666; font-style: italic;")
            self.props_layout.addWidget(lbl)
            return
        
        # Node Header
        node_name = node.name() if callable(node.name) else node.name
        header = QtWidgets.QLabel(f"{node_name}")
        header.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 5px;")
        header.setAlignment(QtCore.Qt.AlignCenter)
        self.props_layout.addWidget(header)
        
        sub_header = QtWidgets.QLabel(f"{node.__identifier__.split('.')[-1].upper()}")
        sub_header.setStyleSheet("color: #888; font-size: 10px; font-weight: bold; margin-bottom: 15px;")
        sub_header.setAlignment(QtCore.Qt.AlignCenter)
        self.props_layout.addWidget(sub_header)
        
        # Route: specialized builders based on node class
        node_class = node.__class__.__name__
        if node_class == 'TopologyOptimizationNode':
            self._build_topopt_ui(node)
        elif node_class == 'CadQueryCodeNode':
            self._build_code_part_ui(node)
        elif node_class == 'InteractiveSelectFaceNode':
            self._build_interactive_select_ui(node)
        elif node_class == 'SelectFaceNode':
            self._build_select_face_ui(node)
        elif node_class in ('ConstraintNode', 'LoadNode', 'PressureLoadNode'):
            self._build_fea_bc_ui(node)
        else:
            self._build_generic_ui(node)

    def _build_code_part_ui(self, node):
        """Inspector for the code-based parametric geometry node.

        The full CadQuery script lives in a separate dialog (opened by the
        ``Edit Code…`` button below, or by double-clicking the node on the
        graph).  Keeping it out of the inspector lets the script grow
        without squeezing everything else into a postage stamp.
        """
        # ── Code-edit launcher ───────────────────────────────────────
        code_group = QtWidgets.QGroupBox("CadQuery Code")
        code_layout = QtWidgets.QVBoxLayout()
        hint = QtWidgets.QLabel(
            "Double-click the node on the graph — or click the button below — "
            "to open the CAD code editor."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        code_layout.addWidget(hint)
        btn_edit = QtWidgets.QPushButton("Edit Code…")
        btn_edit.setStyleSheet(
            "QPushButton {"
            "  background: #1e5aab; color: white; border-radius: 4px;"
            "  padding: 7px 12px; font-weight: bold;"
            "}"
            "QPushButton:hover { background: #2673cc; }"
        )
        btn_edit.clicked.connect(lambda _checked=False, n=node: self._open_cad_code_editor(n))
        code_layout.addWidget(btn_edit)
        btn_preview = QtWidgets.QPushButton("Preview in 3D")
        btn_preview.setToolTip(
            "Run the graph (CAD only — skips FEA/crash) and render this part."
        )
        btn_preview.clicked.connect(
            lambda _checked=False, n=node: self._preview_cad_part(n)
        )
        code_layout.addWidget(btn_preview)
        code_group.setLayout(code_layout)
        self.props_layout.addWidget(code_group)

        # ── Parameters (small, named scalars that flow into the script) ──
        param_group = QtWidgets.QGroupBox("Parameters")
        param_group.setToolTip(
            "Up to six named scalars.  Each becomes a top-level variable inside\n"
            "the CadQuery script.  Set 'exposed_name' on Number/Variable nodes\n"
            "upstream to drive these via cad.fea(...) from the sysmod tab."
        )
        param_layout = QtWidgets.QFormLayout()
        for idx in range(1, 7):
            name_prop = f'param_{idx}_name'
            value_prop = f'param_{idx}_value'
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            name_edit = QtWidgets.QLineEdit(str(node.get_property(name_prop) or ''))
            name_edit.setPlaceholderText("name")
            value_edit = ExpressionEdit(node.get_property(value_prop) or 0.0)
            name_edit.editingFinished.connect(
                lambda n=name_prop, w=name_edit: self.update_property(n, w.text())
            )
            value_edit.value_changed.connect(
                lambda v, n=value_prop: self.update_property(n, v)
            )
            row_layout.addWidget(name_edit, 1)
            row_layout.addWidget(value_edit, 1)
            param_layout.addRow(f"P{idx}:", row)
        param_group.setLayout(param_layout)
        self.props_layout.addWidget(param_group)

        # ── Extra parameters (free-form dict / k=v lines) ────────────
        extra_group = QtWidgets.QGroupBox("Extra Parameters")
        extra_group.setToolTip(
            "Free-form parameters in 'name=value' lines (one per line) or a\n"
            "Python dict.  These override the 6 numbered slots if they collide."
        )
        extra_layout = QtWidgets.QVBoxLayout()
        extra_editor = QtWidgets.QPlainTextEdit(str(node.get_property('parameters') or ''))
        extra_editor.setPlaceholderText("name=value lines or a Python dict")
        mono = QtGui.QFont("Consolas")
        mono.setStyleHint(QtGui.QFont.Monospace)
        extra_editor.setFont(mono)
        extra_editor.setMinimumHeight(90)
        extra_editor.focusOutEvent = (
            lambda ev, w=extra_editor, _orig=extra_editor.focusOutEvent:
                (self.update_property('parameters', w.toPlainText()), _orig(ev))[-1]
        )
        extra_layout.addWidget(extra_editor)
        extra_group.setLayout(extra_layout)
        self.props_layout.addWidget(extra_group)

    # ── helpers shared with the double-click handler ────────────────
    def _open_cad_code_editor(self, node) -> None:
        """Open the full-screen CadQuery script editor for ``node``."""
        current = str(node.get_property('code') or '')
        dlg = CadCodeEditorDialog(current, node=node, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            new_code = dlg.get_code()
            if new_code != current:
                self.update_property('code', new_code)

    def _preview_cad_part(self, node) -> None:
        app = self._get_main_app()
        if app is None:
            return
        # Force re-execution of this node next run.
        try:
            setattr(node, '_dirty', True)
            setattr(node, '_force_execute', True)
        except Exception:
            pass
        app._last_rendered_node = node
        try:
            app._execute_graph(skip_simulation=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Preview failed",
                f"Code part couldn't be evaluated:\n\n{exc}"
            )
            
    def _build_topopt_ui(self, node):
        """Custom UI for Topology Optimization."""
        
        # 1. OPTIMIZATION GOAL
        group = QtWidgets.QGroupBox("Optimization Goal")
        layout = QtWidgets.QVBoxLayout()
        
        # Volume Fraction Slider
        vf_val = float(node.get_property('vol_frac'))
        lbl_vf = QtWidgets.QLabel(f"Keep Volume: {int(vf_val*100)}%")
        slider_vf = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_vf.setRange(1, 99)
        slider_vf.setValue(int(vf_val * 100))
        
        def update_vf(val):
            lbl_vf.setText(f"Keep Volume: {val}%")
            self.update_property('vol_frac', val / 100.0)
            
        slider_vf.valueChanged.connect(update_vf)
        layout.addWidget(lbl_vf)
        layout.addWidget(slider_vf)
        group.setLayout(layout)
        self.props_layout.addWidget(group)
        
        # 2. SOLVER SETTINGS
        group_s = QtWidgets.QGroupBox("Solver Settings")
        layout_s = QtWidgets.QFormLayout()
        
        # Iterations
        spin_iter = QtWidgets.QSpinBox()
        spin_iter.setRange(1, 1000)
        spin_iter.setValue(int(node.get_property('iterations')))
        spin_iter.valueChanged.connect(lambda v: self.update_property('iterations', v))
        layout_s.addRow("Iterations:", spin_iter)
        
        # Penalization (SIMP)
        spin_penal = QtWidgets.QDoubleSpinBox()
        spin_penal.setRange(1.0, 10.0)
        spin_penal.setSingleStep(0.1)
        spin_penal.setValue(float(node.get_property('penal')))
        spin_penal.valueChanged.connect(lambda v: self.update_property('penal', v))
        layout_s.addRow("Penalization (p):", spin_penal)
        
        # Min Density
        spin_min = QtWidgets.QDoubleSpinBox()
        spin_min.setRange(0.0001, 0.1)
        spin_min.setDecimals(4)
        spin_min.setSingleStep(0.001)
        spin_min.setValue(float(node.get_property('min_density')))
        spin_min.valueChanged.connect(lambda v: self.update_property('min_density', v))
        layout_s.addRow("Min Density:", spin_min)
        
        group_s.setLayout(layout_s)
        self.props_layout.addWidget(group_s)
        
        # 3. CONSTRAINTS (Symmetry)
        group_c = QtWidgets.QGroupBox("Symmetry Constraints")
        layout_c = QtWidgets.QGridLayout()
        
        sym_axes = [('X', 'symmetry_x'), ('Y', 'symmetry_y'), ('Z', 'symmetry_z')]
        for i, (axis, prop) in enumerate(sym_axes):
            val = node.get_property(prop)
            chk = QtWidgets.QCheckBox(f"{axis}-Plane")
            chk.setChecked(val is not None)
            
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDisabled(val is None)
            if val is not None: spin.setValue(float(val))
            
            def on_check(state, p=prop, s=spin):
                s.setEnabled(state)
                new_val = s.value() if state else None
                self.update_property(p, new_val)
                
            def on_spin(v, p=prop):
                self.update_property(p, v)
                
            chk.stateChanged.connect(on_check)
            spin.valueChanged.connect(on_spin)
            
            layout_c.addWidget(chk, i, 0)
            layout_c.addWidget(spin, i, 1)
            
        group_c.setLayout(layout_c)
        self.props_layout.addWidget(group_c)
        
        # 4. VISUALIZATION
        group_v = QtWidgets.QGroupBox("Visualization")
        layout_v = QtWidgets.QFormLayout()
        
        combo_vis = QtWidgets.QComboBox()
        combo_vis.addItems(['Density', 'Recovered Shape', 'Von Mises Stress'])
        combo_vis.setCurrentText(str(node.get_property('visualization')))
        combo_vis.currentTextChanged.connect(lambda v: self.update_property('visualization', v))
        layout_v.addRow("Mode:", combo_vis)
        
        # Density Cutoff
        cutoff_val = float(node.get_property('density_cutoff'))
        lbl_cut = QtWidgets.QLabel(f"Cutoff: {cutoff_val:.2f}")
        slider_cut = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_cut.setRange(0, 100)
        slider_cut.setValue(int(cutoff_val * 100))
        
        def update_cut(val):
            fval = val / 100.0
            lbl_cut.setText(f"Cutoff: {fval:.2f}")
            self.update_property('density_cutoff', fval)
            
        slider_cut.valueChanged.connect(update_cut)
        layout_v.addRow(lbl_cut, slider_cut)
        
        group_v.setLayout(layout_v)
        self.props_layout.addWidget(group_v)
        
        # 5. ADVANCED SETTINGS (NEW)
        group_adv = QtWidgets.QGroupBox("Advanced Settings")
        layout_adv = QtWidgets.QFormLayout()
        
        # Filter Radius
        spin_filter = QtWidgets.QDoubleSpinBox()
        spin_filter.setRange(0.0, 50.0)
        spin_filter.setDecimals(2)
        spin_filter.setSingleStep(0.5)
        spin_filter.setValue(float(node.get_property('filter_radius') or 1.5))
        spin_filter.valueChanged.connect(lambda v: self.update_property('filter_radius', v))
        layout_adv.addRow("Filter Radius:", spin_filter)
        
        # Convergence Tol
        spin_tol = QtWidgets.QDoubleSpinBox()
        spin_tol.setRange(0.001, 1.0)
        spin_tol.setDecimals(3)
        spin_tol.setSingleStep(0.005)
        spin_tol.setValue(float(node.get_property('convergence_tol') or 0.01))
        spin_tol.valueChanged.connect(lambda v: self.update_property('convergence_tol', v))
        layout_adv.addRow("Convergence Tol:", spin_tol)
        
        # Move Limit
        spin_move = QtWidgets.QDoubleSpinBox()
        spin_move.setRange(0.01, 1.0)
        spin_move.setDecimals(2)
        spin_move.setSingleStep(0.05)
        spin_move.setValue(float(node.get_property('move_limit') or 0.2))
        spin_move.valueChanged.connect(lambda v: self.update_property('move_limit', v))
        layout_adv.addRow("Move Limit:", spin_move)
        
        # Update Scheme
        combo_scheme = QtWidgets.QComboBox()
        combo_scheme.addItems(['MMA', 'OC'])
        combo_scheme.setCurrentText(str(node.get_property('update_scheme') or 'MMA'))
        combo_scheme.currentTextChanged.connect(lambda v: self.update_property('update_scheme', v))
        layout_adv.addRow("Update Scheme:", combo_scheme)
        
        # NOTE: 'Element Type' combo removed — CalculiX uses C3D4 (linear tet)
        # always now that the in-house skfem path is gone.  The property is
        # still declared on the node for backward-compat with old .cad files.

        # Projection
        combo_proj = QtWidgets.QComboBox()
        combo_proj.addItems(['None', 'Heaviside'])
        combo_proj.setCurrentText(str(node.get_property('projection') or 'Heaviside'))
        combo_proj.currentTextChanged.connect(lambda v: self.update_property('projection', v))
        layout_adv.addRow("Projection:", combo_proj)
        
        # Filter Type
        combo_filter = QtWidgets.QComboBox()
        combo_filter.addItems(['density', 'sensitivity'])
        combo_filter.setCurrentText(str(node.get_property('filter_type') or 'density'))
        combo_filter.currentTextChanged.connect(lambda v: self.update_property('filter_type', v))
        layout_adv.addRow("Filter Type:", combo_filter)
        
        # Recovery Resolution
        spin_res = QtWidgets.QSpinBox()
        spin_res.setRange(10, 500)
        spin_res.setSingleStep(10)
        spin_res.setValue(int(node.get_property('recovery_resolution') or 50))
        spin_res.valueChanged.connect(lambda v: self.update_property('recovery_resolution', v))
        layout_adv.addRow("Resolution:", spin_res)
        
        # Smoothing Iterations
        spin_smooth = QtWidgets.QSpinBox()
        spin_smooth.setRange(0, 50)
        spin_smooth.setValue(int(node.get_property('smoothing_iterations') or 3))
        spin_smooth.valueChanged.connect(lambda v: self.update_property('smoothing_iterations', v))
        layout_adv.addRow("Smoothing Set:", spin_smooth)
        
        group_adv.setLayout(layout_adv)
        self.props_layout.addWidget(group_adv)

        # 6. EXPORT RECOVERED SHAPE
        group_exp = QtWidgets.QGroupBox("Export Recovered Shape")
        layout_exp = QtWidgets.QVBoxLayout()

        btn_export_stl = QtWidgets.QPushButton("Export to STL")
        btn_export_stl.setToolTip("Export the topology-optimised recovered shape as an STL file")
        btn_export_stl.clicked.connect(lambda: self._export_topopt_stl(node))
        layout_exp.addWidget(btn_export_stl)

        btn_export_obj = QtWidgets.QPushButton("Export to OBJ")
        btn_export_obj.setToolTip("Export the topology-optimised recovered shape as a Wavefront OBJ")
        btn_export_obj.clicked.connect(lambda: self._export_topopt_obj(node))
        layout_exp.addWidget(btn_export_obj)

        group_exp.setLayout(layout_exp)
        self.props_layout.addWidget(group_exp)

    # ──────────────────────────────────────────────────
    # TopOpt direct mesh export (vertices / faces)
    # ──────────────────────────────────────────────────
    def _export_topopt_stl(self, node):
        """Export recovered shape from topology optimisation as binary STL."""
        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict) or 'recovered_shape' not in result or result['recovered_shape'] is None:
            QtWidgets.QMessageBox.warning(self, "No Shape",
                "Run topology optimisation first — no recovered shape available.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)")
        if not path:
            return
        try:
            import numpy as np
            from stl import mesh as stl_mesh   # numpy-stl
            verts = result['recovered_shape']['vertices']
            faces = result['recovered_shape']['faces']
            stl_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    stl_obj.vectors[i][j] = verts[f[j]]
            stl_obj.save(path)
            if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                self.window().statusBar().showMessage(f"\u2713 Exported {len(faces)} triangles to {path}")
        except ImportError:
            # Fallback: raw binary STL without numpy-stl
            self._write_binary_stl(path, result['recovered_shape'])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _export_topopt_obj(self, node):
        """Export recovered shape as Wavefront OBJ."""
        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict) or 'recovered_shape' not in result or result['recovered_shape'] is None:
            QtWidgets.QMessageBox.warning(self, "No Shape",
                "Run topology optimisation first — no recovered shape available.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export OBJ", "", "OBJ Files (*.obj)")
        if not path:
            return
        try:
            verts = result['recovered_shape']['vertices']
            faces = result['recovered_shape']['faces']
            with open(path, 'w') as f:
                f.write("# PyLCSS TopOpt recovered shape\n")
                for v in verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                self.window().statusBar().showMessage(f"\u2713 Exported {len(faces)} faces to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _write_binary_stl(self, path, shape_data):
        """Write binary STL without numpy-stl dependency."""
        import struct
        import numpy as np
        verts = shape_data['vertices']
        faces = shape_data['faces']
        with open(path, 'wb') as f:
            f.write(b'\x00' * 80)  # header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                # compute normal
                e1 = np.array(v1) - np.array(v0)
                e2 = np.array(v2) - np.array(v0)
                n = np.cross(e1, e2)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n = n / norm
                f.write(struct.pack('<3f', *n))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))
        if hasattr(self.window(), 'statusBar') and self.window().statusBar():
            self.window().statusBar().showMessage(f"\u2713 Exported {len(faces)} triangles to {path}")

    # NOTE: Five legacy node-class-specific builders used to live here
    # (_build_primitive_ui / _build_simulation_ui / _build_operation_ui /
    # _build_modification_ui / _build_transform_ui).  They were never reached:
    # display_node only dispatches to TopologyOptimizationNode, CadQueryCodeNode,
    # InteractiveSelectFaceNode, SelectFaceNode, and the FEA-BC trio; every
    # other node falls through to _build_generic_ui (which renders correctly
    # via the sectioned, items-aware widget loop).  Removed to keep this file
    # focused on the UI that's actually displayed.

    # Ordered list of (section title, prefix list) — first match wins.
    # Properties that match no section land in "General".
    _PROPERTY_SECTIONS = [
        ("External Solver", ("external_", "openradioss_", "calculix_", "run_external", "deck_only", "solver_backend",
                              "deck_path", "engine_path", "engine_executable_path", "starter_path",
                              "work_dir", "timeout_s")),
        ("Visualization",   ("visualization", "deformation_scale", "disp_scale", "n_frames")),
        ("Solver",          ("end_time", "time_steps", "damping", "enable_", "contact_", "mass_scaling", "iterations",
                             "convergence_tol", "move_limit", "min_density", "penal", "filter_radius",
                             "update_scheme", "filter_type", "projection", "heaviside_", "continuation",
                             "element_type", "shape_recovery", "recovery_resolution", "smoothing_iterations",
                             "density_cutoff", "vol_frac", "symmetry_")),
        ("Material",        ("preset", "E", "nu", "rho", "density", "poissons_ratio", "yield_strength",
                             "tangent_modulus", "failure_strain", "enable_fracture")),
        ("Mesh",            ("mesh_type", "element_size", "max_size", "min_size", "order")),
        ("Impact",          ("velocity_", "application_scope", "node_tolerance", "impactor_mass_kg")),
        ("Geometry",        ("box_", "length", "width", "depth", "height", "radius", "thickness",
                             "near_", "selector_type", "tag", "range_expr", "direction")),
        ("Load",            ("load_type", "force_", "vector", "magnitude", "pressure", "gravity_",
                             "accel")),
        ("Constraint",      ("constraint_type", "fixed_dofs", "displacement_")),
    ]
    # Hide these unless they have a meaningful value (truthy non-empty).
    _PROPERTY_HIDE_IF_EMPTY = ("condition", "range_expr", "tag")

    # Hide these *always* — they exist on the node only so projects saved
    # before the CalculiX-only / OpenRadioss-only cut keep deserialising.  The
    # node code ignores them; showing them in the inspector just confuses the
    # user with knobs that look meaningful but do nothing.
    _PROPERTY_HIDE_ALWAYS = frozenset({
        # Solver-backend selectors collapsed to a single backend.
        'solver_backend', 'run_external_solver',
        # In-house topology-opt element choice (CalculiX uses C3D4 always).
        'element_type',
        # In-house crash-solver tuning knobs (OpenRadioss has its own).  The
        # `time_steps` property remains visible because the OpenRadioss path
        # uses it as the mass-scaling cycle target (end_time / time_steps).
        'damping_alpha',
        'enable_corotation', 'enable_contact',
        'contact_stiffness', 'contact_thickness', 'contact_update_interval',
        'mass_scaling_threshold',
        # Duplicate of the on-canvas value_input text field — kept for
        # NodeGraphQt back-compat but redundant in the inspector.
        'value',
        # Carried on legacy nodes but not honoured by the new solver paths:
        # 'min_safety_factor' is computed by the optimiser, not set by user;
        # 'fixed_faces' was a JSON-string condition list — the new ShapeOpt
        #   only supports SelectFaceNode-driven constraints (logged warning);
        # 'moment_x/y/z' on LoadNode were never wired to either CalculiX
        #   *CLOAD or OpenRadioss; only force_x/y/z are exported.
        'min_safety_factor',
        'fixed_faces',
        'moment_x', 'moment_y', 'moment_z',
    })

    # Tooltips for properties that aren't self-explanatory.  Looked up by
    # exact property name in the generic-UI builder.  Plain English so a
    # mechanical engineer reading the inspector knows what to change.
    _PROPERTY_TOOLTIPS = {
        'analysis_type':
            "Linear           — fastest, valid for small deflections and elastic only.\n"
            "Nonlinear (Geometric) — large rotations / deflections (NLGEOM).\n"
            "Nonlinear (Plastic)   — plastic yielding too (requires yield_strength on Material).",
        'analysis_mode': "Same as analysis_type — legacy alias.",
        'visualization': "Field used to colour the result mesh in the 3-D viewer.",
        'deformation_scale':
            "Multiplier on the displayed displacement (visual only). "
            "'Auto' scales so the peak motion is ~5 % of the bounding box.",
        'disp_scale': "Multiplier on the displayed displacement (visual only).",
        'n_frames': "Number of animation frames recorded for playback.",
        'end_time': "Simulation duration. For crash: milliseconds.",
        'time_steps':
            "Crash only: when mass scaling is enabled, the OpenRadioss target\n"
            "time step is end_time / time_steps.  Higher = less added mass but\n"
            "a slower run; lower = faster but more artificial inertia.",
        'enable_mass_scaling':
            "Hold the explicit time step at end_time / time_steps by adding mass to\n"
            "nodes whose CFL bound is below that value (Radioss /DT/NODA/CST).\n"
            "Prevents the 'estimated remaining time' from drifting upward during\n"
            "the run.  Slight artificial inertia is added — turn off for inertia-\n"
            "sensitive impact studies.",
        'deck_only':
            "Write the solver deck (.inp for CalculiX / .k+.rad for OpenRadioss)\n"
            "but do NOT launch the solver — useful for inspecting the deck.",
        'external_solver_path':  "Override the auto-discovered CalculiX ccx binary path.",
        'external_work_dir':     "Solver working directory.  Empty = a temp dir is created per run.",
        'external_timeout_s':    "Wall-clock timeout for the external solver run (seconds).",
        'openradioss_starter_path': "Override the auto-discovered OpenRadioss starter binary.",
        'openradioss_engine_path':  "Override the auto-discovered OpenRadioss engine binary.",
        'preset':           "Pick a material from the built-in database, or 'Custom' to set fields manually.",
        'youngs_modulus':   "Young's modulus E (MPa in the standard mm/t/s unit system).",
        'poissons_ratio':   "Poisson's ratio ν (typical 0.27–0.34 for metals).",
        'density':          "Mass density ρ (tonne/mm³ — 7.85e-9 for steel).",
        'yield_strength':   "Initial yield stress σ_y (MPa).  Non-zero triggers *PLASTIC in CalculiX.",
        'tangent_modulus':  "Bilinear hardening slope after yield (MPa).  Set to 0 for perfectly plastic.",
        'failure_strain':   "Equivalent plastic strain at element deletion (crash only).",
        'exposed_name':
            "Name this Number/Variable becomes when the .cad file is called from\n"
            "the system-modeling tab via cad.fea(...)/cad.crash(...)/cad.topopt(...).\n"
            "Empty = not exposed; for VariableNode it defaults to 'variable_name'.",

        # ── SizeOptimization ───────────────────────────────────────────
        'parameters':
            "JSON list of upstream-shape property names to optimise, e.g.\n"
            '    ["wall_thickness", "fillet_r"]',
        'bounds':
            "JSON dict of (min, max) per parameter, e.g.\n"
            '    {"wall_thickness": [1.0, 20.0], "fillet_r": [0.5, 5.0]}',
        'optimizer':
            "SciPy optimiser. COBYLA is the safest gradient-free choice for 1–5\n"
            "parameters when meshing topology may change between evaluations.\n"
            "SLSQP / L-BFGS-B / trust-constr use finite-difference gradients —\n"
            "each step costs (n_params+1) full CAD→mesh→CalculiX evaluations.",
        'gradient_step':
            "FD step size for the gradient-based optimisers (SLSQP / L-BFGS-B /\n"
            "trust-constr).  Ignored by COBYLA and Nelder-Mead.",
        'max_iterations': "Outer iteration cap on the optimiser.",
        'tolerance':      "Convergence tolerance for the optimiser.",
        'element_size':   "Target element size [mm] used when re-meshing each evaluation.",
        'max_stress':     "Stress constraint [MPa] — used when chosen optimiser supports inequality constraints.",
        'max_volume':     "Volume constraint [mm³].  Zero disables the constraint.",
        'objective':
            "What the optimiser minimises.  For SizeOpt: Min Weight = volume,\n"
            "Min Compliance = u·f, Min Max Stress = peak Von Mises.",

        # ── ShapeOptimization ──────────────────────────────────────────
        'sensitivity_method':
            "Biological Stress Leveling: heuristic — moves boundary inward at\n"
            "low-stress nodes, outward at high-stress nodes.  Reduces stress\n"
            "concentrations.\n"
            "Adjoint Compliance: rigorous shape derivative −2·W(u).  Drives the\n"
            "boundary toward uniform strain-energy density (minimum compliance).",
        'volume_preservation':
            "If checked, rescale boundary motion each iteration so total volume\n"
            "stays equal to the initial volume.  Recommended for compliance opt.",
        'step_size':        "Per-iteration boundary motion scale (mm).",
        'smoothing_weight': "Laplacian sensitivity smoothing — higher = smoother boundary updates.",
        'max_displacement': "Cap on per-node motion across the whole optimisation [mm].",
        'convergence_tol':  "Relative-objective change below which the optimiser stops.",

        # ── TopologyOptimization (advanced) ────────────────────────────
        'penal':             "SIMP penalisation p.  Higher = sharper 0/1 split (3.0 is standard).",
        'min_density':       "Ersatz minimum density ρ_min.  Prevents singular stiffness.",
        'move_limit':        "Max change in ρ_e per iteration (0.2 = ±20 %).",
        'filter_radius':     "Density / sensitivity filter radius [mm] — controls minimum feature size.",
        'filter_type':       "density: physical density filter (preferred).\nsensitivity: heuristic sensitivity smoothing.",
        'projection':        "Heaviside projection sharpens grey densities into crisp 0/1 designs.",
        'heaviside_beta':    "Heaviside sharpness β.  Higher = crisper, but harder to converge.",
        'heaviside_eta':     "Heaviside threshold η (default 0.5).",
        'continuation':      "Gradually ramp β from 1 → heaviside_beta over the run.",
        'update_scheme':     "MMA: gradient-based, robust.  OC: simpler optimality-criteria update.",
        'simp_bins':         "Number of discrete moduli the CalculiX deck quantises into per iteration.",
        'shape_recovery':    "Run marching-cubes shape recovery with Taubin smoothing on the final density field.",
        'recovery_resolution': "Marching-cubes voxel-grid resolution for recovered shape extraction.",
        'smoothing_iterations': "Gaussian smoothing passes on the recovered shape.",
        'density_cutoff':    "Density threshold below which material is removed in the recovered shape.",
        'vol_frac':          "Target volume fraction (kept / total) — the SIMP constraint.",
        'symmetry_x':        "Mirror-plane X coordinate [mm].  None = no X-symmetry.",
        'symmetry_y':        "Mirror-plane Y coordinate [mm].  None = no Y-symmetry.",
        'symmetry_z':        "Mirror-plane Z coordinate [mm].  None = no Z-symmetry.",
        'iterations':        "Outer iteration cap on the SIMP loop.",

        # ── Mesh / Remesh / Impact ─────────────────────────────────────
        'mesh_type':         "Tet:  linear C3D4 (fast).  Tet10: quadratic C3D10 (more accurate, ~4× slower).",
        'refinement_size':   "Local element size at refinement zones [mm].  0 = no local refinement.",
        'close_holes':       "Cap small holes during remesh — helps surface watertightness.",
        'repair_surface':    "Run topological repair before remeshing.",
        'mesh_quality':      "Target mesh-quality factor for remeshing.",
        'velocity_x':        "Initial impactor velocity along X [mm/ms = m/s].",
        'velocity_y':        "Initial impactor velocity along Y [mm/ms = m/s].",
        'velocity_z':        "Initial impactor velocity along Z [mm/ms = m/s].",
        'application_scope':
            "Impact Face: a moving rigid wall/impactor drives the selected\n"
            "face; connected constraints stay active (fixed-rear crush tests).\n"
            "Moving Body: the whole mesh receives velocity and hits a generated\n"
            "rigid wall; connected constraints are intentionally ignored.",
        'node_tolerance':    "Distance [mm] within which a mesh node is treated as belonging to the impact face.",
        'impactor_mass_kg':
            "Optional sled/impactor mass [kg].  In Impact Face scope this is\n"
            "the moving rigid wall mass.  In Moving Body scope it is lumped on\n"
            "the trailing edge of the projectile.",
        'enable_fracture':   "Delete elements whose plastic strain exceeds failure_strain.",

        # ── Constraint / Load extras ───────────────────────────────────
        'condition':
            "Optional NumPy boolean expression over the mesh-node coordinates\n"
            "x, y, z (mm).  Used only when no SelectFace node is connected.\n"
            "Example:  z < 0.01   or   (x < 1) & (y > 9).",
        'displacement_x':    "Prescribed X-displacement [mm] (only when constraint_type = Displacement).",
        'displacement_y':    "Prescribed Y-displacement [mm] (only when constraint_type = Displacement).",
        'displacement_z':    "Prescribed Z-displacement [mm] (only when constraint_type = Displacement).",
        'gravity_accel':     "Gravity magnitude [mm/s²].  9810 = standard Earth gravity.",
        'gravity_direction': "Sign and axis of the gravity vector.",
        'pressure':          "Surface pressure [Pa].  Positive = outward, negative = inward.",

        # ── SelectFace ─────────────────────────────────────────────────
        'selector_type':     "How this node picks faces (see the dropdown's own tooltip).",
        'direction':         "Outward-normal direction the selected face(s) must point in.",
    }

    _PROPERTY_LABELS = {
        'application_scope': 'Impact Setup',
        'velocity_x': 'Velocity X (mm/ms)',
        'velocity_y': 'Velocity Y (mm/ms)',
        'velocity_z': 'Velocity Z (mm/ms)',
        'node_tolerance': 'Node Tolerance (mm)',
        'impactor_mass_kg': 'Sled Mass (kg)',
        'end_time': 'End Time (ms)',
        'time_steps': 'Mass-Scaling Steps',
        'n_frames': 'Animation Frames',
        'disp_scale': 'Display Scale',
        'enable_mass_scaling': 'Mass Scaling',
        'external_timeout_s': 'Solver Timeout (s)',
        'openradioss_starter_path': 'Starter Path',
        'openradioss_engine_path': 'Engine Path',
        'external_work_dir': 'Work Directory',
        'deck_only': 'Deck Only',
        'youngs_modulus': "Young's Modulus (MPa)",
        'poissons_ratio': "Poisson's Ratio",
        'density': 'Density (t/mm^3)',
        'yield_strength': 'Yield Strength (MPa)',
        'tangent_modulus': 'Tangent Modulus (MPa)',
        'failure_strain': 'Failure Strain',
    }

    @classmethod
    def _section_for(cls, prop_name):
        for title, prefixes in cls._PROPERTY_SECTIONS:
            for pref in prefixes:
                if prop_name == pref or prop_name.startswith(pref):
                    return title
        return "General"

    # ── Path / directory property helpers ─────────────────────────────────
    # File-filter strings keyed by property name (most specific first).
    # Each entry: (substring match, dialog title, filter pattern).
    _PATH_PROP_FILTERS = (
        ("deck_path",                "Select OpenRadioss / LS-DYNA deck",
         "Solver decks (*.k *.key *.rad *.inp);;All files (*)"),
        ("engine_path",              "Select OpenRadioss engine file",
         "Radioss engine files (*.rad *_0001.rad);;All files (*)"),
        ("engine_executable_path",   "Select OpenRadioss engine binary",
         "Executables (*.exe);;All files (*)"),
        ("starter_path",             "Select OpenRadioss starter binary",
         "Executables (*.exe);;All files (*)"),
        ("openradioss_starter_path", "Select OpenRadioss starter binary",
         "Executables (*.exe);;All files (*)"),
        ("openradioss_engine_path",  "Select OpenRadioss engine binary",
         "Executables (*.exe);;All files (*)"),
        ("external_solver_path",     "Select CalculiX `ccx` binary",
         "Executables (*.exe);;All files (*)"),
        ("filepath",                 "Select CAD file",
         "CAD files (*.step *.stp *.iges *.igs *.brep *.stl *.obj);;All files (*)"),
    )

    @classmethod
    def _is_directory_prop(cls, name):
        n = name.lower()
        return n.endswith("_dir") or n == "work_dir" or n.endswith("_directory")

    @classmethod
    def _looks_like_path_prop(cls, name):
        n = name.lower()
        if cls._is_directory_prop(name):
            return True
        # Match anything that ends with _path, equals 'filepath', or is one
        # of our explicitly known file properties.
        if n.endswith("_path") or n == "filepath":
            return True
        for sub, _, _ in cls._PATH_PROP_FILTERS:
            if sub in n:
                return True
        return False

    def _build_path_widget(self, name, val):
        """QLineEdit + Browse button for path / directory properties."""
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        edit = QtWidgets.QLineEdit(str(val) if val is not None else "")
        edit.setPlaceholderText(
            "Browse to a folder…" if self._is_directory_prop(name)
            else "Browse to a file…"
        )
        edit.editingFinished.connect(
            lambda n=name, w=edit: self.update_property(n, w.text())
        )
        h.addWidget(edit, 1)

        btn = QtWidgets.QToolButton()
        btn.setText("...")
        btn.setToolTip(
            "Pick a folder" if self._is_directory_prop(name) else "Pick a file"
        )
        btn.setMinimumWidth(28)

        def browse():
            current = edit.text().strip()
            start_dir = current if (current and os.path.isdir(os.path.dirname(current) or current)) else ""
            if self._is_directory_prop(name):
                chosen = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Select directory", start_dir or current
                )
            else:
                title = "Select file"
                filt = "All files (*)"
                for sub, t, f in self._PATH_PROP_FILTERS:
                    if sub in name.lower():
                        title, filt = t, f
                        break
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, title, start_dir or current, filt
                )
            if chosen:
                edit.setText(chosen)
                self.update_property(name, chosen)

        btn.clicked.connect(browse)
        h.addWidget(btn)
        return container

    def _build_generic_ui(self, node):
        """Generic Clean UI - uses node.properties() for custom properties.

        Properties are grouped into semantic sections (External Solver,
        Visualization, Solver, Material, Mesh, Geometry, Load, Constraint,
        General) and rendered with the right widget per type — sliders when
        the node declared a range, checkboxes for bools, combos when items
        are declared, expression-aware edits for numerics.
        """
        try:
            all_props = node.properties()
            props = all_props.get('custom', {})
        except Exception:
            props = {}

        if not props:
            lbl = QtWidgets.QLabel("No editable properties")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.props_layout.addWidget(lbl)
            return

        # Collect NodeGraphQt's per-property metadata once.  Items and ranges
        # both live on the model — items are how we know it's a combo, range
        # is how we pick a slider over a spinbox.
        prop_attrs = {}
        try:
            if hasattr(node, 'model'):
                model = node.model
                temp_attrs = getattr(model, '_TEMP_property_attrs', None) or {}
                if isinstance(temp_attrs, dict):
                    prop_attrs.update({k: dict(v) for k, v in temp_attrs.items() if isinstance(v, dict)})
                if getattr(model, '_graph_model', None) is not None:
                    try:
                        common = model._graph_model.get_node_common_properties(model.type_) or {}
                        for k, v in common.items():
                            if isinstance(v, dict):
                                merged = prop_attrs.get(k, {})
                                merged.update(v)
                                prop_attrs[k] = merged
                    except Exception:
                        pass
        except Exception:
            pass

        # Skip empty/optional properties so the inspector isn't a wall of blanks.
        def _should_skip(name, val):
            if name.startswith('_'):
                return True
            if name in self._PROPERTY_HIDE_ALWAYS:
                return True
            if name in self._PROPERTY_HIDE_IF_EMPTY:
                if val is None or val == '' or val == 0 or val == 0.0:
                    return True
            return False

        # Bucket properties into sections, preserving the section order above.
        section_order = [t for t, _ in self._PROPERTY_SECTIONS] + ["General"]
        buckets = {title: [] for title in section_order}
        for name in sorted(props.keys()):
            val = props[name]
            if _should_skip(name, val):
                continue
            buckets[self._section_for(name)].append((name, val))

        # Hard-coded fallback combo items for nodes whose ``items`` metadata
        # is missing from NodeGraphQt's per-instance attrs (happens for older
        # saved files and for properties created via add_text_input()).  The
        # entry whose superset value list covers the saved string wins.
        known_combos = {
            'operation': ['Union', 'Cut', 'Intersect'],
            'preset': ['Custom', 'Steel (Structural)', 'Steel (Stainless 304)',
                       'Aluminum 6061-T6', 'Aluminum 7075-T6', 'Titanium Ti-6Al-4V',
                       'Copper (Annealed)', 'Brass', 'Cast Iron (Gray)', 'Magnesium AZ31',
                       'Nickel Alloy 718', 'CFRP (Quasi-Isotropic)', 'GFRP (E-Glass)',
                       'Concrete (Normal)', 'ABS Plastic', 'Nylon 6/6', 'PEEK', 'Wood (Oak)'],
            'mesh_type': ['Tet', 'Tet10'],
            'constraint_type': ['Fixed', 'Roller X', 'Roller Y', 'Roller Z',
                                'Pinned', 'Symmetry X', 'Symmetry Y', 'Symmetry Z', 'Displacement'],
            'load_type': ['Force', 'Gravity', 'Pressure'],
            'gravity_direction': ['-Y', '-Z', '-X', '+Y', '+Z', '+X'],
            # Union of every solver node's visualization vocabulary.
            'visualization': ['Von Mises Stress', 'Displacement',
                              'Plastic Strain', 'Failed Elements', 'Density',
                              'Recovered Shape'],
            'filter_type': ['sensitivity', 'density'],
            'update_scheme': ['MMA', 'OC'],
            'projection': ['None', 'Heaviside'],
            # New combos introduced by the CalculiX / OpenRadioss rewrites.
            'analysis_type': ['Linear', 'Nonlinear (Geometric)', 'Nonlinear (Plastic)'],
            'deformation_scale': ['Auto', '1x', '5x', '10x', '50x', '100x', '200x'],
            'application_scope': ['Impact Face', 'Moving Body'],
            # Optimisation-node combos that were always there but worth pinning.
            'objective': ['Min Weight', 'Min Compliance', 'Min Max Stress',
                          'Uniform Stress'],
            'optimizer':  ['COBYLA', 'Nelder-Mead', 'SLSQP', 'L-BFGS-B',
                           'trust-constr', 'Powell'],
            'sensitivity_method': ['Biological Stress Leveling', 'Adjoint Compliance'],
            'element_type': ['Fast (Linear P1)', 'Accurate (Quadratic P2)'],
        }
        if node.__class__.__name__ == 'CrashMaterialNode':
            try:
                from pylcss.cad.nodes.crash.materials import CRASH_MATERIAL_PRESETS
                known_combos['preset'] = list(CRASH_MATERIAL_PRESETS.keys())
            except Exception:
                known_combos['preset'] = [
                    'Custom',
                    'Steel (Structural A36)',
                    'Steel (High-Strength DP780)',
                    'DP780 Dual-Phase',
                    'Steel (Ultra-High UHSS 1500)',
                    'Aluminum 6061-T6',
                    'Aluminum 5052-H32 (Crush)',
                    'CFRP (Quasi-Isotropic)',
                ]

        rendered_any = False
        for title in section_order:
            entries = buckets[title]
            if not entries:
                continue
            rendered_any = True
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QFormLayout()
            layout.setLabelAlignment(QtCore.Qt.AlignRight)
            layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

            for name, val in entries:
                label_text = self._PROPERTY_LABELS.get(name, name.replace('_', ' ').title())
                attrs = prop_attrs.get(name, {})
                combo_items = attrs.get('items') if isinstance(attrs, dict) else None
                if not combo_items and name in known_combos:
                    combo_items = known_combos[name]
                value_range = attrs.get('range') if isinstance(attrs, dict) else None
                tooltip = attrs.get('tooltip') if isinstance(attrs, dict) else None

                if combo_items:
                    widget = QtWidgets.QComboBox()
                    item_texts = [str(item) for item in combo_items]
                    if val is not None and str(val) not in item_texts:
                        item_texts.append(str(val))
                    widget.addItems(item_texts)
                    if val is not None:
                        idx = widget.findText(str(val))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                    widget.currentTextChanged.connect(lambda v, n=name: self.update_property(n, v))
                elif isinstance(val, bool):
                    widget = QtWidgets.QCheckBox()
                    widget.setChecked(val)
                    widget.stateChanged.connect(lambda s, n=name: self.update_property(n, bool(s)))
                elif isinstance(val, (int, float)) and value_range and len(value_range) == 2:
                    # Range-bounded numerics get a slider+spin combo for clarity.
                    lo, hi = float(value_range[0]), float(value_range[1])
                    is_int = isinstance(val, int) and float(lo).is_integer() and float(hi).is_integer()
                    container = QtWidgets.QWidget()
                    h = QtWidgets.QHBoxLayout(container)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.setSpacing(6)
                    if is_int:
                        spin = QtWidgets.QSpinBox()
                        spin.setRange(int(lo), int(hi))
                        spin.setValue(int(val))
                    else:
                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setDecimals(4)
                        spin.setRange(lo, hi)
                        spin.setValue(float(val))
                    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                    slider_steps = 1000 if not is_int else max(1, int(hi - lo))
                    slider.setRange(0, slider_steps)
                    def _to_slider(v, lo=lo, hi=hi, steps=slider_steps):
                        if hi <= lo:
                            return 0
                        return int(round((v - lo) / (hi - lo) * steps))
                    def _from_slider(s, lo=lo, hi=hi, steps=slider_steps):
                        return lo + (s / steps) * (hi - lo)
                    slider.setValue(_to_slider(float(val)))
                    def on_slider(s, n=name, sp=spin, conv=_from_slider, is_int=is_int):
                        v = conv(s)
                        if is_int:
                            v = int(round(v))
                        sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
                        self.update_property(n, v)
                    def on_spin(v, n=name, sl=slider, conv=_to_slider):
                        sl.blockSignals(True); sl.setValue(conv(float(v))); sl.blockSignals(False)
                        self.update_property(n, v)
                    slider.valueChanged.connect(on_slider)
                    spin.valueChanged.connect(on_spin)
                    h.addWidget(slider, 1)
                    h.addWidget(spin)
                    widget = container
                elif isinstance(val, (int, float)):
                    widget = ExpressionEdit(val)
                    widget.value_changed.connect(lambda v, n=name: self.update_property(n, v))
                elif self._looks_like_path_prop(name):
                    widget = self._build_path_widget(name, val)
                elif val is not None:
                    widget = QtWidgets.QLineEdit(str(val))
                    widget.editingFinished.connect(lambda n=name, w=widget: self.update_property(n, w.text()))
                else:
                    continue

                # Fall back to the panel's static tooltip table when the
                # node itself didn't ship one.  Lets us document the meaning
                # of obscure knobs without touching every node class.
                if not tooltip:
                    tooltip = self._PROPERTY_TOOLTIPS.get(name)
                if tooltip:
                    widget.setToolTip(str(tooltip))
                layout.addRow(label_text, widget)

            group.setLayout(layout)
            self.props_layout.addWidget(group)

        if not rendered_any:
            lbl = QtWidgets.QLabel("No editable properties")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.props_layout.addWidget(lbl)
    
    # ──────────────────────────────────────────────────
    # Interactive Select Face UI
    # ──────────────────────────────────────────────────

    def _build_select_face_ui(self, node):
        """Selector-aware UI for SelectFaceNode.

        SelectFaceNode is a swiss-army-knife with seven different selection
        strategies (bounding box, nearest point, face index, direction tag,
        range expression…).  The generic inspector shows *all* of their
        fields at once which is overwhelming.  Here we render only the
        fields relevant to the currently chosen ``selector_type``.
        """
        # The selector type drives which field group is shown.
        sel_type = node.get_property('selector_type') or 'Bounding Box'
        type_options = [
            'Bounding Box', 'Nearest Point', 'Direction', 'Face Index',
            'Range Expression', 'Tag',
        ]

        # ── 1. Selector type combo ──────────────────────────────────────
        grp_type = QtWidgets.QGroupBox("Selector")
        lay_type = QtWidgets.QFormLayout()
        combo = QtWidgets.QComboBox()
        combo.addItems(type_options)
        if sel_type in type_options:
            combo.blockSignals(True)
            combo.setCurrentText(sel_type)
            combo.blockSignals(False)
        combo.setToolTip(
            "How this node picks faces:\n"
            "  Bounding Box     — every face whose centroid is inside the box\n"
            "  Nearest Point    — the single face closest to (near_x, near_y, near_z)\n"
            "  Direction        — every face whose normal is +X / −Z / …\n"
            "  Face Index       — pick by integer face id (1-based, brittle)\n"
            "  Range Expression — Python boolean over x, y, z of the face centroid\n"
            "  Tag              — match a user-set tag string on the upstream node"
        )
        combo.currentTextChanged.connect(
            lambda v: (self.update_property('selector_type', v),
                       self.display_node(node))  # rebuild panel for the new type
        )
        lay_type.addRow("Type:", combo)
        grp_type.setLayout(lay_type)
        self.props_layout.addWidget(grp_type)

        # ── 2. Type-specific field group ────────────────────────────────
        grp = QtWidgets.QGroupBox("Parameters")
        lay = QtWidgets.QFormLayout()

        def _spin(prop, lo=-1e4, hi=1e4, dec=3):
            val = float(node.get_property(prop) or 0.0)
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(dec); w.setValue(val)
            w.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return w

        def _intspin(prop, lo=0, hi=10_000):
            try: val = int(node.get_property(prop) or 0)
            except Exception: val = 0
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi); w.setValue(val)
            w.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return w

        def _line(prop, placeholder=''):
            val = node.get_property(prop) or ''
            w = QtWidgets.QLineEdit(str(val))
            w.setPlaceholderText(placeholder)
            w.editingFinished.connect(lambda p=prop, ww=w: self.update_property(p, ww.text()))
            return w

        if sel_type == 'Bounding Box':
            lay.addRow("Min X:", _spin('box_min_x'))
            lay.addRow("Min Y:", _spin('box_min_y'))
            lay.addRow("Min Z:", _spin('box_min_z'))
            lay.addRow("Max X:", _spin('box_max_x'))
            lay.addRow("Max Y:", _spin('box_max_y'))
            lay.addRow("Max Z:", _spin('box_max_z'))
        elif sel_type == 'Nearest Point':
            lay.addRow("Near X:", _spin('near_x'))
            lay.addRow("Near Y:", _spin('near_y'))
            lay.addRow("Near Z:", _spin('near_z'))
        elif sel_type == 'Direction':
            dir_combo = QtWidgets.QComboBox()
            dir_combo.addItems(['+X', '-X', '+Y', '-Y', '+Z', '-Z'])
            dir_combo.setToolTip(
                "Pick every face whose outward normal points in this direction "
                "(within ~10° tolerance)."
            )
            cur = node.get_property('direction') or '+Z'
            if cur in [dir_combo.itemText(i) for i in range(dir_combo.count())]:
                dir_combo.setCurrentText(cur)
            dir_combo.currentTextChanged.connect(
                lambda v: self.update_property('direction', v)
            )
            lay.addRow("Normal:", dir_combo)
        elif sel_type == 'Face Index':
            w = _intspin('face_index', 0, 100_000)
            w.setToolTip(
                "Zero-based face index from CadQuery's `faces()` iteration order.\n"
                "Fragile — adding a fillet or boolean upstream renumbers faces."
            )
            lay.addRow("Index:", w)
        elif sel_type == 'Range Expression':
            w = _line('range_expr', placeholder='e.g. z > 0.99 * z_max')
            w.setToolTip(
                "Python boolean over the face-centroid coordinates x, y, z.\n"
                "Face is picked when this expression is True for its centroid."
            )
            lay.addRow("Expr:", w)
        elif sel_type == 'Tag':
            w = _line('tag', placeholder='e.g. top_face')
            w.setToolTip(
                "Match faces that were tagged with this string on the upstream node."
            )
            lay.addRow("Tag:", w)

        grp.setLayout(lay)
        self.props_layout.addWidget(grp)

    def _build_interactive_select_ui(self, node):
        """Dedicated Properties Panel UI for InteractiveSelectFaceNode."""
        # -- Status banner --
        sel_label = node.get_property('selection_label') or 'No faces selected'
        raw_indices = node.get_property('picked_face_indices') or ''
        face_indices = [int(t.strip()) for t in raw_indices.split(',') if t.strip().isdigit()]

        banner = QtWidgets.QLabel(sel_label)
        banner.setWordWrap(True)
        if face_indices:
            banner.setStyleSheet(
                "background:#1a5c2a; color:#6dde8d; font-weight:bold;"
                "padding:8px; border-radius:4px; margin-bottom:6px;"
            )
        else:
            banner.setStyleSheet(
                "background:#3a2800; color:#f0b040; font-weight:bold;"
                "padding:8px; border-radius:4px; margin-bottom:6px;"
            )
        self.props_layout.addWidget(banner)
        self._pick_banner = banner

        # -- Face list --
        if face_indices:
            group_list = QtWidgets.QGroupBox(f"Selected Faces ({len(face_indices)})")
            vbox = QtWidgets.QVBoxLayout(group_list)
            for idx in face_indices:
                lbl = QtWidgets.QLabel(f"  Face index {idx}")
                lbl.setStyleSheet("color:#aad4ff; font-size:11px;")
                vbox.addWidget(lbl)
            self.props_layout.addWidget(group_list)

        # -- Pick button --
        btn_pick = QtWidgets.QPushButton("Pick Faces in 3D Viewer")
        btn_pick.setStyleSheet(
            "QPushButton { background:#1e5ab4; color:white; border-radius:5px;"
            "  padding:8px; font-weight:bold; font-size:13px; }"
            "QPushButton:hover { background:#2470d8; }"
        )
        btn_pick.setToolTip(
            "Click to enter face-picking mode.\n"
            "Then click faces on the 3D model. Ctrl+Click for multi-select."
        )
        btn_pick.clicked.connect(lambda: self._start_picking_session(node))
        self.props_layout.addWidget(btn_pick)

        # -- Clear button --
        btn_clear = QtWidgets.QPushButton("Clear Selection")
        btn_clear.setStyleSheet(
            "QPushButton { background:#3a1010; color:#f08080; border-radius:5px;"
            "  padding:6px; font-size:12px; }"
            "QPushButton:hover { background:#5a1010; }"
        )
        btn_clear.clicked.connect(lambda: self._clear_face_selection(node))
        self.props_layout.addWidget(btn_clear)

        # -- Hint --
        hint = QtWidgets.QLabel(
            "<i>Note: Execute the graph first so the 3D viewer has geometry to pick from.</i>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:10px; margin-top:8px;")
        self.props_layout.addWidget(hint)

    def _start_picking_session(self, node):
        """Enable picking mode in the 3D viewer for this node."""
        # Walk up to find ProfessionalCadApp
        app = self._get_main_app()
        if app is None:
            QtWidgets.QMessageBox.warning(
                self, "No Viewer",
                "Cannot access the 3D viewer. Make sure the application is fully loaded."
            )
            return

        viewer = getattr(app, 'viewer', None)
        if viewer is None:
            QtWidgets.QMessageBox.warning(self, "No Viewer", "3D viewer not found.")
            return

        # Build index→OCC mapping from the viewer's stored face list
        # The viewer must have already rendered the upstream shape
        if not viewer._all_occ_faces:
            QtWidgets.QMessageBox.information(
                self, "Run Graph First",
                "Please execute the graph (▶ Run) so the 3D viewer has the shape loaded,"
                " then try picking again."
            )
            return

        viewer.enable_picking_mode(multi_select=True)

        # Wire done signal
        def _on_faces_picked(occ_faces):
            try:
                viewer.face_picked.disconnect(_on_faces_picked)
            except Exception:
                pass
            # Map OCC face objects → indices in the viewer's list
            all_occ = viewer._all_occ_faces
            picked_indices = []
            for face in occ_faces:
                for i, f in enumerate(all_occ):
                    # Identity check via hash code (OCC)
                    try:
                        if face.hashCode(10000) == f.hashCode(10000):
                            picked_indices.append(i)
                            break
                    except Exception:
                        if face is f:
                            picked_indices.append(i)
                            break

            if hasattr(node, 'set_picked_faces'):
                node.set_picked_faces(picked_indices)
            else:
                node.set_property('picked_face_indices',
                                  ','.join(str(i) for i in picked_indices))

            # Invalidate and re-run
            if hasattr(node, '_last_hash'):
                node._last_hash = None
            try:
                self.property_changed.emit(node, 'picked_face_indices',
                                           '', node.get_property('picked_face_indices'))
            except Exception:
                pass

            # Refresh the panel
            self.display_node(node)

            if hasattr(app, '_execute_graph'):
                app._execute_graph(skip_simulation=True)

        def _on_cancelled():
            try:
                viewer.picking_cancelled.disconnect(_on_cancelled)
                viewer.face_picked.disconnect(_on_faces_picked)
            except Exception:
                pass

        viewer.face_picked.connect(_on_faces_picked)
        viewer.picking_cancelled.connect(_on_cancelled)

    def _clear_face_selection(self, node):
        """Clear all picked faces from an InteractiveSelectFaceNode."""
        if hasattr(node, 'set_picked_faces'):
            node.set_picked_faces([])
        else:
            node.set_property('picked_face_indices', '')
            node.set_property('selection_label', 'No faces selected')
        if hasattr(node, '_last_hash'):
            node._last_hash = None
        self.display_node(node)

    def _get_main_app(self):
        """Walk up the parent chain to find ProfessionalCadApp."""
        widget = self.parent()
        while widget is not None:
            if widget.__class__.__name__ == 'ProfessionalCadApp':
                return widget
            widget = widget.parent() if hasattr(widget, 'parent') else None
        return None

    # ──────────────────────────────────────────────────
    # FEA Boundary Condition Rich UI
    # ──────────────────────────────────────────────────

    def _build_fea_bc_ui(self, node):
        """Rich Properties Panel UI for ConstraintNode, LoadNode, PressureLoadNode."""
        node_class = node.__class__.__name__
        props = node.model.properties

        if node_class == 'ConstraintNode':
            # Use get_property (NodeGraphQt API) so we always read the live value,
            # not a potentially stale snapshot from node.model.properties.
            ct = node.get_property('constraint_type') or 'Fixed'

            grp = QtWidgets.QGroupBox("Constraint Type")
            lay = QtWidgets.QFormLayout()

            combo = QtWidgets.QComboBox()
            combo.addItems(['Fixed', 'Roller X', 'Roller Y', 'Roller Z',
                            'Pinned', 'Symmetry X', 'Symmetry Y', 'Symmetry Z', 'Displacement'])
            # Block signals while setting the initial value so construction doesn't
            # fire currentTextChanged and cause a spurious property-change loop.
            combo.blockSignals(True)
            combo.setCurrentText(str(ct))
            combo.blockSignals(False)
            combo.currentTextChanged.connect(lambda v: self.update_property('constraint_type', v))
            lay.addRow("Type:", combo)

            if ct == 'Displacement':
                for ax in ['displacement_x', 'displacement_y', 'displacement_z']:
                    val = node.get_property(ax)
                    if val is not None:
                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setRange(-1e6, 1e6)
                        spin.setDecimals(4)
                        spin.setValue(float(val))
                        spin.valueChanged.connect(lambda v, p=ax: self.update_property(p, v))
                        lay.addRow(ax.replace('displacement_', 'U') + ':', spin)

            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

            # Condition expression — used when no face is connected (e.g. all .cad examples)
            cond_val = node.get_property('condition') or ''
            grp2 = QtWidgets.QGroupBox("Applied To (Condition Expression)")
            grp2.setToolTip("NumPy boolean expression over mesh node coordinates x, y, z.\n"
                            "Nodes where the expression is True receive this constraint.\n"
                            "Used when no SelectFace node is connected.")
            lay2 = QtWidgets.QFormLayout()
            edit = QtWidgets.QLineEdit(str(cond_val))
            edit.setPlaceholderText("e.g. z < 0.01  or  (x < 1) & (y > 9)")
            edit.editingFinished.connect(lambda: self.update_property('condition', edit.text()))
            lay2.addRow("Expression:", edit)
            info = QtWidgets.QLabel("Variables: x, y, z (node coords in mm).  Leave blank if a face is connected.")
            info.setStyleSheet("color:#888; font-size:10px;")
            info.setWordWrap(True)
            lay2.addRow(info)
            grp2.setLayout(lay2)
            self.props_layout.addWidget(grp2)

        elif node_class == 'LoadNode':
            lt = node.get_property('load_type') or 'Force'
            grp = QtWidgets.QGroupBox("Load Settings")
            lay = QtWidgets.QFormLayout()

            combo = QtWidgets.QComboBox()
            combo.addItems(['Force', 'Gravity'])
            combo.blockSignals(True)
            combo.setCurrentText(str(lt) if str(lt) in ['Force', 'Gravity'] else 'Force')
            combo.blockSignals(False)
            combo.currentTextChanged.connect(lambda v: self.update_property('load_type', v))
            lay.addRow("Type:", combo)

            if lt == 'Force':
                fx = float(node.get_property('force_x') or 0.0)
                fy = float(node.get_property('force_y') or 0.0)
                fz = float(node.get_property('force_z') or 0.0)
                for axis, prop, val in [('X', 'force_x', fx), ('Y', 'force_y', fy), ('Z', 'force_z', fz)]:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(-1e12, 1e12)
                    spin.setDecimals(2)
                    spin.setValue(val)
                    spin.setSuffix(' N')
                    spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                    lay.addRow(f"F{axis}:", spin)
                mag = (fx**2 + fy**2 + fz**2) ** 0.5
                # Direction arrow label  e.g. "→ (-1000, 0, 0)"
                def _dir_arrow(x, y, z):
                    dominant = max([(abs(x),'X',x),(abs(y),'Y',y),(abs(z),'Z',z)], key=lambda t: t[0])
                    sign = '−' if dominant[2] < 0 else '+'
                    return f"{sign}{dominant[1]}"
                dir_lbl = QtWidgets.QLabel(
                    f"({fx:+.0f}, {fy:+.0f}, {fz:+.0f}) N   │   {mag:.2f} N   │   dir ≈ {_dir_arrow(fx,fy,fz)}")
                dir_lbl.setStyleSheet("color:#6dde8d; font-weight:bold; font-size:11px;")
                lay.addRow("", dir_lbl)

            elif lt == 'Gravity':
                grav_accel = float(node.get_property('gravity_accel') or 9810.0)
                grav_dir   = node.get_property('gravity_direction') or '-Y'
                spin_g = QtWidgets.QDoubleSpinBox()
                spin_g.setRange(0, 100000)
                spin_g.setDecimals(2)
                spin_g.setValue(grav_accel)
                spin_g.setSuffix(' mm/s²')
                spin_g.valueChanged.connect(lambda v: self.update_property('gravity_accel', v))
                lay.addRow("Accel:", spin_g)
                cb_dir = QtWidgets.QComboBox()
                cb_dir.addItems(['-Y', '-Z', '-X', '+Y', '+Z', '+X'])
                cb_dir.blockSignals(True)
                cb_dir.setCurrentText(str(grav_dir))
                cb_dir.blockSignals(False)
                cb_dir.currentTextChanged.connect(lambda v: self.update_property('gravity_direction', v))
                lay.addRow("Direction:", cb_dir)

            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

            # Applied-to condition expression
            cond_val = node.get_property('condition') or ''
            grp2 = QtWidgets.QGroupBox("Applied To (Condition Expression)")
            grp2.setToolTip("NumPy boolean expression over mesh node coordinates x, y, z.\n"
                            "Nodes where the expression is True receive this load.\n"
                            "Used when no SelectFace node is connected.")
            lay2 = QtWidgets.QFormLayout()
            edit = QtWidgets.QLineEdit(str(cond_val))
            edit.setPlaceholderText("e.g. z > 19   or   (np.abs(z) < 1.5) & (x > 9)")
            edit.editingFinished.connect(lambda: self.update_property('condition', edit.text()))
            lay2.addRow("Expression:", edit)
            info = QtWidgets.QLabel("Variables: x, y, z (node coords in mm).  Leave blank if a SelectFace is connected.")
            info.setStyleSheet("color:#888; font-size:10px;")
            info.setWordWrap(True)
            lay2.addRow(info)
            grp2.setLayout(lay2)
            self.props_layout.addWidget(grp2)

        # ── Preview in 3D button (Constraint + Load) ──────────────────────────
        if node_class in ('ConstraintNode', 'LoadNode'):
            btn_preview = QtWidgets.QPushButton("👁  Preview in 3D")
            btn_preview.setToolTip(
                "Run the CAD graph (geometry only — no FEA/crash solve) and "
                "highlight the faces selected by the SelectFace node upstream\n"
                "of this BC.  Requires a SelectFace → Constraint / Load link;\n"
                "a bare condition-string BC has no face to highlight."
            )
            btn_preview.setStyleSheet(
                "QPushButton {"
                "  background: #1e5aab; color: white; border-radius: 4px;"
                "  padding: 5px 10px; font-weight: bold; font-size: 12px;"
                "  margin-top: 6px;"
                "}"
                "QPushButton:hover { background: #2673cc; }"
            )

            def _on_preview(checked=False, _node=node):
                app = self._get_main_app()
                if app is None:
                    QtWidgets.QMessageBox.warning(
                        self, "No viewer",
                        "Cannot reach the main CAD widget; preview unavailable."
                    )
                    return

                # 1. Render the upstream geometry first.  Many bugs reported as
                # "Preview does nothing" turn out to be that the user never ran
                # the graph — _last_result is None on every upstream node, so
                # the viewer has no shape to draw the overlay on.
                source, renderable = app._get_render_context_for_node(_node)
                if renderable is None:
                    app._last_rendered_node = _node
                    try:
                        app._execute_graph(skip_simulation=True)
                    except Exception as exc:
                        QtWidgets.QMessageBox.critical(
                            self, "Graph execution failed",
                            f"Couldn't preview because the CAD graph errored:\n\n{exc}"
                        )
                        return
                    source, renderable = app._get_render_context_for_node(_node)
                if renderable is not None:
                    app._render_result_in_viewer(renderable)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Nothing to preview",
                        "No upstream geometry is connected to this BC, or the "
                        "CAD graph hasn't produced one yet.\n\n"
                        "Connect a primitive / sketch / import node → … → this "
                        "BC and try again."
                    )
                    return

                # 2. Now draw the BC overlay on top of the shape.
                try:
                    app._show_bc_for_node(_node)
                except Exception as exc:
                    # _show_bc_for_node has its own try/except; this is belt-
                    # and-suspenders.  We still want the user to see *why*.
                    QtWidgets.QMessageBox.warning(
                        self, "Overlay failed",
                        f"Geometry rendered, but the BC overlay couldn't be "
                        f"drawn:\n\n{exc}\n\n"
                        "Most common cause: the SelectFace node upstream of "
                        "this BC hasn't selected any faces yet."
                    )
                    return

                # 3. Verify the overlay actually had something to draw.
                # _collect_bc_for_node is cheap; surface a friendly hint when
                # the overlay was a no-op (e.g. the BC uses a condition
                # string that doesn't match any nodes).
                try:
                    c_faces, l_faces, l_vecs = app._collect_bc_for_node(_node)
                    if not (c_faces or l_faces or l_vecs):
                        sb = getattr(app, 'statusBar', lambda: None)()
                        if sb is not None:
                            sb.showMessage(
                                "Preview: geometry rendered, but no BC face/vector "
                                "to highlight (connect a SelectFace upstream).",
                                6000,
                            )
                except Exception:
                    pass

            btn_preview.clicked.connect(_on_preview)
            self.props_layout.addWidget(btn_preview)

        elif node_class == 'PressureLoadNode':
            grp = QtWidgets.QGroupBox("Pressure Load")
            lay = QtWidgets.QFormLayout()

            pval = float(props.get('pressure', 1000000.0))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e15, 1e15)
            spin.setDecimals(2)
            spin.setValue(pval)
            spin.setSuffix(' Pa')
            spin.valueChanged.connect(lambda v: self.update_property('pressure', v))
            lay.addRow("Pressure:", spin)

            pval_mpa = pval / 1e6
            info_lbl = QtWidgets.QLabel(f"{pval_mpa:.4g} MPa  (positive = outward, negative = inward)")
            info_lbl.setStyleSheet("color:#aad4ff; font-size:11px;")
            lay.addRow("", info_lbl)

            grp.setLayout(lay)
            self.props_layout.addWidget(grp)

        else:
            self._build_generic_ui(node)

    # ──────────────────────────────────────────────────
    # Property update
    # ──────────────────────────────────────────────────

    def update_property(self, prop_name, value):
        """Update node property and mark as dirty for recalculation."""
        if self.current_node:
            try:
                self._updating_property = True
                old = self.current_node.get_property(prop_name)
                self.current_node.set_property(prop_name, value)
                # Mark node as dirty for recalculation
                if hasattr(self.current_node, '_last_hash'):
                    self.current_node._last_hash = None  # Invalidate hash cache
                try:
                    self.property_changed.emit(self.current_node, prop_name, old, value)
                except Exception: pass
            except Exception:
                pass
            finally:
                self._updating_property = False


class TimelinePanel(QtWidgets.QWidget):
    """Timeline/History panel for tracking changes."""

    def __init__(self):
        super(TimelinePanel, self).__init__()
        self.layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Timeline")
        title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        self.layout.addWidget(title)

        self.history_list = QtWidgets.QListWidget()
        self.layout.addWidget(self.history_list)

    def add_event(self, event_text):
        """Add an event to the timeline."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.addItem(f"[{timestamp}] {event_text}")
        # Scroll to bottom
        self.history_list.scrollToBottom()


class ResultsPanel(QtWidgets.QWidget):
    """Summary of the most recent FEA / Crash / TopOpt solve.

    Pulled from the result dict that the solver nodes already produce — so we
    do not duplicate any computation; this is purely a presentation surface
    for what otherwise only goes to stdout.
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            """
            QGroupBox { font-weight: bold; margin-top: 8px; padding-top: 10px; border: 1px solid #444; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #BBDEFB; }
            QLabel#empty { color: #888; font-style: italic; padding: 24px; }
            QLabel.metric-key { color: #B0BEC5; }
            QLabel.metric-val { color: #FAFAFA; font-weight: bold; }
            """
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self._empty = QtWidgets.QLabel("No solver results yet — run an FEA, Crash, or Topology node.")
        self._empty.setObjectName("empty")
        self._empty.setAlignment(QtCore.Qt.AlignCenter)
        self._empty.setWordWrap(True)
        outer.addWidget(self._empty)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._scroll.setWidget(self._content)
        self._scroll.setVisible(False)
        outer.addWidget(self._scroll, 1)

    @staticmethod
    def _fmt(value, unit=""):
        if value is None:
            return "—"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if v == 0.0:
            return f"0 {unit}".strip()
        if abs(v) >= 1e3 or abs(v) < 1e-2:
            return f"{v:.3e} {unit}".strip()
        return f"{v:.4g} {unit}".strip()

    def _clear(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _add_section(self, title, rows):
        """rows: list of (label, value-string)."""
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(10, 6, 10, 10)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(4)
        for label, val in rows:
            lk = QtWidgets.QLabel(label)
            lk.setProperty("class", "metric-key")
            lk.setStyleSheet("color: #B0BEC5;")
            lv = QtWidgets.QLabel(val)
            lv.setStyleSheet("color: #FAFAFA; font-weight: bold;")
            lv.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(lk, lv)
        self._content_layout.addWidget(group)

    def _add_warnings(self, warnings):
        if not warnings:
            return
        group = QtWidgets.QGroupBox(f"Warnings ({len(warnings)})")
        v = QtWidgets.QVBoxLayout(group)
        v.setContentsMargins(10, 6, 10, 10)
        for w in warnings:
            label = QtWidgets.QLabel(f"• {w}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #FFCC80;")
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            v.addWidget(label)
        self._content_layout.addWidget(group)

    def show_result(self, data):
        """Populate from a solver result dict.  Safe to call with None."""
        if not isinstance(data, dict) or 'type' not in data:
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return
        rtype = data.get('type')
        if rtype not in ('fea', 'crash', 'topopt', 'external_solver'):
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return

        self._clear()
        self._empty.setVisible(False)
        self._scroll.setVisible(True)

        backend = data.get('backend') or ('CalculiX' if rtype == 'fea' else
                                          'OpenRadioss' if rtype == 'crash' else '—')
        meta_rows = [
            ("Type",            rtype.upper()),
            ("Backend",         str(backend)),
        ]
        if 'visualization_mode' in data:
            meta_rows.append(("Visualization", str(data['visualization_mode'])))
        if 'external_status' in data:
            meta_rows.append(("Solver status", str(data['external_status'])))
        if 'work_dir' in data:
            meta_rows.append(("Work directory", str(data['work_dir'])))
        self._add_section("Run", meta_rows)

        if rtype == 'fea' or rtype == 'external_solver':
            metrics = []
            disp = data.get('displacement')
            if disp is not None:
                try:
                    import numpy as np
                    arr = np.asarray(disp, dtype=float)
                    if arr.size:
                        max_disp = float(np.max(np.abs(arr)))
                        metrics.append(("Max |u|", self._fmt(max_disp, "mm")))
                except Exception:
                    pass
            stress = data.get('stress')
            if stress is not None:
                try:
                    import numpy as np
                    arr = np.asarray(stress, dtype=float)
                    if arr.size:
                        metrics.append(("Peak stress (nodal)", self._fmt(float(np.max(arr)), "MPa")))
                except Exception:
                    pass
            if 'max_stress_gauss' in data:
                metrics.append(("Peak stress (Gauss)", self._fmt(data['max_stress_gauss'], "MPa")))
            if 'deformation_scale' in data:
                metrics.append(("Deformation scale", f"{float(data['deformation_scale']):.1f}×"))
            if metrics:
                self._add_section("Result", metrics)

        if rtype == 'crash':
            crash_rows = []
            if 'peak_displacement' in data:
                crash_rows.append(("Peak displacement", self._fmt(data['peak_displacement'], "mm")))
            if 'peak_stress' in data:
                crash_rows.append(("Peak Von Mises", self._fmt(data['peak_stress'], "MPa")))
            if 'absorbed_energy' in data:
                crash_rows.append(("Plastic dissipation", self._fmt(data['absorbed_energy'], "N·mm")))
            if 'n_failed' in data:
                crash_rows.append(("Failed elements", str(data['n_failed'])))
            if 'frames' in data and data['frames']:
                crash_rows.append(("Animation frames", str(len(data['frames']))))
            if 'energy_balance_max_error' in data:
                crash_rows.append(("Energy balance error", f"{float(data['energy_balance_max_error']) * 100:.1f}%"))
            if crash_rows:
                self._add_section("Crash result", crash_rows)

        if rtype == 'topopt':
            topopt_rows = []
            if 'final_vol_frac' in data:
                topopt_rows.append(("Final volume fraction", f"{float(data['final_vol_frac']) * 100:.1f}%"))
            elif data.get('density') is not None:
                try:
                    import numpy as np
                    density = np.asarray(data.get('density'), dtype=float)
                    elem_vol = np.asarray(data.get('element_volumes'), dtype=float)
                    if density.size and elem_vol.size == density.size:
                        denom = float(np.sum(elem_vol))
                        if denom > 0.0:
                            vf = float(np.sum(density * elem_vol) / denom)
                            topopt_rows.append(("Final volume fraction", f"{vf * 100:.1f}%"))
                except Exception:
                    pass
            if 'target_vol_frac' in data:
                topopt_rows.append(("Target volume fraction", f"{float(data['target_vol_frac']) * 100:.1f}%"))
            if 'compliance' in data:
                topopt_rows.append(("Compliance", self._fmt(data['compliance'], "N·mm")))
            if 'volume' in data:
                topopt_rows.append(("Retained volume", self._fmt(data['volume'], "mm³")))
            if 'total_volume' in data:
                topopt_rows.append(("Design-domain volume", self._fmt(data['total_volume'], "mm³")))
            if 'mass' in data:
                topopt_rows.append(("Effective mass", self._fmt(data['mass'], "t")))
            recovered = data.get('recovered_shape')
            if isinstance(recovered, dict):
                verts = recovered.get('vertices')
                faces = recovered.get('faces')
                try:
                    topopt_rows.append(("Recovered surface", f"{len(verts)} vertices / {len(faces)} faces"))
                except Exception:
                    pass
            if topopt_rows:
                self._add_section("Topology result", topopt_rows)

        # Warnings from the external backends
        warnings = data.get('warnings') or []
        if warnings:
            self._add_warnings(list(warnings))


class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""

    # QtAwesome icon names per category prefix.  The first prefix that matches
    # wins, so order from most specific to most general.
    _CATEGORY_ICONS = (
        ("Geometry",                     "fa5s.code",            "#81C784"),
        ("Simulation - Pre-Processing",  "fa5s.project-diagram", "#80CBC4"),
        ("Simulation - Loads",           "fa5s.weight-hanging",  "#FF8A65"),
        ("Simulation - Solve",           "fa5s.calculator",      "#9CCC65"),
        ("Crash Simulation",             "fa5s.car-crash",       "#EF5350"),
        ("Analysis",                     "fa5s.balance-scale",   "#B39DDB"),
        ("IO",                           "fa5s.file-export",     "#90A4AE"),
    )

    @staticmethod
    def _icon_for(label: str):
        try:
            import qtawesome as qta
        except Exception:
            return None
        for prefix, icon_name, color in LibraryPanel._CATEGORY_ICONS:
            if label.startswith(prefix):
                try:
                    return qta.icon(icon_name, color=color)
                except Exception:
                    return None
        return None

    def __init__(self, spawn_callback):
        super(LibraryPanel, self).__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)

        # Search box — title was redundant with the dock title, dropped.
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search components...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        self.layout.addWidget(self.search)

        # Tree view for categories
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(True)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                border: none;
                background: transparent;
                outline: none;
            }
            QTreeView::item {
                padding: 4px 2px;
                border: 0;
            }
            QTreeView::item:hover {
                background: rgba(255, 255, 255, 18);
            }
            QTreeView::item:selected {
                background: rgba(33, 150, 243, 60);
                color: white;
            }
            """
        )
        # enable dragging from the library into the graph
        self.tree.setDragEnabled(True)
        self.tree.itemPressed.connect(self._start_drag)
        
        # Component categories — only entries reachable in the new code-first
        # workflow are listed.  Hand-placed primitives / sketches / 3-D ops /
        # transforms / patterns have been removed; geometry is created either
        # in a Code Part / Assembly node (one readable CadQuery script) or
        # imported from a STEP / mesh file.
        # Format: (Label, node_id, tooltip_description)
        categories = {
            "Geometry": [
                ("Code Part / Assembly", "com.cad.code_part",
                 "Parametric CAD as one readable CadQuery script.\n"
                 "Set up to 6 named parameters (with optional inputs from Number / Variable\n"
                 "nodes) and write `result = cq.Workplane(...)...`.  The output 'shape' port\n"
                 "feeds straight into Mesh / Select Face / Assembly nodes."),
                ("Import STEP", "com.cad.import_step",
                 "Import a STEP / IGES CAD file as the upstream geometry."),
                ("Import Mesh", "com.cad.import_stl",
                 "Import an STL / OBJ surface mesh."),
            ],

            # ═══════════════════════════════════════════════════════════════
            # WORKBENCH: SIMULATION (FEA)
            # ═══════════════════════════════════════════════════════════════
            "Simulation - Pre-Processing": [
                ("Select Face", "com.cad.select_face", "Select face for BCs using text selectors (Direction, Index, Box…)"),
                ("Select Face (Interactive)", "com.cad.select_face_interactive",
                 "Click faces directly in the 3D viewer to select them — no code required"),
                ("Material", "com.cad.sim.material", "Define material"),
                ("Generate Mesh", "com.cad.sim.mesh", "Create FEM mesh"),
            ],
            "Simulation - Loads & Constraints": [
                ("Constraint", "com.cad.sim.constraint", "Apply fixation/support"),
                ("Force Load", "com.cad.sim.load", "Apply force"),
                ("Pressure Load", "com.cad.sim.pressure_load", "Apply uniform pressure"),
            ],
            "Simulation - Solve & Optimize": [
                ("Solver", "com.cad.sim.solver", "Run FEA solver"),
                ("Topology Opt", "com.cad.sim.topopt", "Optimize topology (SIMP)"),
                ("Size Opt", "com.cad.sim.sizeopt", "Optimize parametric dimensions"),
                ("Shape Opt", "com.cad.sim.shapeopt", "Optimize boundary shape"),
                ("Remesh Surface", "com.cad.sim.remesh", "Convert TopOpt surface to volume mesh"),
            ],

            # ═══════════════════════════════════════════════════════════════
            # WORKBENCH: CRASH / IMPACT SIMULATION
            # ═══════════════════════════════════════════════════════════════
            "Crash Simulation": [
                ("Crash Material",    "com.cad.sim.crash_material",
                 "Elasto-plastic material with yield strength, hardening and failure strain (presets: A36, DP780, UHSS 1500, Al 6061, Al 5052, CFRP)"),
                ("Impact Condition",  "com.cad.sim.impact",
                 "Define initial velocity (mm/ms = m/s) applied to impact face nodes"),
                ("Crash Solver",       "com.cad.sim.crash_solver",
                 "Explicit transient crash solver — runs OpenRadioss on the connected mesh, material, constraints and impact condition"),
                ("Run Radioss Deck",    "com.cad.sim.radioss_deck",
                 "Run an existing OpenRadioss/LS-DYNA `.k` or `.rad` deck (e.g. the Chrysler Neon HPC benchmark) end-to-end and play the animation in the viewer"),
            ],

            # ═══════════════════════════════════════════════════════════════
            # WORKBENCH: ANALYSIS & UTILITIES
            # ═══════════════════════════════════════════════════════════════
            "Analysis & Assembly": [
                ("Assembly", "com.cad.assembly", "Combine parts"),
                ("Mass Properties", "com.cad.mass_properties", "Calculate mass/volume"),
                ("Bounding Box", "com.cad.bounding_box", "Measure dimensions"),
            ],
            "IO & Parameters": [
                ("Number", "com.cad.number",
                 "Numeric constant.  Set its `exposed_name` to make it a kwarg on cad.fea / cad.crash / cad.topopt from the system-modeling tab."),
                ("Variable", "com.cad.variable",
                 "Named variable.  Like Number but with a label; falls back to `variable_name` if `exposed_name` is blank."),
                ("Export STEP", "com.cad.export_step", "Export the current shape to a .step file"),
                ("Export STL", "com.cad.export_stl",  "Export the current mesh / shape to a .stl file"),
            ],
        }
        
        cat_font = QtGui.QFont()
        cat_font.setBold(True)
        for category, items in categories.items():
            cat_item = QtWidgets.QTreeWidgetItem([category])
            cat_item.setFont(0, cat_font)
            icon = self._icon_for(category)
            if icon is not None:
                cat_item.setIcon(0, icon)
            # Category rows are not draggable / not selectable as a target.
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemIsSelectable)

            for item_data in items:
                label, node_id, tooltip = item_data
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, QtCore.Qt.UserRole, node_id)
                item.setToolTip(0, tooltip)  # Show description on hover
                cat_item.addChild(item)

            self.tree.addTopLevelItem(cat_item)

        self.tree.expandAll()
        self.tree.itemDoubleClicked.connect(self._on_component_selected)
        self.layout.addWidget(self.tree)
    
    def _filter_tree(self, text):
        """Filter tree items based on search text."""
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(category.childCount()):
                item = category.child(j)
                matches = text in item.text(0).lower() if text else True
                item.setHidden(not matches)
                if matches:
                    visible_children += 1
            # Show category if any children match, or if search is empty
            category.setHidden(visible_children == 0 and bool(text))
            if visible_children > 0 or not text:
                category.setExpanded(bool(text))  # Auto-expand when searching
    
    def _on_component_selected(self, item, column):
        """Handle component selection from library."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if node_id:
            self.spawn_callback(node_id, item.text(0))

    def _start_drag(self, item, column):
        """Start a drag operation carrying the node identifier."""
        node_id = item.data(0, QtCore.Qt.UserRole)
        if not node_id:
            return
        drag = QDrag(self.tree)
        mime = QMimeData()
        mime.setData('application/x-node-id', str(node_id).encode('utf-8'))
        mime.setText(item.text(0))
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.CopyAction)


# Lazy imports to avoid circular dependency with hands_free module
# These are imported at runtime inside __init__ instead of at module level


class ProfessionalCadApp(QtWidgets.QMainWindow):
    """Main application window for parametric design and simulation."""
    
    def __init__(self, parent=None):
        super(ProfessionalCadApp, self).__init__(parent)
        
        self.setWindowTitle("Engineering Design Studio")
        self.resize(1200, 800)
        
        # Initialize data
        self.undo_stack = []
        self.redo_stack = []
        self.current_file = None
        
        # Initialize worker as None
        self.worker = None
        self._last_preview_update_time = 0.0
        
        # Loading state flag to suppress events during project load
        self._is_loading = False
        
        # Add mutex for thread safety
        self.result_mutex = QtCore.QMutex()
        
        # Create graph
        self.graph = NodeGraph()
        self._register_nodes()
        
        # Initialize Command Dispatcher for LLM actions (lazy import to avoid circular)
        from pylcss.assistant_systems.api.dispatcher import CommandDispatcher
        self.command_dispatcher = CommandDispatcher(main_window=self)
        self.llm_chat_dialog = None
        
        # Connect graph signals
        self.graph.property_changed.connect(self._on_graph_property_changed)
        self.graph.port_connected.connect(self._on_connection_changed)
        self.graph.port_disconnected.connect(self._on_connection_changed)
        
        # Prevent double-click popup
        self.graph.node_double_clicked.connect(self._on_node_double_clicked)
        
        # Setup UI
        self._setup_toolbar() 
        self._create_ui()
        self._setup_shortcuts()

        # Connect graph selection
        self.graph.node_selected.connect(self._on_node_selected)
    
    def _register_nodes(self):
        """Register all available nodes."""
        for node_class in NODE_CLASS_MAPPING.values():
            try:
                self.graph.register_node(node_class)
            except Exception:
                pass
    
    def _create_ui(self):
        """Create the main UI layout with 3D Viewer."""
        # Main content widget for QMainWindow
        content_widget = QtWidgets.QWidget()
        main_h_layout = QtWidgets.QHBoxLayout(content_widget)
        main_h_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel: Library
        self.library = LibraryPanel(self._spawn_node)
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.addWidget(self.library)
        left_splitter.setSizes([500])
        
        # CENTER AREA
        center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.viewer = CQ3DViewer()
        self.viewer.face_picking_requested.connect(self._start_viewer_face_picking)
        center_splitter.addWidget(self.viewer)
        
        graph_widget = self.graph.widget
        # Enable drops
        try:
            graph_widget.setAcceptDrops(True)
            graph_widget.installEventFilter(self)
        except Exception:
            pass
        self._graph_widget = graph_widget
        center_splitter.addWidget(graph_widget)
        center_splitter.setSizes([600, 400])
        
        # Setup context menu for the graph
        try:
            menu = self.graph.get_context_menu('graph')
            menu.add_command('Fit to View', self._fit_all, 'F')
        except Exception:
            pass

        # Right panel
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.properties = PropertiesPanel()
        try:
            self.properties.property_changed.connect(self._on_property_changed)
        except Exception:
            pass
        self.results = ResultsPanel()
        self.timeline = TimelinePanel()
        right_splitter.addWidget(self.properties)
        right_splitter.addWidget(self.results)
        right_splitter.addWidget(self.timeline)
        right_splitter.setSizes([380, 280, 180])
        
        # Main splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([200, 1000, 300])
        
        main_h_layout.addWidget(main_splitter)
        
        # Set central widget for QMainWindow
        self.setCentralWidget(content_widget)

    def _active_interactive_face_node(self):
        """Return the interactive face-selection node implied by the current UI."""
        candidates = []
        try:
            candidates.extend(list(self.graph.selected_nodes()))
        except Exception:
            pass

        for attr in ('current_node',):
            node = getattr(self.properties, attr, None)
            if node is not None and node not in candidates:
                candidates.append(node)

        last = getattr(self, '_last_rendered_node', None)
        if last is not None and last not in candidates:
            candidates.append(last)

        for node in candidates:
            if node is not None and node.__class__.__name__ == 'InteractiveSelectFaceNode':
                return node

        try:
            all_interactive = [
                n for n in self.graph.all_nodes()
                if n.__class__.__name__ == 'InteractiveSelectFaceNode'
            ]
        except Exception:
            all_interactive = []

        if len(all_interactive) == 1:
            return all_interactive[0]
        return None

    def _start_viewer_face_picking(self):
        """Start face picking from the VTK viewer toolbar."""
        if self._execution_is_active():
            message = "Wait for the current CAD preview to finish before picking faces."
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.information(self, "Preview In Progress", message)
            return

        node = self._active_interactive_face_node()
        if node is None:
            QtWidgets.QMessageBox.information(
                self,
                "Select Face Picking Node",
                "Add or select a Select Face (Interactive) node connected to the shape, "
                "then click Pick Faces again.",
            )
            return

        self.properties.display_node(node)
        self._last_rendered_node = node

        _, geometry = self._get_render_context_for_node(node)
        if geometry is None:
            self._execute_graph(skip_simulation=True)
            self.statusBar().showMessage(
                "Updating CAD preview for face picking; click Pick Faces again when it finishes."
            )
            return

        self._render_result_in_viewer(geometry)
        self.properties._start_picking_session(node)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PySide6.QtGui import QShortcut, QKeySequence
        
        # File shortcuts
        QShortcut(QKeySequence.New, self, self._new_project)
        QShortcut(QKeySequence.Open, self, self._open_project)
        QShortcut(QKeySequence.Save, self, self._save_project)
        
        # Edit shortcuts
        QShortcut(QKeySequence.Undo, self, self._undo)
        QShortcut(QKeySequence.Redo, self, self._redo)
        QShortcut(QKeySequence.Delete, self, self._delete_selected)
        
        # Custom shortcuts
        QShortcut(QKeySequence("F5"), self, self._execute_graph)  # Run
        QShortcut(QKeySequence("F"), self, self._fit_all)  # Fit all
        QShortcut(QKeySequence("R"), self, self._reset_view)  # Reset view
    
    def _setup_toolbar(self):
        """Create toolbar and add to layout."""
        try:
            import qtawesome as qta
        except Exception:
            qta = None

        def _icon(name, color="#E0E0E0"):
            if qta is None:
                return QtGui.QIcon()
            try:
                return qta.icon(name, color=color)
            except Exception:
                return QtGui.QIcon()

        self.toolbar = QtWidgets.QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.setStyleSheet(
            """
            QToolBar { spacing: 4px; padding: 4px; border: 0; }
            QToolButton { padding: 4px 8px; border-radius: 4px; }
            QToolButton:hover { background: rgba(255,255,255,18); }
            QToolBar::separator { background: rgba(255,255,255,28); width: 1px; margin: 4px 4px; }
            """
        )
        self.addToolBar(self.toolbar)

        # ── File / project ──────────────────────────────────────────────
        self.toolbar.addAction(_icon("fa5s.file"),       "New",    self._new_project).setShortcut("Ctrl+N")
        self.toolbar.addAction(_icon("fa5s.folder-open"),"Open",   self._open_project).setShortcut("Ctrl+O")
        self.toolbar.addAction(_icon("fa5s.save"),       "Save",   self._save_project).setShortcut("Ctrl+S")
        act_import = self.toolbar.addAction(_icon("fa5s.file-import"), "Import Geometry", self._import_cad)
        act_import.setToolTip("Import a STEP / IGES / STL / OBJ file as a node")
        self.toolbar.addSeparator()

        # ── Edit (icon-only — short labels add noise) ───────────────────
        act_undo = self.toolbar.addAction(_icon("fa5s.undo"), "", self._undo)
        act_undo.setToolTip("Undo (Ctrl+Z)")
        act_redo = self.toolbar.addAction(_icon("fa5s.redo"), "", self._redo)
        act_redo.setToolTip("Redo (Ctrl+Y)")
        self.toolbar.addSeparator()

        # ── Run is the most-used action — give it accent color via icon ─
        run_act = self.toolbar.addAction(_icon("fa5s.play", "#66BB6A"), "Run", self._execute_graph)
        run_act.setShortcut("F5")
        run_act.setToolTip("Execute the node graph (F5)")

        self.toolbar.addAction(_icon("fa5s.check-circle", "#4FC3F7"), "Validate", self._validate_model
                               ).setToolTip("Check the graph for disconnected nodes and obvious mistakes")
        self.toolbar.addAction(_icon("fa5s.clipboard-list"), "Report", self._generate_report
                               ).setToolTip("Generate a text summary of the current model")
        self.toolbar.addAction(_icon("fa5s.file-export"), "Export Results", self._export_simulation_results
                               ).setToolTip("Save FEA / TopOpt / Crash results to disk")
        self.toolbar.addSeparator()

        # ── View ────────────────────────────────────────────────────────
        self.toolbar.addAction(_icon("fa5s.expand"), "", self._fit_all
                               ).setToolTip("Fit view to all (F)")

        # ── Right-aligned cluster: Auto-update toggle ───────────────────
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        self.auto_update_cb = QtWidgets.QCheckBox("Auto-update")
        self.auto_update_cb.setChecked(True)  # Default ON for CAD rendering
        self.auto_update_cb.setToolTip("Re-execute the graph automatically when properties change (skips FEA/TopOpt)")
        self.toolbar.addWidget(self.auto_update_cb)
    
    def _spawn_node(self, node_id, label, x=None, y=None):
        """Spawn a new node in the graph."""
        # Choose a default position if none provided
        existing = [n.pos() for n in self.graph.all_nodes()]
        x0, y0 = 250, 50
        if existing:
            y0 = max(n[1] for n in existing) + 150
        if x is None:
            x = x0
        if y is None:
            y = y0
        
        node_classes = NODE_CLASS_MAPPING
        
        node_class = node_classes.get(node_id)
        if not node_class:
            self.statusBar().showMessage(f"Unknown node: {node_id}")
            return
        
        try:
            node = node_class()
            node.set_name(label)
            node.set_pos(x, y)
            self.graph.add_node(node)
            # record undo action
            try:
                self._push_undo({'type': 'add_node', 'node': node})
            except Exception:
                pass

            self.timeline.add_event(f"Added {label} node")
            self.statusBar().showMessage(f"✓ Created {label}")
            return node
        except Exception as e:
            self.statusBar().showMessage(f"✗ Error: {e}")
            return None
            
    def _import_cad(self):
        """Prompt user for a geometry file, add an import node, and set its filepath."""
        try:
            from pylcss.io_manager.cad_io import CADImporter
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import Geometry File", "", CADImporter.get_filter_string()
            )
            if not filepath:
                return

            ext = os.path.splitext(filepath)[1].lower()
            if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
                node = self._spawn_node("com.cad.import_step", f"Import {ext.upper()[1:]}")
                if node:
                    node.set_property("filepath", filepath)
                    self._execute_graph()
            elif ext in (".stl", ".obj", ".3mf"):
                node = self._spawn_node("com.cad.import_stl", f"Import {ext.upper()[1:]}")
                if node:
                    node.set_property("filepath", filepath)
                    self._execute_graph()
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported Format", f"Format {ext} not supported for direct node importing.")
        except Exception as e:
            self.statusBar().showMessage(f"Error importing geometry: {e}")
    
    def _is_simulation_render_result(self, obj):
        """Return True for objects that should be drawn via render_simulation()."""
        if obj is None:
            return False
        if hasattr(obj, 'p') and hasattr(obj, 't'):
            return True
        if isinstance(obj, dict):
            return 'mesh' in obj or (obj.get('type') == 'crash' and bool(obj.get('frames')))
        return False

    def _is_renderable_result(self, obj):
        """Return True when the viewer can draw *obj* directly."""
        if obj is None:
            return False
        if self._is_simulation_render_result(obj):
            return True
        if hasattr(obj, 'tessellate') or hasattr(obj, 'val') or hasattr(obj, 'toCompound'):
            return True
        if hasattr(obj, 'ctx') and hasattr(obj.ctx, 'pendingWires'):
            return bool(obj.ctx.pendingWires)
        return False

    def _render_result_in_viewer(self, obj):
        """Render a previously resolved object using the right viewer path."""
        if obj is None:
            return
        if self._is_simulation_render_result(obj):
            self.viewer.render_simulation(obj)
            if isinstance(obj, dict):
                try:
                    self.results.show_result(obj)
                except Exception:
                    pass
        elif self._is_2d_sketch(obj):
            self.viewer.render_sketch(obj)
        else:
            self.viewer.render_shape(obj)

    @staticmethod
    def _port_name(port):
        try:
            return port.name()
        except Exception:
            return ''

    def _preferred_render_ports(self, node):
        cls = node.__class__.__name__
        if cls in ('ConstraintNode', 'LoadNode', 'PressureLoadNode'):
            return ('target_face', 'mesh', 'shape')
        if cls == 'ImpactConditionNode':
            return ('impact_face', 'mesh', 'shape')
        if cls in ('SelectFaceNode', 'InteractiveSelectFaceNode', 'MeshNode'):
            return ('shape',)
        return ()

    def _ordered_input_ports(self, node, preferred_names=()):
        try:
            ports = node.input_ports()
            if isinstance(ports, dict):
                ports = list(ports.values())
            else:
                ports = list(ports)
        except Exception:
            return []
        preferred = []
        rest = []
        preferred_names = tuple(preferred_names or ())
        for port in ports:
            if self._port_name(port) in preferred_names:
                preferred.append(port)
            else:
                rest.append(port)
        preferred.sort(key=lambda p: preferred_names.index(self._port_name(p)))
        return preferred + rest

    def _find_upstream_renderable(self, node, visited=None, preferred_ports=()):
        """Walk upstream and return (source_node, renderable_result)."""
        if node is None:
            return None, None
        if visited is None:
            visited = set()
        marker = id(node)
        if marker in visited:
            return None, None
        visited.add(marker)

        result = getattr(node, '_last_result', None)
        if self._is_renderable_result(result):
            return node, result

        ports = self._ordered_input_ports(
            node,
            preferred_ports or self._preferred_render_ports(node),
        )
        for port in ports:
            try:
                connected_ports = list(port.connected_ports())
            except Exception:
                connected_ports = []
            for conn_port in connected_ports:
                upstream = conn_port.node()
                upstream_result = getattr(upstream, '_last_result', None)
                if self._is_renderable_result(upstream_result):
                    return upstream, upstream_result
                source, renderable = self._find_upstream_renderable(upstream, visited)
                if renderable is not None:
                    return source, renderable
        return None, None

    def _get_render_context_for_node(self, node):
        """Return the best render target for a selected graph node."""
        result = getattr(node, '_last_result', None)
        if self._is_renderable_result(result):
            return node, result
        return self._find_upstream_renderable(
            node,
            preferred_ports=self._preferred_render_ports(node),
        )

    def _on_node_selected(self, node):
        """Handle node selection."""
        if node:
            # --- PATCH: force-update graph model combo items for TopOpt ---
            # NodeGraphQt bakes combo items into the graph model at node-creation
            # time. If the .cad file was saved with an older items list, the new
            # options won't appear.  We patch ALL TopOpt combos here before
            # calling display_node so stale .cad files always show current items.
            try:
                if node.type_ == 'com.cad.sim.topopt':
                    graph_model = node.model._graph_model
                    if graph_model is not None:
                        common = graph_model.get_node_common_properties(node.type_)
                        if common:
                            _topopt_combos = {
                                'visualization': ['Density', 'Recovered Shape', 'Von Mises Stress'],
                                'element_type':  ['Fast (Linear P1)', 'Accurate (Quadratic P2)'],
                                'filter_type':   ['sensitivity', 'density'],
                                'update_scheme': ['MMA', 'OC'],
                                'projection':    ['None', 'Heaviside'],
                            }
                            for prop, items in _topopt_combos.items():
                                if prop in common:
                                    common[prop]['items'] = items
            except Exception:
                pass
            # ---------------------------------------------------------------

            self.properties.display_node(node)
            self.statusBar().showMessage(f"Selected: {node.name}")


            _, geometry = self._get_render_context_for_node(node)

            if geometry is not None:
                self._last_rendered_node = node
                self._render_result_in_viewer(geometry)

                # Re-apply face highlights if it's the interactive picker
                if node.__class__.__name__ == 'InteractiveSelectFaceNode':
                    raw = node.get_property('picked_face_indices') or ''
                    idx_list = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
                    if idx_list:
                        if hasattr(self.viewer, 'highlight_faces'):
                            self.viewer.highlight_faces(idx_list)

                # Show BC overlays for this node (load/support highlights + arrows)
                try:
                    self._show_bc_for_node(node)
                except Exception:
                    pass
            else:
                # No cached result yet — execute the shape pipeline automatically
                # (skip_simulation=True means heavy FEA/TopOpt nodes are skipped,
                # so this is fast and behaves the same as when any property changes).
                self._last_rendered_node = node
                self._execute_graph(skip_simulation=True)

    def _get_upstream_shape(self, node):
        """Walk input ports to find the first cached shape result upstream."""
        _, result = self._find_upstream_renderable(
            node,
            preferred_ports=self._preferred_render_ports(node),
        )
        if result is not None and not self._is_simulation_render_result(result):
            return result
        return None
        try:
            for port in node.input_ports():
                for conn_port in port.connected_ports():
                    upstream = conn_port.node()
                    upstream_result = getattr(upstream, '_last_result', None)
                    if upstream_result is None:
                        continue
                    # Skip face-dicts — keep walking up
                    if isinstance(upstream_result, dict) and 'faces' in upstream_result:
                        shape = self._get_upstream_shape(upstream)
                        if shape is not None:
                            return shape
                    # Return if it looks like a renderable shape
                    if hasattr(upstream_result, 'tessellate') or hasattr(upstream_result, 'val'):
                        return upstream_result
                    if hasattr(upstream_result, 'toCompound'):
                        return upstream_result
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # BC OVERLAY HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_face_centroid(occ_face):
        """Compute centroid of an OCC face from its bounding-box mid-point."""
        try:
            bb = occ_face.BoundingBox()
            return [
                (bb.xmin + bb.xmax) / 2.0,
                (bb.ymin + bb.ymax) / 2.0,
                (bb.zmin + bb.zmax) / 2.0,
            ]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _collect_bc_for_node(self, anchor_node):
        """
        Walk the entire graph and collect load/constraint BC data that feeds
        into (or is associated with) *anchor_node*.

        Returns:
            (constraint_faces, load_faces, load_vectors)
            Each is a list (possibly empty). load_vectors is a list of
            ([cx,cy,cz], [fx,fy,fz]) tuples.
        """
        constraint_faces = []
        load_faces = []
        load_vectors = []

        anchor_cls = anchor_node.__class__.__name__
        solver_overlay_scope = None
        if anchor_cls in ('SolverNode', 'CrashSolverNode', 'TopologyOptimizationNode'):
            solver_overlay_scope = set()

            def _walk_upstream(n):
                marker = id(n)
                if marker in solver_overlay_scope:
                    return
                solver_overlay_scope.add(marker)
                try:
                    ports = n.input_ports()
                    if isinstance(ports, dict):
                        ports = list(ports.values())
                except Exception:
                    ports = []
                for port in ports:
                    try:
                        connected = list(port.connected_ports())
                    except Exception:
                        connected = []
                    for conn_port in connected:
                        try:
                            _walk_upstream(conn_port.node())
                        except Exception:
                            pass

            _walk_upstream(anchor_node)

        def _get_faces_for_bc_node(bc_node, extra_port_names=None):
            """Try two sources: (1) _last_result geometries, (2) named input port to SelectFaceNode.

            extra_port_names: additional port names to walk (e.g. ['impact_face'])
            in addition to the default 'target_face'.
            """
            faces = []
            # Source 1: cached result of BC node itself (populated after full run)
            try:
                result = getattr(bc_node, '_last_result', None)
                if isinstance(result, dict):
                    geoms = result.get('geometries') or []
                    # PressureLoadNode and older ConstraintNode use 'geometry' (singular)
                    if not geoms and result.get('geometry'):
                        geoms = [result['geometry']]
                    # ImpactConditionNode stores faces under 'face_list'
                    if not geoms:
                        geoms = result.get('face_list') or []
                    faces.extend([g for g in geoms if g is not None])
            except Exception:
                pass

            # Source 2: walk named input ports to SelectFaceNode result
            if not faces:
                port_names_to_check = {'target_face'}
                if extra_port_names:
                    port_names_to_check.update(extra_port_names)
                try:
                    for port in bc_node.input_ports():
                        try:
                            pname = port.name()
                        except Exception:
                            pname = ''
                        if pname not in port_names_to_check:
                            continue
                        for conn_port in port.connected_ports():
                            upstream = conn_port.node()
                            upstream_result = getattr(upstream, '_last_result', None)
                            if isinstance(upstream_result, dict):
                                sel_faces = upstream_result.get('faces') or []
                                faces.extend([f for f in sel_faces if f is not None])
                                if not faces and upstream_result.get('face') is not None:
                                    faces.append(upstream_result['face'])
                except Exception:
                    pass

            return faces

        def _get_mesh_for_bc_node(bc_node):
            """Return the mesh connected to a BC node, if it has already run."""
            try:
                mesh_port = bc_node.get_input('mesh')
                if mesh_port:
                    for conn_port in mesh_port.connected_ports():
                        upstream = conn_port.node()
                        mesh = getattr(upstream, '_last_result', None)
                        if mesh is not None and hasattr(mesh, 'p') and hasattr(mesh, 't'):
                            return mesh
            except Exception:
                pass
            return None

        def _condition_centroid_and_points(bc_node):
            """Centroid/sample points for legacy condition-expression BCs."""
            condition = ''
            try:
                condition = str(bc_node.get_property('condition') or '').strip()
            except Exception:
                condition = ''
            if not condition:
                return None, []
            mesh = _get_mesh_for_bc_node(bc_node)
            if mesh is None:
                return None, []
            try:
                import numpy as _np
                from pylcss.solver_backends.common import nodes_matching_condition

                node_ids = nodes_matching_condition(mesh, condition)
                if node_ids.size == 0:
                    return None, []
                pts = _np.asarray(mesh.p, dtype=float).T[node_ids]
                centroid = _np.mean(pts, axis=0).tolist()
                if len(pts) <= 5:
                    samples = pts.tolist()
                else:
                    order = _np.linspace(0, len(pts) - 1, 5, dtype=int)
                    samples = pts[order].tolist()
                return centroid, samples
            except Exception:
                return None, []

        for node in self.graph.all_nodes():
            cls = node.__class__.__name__
            if solver_overlay_scope is not None and id(node) not in solver_overlay_scope:
                continue

            # ---- ConstraintNode ----
            if cls == 'ConstraintNode':
                if anchor_cls == 'ConstraintNode' and node is not anchor_node:
                    continue
                if anchor_cls == 'LoadNode':
                    continue
                faces = _get_faces_for_bc_node(node)
                # FIX #V4: Carry viz metadata (per-type color) alongside the face
                # object so the viewer can render each constraint type distinctly.
                result = getattr(node, '_last_result', None)
                viz_meta = (result.get('viz') if isinstance(result, dict) else None) or {}
                for g in faces:
                    constraint_faces.append({'face': g, 'viz': viz_meta})
                if not faces:
                    centroid, samples = _condition_centroid_and_points(node)
                    if centroid is not None:
                        if not viz_meta:
                            try:
                                fixed_dofs = list((result or {}).get('fixed_dofs') or [])
                            except Exception:
                                fixed_dofs = []
                            viz_meta = {
                                'constraint_type': node.get_property('constraint_type') or 'Fixed',
                                'color': '#2979FF',
                                'fixed_dofs': fixed_dofs or [0, 1, 2],
                            }
                        constraint_faces.append({
                            'pos': centroid,
                            'points': samples,
                            'viz': viz_meta,
                        })

            # ---- LoadNode ----
            elif cls == 'LoadNode':
                if anchor_cls == 'LoadNode' and node is not anchor_node:
                    continue
                if anchor_cls == 'ConstraintNode':
                    continue
                faces = _get_faces_for_bc_node(node)
                try:
                    fx = float(node.get_property('force_x') or 0.0)
                    fy = float(node.get_property('force_y') or 0.0)
                    fz = float(node.get_property('force_z') or 0.0)
                    vec = [fx, fy, fz]
                    import math as _m
                    force_mag = _m.sqrt(fx*fx + fy*fy + fz*fz)
                    for g in faces:
                        load_faces.append(g)
                        centroid = self._get_face_centroid(g)
                        # FIX #V2: pass magnitude for log-scaled arrow in the viewer
                        load_vectors.append({
                            'centroid': centroid,
                            'face': g,
                            'vector': vec,
                            'magnitude_N': force_mag,
                        })
                    if not faces and force_mag > 1e-9:
                        centroid, samples = _condition_centroid_and_points(node)
                        if centroid is not None:
                            load_vectors.append({
                                'centroid': centroid,
                                'points': samples,
                                'vector': vec,
                                'magnitude_N': force_mag,
                            })
                except Exception:
                    pass

            # ---- PressureLoadNode ─ yellow face highlight, no arrow (normal varies) ----
            elif cls == 'PressureLoadNode':
                if anchor_cls == 'ConstraintNode':
                    continue
                faces = _get_faces_for_bc_node(node)
                for g in faces:
                    load_faces.append(g)

            # ---- ImpactConditionNode ─ yellow face highlight + cyan velocity arrow ----
            elif cls == 'ImpactConditionNode':
                if anchor_cls == 'ConstraintNode':
                    continue
                faces = _get_faces_for_bc_node(node, extra_port_names={'impact_face'})
                try:
                    vx = float(node.get_property('velocity_x') or 0.0)
                    vy = float(node.get_property('velocity_y') or 0.0)
                    vz = float(node.get_property('velocity_z') or 0.0)
                except Exception:
                    vx, vy, vz = 0.0, 0.0, 0.0
                import math as _m
                v_mag = _m.sqrt(vx*vx + vy*vy + vz*vz)
                arrow_emitted = False
                for g in faces:
                    try:
                        load_faces.append(g)
                        centroid = self._get_face_centroid(g)
                        load_vectors.append({
                            'centroid': centroid,
                            'face': g,
                            'vector': [vx, vy, vz],
                            'magnitude_N': v_mag,
                            'color': '#00e5ff',
                        })
                        arrow_emitted = True
                    except Exception:
                        try:
                            load_faces.append(g)
                        except Exception:
                            pass
                # Fallback: draw arrow at centroid (0,0,0) if no face centroid found
                if not arrow_emitted and v_mag > 1e-9:
                    load_vectors.append({
                        'centroid': [0.0, 0.0, 0.0],
                        'vector': [vx, vy, vz],
                        'magnitude_N': v_mag,
                        'color': '#00e5ff',
                    })

        return constraint_faces, load_faces, load_vectors

    def _show_bc_for_node(self, node):
        """
        Collect all BC data visible from *node* and push it to both the viewer
        cache and render_bc_overlays().  Safe to call after render_shape() or
        render_simulation() — overlays are layered on top.
        """
        try:
            cls = node.__class__.__name__
            if cls in ('SelectFaceNode', 'InteractiveSelectFaceNode'):
                result = getattr(node, '_last_result', None)
                faces = []
                if isinstance(result, dict):
                    faces = [f for f in (result.get('faces') or []) if f is not None]
                    if not faces and result.get('face') is not None:
                        faces = [result.get('face')]
                if faces:
                    self.viewer.set_bc_overlay_data(load_faces=faces)
                    self.viewer.render_bc_overlays(load_faces=faces)
                    return

            c_faces, l_faces, l_vecs = self._collect_bc_for_node(node)
            has_any = bool(c_faces or l_faces or l_vecs)
            if has_any:
                self.viewer.set_bc_overlay_data(
                    constraint_faces=c_faces or None,
                    load_faces=l_faces or None,
                    load_vectors=l_vecs or None,
                )
                self.viewer.render_bc_overlays(
                    constraint_faces=c_faces or None,
                    load_faces=l_faces or None,
                    load_vectors=l_vecs or None,
                )
            else:
                # Nothing to show — clear any stale overlays
                self.viewer.set_bc_overlay_data()
                self.viewer.render_bc_overlays()
        except Exception:
            pass

    def _on_node_double_clicked(self, node):
        """Handle a double-click on a node on the graph canvas.

        For ``CadQueryCodeNode`` this opens the full-screen CAD code editor —
        the same one the inspector's *Edit Code…* button uses.  Other node
        types swallow the double-click so NodeGraphQt's default subgraph
        popup doesn't appear.
        """
        try:
            if node.__class__.__name__ == 'CadQueryCodeNode':
                # The inspector holds the editor-open helper.
                self.properties._open_cad_code_editor(node)
                self.timeline.add_event(f"Opened code editor for {node.name()}")
                return
        except Exception:
            # Fall through to the silent default behaviour on any error.
            pass
        try:
            node_label = node.name() if callable(node.name) else node.name
        except Exception:
            node_label = '<unknown>'
        self.timeline.add_event(f"Double-clicked {node_label} (Popup disabled)")


    def _on_graph_property_changed(self, node, prop_name, prop_value):
        """Handle property changes from the graph (including widgets on nodes)."""
        # Mark node as dirty so it re-executes
        setattr(node, '_dirty', True)

        # Update the properties panel if this node is selected.
        # Skip if the inspector itself triggered the change to avoid a reset loop.
        if self.properties.current_node == node and not self.properties._updating_property:
            self.properties.display_node(node)
        
        # SPECIAL CASE: Visualization mode changes should update display immediately
        # without requiring full graph re-execution
        if prop_name == 'visualization':
            cached_result = getattr(node, '_last_result', None)
            if cached_result is not None and isinstance(cached_result, dict):
                # Update the visualization_mode in the cached result
                cached_result['visualization_mode'] = prop_value
                
                # Re-render with updated visualization mode
                try:
                    self.viewer.render_simulation(cached_result)
                    try:
                        self._show_bc_for_node(node)
                    except Exception:
                        pass
                except Exception:
                    pass
                
                # Don't mark as dirty for visualization-only changes
                setattr(node, '_dirty', False)
                return
            
        # Auto-update if enabled (skip simulation nodes for performance)
        if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
            self._execute_graph(skip_simulation=True)
            
    def _on_connection_changed(self, port_in, port_out):
        """Handle connection changes (connect/disconnect)."""
        # Skip events during project loading to prevent spam
        if self._is_loading:
            return
            
        # Mark both nodes as dirty
        if port_in:
            node = port_in.node()
            setattr(node, '_dirty', True)
        if port_out:
            node = port_out.node()
            setattr(node, '_dirty', True)
            
        self.timeline.add_event("Connection changed")
        # Auto-execute with skip_simulation for fast CAD preview
        if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
            self._execute_graph(skip_simulation=True)

    def eventFilter(self, source, event):
        """Handle drag/drop events on the graph widget to spawn nodes at drop location."""
        try:
            if source is getattr(self, '_graph_widget', None):
                if event.type() == QtCore.QEvent.DragEnter or event.type() == QtCore.QEvent.DragMove:
                    mime = event.mimeData()
                    if mime and (mime.hasFormat('application/x-node-id') or mime.hasText()):
                        event.accept()
                        return True
                if event.type() == QtCore.QEvent.Drop:
                    mime = event.mimeData()
                    if mime and mime.hasFormat('application/x-node-id'):
                        node_id = bytes(mime.data('application/x-node-id')).decode('utf-8')
                        label = mime.text() or str(node_id)
                        pos = event.pos()
                        # spawn node using explicit coordinates
                        self._spawn_node(node_id, label, x=pos.x(), y=pos.y())
                        event.accept()
                        return True
        except Exception:
            pass
        return super(ProfessionalCadApp, self).eventFilter(source, event)

    def _on_execution_finished(self, results):
        """Called when the background thread completes."""
        self.worker = None
        # Lock before processing results
        self.result_mutex.lock()
        try:
            # 1. Unlock UI
            self.graph.widget.setEnabled(True)
            self.toolbar.setEnabled(True)
            self.statusBar().showMessage("✓ Computation Complete")
            self.timeline.add_event("Graph execution finished")

            # 2. Update Visualization (Must be done on Main Thread!)
            try:
                # Logic to find what to render (copied from your old _execute_graph)
                target_node = None
                selected = next(iter(self.graph.selected_nodes()), None)

                # Helper to check renderability
                def is_renderable(obj):
                    if obj is None: return False
                    if isinstance(obj, dict) and ('mesh' in obj or 'vertices' in obj): return True # Sim result
                    if hasattr(obj, 'tessellate'): return True # Shape
                    if hasattr(obj, 'val'): return True # Wrapper
                    # Check for 2D sketch (has pending wires but no solid)
                    if hasattr(obj, 'ctx') and hasattr(obj.ctx, 'pendingWires'):
                        return bool(obj.ctx.pendingWires)
                    return False

                if selected:
                    # Check current result
                    res = results.get(selected)
                    # Or check cached result if not in new results
                    if res is None:
                        res = getattr(selected, '_last_result', None)

                    if is_renderable(res):
                        target_node = selected
                    else:
                        _, upstream_geom = self._get_render_context_for_node(selected)
                        if upstream_geom is not None:
                            target_node = selected

                # Fallback to last rendered
                if target_node is None:
                    last = getattr(self, '_last_rendered_node', None)
                    if last:
                        target_node = last

                # Render
                if target_node:
                    self._last_rendered_node = target_node
                    # Get result from the results dict or the node cache
                    geom = results.get(target_node, getattr(target_node, '_last_result', None))
                    if not self._is_renderable_result(geom):
                        _, geom = self._get_render_context_for_node(target_node)

                    self._render_result_in_viewer(geom)

                    # Re-collect and cache BC overlays so they show on the
                    # freshly rendered shape / simulation result.
                    try:
                        self._show_bc_for_node(target_node)
                    except Exception:
                        pass
                else:
                    self.viewer.clear()

            except Exception:
                pass
        finally:
            self.result_mutex.unlock()

    def _is_2d_sketch(self, obj):
        """Check if object is a 2D sketch (has wires but NO solids)."""
        if obj is None:
            return False
        
        # First check if this has solids - if so, it's a 3D shape, NOT a sketch
        try:
            # CadQuery Workplane with solids
            if hasattr(obj, 'val'):
                val = obj.val()
                # Check for solid or compound
                if hasattr(val, 'Solids') and val.Solids():
                    return False  # Has solids = 3D shape
            # Or direct solid
            if hasattr(obj, 'Solids') and obj.Solids():
                return False  # Has solids = 3D shape
        except Exception:
            pass
        
        # Now check for pending wires (2D sketch)
        # Only treat as sketch if there are wires but no solids
        if hasattr(obj, 'ctx') and hasattr(obj.ctx, 'pendingWires'):
            if obj.ctx.pendingWires:
                return True
        return False

    def _on_execution_error(self, error_msg):
        """Called if background thread fails."""
        self.worker = None
        self.graph.widget.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.statusBar().showMessage(f"❌ Error: {error_msg}")
        self.timeline.add_event(f"Execution failed: {error_msg}")
        try:
            self.viewer.set_bc_overlay_data()
            self.viewer.render_bc_overlays()
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Computation Error", error_msg)
    
    def _update_property(self, prop_name, value):
        """Update node property."""
        if self.properties.current_node:
            try:
                self.properties.current_node.set_property(prop_name, value)
                self.timeline.add_event(f"Updated {prop_name} = {value}")
            except Exception:
                pass
    
    def _undo(self):
        """Undo last action."""
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        action = self.undo_stack.pop()
        try:
            typ = action.get('type')
            if typ == 'add_node':
                node = action.get('node')
                # remove node
                try:
                    self.graph.remove_node(node)
                except Exception:
                    pass
                # push redo
                self.redo_stack.append({'type': 'add_node', 'node': node})
                self.timeline.add_event(f"Undid add node {getattr(node, 'name', '')}")
            elif typ == 'remove_nodes':
                nodes = action.get('nodes', [])
                for n in nodes:
                    try:
                        self.graph.add_node(n)
                        # restore position if available
                        try:
                            pos = action.get('positions', {}).get(id(n))
                            if pos:
                                n.set_pos(pos[0], pos[1])
                        except Exception:
                            pass
                    except Exception:
                        pass
                self.redo_stack.append(action)
                self.timeline.add_event(f"Undid delete of {len(nodes)} node(s)")
            elif typ == 'prop_change':
                node = action.get('node')
                prop = action.get('prop')
                old = action.get('old')
                new = action.get('new')
                try:
                    node.set_property(prop, old)
                except Exception:
                    pass
                # push redo
                self.redo_stack.append({'type': 'prop_change', 'node': node, 'prop': prop, 'old': new, 'new': old})
                self.timeline.add_event(f"Undid property {prop} on {getattr(node,'name','node')}")
            else:
                self.timeline.add_event('Unknown undo action')
        except Exception:
            pass
        self.statusBar().showMessage("Undo")
    
    def _redo(self):
        """Redo last action."""
        if not self.redo_stack:
            self.statusBar().showMessage("Nothing to redo")
            return
        action = self.redo_stack.pop()
        try:
            typ = action.get('type')
            if typ == 'add_node':
                node = action.get('node')
                try:
                    self.graph.add_node(node)
                except Exception:
                    pass
                self.undo_stack.append(action)
                self.timeline.add_event(f"Redid add node {getattr(node,'name','')}")
            elif typ == 'remove_nodes':
                nodes = action.get('nodes', [])
                positions = action.get('positions', {})
                for n in nodes:
                    try:
                        self.graph.remove_node(n)
                    except Exception:
                        pass
                self.undo_stack.append(action)
                self.timeline.add_event(f"Redid delete of {len(nodes)} node(s)")
            elif typ == 'prop_change':
                node = action.get('node')
                prop = action.get('prop')
                new = action.get('new')
                try:
                    node.set_property(prop, new)
                except Exception:
                    pass
                self.undo_stack.append(action)
                self.timeline.add_event(f"Redid property {prop} on {getattr(node,'name','node')}")
            else:
                self.timeline.add_event('Unknown redo action')
        except Exception:
            pass
        self.statusBar().showMessage("Redo")
    
    def _delete_selected(self):
        """Delete selected nodes."""
        selected = list(self.graph.selected_nodes())
        if not selected:
            return

        # record positions so undo can restore
        positions = {id(n): n.pos() for n in selected}
        try:
            self._push_undo({'type': 'remove_nodes', 'nodes': selected, 'positions': positions})
        except Exception:
            pass

        for node in selected:
            try:
                self.graph.remove_node(node)
            except Exception:
                pass
        self.timeline.add_event(f"Deleted {len(selected)} selected nodes")
    
    def _clear_graph(self):
        """Clear entire graph."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Clear All', 'Remove all nodes?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            # record all nodes for undo
            all_nodes = list(self.graph.all_nodes())
            positions = {id(n): n.pos() for n in all_nodes}
            try:
                self._push_undo({'type': 'remove_nodes', 'nodes': all_nodes, 'positions': positions})
            except Exception:
                pass
            self.graph.clear_session()
            self._last_rendered_node = None
            try:
                self.viewer.clear()
            except Exception:
                pass
            self.timeline.add_event("Cleared graph")
    
    def _fit_all(self):
        """Fit all nodes in view."""
        try:
            # Fit the node graph view
            self.graph.fit_to_selection()
            # If no selection, center on all nodes
            if not self.graph.selected_nodes():
                self.graph.center_on_nodes(self.graph.all_nodes())
            self.statusBar().showMessage("✓ Fit to view")
        except Exception as e:
            # Fallback - try basic centering
            try:
                self.graph.center_selection()
            except:
                pass
            self.statusBar().showMessage("✓ View adjusted")
    
    def _reset_view(self):
        """Reset the 3D viewer to default orientation."""
        try:
            # Reset the 3D viewer camera
            if hasattr(self.viewer, 'renderer') and self.viewer.renderer:
                self.viewer.renderer.ResetCamera()
                if hasattr(self.viewer, 'iren') and self.viewer.iren:
                    self.viewer.iren.GetRenderWindow().Render()
            self.statusBar().showMessage("✓ 3D View reset")
            self.timeline.add_event("3D view reset to default")
        except Exception as e:
            self.statusBar().showMessage("View reset")
    
    def _run_simulation(self):
        """Run simulation by executing the graph and finding simulation nodes."""
        self.statusBar().showMessage("Running simulation...")
        self.timeline.add_event("Simulation started")
        
        # Find simulation-related nodes (Solver, TopOpt, etc.)
        sim_nodes = []
        for node in self.graph.all_nodes():
            node_class = node.__class__.__name__
            if node_class in ['SolverNode', 'TopologyOptimizationNode', 'MeshNode', 'CrashSolverNode']:
                sim_nodes.append(node)
        
        if not sim_nodes:
            QtWidgets.QMessageBox.information(
                self, "No Simulation",
                "No simulation nodes found in the graph.\n\n"
                "Add FEA nodes (Material, Mesh, Constraint, Load, Solver) "
                "or a Topology Optimization node to run a simulation."
            )
            self.statusBar().showMessage("No simulation nodes found")
            return
        
        # Execute the graph which will run the simulation
        self._execute_graph()
        self.timeline.add_event(f"Simulation executed ({len(sim_nodes)} sim nodes)")
    
    def _generate_report(self):
        """Generate a report from the model with node information."""
        self.statusBar().showMessage("Generating report...")
        self.timeline.add_event("Report generation started")
        
        # Collect model information
        all_nodes = list(self.graph.all_nodes())
        if not all_nodes:
            QtWidgets.QMessageBox.information(
                self, "Empty Model",
                "No nodes in the graph to report on."
            )
            self.statusBar().showMessage("No nodes to report")
            return
        
        # Build report content
        report_lines = [
            "=" * 60,
            "CAD MODEL REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {self.current_file or 'Unsaved'}",
            "",
            f"Total Nodes: {len(all_nodes)}",
            "",
            "NODE SUMMARY:",
            "-" * 40,
        ]
        
        # Categorize nodes
        node_types = {}
        for node in all_nodes:
            node_class = node.__class__.__name__
            node_types[node_class] = node_types.get(node_class, 0) + 1
        
        for node_type, count in sorted(node_types.items()):
            report_lines.append(f"  {node_type}: {count}")
        
        report_lines.extend([
            "",
            "NODE DETAILS:",
            "-" * 40,
        ])
        
        for node in all_nodes:
            report_lines.append(f"  [{node.__class__.__name__}] {node.name()}")
            # Add key properties
            try:
                props = node.model.properties
                for key, val in list(props.items())[:5]:  # First 5 properties
                    if not key.startswith('_'):
                        report_lines.append(f"      {key}: {val}")
            except:
                pass
        
        report_lines.append("\n" + "=" * 60)
        report_text = "\n".join(report_lines)
        
        # Show in a dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Model Report")
        dialog.resize(600, 500)
        layout = QtWidgets.QVBoxLayout(dialog)
        
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(report_text)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(text_edit)
        
        # Save button
        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save Report...")
        close_btn = QtWidgets.QPushButton("Close")
        
        def save_report():
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog, "Save Report", "model_report.txt", "Text Files (*.txt)"
            )
            if fname:
                with open(fname, 'w') as f:
                    f.write(report_text)
                self.statusBar().showMessage(f"✓ Report saved to {fname}")
        
        save_btn.clicked.connect(save_report)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()
        self.statusBar().showMessage("✓ Report generated")
        self.timeline.add_event("Report generated")
    
    def _validate_model(self):
        """Validate the current model for issues."""
        self.statusBar().showMessage("Validating model...")
        self.timeline.add_event("Model validation started")
        
        issues = []
        warnings = []
        
        all_nodes = list(self.graph.all_nodes())
        
        if not all_nodes:
            issues.append("Model is empty - no nodes found")
        else:
            # Check for disconnected nodes
            for node in all_nodes:
                has_input = False
                has_output = False
                
                for port in node.input_ports():
                    if port.connected_ports():
                        has_input = True
                        break
                
                for port in node.output_ports():
                    if port.connected_ports():
                        has_output = True
                        break
                
                # Primitive nodes don't need inputs
                node_class = node.__class__.__name__
                is_primitive = node_class in ['BoxNode', 'CylinderNode', 'SphereNode', 
                                               'ConeNode', 'TorusNode', 'NumberNode']
                is_export = 'Export' in node_class
                
                if not is_primitive and not has_input:
                    warnings.append(f"{node.name()} has no connected inputs")
                
                if not is_export and not has_output:
                    # Check if it's a terminal node (not an issue)
                    if node_class not in ['SolverNode', 'TopologyOptimizationNode']:
                        pass  # Non-terminal nodes without outputs are fine
            
            # Check for simulation setup
            has_mesh = any(n.__class__.__name__ == 'MeshNode' for n in all_nodes)
            has_solver = any(n.__class__.__name__ == 'SolverNode' for n in all_nodes)
            has_material = any(n.__class__.__name__ == 'MaterialNode' for n in all_nodes)
            has_constraint = any(n.__class__.__name__ == 'ConstraintNode' for n in all_nodes)
            has_load = any(n.__class__.__name__ == 'LoadNode' for n in all_nodes)
            
            if has_solver:
                if not has_mesh:
                    issues.append("Solver requires a Mesh node")
                if not has_material:
                    issues.append("Solver requires a Material node")
                if not has_constraint:
                    warnings.append("Solver may need constraint nodes (fixed supports)")
                if not has_load:
                    warnings.append("Solver may need load nodes")
        
        # Show results
        if not issues and not warnings:
            QtWidgets.QMessageBox.information(
                self, "Validation Complete",
                "✓ Model is valid!\n\n"
                f"Total nodes: {len(all_nodes)}"
            )
            self.statusBar().showMessage("✓ Model valid")
        else:
            msg = ""
            if issues:
                msg += "ERRORS:\n" + "\n".join(f"  ❌ {i}" for i in issues) + "\n\n"
            if warnings:
                msg += "WARNINGS:\n" + "\n".join(f"  ⚠️ {w}" for w in warnings)
            
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Validation Results")
            box.setText(f"Found {len(issues)} errors and {len(warnings)} warnings")
            box.setDetailedText(msg)
            box.setIcon(QtWidgets.QMessageBox.Warning if issues else QtWidgets.QMessageBox.Information)
            box.exec()
            
            if issues:
                self.statusBar().showMessage(f"⚠️ {len(issues)} validation errors")
            else:
                self.statusBar().showMessage(f"✓ Valid with {len(warnings)} warnings")
        
        self.timeline.add_event(f"Validation: {len(issues)} errors, {len(warnings)} warnings")
    
    def _new_project(self):
        """Create a new project."""
        if not self._ensure_idle_for_io("creating a new project"):
            return

        self.graph.clear_session()
        self.current_file = None
        self._last_rendered_node = None
        try:
            self.viewer.clear()
        except Exception:
            pass
        self.timeline.add_event("New project created")
        self.statusBar().showMessage("New project")
    
    def _open_project(self):
        """Open a project file."""
        if not self._ensure_idle_for_io("opening a project"):
            return

        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", "", "Design Projects (*.cad);;All Files (*)"
        )
        if fname:
            # Disable auto-update and set loading flag to suppress events
            was_auto_update_enabled = self.auto_update_cb.isChecked()
            self.auto_update_cb.setChecked(False)
            self._is_loading = True
            previous_file = self.current_file
            backup_session = None
            try:
                backup_session = self.graph.serialize_session()
            except Exception:
                backup_session = None
            
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                self.graph.clear_session()
                self.graph.deserialize_session(session_data)

                self._last_rendered_node = None
                try:
                    self.viewer.clear()
                except Exception:
                    pass
                
                # Automatically fit all nodes in view after loading
                try:
                    self._fit_all()
                except Exception:
                    pass
                
                self.current_file = fname
                self.timeline.add_event(f"Opened project: {fname}")
                self.statusBar().showMessage(f"Opened: {fname}")
                
                # Execute to restore view (but skip heavy simulation)
                self._execute_graph(skip_simulation=True)
            except Exception as e:
                self.current_file = previous_file
                if backup_session is not None:
                    try:
                        self.graph.clear_session()
                        self.graph.deserialize_session(backup_session)
                        self._fit_all()
                    except Exception:
                        pass
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open project: {e}")
            finally:
                self._is_loading = False
                self.auto_update_cb.setChecked(was_auto_update_enabled)

    def _execution_is_active(self):
        return bool(self.worker and self.worker.isRunning())

    def _ensure_idle_for_io(self, action_name):
        if not self._execution_is_active():
            return True

        message = f"Wait for the current computation to finish before {action_name}."
        self.statusBar().showMessage(f"⚠️ {message}")
        QtWidgets.QMessageBox.information(self, "Computation In Progress", message)
        return False
    
    def _get_node_class(self, class_name):
        """Get node class by name."""
        return NODE_NAME_MAPPING.get(class_name)

    def _get_exportable_result_node(self):
        """Return the best candidate node with cached simulation results."""
        candidates = []
        selected = next(iter(self.graph.selected_nodes()), None)
        if selected is not None:
            candidates.append(selected)

        current = getattr(self.properties, 'current_node', None)
        if current is not None and current not in candidates:
            candidates.append(current)

        last = getattr(self, '_last_rendered_node', None)
        if last is not None and last not in candidates:
            candidates.append(last)

        for node in reversed(list(self.graph.all_nodes())):
            if node not in candidates:
                candidates.append(node)

        for node in candidates:
            result = getattr(node, '_last_result', None)
            if isinstance(result, dict) and result.get('mesh') is not None and result.get('type') in {'fea', 'topopt', 'crash'}:
                return node
        return None

    def _build_simulation_export_payload(self, node):
        """Create portable JSON/HDF5 payloads from a cached simulation result."""
        import numpy as np

        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict) or result.get('mesh') is None:
            raise ValueError("The selected node has no exportable simulation result.")

        mesh = result['mesh']
        if not hasattr(mesh, 'p') or not hasattr(mesh, 't'):
            raise ValueError("Only scikit-fem style simulation meshes are supported for export.")

        points = np.asarray(mesh.p.T, dtype=float)
        if points.ndim != 2:
            raise ValueError("Invalid mesh point array.")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        cells = np.asarray(mesh.t.T, dtype=int)
        if cells.ndim != 2:
            raise ValueError("Invalid mesh connectivity array.")

        n_points = points.shape[0]
        n_cells = cells.shape[0]
        nodes_per_cell = cells.shape[1]
        cell_type_map = {2: 'line', 3: 'triangle', 4: 'tetra', 8: 'hexahedron'}
        cell_type = cell_type_map.get(nodes_per_cell, f'{nodes_per_cell}-node')

        def _as_numeric_array(value):
            if value is None:
                return None
            arr = np.asarray(value)
            if arr.size == 0 or arr.dtype == object:
                return None
            return arr

        point_data = {}
        cell_data = {}
        history = {}
        recovered_shape = None

        displacement = _as_numeric_array(result.get('displacement'))
        displacement_vec = None
        if displacement is not None and displacement.ndim == 1 and displacement.size == 3 * n_points:
            displacement_vec = displacement.reshape(n_points, 3)
        elif displacement is not None and displacement.ndim == 2 and displacement.shape[0] == n_points:
            if displacement.shape[1] == 2:
                displacement_vec = np.column_stack([displacement, np.zeros(n_points)])
            elif displacement.shape[1] >= 3:
                displacement_vec = displacement[:, :3]

        if displacement_vec is not None:
            point_data['displacement'] = displacement_vec
            point_data['displacement_magnitude'] = np.linalg.norm(displacement_vec, axis=1)

        stress = _as_numeric_array(result.get('stress'))
        if stress is not None:
            if stress.ndim == 1 and stress.size == n_points:
                point_data['stress'] = stress
            elif stress.ndim == 1 and stress.size == n_cells:
                cell_data['stress'] = stress

        density = _as_numeric_array(result.get('density'))
        if density is not None:
            if density.ndim == 1 and density.size == n_cells:
                cell_data['density'] = density
            elif density.ndim == 1 and density.size == n_points:
                point_data['density'] = density

        design_density = _as_numeric_array(result.get('design_density'))
        if design_density is not None and design_density.ndim == 1 and design_density.size == n_cells:
            cell_data['design_density'] = design_density

        element_stress = _as_numeric_array(result.get('element_stress'))
        if element_stress is not None and element_stress.ndim == 1 and element_stress.size == n_cells:
            cell_data['element_stress'] = element_stress

        plastic_strain = _as_numeric_array(result.get('plastic_strain'))
        if plastic_strain is not None and plastic_strain.ndim == 1 and plastic_strain.size == n_cells:
            cell_data['plastic_strain'] = plastic_strain

        failed_elements = _as_numeric_array(result.get('failed_elements'))
        if failed_elements is not None and failed_elements.ndim == 1 and failed_elements.size == n_cells:
            cell_data['failed_elements'] = failed_elements.astype(np.int8)

        for key in ('time', 'energy_kinetic', 'energy_strain', 'energy_plastic', 'energy_balance'):
            arr = _as_numeric_array(result.get(key))
            if arr is not None:
                history[key] = arr

        topopt_shape = result.get('recovered_shape')
        if isinstance(topopt_shape, dict):
            vertices = _as_numeric_array(topopt_shape.get('vertices'))
            faces = _as_numeric_array(topopt_shape.get('faces'))
            if vertices is not None and faces is not None:
                recovered_shape = {
                    'vertices': vertices,
                    'faces': faces,
                }

        node_name = getattr(node, 'name', None)
        if callable(node_name):
            node_name = node_name()
        if not node_name:
            node_name = getattr(node, 'NODE_NAME', None) or node.__class__.__name__

        metadata = {
            'node_name': str(node_name),
            'node_class': node.__class__.__name__,
            'simulation_type': str(result.get('type', 'unknown')),
            'visualization_mode': str(result.get('visualization_mode', '')),
            'exported_at': datetime.now().isoformat(),
            'cell_type': cell_type,
            'point_count': int(n_points),
            'cell_count': int(n_cells),
        }

        summary = {}
        for key in (
            'peak_displacement',
            'peak_stress',
            'absorbed_energy',
            'n_failed',
            'energy_balance_max_error',
            'max_stress_gauss',
            'deformation_scale',
            'density_cutoff',
        ):
            value = result.get(key)
            if isinstance(value, (int, float)):
                summary[key] = float(value)

        json_payload = {
            'metadata': metadata,
            'summary': summary,
            'mesh': {
                'points': points.tolist(),
                'cell_type': cell_type,
                'cells': cells.tolist(),
            },
            'point_data': {name: np.asarray(values).tolist() for name, values in point_data.items()},
            'cell_data': {name: np.asarray(values).tolist() for name, values in cell_data.items()},
            'history': {name: np.asarray(values).tolist() for name, values in history.items()},
        }

        if recovered_shape is not None:
            json_payload['recovered_shape'] = {
                'vertices': recovered_shape['vertices'].tolist(),
                'faces': recovered_shape['faces'].tolist(),
            }

        hdf5_datasets = {
            'mesh/points': points,
            'mesh/cells': cells,
        }
        for name, values in point_data.items():
            hdf5_datasets[f'point_data/{name}'] = np.asarray(values)
        for name, values in cell_data.items():
            hdf5_datasets[f'cell_data/{name}'] = np.asarray(values)
        for name, values in history.items():
            hdf5_datasets[f'history/{name}'] = np.asarray(values)
        if recovered_shape is not None:
            hdf5_datasets['recovered_shape/vertices'] = recovered_shape['vertices']
            hdf5_datasets['recovered_shape/faces'] = recovered_shape['faces']

        return json_payload, hdf5_datasets, metadata

    def _export_simulation_results(self):
        """Export cached FEA/TopOpt/Crash results from the active node."""
        if not self._ensure_idle_for_io("exporting simulation results"):
            return

        node = self._get_exportable_result_node()
        if node is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Results",
                "Run or select an FEA, topology-optimization, or crash node before exporting results.",
            )
            return

        try:
            json_payload, hdf5_datasets, metadata = self._build_simulation_export_payload(node)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
            return

        node_name = metadata.get('node_name', 'simulation_results')
        safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in node_name).strip('_') or 'simulation_results'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Simulation Results",
            f"{safe_name}.json",
            "JSON Files (*.json);;HDF5 Files (*.h5)",
        )
        if not fname:
            return

        ext = os.path.splitext(fname)[1].lower()
        if not ext:
            fname += '.json'
            ext = '.json'

        try:
            from pylcss.io_manager.data_io import DataExporter

            if ext == '.json':
                DataExporter.to_json(fname, json_payload)
            elif ext == '.h5':
                attrs = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
                for key, value in json_payload.get('summary', {}).items():
                    attrs[f'summary_{key}'] = value
                DataExporter.to_hdf5(fname, hdf5_datasets, attrs=attrs)
            else:
                raise ValueError(f"Unsupported export format: {ext}")

            self.timeline.add_event(f"Exported simulation results: {fname}")
            self.statusBar().showMessage(f"Exported results: {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))


    # --- Undo stack helpers ---
    def _push_undo(self, action):
        try:
            self.undo_stack.append(action)
            # clear redo on new action
            self.redo_stack.clear()
        except Exception:
            pass

    def _on_property_changed(self, node, prop_name, old, new):
        try:
            self._push_undo({'type': 'prop_change', 'node': node, 'prop': prop_name, 'old': old, 'new': new})
            self.timeline.add_event(f"Property changed: {prop_name} = {new} ({getattr(node,'name','')})")
            
            # IMPORTANT: When a property changes, the modified node should be rendered
            # Store it as last_rendered_node so _on_execution_finished renders the right node
            self._last_rendered_node = node
            
            # OPTIMIZATION: Check if this is a visualization-only property change
            # These properties don't need a re-run, just a re-render
            visualization_only_props = ['visualization', 'density_cutoff', 'element_type', 'projection']
            
            if prop_name in visualization_only_props:
                # Check if the node has cached results (_last_result)
                cached_result = getattr(node, '_last_result', None)
                if cached_result is not None:
                    # Update the cached dictionary so the renderer knows what to draw
                    if isinstance(cached_result, dict):
                        if prop_name == 'visualization':
                            cached_result['visualization_mode'] = new
                        elif prop_name == 'density_cutoff':
                            cached_result['density_cutoff'] = new

                    # Just re-render with existing results instead of re-executing
                    try:
                        if isinstance(cached_result, dict) and ('mesh' in cached_result or 'displacement' in cached_result or 'recovered_shape' in cached_result):
                            self.viewer.render_simulation(cached_result)
                        elif hasattr(cached_result, 'p') and hasattr(cached_result, 't'):
                            # Direct Mesh object from skfem
                            self.viewer.render_simulation(cached_result)
                        else:
                            self.viewer.render_shape(cached_result)
                        try:
                            self._show_bc_for_node(node)
                        except Exception:
                            pass
                        if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                            self.window().statusBar().showMessage(f"✓ Updated {prop_name} display")
                        return  # Skip full graph execution
                    except Exception as e:
                        print(f"Warning: Render failed during viz update for {prop_name}: {e}")
                        
                # Unconditionally return for visualization properties to prevent full recompute
                return
            # Auto-execute if enabled (for non-visualization properties)
            if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
                self._execute_graph()
                
        except Exception:
            pass
    
    def _save_project(self):
        """Save current project."""
        if not self._ensure_idle_for_io("saving the project"):
            return

        if not self.current_file:
            self._save_as_project()
            return
        
        try:
            # Serialize graph using NodeGraphQt's built-in session manager
            project_data = self.graph.serialize_session()

            target_dir = os.path.dirname(self.current_file) or None
            fd, temp_path = tempfile.mkstemp(prefix='pylcss_cad_', suffix='.tmp', dir=target_dir)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.current_file)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            
            self.timeline.add_event(f"Saved project: {self.current_file}")
            self.statusBar().showMessage(f"Saved: {self.current_file}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
    
    def _save_as_project(self):
        """Save project with a new name."""
        if not self._ensure_idle_for_io("saving the project"):
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Design Projects (*.cad);;All Files (*)"
        )
        if fname:
            # Ensure .cad extension
            if not fname.endswith('.cad'):
                fname += '.cad'
            self.current_file = fname
            self._save_project()
    
    def _show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.information(
            self, "About",
        )
    
    def _execute_graph(self, skip_simulation=False):
        """Start graph execution in a background thread.
        
        Args:
            skip_simulation: If True, skip FEA/TopOpt nodes (for auto-update mode)
        """
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("⚠️ Computation already in progress...")
            return

        # Keep UI responsive during optimization (don't disable)
        # self.graph.widget.setEnabled(False)  # Removed for real-time viz
        # self.toolbar.setEnabled(False)  # Removed for real-time viz
        
        if skip_simulation:
            self.statusBar().showMessage("⏳ Updating design preview...")
        else:
            self.statusBar().showMessage("⏳ Computing... (watch 3D viewer for live updates)")
            self.timeline.add_event("Graph execution started (Full)")

        # Capture the list of nodes on the MAIN THREAD
        all_nodes_snapshot = list(self.graph.all_nodes())
        
        # Initialize worker with skip_simulation parameter
        self.worker = GraphExecutionWorker(all_nodes_snapshot, skip_simulation=skip_simulation, parent=self)

        self.worker.computation_finished.connect(self._on_execution_finished)
        self.worker.computation_error.connect(self._on_execution_error)
        # Connect optimization step for real-time visualization
        self.worker.optimization_step.connect(self._on_optimization_step)
        self.worker.start()

    @QtCore.Slot(bool)
    @QtCore.Slot()  # Allow calling without arguments (default=False)
    def execute_graph(self, skip_simulation=False):
        """Public alias for _execute_graph, allowed to be called by external agents."""
        self._execute_graph(skip_simulation)
    
    def _on_optimization_step(self, mesh, densities, step, total):
        """Update the 3D viewer with current optimization state (real-time viz)."""
        try:
            import numpy as np
            now = time.monotonic()
            is_final_step = (step + 1) >= total
            if not is_final_step and (now - self._last_preview_update_time) < 0.1:
                return

            self._last_preview_update_time = now
            self.statusBar().showMessage(f"⏳ TopOpt: Iteration {step+1}/{total} (Vol: {np.mean(densities):.1%})")
            
            # Update viewer with current density field
            result = {
                'mesh': mesh,
                'density': densities,
                'type': 'topopt',
                'visualization_mode': 'Density',
                'density_cutoff': 0.3,  # Use default cutoff for preview
                '_preview': True,
            }
            self.viewer.render_simulation(result)
            
        except Exception:
            pass


def main():
    """Launch the professional CAD software."""
    os.environ['QT_API'] = 'pyside6'
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    app.setPalette(palette)
    
    try:
        window = ProfessionalCadApp()
        window.show()
        sys.exit(app.exec())
    except RuntimeError as e:
        # Suppress NodeGraphQt internal errors
        if "layout direction not valid" in str(e):
            pass
            sys.exit(0)
        else:
            raise


if __name__ == '__main__':
    main()
