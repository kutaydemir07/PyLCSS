# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Engineering design studio - full-featured interface.

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
import logging
from datetime import datetime
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import cadquery as cq
from .cad_viewer import CQ3DViewer
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag
from NodeGraphQt import NodeGraph
from pylcss.design_studio.engine import execute_graph
from pylcss.design_studio.node_library import NODE_CLASS_MAPPING, NODE_NAME_MAPPING
from pylcss.design_studio.topology_optimization.presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_WORKFLOW_MODES,
    INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
)

logger = logging.getLogger(__name__)

try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None  # Fallback if not installed

# Import all node types
from pylcss.design_studio.nodes import NODE_REGISTRY, NumberNode, ExportStepNode, ExportStlNode
from pylcss.design_studio.nodes.modeling import InteractiveSelectFaceNode

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
            from pylcss.design_studio.engine import execute_graph
            
            # Callback for real-time updates
            def progress_cb(mesh, densities, step, total):
                if self._is_running:
                    self.optimization_step.emit(mesh, densities, step, total)

            # Pass skip_simulation and callback to engine
            results = execute_graph(
                self.nodes, 
                skip_simulation=self.skip_simulation,
                cancel_callback=lambda: not self._is_running,
                progress_callback=progress_cb
            )

            self.computation_finished.emit(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.computation_error.emit(str(e))
        finally:
            self._is_running = False
    def cancel(self):
        """Ask long-running simulation nodes to stop cleanly."""
        self._is_running = False
        self.requestInterruption()
        for node in self.nodes:
            request_stop = getattr(node, 'request_stop', None)
            if callable(request_stop):
                request_stop()




def _external_write_cad_step(payload, path):
    import cadquery as cq
    from pylcss.design_studio.topology_optimization.cad_reconstruction import reconstruct_topopt_cad
    shape = reconstruct_topopt_cad(
        payload,
        source_geometry="Recovered Shape",
        sew_tolerance=1e-4,
    )
    cq.exporters.export(shape, str(path), exportType="STEP")
    return True

class TopOptStepExportWorker(QtCore.QThread):
    """Background worker for TopOpt STEP export."""

    export_finished = QtCore.Signal(str)
    export_error = QtCore.Signal(str)

    def __init__(
        self,
        topo_output,
        path,
        *,
        density_cutoff=0.45,
        print_ready=False,
        decimate_ratio=1.0,
        extrusion_axis="none",
        passive_regions=None,
        parent=None,
    ):
        super().__init__(parent)
        self.topo_output = dict(topo_output or {})
        self.path = str(path)
        self.density_cutoff = float(density_cutoff or 0.45)
        self.print_ready = bool(print_ready)
        self.decimate_ratio = float(decimate_ratio or 1.0)
        self.extrusion_axis = str(extrusion_axis or "none").strip().lower()
        self.passive_regions = dict(passive_regions or {})

    @staticmethod
    def _bounds_tuple(bounds_payload):
        import numpy as np

        if (
            isinstance(bounds_payload, dict)
            and "min" in bounds_payload
            and "max" in bounds_payload
        ):
            mins = np.asarray(bounds_payload["min"], dtype=float)
            maxs = np.asarray(bounds_payload["max"], dtype=float)
            if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                return mins[:3], maxs[:3]
        return None

    def _refresh_recovered_shape(self, payload):
        if payload.get("density") is None:
            return
        import numpy as np
        from pylcss.design_studio.topology_optimization.recovery import _recover_voxel_shape

        passive = self.passive_regions or payload.get("passive_regions") or {}
        recovered = _recover_voxel_shape(
            np.asarray(payload["density"], dtype=float),
            self._bounds_tuple(payload.get("bounds")),
            self.density_cutoff,
            print_ready=self.print_ready,
            decimate_ratio=self.decimate_ratio,
            solid_boxes=passive.get("solid_boxes", ()),
            void_boxes=passive.get("void_boxes", ()),
            solid_cylinders=passive.get("solid_cylinders", ()),
            void_cylinders=passive.get("void_cylinders", ()),
            extrusion_axis=self.extrusion_axis,
            source_mask=payload.get("design_domain"),
        )
        if recovered is not None and len(recovered.get("faces", [])) > 0:
            payload["recovered_shape"] = recovered

    def _write_cad_step(self, payload, path):
        import cadquery as cq

        from pylcss.design_studio.topology_optimization.cad_reconstruction import (
            reconstruct_topopt_cad,
        )

        shape = reconstruct_topopt_cad(
            payload,
            source_geometry="Recovered Shape",
            sew_tolerance=1e-4,
        )
        cq.exporters.export(shape, str(path), exportType="STEP")

    @staticmethod
    def _write_faceted_step(recovered, path):
        import numpy as np

        vertices = np.asarray(recovered.get("vertices"), dtype=float)
        faces = np.asarray(recovered.get("faces"), dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] < 3:
            raise RuntimeError("Recovered shape does not contain triangle vertices/faces.")

        max_faceted_faces = 5000
        if len(faces) > max_faceted_faces:
            raise RuntimeError(
                f"Recovered mesh has {len(faces)} triangles; dense faceted STEP "
                "export is disabled. Use STL/OBJ for mesh output or CAD STEP "
                "reconstruction for an interactive B-rep."
            )

        from OCP.BRep import BRep_Builder
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
        from OCP.gp import gp_Pnt
        from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer
        from OCP.TopoDS import TopoDS_Compound

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        n_faces = 0
        skipped = 0
        for tri in faces:
            try:
                pts = vertices[np.asarray(tri[:3], dtype=int), :3]
            except Exception:
                skipped += 1
                continue
            if not np.all(np.isfinite(pts)):
                skipped += 1
                continue
            area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            if area <= 1e-12:
                skipped += 1
                continue

            poly = BRepBuilderAPI_MakePolygon(
                gp_Pnt(float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2])),
                gp_Pnt(float(pts[1, 0]), float(pts[1, 1]), float(pts[1, 2])),
                gp_Pnt(float(pts[2, 0]), float(pts[2, 1]), float(pts[2, 2])),
                True,
            )
            if not poly.IsDone():
                skipped += 1
                continue
            face_builder = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if not face_builder.IsDone():
                skipped += 1
                continue
            builder.Add(compound, face_builder.Face())
            n_faces += 1

        if n_faces < 1:
            raise RuntimeError("Recovered shape did not contain any valid triangles.")

        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_StepModelType.STEPControl_AsIs)
        writer.Write(str(path))
        return n_faces, skipped

    def run(self):
        try:
            payload = dict(self.topo_output)
            payload["density_cutoff"] = self.density_cutoff
            if self.passive_regions:
                payload["passive_regions"] = self.passive_regions
            else:
                payload.setdefault("passive_regions", {})
            payload["extrusion_axis"] = self.extrusion_axis

            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_external_write_cad_step, payload, self.path)
                future.result()

            self.export_finished.emit(self.path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.export_error.emit(str(exc))


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
    
    # Cohesive, flat dark styling for the whole inspector subtree.  Inline
    # per-widget styles set elsewhere still win (Qt applies the most specific
    # rule), so this only supplies the modern defaults: card-style groups,
    # flat focus-highlighted inputs, and a single #4a9eff accent.
    #
    # IMPORTANT (Qt stylesheet rule): styling a complex widget (QComboBox,
    # QSpinBox) puts it into stylesheet-render mode, which DISABLES its native
    # sub-controls; the CSS-border arrow trick does not render reliably in Qt6,
    # leaving blank "white rectangle" buttons.  So combo boxes and spin boxes
    # are intentionally LEFT NATIVE here (no QComboBox / QSpinBox rule) — they
    # pick up the application's dark Fusion theme and keep working arrows.  Only
    # widgets that are safe to style (QLineEdit has no sub-controls; QCheckBox's
    # ::indicator is fully specified) are themed.
    _INSPECTOR_QSS = """
        #InspectorPanel { background: #1c1e22; }
        QScrollArea { background: transparent; border: none; }
        #qt_scrollarea_viewport { background: transparent; }
        QGroupBox {
            background: #24272d;
            border: 1px solid #2f333a;
            border-radius: 8px;
            margin-top: 16px;
            padding: 12px 10px 10px 10px;
            font-size: 12px;
            font-weight: 600;
            color: #cdd2d9;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            color: #6fb3ff;
            font-weight: 700;
        }
        QLabel { color: #aab0b8; font-size: 12px; background: transparent; }
        QLineEdit {
            background: #14161a;
            border: 1px solid #313641;
            border-radius: 6px;
            padding: 5px 8px;
            color: #eef1f5;
            min-height: 18px;
            selection-background-color: #4a9eff;
        }
        QLineEdit:focus { border: 1px solid #4a9eff; background: #181b20; }
        QLineEdit:disabled { color: #6b7178; background: #1a1c20; border-color: #2a2d33; }

        /* Check box — explicit indicator so the tick box stays visible.
           Checked = filled accent box, unchecked = empty dark box. */
        QCheckBox { color: #aab0b8; spacing: 7px; background: transparent; }
        QCheckBox::indicator {
            width: 16px; height: 16px; border-radius: 4px;
            border: 1px solid #3a3f48; background: #14161a;
        }
        QCheckBox::indicator:hover { border: 1px solid #4a9eff; }
        QCheckBox::indicator:checked {
            background: #4a9eff; border: 1px solid #4a9eff;
        }
        QCheckBox::indicator:disabled {
            border: 1px solid #2a2d33; background: #1a1c20;
        }

        QPushButton {
            background: #2a2e35; border: 1px solid #383d46;
            border-radius: 6px; padding: 6px 12px; color: #d6dae0; font-weight: 600;
        }
        QPushButton:hover { background: #323843; border-color: #4a9eff; color: #ffffff; }
        QPushButton:pressed { background: #2a2e35; }
    """

    def __init__(self):
        super(PropertiesPanel, self).__init__()
        self.setObjectName("InspectorPanel")
        self.setStyleSheet(self._INSPECTOR_QSS)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
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

        # Clear previous UI. takeAt() drops the item from the layout but
        # leaves the widget parented to props_widget, so deleteLater()
        # alone keeps it painted at its old geometry until the event loop
        # fires — when display_node() is called several times in a row
        # (e.g. after a face pick) the survivors stack up as ghost
        # widgets. setParent(None) detaches them from the paint tree
        # immediately so only the freshly built panel is visible.
        self.props_widget.setUpdatesEnabled(False)
        try:
            while self.props_layout.count():
                item = self.props_layout.takeAt(0)
                w = item.widget() if item is not None else None
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
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
            if node_class == 'TopologyOptVoxelNode':
                self._build_topopt_voxel_ui(node)
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
        finally:
            self.props_widget.setUpdatesEnabled(True)

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
            
    def _build_topopt_voxel_ui(self, node):
        """Compact inspector for the structured voxel topology optimizer."""

        def _get_int(prop, default):
            try:
                return int(node.get_property(prop))
            except Exception:
                return default

        def _get_float(prop, default):
            try:
                return float(node.get_property(prop))
            except Exception:
                return default

        def _get_bool(prop, default=False):
            value = node.get_property(prop)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value) if value is not None else bool(default)

        def _combo(prop, items, default=None):
            widget = QtWidgets.QComboBox()
            item_texts = [str(item) for item in items]
            widget.addItems(item_texts)
            current = str(node.get_property(prop) or default or item_texts[0])
            idx = widget.findText(current)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.currentTextChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return widget

        def _check(prop, label=""):
            widget = QtWidgets.QCheckBox(label)
            widget.setChecked(_get_bool(prop))
            widget.stateChanged.connect(
                lambda state, p=prop: self.update_property(p, bool(state))
            )
            return widget

        def _double(prop, default, lo, hi, decimals=3, step=0.1):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(float(lo), float(hi))
            widget.setDecimals(int(decimals))
            widget.setSingleStep(float(step))
            widget.setValue(_get_float(prop, default))
            widget.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return widget

        def _int(prop, default, lo, hi):
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(lo), int(hi))
            widget.setValue(_get_int(prop, default))
            widget.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return widget

        def _json_editor(prop, placeholder, min_height=72):
            widget = QtWidgets.QPlainTextEdit(str(node.get_property(prop) or "[]"))
            widget.setPlaceholderText(placeholder)
            mono = QtGui.QFont("Consolas")
            mono.setStyleHint(QtGui.QFont.Monospace)
            widget.setFont(mono)
            widget.setMinimumHeight(int(min_height))
            widget.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            widget.focusOutEvent = (
                lambda ev, w=widget, p=prop, _orig=widget.focusOutEvent:
                    (self.update_property(p, w.toPlainText()), _orig(ev))[-1]
            )
            return widget

        def _spin_row(widgets):
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            for widget in widgets:
                layout.addWidget(widget, 1)
            return row

        def _refresh_topopt_later():
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        def _points_bounds(points, *, column_major=False):
            try:
                import numpy as np

                pts = np.asarray(points, dtype=float)
                if pts.ndim != 2 or pts.size == 0:
                    return None
                if column_major:
                    if pts.shape[0] < 3 or pts.shape[1] == 0:
                        return None
                    coords = pts[:3, :]
                    return coords.min(axis=1), coords.max(axis=1)
                if pts.shape[1] >= 3:
                    coords = pts[:, :3]
                    return coords.min(axis=0), coords.max(axis=0)
                if pts.shape[0] >= 3:
                    coords = pts[:3, :]
                    return coords.min(axis=1), coords.max(axis=1)
            except Exception:
                return None
            return None

        def _bbox_bounds(value):
            try:
                bb = value.BoundingBox()
            except Exception:
                bb = value
            try:
                return (
                    (float(bb.xmin), float(bb.ymin), float(bb.zmin)),
                    (float(bb.xmax), float(bb.ymax), float(bb.zmax)),
                )
            except Exception:
                return None

        def _bounds_from_value(value, depth=0):
            if value is None or depth > 3:
                return None
            if isinstance(value, dict):
                for key in ("mesh", "shape", "cad", "cadquery_object", "result"):
                    bounds = _bounds_from_value(value.get(key), depth + 1)
                    if bounds is not None:
                        return bounds
                return None
            if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                bounds_list = [
                    bounds for bounds in (_bounds_from_value(item, depth + 1) for item in value)
                    if bounds is not None
                ]
                if bounds_list:
                    try:
                        import numpy as np

                        mins = np.vstack([b[0] for b in bounds_list]).min(axis=0)
                        maxs = np.vstack([b[1] for b in bounds_list]).max(axis=0)
                        return mins, maxs
                    except Exception:
                        return None
                return None
            if hasattr(value, "p"):
                bounds = _points_bounds(value.p, column_major=True)
                if bounds is not None:
                    return bounds
            for attr in ("vertices", "points"):
                if hasattr(value, attr):
                    bounds = _points_bounds(getattr(value, attr))
                    if bounds is not None:
                        return bounds
            if hasattr(value, "val"):
                try:
                    bounds = _bounds_from_value(value.val(), depth + 1)
                    if bounds is not None:
                        return bounds
                except Exception:
                    pass
            if hasattr(value, "wrapped"):
                try:
                    bounds = _bounds_from_value(value.wrapped, depth + 1)
                    if bounds is not None:
                        return bounds
                except Exception:
                    pass
            return _bbox_bounds(value)

        def _topopt_input_spans():
            try:
                bounds = _bounds_from_value(node.get_input_value("mesh", None))
            except Exception:
                bounds = None
            if bounds is None:
                return None
            try:
                import numpy as np

                mins, maxs = bounds
                spans = np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float)
                if spans.shape[0] < 3 or not np.all(np.isfinite(spans)):
                    return None
                positive = spans[spans > 1e-9]
                if positive.size == 0:
                    return None
                fill = float(positive.min())
                return np.where(spans > 1e-9, spans, fill)
            except Exception:
                return None

        def _generalized_grid():
            import numpy as np

            spans = _topopt_input_spans()
            if spans is None:
                spans = np.asarray([
                    max(1, _get_int("nelx", 30)),
                    max(1, _get_int("nely", 20)),
                    max(1, _get_int("nelz", 10)),
                ], dtype=float)

            target_cells = 24000.0
            max_cells = 50000
            min_axis = 6
            max_axis = 160

            voxel = max(float(np.prod(spans) / target_cells) ** (1.0 / 3.0), 1e-9)
            dims = np.ceil(spans / voxel).astype(int)
            dims = np.maximum(dims, min_axis)

            if int(dims.max()) > max_axis:
                scale = max_axis / float(dims.max())
                dims = np.maximum(np.floor(dims * scale).astype(int), min_axis)

            while int(np.prod(dims)) > max_cells and int(dims.max()) > min_axis:
                scale = (max_cells / float(np.prod(dims))) ** (1.0 / 3.0) * 0.98
                dims = np.maximum(np.floor(dims * scale).astype(int), min_axis)

            return [int(v) for v in dims[:3]]

        def _apply_generalized_defaults(refresh=True):
            nelx, nely, nelz = _generalized_grid()
            stress_enabled = _get_bool("stress_constraint")
            goal = str(node.get_property("design_goal") or "").lower()
            stress_goal = "stress" in goal
            optimizer = "MMA" if stress_enabled or stress_goal else "OC"
            max_dim = max(nelx, nely, nelz)
            rmin = round(max(1.2, min(5.0, max_dim * 0.030)), 2)

            settings = {
                "advanced_settings_visible": False,
                "nelx": nelx,
                "nely": nely,
                "nelz": nelz,
                "rmin": rmin,
                "penal": 3.0,
                "density_cutoff": 0.45,
                "optimizer": optimizer,
                "max_iter": 100,
                "tol": 0.005,
                "convergence_patience": 5,
                "print_ready_mesh": False,
                "mesh_decimate_ratio": 1.0,
            }
            for key, value in settings.items():
                self.update_property(key, value)
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    "Applied generalized fine topology defaults"
                )
            if refresh:
                _refresh_topopt_later()

        def _intent_combo(prop, items, default=None):
            widget = _combo(prop, items, default)

            def _changed(value, p=prop):
                self.update_property(p, value)
                goal = value if p == 'design_goal' else node.get_property('design_goal')
                manufacturing = (
                    value if p == 'manufacturing_process'
                    else node.get_property('manufacturing_process')
                )
                settings = industrial_topopt_defaults(
                    goal,
                    'Automatic',
                    manufacturing,
                    nelx=node.get_property('nelx') or 30,
                    nely=node.get_property('nely') or 20,
                    nelz=node.get_property('nelz') or 10,
                )
                for key, setting in settings.items():
                    self.update_property(key, setting)
                _refresh_topopt_later()

            try:
                widget.currentTextChanged.disconnect()
            except Exception:
                pass
            widget.currentTextChanged.connect(_changed)
            return widget


        intent_group = QtWidgets.QGroupBox("Design Intent")
        intent_layout = QtWidgets.QFormLayout()
        intent_layout.addRow(
            "Workflow:",
            _combo("workflow_mode", INDUSTRIAL_WORKFLOW_MODES, "Guided"),
        )
        intent_layout.addRow(
            "Goal:",
            _intent_combo("design_goal", INDUSTRIAL_DESIGN_GOALS, "Lightweight Stiffness"),
        )
        intent_layout.addRow(
            "Manufacturing:",
            _intent_combo("manufacturing_process", INDUSTRIAL_MANUFACTURING_PROCESSES, "None"),
        )

        volfrac = max(0.01, min(0.99, _get_float("volfrac", 0.5)))
        material_container = QtWidgets.QWidget()
        material_layout = QtWidgets.QHBoxLayout(material_container)
        material_layout.setContentsMargins(0, 0, 0, 0)
        material_layout.setSpacing(6)
        material_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        material_slider.setRange(1, 99)
        material_slider.setValue(int(round(volfrac * 100)))
        material_spin = QtWidgets.QSpinBox()
        material_spin.setRange(1, 99)
        material_spin.setSuffix("%")
        material_spin.setValue(int(round(volfrac * 100)))

        def update_material_volfrac(percent):
            material_slider.blockSignals(True)
            material_spin.blockSignals(True)
            material_slider.setValue(percent)
            material_spin.setValue(percent)
            material_slider.blockSignals(False)
            material_spin.blockSignals(False)
            self.update_property("volfrac", percent / 100.0)

        material_slider.valueChanged.connect(update_material_volfrac)
        material_spin.valueChanged.connect(update_material_volfrac)
        material_layout.addWidget(material_slider, 1)
        material_layout.addWidget(material_spin)
        intent_layout.addRow(
            "Allowable Stress (MPa):",
            _double("yield_stress", 250.0, 0.001, 1_000_000.0, decimals=3, step=10.0),
        )
        advanced_toggle = QtWidgets.QCheckBox("Show solver and recovery controls")
        advanced_toggle.setChecked(_get_bool("advanced_settings_visible"))

        def _toggle_advanced(state):
            self.update_property("advanced_settings_visible", bool(state))
            _refresh_topopt_later()

        advanced_toggle.stateChanged.connect(_toggle_advanced)
        intent_layout.addRow("Advanced:", advanced_toggle)
        intent_layout.addRow("Material Budget:", material_container)

        intent_group.setLayout(intent_layout)
        self.props_layout.addWidget(intent_group)

        pipeline_group = QtWidgets.QGroupBox("CAD Export")
        pipeline_layout = QtWidgets.QFormLayout()
        step_name = QtWidgets.QLineEdit(str(node.get_property("cad_export_filename") or "topology_optimized.step"))
        step_name.editingFinished.connect(
            lambda w=step_name: self.update_property("cad_export_filename", w.text())
        )
        pipeline_layout.addRow("STEP Name:", step_name)
        btn_export_step = QtWidgets.QPushButton("Export STEP")
        btn_export_step.clicked.connect(lambda: self._export_topopt_step(node))
        pipeline_layout.addRow("Export:", btn_export_step)
        pipeline_group.setToolTip("Export the recovered topology shape after a run.")
        pipeline_group.setLayout(pipeline_layout)
        self.props_layout.addWidget(pipeline_group)

        advanced_group = QtWidgets.QGroupBox("Solver Settings")
        advanced_group.setToolTip("Raw solver controls for topology studies and reproducibility.")
        advanced_layout = QtWidgets.QVBoxLayout()

        domain_group = QtWidgets.QGroupBox("Voxel Grid")
        domain_layout = QtWidgets.QFormLayout()
        for label, prop, default in (
            ("X Cells:", "nelx", 30),
            ("Y Cells:", "nely", 20),
            ("Z Cells:", "nelz", 10),
        ):
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 500)
            spin.setValue(_get_int(prop, default))
            spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            domain_layout.addRow(label, spin)
        domain_group.setLayout(domain_layout)
        advanced_layout.addWidget(domain_group)

        opt_group = QtWidgets.QGroupBox("Optimization")
        opt_layout = QtWidgets.QFormLayout()

        volfrac = max(0.01, min(0.99, _get_float("volfrac", 0.5)))
        vol_container = QtWidgets.QWidget()
        vol_layout = QtWidgets.QHBoxLayout(vol_container)
        vol_layout.setContentsMargins(0, 0, 0, 0)
        vol_layout.setSpacing(6)
        vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        vol_slider.setRange(1, 99)
        vol_slider.setValue(int(round(volfrac * 100)))
        vol_spin = QtWidgets.QSpinBox()
        vol_spin.setRange(1, 99)
        vol_spin.setSuffix("%")
        vol_spin.setValue(int(round(volfrac * 100)))

        def update_volfrac(percent):
            vol_slider.blockSignals(True)
            vol_spin.blockSignals(True)
            vol_slider.setValue(percent)
            vol_spin.setValue(percent)
            vol_slider.blockSignals(False)
            vol_spin.blockSignals(False)
            self.update_property("volfrac", percent / 100.0)

        vol_slider.valueChanged.connect(update_volfrac)
        vol_spin.valueChanged.connect(update_volfrac)
        vol_layout.addWidget(vol_slider, 1)
        vol_layout.addWidget(vol_spin)
        opt_layout.addRow("Keep Volume:", vol_container)

        rmin = QtWidgets.QDoubleSpinBox()
        rmin.setRange(0.0, 100.0)
        rmin.setDecimals(2)
        rmin.setSingleStep(0.1)
        rmin.setValue(_get_float("rmin", 1.5))
        rmin.valueChanged.connect(lambda v: self.update_property("rmin", v))
        opt_layout.addRow("Filter Radius:", rmin)

        penal = QtWidgets.QDoubleSpinBox()
        penal.setRange(1.0, 10.0)
        penal.setDecimals(2)
        penal.setSingleStep(0.1)
        penal.setValue(_get_float("penal", 3.0))
        penal.valueChanged.connect(lambda v: self.update_property("penal", v))
        opt_layout.addRow("Penalization:", penal)

        cutoff = QtWidgets.QDoubleSpinBox()
        cutoff.setRange(0.0, 1.0)
        cutoff.setDecimals(2)
        cutoff.setSingleStep(0.05)
        cutoff.setValue(_get_float("density_cutoff", 0.45))
        cutoff.valueChanged.connect(lambda v: self.update_property("density_cutoff", v))
        opt_layout.addRow("Density Cutoff:", cutoff)

        opt_group.setLayout(opt_layout)
        advanced_layout.addWidget(opt_group)

        solver_group = QtWidgets.QGroupBox("Solver")
        solver_layout = QtWidgets.QFormLayout()

        optimizer = QtWidgets.QComboBox()
        optimizer.addItems(["OC", "MMA"])
        current_optimizer = str(node.get_property("optimizer") or "OC")
        index = optimizer.findText(current_optimizer)
        optimizer.setCurrentIndex(index if index >= 0 else 0)
        optimizer.currentTextChanged.connect(
            lambda v: self.update_property("optimizer", v)
        )
        solver_layout.addRow("Optimizer:", optimizer)

        max_iter = QtWidgets.QSpinBox()
        max_iter.setRange(1, 1000)
        max_iter.setValue(_get_int("max_iter", 80))
        max_iter.valueChanged.connect(lambda v: self.update_property("max_iter", v))
        solver_layout.addRow("Iterations:", max_iter)

        tol = QtWidgets.QDoubleSpinBox()
        tol.setRange(0.00001, 1.0)
        tol.setDecimals(5)
        tol.setSingleStep(0.001)
        tol.setValue(_get_float("tol", 0.01))
        tol.valueChanged.connect(lambda v: self.update_property("tol", v))
        solver_layout.addRow("Tolerance:", tol)

        solver_group.setLayout(solver_layout)
        advanced_layout.addWidget(solver_group)

        stress_group = QtWidgets.QGroupBox("Stress Constraint")
        stress_layout = QtWidgets.QFormLayout()
        stress_layout.addRow("Enable:", _check("stress_constraint"))
        stress_layout.addRow(
            "Yield Stress:",
            _double("yield_stress", 1.0, 0.0, 1_000_000.0, decimals=3, step=10.0),
        )
        stress_group.setToolTip(
            "P-norm von Mises stress constraint aggregated over ALL load cases.\n"
            "Forces the optimiser to MMA. The numerical relaxation and\n"
            "aggregation policy is selected internally by the solver."
        )
        stress_group.setLayout(stress_layout)
        advanced_layout.addWidget(stress_group)

        mfg_group = QtWidgets.QGroupBox("Manufacturing Constraints")
        mfg_layout = QtWidgets.QFormLayout()
        mfg_layout.addRow(
            "Symmetry:",
            _combo("symmetry", ["None", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"], "None"),
        )
        mfg_layout.addRow("Extrusion:", _combo("extrusion", ["None", "X", "Y", "Z"], "None"))
        mfg_layout.addRow(
            "Build Axis:",
            _combo("overhang_build_axis", ["None", "+X", "-X", "+Y", "-Y", "+Z", "-Z"], "None"),
        )
        mfg_layout.addRow(
            "Max Member Radius:",
            _double("max_member_size_voxels", 0.0, 0.0, 100.0, decimals=2, step=0.5),
        )
        mfg_layout.addRow("Pattern Count:", _int("pattern_repeat", 1, 1, 64))
        mfg_layout.addRow("Pattern Axis:", _combo("pattern_axis", ["X", "Y", "Z"], "Y"))
        mfg_group.setToolTip(
            "Projection-style constraints applied after each density update: symmetry, extrusion, AM overhang, max member size, and repeated patterns."
        )
        mfg_group.setLayout(mfg_layout)
        self.props_layout.addWidget(mfg_group)

        advanced_group.setLayout(advanced_layout)
        # Industrial TopOpt keeps solver/grid knobs automated.  The controls
        # above still exist as node properties for reproducibility and saved
        # studies, but the normal GUI exposes only design intent, setup, CAD,
        # and manufacturing choices.

        if _get_bool("advanced_settings_visible"):
            self.props_layout.addWidget(advanced_group)
        setup_group = QtWidgets.QGroupBox("Setup & Load Cases")
        setup_layout = QtWidgets.QFormLayout()
        support_items = ["None", "Fix X", "Fix Y", "Fix Z", "Fix XY", "Fix YZ", "Fix XZ", "Fix XYZ"]
        support_row = QtWidgets.QWidget()
        support_grid = QtWidgets.QGridLayout(support_row)
        support_grid.setContentsMargins(0, 0, 0, 0)
        support_grid.setSpacing(4)
        for idx, (label, prop) in enumerate((
            ("Left", "left_support"),
            ("Right", "right_support"),
            ("Top", "top_support"),
            ("Bottom", "bottom_support"),
            ("Front", "front_support"),
            ("Back", "back_support"),
        )):
            support_grid.addWidget(QtWidgets.QLabel(label), idx // 2, (idx % 2) * 2)
            support_grid.addWidget(_combo(prop, support_items, "None"), idx // 2, (idx % 2) * 2 + 1)
        setup_layout.addRow("Face Supports:", support_row)
        setup_layout.addRow(
            "Force Type:",
            _combo("force_type", ["Point", "Distributed Face"], "Point"),
        )
        setup_layout.addRow(
            "Force Face:",
            _combo("force_face", ["Left", "Right", "Top", "Bottom", "Front", "Back"], "Right"),
        )
        setup_layout.addRow(
            "Point X/Y/Z:",
            _spin_row((
                _double("force_ix_frac", 1.0, 0.0, 1.0, decimals=3, step=0.05),
                _double("force_iy_frac", 0.5, 0.0, 1.0, decimals=3, step=0.05),
                _double("force_iz_frac", 0.5, 0.0, 1.0, decimals=3, step=0.05),
            )),
        )
        setup_layout.addRow(
            "Force XYZ:",
            _spin_row((
                _double("force_dir_x", 0.0, -1.0, 1.0, decimals=3, step=0.1),
                _double("force_dir_y", -1.0, -1.0, 1.0, decimals=3, step=0.1),
                _double("force_dir_z", 0.0, -1.0, 1.0, decimals=3, step=0.1),
            )),
        )
        setup_layout.addRow(
            "Magnitude:",
            _double("force_magnitude", 1.0, 0.0, 1_000_000.0, decimals=3, step=1.0),
        )
        setup_layout.addRow(
            "Support Regions:",
            _json_editor("support_regions", '[{"x":[0,0.05],"y":[0,1],"z":[0,1],"dofs":"Fix XYZ"}]'),
        )
        setup_layout.addRow(
            "Solid Regions:",
            _json_editor("solid_regions", '[{"type":"cylinder","axis":"z","center":[0.5,0.5],"radius":0.25,"z":[0,1]}]'),
        )
        setup_layout.addRow(
            "Void Regions:",
            _json_editor("void_regions", '[{"type":"cylinder","axis":"z","center":[0.5,0.5],"radius":0.2,"z":[0,1]}]'),
        )
        setup_group.setToolTip(
            "Standalone defaults. In normal graph workflows, use "
            "the topology block's constraints and loads input ports."
        )
        setup_group.setLayout(setup_layout)
        # Boundary conditions are supplied through the TopOpt input ports.
        # Keep the legacy property editors available in saved projects, but do
        # not expose duplicate setup/load controls in the normal inspector.

        post_group = QtWidgets.QGroupBox("Post-Processing")
        post_layout = QtWidgets.QFormLayout()
        post_layout.addRow("Print-Ready Mesh:", _check("print_ready_mesh"))
        post_layout.addRow(
            "Decimate Ratio:",
            _double("mesh_decimate_ratio", 1.0, 0.01, 1.0, decimals=2, step=0.05),
        )
        post_group.setToolTip(
            "Controls marching-cubes cleanup, hole filling, smoothing, and optional mesh decimation before STL/CAD handoff."
        )
        post_group.setLayout(post_layout)
        # Post-processing is part of the automated industrial preset now.
        # Keep the properties for saved studies, but do not show this group.

        view_group = QtWidgets.QGroupBox("Visualization")
        view_layout = QtWidgets.QFormLayout()
        if _get_bool("advanced_settings_visible"):
            self.props_layout.addWidget(post_group)

        validation_group = QtWidgets.QGroupBox("Validation & CAD Handoff")
        validation_layout = QtWidgets.QFormLayout()
        validation_layout.addRow("Validate after solve:", _check("validate_after_optimize"))
        validation_layout.addRow(
            "Validation quality:",
            _combo("validation_quality", ["Standard", "Mesh Convergence"], "Standard"),
        )
        validation_layout.addRow("Build CAD after solve:", _check("generate_cad_after_optimize"))
        validation_group.setToolTip(
            "Optional CalculiX re-analysis and automatic recovered-shape CAD reconstruction."
        )
        validation_group.setLayout(validation_layout)
        self.props_layout.addWidget(validation_group)

        visualization = QtWidgets.QComboBox()
        visualization.addItems(["Density", "Recovered Shape"])
        current_view = str(node.get_property("visualization") or "Density")
        view_index = visualization.findText(current_view)
        visualization.setCurrentIndex(view_index if view_index >= 0 else 0)
        visualization.currentTextChanged.connect(
            lambda v: self.update_property("visualization", v)
        )
        view_layout.addRow("Mode:", visualization)

        btn_export_stl = QtWidgets.QPushButton("Export to STL")
        btn_export_stl.setToolTip(
            "Export the recovered voxel topology surface as an STL file"
        )
        btn_export_stl.clicked.connect(lambda: self._export_topopt_stl(node))
        view_layout.addRow("Recovered Shape:", btn_export_stl)

        view_group.setLayout(view_layout)
        self.props_layout.addWidget(view_group)

    def _refresh_topopt_recovered_shape(self, node, result):
        """Rebuild recovered_shape from the current density before export."""
        if not isinstance(result, dict) or result.get('density') is None:
            return result.get('recovered_shape') if isinstance(result, dict) else None
        try:
            import numpy as np
            from pylcss.design_studio.topology_optimization.recovery import _recover_voxel_shape

            bounds_payload = result.get('bounds')
            bounds = None
            if (
                isinstance(bounds_payload, dict)
                and 'min' in bounds_payload
                and 'max' in bounds_payload
            ):
                mins = np.asarray(bounds_payload['min'], dtype=float)
                maxs = np.asarray(bounds_payload['max'], dtype=float)
                if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                    bounds = (mins[:3], maxs[:3])

            bc = node._build_bc() if hasattr(node, '_build_bc') else None
            recovered = _recover_voxel_shape(
                np.asarray(result['density'], dtype=float),
                bounds,
                float(result.get('density_cutoff') or node.get_property('density_cutoff') or 0.45),
                print_ready=bool(node.get_property('print_ready_mesh')),
                decimate_ratio=float(node.get_property('mesh_decimate_ratio') or 1.0),
                solid_boxes=getattr(bc, 'solid_boxes', ()),
                void_boxes=getattr(bc, 'void_boxes', ()),
                solid_cylinders=getattr(bc, 'solid_cylinders', ()),
                void_cylinders=getattr(bc, 'void_cylinders', ()),
                extrusion_axis=str(
                    result.get('extrusion_axis')
                    or node.get_property('extrusion')
                    or 'none'
                ).lower(),
                source_mask=result.get('design_domain'),
            )
            if recovered is not None and len(recovered.get('faces', [])) > 0:
                result['recovered_shape'] = recovered
                setattr(node, '_last_result', result)
                return recovered
        except Exception:
            pass
        return result.get('recovered_shape')

    def _export_topopt_step(self, node):
        """Reconstruct and export the topology result as a STEP body."""
        worker = getattr(self, "_topopt_step_export_worker", None)
        if worker is not None and worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("STEP export already running...")
            return

        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict) or result.get('type') not in {'topopt_voxel'}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Topology Result",
                "Run topology optimisation before exporting STEP.",
            )
            return

        default_name = str(node.get_property("cad_export_filename") or "topology_optimized.step")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export STEP",
            default_name,
            "STEP Files (*.step *.stp)",
        )
        if not path:
            return
        if not path.lower().endswith((".step", ".stp")):
            path += ".step"

        passive_regions = result.get("passive_regions")
        if not isinstance(passive_regions, dict) and hasattr(node, "_build_bc"):
            try:
                bc = node._build_bc()
                passive_regions = {
                    "solid_boxes": list(getattr(bc, "solid_boxes", ())),
                    "void_boxes": list(getattr(bc, "void_boxes", ())),
                    "solid_cylinders": list(getattr(bc, "solid_cylinders", ())),
                    "void_cylinders": list(getattr(bc, "void_cylinders", ())),
                }
            except Exception:
                passive_regions = {}

        node.set_property("cad_reconstruction_method", "Recovered Shape")
        cutoff = float(result.get("density_cutoff") or node.get_property("density_cutoff") or 0.45)
        print_ready = bool(node.get_property("print_ready_mesh"))
        decimate = float(node.get_property("mesh_decimate_ratio") or 1.0)
        extrusion_axis = str(
            result.get("extrusion_axis") or node.get_property("extrusion") or "none"
        ).strip().lower()

        node.set_property("cad_export_filename", path)
        worker = TopOptStepExportWorker(
            result,
            path,
            density_cutoff=cutoff,
            print_ready=print_ready,
            decimate_ratio=decimate,
            extrusion_axis=extrusion_axis,
            passive_regions=passive_regions,
            parent=self,
        )
        self._topopt_step_export_worker = worker

        def _finish(export_path):
            self._topopt_step_export_worker = None
            if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                self.window().statusBar().showMessage(f"Exported CAD STEP to {export_path}")

        def _fail(message):
            self._topopt_step_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", message)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), 'statusBar') and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting CAD STEP in background...")
        worker.start()

    def _export_topopt_stl(self, node):
        """Export recovered shape from topology optimisation as binary STL."""
        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(self, "No Shape",
                "Run topology optimisation first — no recovered shape available.")
            return
        recovered = self._refresh_topopt_recovered_shape(node, result)
        if recovered is None:
            QtWidgets.QMessageBox.warning(self, "No Shape",
                "Run topology optimisation first - no recovered shape available.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)")
        if not path:
            return
        try:
            import numpy as np
            verts = np.asarray(recovered['vertices'], dtype=float)
            faces = np.asarray(recovered['faces'], dtype=int)
            try:
                import trimesh
                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
                tm.export(path, file_type='stl')
                if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                    self.window().statusBar().showMessage(
                        f"Exported {len(tm.faces)} triangles to {path}"
                    )
                return
            except Exception:
                pass

            from stl import mesh as stl_mesh   # numpy-stl fallback
            stl_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    stl_obj.vectors[i][j] = verts[f[j]]
            stl_obj.save(path)
            if hasattr(self.window(), 'statusBar') and self.window().statusBar():
                self.window().statusBar().showMessage(f"Exported {len(faces)} triangles to {path}")
        except ImportError:
            # Fallback: raw binary STL without numpy-stl
            self._write_binary_stl(path, recovered)
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
                self.window().statusBar().showMessage(f"Exported {len(faces)} faces to {path}")
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
            self.window().statusBar().showMessage(f"Exported {len(faces)} triangles to {path}")

    # NOTE: Five legacy node-class-specific builders used to live here
    # (_build_primitive_ui / _build_simulation_ui / _build_operation_ui /
    # _build_modification_ui / _build_transform_ui).  They were never reached:
    # display_node only dispatches to CadQueryCodeNode,
    # InteractiveSelectFaceNode, SelectFaceNode, and the FEA-BC trio; every
    # other node falls through to _build_generic_ui (which renders correctly
    # via the sectioned, items-aware widget loop).  Removed to keep this file
    # focused on the UI that's actually displayed.

    # Ordered list of (section title, prefix list) — first match wins.
    # Properties that match no section land in "General".
    _PROPERTY_SECTIONS = [
        ("External Solver", ("external_", "openradioss_", "calculix_", "run_external", "deck_only", "solver_backend",
                              "deck_path", "engine_path", "engine_executable_path", "starter_path",
                              "work_dir", "timeout_s", "stress_scale_to_mpa")),
        ("Visualization",   ("visualization", "deformation_scale", "disp_scale", "n_frames")),
        ("Solver",          ("end_time", "time_steps", "damping", "enable_", "contact_", "mass_scaling", "iterations",
                             "convergence_tol", "move_limit", "min_density", "penal", "filter_radius",
                             "update_scheme", "filter_type",
                             "element_type", "shape_recovery", "recovery_resolution", "smoothing_iterations",
                             "density_cutoff", "vol_frac", "symmetry_",
                             # TopOpt stress constraint. Use exact name for
                             # yield_stress so we don't steal Material's
                             # yield_strength into Solver.
                             "yield_stress", "stress_")),
        ("Material",        ("preset", "E", "nu", "rho", "density", "poissons_ratio", "yield_strength",
                             "tangent_modulus", "failure_strain", "enable_fracture",
                             # Crash-only: engineering-facing rate sensitivity.
                             "strain_rate_")),
        ("Mesh",            ("mesh_type", "element_size", "max_size", "min_size", "order")),
        ("Impact",          ("velocity_", "application_scope", "node_tolerance", "wall_", "impactor_mass_kg")),
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
        'damping_alpha', 'damping_beta',
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
        'bc_preset',
        'quality_preset',
        'moment_x', 'moment_y', 'moment_z',
        # Internal topology-optimization numerical policy. These may exist in
        # old project JSON but are no longer engineering-facing controls.
        'projection', 'stress_penalty', 'stress_pnorm_p',
        'heaviside_projection', 'heaviside_beta_init', 'heaviside_beta_max',
        'heaviside_beta_step_iters', 'heaviside_eta', 'continuation',
        # Internal OpenRadioss/crash numerical policy.
        'hourglass_formulation', 'hourglass_coefficient',
        # Replaced by the engineering-facing strain_rate_sensitive checkbox.
        'strain_rate_c', 'strain_rate_p',
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
        'stress_scale_to_mpa':
            "Multiplier from anim_to_vtk's native deck stress unit to MPa. "
            "Use 1e6 for tonne-mm-ms decks, 1000 for kg-mm-ms, or 1 when "
            "the converted stress is already MPa.",
        'preset':           "Pick a material from the built-in database, or 'Custom' to set fields manually.",
        'youngs_modulus':   "Young's modulus E (MPa in the standard mm/t/s unit system).",
        'poissons_ratio':   "Poisson's ratio Î½ (typical 0.27–0.34 for metals).",
        'density':          "Mass density ρ (tonne/mm³ — 7.85e-9 for steel).",
        'yield_strength':   "Initial yield stress Ïƒ_y (MPa).  Non-zero triggers *PLASTIC in CalculiX.",
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
        'yield_stress':       "Yield stress σ_y for the PNorm constraint: ||vm||_PNorm ≤ σ_y.",
        'stress_constraint':  "Add a P-norm von Mises stress constraint.  Forces the optimiser to MMA.",
        'strain_rate_sensitive':
            "Use the material preset's internal strain-rate sensitivity for crash runs.",
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
        'mesh_type':         "Tet:  linear C3D4 (fast).  Tet10: quadratic C3D10 (more accurate, ~4Ã— slower).",
        'refinement_size':   "Local element size at refinement zones [mm].  0 = no local refinement.",
        'close_holes':       "Cap small holes during remesh — helps surface watertightness.",
        'repair_surface':    "Run topological repair before remeshing.",
        'mesh_quality':      "Target mesh-quality factor for remeshing.",
        'velocity_x':        "Initial impactor velocity along X [mm/ms = m/s].",
        'velocity_y':        "Initial impactor velocity along Y [mm/ms = m/s].",
        'velocity_z':        "Initial impactor velocity along Z [mm/ms = m/s].",
        'application_scope':
            "Fixed specimen + moving impactor: selected face is hit by a moving\n"
            "finite-mass impactor/wall; connected constraints stay active.\n"
            "Moving body + fixed wall: the mesh receives initial velocity and\n"
            "hits a generated fixed wall; connected constraints are ignored.\n"
            "Prescribed moving wall: selected face is driven by a massless platen\n"
            "with imposed velocity, useful for controlled crush.",
        'node_tolerance':    "Distance [mm] within which a mesh node is treated as belonging to the impact face.",
        'wall_friction':
            "Rigid-wall Coulomb friction.  Use -1 for the scenario default\n"
            "(0.0 for fixed barrier, 0.08 for moving platen/impactor).",
        'wall_gap_mm':
            "Initial clearance from wall to selected/leading face [mm].\n"
            "Use 0 for automatic clearance based on model size.",
        'impactor_mass_kg':
            "Optional sled/impactor mass [kg].  In Fixed specimen + moving\n"
            "impactor this is the moving rigid wall mass.  In Moving body +\n"
            "fixed wall it is lumped onto the projectile trailing edge.\n"
            "A zero mass moving wall is prescribed velocity, not inertial impact.",
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
        'pressure':          "Surface pressure [N/mm² = MPa].  Positive = outward, negative = inward.",

        # ── SelectFace ─────────────────────────────────────────────────
        'selector_type':     "How this node picks faces (see the dropdown's own tooltip).",
        'direction':         "Outward-normal direction the selected face(s) must point in.",
    }

    _PROPERTY_LABELS = {
        'application_scope': 'Crash Scenario',
        'velocity_x': 'Velocity X (mm/ms)',
        'velocity_y': 'Velocity Y (mm/ms)',
        'velocity_z': 'Velocity Z (mm/ms)',
        'node_tolerance': 'Node Tolerance (mm)',
        'wall_friction': 'Wall Friction',
        'wall_gap_mm': 'Wall Gap (mm)',
        'impactor_mass_kg': 'Impactor / Sled Mass (kg)',
        'end_time': 'End Time (ms)',
        'time_steps': 'Mass-Scaling Steps',
        'n_frames': 'Animation Frames',
        'disp_scale': 'Display Scale',
        'stress_scale_to_mpa': 'Native Stress to MPa',
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
        'strain_rate_sensitive': 'Strain-Rate Sensitive',
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
            'mesh_type': ['Tet', 'Tet10', 'Shell'],
            'constraint_type': ['Fixed', 'Roller X', 'Roller Y', 'Roller Z',
                                'Pinned', 'Symmetry X', 'Symmetry Y', 'Symmetry Z', 'Displacement'],
            'load_type': ['Force', 'Gravity', 'Pressure'],
            'gravity_direction': ['-Y', '-Z', '-X', '+Y', '+Z', '+X'],
            # Union of every solver node's visualization vocabulary.
            'visualization': ['Von Mises Stress', 'Displacement',
                              'Plastic Strain', 'Failed Elements', 'Density',
                              'Recovered Shape'],
            'cad_reconstruction_method': [
                'Recovered Shape',
            ],
            'source_geometry': [
                'Recovered Shape',
            ],
            'filter_type': ['sensitivity', 'density'],
            'update_scheme': ['MMA', 'OC'],
            # New combos introduced by the CalculiX / OpenRadioss rewrites.
            'analysis_type': ['Linear', 'Nonlinear (Geometric)', 'Nonlinear (Plastic)'],
            'deformation_scale': ['Auto', '1x', '5x', '10x', '50x', '100x', '200x'],
            'application_scope': [
                'Fixed specimen + moving impactor',
                'Moving body + fixed wall',
                'Prescribed moving wall',
            ],
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
                from pylcss.design_studio.crash.materials import CRASH_MATERIAL_PRESETS
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
                    'CFRP (Quasi-Isotropic, proxy)',
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
                if name == 'application_scope':
                    combo_items = known_combos[name]
                elif not combo_items and name in known_combos:
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
        def _canonical_selector(value):
            aliases = {
                'direction': 'Direction',
                'nearesttopoint': 'NearestToPoint',
                'nearest point': 'NearestToPoint',
                'nearest_point': 'NearestToPoint',
                'index': 'Index',
                'face index': 'Index',
                'face_index': 'Index',
                'largest area': 'Largest Area',
                'largest_area': 'Largest Area',
                'tag': 'Tag',
                'box': 'Box',
                'bounding box': 'Box',
                'bounding_box': 'Box',
                'coordinate range': 'Coordinate Range',
                'range expression': 'Coordinate Range',
                'range_expression': 'Coordinate Range',
            }
            text = str(value or 'Direction').strip()
            return aliases.get(text.lower(), text)

        def _canonical_direction(value):
            aliases = {
                '+X': '>X', '-X': '<X',
                '+Y': '>Y', '-Y': '<Y',
                '+Z': '>Z', '-Z': '<Z',
                'X+': '>X', 'X-': '<X',
                'Y+': '>Y', 'Y-': '<Y',
                'Z+': '>Z', 'Z-': '<Z',
            }
            text = str(value or '>Z').strip().upper()
            return aliases.get(text, text)

        # The selector type drives which field group is shown. The combo shows
        # friendly labels but stores the exact values SelectFaceNode executes.
        sel_type = _canonical_selector(node.get_property('selector_type'))
        type_options = [
            ('Direction', 'Direction'),
            ('Nearest Point', 'NearestToPoint'),
            ('Face Index', 'Index'),
            ('Largest Area', 'Largest Area'),
            ('Bounding Box', 'Box'),
            ('Range Expression', 'Coordinate Range'),
            ('Tag', 'Tag'),
        ]

        # ── 1. Selector type combo ──────────────────────────────────────
        grp_type = QtWidgets.QGroupBox("Selector")
        lay_type = QtWidgets.QFormLayout()
        combo = QtWidgets.QComboBox()
        for label, value in type_options:
            combo.addItem(label, value)
        current_idx = combo.findData(sel_type)
        if current_idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(current_idx)
            combo.blockSignals(False)
        combo.setToolTip(
            "How this node picks faces:\n"
            "  Bounding Box     — every face whose centroid is inside the box\n"
            "  Nearest Point    — the single face closest to (near_x, near_y, near_z)\n"
            "  Direction        — every face whose normal is +X / −Z / …\n"
            "  Face Index       — pick by integer face id (zero-based, brittle)\n"
            "  Range Expression — Python boolean over x, y, z of the face centroid\n"
            "  Tag              — match a user-set tag string on the upstream node"
        )
        combo.currentIndexChanged.connect(
            lambda _i, c=combo: (self.update_property('selector_type', c.currentData()),
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

        if sel_type == 'Box':
            lay.addRow("Min X:", _spin('box_min_x'))
            lay.addRow("Min Y:", _spin('box_min_y'))
            lay.addRow("Min Z:", _spin('box_min_z'))
            lay.addRow("Max X:", _spin('box_max_x'))
            lay.addRow("Max Y:", _spin('box_max_y'))
            lay.addRow("Max Z:", _spin('box_max_z'))
        elif sel_type == 'NearestToPoint':
            lay.addRow("Near X:", _spin('near_x'))
            lay.addRow("Near Y:", _spin('near_y'))
            lay.addRow("Near Z:", _spin('near_z'))
        elif sel_type == 'Direction':
            dir_combo = QtWidgets.QComboBox()
            direction_options = [
                ('+X (X max face)', '>X'),
                ('-X (X min face)', '<X'),
                ('+Y (Y max face)', '>Y'),
                ('-Y (Y min face)', '<Y'),
                ('+Z (Z max face)', '>Z'),
                ('-Z (Z min face)', '<Z'),
            ]
            for label, value in direction_options:
                dir_combo.addItem(label, value)
            dir_combo.setToolTip(
                "Pick every face whose outward normal points in this direction "
                "(within ~10° tolerance)."
            )
            cur = _canonical_direction(node.get_property('direction'))
            current_idx = dir_combo.findData(cur)
            if current_idx >= 0:
                dir_combo.setCurrentIndex(current_idx)
            dir_combo.currentIndexChanged.connect(
                lambda _i, c=dir_combo: self.update_property('direction', c.currentData())
            )
            lay.addRow("Normal:", dir_combo)
        elif sel_type == 'Index':
            w = _intspin('face_index', 0, 100_000)
            w.setToolTip(
                "Zero-based face index from CadQuery's `faces()` iteration order.\n"
                "Fragile — adding a fillet or boolean upstream renumbers faces."
            )
            lay.addRow("Index:", w)
        elif sel_type == 'Coordinate Range':
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

        summary_group = QtWidgets.QGroupBox("Resolved Selection")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        result = getattr(node, '_last_result', None)
        summaries = result.get('face_summaries') if isinstance(result, dict) else None
        if summaries:
            count = int(result.get('face_count') or len(summaries))
            summary_layout.addWidget(QtWidgets.QLabel(f"{count} face(s) currently matched."))
            for idx, info in enumerate(summaries[:6], start=1):
                center = info.get('center') or []
                bbox = info.get('bbox') or {}
                area = info.get('area')
                if len(center) != 3 or not bbox:
                    continue
                text = (
                    f"Face {idx}: center=({center[0]:.3g}, {center[1]:.3g}, {center[2]:.3g}), "
                    f"bbox X[{bbox.get('xmin', 0):.3g}, {bbox.get('xmax', 0):.3g}] "
                    f"Y[{bbox.get('ymin', 0):.3g}, {bbox.get('ymax', 0):.3g}] "
                    f"Z[{bbox.get('zmin', 0):.3g}, {bbox.get('zmax', 0):.3g}]"
                )
                if area is not None:
                    text += f", area={area:.3g}"
                label = QtWidgets.QLabel(text)
                label.setWordWrap(True)
                summary_layout.addWidget(label)
        else:
            label = QtWidgets.QLabel(
                "Run or preview the graph to list the matched face centers and bounds."
            )
            label.setWordWrap(True)
            label.setStyleSheet("color:#888; font-style:italic;")
            summary_layout.addWidget(label)

        btn_refresh = QtWidgets.QPushButton("Preview Selection")
        btn_refresh.setToolTip("Execute CAD-only graph preview and redraw the selected face overlay.")
        btn_refresh.clicked.connect(
            lambda _checked=False: (
                self._get_main_app()._execute_graph(skip_simulation=True)
                if self._get_main_app() is not None else None
            )
        )
        summary_layout.addWidget(btn_refresh)
        self.props_layout.addWidget(summary_group)

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
            "<i>Note: The picker prepares cached upstream geometry when possible. "
            "If the upstream topology optimization has not run yet, run it first.</i>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:10px; margin-top:8px;")
        self.props_layout.addWidget(hint)

    @staticmethod
    def _viewer_has_pickable_faces(viewer):
        try:
            faces = list(getattr(viewer, '_all_occ_faces', []) or [])
            polydata = list(getattr(viewer, '_face_polydata_list', []) or [])
            return bool(faces) and any(pd is not None for pd in polydata)
        except Exception:
            return False

    @staticmethod
    def _upstream_nodes_for(node):
        ordered = []
        visited = set()

        def _walk(current):
            marker = id(current)
            if marker in visited:
                return
            visited.add(marker)
            try:
                ports = current.input_ports()
                if isinstance(ports, dict):
                    ports = list(ports.values())
                else:
                    ports = list(ports)
            except Exception:
                ports = []
            for port in ports:
                try:
                    connected = list(port.connected_ports())
                except Exception:
                    connected = []
                for conn_port in connected:
                    try:
                        _walk(conn_port.node())
                    except Exception:
                        pass
            ordered.append(current)

        _walk(node)
        return ordered

    def _prepare_viewer_for_picking(self, app, node):
        """Render or compute the nearest upstream geometry for face picking."""
        viewer = getattr(app, 'viewer', None)
        if viewer is None:
            return False

        # Always prefer the selected picker node's own upstream geometry over
        # whatever happens to be displayed.  This prevents a Crash/FEA picker
        # from trying to split a stale dense TopOpt STL currently in the viewer.
        try:
            source_node, geometry = app._get_render_context_for_node(node)
        except Exception:
            source_node, geometry = None, None
        if geometry is not None:
            try:
                app._last_rendered_node = source_node or node
                app._render_result_in_viewer(geometry)
            except Exception:
                pass
            if self._viewer_has_pickable_faces(viewer):
                return True
            if hasattr(viewer, 'ensure_mesh_face_picking') and viewer.ensure_mesh_face_picking():
                return True

        if self._viewer_has_pickable_faces(viewer):
            return True
        if hasattr(viewer, 'ensure_mesh_face_picking') and viewer.ensure_mesh_face_picking():
            return True

        if hasattr(app, '_execution_is_active') and app._execution_is_active():
            message = "Wait for the current graph run to finish before picking faces."
            try:
                app.statusBar().showMessage(message)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "Graph Running", message)
            return "pending"

        worker = getattr(self, '_pick_prepare_worker', None)
        try:
            worker_running = bool(worker is not None and worker.isRunning())
        except Exception:
            worker_running = False
        if worker_running:
            message = "Geometry is already being prepared for face picking."
            try:
                app.statusBar().showMessage(message)
            except Exception:
                pass
            return "pending"

        upstream_nodes = [n for n in self._upstream_nodes_for(node) if n is not node]
        blocked_ids = {
            'com.cad.sim.solver',
            'com.cad.sim.crash_solver',
            'com.cad.sim.topopt_voxel',
        }
        blocked = [
            n for n in upstream_nodes
            if getattr(n, '__identifier__', '') in blocked_ids
            and getattr(n, '_last_result', None) is None
        ]
        if blocked:
            message = (
                "Run the upstream topology/solver node once, then pick faces. "
                "The picker can prepare STL import and remesh geometry, but it "
                "will not silently start a new topology optimization."
            )
            try:
                app.statusBar().showMessage(message)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "Run Upstream First", message)
            return "pending"

        if not upstream_nodes:
            if bool(getattr(viewer, '_mesh_picking_too_dense', False)):
                message = (
                    "This displayed topology/STL surface is too dense for direct "
                    "interactive patch picking. Pick from a face selector connected "
                    "after Remesh, so the picker can use the volume-mesh surface."
                )
                try:
                    app.statusBar().showMessage(message)
                except Exception:
                    pass
                QtWidgets.QMessageBox.information(self, "Use Remeshed Surface", message)
                return "pending"
            return False

        try:
            try:
                app.statusBar().showMessage("Preparing geometry for face picking...")
            except Exception:
                pass
            worker = GraphExecutionWorker(upstream_nodes, skip_simulation=False, parent=self)
            self._pick_prepare_worker = worker

            def _finish(_results):
                try:
                    self._pick_prepare_worker = None
                    source_node, geometry = app._get_render_context_for_node(node)
                    if geometry is not None:
                        app._last_rendered_node = source_node or node
                        app._render_result_in_viewer(geometry)
                    ready = self._viewer_has_pickable_faces(viewer)
                    if not ready and hasattr(viewer, 'ensure_mesh_face_picking'):
                        ready = viewer.ensure_mesh_face_picking()
                    if ready:
                        try:
                            app.statusBar().showMessage("Geometry ready. Pick faces in the viewer.")
                        except Exception:
                            pass
                        self._start_picking_session(node)
                    else:
                        if bool(getattr(viewer, '_mesh_picking_too_dense', False)):
                            text = (
                                "The rendered surface is too dense for direct "
                                "interactive patch picking. Run Remesh first, "
                                "then pick on the remeshed surface."
                            )
                        else:
                            text = "The upstream geometry finished, but no pickable face patches were found."
                        QtWidgets.QMessageBox.information(
                            self,
                            "No Pickable Faces",
                            text,
                        )
                finally:
                    try:
                        worker.deleteLater()
                    except Exception:
                        pass

            def _error(message):
                try:
                    self._pick_prepare_worker = None
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Could Not Prepare Picker",
                        f"Could not prepare upstream geometry for face picking:\n{message}",
                    )
                finally:
                    try:
                        worker.deleteLater()
                    except Exception:
                        pass

            worker.computation_finished.connect(_finish)
            worker.computation_error.connect(_error)
            worker.start()
            return "pending"
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Could Not Prepare Picker",
                f"Could not prepare upstream geometry for face picking:\n{exc}",
            )
            return False

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

        prepared = self._prepare_viewer_for_picking(app, node)
        if prepared is not True:
            if prepared == "pending":
                return
            QtWidgets.QMessageBox.information(
                self,
                "No Pickable Faces",
                "No pickable face patches are available yet. Run the upstream "
                "geometry/remesh or topology step, then try picking again."
            )
            return

        viewer.enable_picking_mode(multi_select=True)

        # Wire done signal
        def _on_faces_picked(occ_faces):
            try:
                viewer.face_picked.disconnect(_on_faces_picked)
            except Exception:
                pass
            # Map picked face objects → indices that the InteractiveSelectFace
            # node understands. Two flavours:
            #   * Mesh virtual-face dicts already carry their own 'stored_index'
            #     (a >=1000 patch id encoding direction+component). Read it
            #     directly — no identity check needed.
            #   * OCC face objects: locate them in viewer._all_occ_faces by
            #     hashCode equality so we can record their position.
            all_occ = viewer._all_occ_faces
            picked_indices = []
            picked_labels = []
            for face in occ_faces:
                if isinstance(face, dict):
                    stored = face.get('stored_index')
                    if stored is None:
                        stored = face.get('face_index', 0)
                    picked_indices.append(int(stored))
                    picked_labels.append(
                        face.get('label')
                        or face.get('selector')
                        or None
                    )
                    continue
                for i, f in enumerate(all_occ):
                    try:
                        if face.hashCode(10000) == f.hashCode(10000):
                            picked_indices.append(i)
                            picked_labels.append(None)
                            break
                    except Exception:
                        if face is f:
                            picked_indices.append(i)
                            picked_labels.append(None)
                            break

            # Suppress the graph's property-changed handler while we write
            # both properties; otherwise each set_property fires its own
            # display_node() and the panel rebuilds 3-4 times in a row,
            # leaving stale Clear-Selection buttons (red, deleteLater-pending)
            # stacked on top of each other until the event loop drains.
            app._suppress_graph_property_changed = True
            try:
                if hasattr(node, 'set_picked_faces'):
                    node.set_picked_faces(picked_indices)
                else:
                    node.set_property('picked_face_indices',
                                      ','.join(str(i) for i in picked_indices))
                if picked_indices and any(picked_labels):
                    chunks = []
                    for idx, label in zip(picked_indices, picked_labels):
                        chunks.append(f"{idx} / {label}" if label else str(idx))
                    node.set_property(
                        'selection_label',
                        f"{len(picked_indices)} face{'s' if len(picked_indices) != 1 else ''} selected  (idx: {', '.join(chunks)})",
                    )
                if hasattr(node, '_last_hash'):
                    node._last_hash = None
            finally:
                app._suppress_graph_property_changed = False

            # Single rebuild of the inspector with final state.
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
    # ——————————————————————————————————————————————————————————

    def _build_fea_bc_ui(self, node):
        """Rich Properties Panel UI for ConstraintNode, LoadNode, PressureLoadNode."""
        node_class = node.__class__.__name__
        props = node.model.properties

        if node_class == 'ConstraintNode':
            # Use get_property (NodeGraphQt API) so we always read the live value,
            # not a potentially stale snapshot from node.model.properties.
            ct = node.get_property('constraint_type') or 'Fixed'
            if ct == 'Pinned (Fixed for solids)':
                ct = 'Pinned'

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
                for axis, ax in [('X', 'displacement_x'),
                                 ('Y', 'displacement_y'),
                                 ('Z', 'displacement_z')]:
                    val = node.get_property(ax)
                    if val is not None:
                        enabled_prop = f'{ax}_enabled'
                        enabled_value = node.get_property(enabled_prop)
                        enabled = True if enabled_value is None else bool(enabled_value)
                        row = QtWidgets.QWidget()
                        row_layout = QtWidgets.QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        active = QtWidgets.QCheckBox("Prescribe")
                        active.setChecked(enabled)
                        active.toggled.connect(
                            lambda checked, p=enabled_prop: self.update_property(p, checked)
                        )
                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setRange(-1e6, 1e6)
                        spin.setDecimals(4)
                        spin.setValue(float(val))
                        spin.valueChanged.connect(lambda v, p=ax: self.update_property(p, v))
                        row_layout.addWidget(active)
                        row_layout.addWidget(spin, 1)
                        lay.addRow(f'U{axis}:', row)

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
            btn_preview = QtWidgets.QPushButton("Preview in 3D")
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

            raw_pressure = node.get_property('pressure')
            pval = float(1.0 if raw_pressure is None else raw_pressure)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.blockSignals(True)
            spin.setValue(pval)
            spin.blockSignals(False)
            spin.setSuffix(' MPa')
            spin.valueChanged.connect(lambda v: self.update_property('pressure', v))
            lay.addRow("Pressure:", spin)

            pdir = node.get_property('direction') or 'Outward'
            dir_combo = QtWidgets.QComboBox()
            dir_combo.addItems(['Outward', 'Inward'])
            dir_combo.blockSignals(True)
            dir_combo.setCurrentText(str(pdir) if str(pdir) in ['Outward', 'Inward'] else 'Outward')
            dir_combo.blockSignals(False)
            dir_combo.currentTextChanged.connect(lambda v: self.update_property('direction', v))
            lay.addRow("Direction:", dir_combo)

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
            QLabel.metric-key { color: #B0BEC5; }
            QLabel.metric-val { color: #FAFAFA; font-weight: bold; }
            """
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self._empty = QtWidgets.QLabel("No solver results yet — run an FEA, Crash, or Topology node.")
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
        if rtype not in ('fea', 'crash', 'external_solver', 'topopt_voxel'):
            self._empty.setVisible(True)
            self._scroll.setVisible(False)
            return

        self._clear()
        self._empty.setVisible(False)
        self._scroll.setVisible(True)

        backend = data.get('backend') or ('CalculiX' if rtype == 'fea' else
                                          'OpenRadioss' if rtype == 'crash' else '—')
        if rtype == 'topopt_voxel':
            backend = data.get('backend') or 'pyMOTO'
        meta_rows = [
            ("Type",            rtype.upper()),
            ("Backend",         str(backend)),
        ]
        if 'visualization_mode' in data:
            meta_rows.append(("Visualization", str(data['visualization_mode'])))
        if 'analysis_type' in data:
            meta_rows.append(("Analysis", str(data['analysis_type'])))
        if 'external_status' in data:
            meta_rows.append(("Solver status", str(data['external_status'])))
        if 'work_dir' in data:
            meta_rows.append(("Work directory", str(data['work_dir'])))
        self._add_section("Run", meta_rows)

        if rtype == 'fea' or rtype == 'external_solver':
            metrics = []
            peak_disp = data.get('peak_displacement')
            if peak_disp is None and data.get('displacement') is not None:
                try:
                    import numpy as np
                    arr = np.asarray(data['displacement'], dtype=float)
                    if arr.size:
                        if arr.ndim == 1 and arr.size % 3 == 0:
                            arr = arr.reshape((-1, 3))
                        peak_disp = float(np.max(np.linalg.norm(arr, axis=1)))
                except Exception:
                    pass
            if peak_disp is not None:
                metrics.append(("Peak displacement", self._fmt(peak_disp, "mm")))

            peak_stress = data.get('peak_stress_nodal')
            if peak_stress is None and data.get('stress') is not None:
                try:
                    import numpy as np
                    arr = np.asarray(data['stress'], dtype=float)
                    if arr.size:
                        peak_stress = float(np.max(arr))
                except Exception:
                    pass
            if peak_stress is not None:
                metrics.append(("Peak stress (nodal extrapolated)", self._fmt(peak_stress, "MPa")))
            if data.get('stress_location') == 'gauss' and 'max_stress_gauss' in data:
                metrics.append(("Peak stress (Gauss)", self._fmt(data['max_stress_gauss'], "MPa")))
            if 'strain_energy' in data:
                metrics.append(("Strain energy", self._fmt(data['strain_energy'], "N mm")))
            if data.get('compliance') is not None:
                metrics.append(("Compliance", self._fmt(data['compliance'], "N mm")))
            if 'volume' in data:
                metrics.append(("Volume", self._fmt(data['volume'], "mm^3")))
            if 'mass' in data:
                metrics.append(("Mass", self._fmt(float(data['mass']) * 1000.0, "kg")))
            if 'deformation_scale' in data:
                try:
                    raw_scale = data['deformation_scale']
                    if isinstance(raw_scale, str):
                        text_scale = raw_scale.strip().lower()
                        scale = 1.0 if text_scale == 'auto' else float(text_scale.rstrip('x'))
                    else:
                        scale = float(raw_scale)
                    metrics.append(("Deformation scale", f"{scale:.1f}Ã—"))
                except Exception:
                    pass
            if metrics:
                self._add_section("Result", metrics)

        if rtype == 'crash':
            crash_rows = []
            if 'peak_displacement' in data:
                crash_rows.append(("Peak displacement", self._fmt(data['peak_displacement'], "mm")))
            if 'peak_stress' in data:
                crash_rows.append(("Peak Von Mises", self._fmt(data['peak_stress'], "MPa")))
            if 'absorbed_energy' in data:
                crash_rows.append(("Absorbed / internal energy", self._fmt(data['absorbed_energy'], "N·mm")))
            if 'n_failed' in data:
                crash_rows.append(("Failed elements", str(data['n_failed'])))
            if 'frames' in data and data['frames']:
                crash_rows.append(("Animation frames", str(len(data['frames']))))
            if 'energy_balance_max_error' in data:
                crash_rows.append(("Energy balance error", f"{float(data['energy_balance_max_error']) * 100:.1f}%"))
            if crash_rows:
                self._add_section("Crash result", crash_rows)

        if rtype == 'topopt_voxel':
            topo_rows = []
            if data.get('design_goal'):
                topo_rows.append(("Goal", str(data['design_goal'])))
            if data.get('target_vol_frac') is not None:
                topo_rows.append(("Material budget", f"{float(data['target_vol_frac']) * 100:.1f}%"))
            if data.get('final_vol_frac') is not None:
                topo_rows.append(("Final material", f"{float(data['final_vol_frac']) * 100:.1f}%"))
            if data.get('compliance') is not None:
                topo_rows.append(("Compliance", self._fmt(data['compliance'], "N mm")))
            if data.get('stress_pnorm') is not None:
                topo_rows.append(("Stress P-norm proxy", self._fmt(data['stress_pnorm'], "MPa")))
            if data.get('volume') is not None:
                topo_rows.append(("Recovered volume", self._fmt(data['volume'], "mm^3")))
            if data.get('mass') is not None:
                topo_rows.append(("Mass", self._fmt(float(data['mass']) * 1000.0, "kg")))
            topo_rows.append(("Iterations", str(int(data.get('iterations') or 0))))
            topo_rows.append(("Converged", "Yes" if data.get('converged') else "No"))
            self._add_section("Topology result", topo_rows)

            validation = data.get('validation_summary')
            if isinstance(validation, dict):
                validation_rows = []
                if validation.get('max_stress') is not None:
                    validation_rows.append(("Validated peak stress", self._fmt(validation['max_stress'], "MPa")))
                if validation.get('compliance') is not None:
                    validation_rows.append(("Validated compliance", self._fmt(validation['compliance'], "N mm")))
                if validation_rows:
                    self._add_section("CalculiX validation", validation_rows)

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

    @staticmethod
    def _calculix_status():
        """Return a concise, non-launching CalculiX availability check."""
        try:
            from pylcss.solver_backends.calculix import resolve_calculix_executable
            executable = resolve_calculix_executable()
        except Exception as exc:
            return False, f"CalculiX availability check failed: {exc}"
        if not executable:
            return False, (
                "CalculiX unavailable. Deck-only generation remains available; "
                "install with `python scripts/install_solvers.py --only ccx`."
            )
        return True, f"CalculiX ready: {executable}"


    @staticmethod
    def _openradioss_status():
        """Return a concise, non-launching availability check for the palette."""
        try:
            from pylcss.solver_backends.common import resolve_executable
            from pylcss.solver_backends.radioss_reader import resolve_anim_to_vtk

            starter = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
                ("starter_win64.exe", "starter_win64", "starter_linux64_gf"),
            )
            engine = resolve_executable(
                None,
                ("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
                ("engine_win64.exe", "engine_win64", "engine_linux64_gf"),
            )
            converter = resolve_anim_to_vtk()
        except Exception as exc:
            return False, f"Availability check failed: {exc}"

        missing = [
            label for label, value in (
                ("Starter", starter), ("Engine", engine), ("anim_to_vtk", converter)
            ) if not value
        ]
        if missing:
            return False, (
                "OpenRadioss unavailable: missing " + ", ".join(missing) + ".\n"
                "Deck-only generation remains available; install with "
                "`python scripts/install_solvers.py --only radioss`."
            )
        return True, "OpenRadioss ready: Starter, Engine, and anim_to_vtk detected."

    def __init__(self, spawn_callback):
        super(LibraryPanel, self).__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)
        radioss_ready, radioss_status = self._openradioss_status()
        calculix_ready, calculix_status = self._calculix_status()

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
                ("FreeCAD Part (interactive)", "com.cad.freecad_part",
                 "Interactive parametric CAD authored in FreeCAD's own GUI.\n"
                 "Double-click the node to launch FreeCAD on a node-owned .FCStd file.\n"
                 "Draw sketches, add PartDesign features, set named faces / FEM loads;\n"
                 "save inside FreeCAD and PyLCSS auto-imports the geometry via BREP +\n"
                 "sidecar JSON.  Requires FreeCAD installed locally\n"
                 "(`python scripts/install_solvers.py --only freecad`)."),
                ("Import STEP", "com.cad.import_step",
                 "Import a STEP / IGES CAD file as the upstream geometry."),
                ("Import Mesh", "com.cad.import_stl",
                 "Import an STL / OBJ surface mesh."),
            ],

            # ───────────────────────────────────────────────────────────────
            # WORKBENCH: SIMULATION (FEA)
            # ───────────────────────────────────────────────────────────────
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
                ("Topology Opt (Voxel)", "com.cad.sim.topopt_voxel", "Run structured 3D voxel topology optimization with pyMOTO"),
                ("Remesh Surface", "com.cad.sim.remesh", "Convert TopOpt surface to volume mesh"),
            ],

            # ───────────────────────────────────────────────────────────────
            # WORKBENCH: CRASH / IMPACT SIMULATION
            # ───────────────────────────────────────────────────────────────
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

            # ───────────────────────────────────────────────────────────────
            # WORKBENCH: ANALYSIS & UTILITIES
            # ───────────────────────────────────────────────────────────────
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
                if node_id == "com.cad.sim.solver":
                    tooltip += "\n\n" + calculix_status
                    item.setData(0, QtCore.Qt.UserRole + 1, calculix_ready)
                    if not calculix_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
                elif node_id in ("com.cad.sim.crash_solver", "com.cad.sim.radioss_deck"):
                    tooltip += "\n\n" + radioss_status
                    item.setData(0, QtCore.Qt.UserRole + 1, radioss_ready)
                    if not radioss_ready:
                        item.setForeground(0, QtGui.QColor("#FFB74D"))
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


# Lazy imports to avoid circular dependency with the assistant module
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
        # Tear down per-node FreeCAD launcher / watcher state when a node is
        # removed from the graph so the QProcess + QFileSystemWatcher don't
        # leak and so reopening the deleted node's file ID later starts
        # fresh state.
        try:
            self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        except Exception:
            pass
        
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
            except Exception as exc:
                logger.warning(
                    "Could not register CAD node %s: %s",
                    getattr(node_class, "__name__", node_class), exc,
                )
    
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

        # Per-node context menu commands. NodeGraphQt scopes node-context
        # menu items to a specific node type via its identifier, so this
        # only appears on FreeCadPartNode right-click. The handler receives
        # (graph, node).
        try:
            node_menu = self.graph.get_context_menu('nodes')
            node_menu.add_command(
                'Open in FreeCAD',
                lambda g, n: self._open_freecad_for_node(n),
                node_type='com.cad.freecad_part.FreeCadPartNode',
            )
        except Exception:
            # NodeGraphQt's node-context menu API changes between versions;
            # if this signature isn't supported, double-click still works.
            pass

        # Right panel
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.properties = PropertiesPanel()
        try:
            self.properties.property_changed.connect(self._on_property_changed)
        except Exception:
            pass
        self.results = None
        self.timeline = TimelinePanel()
        right_splitter.addWidget(self.properties)
        right_splitter.addWidget(self.timeline)
        right_splitter.setSizes([520, 180])
        
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

        # Edit shortcuts.  NodeGraphQt's graph_widget already binds Ctrl+Z /
        # Ctrl+Y internally, so registering them as QShortcut on the same
        # parent caused Qt's "Ambiguous shortcut overload" warning at
        # runtime.  We bind through the toolbar actions instead (see
        # _create_toolbar's act_undo / act_redo) so there's exactly one
        # owner per key chord.  Delete is unique to us -- keep it.
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
        # setShortcut() on the action makes Qt route Ctrl+Z / Ctrl+Y to our
        # handler from a single owner; the previous QShortcut bindings were
        # removed in _setup_shortcuts to stop the "Ambiguous shortcut" warning.
        # QKeySequence is imported locally because _setup_shortcuts (the
        # other caller in this file) does the same -- there's no
        # module-level import to lean on.
        from PySide6.QtGui import QKeySequence
        act_undo = self.toolbar.addAction(_icon("fa5s.undo"), "", self._undo)
        act_undo.setShortcut(QKeySequence.Undo)
        act_undo.setShortcutContext(QtCore.Qt.WindowShortcut)
        act_undo.setToolTip("Undo (Ctrl+Z)")
        act_redo = self.toolbar.addAction(_icon("fa5s.redo"), "", self._redo)
        act_redo.setShortcut(QKeySequence.Redo)
        act_redo.setShortcutContext(QtCore.Qt.WindowShortcut)
        act_redo.setToolTip("Redo (Ctrl+Y)")
        self.toolbar.addSeparator()

        # ── Run is the most-used action — give it accent color via icon ─
        self.run_action = self.toolbar.addAction(_icon("fa5s.play", "#66BB6A"), "Run", self._run_action)
        self.run_action.setToolTip(
            "Run the selected node and its inputs only (siblings/downstream are "
            "skipped). With nothing selected, runs the whole graph."
        )
        self.run_action.setShortcut("F5")
        self.run_action.setToolTip("Execute the node graph (F5)")
        self.stop_action = self.toolbar.addAction(_icon("fa5s.stop", "#EF5350"), "Stop", self._cancel_execution)
        self.stop_action.setShortcut("Shift+F5")
        self.stop_action.setToolTip("Stop TopOpt at the next iteration boundary (Shift+F5)")

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
            if node_id == "com.cad.sim.topopt_voxel":
                try:
                    self._suppress_graph_property_changed = True
                    self._apply_topopt_industrial_defaults(node)
                except Exception:
                    pass
                finally:
                    self._suppress_graph_property_changed = False
            self.graph.add_node(node)
            # record undo action
            try:
                self._push_undo({'type': 'add_node', 'node': node})
            except Exception:
                pass

            self.timeline.add_event(f"Added {label} node")
            self.statusBar().showMessage(f"Created {label}")
            return node
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
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
            return (
                'mesh' in obj
                or ('vertices' in obj and 'faces' in obj)
                or obj.get('type') == 'topopt_voxel'
                or (obj.get('type') == 'crash' and bool(obj.get('frames')))
            )
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
            if isinstance(obj, dict) and self.results is not None:
                try:
                    self.results.show_result(obj)
                except Exception:
                    pass
        elif self._is_2d_sketch(obj):
            self.viewer.render_sketch(obj)
        else:
            self.viewer.render_shape(obj)
        # Remember what's on screen so _on_node_selected can skip re-rendering
        # the same geometry when the user clicks a sibling node that resolves
        # to the same upstream payload.
        self._last_rendered_geom_id = id(obj)

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

    @staticmethod
    def _is_topopt_result_consumer(node):
        return False

    @staticmethod
    def _is_topopt_render_result(obj):
        return (
            isinstance(obj, dict)
            and obj.get('type') in {'topopt_voxel'}
        )

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

    def _find_upstream_topopt_result(self, node, visited=None):
        """Return the nearest cached topology result feeding a topology consumer.

        Unlike _find_upstream_renderable, this intentionally does not keep
        walking into the design-domain mesh/shape when the TopOpt node has not
        run yet.  That prevents uncomputed Validation/CAD nodes from showing an
        unrelated box or base CAD body.
        """
        if node is None:
            return None, None
        if visited is None:
            visited = set()
        marker = id(node)
        if marker in visited:
            return None, None
        visited.add(marker)

        result = getattr(node, '_last_result', None)
        if self._is_topopt_render_result(result):
            return node, result

        try:
            port = node.get_input('topology_result')
        except Exception:
            port = None
        if port is None:
            return None, None

        try:
            connected_ports = list(port.connected_ports())
        except Exception:
            connected_ports = []

        for conn_port in connected_ports:
            upstream = conn_port.node()
            result = getattr(upstream, '_last_result', None)
            if self._is_topopt_render_result(result):
                return upstream, result
            if self._is_topopt_result_consumer(upstream):
                source, renderable = self._find_upstream_topopt_result(upstream, visited)
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
        """Handle node selection.

        NodeGraphQt can fire node_selected rapidly (rubber-band drag,
        keyboard arrow selection, programmatic selection from undo/redo),
        and each tick triggers a synchronous inspector rebuild + viewer
        re-render which causes a perceptible freeze for large meshes.
        Two cheap guards make this robust:
          * Skip everything when the node is already the current selection.
          * Skip the heavy viewer re-render when the geometry to be drawn
            is the same object as what's already on screen — overlays and
            highlights still update because they're cheap.
        """
        if not node:
            return
        if self.properties.current_node is node and self._last_rendered_node is node:
            return

        self.properties.display_node(node)
        self.statusBar().showMessage(f"Selected: {node.name}")

        if self._is_topopt_result_consumer(node):
            own = getattr(node, '_last_result', None)
            if self._is_renderable_result(own):
                source_node, geometry = node, own
            else:
                source_node, geometry = self._find_upstream_topopt_result(node)
        else:
            source_node, geometry = self._get_render_context_for_node(node)

        if geometry is not None:
            self._last_rendered_node = source_node or node
            if getattr(self, '_last_rendered_geom_id', None) != id(geometry):
                self._render_result_in_viewer(geometry)  # also updates _last_rendered_geom_id

            # Re-apply face highlights if it's the interactive picker
            if node.__class__.__name__ == 'InteractiveSelectFaceNode':
                raw = node.get_property('picked_face_indices') or ''
                idx_list = [int(x.strip()) for x in raw.split(',') if x.strip().isdigit()]
                if idx_list and hasattr(self.viewer, 'highlight_faces'):
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
            if self._is_topopt_result_consumer(node):
                self._last_rendered_node = node
                self._last_rendered_geom_id = None
                self.viewer.clear()
                self.statusBar().showMessage(
                    "Run Simulation to compute the upstream topology result first."
                )
                return
            self._last_rendered_node = node
            self._last_rendered_geom_id = None
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
        if isinstance(occ_face, dict):
            center = occ_face.get('center')
            if center is not None:
                return [float(v) for v in center[:3]]
            bbox = occ_face.get('bbox') or {}
            try:
                return [
                    (float(bbox['xmin']) + float(bbox['xmax'])) / 2.0,
                    (float(bbox['ymin']) + float(bbox['ymax'])) / 2.0,
                    (float(bbox['zmin']) + float(bbox['zmax'])) / 2.0,
                ]
            except Exception:
                return [0.0, 0.0, 0.0]
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
        if anchor_cls in ('SolverNode', 'CrashSolverNode', 'TopologyOptVoxelNode'):
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
                    if isinstance(g, dict) and (g.get('mesh_selection') or g.get('node_ids') is not None):
                        mesh_viz = dict(viz_meta)
                        if not mesh_viz.get('faces'):
                            mesh_viz['faces'] = [{
                                'center': g.get('center'),
                                'points': g.get('points'),
                                'bbox': g.get('bbox'),
                            }]
                        constraint_faces.append({
                            'pos': g.get('center'),
                            'points': g.get('points'),
                            'viz': mesh_viz,
                            'mesh_selection': g,
                        })
                    else:
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
                        is_mesh_selection = isinstance(g, dict) and (
                            g.get('mesh_selection') or g.get('node_ids') is not None
                        )
                        if is_mesh_selection:
                            load_faces.append({'mesh_selection': g})
                        else:
                            load_faces.append(g)
                        centroid = self._get_face_centroid(g)
                        # FIX #V2: pass magnitude for log-scaled arrow in the viewer
                        load_vectors.append({
                            'centroid': centroid,
                            'face': None if is_mesh_selection else g,
                            'mesh_selection': g if is_mesh_selection else None,
                            'vector': vec,
                            'magnitude_N': force_mag,
                            'points': g.get('points') if isinstance(g, dict) else None,
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
                        is_mesh_selection = isinstance(g, dict) and (
                            g.get('mesh_selection') or g.get('node_ids') is not None
                        )
                        centroid = self._get_face_centroid(g)
                        load_vectors.append({
                            'centroid': centroid,
                            'face': None if is_mesh_selection else g,
                            'mesh_selection': g if is_mesh_selection else None,
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
            # During crash animation playback the part is deforming, but BC
            # overlays are built from the *undeformed* face geometry — they'd
            # float in the original location and never move with the animation
            # (the "impact velocity face stays same" artifact).  Skip them; the
            # impact direction is conveyed by the moving wall instead.
            if getattr(self.viewer, '_crash_base_data', None) is not None:
                self.viewer.set_bc_overlay_data()
                self.viewer.render_bc_overlays()
                return

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
        the same one the inspector's *Edit Code…* button uses.
        For ``FreeCadPartNode`` this launches the FreeCAD GUI subprocess on
        the node's .FCStd file and wires the save-watcher so the viewer
        refreshes automatically when the user saves inside FreeCAD.
        Other node types swallow the double-click so NodeGraphQt's default
        subgraph popup doesn't appear.
        """
        try:
            class_name = node.__class__.__name__
            if class_name == 'CadQueryCodeNode':
                # The inspector holds the editor-open helper.
                self.properties._open_cad_code_editor(node)
                self.timeline.add_event(f"Opened code editor for {node.name()}")
                return
            if class_name == 'FreeCadPartNode':
                self._open_freecad_for_node(node)
                return
        except Exception:
            # Fall through to the silent default behaviour on any error.
            pass
        try:
            node_label = node.name() if callable(node.name) else node.name
        except Exception:
            node_label = '<unknown>'
        self.timeline.add_event(f"Double-clicked {node_label} (Popup disabled)")

    def _open_freecad_for_node(self, node):
        """Spawn FreeCAD on a FreeCadPartNode's .FCStd file and wire a
        per-node :class:`FCStdWatcher` so saves trigger a viewer refresh.

        Idempotent: opening the same node twice re-focuses the existing
        FreeCAD instance (the launcher's per-file registry guarantees this)
        and reuses the watcher already attached to the node.
        """
        try:
            from pylcss.design_studio.freecad_bridge.launcher import FreeCadLauncher
            from pylcss.design_studio.freecad_bridge.watcher import FCStdWatcher
        except ImportError as exc:
            self.timeline.add_event(f"FreeCAD bridge unavailable: {exc}")
            return

        fcstd_path = node.fcstd_path()

        launcher = getattr(node, '_freecad_launcher', None)
        if launcher is None:
            launcher = FreeCadLauncher(parent=self)
            node._freecad_launcher = launcher
            launcher.error_occurred.connect(
                lambda path, msg, n=node: self.timeline.add_event(
                    f"FreeCAD error for {n.name()}: {msg}"
                )
            )
            launcher.process_exited.connect(
                lambda path, code, n=node: self.timeline.add_event(
                    f"FreeCAD exited (code {code}) for {n.name()}"
                )
            )

        if not launcher.is_available():
            self.timeline.add_event(
                "FreeCAD not installed — run "
                "`python scripts/install_solvers.py --only freecad`"
            )
            return

        # Wire (or re-wire) the save-watcher exactly once per node so each
        # save in FreeCAD triggers _cad_execute and the viewer updates.
        watcher = getattr(node, '_freecad_watcher', None)
        if watcher is None or str(watcher.fcstd_path) != str(fcstd_path):
            if watcher is not None:
                try:
                    watcher.stop()
                except Exception:
                    pass
            watcher = FCStdWatcher(fcstd_path, parent=self)
            watcher.saved.connect(
                lambda _p, n=node: self._on_freecad_save(n)
            )
            node._freecad_watcher = watcher

        ok = launcher.open(fcstd_path)
        if ok:
            self.timeline.add_event(f"Opened FreeCAD for {node.name()}")

    def _on_freecad_save(self, node):
        """FCStdWatcher fired -- mark the node dirty and trigger a CAD
        execute so the new geometry shows up in the viewer."""
        setattr(node, '_dirty', True)
        self.timeline.add_event(f"FreeCAD saved: refreshing {node.name()}")
        try:
            # Re-use whichever execution entry point the rest of the widget
            # uses for "graph property changed -> re-run".
            if hasattr(self, '_cad_execute'):
                self._cad_execute()
            elif hasattr(self, 'execute_graph'):
                self.execute_graph()
        except Exception as exc:
            self.timeline.add_event(f"CAD re-execute failed: {exc}")

    def _on_nodes_deleted(self, node_ids):
        """Release per-node FreeCAD launcher + watcher when the node is
        removed from the graph. NodeGraphQt emits node IDs (not nodes) here
        because the node objects have already been torn down -- we keep
        the launcher/watcher in a per-node attribute, so look them up via
        the still-alive references in our own bookkeeping."""
        # NodeGraphQt API gives us only IDs; we can't fetch the nodes back
        # because they're gone. Best-effort cleanup: walk every still-alive
        # node and ensure orphans get shut down on the next idle cycle.
        try:
            for node in self.graph.all_nodes():
                if not hasattr(node, '_freecad_watcher'):
                    continue
                # Node still alive -- nothing to do here.
        except Exception:
            pass
        # Anything we forget here gets garbage-collected when the cad_widget
        # itself is torn down (the launcher's QProcess is parented to us).


    # Property names whose changes don't affect anything the inspector renders
    # (purely internal book-keeping written by node.run() / set_error /
    # clear_error). Rebuilding the inspector for these is wasted work — every
    # worker tick used to fire 2-4 of them per executed node.
    _SILENT_PROP_NAMES = frozenset({
        'error_state',
        'error_message',
    })

    def _on_graph_property_changed(self, node, prop_name, prop_value):
        """Handle property changes from the graph (including widgets on nodes)."""
        # Mark node as dirty so it re-executes
        setattr(node, '_dirty', True)

        if getattr(self, '_suppress_graph_property_changed', False):
            return

        # Update the properties panel if this node is selected.
        # Skip if the inspector itself triggered the change to avoid a reset loop.
        # Skip "silent" book-keeping props that the panel never displays —
        # rebuilding for those just causes UI freezes during graph execution.
        if (prop_name not in self._SILENT_PROP_NAMES
                and self.properties.current_node == node
                and not self.properties._updating_property):
            self.properties.display_node(node)

        # SPECIAL CASE: Visualization mode changes should update display immediately
        # without requiring full graph re-execution
        if prop_name in ('visualization', 'deformation_scale', 'disp_scale'):
            cached_result = getattr(node, '_last_result', None)
            if cached_result is not None and isinstance(cached_result, dict):
                # Update the visualization_mode in the cached result
                if prop_name == 'visualization':
                    cached_result['visualization_mode'] = prop_value
                elif prop_name == 'deformation_scale':
                    text = str(prop_value).strip().lower()
                    if text == 'auto':
                        cached_result['deformation_scale'] = cached_result.get(
                            'auto_deformation_scale', 1.0)
                    else:
                        try:
                            cached_result['deformation_scale'] = float(text.rstrip('x'))
                        except ValueError:
                            pass
                else:
                    cached_result['disp_scale'] = float(prop_value)
                
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
            self.statusBar().showMessage("Computation complete")
            self.timeline.add_event("Graph execution finished")

            # 2. Update Visualization (Must be done on Main Thread!)
            try:
                # Decide what to draw after a run.  Priority:
                #   1. The selected node's OWN result, if it's renderable
                #      (so clicking CAD shows the B-rep, validation shows FEA,
                #       topopt shows density — note _is_renderable_result
                #       recognises the 'topopt_voxel' dict, which has a
                #       'density' field but no 'mesh'/'vertices' key).
                #   2. Otherwise the optimisation / FEA result produced in this
                #      graph, so the topopt outcome stays visible even when
                #      downstream CAD / STEP-export nodes are wired after it.
                #   3. Otherwise an upstream preview (design domain / mesh).
                #   4. Otherwise the last-rendered node.
                selected = next(iter(self.graph.selected_nodes()), None)
                target_node = None
                geom = None
                prefer_topopt = bool(getattr(self, '_prefer_topopt_after_run', False))

                if prefer_topopt:
                    sim_node = self._find_renderable_simulation_node()
                    if sim_node is not None:
                        result = getattr(sim_node, '_last_result', None)
                        if self._is_topopt_render_result(result):
                            target_node, geom = sim_node, result
                    if geom is None:
                        preview = getattr(self, '_last_topopt_preview_payload', None)
                        if self._is_renderable_result(preview):
                            geom = preview

                if geom is None and selected is not None:
                    own = results.get(selected, getattr(selected, '_last_result', None))
                    if self._is_renderable_result(own):
                        target_node, geom = selected, own

                if geom is None and selected is not None:
                    src, upstream_geom = self._get_render_context_for_node(selected)
                    if upstream_geom is not None:
                        target_node, geom = (src or selected), upstream_geom

                if geom is None:
                    sim_node = self._find_renderable_simulation_node()
                    if sim_node is not None:
                        target_node = sim_node
                        geom = getattr(sim_node, '_last_result', None)

                if geom is None:
                    last = getattr(self, '_last_rendered_node', None)
                    if last is not None:
                        cached = getattr(last, '_last_result', None)
                        if self._is_renderable_result(cached):
                            target_node, geom = last, cached

                if geom is None:
                    preview = getattr(self, '_last_topopt_preview_payload', None)
                    if self._is_renderable_result(preview):
                        geom = preview

                if geom is not None:
                    if target_node is not None:
                        self._last_rendered_node = target_node
                    self._render_result_in_viewer(geom)
                    try:
                        self._show_bc_for_node(target_node)
                    except Exception:
                        pass
                else:
                    self.viewer.clear()

            except Exception:
                pass
        finally:
            self._prefer_topopt_after_run = False
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
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.timeline.add_event(f"Execution failed: {error_msg}")
        try:
            sim_node = self._find_renderable_simulation_node()
            if sim_node is not None:
                result = getattr(sim_node, '_last_result', None)
                if self._is_renderable_result(result):
                    self._last_rendered_node = sim_node
                    self._render_result_in_viewer(result)
            elif self._is_renderable_result(getattr(self, '_last_topopt_preview_payload', None)):
                self._render_result_in_viewer(self._last_topopt_preview_payload)
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
            self.statusBar().showMessage("Fit to view")
        except Exception as e:
            # Fallback - try basic centering
            try:
                self.graph.center_selection()
            except:
                pass
            self.statusBar().showMessage("View adjusted")
    
    def _reset_view(self):
        """Reset the 3D viewer to default orientation."""
        try:
            # Reset the 3D viewer camera
            if hasattr(self.viewer, 'renderer') and self.viewer.renderer:
                self.viewer.renderer.ResetCamera()
                if hasattr(self.viewer, 'iren') and self.viewer.iren:
                    self.viewer.iren.GetRenderWindow().Render()
            self.statusBar().showMessage("3D view reset")
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
            if node_class in [
                'SolverNode', 'MeshNode', 'CrashSolverNode', 'RunRadiossDeckNode'
            ]:
                sim_nodes.append(node)
        
        if not sim_nodes:
            QtWidgets.QMessageBox.information(
                self, "No Simulation",
                "No simulation nodes found in the graph.\n\n"
                "Add FEA nodes (Material, Mesh, Constraint, Load, Solver), "
                "a Crash Solver / Run Radioss Deck node, or a Topology "
                "Optimization node to run a simulation."
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
                self.statusBar().showMessage(f"Report saved to {fname}")
        
        save_btn.clicked.connect(save_report)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()
        self.statusBar().showMessage("Report generated")
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
                
                # Code-first and scalar nodes don't need inputs
                node_class = node.__class__.__name__
                is_primitive = node_class in [
                    'CadQueryCodeNode', 'NumberNode', 'VariableNode',
                    'MaterialNode', 'CrashMaterialNode', 'RunRadiossDeckNode',
                ]
                is_export = 'Export' in node_class
                
                if not is_primitive and not has_input:
                    warnings.append(f"{node.name()} has no connected inputs")
                
                if not is_export and not has_output:
                    # Check if it's a terminal node (not an issue)
                    if node_class not in ['SolverNode']:
                        pass  # Non-terminal nodes without outputs are fine
            
            # Check for simulation setup
            has_mesh = any(n.__class__.__name__ in ('MeshNode', 'RemeshNode')
                           for n in all_nodes)
            has_solver = any(n.__class__.__name__ == 'SolverNode' for n in all_nodes)
            has_material = any(n.__class__.__name__ == 'MaterialNode' for n in all_nodes)
            has_constraint = any(n.__class__.__name__ == 'ConstraintNode' for n in all_nodes)
            has_load = any(n.__class__.__name__ in ('LoadNode', 'PressureLoadNode')
                           for n in all_nodes)
            has_prescribed_displacement = any(
                n.__class__.__name__ == 'ConstraintNode'
                and n.get_property('constraint_type') == 'Displacement'
                and any(
                    n.get_property(f'displacement_{axis}_enabled') is not False
                    and abs(float(n.get_property(f'displacement_{axis}') or 0.0)) > 1e-15
                    for axis in ('x', 'y', 'z')
                ) for n in all_nodes
            )
            
            if has_solver:
                if not has_mesh:
                    issues.append("Solver requires a Mesh node")
                if not has_material:
                    issues.append("Solver requires a Material node")
                if not has_constraint:
                    warnings.append("Solver may need constraint nodes (fixed supports)")
                if not has_load and not has_prescribed_displacement:
                    warnings.append("Solver may need load nodes")
        
        # Show results
        if not issues and not warnings:
            QtWidgets.QMessageBox.information(
                self, "Validation Complete",
                "Model is valid.\n\n"
                f"Total nodes: {len(all_nodes)}"
            )
            self.statusBar().showMessage("Model valid")
        else:
            msg = ""
            if issues:
                msg += "ERRORS:\n" + "\n".join(f"  - {i}" for i in issues) + "\n\n"
            if warnings:
                msg += "WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)
            
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Validation Results")
            box.setText(f"Found {len(issues)} errors and {len(warnings)} warnings")
            box.setDetailedText(msg)
            box.setIcon(QtWidgets.QMessageBox.Warning if issues else QtWidgets.QMessageBox.Information)
            box.exec()
            
            if issues:
                self.statusBar().showMessage(f"{len(issues)} validation errors")
            else:
                self.statusBar().showMessage(f"Valid with {len(warnings)} warnings")
        
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
        self.statusBar().showMessage(message)
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
            if not isinstance(result, dict):
                continue
            rtype = result.get('type')
            # Voxel topology-opt results carry 'density'/'recovered_shape'
            # instead of a scikit-fem 'mesh', so accept them explicitly.
            if rtype == 'topopt_voxel':
                return node
            if result.get('mesh') is not None and rtype in {'fea', 'crash'}:
                return node
        return None

    def _find_renderable_simulation_node(self):
        """Return a graph node whose cached result is a renderable simulation
        (topopt density, FEA field, crash frames, or a recovered shape).

        Prefers the optimisation / FEA *producers* over downstream consumers
        so the topopt result stays visible after a run even when CAD / export
        nodes are wired after it.
        """
        sim_nodes = []
        try:
            all_nodes = list(self.graph.all_nodes())
        except Exception:
            return None
        for node in all_nodes:
            result = getattr(node, '_last_result', None)
            if self._is_simulation_render_result(result):
                sim_nodes.append(node)

        def _rank(node):
            result = getattr(node, '_last_result', None)
            rtype = result.get('type') if isinstance(result, dict) else None
            if rtype in ('topopt_voxel',):
                return 0
            if rtype == 'fea':
                return 1
            if rtype == 'crash':
                return 2
            if rtype == 'remesh':
                return 3
            if isinstance(result, dict) and 'vertices' in result and 'faces' in result:
                return 4
            return 3

        sim_nodes.sort(key=_rank)
        return sim_nodes[0] if sim_nodes else None

    @staticmethod
    def _build_topopt_export_payload(node, result):
        """Create portable JSON/HDF5 data for a structured voxel result."""
        import numpy as np

        density = np.asarray(result.get('density'), dtype=float)
        if density.ndim != 3 or density.size == 0:
            raise ValueError("The topology result has no 3-D density field to export.")

        node_name = getattr(node, 'name', None)
        if callable(node_name):
            node_name = node_name()
        node_name = node_name or getattr(node, 'NODE_NAME', None) or node.__class__.__name__
        metadata = {
            'node_name': str(node_name),
            'node_class': node.__class__.__name__,
            'simulation_type': 'topopt_voxel',
            'visualization_mode': str(result.get('visualization_mode', '')),
            'exported_at': datetime.now().isoformat(),
            'cell_type': 'voxel',
            'cell_count': int(density.size),
            'grid_shape': [int(v) for v in density.shape],
        }

        summary = {}
        for key in (
            'target_vol_frac', 'final_vol_frac', 'bounding_vol_frac',
            'compliance', 'stress_pnorm', 'volume', 'mass', 'total_volume',
            'iterations', 'density_cutoff',
        ):
            value = result.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary[key] = float(value)
        summary['converged'] = bool(result.get('converged'))
        summary['message'] = str(result.get('message') or '')
        summary['design_goal'] = str(result.get('design_goal') or '')

        fields = {'density': density.tolist()}
        hdf5_datasets = {'voxel/density': density}
        for key in ('design_density', 'design_domain'):
            value = result.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.shape == density.shape:
                fields[key] = arr.tolist()
                hdf5_datasets[f'voxel/{key}'] = arr

        history = {}
        for key in ('compliance_history', 'change_history', 'stress_history', 'objective_history'):
            value = result.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            history[key] = arr.tolist()
            hdf5_datasets[f'history/{key}'] = arr

        json_payload = {
            'metadata': metadata,
            'summary': summary,
            'bounds': result.get('bounds'),
            'voxel_fields': fields,
            'history': history,
        }
        recovered = result.get('recovered_shape')
        if isinstance(recovered, dict) and recovered.get('vertices') is not None and recovered.get('faces') is not None:
            vertices = np.asarray(recovered['vertices'], dtype=float)
            faces = np.asarray(recovered['faces'], dtype=int)
            json_payload['recovered_shape'] = {
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
            }
            hdf5_datasets['recovered_shape/vertices'] = vertices
            hdf5_datasets['recovered_shape/faces'] = faces
        return json_payload, hdf5_datasets, metadata

    def _build_simulation_export_payload(self, node):
        """Create portable JSON/HDF5 payloads from a cached simulation result."""
        import numpy as np

        result = getattr(node, '_last_result', None)
        if not isinstance(result, dict):
            raise ValueError("The selected node has no exportable simulation result.")
        if result.get('type') == 'topopt_voxel':
            return self._build_topopt_export_payload(node, result)
        if result.get('mesh') is None:
            raise ValueError("The selected node has no exportable simulation mesh.")

        mesh = result['mesh']
        if not hasattr(mesh, 'p') or not hasattr(mesh, 't'):
            raise ValueError("Only scikit-fem style simulation meshes are supported for export.")

        points = np.asarray(mesh.p.T, dtype=float)
        if points.ndim != 2:
            raise ValueError("Invalid mesh point array.")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        from pylcss.solver_backends.common import tet10_connectivity
        quadratic_connectivity = tet10_connectivity(mesh)
        connectivity = quadratic_connectivity if quadratic_connectivity is not None else mesh.t
        cells = np.asarray(connectivity.T, dtype=int)
        if cells.ndim != 2:
            raise ValueError("Invalid mesh connectivity array.")

        n_points = points.shape[0]
        n_cells = cells.shape[0]
        nodes_per_cell = cells.shape[1]
        cell_type_map = {
            2: 'line', 3: 'triangle', 4: 'tetra', 8: 'hexahedron', 10: 'tetra10'}
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
            'peak_stress_nodal',
            'strain_energy',
            'compliance',
            'volume',
            'mass',
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
            visualization_only_props = [
                'visualization', 'deformation_scale', 'disp_scale', 'density_cutoff', 'element_type']
            
            if prop_name in visualization_only_props:
                # Check if the node has cached results (_last_result)
                cached_result = getattr(node, '_last_result', None)
                if cached_result is not None:
                    # Update the cached dictionary so the renderer knows what to draw
                    if isinstance(cached_result, dict):
                        if prop_name == 'visualization':
                            cached_result['visualization_mode'] = new
                        elif prop_name == 'deformation_scale':
                            text = str(new).strip().lower()
                            if text == 'auto':
                                cached_result['deformation_scale'] = cached_result.get(
                                    'auto_deformation_scale', 1.0)
                            else:
                                try:
                                    cached_result['deformation_scale'] = float(text.rstrip('x'))
                                except ValueError:
                                    pass
                        elif prop_name == 'disp_scale':
                            cached_result['disp_scale'] = float(new)
                        elif prop_name == 'density_cutoff':
                            cached_result['density_cutoff'] = new
                        if (
                            cached_result.get('type') == 'topopt_voxel'
                            and cached_result.get('density') is not None
                            and (
                                prop_name == 'density_cutoff'
                                or cached_result.get('visualization_mode') == 'Recovered Shape'
                            )
                        ):
                            self._refresh_topopt_recovered_shape(node, cached_result)

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
                            self.window().statusBar().showMessage(f"Updated {prop_name} display")
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

    @staticmethod
    def _port_has_connections(node, port_name):
        try:
            port = node.get_input(port_name)
            return bool(port and port.connected_ports())
        except Exception:
            return False

    @staticmethod
    def _topopt_has_property_support(node):
        support_props = (
            'left_support', 'right_support', 'top_support',
            'bottom_support', 'front_support', 'back_support',
        )
        for prop in support_props:
            value = str(node.get_property(prop) or '').strip().lower()
            if value and value != 'none':
                return True

        text = str(node.get_property('support_regions') or '').strip()
        if not text or text == '[]':
            return False
        try:
            regions = json.loads(text)
            return bool(regions)
        except Exception:
            return True

    def _topopt_preflight_error(self, node):
        if getattr(node, "__identifier__", "") != "com.cad.sim.topopt_voxel":
            return None
        # Design domain: a mesh input is mandatory. Without one the optimiser
        # falls back to a default unit voxel grid and produces meaningless
        # results — so block the run rather than let it appear to "succeed".
        if not self._port_has_connections(node, 'mesh'):
            return (
                "Topology Opt needs a design-domain mesh. Connect a Remesh, "
                "Import STL, or FreeCAD Part node to the TopOpt 'mesh' input."
            )
        if not self._port_has_connections(node, 'material'):
            return (
                "Topology Opt needs a Material connection so stiffness, stress, "
                "mass, and downstream FEA use consistent units."
            )

        # Fixed support: either a property-based support (e.g. left_support)
        # or a wired Constraint node on the 'constraints' input.
        if not self._port_has_connections(node, 'constraints'):
            return (
                "Topology Opt needs at least one connected Constraint selected "
                "on the current design domain."
            )
        if not self._port_has_connections(node, 'loads'):
            return (
                "Topology Opt needs at least one connected Force Load or Pressure Load. "
                "The GUI does not use hidden fallback loads."
            )

        return None

    @staticmethod
    def _topopt_cached_mesh_value(node):
        try:
            port = node.get_input('mesh')
            connected = list(port.connected_ports()) if port else []
        except Exception:
            connected = []
        if not connected:
            return None
        source_port = connected[0]
        source = source_port.node()
        value = getattr(source, '_last_result', None)
        if isinstance(value, dict):
            try:
                output_name = source_port.name()
            except Exception:
                output_name = ''
            if output_name and output_name in value:
                value = value[output_name]
            elif 'mesh' in value:
                value = value['mesh']
        return value

    @staticmethod
    def _topopt_spans_from_value(value):
        try:
            import numpy as np

            if isinstance(value, dict):
                if 'mesh' in value:
                    return ProfessionalCadApp._topopt_spans_from_value(value['mesh'])
                if 'vertices' in value:
                    pts = np.asarray(value['vertices'], dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
                        spans = pts[:, :3].max(axis=0) - pts[:, :3].min(axis=0)
                        positive = spans[spans > 1e-9]
                        if positive.size:
                            return np.where(spans > 1e-9, spans, float(positive.min()))
            if hasattr(value, 'p'):
                pts = np.asarray(value.p, dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] > 0:
                    spans = pts[:3].max(axis=1) - pts[:3].min(axis=1)
                    positive = spans[spans > 1e-9]
                    if positive.size:
                        return np.where(spans > 1e-9, spans, float(positive.min()))
        except Exception:
            return None
        return None

    @staticmethod
    def _topopt_industrial_grid_from_spans(spans):
        import numpy as np

        spans = np.asarray(spans, dtype=float)
        if spans.shape[0] < 3 or not np.all(np.isfinite(spans)):
            spans = np.asarray([80.0, 28.0, 8.0])
        positive = spans[spans > 1e-9]
        if positive.size == 0:
            spans = np.asarray([80.0, 28.0, 8.0])

        target_cells = 24000.0
        max_cells = 50000
        min_axis = 6
        max_axis = 160

        voxel = max(float(np.prod(spans[:3]) / target_cells) ** (1.0 / 3.0), 1e-9)
        dims = np.ceil(spans[:3] / voxel).astype(int)
        dims = np.maximum(dims, min_axis)
        if int(dims.max()) > max_axis:
            dims = np.maximum(
                np.floor(dims * (max_axis / float(dims.max()))).astype(int),
                min_axis,
            )
        while int(np.prod(dims)) > max_cells and int(dims.max()) > min_axis:
            scale = (max_cells / float(np.prod(dims))) ** (1.0 / 3.0) * 0.98
            dims = np.maximum(np.floor(dims * scale).astype(int), min_axis)
        return [int(v) for v in dims]

    def _apply_topopt_industrial_defaults(self, node):
        if getattr(node, "__identifier__", "") != "com.cad.sim.topopt_voxel":
            return

        spans = self._topopt_spans_from_value(self._topopt_cached_mesh_value(node))
        if spans is None:
            try:
                spans = [
                    max(1, int(node.get_property('nelx') or 80)),
                    max(1, int(node.get_property('nely') or 28)),
                    max(1, int(node.get_property('nelz') or 8)),
                ]
            except Exception:
                spans = [80, 28, 8]
        nelx, nely, nelz = self._topopt_industrial_grid_from_spans(spans)
        stress_enabled = str(node.get_property('stress_constraint') or '').lower() in {
            '1', 'true', 'yes', 'on',
        }
        goal = str(node.get_property('design_goal') or '').lower()
        stress_goal = 'stress' in goal
        settings = {
            'advanced_settings_visible': False,
            'nelx': nelx,
            'nely': nely,
            'nelz': nelz,
            'rmin': round(max(1.2, min(5.0, max(nelx, nely, nelz) * 0.030)), 2),
            'penal': 3.0,
            'density_cutoff': 0.45,
            'optimizer': 'MMA' if stress_enabled or stress_goal else 'OC',
            'max_iter': 100,
            'tol': 0.005,
            'convergence_patience': 5,
            'print_ready_mesh': False,
            'mesh_decimate_ratio': 1.0,
        }
        for key, value in settings.items():
            try:
                node.set_property(key, value)
            except Exception:
                pass
    
    def _upstream_closure(self, node):
        """Return *node* plus every node transitively feeding its inputs.

        This is the dependency subgraph needed to compute *node* — its
        upstream ancestors only, never downstream consumers or sibling
        branches.  Used to scope a "Run" to the selected node so that, e.g.,
        running a Mesh node does not also trigger a downstream Topology
        Optimization, and running an FEA Solver does not run a sibling TopOpt
        that merely shares the same geometry.
        """
        seen = set()
        order = []
        stack = [node]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            order.append(n)
            try:
                ports = n.input_ports()
                if isinstance(ports, dict):
                    ports = list(ports.values())
            except Exception:
                ports = []
            for port in ports:
                try:
                    conns = list(port.connected_ports())
                except Exception:
                    conns = []
                for cp in conns:
                    try:
                        up = cp.node()
                    except Exception:
                        continue
                    if id(up) not in seen:
                        stack.append(up)
        return order

    def _run_action(self):
        """Toolbar "Run".

        If exactly one node is selected, run only that node and its upstream
        dependency chain (so siblings/downstream — e.g. a Topology Opt that
        shares the geometry — are NOT executed).  With no single selection,
        run the whole graph as before.
        """
        try:
            selected = list(self.graph.selected_nodes())
        except Exception:
            selected = []
        if len(selected) == 1:
            target = selected[0]
            scoped = self._upstream_closure(target)
            self._last_rendered_node = target
            self._last_rendered_geom_id = None
            try:
                name = target.name() if callable(target.name) else target.name
            except Exception:
                name = "node"
            self.statusBar().showMessage(f"Running '{name}' and its inputs ({len(scoped)} nodes)...")
            self._execute_graph(nodes=scoped)
        else:
            self._execute_graph()

    def _execute_graph(self, skip_simulation=False, nodes=None):
        """Start graph execution in a background thread.

        Args:
            skip_simulation: If True, skip FEA/TopOpt nodes (for auto-update mode)
            nodes: Optional explicit node list to execute (a scoped subgraph,
                e.g. a selected node's upstream closure).  When None the whole
                graph runs.
        """
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Computation already in progress...")
            return

        # Keep UI responsive during optimization (don't disable)
        # self.graph.widget.setEnabled(False)  # Removed for real-time viz
        # self.toolbar.setEnabled(False)  # Removed for real-time viz

        if skip_simulation:
            self.statusBar().showMessage("Updating design preview...")
        else:
            self.statusBar().showMessage("Computing... (watch 3D viewer for live updates)")
            self._last_topopt_preview_payload = None
            self.timeline.add_event("Graph execution started (Full)")

        # Capture the list of nodes on the MAIN THREAD.  Do not rewrite TopOpt
        # solver settings here; saved studies and explicit user edits must run
        # as-authored. Defaults remain available from the TopOpt property panel.
        all_nodes_snapshot = list(nodes) if nodes is not None else list(self.graph.all_nodes())
        has_topopt_run = False
        for node in all_nodes_snapshot:
            if getattr(node, "__identifier__", "") == "com.cad.sim.topopt_voxel":
                has_topopt_run = True
                if not skip_simulation:
                    message = self._topopt_preflight_error(node)
                    if message:
                        try:
                            node.set_error(message)
                        except Exception:
                            pass
                        self.statusBar().showMessage(message)
                        self.timeline.add_event(message)
                        QtWidgets.QMessageBox.warning(self, "Topology Opt Setup", message)
                        return
        self._prefer_topopt_after_run = bool(has_topopt_run and not skip_simulation)
        
        # Initialize worker with skip_simulation parameter
        self.worker = GraphExecutionWorker(all_nodes_snapshot, skip_simulation=skip_simulation, parent=self)

        self.worker.computation_finished.connect(self._on_execution_finished)
        self.worker.computation_error.connect(self._on_execution_error)
        # Connect optimization step for real-time visualization
        self.worker.optimization_step.connect(self._on_optimization_step)
        self.worker.start()

    def _cancel_execution(self):
        worker = getattr(self, 'worker', None)
        if worker is None or not worker.isRunning():
            self.statusBar().showMessage("No computation is running")
            return
        worker.cancel()
        self.statusBar().showMessage("Stopping computation at the next safe iteration...")
        self.timeline.add_event("Computation stop requested")


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
            stage = mesh.get('stage') if isinstance(mesh, dict) else None
            vol_frac = None
            if isinstance(mesh, dict):
                try:
                    vol_frac = float(mesh.get('final_vol_frac'))
                except Exception:
                    vol_frac = None
            if vol_frac is None or not np.isfinite(vol_frac):
                vol_frac = float(np.mean(densities))
            is_final_step = (step + 1) >= total
            if not stage and not is_final_step and (now - self._last_preview_update_time) < 0.1:
                return

            self._last_preview_update_time = now
            if stage:
                self.statusBar().showMessage(f"TopOpt: {stage} (Vol: {vol_frac:.1%})")
            else:
                self.statusBar().showMessage(
                    f"TopOpt: Iteration {step+1}/{total} (Vol: {vol_frac:.1%})"
                )
            
            if isinstance(mesh, dict) and mesh.get('type') == 'topopt_voxel':
                result = dict(mesh)
                result['_preview'] = True
                self._last_topopt_preview_payload = result
                try:
                    for candidate in self.graph.all_nodes():
                        if getattr(candidate, '__identifier__', '') == 'com.cad.sim.topopt_voxel':
                            self._last_rendered_node = candidate
                            break
                except Exception:
                    pass
                self.viewer.render_simulation(result)
                return

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
