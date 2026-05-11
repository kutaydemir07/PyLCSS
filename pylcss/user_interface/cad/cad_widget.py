# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Professional CAD Software - Full-Featured Simulink-like Interface.

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
        elif node_class == 'InteractiveSelectFaceNode':
            self._build_interactive_select_ui(node)
        elif node_class in ('ConstraintNode', 'LoadNode', 'PressureLoadNode'):
            self._build_fea_bc_ui(node)
        else:
            self._build_generic_ui(node)
            
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
        
        # Element Type
        combo_elem = QtWidgets.QComboBox()
        combo_elem.addItems(['Fast (Linear P1)', 'Accurate (Quadratic P2)'])
        combo_elem.setCurrentText(str(node.get_property('element_type') or 'Fast (Linear P1)'))
        combo_elem.currentTextChanged.connect(lambda v: self.update_property('element_type', v))
        layout_adv.addRow("Element Type:", combo_elem)

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

    def _build_primitive_ui(self, node):
        """UI for Primitive nodes (Box, Cylinder, Sphere, etc.)."""
        group = QtWidgets.QGroupBox("Dimensions")
        layout = QtWidgets.QFormLayout()
        
        # Actual property names from node implementations
        dim_props = [
            'box_length', 'box_width', 'box_depth',  # BoxNode
            'cyl_radius', 'cyl_height',  # CylinderNode
            'sphere_radius',  # SphereNode
            'bottom_radius', 'top_radius', 'cone_height',  # ConeNode
            'major_radius', 'minor_radius',  # TorusNode
        ]
        props = node.model.properties
        
        for prop in dim_props:
            if prop in props:
                val = props[prop]
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.001, 10000)
                spin.setDecimals(3)
                spin.setValue(float(val) if val else 1.0)
                spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                label = prop.replace('_', ' ').replace('box ', '').replace('cyl ', '').replace('sphere ', '').title()
                layout.addRow(f"{label}:", spin)
        
        group.setLayout(layout)
        self.props_layout.addWidget(group)
        
        # Position group
        group_pos = QtWidgets.QGroupBox("Position")
        layout_pos = QtWidgets.QFormLayout()
        
        for axis in ['center_x', 'center_y', 'center_z']:
            if axis in props:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-10000, 10000)
                spin.setValue(float(props[axis]) if props[axis] else 0.0)
                spin.valueChanged.connect(lambda v, p=axis: self.update_property(p, v))
                layout_pos.addRow(f"{axis.replace('center_', '').upper()}:", spin)
        
        if layout_pos.rowCount() > 0:
            group_pos.setLayout(layout_pos)
            self.props_layout.addWidget(group_pos)
    
    def _build_simulation_ui(self, node):
        """UI for Simulation nodes (Material, Mesh, Solver, etc.)."""
        node_class = node.__class__.__name__
        props = node.model.properties
        
        if node_class == 'MaterialNode':
            group = QtWidgets.QGroupBox("Material Properties")
            layout = QtWidgets.QFormLayout()
            
            for prop in ['youngs_modulus', 'poissons_ratio', 'density']:
                if prop in props:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setDecimals(6)
                    if prop == 'youngs_modulus':
                        spin.setRange(1e3, 1e12)
                        spin.setSingleStep(1e9)
                    elif prop == 'poissons_ratio':
                        spin.setRange(0.0, 0.5)
                        spin.setSingleStep(0.01)
                    else:
                        spin.setRange(0.1, 100000)
                    spin.setValue(float(props[prop]))
                    spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                    layout.addRow(f"{prop.replace('_', ' ').title()}:", spin)
            
            group.setLayout(layout)
            self.props_layout.addWidget(group)
            
        elif node_class == 'MeshNode':
            group = QtWidgets.QGroupBox("Mesh Settings")
            layout = QtWidgets.QFormLayout()
            
            if 'mesh_size' in props:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.01, 100)
                spin.setDecimals(2)
                spin.setValue(float(props['mesh_size']))
                spin.valueChanged.connect(lambda v: self.update_property('mesh_size', v))
                layout.addRow("Mesh Size:", spin)
            
            if 'order' in props:
                spin_ord = QtWidgets.QSpinBox()
                spin_ord.setRange(1, 2)
                spin_ord.setValue(int(props['order']))
                spin_ord.valueChanged.connect(lambda v: self.update_property('order', v))
                layout.addRow("Element Order:", spin_ord)
            
            group.setLayout(layout)
            self.props_layout.addWidget(group)
            
        elif node_class == 'SolverNode':
            group = QtWidgets.QGroupBox("Solver Settings")
            layout = QtWidgets.QFormLayout()
            
            if 'visualization' in props:
                combo = QtWidgets.QComboBox()
                combo.addItems(['Displacement', 'Von Mises Stress'])
                combo.setCurrentText(str(props['visualization']))
                combo.currentTextChanged.connect(lambda v: self.update_property('visualization', v))
                layout.addRow("Display:", combo)
            
            group.setLayout(layout)
            self.props_layout.addWidget(group)
            
        else:
            self._build_generic_ui(node)
    
    def _build_operation_ui(self, node):
        """UI for Operation nodes (Extrude, Revolve, etc.)."""
        group = QtWidgets.QGroupBox("Operation Parameters")
        layout = QtWidgets.QFormLayout()
        props = node.model.properties
        
        for prop in ['extrude_distance', 'pocket_depth', 'angle', 'direction']:
            if prop in props:
                val = props[prop]
                if isinstance(val, (int, float)):
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(-10000, 10000)
                    spin.setDecimals(2)
                    spin.setValue(float(val))
                    spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                    layout.addRow(f"{prop.replace('_', ' ').title()}:", spin)
                else:
                    edit = QtWidgets.QLineEdit(str(val))
                    edit.editingFinished.connect(lambda p=prop: self.update_property(p, edit.text()))
                    layout.addRow(f"{prop.replace('_', ' ').title()}:", edit)
        
        group.setLayout(layout)
        self.props_layout.addWidget(group)
    
    def _build_modification_ui(self, node):
        """UI for Modification nodes (Fillet, Chamfer, Shell)."""
        group = QtWidgets.QGroupBox("Modification")
        layout = QtWidgets.QFormLayout()
        props = node.model.properties
        
        for prop in ['radius', 'fillet_radius', 'chamfer_distance', 'thickness', 'offset']:
            if prop in props:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.001, 1000)
                spin.setDecimals(3)
                spin.setValue(float(props[prop]))
                spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                layout.addRow(f"{prop.replace('_', ' ').title()}:", spin)
        
        group.setLayout(layout)
        self.props_layout.addWidget(group)
    
    def _build_transform_ui(self, node):
        """UI for Transform nodes (Translate, Rotate, Scale, Mirror)."""
        group = QtWidgets.QGroupBox("Transform")
        layout = QtWidgets.QFormLayout()
        props = node.model.properties
        
        # Translation / Rotation / Scale components
        for prop in ['x', 'y', 'z', 'dx', 'dy', 'dz', 'angle', 'scale_x', 'scale_y', 'scale_z', 'uniform_scale']:
            if prop in props:
                val = props[prop]
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-10000, 10000)
                spin.setDecimals(3)
                spin.setValue(float(val) if val else 0.0)
                spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                layout.addRow(f"{prop.replace('_', ' ').title()}:", spin)
        
        group.setLayout(layout)
        self.props_layout.addWidget(group)

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
        ("Geometry",        ("box_", "length", "width", "depth", "height", "radius", "thickness",
                             "near_", "selector_type", "tag", "range_expr", "direction")),
        ("Load",            ("load_type", "force_", "vector", "magnitude", "pressure", "gravity_",
                             "accel", "velocity_", "node_tolerance")),
        ("Constraint",      ("constraint_type", "fixed_dofs", "displacement_")),
    ]
    # Hide these unless they have a meaningful value (truthy non-empty).
    _PROPERTY_HIDE_IF_EMPTY = ("condition", "range_expr", "tag", "external_solver_path",
                               "external_work_dir", "openradioss_starter_path",
                               "openradioss_engine_path")

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

        # Hard-coded fallback combo items for legacy nodes that don't declare
        # their items via NodeGraphQt's metadata (kept as a safety net).
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
            'visualization': ['Density', 'Von Mises Stress', 'Displacement'],
            'filter_type': ['sensitivity', 'density'],
            'update_scheme': ['MMA', 'OC'],
            'projection': ['None', 'Heaviside'],
            'solver_backend': ['Internal scikit-fem', 'CalculiX',
                               'Internal PyLCSS', 'OpenRadioss (experimental)'],
        }

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
                label_text = name.replace('_', ' ').title()
                attrs = prop_attrs.get(name, {})
                combo_items = attrs.get('items') if isinstance(attrs, dict) else None
                if not combo_items and name in known_combos:
                    combo_items = known_combos[name]
                value_range = attrs.get('range') if isinstance(attrs, dict) else None
                tooltip = attrs.get('tooltip') if isinstance(attrs, dict) else None

                if combo_items:
                    widget = QtWidgets.QComboBox()
                    widget.addItems([str(item) for item in combo_items])
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
                "Highlight this load / support face in the 3D viewer"
            )
            btn_preview.setStyleSheet(
                "QPushButton {"
                "  background: #1e5aab; color: white; border-radius: 4px;"
                "  padding: 5px 10px; font-weight: bold; font-size: 12px;"
                "  margin-top: 6px;"
                "}"
                "QPushButton:hover { background: #2673cc; }"
            )
            # Use a local default so the closure captures the right node.
            def _on_preview(checked=False, _node=node):
                app = self._get_main_app()
                if app:
                    # Render the upstream shape first, then overlay BCs.
                    shape = app._get_upstream_shape(_node)
                    if shape is None:
                        # Maybe the node itself has a cached shape in a parent
                        shape = getattr(_node, '_last_result', None)
                    if shape and hasattr(shape, 'tessellate'):
                        app.viewer.render_shape(shape)
                    elif shape and isinstance(shape, dict) and 'mesh' in shape:
                        app.viewer.render_simulation(shape)
                    try:
                        app._show_bc_for_node(_node)
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

        backend = data.get('backend') or ('Internal scikit-fem' if rtype == 'fea' else
                                          'Internal PyLCSS' if rtype == 'crash' else '—')
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

        # Warnings from the external backends
        warnings = data.get('warnings') or []
        if warnings:
            self._add_warnings(list(warnings))


class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""

    # QtAwesome icon names per category prefix.  The first prefix that matches
    # wins, so order from most specific to most general.
    _CATEGORY_ICONS = (
        ("Sketcher",              "fa5s.pencil-ruler",   "#4FC3F7"),
        ("Part Design - Primitives",     "fa5s.cube",            "#FFCA28"),
        ("Part Design - Create",         "fa5s.industry",        "#FFCA28"),
        ("Part Design - Modify",         "fa5s.tools",           "#FFCA28"),
        ("Part Design - Hole Wizard",    "fa5s.dot-circle",      "#FFCA28"),
        ("Part Design - Transform",      "fa5s.arrows-alt",      "#FFCA28"),
        ("Simulation - Pre-Processing",  "fa5s.project-diagram", "#80CBC4"),
        ("Simulation - Loads",           "fa5s.weight-hanging",  "#FF8A65"),
        ("Simulation - Solve",           "fa5s.calculator",      "#9CCC65"),
        ("Crash Simulation",      "fa5s.car-crash",       "#EF5350"),
        ("Analysis",              "fa5s.balance-scale",   "#B39DDB"),
        ("IO",                    "fa5s.file-export",     "#90A4AE"),
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
        
        # Component categories - Professional CAD/FEA/TopOpt Library
        # Format: (Label, node_id, tooltip_description)
        categories = {
            # ═══════════════════════════════════════════════════════════════
            # WORKBENCH: SKETCHER
            # ═══════════════════════════════════════════════════════════════
            "Sketcher": [
                ("Create Sketch", "com.cad.sketch", "Start a new 2D sketch on a plane"),
                ("Line", "com.cad.sketch.line", "Draw a straight line segment"),
                ("Rectangle", "com.cad.sketch.rectangle", "Draw a rectangle"),
                ("Circle", "com.cad.sketch.circle", "Draw a circle"),
                ("Arc", "com.cad.sketch.arc", "Draw an arc"),
                ("Polygon", "com.cad.sketch.polygon", "Draw a regular polygon"),
                ("Ellipse", "com.cad.ellipse", "Draw an ellipse"),
                ("Spline", "com.cad.spline", "Draw a smooth B-spline curve"),
            ],
            
            # ═══════════════════════════════════════════════════════════════
            # WORKBENCH: PART DESIGN
            # ═══════════════════════════════════════════════════════════════
            "Part Design - Primitives": [
                ("Box", "com.cad.box", "Create a rectangular block"),
                ("Cylinder", "com.cad.cylinder", "Create a cylinder"),
                ("Sphere", "com.cad.sphere", "Create a sphere"),
                ("Cone", "com.cad.cone", "Create a cone"),
                ("Torus", "com.cad.torus", "Create a torus (donut)"),
                ("Wedge", "com.cad.wedge", "Create a wedge"),
                ("Pyramid", "com.cad.pyramid", "Create a pyramid"),
            ],
            "Part Design - Create": [
                ("Extrude", "com.cad.extrude", "Extrude 2D sketch to 3D"),
                ("Twisted Extrude", "com.cad.twisted_extrude", "Extrude with helical twist"),
                ("Revolve", "com.cad.revolve", "Revolve sketch around axis"),
                ("Sweep", "com.cad.sweep", "Sweep profile along path"),
                ("Loft", "com.cad.loft", "Loft between profiles"),
                ("Helix", "com.cad.helix", "Create a helix/spiral"),
            ],
            "Part Design - Modify": [
                ("Boolean", "com.cad.boolean", "Union, Cut, or Intersect shapes"),
                ("Fillet", "com.cad.fillet", "Round edges"),
                ("Chamfer", "com.cad.chamfer", "Bevel edges"),
                ("Shell", "com.cad.shell", "Hollow out part"),
                ("Offset", "com.cad.offset", "Offset 3D shape"),
            ],
            "Part Design - Hole Wizard": [
                ("Hole (Point)", "com.cad.hole_at_coords", "Drill hole at coordinates"),
                ("Multi-Hole", "com.cad.multi_hole", "Drill multiple holes"),
                ("Pocket", "com.cad.pocket", "Extrude cut"),
                ("Slot", "com.cad.slot_cut", "Cut a slot"),
                ("Rectangular Cut", "com.cad.rectangular_cut", "Square cut"),
                ("Cylinder Cut", "com.cad.cylinder_cut", "Cylindrical cut"),
                ("Array Holes", "com.cad.array_holes", "Pattern of holes"),
            ],
            "Part Design - Transform & Pattern": [
                ("Translate", "com.cad.translate", "Move object"),
                ("Rotate", "com.cad.rotate", "Rotate object"),
                ("Scale", "com.cad.scale", "Resize object"),
                ("Mirror", "com.cad.mirror", "Mirror object"),
                ("Linear Pattern", "com.cad.linear_pattern", "Linear repetition"),
                ("Circular Pattern", "com.cad.circular_pattern", "Circular repetition"),
                ("Radial Pattern", "com.cad.pattern.radial", "Radial repetition"),
                ("Mirror Pattern", "com.cad.pattern.mirror", "Mirror pattern"),
                ("Grid Pattern", "com.cad.pattern.grid", "Grid repetition"),
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
                ("Crash Solver (CPU)", "com.cad.sim.crash_solver",
                 "Explicit central-difference transient solver with J2 plasticity, element deletion and frame-by-frame playback"),
                ("Crash Solver (GPU)",  "com.cad.sim.crash_solver_gpu",
                 "GPU-accelerated explicit crash solver (CUDA); falls back to CPU if no CUDA device is found)"),
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
                ("Number", "com.cad.number", "Numeric constant"),
                ("Variable", "com.cad.variable", "Named variable"),
                ("Import STEP", "com.cad.import_step", "Import a STEP/IGES CAD file"),
                ("Import Mesh", "com.cad.import_stl", "Import an STL/OBJ mesh file"),
                ("Export STEP", "com.cad.export_step", "Export to .step"),
                ("Export STL", "com.cad.export_stl", "Export to .stl"),
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
    """Main Application Window (formerly CADWidget)."""
    
    def __init__(self, parent=None):
        super(ProfessionalCadApp, self).__init__(parent)
        
        self.setWindowTitle("Professional CAD")
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
        act_import = self.toolbar.addAction(_icon("fa5s.file-import"), "Import CAD", self._import_cad)
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
        """Prompt user for a CAD file, add an import node, and set its filepath."""
        try:
            from pylcss.io_manager.cad_io import CADImporter
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import CAD File", "", CADImporter.get_filter_string()
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
            self.statusBar().showMessage(f"Error importing CAD: {e}") 
    
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


            # Only render if we have a CACHED result.
            # Do NOT call execute_graph() here to avoid freezing on selection.
            geometry = getattr(node, '_last_result', None)

            # If the result is a face-selection dict (from SelectFaceNode /
            # InteractiveSelectFaceNode), render the upstream shape instead so
            # the user can see the full 3D model and pick faces on it.
            if isinstance(geometry, dict) and 'faces' in geometry and 'mesh' not in geometry:
                geometry = self._get_upstream_shape(node)

            if geometry:
                # Check if it's simulation data or mesh object
                is_sim = False
                if isinstance(geometry, dict) and 'mesh' in geometry:
                    is_sim = True
                elif hasattr(geometry, 'p') and hasattr(geometry, 't'):
                    # Direct Mesh object from skfem
                    is_sim = True

                if is_sim:
                    self.viewer.render_simulation(geometry)
                    try:
                        self.results.show_result(geometry)
                    except Exception:
                        pass
                elif self._is_2d_sketch(geometry):
                    self.viewer.render_sketch(geometry)
                else:
                    self.viewer.render_shape(geometry)

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
                if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
                    self._execute_graph(skip_simulation=True)

    def _get_upstream_shape(self, node):
        """Walk input ports to find the first cached shape result upstream."""
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

        for node in self.graph.all_nodes():
            cls = node.__class__.__name__

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
        """
        Handle double-click.
        Intentionally left empty (or just log) to consume the event and prevent
        any default popup/dialog from appearing.
        """
        self.timeline.add_event(f"Double-clicked {node.name} (Popup disabled)")


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

                    if isinstance(geom, dict) and ('mesh' in geom or 'displacement' in geom):
                        self.viewer.render_simulation(geom)
                        try:
                            self.results.show_result(geom)
                        except Exception:
                            pass
                    elif hasattr(geom, 'p') and hasattr(geom, 't'):
                        # Direct Mesh object from skfem
                        self.viewer.render_simulation(geom)
                    elif self._is_2d_sketch(geom):
                        # 2D sketch - render as polylines
                        self.viewer.render_sketch(geom)
                    else:
                        self.viewer.render_shape(geom)

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
            if node_class in ['SolverNode', 'TopologyOptimizationNode', 'MeshNode', 'CrashSolverNode', 'CrashSolverGPUNode']:
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
            self, "Open Project", "", "CAD Projects (*.cad);;All Files (*)"
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
            self, "Save Project As", "", "CAD Projects (*.cad);;All Files (*)"
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
            self.statusBar().showMessage("⏳ Updating CAD preview...")
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
                'density_cutoff': 0.3  # Use default cutoff for preview
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
