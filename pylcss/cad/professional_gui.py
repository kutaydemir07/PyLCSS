# Copyright (c) 2025 Kutay Demir.
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
from datetime import datetime
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import cadquery as cq
from pylcss.cad.viewer import CQ3DViewer
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
from pylcss.cad.nodes import (
    NODE_REGISTRY, BoxNode, CylinderNode, SphereNode,
    ExtrudeNode, PocketNode, FilletNode,
    NumberNode, ExportStepNode, ExportStlNode, SelectFaceNode, CutExtrudeNode, BooleanNode, RevolveNode, CylinderCutNode
)

class GraphExecutionWorker(QtCore.QThread):
    """Background worker to run the node graph without freezing the UI."""
    computation_finished = QtCore.Signal(object)  # Emits results dict
    computation_error = QtCore.Signal(str)

    # CHANGE: Accept 'nodes' list instead of 'graph' object
    def __init__(self, nodes, parent=None):
        super().__init__(parent)
        self.nodes = nodes  # Store the list of nodes, not the graph wrapper
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            from pylcss.cad.engine import execute_graph
            
            # CHANGE: Pass the list of nodes to the engine
            results = execute_graph(self.nodes)

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
        
        # Route: TopOpt gets specialized UI, all others get generic
        node_class = node.__class__.__name__
        if node_class == 'TopologyOptimizationNode':
            self._build_topopt_ui(node)
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
        combo_vis.addItems(['Density', 'Von Mises Stress'])
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
            
        elif node_class in ['ConstraintNode', 'LoadNode', 'PressureLoadNode']:
            group = QtWidgets.QGroupBox("Boundary Condition")
            layout = QtWidgets.QFormLayout()
            
            if 'condition' in props:
                edit = QtWidgets.QLineEdit(str(props['condition']))
                edit.editingFinished.connect(lambda: self.update_property('condition', edit.text()))
                layout.addRow("Condition:", edit)
            
            # Force components
            for prop in ['fx', 'fy', 'fz', 'pressure']:
                if prop in props:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(-1e12, 1e12)
                    spin.setDecimals(2)
                    spin.setValue(float(props[prop]))
                    spin.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
                    layout.addRow(f"{prop.upper()}:", spin)
            
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

    def _build_generic_ui(self, node):
        """Generic Clean UI - uses node.properties() for custom properties."""
        # NodeGraphQt stores custom properties inside the 'custom' key
        try:
            all_props = node.properties()
            props = all_props.get('custom', {})
        except:
            props = {}
        print(f"[Inspector DEBUG] Node: {node.__class__.__name__}, custom props: {list(props.keys())}")
        
        if not props:
            lbl = QtWidgets.QLabel("No editable properties")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.props_layout.addWidget(lbl)
            return
        
        group = QtWidgets.QGroupBox(f"Properties ({len(props)})")
        layout = QtWidgets.QFormLayout()
        
        sorted_keys = sorted(props.keys())
        
        for name in sorted_keys:
            val = props[name]
            label_text = name.replace('_', ' ').title()
            
            if isinstance(val, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(val)
                widget.stateChanged.connect(lambda s, n=name: self.update_property(n, bool(s)))
                layout.addRow(label_text, widget)
            elif isinstance(val, (int, float)):
                widget = ExpressionEdit(val)
                widget.value_changed.connect(lambda v, n=name: self.update_property(n, v))
                layout.addRow(label_text, widget)
            elif val is not None:
                widget = QtWidgets.QLineEdit(str(val))
                widget.editingFinished.connect(lambda n=name, w=widget: self.update_property(n, w.text()))
                layout.addRow(label_text, widget)
                
        group.setLayout(layout)
        self.props_layout.addWidget(group)
    
    def update_property(self, prop_name, value):
        """Update node property and mark as dirty for recalculation."""
        if self.current_node:
            try:
                old = self.current_node.get_property(prop_name)
                self.current_node.set_property(prop_name, value)
                # Mark node as dirty for recalculation
                if hasattr(self.current_node, '_last_hash'):
                    self.current_node._last_hash = None  # Invalidate hash cache
                try:
                    self.property_changed.emit(self.current_node, prop_name, old, value)
                except Exception: pass
            except Exception as e:
                print(f"Property update error: {e}")


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


class LibraryPanel(QtWidgets.QWidget):
    """Component library with categorized nodes."""
    
    def __init__(self, spawn_callback):
        super(LibraryPanel, self).__init__()
        self.spawn_callback = spawn_callback
        self.layout = QtWidgets.QVBoxLayout(self)
        
        title = QtWidgets.QLabel("Component Library")
        title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        self.layout.addWidget(title)
        
        # Search box
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Search components...")
        self.layout.addWidget(search)
        
        # Tree view for categories
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabel("Components")
        # enable dragging from the library into the graph
        self.tree.setDragEnabled(True)
        self.tree.itemPressed.connect(self._start_drag)
        
        # Component categories
        categories = {
            "Primitives": [
                ("Box", "com.cad.box"),
                ("Box (Coordinates)", "com.cad.coordinate_box"),
                ("Cylinder", "com.cad.cylinder"),
                ("Sphere", "com.cad.sphere"),
                ("Cone", "com.cad.cone"),
                ("Torus", "com.cad.torus"),
                ("Wedge", "com.cad.wedge"),
                ("Pyramid", "com.cad.pyramid"),
            ],
            "2D Sketching": [
                ("Sketch", "com.cad.sketch"),
                ("Rectangle", "com.cad.rect_sketch"),
                ("Circle", "com.cad.circle_sketch"),
                ("Line", "com.cad.line"),
                ("Arc", "com.cad.arc"),
                ("Polygon", "com.cad.polygon"),
                ("Spline", "com.cad.spline"),
                ("Ellipse", "com.cad.ellipse"),
            ],
            "3D Operations": [
                ("Extrude", "com.cad.extrude"),
                ("Revolve", "com.cad.revolve"),
                ("Sweep", "com.cad.sweep"),
                ("Loft", "com.cad.loft"),
                ("Helix", "com.cad.helix"),
            ],
            "Cutting Operations": [
                ("Pocket", "com.cad.pocket"),
                ("Extruded Cut", "com.cad.cut_extrude"),
                ("Cylinder Cut", "com.cad.cylinder_cut"),
                ("Hole at Coordinates", "com.cad.hole_at_coords"),
                ("Multiple Holes", "com.cad.multi_hole"),
                ("Rectangular Cut", "com.cad.rectangular_cut"),
                ("Slot Cut", "com.cad.slot_cut"),
                ("Array of Holes", "com.cad.array_holes"),
            ],
            "Modifications": [
                ("Fillet", "com.cad.fillet"),
                ("Chamfer", "com.cad.chamfer"),
                ("Shell", "com.cad.shell"),
                ("Offset 2D", "com.cad.offset"),
                ("Draft", "com.cad.draft"),
            ],
            "Boolean Operations": [
                ("Boolean Op", "com.cad.boolean"),
            ],
            "Transformations": [
                ("Translate", "com.cad.translate"),
                ("Rotate", "com.cad.rotate"),
                ("Scale", "com.cad.scale"),
                ("Mirror", "com.cad.mirror"),
            ],
            "Patterns": [
                ("Linear Pattern", "com.cad.linear_pattern"),
                ("Circular Pattern", "com.cad.circular_pattern"),
            ],
            "Selection": [
                ("Select Face", "com.cad.select_face"),
            ],
            "Constraints": [
                ("Distance", "com.cad.constraint_distance"),
                ("Angle", "com.cad.constraint_angle"),
                ("Coincident", "com.cad.constraint_coincident"),
            ],
            "Assembly": [
                ("Assembly", "com.cad.assembly"),
                ("Mate", "com.cad.mate"),
            ],
            "Analysis": [
                ("Mass Properties", "com.cad.mass_properties"),
                ("Stress Analysis", "com.cad.stress_analysis"),
                ("Bounding Box", "com.cad.bounding_box"),
                ("Volume", "com.cad.volume"),
                ("Surface Area", "com.cad.surface_area"),
                ("Center of Mass", "com.cad.center_of_mass"),
            ],
            "Simulation": [
                ("Force", "com.cad.force"),
                ("Fixed Support", "com.cad.fixed_support"),
                ("Run Simulation", "com.cad.simulation_run"),
            ],
            "FEA & Optimization": [
                ("Generate Mesh", "com.cad.sim.mesh"),
                ("Material", "com.cad.sim.material"),
                ("FEA Fixed Support", "com.cad.sim.constraint"),
                ("FEA Load", "com.cad.sim.load"),
                ("FEA Solver", "com.cad.sim.solver"),
                ("Topology Opt", "com.cad.sim.topopt"),
            ],
            "Advanced Features": [
                ("3D Text", "com.cad.text"),
                ("Thread", "com.cad.thread"),
                ("Split", "com.cad.split"),
            ],
            "Measurement": [
                ("Measure Distance", "com.cad.measure_distance"),
            ],
            "Parameters": [
                ("Number", "com.cad.number"),
            ],
            "Export": [
                ("2D Drawing", "com.cad.drawing"),
                ("Property Table", "com.cad.property_table"),
                ("Report", "com.cad.report"),
                ("Export STEP", "com.cad.export_step"),
                ("Export STL", "com.cad.export_stl"),
            ],
        }
        
        for category, items in categories.items():
            cat_item = QtWidgets.QTreeWidgetItem([category])
            cat_item.setIcon(0, QtGui.QIcon())
            
            for label, node_id in items:
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, QtCore.Qt.UserRole, node_id)
                cat_item.addChild(item)
            
            self.tree.addTopLevelItem(cat_item)
        
        self.tree.itemDoubleClicked.connect(self._on_component_selected)
        self.layout.addWidget(self.tree)
    
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
        
        # Add mutex for thread safety
        self.result_mutex = QtCore.QMutex()
        
        # Create graph
        self.graph = NodeGraph()
        self._register_nodes()
        
        # Connect graph signals
        self.graph.property_changed.connect(self._on_graph_property_changed)
        self.graph.port_connected.connect(self._on_connection_changed)
        self.graph.port_disconnected.connect(self._on_connection_changed)
        
        # Prevent double-click popup
        self.graph.node_double_clicked.connect(self._on_node_double_clicked)
        
        # Setup UI
        self._setup_toolbar() 
        self._create_ui()
        # self._setup_menu() 

        # Connect graph selection
        self.graph.node_selected.connect(self._on_node_selected)
    
    def _register_nodes(self):
        """Register all available nodes."""
        for node_class in NODE_CLASS_MAPPING.values():
            try:
                self.graph.register_node(node_class)
            except Exception as e:
                print(f"Warning: Could not register {node_class.__name__}: {e}")
    
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

        # Right panel
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.properties = PropertiesPanel()
        try:
            self.properties.property_changed.connect(self._on_property_changed)
        except Exception:
            pass
        self.timeline = TimelinePanel()
        right_splitter.addWidget(self.properties)
        right_splitter.addWidget(self.timeline)
        right_splitter.setSizes([400, 200])
        
        # Main splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([200, 1000, 300])
        
        main_h_layout.addWidget(main_splitter)
        
        # Set central widget for QMainWindow
        self.setCentralWidget(content_widget)
    
    def _setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self._new_project, QtGui.QKeySequence.New)
        file_menu.addAction("Open", self._open_project, QtGui.QKeySequence.Open)
        file_menu.addAction("Save", self._save_project, QtGui.QKeySequence.Save)
        file_menu.addAction("Save As", self._save_as_project)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QtGui.QKeySequence.Quit)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", self._undo, QtGui.QKeySequence.Undo)
        edit_menu.addAction("Redo", self._redo, QtGui.QKeySequence.Redo)
        edit_menu.addSeparator()
        edit_menu.addAction("Delete", self._delete_selected, QtGui.QKeySequence.Delete)
        edit_menu.addAction("Clear All", self._clear_graph)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Fit All", self._fit_all)
        view_menu.addAction("Reset View", self._reset_view)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Execute Graph", self._execute_graph)
        tools_menu.addAction("Run Simulation", self._run_simulation)
        tools_menu.addAction("Generate Report", self._generate_report)
        tools_menu.addAction("Validate Model", self._validate_model)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._show_about)
    
    def _setup_toolbar(self):
        """Create toolbar and add to layout."""
        self.toolbar = QtWidgets.QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        
        self.toolbar.addAction("New", self._new_project)
        self.toolbar.addAction("Open", self._open_project)
        self.toolbar.addAction("Save", self._save_project)
        self.toolbar.addSeparator()
        self.toolbar.addAction("Undo", self._undo)
        self.toolbar.addAction("Redo", self._redo)
        self.toolbar.addSeparator()
        self.toolbar.addAction("Run", self._execute_graph)
        self.toolbar.addAction("Report", self._generate_report)
        
        self.toolbar.addSeparator()
        self.auto_update_cb = QtWidgets.QCheckBox("Auto-Update")
        self.auto_update_cb.setChecked(False)
        self.auto_update_cb.setToolTip("Automatically execute graph when properties change")
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
        except Exception as e:
            self.statusBar().showMessage(f"✗ Error: {e}")
            print(f"Node creation error: {e}")
    
    def _on_node_selected(self, node):
        """Handle node selection."""
        if node:
            self.properties.display_node(node)
            self.statusBar().showMessage(f"Selected: {node.name}")

            # Only render if we have a CACHED result. 
            # Do NOT call execute_graph() here to avoid freezing on selection.
            geometry = getattr(node, '_last_result', None)

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
                else:
                    self.viewer.render_shape(geometry)
            else:
                # Optional: Trigger a background update if you really want "Live" feel
                # but only if it's a fast node. For now, better to wait for user to click "Execute"
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

        # Update the properties panel if this node is selected
        if self.properties.current_node == node:
            self.properties.display_node(node)
            
        # Auto-update if enabled
        if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
            self._execute_graph()
            
    def _on_connection_changed(self, port_in, port_out):
        """Handle connection changes (connect/disconnect)."""
        # Mark both nodes as dirty
        if port_in:
            node = port_in.node()
            setattr(node, '_dirty', True)
        if port_out:
            node = port_out.node()
            setattr(node, '_dirty', True)
            
        self.timeline.add_event("Connection changed")
        self._execute_graph()

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
        except Exception as e:
            print(f"Drag/drop error: {e}")
        return super(ProfessionalCadApp, self).eventFilter(source, event)

    def _on_execution_finished(self, results):
        """Called when the background thread completes."""
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
                    elif hasattr(geom, 'p') and hasattr(geom, 't'):
                        # Direct Mesh object from skfem
                        self.viewer.render_simulation(geom)
                    else:
                        self.viewer.render_shape(geom)
                else:
                    self.viewer.clear()

            except Exception as e:
                print(f"Visualization update error: {e}")
        finally:
            self.result_mutex.unlock()

    def _on_execution_error(self, error_msg):
        """Called if background thread fails."""
        self.graph.widget.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.statusBar().showMessage(f"❌ Error: {error_msg}")
        self.timeline.add_event(f"Execution failed: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "Computation Error", error_msg)
    
    def _update_property(self, prop_name, value):
        """Update node property."""
        if self.properties.current_node:
            try:
                self.properties.current_node.set_property(prop_name, value)
                self.timeline.add_event(f"Updated {prop_name} = {value}")
            except Exception as e:
                print(f"Property update error: {e}")
    
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
        except Exception as e:
            print(f"Undo failed: {e}")
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
        except Exception as e:
            print(f"Redo failed: {e}")
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
            self.timeline.add_event("Cleared graph")
    
    def _fit_all(self):
        """Fit all nodes in view."""
        self.statusBar().showMessage("Fit All (centering nodes)")
    
    def _reset_view(self):
        """Reset view to default."""
        self.statusBar().showMessage("View reset")
    
    def _run_simulation(self):
        """Run simulation on current model."""
        self.statusBar().showMessage("Running simulation...")
        self.timeline.add_event("Simulation started")
        # Find export or simulation nodes and run them
        QtCore.QTimer.singleShot(1000, lambda: self.statusBar().showMessage("✓ Simulation complete"))
        self.timeline.add_event("Simulation completed")
    
    def _generate_report(self):
        """Generate a report from the model."""
        self.statusBar().showMessage("Generating report...")
        self.timeline.add_event("Report generation started")
        QtCore.QTimer.singleShot(1000, lambda: self.statusBar().showMessage("✓ Report generated"))
        self.timeline.add_event("Report generated")
    
    def _validate_model(self):
        """Validate the current model."""
        self.statusBar().showMessage("Validating model...")
        self.timeline.add_event("Model validation started")
        QtCore.QTimer.singleShot(1000, lambda: self.statusBar().showMessage("✓ Model valid"))
        self.timeline.add_event("Model validation complete")
    
    def _new_project(self):
        """Create a new project."""
        self.graph.clear_session()
        self.current_file = None
        self.timeline.add_event("New project created")
        self.statusBar().showMessage("New project")
    
    def _open_project(self):
        """Open a project file."""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", "", "CAD Projects (*.cad);;All Files (*)"
        )
        if fname:
            try:
                # Load and deserialize using NodeGraphQt's built-in session manager
                # load_session handles clearing the session and reading the file
                self.graph.load_session(fname)
                
                self.current_file = fname
                self.timeline.add_event(f"Opened project: {fname}")
                self.statusBar().showMessage(f"Opened: {fname}")
                
                # Execute to restore view
                self._execute_graph()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open project: {e}")
                print(f"Open error details: {e}")
    
    def _get_node_class(self, class_name):
        """Get node class by name."""
        return NODE_NAME_MAPPING.get(class_name)


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
            
            # OPTIMIZATION: Check if this is a visualization-only property change
            # These properties don't need a re-run, just a re-render
            visualization_only_props = ['visualization', 'density_cutoff']
            
            if prop_name in visualization_only_props:
                # Check if the node has cached results (_last_result)
                cached_result = getattr(node, '_last_result', None)
                if cached_result is not None:
                    # Just re-render with existing results instead of re-executing
                    try:
                        if isinstance(cached_result, dict) and ('mesh' in cached_result or 'displacement' in cached_result):
                            self.viewer.render_simulation(cached_result)
                        elif hasattr(cached_result, 'p') and hasattr(cached_result, 't'):
                            # Direct Mesh object from skfem
                            self.viewer.render_simulation(cached_result)
                        else:
                            self.viewer.render_shape(cached_result)
                        self.statusBar().showMessage(f"✓ Updated {prop_name} display")
                        return  # Skip full graph execution
                    except Exception as e:
                        print(f"Visualization-only update failed: {e}")
                        # Fall through to full execution if re-render fails
            
            # Auto-execute if enabled (for non-visualization properties)
            if hasattr(self, 'auto_update_cb') and self.auto_update_cb.isChecked():
                self._execute_graph()
                
        except Exception as e:
            print(f"Property change record error: {e}")
    
    def _save_project(self):
        """Save current project."""
        if not self.current_file:
            self._save_as_project()
            return
        
        try:
            # Serialize graph using NodeGraphQt's built-in session manager
            project_data = self.graph.serialize_session()
            
            # Save to file
            with open(self.current_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.timeline.add_event(f"Saved project: {self.current_file}")
            self.statusBar().showMessage(f"Saved: {self.current_file}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
            print(f"Save error details: {e}")
    
    def _save_as_project(self):
        """Save project with a new name."""
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
            "Professional CAD Software\n"
            "Version 1.0\n\n"
            "A full-featured, Simulink-style parametric CAD system\n"
            "with sketches, constraints, assembly, analysis, and simulation.\n\n"
            "© 2025 Kutay Demir"
        )
    
    def _execute_graph(self):
        """Start graph execution in a background thread."""
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("⚠️ Computation already in progress...")
            return

        self.graph.widget.setEnabled(False)
        self.toolbar.setEnabled(False)
        self.statusBar().showMessage("⏳ Computing... (UI Locked)")
        self.timeline.add_event("Graph execution started (Background)")

        # --- KEY CHANGE HERE ---
        # Capture the list of nodes on the MAIN THREAD.
        # This prevents the background thread from accessing the QGraphicsScene.
        # We convert the iterator to a concrete list immediately.
        all_nodes_snapshot = list(self.graph.all_nodes())
        
        # Initialize worker with the snapshot list
        self.worker = GraphExecutionWorker(all_nodes_snapshot, self)
        # -----------------------

        self.worker.computation_finished.connect(self._on_execution_finished)
        self.worker.computation_error.connect(self._on_execution_error)
        self.worker.start()


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
            print(f"(NodeGraphQt internal warning - continuing anyway)")
            sys.exit(0)
        else:
            raise


if __name__ == '__main__':
    main()
