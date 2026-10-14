# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""StudyInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging
import math
import numpy as np


from PySide6 import QtCore, QtGui, QtWidgets


from .code_editor import CadCodeEditorDialog
from .crash_plot_dialog import CrashPlotDialog
from .inspector_controls import ExpressionEdit

__all__ = ["StudyInspectorMixin"]


class StudyInspectorMixin:
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
        btn_edit.clicked.connect(
            lambda _checked=False, n=node: self._open_cad_code_editor(n)
        )
        code_layout.addWidget(btn_edit)
        btn_preview = QtWidgets.QPushButton("Preview in 3D")
        btn_preview.setToolTip(
            "Run the graph (CAD only — skips FEA/impact) and render this part."
        )
        btn_preview.clicked.connect(
            lambda _checked=False, n=node: self._preview_cad_part(n)
        )
        code_layout.addWidget(btn_preview)
        code_group.setLayout(code_layout)
        self.props_layout.addWidget(code_group)

        # ── Scalable Parameters ──────────────────────────────────────
        self._build_scalable_parameter_group(
            node,
            title="Parameters",
            tooltip=(
                "Named scalars that flow into the CadQuery script as top-level variables.\n"
                "Adjust 'Ports' to add or remove parameter sockets."
            ),
        )

        # ── Extra parameters (free-form dict / k=v lines) ────────────
        extra_group = QtWidgets.QGroupBox("Extra Parameters")
        extra_group.setToolTip(
            "Free-form parameters in 'name=value' lines (one per line) or a\n"
            "Python dict.  These override numbered slots if they collide."
        )
        extra_layout = QtWidgets.QVBoxLayout()
        extra_editor = QtWidgets.QPlainTextEdit(
            str(node.get_property("parameters") or "")
        )
        extra_editor.setPlaceholderText("name=value lines or a Python dict")
        mono = QtGui.QFont("Consolas")
        mono.setStyleHint(QtGui.QFont.Monospace)
        extra_editor.setFont(mono)
        extra_editor.setMinimumHeight(90)
        extra_editor.focusOutEvent = (
            lambda ev, w=extra_editor, _orig=extra_editor.focusOutEvent: (
                self.update_property("parameters", w.toPlainText()),
                _orig(ev),
            )[-1]
        )
        extra_layout.addWidget(extra_editor)
        extra_group.setLayout(extra_layout)
        self.props_layout.addWidget(extra_group)

    def _build_scalable_parameter_group(
        self,
        node,
        title: str = "Parameters",
        tooltip: str = (
            "Named parameter scalars. Each maps to an input socket on the node.\n"
            "Use the controls above to change the number of parameter ports."
        ),
    ) -> None:
        """Render a scalable parameter slots group with dynamic port count controls."""
        param_group = QtWidgets.QGroupBox(title)
        param_group.setToolTip(tooltip)
        param_vlayout = QtWidgets.QVBoxLayout()

        # Toolbar: [ Ports: ] [ QSpinBox ] [ + Add ] [ - Remove ]
        ctrl_bar = QtWidgets.QHBoxLayout()
        ctrl_bar.setContentsMargins(0, 0, 0, 4)
        lbl_ports = QtWidgets.QLabel("Ports:")
        lbl_ports.setStyleSheet("color: #aab0b8; font-weight: 600;")

        current_count = node.get_param_count() if hasattr(node, "get_param_count") else 6
        spin_ports = QtWidgets.QSpinBox()
        has_limit = node.has_property("parameter_limit")
        maximum = max(current_count, int(node.get_property("parameter_limit"))) if has_limit else 2_147_483_647
        spin_ports.setRange(0, maximum)
        spin_ports.setValue(current_count)
        spin_ports.setToolTip("Set the number of parameter input sockets for this node.")

        btn_add = QtWidgets.QPushButton("+ Add")
        btn_add.setToolTip("Add an additional parameter port.")
        btn_add.setStyleSheet("padding: 3px 8px; font-weight: 600;")

        btn_remove = QtWidgets.QPushButton("- Remove")
        btn_remove.setToolTip("Remove the highest parameter port.")
        btn_remove.setStyleSheet("padding: 3px 8px; font-weight: 600;")

        ctrl_bar.addWidget(lbl_ports)
        ctrl_bar.addWidget(spin_ports)
        ctrl_bar.addWidget(btn_add)
        ctrl_bar.addWidget(btn_remove)
        ctrl_bar.addStretch()
        param_vlayout.addLayout(ctrl_bar)
        if has_limit:
            limit_row = QtWidgets.QHBoxLayout()
            limit_row.addWidget(QtWidgets.QLabel("Port limit:"))
            limit_spin = QtWidgets.QSpinBox()
            limit_spin.setRange(max(1, current_count), 2_147_483_647)
            limit_spin.setValue(maximum)
            limit_spin.setToolTip("Increase this limit when the part needs more parameter sockets.")

            def _on_limit_change(value):
                spin_ports.setMaximum(value)
                self.update_property("parameter_limit", value)

            limit_spin.valueChanged.connect(_on_limit_change)
            limit_row.addWidget(limit_spin)
            limit_row.addStretch()
            param_vlayout.addLayout(limit_row)

        def _on_count_change(new_val: int):
            new_val = max(0, min(int(new_val), spin_ports.maximum()))
            curr = node.get_param_count() if hasattr(node, "get_param_count") else 6
            if new_val == curr:
                return
            if hasattr(node, "set_param_count"):
                node.set_param_count(new_val)
            self.update_property("param_count", new_val)
            self.display_node(node)

        btn_add.clicked.connect(lambda: _on_count_change(spin_ports.value() + 1))
        btn_remove.clicked.connect(lambda: _on_count_change(spin_ports.value() - 1))
        spin_ports.valueChanged.connect(_on_count_change)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 2, 0, 0)
        form_layout.setSpacing(4)
        for idx in range(1, current_count + 1):
            name_prop = f"param_{idx}_name"
            value_prop = f"param_{idx}_value"
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            name_edit = QtWidgets.QLineEdit(str(node.get_property(name_prop) or ""))
            name_edit.setPlaceholderText("name")
            value_edit = ExpressionEdit(node.get_property(value_prop) or 0.0)
            if hasattr(self, "property_widgets"):
                self.property_widgets[value_prop] = value_edit
            name_edit.editingFinished.connect(
                lambda n=name_prop, w=name_edit: self.update_property(n, w.text())
            )
            value_edit.value_changed.connect(
                lambda v, n=value_prop: self.update_property(n, v)
            )
            row_layout.addWidget(name_edit, 1)
            row_layout.addWidget(value_edit, 1)
            form_layout.addRow(f"P{idx}:", row)
        param_vlayout.addLayout(form_layout)
        param_group.setLayout(param_vlayout)
        self.props_layout.addWidget(param_group)

    def _build_freecad_part_ui(self, node):
        """Inspector for the FreeCAD-backed parametric part node."""
        doc_group = QtWidgets.QGroupBox("FreeCAD Document")
        doc_layout = QtWidgets.QVBoxLayout()

        fcstd_name = str(node.get_property("fcstd_filename") or "")
        lbl_file = QtWidgets.QLabel(f"Document: {fcstd_name}" if fcstd_name else "Document: (auto-created on first open)")
        lbl_file.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        lbl_file.setWordWrap(True)
        doc_layout.addWidget(lbl_file)

        btn_open = QtWidgets.QPushButton("Open in FreeCAD…")
        btn_open.setStyleSheet(
            "QPushButton {"
            "  background: #1e5aab; color: white; border-radius: 4px;"
            "  padding: 7px 12px; font-weight: bold;"
            "}"
            "QPushButton:hover { background: #2673cc; }"
        )
        btn_open.clicked.connect(
            lambda _checked=False, n=node: self._open_freecad_part(n)
        )
        doc_layout.addWidget(btn_open)

        chk_auto = QtWidgets.QCheckBox("Auto-open in FreeCAD on double-click")
        chk_auto.setChecked(bool(node.get_property("auto_open_on_double_click")))
        chk_auto.toggled.connect(
            lambda checked, n=node: self.update_property("auto_open_on_double_click", checked)
        )
        doc_layout.addWidget(chk_auto)

        timeout_layout = QtWidgets.QFormLayout()
        timeout_value = node.get_property("parameter_timeout_s")
        timeout_edit = ExpressionEdit(60.0 if timeout_value is None else timeout_value)
        timeout_edit.setToolTip("Maximum time allowed for a parameter update, in seconds.")
        timeout_edit.value_changed.connect(
            lambda value: self.update_property("parameter_timeout_s", value)
        )
        timeout_layout.addRow("Parameter update timeout (s):", timeout_edit)
        doc_layout.addLayout(timeout_layout)

        btn_preview = QtWidgets.QPushButton("Preview in 3D")
        btn_preview.setToolTip("Run the graph (CAD only) and render this part.")
        btn_preview.clicked.connect(
            lambda _checked=False, n=node: self._preview_cad_part(n)
        )
        doc_layout.addWidget(btn_preview)

        doc_group.setLayout(doc_layout)
        self.props_layout.addWidget(doc_group)

        # Scalable parameter sockets populated from FreeCAD spreadsheet aliases or user input
        self._build_scalable_parameter_group(
            node,
            title="Parameters (Spreadsheet Aliases)",
            tooltip=(
                "Parameters linked to FreeCAD spreadsheet aliases.\n"
                "Use SheetName.alias when multiple sheets share an alias.\n"
                "Adjust 'Ports' to scale available sockets up or down."
            ),
        )

    def _open_freecad_part(self, node) -> None:
        """Launch the FreeCAD subprocess on the part's .FCStd file."""
        app = self._get_main_app()
        if app and hasattr(app, "_open_freecad_for_node"):
            app._open_freecad_for_node(node)
        elif hasattr(node, "open_in_freecad"):
            node.open_in_freecad(parent_qobject=self)

    def _open_cad_code_editor(self, node) -> None:
        """Open the full-screen CadQuery script editor for ``node``."""
        current = str(node.get_property("code") or "")
        dlg = CadCodeEditorDialog(current, node=node, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            new_code = dlg.get_code()
            if new_code != current:
                self.update_property("code", new_code)

    def _preview_cad_part(self, node) -> None:
        app = self._get_main_app()
        if app is None:
            return
        # Force re-execution of this node next run.
        try:
            setattr(node, "_dirty", True)
            setattr(node, "_force_execute", True)
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        app._last_rendered_node = node
        try:
            app._execute_graph(skip_simulation=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Preview failed", f"Code part couldn't be evaluated:\n\n{exc}"
            )

    def _build_material_ui(self, node):
        """Show the material values that the solver will actually consume.

        Preset elastic/plastic fields are resolved from the databases in the
        node ``run()`` methods.  Rendering the stale backing properties as
        editable controls made it look as though those edits affected a named
        preset when they were in fact ignored.  Preset-derived values are now
        visible but locked; choosing Custom exposes the stored custom values.
        """
        from pylcss.cae.design_studio.crash.materials import CRASH_MATERIAL_PRESETS
        from pylcss.cae.design_studio.fem._helpers import MATERIAL_DATABASE

        # One Material node serves the static, topology, lattice, and impact
        # studies. Purpose limits the editor to the properties the connected
        # solver uses instead of mixing every material card into one panel.
        purpose = str(node.get_property("analysis_purpose") or "Structural FEA")
        is_impact = purpose == "Impact"
        preset = str(node.get_property("preset") or "Custom")
        presets = (
            dict(CRASH_MATERIAL_PRESETS)
            if is_impact
            else dict(MATERIAL_DATABASE)
        )
        presets["Custom"] = {}
        preset_group = QtWidgets.QGroupBox("Material")
        preset_layout = QtWidgets.QFormLayout(preset_group)
        purpose_combo = QtWidgets.QComboBox()
        purpose_combo.addItems(["Structural FEA", "Topology / Lattice", "Impact"])
        purpose_combo.setCurrentText(purpose)

        def _change_purpose(value):
            available_presets = (
                CRASH_MATERIAL_PRESETS
                if value == "Impact"
                else MATERIAL_DATABASE
            )
            if str(node.get_property("preset") or "Custom") not in available_presets:
                self.update_property("preset", "Custom")
            self.update_property("analysis_purpose", value)
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        purpose_combo.currentTextChanged.connect(_change_purpose)
        preset_layout.addRow("Purpose:", purpose_combo)
        preset_combo = QtWidgets.QComboBox()
        preset_combo.addItems([str(name) for name in presets.keys()])
        preset_combo.setCurrentText(preset if preset in presets else "Custom")

        def _change_preset(value):
            self.update_property("preset", value)
            if value != "Custom" and value in presets:
                p_data = presets[value]
                if "E" in p_data:
                    self.update_property("youngs_modulus", float(p_data["E"]))
                if "nu" in p_data:
                    self.update_property("poissons_ratio", float(p_data["nu"]))
                if "rho" in p_data:
                    self.update_property("density", float(p_data["rho"]))
                if "yield_strength" in p_data:
                    self.update_property("yield_strength", float(p_data["yield_strength"]))
                if "tangent_modulus" in p_data:
                    self.update_property("tangent_modulus", float(p_data["tangent_modulus"]))
                if "failure_strain" in p_data:
                    self.update_property("failure_strain", float(p_data["failure_strain"]))
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        preset_combo.currentTextChanged.connect(_change_preset)
        preset_layout.addRow("Preset:", preset_combo)

        locked = preset != "Custom" and preset in presets
        has_input_overrides = (
            not is_impact
            and any(
                self._connected_input_ports(node, name)
                for name in (
                    "youngs_modulus",
                    "poissons_ratio",
                    "density",
                )
            )
        )
        source_note = QtWidgets.QLabel(
            "Connected values override the preset fields shown below."
            if has_input_overrides
            else "Preset values shown below are the exact solver inputs. Choose Custom "
            "to edit them."
            if locked
            else "Custom material values are editable and are passed directly to the solver."
        )
        source_note.setWordWrap(True)
        source_note.setStyleSheet("color:#8f98a5; font-size:10px;")
        preset_layout.addRow(source_note)

        preset_values = presets.get(preset, {}) if locked else {}
        field_specs = [
            ("youngs_modulus", "E", "Young's modulus (MPa)"),
            ("poissons_ratio", "nu", "Poisson's ratio"),
            ("density", "rho", "Density (t/mm^3)"),
        ]
        if is_impact:
            field_specs.extend(
                [
                    ("yield_strength", "yield_strength", "Yield strength (MPa)"),
                    ("tangent_modulus", "tangent_modulus", "Tangent modulus (MPa)"),
                ]
            )
            if bool(node.get_property("enable_fracture")):
                field_specs.append(
                    ("failure_strain", "failure_strain", "Failure strain")
                )

        for prop_name, preset_key, label in field_specs:
            value = preset_values.get(preset_key, node.get_property(prop_name))
            connected_ports = (
                self._connected_input_ports(node, prop_name)
                if not is_impact
                else []
            )
            if connected_ports:
                try:
                    upstream_value = getattr(
                        connected_ports[0].node(),
                        "_last_result",
                        None,
                    )
                    if isinstance(upstream_value, (int, float)):
                        value = upstream_value
                except Exception:
                    pass
            editor = ExpressionEdit(value if value is not None else 0.0)
            editor.setEnabled(not locked and not connected_ports)
            if connected_ports:
                editor.setToolTip("Driven by the connected input port.")
            elif locked:
                editor.setToolTip(
                    f"Resolved from the '{preset}' database entry. Choose Custom to edit."
                )
            else:
                editor.value_changed.connect(
                    lambda v, p=prop_name: self.update_property(p, v)
                )
            preset_layout.addRow(label + ":", editor)
            self.property_widgets[prop_name] = editor

        self.props_layout.addWidget(preset_group)

        if is_impact:
            behavior = QtWidgets.QGroupBox("Impact Material Behavior")
            behavior_layout = QtWidgets.QFormLayout(behavior)
            fracture = QtWidgets.QCheckBox("Delete failed elements")
            fracture.setChecked(bool(node.get_property("enable_fracture")))
            fracture.toggled.connect(
                lambda checked: self.update_property("enable_fracture", bool(checked))
            )
            fracture.setToolTip(
                "Delete elements after the equivalent plastic strain reaches the failure strain."
            )
            behavior_layout.addRow("Fracture:", fracture)

            rate = QtWidgets.QCheckBox("Use preset strain-rate law")
            rate.setChecked(bool(node.get_property("strain_rate_sensitive")))
            rate.toggled.connect(
                lambda checked: self.update_property(
                    "strain_rate_sensitive", bool(checked)
                )
            )
            rate.setToolTip(
                "Enable the preset Cowper-Symonds rate hardening. The internal constants "
                "are tied to the selected material model."
            )
            behavior_layout.addRow("Rate effects:", rate)
            self.props_layout.addWidget(behavior)
        elif purpose == "Structural FEA":
            plasticity = QtWidgets.QGroupBox("Optional Bilinear Plasticity")
            plastic_layout = QtWidgets.QFormLayout(plasticity)
            for prop_name, label in (
                ("yield_strength", "Yield strength (MPa)"),
                ("tangent_modulus", "Tangent modulus (MPa)"),
            ):
                editor = ExpressionEdit(node.get_property(prop_name) or 0.0)
                editor.value_changed.connect(
                    lambda v, p=prop_name: self.update_property(p, v)
                )
                plastic_layout.addRow(label + ":", editor)
                self.property_widgets[prop_name] = editor
            hint = QtWidgets.QLabel(
                "Yield strength = 0 keeps the material purely elastic. These two "
                "values intentionally override the selected elastic preset."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#8f98a5; font-size:10px;")
            plastic_layout.addRow(hint)
            self.props_layout.addWidget(plasticity)

    @staticmethod
    def _connected_input_ports(node, port_name):
        """Return the output ports currently connected to one node input."""
        try:
            port = node.get_input(port_name)
            return list(port.connected_ports()) if port is not None else []
        except Exception:
            return []

    def _build_mesh_ui(self, node):
        """Inspector for MeshNode: context-aware for CAD vs NPZ and Tet vs Shell."""
        source = str(node.get_property("mesh_source") or "CAD").strip()
        mesh_type = str(node.get_property("mesh_type") or "Tet").strip()
        is_cad = source == "CAD"
        is_shell = mesh_type.lower() == "shell"

        grp = QtWidgets.QGroupBox("Mesh")
        lay = QtWidgets.QFormLayout(grp)

        # 1. Source selector
        combo_source = QtWidgets.QComboBox()
        combo_source.addItems(["CAD", "Mesh NPZ"])
        combo_source.blockSignals(True)
        combo_source.setCurrentText(source)
        combo_source.blockSignals(False)

        def _on_source_change(val):
            self.update_property("mesh_source", val)
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        combo_source.currentTextChanged.connect(_on_source_change)
        lay.addRow("Source:", combo_source)

        if not is_cad:
            # Pre-generated mesh NPZ path
            mesh_path = str(node.get_property("mesh_path") or "")
            path_edit = QtWidgets.QLineEdit(mesh_path)
            path_edit.setPlaceholderText("Path to .npz mesh file")
            path_edit.editingFinished.connect(
                lambda: self.update_property("mesh_path", path_edit.text().strip())
            )
            lay.addRow("Mesh File:", path_edit)
            self.property_widgets["mesh_path"] = path_edit
        else:
            # CAD Meshing controls
            combo_type = QtWidgets.QComboBox()
            type_options = {
                "Tet (Linear C3D4 - Fast)": "Tet",
                "Tet10 (Quadratic C3D10 - High Precision)": "Tet10",
                "Shell (2D Surface - OpenCourant Crash Only)": "Shell",
            }
            rev_type_options = {v: k for k, v in type_options.items()}
            combo_type.addItems(list(type_options.keys()))
            curr_label = rev_type_options.get(mesh_type, "Tet (Linear C3D4 - Fast)")
            combo_type.blockSignals(True)
            combo_type.setCurrentText(curr_label)
            combo_type.blockSignals(False)

            def _on_type_change(val):
                internal = type_options.get(val, "Tet")
                self.update_property("mesh_type", internal)
                QtCore.QTimer.singleShot(
                    0,
                    lambda n=node: self.display_node(n) if self.current_node is n else None,
                )

            combo_type.currentTextChanged.connect(_on_type_change)
            lay.addRow("Element Formulation:", combo_type)

            # Global element size
            elem_size = float(node.get_property("element_size") or 2.0)
            spin_size = QtWidgets.QDoubleSpinBox()
            spin_size.setRange(0.01, 1000.0)
            spin_size.setDecimals(2)
            spin_size.setValue(elem_size)
            spin_size.setSuffix(" mm")
            spin_size.valueChanged.connect(
                lambda v: self.update_property("element_size", v)
            )
            self.property_widgets["element_size"] = spin_size
            lay.addRow("Element Size:", spin_size)

            # Local refinement size
            ref_size = float(node.get_property("refinement_size") or 0.5)
            spin_ref = QtWidgets.QDoubleSpinBox()
            spin_ref.setRange(0.001, 1000.0)
            spin_ref.setDecimals(3)
            spin_ref.setValue(ref_size)
            spin_ref.setSuffix(" mm")
            spin_ref.valueChanged.connect(
                lambda v: self.update_property("refinement_size", v)
            )
            self.property_widgets["refinement_size"] = spin_ref
            lay.addRow("Refinement Size:", spin_ref)

            # Shell properties: ONLY visible when mesh_type == 'Shell'
            if is_shell:
                shell_grp = QtWidgets.QGroupBox("Shell Formulation (Crash)")
                shell_lay = QtWidgets.QFormLayout(shell_grp)

                shell_t = float(node.get_property("shell_thickness") or 1.5)
                spin_t = QtWidgets.QDoubleSpinBox()
                spin_t.setRange(0.01, 100.0)
                spin_t.setDecimals(2)
                spin_t.setValue(shell_t)
                spin_t.setSuffix(" mm")
                spin_t.valueChanged.connect(
                    lambda v: self.update_property("shell_thickness", v)
                )
                self.property_widgets["shell_thickness"] = spin_t
                shell_lay.addRow("Wall Thickness:", spin_t)

                shell_nip = int(node.get_property("shell_nip") or 5)
                spin_nip = QtWidgets.QSpinBox()
                spin_nip.setRange(1, 20)
                spin_nip.setValue(shell_nip)
                spin_nip.valueChanged.connect(
                    lambda v: self.update_property("shell_nip", v)
                )
                self.property_widgets["shell_nip"] = spin_nip
                shell_lay.addRow("Through-Thickness Points:", spin_nip)

                note = QtWidgets.QLabel(
                    "Shell meshing produces 2D surface triangles with assigned wall thickness. "
                    "Used exclusively by OpenCourant Crash Solver for thin structures (tubes, sheet metal); "
                    "CalculiX Static FEA requires solid continuum elements (Tet or Tet10)."
                )
                note.setWordWrap(True)
                note.setStyleSheet("color:#8f98a5; font-size:10px;")
                shell_lay.addRow(note)

                self.props_layout.addWidget(grp)
                self.props_layout.addWidget(shell_grp)
                return

        self.props_layout.addWidget(grp)

    def _build_fea_solver_ui(self, node):
        """FEA solver inspector: solver settings, results summary, and static equilibrium."""
        self._build_generic_ui(node)

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") != "fea":
            return

        # ── Results Summary Group ──
        res_grp = QtWidgets.QGroupBox("FEA Results Summary")
        res_layout = QtWidgets.QFormLayout(res_grp)

        peak_vm = result.get("peak_stress_nodal")
        if peak_vm is not None:
            lbl_vm = QtWidgets.QLabel(f"{float(peak_vm):.2f} MPa")
            lbl_vm.setStyleSheet("font-weight: bold; font-size: 11px;")
            res_layout.addRow("Peak Von Mises Stress", lbl_vm)

        peak_disp = result.get("peak_displacement")
        if peak_disp is not None:
            lbl_disp = QtWidgets.QLabel(f"{float(peak_disp):.4f} mm")
            lbl_disp.setStyleSheet("font-weight: bold; font-size: 11px;")
            res_layout.addRow("Peak Displacement", lbl_disp)

        min_sf = result.get("min_safety_factor")
        if min_sf is not None:
            sf_val = float(min_sf)
            if sf_val < 1.0:
                sf_style = "color: #ff5252; font-weight: bold; font-size: 11px;"
                sf_text = f"{sf_val:.2f} (Failure Risk / Yield Exceeded)"
            elif sf_val < 1.5:
                sf_style = "color: #ffb74d; font-weight: bold; font-size: 11px;"
                sf_text = f"{sf_val:.2f} (Marginal Safety)"
            else:
                sf_style = "color: #66d17a; font-weight: bold; font-size: 11px;"
                sf_text = f"{sf_val:.2f} (Safe)"
            lbl_sf = QtWidgets.QLabel(sf_text)
            lbl_sf.setStyleSheet(sf_style)
            res_layout.addRow("Min Safety Factor (FoS)", lbl_sf)

        self.props_layout.addWidget(res_grp)

        # ── Solution Verification & Static Equilibrium Group ──
        eq_grp = QtWidgets.QGroupBox("Solution Verification and Equilibrium")
        eq_layout = QtWidgets.QFormLayout(eq_grp)

        app_f = result.get("applied_force")
        rxn_f = result.get("reaction_force")
        res_f = result.get("residual_force")
        err_pct = float(result.get("equilibrium_error_pct", 0.0) or 0.0)
        is_balanced = bool(result.get("equilibrium_balanced", True))

        if app_f is not None:
            af = np.asarray(app_f, dtype=float)
            lbl_app = QtWidgets.QLabel(
                f"[{af[0]:.2f}, {af[1]:.2f}, {af[2]:.2f}] N  (|F| = {float(np.linalg.norm(af)):.2f} N)"
            )
            lbl_app.setStyleSheet("font-size: 11px;")
            eq_layout.addRow("Total Applied Force", lbl_app)

        if rxn_f is not None:
            rf = np.asarray(rxn_f, dtype=float)
            lbl_rxn = QtWidgets.QLabel(
                f"[{rf[0]:.2f}, {rf[1]:.2f}, {rf[2]:.2f}] N  (|R| = {float(np.linalg.norm(rf)):.2f} N)"
            )
            lbl_rxn.setStyleSheet("font-size: 11px;")
            eq_layout.addRow("Total Reaction Force", lbl_rxn)

        if res_f is not None:
            rf_res = np.asarray(res_f, dtype=float)
            lbl_res = QtWidgets.QLabel(
                f"[{rf_res[0]:.2e}, {rf_res[1]:.2e}, {rf_res[2]:.2e}] N  (|ΔF| = {float(np.linalg.norm(rf_res)):.2e} N)"
            )
            lbl_res.setStyleSheet("font-size: 11px;")
            eq_layout.addRow("Residual Force", lbl_res)

        status_lbl = QtWidgets.QLabel(
            f"✓ Balanced ({err_pct:.3f}% residual error)"
            if is_balanced
            else f"⚠ Imbalance detected ({err_pct:.1f}% error)"
        )
        status_lbl.setStyleSheet(
            "color: #66d17a; font-weight: bold; font-size: 11px;"
            if is_balanced
            else "color: #ffb74d; font-weight: bold; font-size: 11px;"
        )
        eq_layout.addRow("Equilibrium Status", status_lbl)

        self.props_layout.addWidget(eq_grp)

        # ── Multi-Body Tied Contact & Components Breakdown ──
        connections = result.get("connections") or []
        if connections:
            conn_grp = QtWidgets.QGroupBox(f"Bonded Contact Interfaces (*TIE) [{len(connections)}]")
            conn_lay = QtWidgets.QFormLayout(conn_grp)
            for c in connections:
                c_name = c.get("name", "TIE")
                part_a = c.get("component_a", "Part A")
                part_b = c.get("component_b", "Part B")
                tol = c.get("search_tolerance_mm", 0.25)
                gap = c.get("bounding_box_gap_mm", 0.0)
                lbl_c = QtWidgets.QLabel(f"{part_a} ↔ {part_b} (gap {gap:.3f} mm, tol {tol:.3f} mm)")
                lbl_c.setStyleSheet("color: #66d17a; font-size: 11px;")
                conn_lay.addRow(f"Tie {c_name}:", lbl_c)
            self.props_layout.addWidget(conn_grp)

        comp_summaries = result.get("component_summaries") or []
        if comp_summaries:
            comp_grp = QtWidgets.QGroupBox(f"Component Breakdown [{len(comp_summaries)} Parts]")
            comp_lay = QtWidgets.QFormLayout(comp_grp)
            for cp in comp_summaries:
                pname = cp.get("name", "Part")
                pmaterial = cp.get("material", "Material")
                pvm = cp.get("peak_stress_mpa", 0.0)
                pdisp = cp.get("peak_displacement_mm", 0.0)
                psf = cp.get("yield_safety_factor")
                sf_txt = f", FoS: {psf:.2f}" if psf is not None else ""
                lbl_p = QtWidgets.QLabel(f"{pmaterial} · {pvm:.1f} MPa · {pdisp:.3f} mm{sf_txt}")
                lbl_p.setStyleSheet("font-size: 11px;")
                comp_lay.addRow(f"{pname}:", lbl_p)
            self.props_layout.addWidget(comp_grp)

    def _build_modal_solver_ui(self, node):
        """Modal solver inspector: settings and interactive Natural Frequencies table."""
        self._build_generic_ui(node)

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") != "modal":
            return

        modes = result.get("modes") or []
        if not modes:
            return

        grp = QtWidgets.QGroupBox("Natural Frequencies and Mode Shapes")
        layout = QtWidgets.QVBoxLayout(grp)

        info_lbl = QtWidgets.QLabel("Click a mode row to preview its 3D mode shape:")
        info_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px; margin-bottom: 2px;")
        layout.addWidget(info_lbl)

        table = QtWidgets.QTableWidget(len(modes), 4)
        table.setHorizontalHeaderLabels(
            ["Mode", "Frequency (Hz)", "Angular (rad/s)", "Period (s)"]
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        active_mode = int(result.get("active_mode", 1))

        for row, m in enumerate(modes):
            m_no = m.get("mode", row + 1)
            f_hz = m.get("frequency_hz", 0.0)
            f_rad = m.get("frequency_rad_s", 0.0)
            per = m.get("period_s", 0.0)

            item_mode = QtWidgets.QTableWidgetItem(str(m_no))
            item_mode.setTextAlignment(QtCore.Qt.AlignCenter)
            item_hz = QtWidgets.QTableWidgetItem(f"{f_hz:.2f}")
            item_hz.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item_rad = QtWidgets.QTableWidgetItem(f"{f_rad:.1f}")
            item_rad.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item_per = QtWidgets.QTableWidgetItem(
                f"{per:.4e}" if math.isfinite(per) else "∞"
            )
            item_per.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            table.setItem(row, 0, item_mode)
            table.setItem(row, 1, item_hz)
            table.setItem(row, 2, item_rad)
            table.setItem(row, 3, item_per)

        # Highlight current active mode
        if 1 <= active_mode <= len(modes):
            table.selectRow(active_mode - 1)

        def _on_mode_selected():
            selected_rows = table.selectionModel().selectedRows()
            if not selected_rows:
                return
            row = selected_rows[0].row()
            sel_mode = row + 1
            node.set_property("active_mode", sel_mode)
            result["active_mode"] = sel_mode
            flat_disp = result.get("mode_displacements_flat", {}).get(sel_mode)
            if flat_disp is not None:
                result["displacement"] = flat_disp
            f_val = modes[row].get("frequency_hz", 0.0)
            result["frequency_hz"] = f_val
            result["visualization_mode"] = f"Mode {sel_mode} ({f_val:.1f} Hz)"
            from .result_fields import activate_modal_mode
            activate_modal_mode(result, result["visualization_mode"])
            parent_window = self.parentWidget()
            while parent_window is not None and not hasattr(parent_window, "viewer"):
                parent_window = parent_window.parentWidget()
            viewer = getattr(parent_window, "viewer", None)
            if viewer is not None and hasattr(viewer, "render_simulation"):
                viewer.render_simulation(result)

        table.itemSelectionChanged.connect(_on_mode_selected)
        table.setFixedHeight(min(220, 28 * len(modes) + 30))
        layout.addWidget(table)

        self.props_layout.addWidget(grp)

    def _build_crash_solver_ui(self, node):
        """Impact solver inspector: settings, crashworthiness KPIs, and energy balance."""
        result = getattr(node, "_last_result", None)
        has_results = isinstance(result, dict) and result.get("type") in ("crash", "external_solver")

        if not has_results:
            # No results yet — show setup properties directly (no tab widget needed).
            self._build_generic_ui(node)
            return

        metrics = result.get("crash_metrics") or {}
        quality = result.get("quality") or {}

        # ── Tab widget: Setup | Results ──
        tab_widget = QtWidgets.QTabWidget()
        tab_widget.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #2f333a; border-radius: 5px; background: transparent; }"
            "QTabBar::tab {"
            "  background: #24272d; color: #9aa0a6; border: 1px solid #2f333a;"
            "  border-bottom: none; border-radius: 4px 4px 0 0;"
            "  padding: 4px 12px; font-size: 11px; font-weight: 600;"
            "}"
            "QTabBar::tab:selected { background: #1c1e22; color: #d29922; border-bottom: 2px solid #d29922; }"
            "QTabBar::tab:hover:!selected { background: #2a2e35; color: #cdd2d9; }"
        )

        # -- Setup tab --
        setup_tab = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(setup_tab)
        setup_layout.setContentsMargins(2, 6, 2, 2)
        setup_layout.setSpacing(5)
        setup_layout.setAlignment(QtCore.Qt.AlignTop)

        # Temporarily redirect props_layout so _build_generic_ui populates the
        # Setup tab instead of the outer inspector scroll area.
        _saved_layout = self.props_layout
        _saved_widget = self.props_widget
        self.props_layout = setup_layout
        self.props_widget = setup_tab
        try:
            self._build_generic_ui(node)
        finally:
            self.props_layout = _saved_layout
            self.props_widget = _saved_widget

        setup_layout.addStretch(1)
        tab_widget.addTab(setup_tab, "Setup")

        # -- Results tab --
        results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_tab)
        results_layout.setContentsMargins(2, 6, 2, 2)
        results_layout.setSpacing(5)
        results_layout.setAlignment(QtCore.Qt.AlignTop)

        # ── Crashworthiness Performance Group ──
        kpi_grp = QtWidgets.QGroupBox("Crashworthiness and Energy Absorption")
        kpi_layout = QtWidgets.QFormLayout(kpi_grp)

        peak_f = float(result.get("peak_force") or metrics.get("peak_crushing_force_kN") or 0.0)
        mean_f = float(result.get("mean_force") or metrics.get("mean_crushing_force_kN") or 0.0)
        cfe = float(
            result.get("crush_force_efficiency")
            or metrics.get("crush_force_efficiency")
            or (mean_f / peak_f if peak_f > 0 else 0.0)
        )
        sea = float(
            result.get("specific_energy_absorption")
            or metrics.get("specific_energy_absorption_kJ_kg")
            or 0.0
        )
        absorbed_e = float(
            result.get("absorbed_energy_kj")
            or metrics.get("absorbed_energy_kJ")
            or (float(result.get("absorbed_energy", 0.0)) / 1.0e6)
        )
        stroke = float(
            result.get("crush_distance")
            or metrics.get("useful_crush_stroke_mm")
            or 0.0
        )
        peak_a = float(
            result.get("peak_acceleration_g")
            or metrics.get("peak_acceleration_g")
            or 0.0
        )
        delta_v = float(
            result.get("delta_v")
            or metrics.get("delta_v_m_s")
            or 0.0
        )
        n_failed = int(result.get("n_failed", 0) or 0)

        # CFE row with status coloring
        cfe_pct = cfe * 100.0
        if cfe >= 0.70:
            cfe_style = "color: #66d17a; font-weight: bold; font-size: 11px;"
            cfe_hint = " (Excellent / Progressive Buckling)"
        elif cfe >= 0.50:
            cfe_style = "color: #4fc3f7; font-weight: bold; font-size: 11px;"
            cfe_hint = " (Good / Typical Crush Box)"
        else:
            cfe_style = "color: #ffb74d; font-weight: bold; font-size: 11px;"
            cfe_hint = " (Low Efficiency / High Initial Peak)"
        lbl_cfe = QtWidgets.QLabel(f"{cfe_pct:.1f}%{cfe_hint}")
        lbl_cfe.setStyleSheet(cfe_style)
        kpi_layout.addRow("Crash Force Efficiency (CFE)", lbl_cfe)

        # SEA row
        lbl_sea = QtWidgets.QLabel(f"{sea:.2f} kJ/kg")
        lbl_sea.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 11px;")
        lbl_sea.setToolTip("Specific Energy Absorption per unit structural mass (E_abs / mass)")
        kpi_layout.addRow("Specific Energy Absorption (SEA)", lbl_sea)

        # Peak & Mean Force
        lbl_pf = QtWidgets.QLabel(f"{peak_f:.2f} kN")
        lbl_pf.setStyleSheet("font-weight: bold; font-size: 11px;")
        kpi_layout.addRow("Peak Crushing Force (F_peak)", lbl_pf)

        lbl_mf = QtWidgets.QLabel(f"{mean_f:.2f} kN")
        lbl_mf.setStyleSheet("font-size: 11px;")
        kpi_layout.addRow("Mean Crushing Force (F_mean)", lbl_mf)

        # Absorbed Energy & Stroke
        lbl_ae = QtWidgets.QLabel(f"{absorbed_e:.2f} kJ")
        lbl_ae.setStyleSheet("font-weight: bold; font-size: 11px;")
        kpi_layout.addRow("Total Energy Absorbed", lbl_ae)

        lbl_stroke = QtWidgets.QLabel(f"{stroke:.1f} mm")
        lbl_stroke.setStyleSheet("font-size: 11px;")
        kpi_layout.addRow("Useful Crush Stroke", lbl_stroke)

        if peak_a > 0.0:
            lbl_pa = QtWidgets.QLabel(f"{peak_a:.1f} g")
            lbl_pa.setStyleSheet("font-size: 11px;")
            kpi_layout.addRow("Peak Deceleration", lbl_pa)

        if delta_v > 0.0:
            lbl_dv = QtWidgets.QLabel(f"{delta_v:.2f} m/s")
            lbl_dv.setStyleSheet("font-size: 11px;")
            kpi_layout.addRow("Velocity Change (ΔV)", lbl_dv)

        lbl_failed = QtWidgets.QLabel(
            "0 (Intact - No Rupture)" if n_failed == 0 else f"{n_failed} elements eroded"
        )
        lbl_failed.setStyleSheet(
            "color: #66d17a; font-size: 11px;"
            if n_failed == 0
            else "color: #ff5252; font-weight: bold; font-size: 11px;"
        )
        kpi_layout.addRow("Eroded Elements", lbl_failed)

        # Button to view interactive curves
        btn_curves = QtWidgets.QPushButton("View Time-History Curves (F-d, Energy, Pulse)…")
        btn_curves.setStyleSheet(
            "QPushButton {"
            "  background: #1e5aab; color: white; border-radius: 4px;"
            "  padding: 7px 12px; font-weight: bold; font-size: 11px;"
            "  margin-top: 4px;"
            "}"
            "QPushButton:hover { background: #2673cc; }"
        )
        btn_curves.clicked.connect(
            lambda _checked=False, r=result: CrashPlotDialog(r, parent=self).exec()
        )
        kpi_layout.addRow(btn_curves)
        results_layout.addWidget(kpi_grp)

        # ── Solution Quality & Energy Gates Group ──
        qual_grp = QtWidgets.QGroupBox("Solution Quality and Energy Gates")
        qual_layout = QtWidgets.QFormLayout(qual_grp)

        q_status = str(
            result.get("numerical_status") or quality.get("numerical_status")
            or result.get("quality_status") or quality.get("status") or ""
        ).lower()
        if q_status == "pass":
            stat_lbl = QtWidgets.QLabel("✓ Numerical checks passed")
            stat_lbl.setStyleSheet("color: #66d17a; font-weight: bold; font-size: 11px;")
        elif q_status == "warning":
            stat_lbl = QtWidgets.QLabel("⚠ Numerical checks need review")
            stat_lbl.setStyleSheet("color: #ffb74d; font-weight: bold; font-size: 11px;")
        elif q_status == "fail":
            stat_lbl = QtWidgets.QLabel("✕ Numerical checks failed")
            stat_lbl.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 11px;")
        else:
            stat_lbl = QtWidgets.QLabel("Numerical checks incomplete")
            stat_lbl.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        qual_layout.addRow("Numerical Status", stat_lbl)
        physical_status = str(quality.get("physical_validation_status") or "unverified")
        qual_layout.addRow(
            "Material Validation",
            QtWidgets.QLabel(physical_status.replace("_", " ").title()),
        )
        for check in quality.get("checks", []):
            if isinstance(check, dict) and check.get("status") != "pass":
                detail = QtWidgets.QLabel(
                    f"{str(check.get('status', '')).replace('_', ' ').title()}: "
                    f"{check.get('message', '')}"
                )
                detail.setWordWrap(True)
                qual_layout.addRow(
                    str(check.get("id", "")).replace("_", " ").title(), detail,
                )

        # Extract quality check limits
        checks = {c.get("id"): c for c in quality.get("checks", []) if isinstance(c, dict)}

        # Energy balance check
        eb_chk = checks.get("energy_balance", {})
        eb_val = eb_chk.get("value")
        if eb_val is not None:
            eb_pct = float(eb_val)
            eb_stat = eb_chk.get("status", "pass")
            eb_style = "color: #66d17a; font-size: 11px;" if eb_stat == "pass" else "color: #ffb74d; font-size: 11px;" if eb_stat == "warning" else "color: #ff5252; font-size: 11px;"
            lbl_eb = QtWidgets.QLabel(f"{eb_pct:.2f}% (Limit ≤ 5% pass)")
            lbl_eb.setStyleSheet(eb_style)
            qual_layout.addRow("Energy Balance Error", lbl_eb)

        # Hourglass energy check
        hg_chk = checks.get("hourglass_energy", {})
        hg_val = hg_chk.get("value")
        if hg_val is not None:
            hg_pct = float(hg_val)
            hg_stat = hg_chk.get("status", "pass")
            hg_style = "color: #66d17a; font-size: 11px;" if hg_stat == "pass" else "color: #ffb74d; font-size: 11px;" if hg_stat == "warning" else "color: #ff5252; font-size: 11px;"
            lbl_hg = QtWidgets.QLabel(f"{hg_pct:.2f}% (Limit ≤ 10% pass)")
            lbl_hg.setStyleSheet(hg_style)
            qual_layout.addRow("Hourglass Energy Ratio", lbl_hg)

        # Combined artificial energy
        art_chk = checks.get("hourglass_plus_contact_energy", {})
        art_val = art_chk.get("value")
        if art_val is not None:
            art_pct = float(art_val)
            art_stat = art_chk.get("status", "pass")
            art_style = "color: #66d17a; font-size: 11px;" if art_stat == "pass" else "color: #ff5252; font-size: 11px;"
            lbl_art = QtWidgets.QLabel(f"{art_pct:.2f}% (Limit ≤ 15% pass)")
            lbl_art.setStyleSheet(art_style)
            qual_layout.addRow("Hourglass + Contact Energy", lbl_art)

        # Added mass from mass scaling
        mass_chk = checks.get("added_mass", {})
        mass_val = mass_chk.get("value")
        if mass_val is not None:
            mass_pct = float(mass_val)
            mass_stat = mass_chk.get("status", "pass")
            mass_style = "color: #66d17a; font-size: 11px;" if mass_stat == "pass" else "color: #ffb74d; font-size: 11px;" if mass_stat == "warning" else "color: #ff5252; font-size: 11px;"
            lbl_mass = QtWidgets.QLabel(f"{mass_pct:.2f}% (Limit ≤ 1% pass)")
            lbl_mass.setStyleSheet(mass_style)
            qual_layout.addRow("Added Mass (Scaling)", lbl_mass)

        ext_stat = str(result.get("external_status", "engine_completed"))
        clean_stat = ext_stat.replace("_", " ").title()
        is_normal = "fail" not in ext_stat.lower() and "error" not in ext_stat.lower()
        lbl_term = QtWidgets.QLabel(f"Normal ({clean_stat})" if is_normal else clean_stat)
        lbl_term.setStyleSheet("color: #66d17a; font-size: 11px;" if is_normal else "color: #ff5252; font-size: 11px;")
        qual_layout.addRow("Solver Termination", lbl_term)

        results_layout.addWidget(qual_grp)
        results_layout.addStretch(1)
        tab_widget.addTab(results_tab, "Results")

        # Open on the Results tab when results are present.
        tab_widget.setCurrentIndex(1)

        self.props_layout.addWidget(tab_widget)
