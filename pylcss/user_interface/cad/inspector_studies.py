# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""StudyInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging


from PySide6 import QtCore, QtGui, QtWidgets


from .code_editor import CadCodeEditorDialog
from .inspector_controls import ExpressionEdit
from .panels import LibraryPanel

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
            name_prop = f"param_{idx}_name"
            value_prop = f"param_{idx}_value"
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            name_edit = QtWidgets.QLineEdit(str(node.get_property(name_prop) or ""))
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
        is_crash = node.__class__.__name__ == "CrashMaterialNode"
        if is_crash:
            from pylcss.design_studio.crash.materials import CRASH_MATERIAL_PRESETS

            presets = CRASH_MATERIAL_PRESETS
        else:
            from pylcss.design_studio.fem._helpers import MATERIAL_DATABASE

            presets = MATERIAL_DATABASE

        preset = str(node.get_property("preset") or "Custom")
        preset_group = QtWidgets.QGroupBox("Material Definition")
        preset_layout = QtWidgets.QFormLayout(preset_group)
        preset_combo = QtWidgets.QComboBox()
        preset_combo.addItems([str(name) for name in presets.keys()])
        preset_combo.setCurrentText(preset if preset in presets else "Custom")

        def _change_preset(value):
            self.update_property("preset", value)
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        preset_combo.currentTextChanged.connect(_change_preset)
        preset_layout.addRow("Preset:", preset_combo)

        locked = preset != "Custom" and preset in presets
        source_note = QtWidgets.QLabel(
            "Preset values shown below are the exact solver inputs. Choose Custom "
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
        if not is_crash:
            field_specs.append(
                (
                    "thermal_conductivity",
                    "k",
                    "Thermal conductivity (W/m K)",
                )
            )
        if is_crash:
            field_specs.extend(
                [
                    ("yield_strength", "yield_strength", "Yield strength (MPa)"),
                    ("tangent_modulus", "tangent_modulus", "Tangent modulus (MPa)"),
                    ("failure_strain", "failure_strain", "Failure strain"),
                ]
            )

        for prop_name, preset_key, label in field_specs:
            value = preset_values.get(preset_key, node.get_property(prop_name))
            editor = ExpressionEdit(value if value is not None else 0.0)
            editor.setEnabled(not locked)
            if locked:
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

        if is_crash:
            behavior = QtWidgets.QGroupBox("Crash Material Behavior")
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
        else:
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

    @classmethod
    def _connected_input_nodes(cls, node, port_name):
        nodes = []
        for port in cls._connected_input_ports(node, port_name):
            try:
                nodes.append(port.node())
            except Exception:
                continue
        return nodes

    def _finish_study_definition_edit(self, app, solver_node):
        """Restore the solver inspector once an atomic quick-add has finished."""

        def _finish():
            # NodeGraphQt may select a node while it is being added. Keep the
            # study definition anchored to the solver the user was editing.
            try:
                app.graph.clear_selection()
                solver_node.set_selected(True)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            finally:
                app._batching_study_definition_edit = False

            self._refresh_study_definition_statuses(solver_node)

        QtCore.QTimer.singleShot(0, _finish)

    def _add_solver_setup_node(
        self,
        solver_node,
        *,
        node_id,
        label,
        output_name,
        solver_input,
        mesh_input=None,
    ):
        """Spawn and wire a boundary-condition node from a solver inspector."""
        app = self._get_main_app()
        if app is None:
            return
        app._batching_study_definition_edit = True
        try:
            x, y = solver_node.pos()
        except Exception:
            x, y = (0.0, 0.0)

        existing = len(self._connected_input_ports(solver_node, solver_input))
        created = app._spawn_node(
            node_id,
            label,
            x=float(x) - 330.0,
            y=float(y) + 120.0 * existing,
        )
        if created is None:
            app._batching_study_definition_edit = False
            return

        try:
            created.get_output(output_name).connect_to(
                solver_node.get_input(solver_input)
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

        # A new FEA condition needs the same mesh as its solver. Reuse that
        # upstream connection so quick-add leaves only the face to define.
        if mesh_input:
            mesh_sources = self._connected_input_ports(solver_node, "mesh")
            if mesh_sources:
                try:
                    mesh_sources[0].connect_to(created.get_input(mesh_input))
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

        self._finish_study_definition_edit(app, solver_node)

    @staticmethod
    def _study_status_label(count, *, optional=False, detail=""):
        label = QtWidgets.QLabel()
        StudyInspectorMixin._set_study_status_label(
            label, count, optional=optional, detail=detail
        )
        return label

    @staticmethod
    def _set_study_status_label(label, count, *, optional=False, detail=""):
        if count:
            text, color = f"{count} connected", "#72d38a"
        elif optional:
            text, color = "Optional", "#8f98a5"
        else:
            text, color = "Not connected", "#ffb35c"
        if detail:
            text += f" - {detail}"
        label.setText(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color:{color};")

    def _refresh_study_definition_statuses(self, node):
        """Update connection badges without rebuilding the whole inspector."""
        if self.current_node is not node:
            return

        labels = self._study_status_labels
        node_class = node.__class__.__name__
        impact_scope = ""
        support_optional = False
        driven_displacement = False

        if node_class == "SolverNode":
            driven_displacement = self._has_driven_displacement(node)
        elif node_class == "CrashSolverNode":
            impact_nodes = self._connected_input_nodes(node, "impact")
            if impact_nodes:
                try:
                    impact_scope = str(
                        impact_nodes[0].get_property("application_scope") or ""
                    )
                except Exception:
                    impact_scope = ""
            normalized_scope = impact_scope.strip().lower().replace("_", " ")
            support_optional = normalized_scope.startswith(
                ("moving body", "prescribed")
            )

        for port_name, label in labels.items():
            count = len(self._connected_input_ports(node, port_name))
            optional = False
            detail = ""
            if node_class == "SolverNode" and port_name == "loads":
                optional = driven_displacement
                if driven_displacement and not count:
                    detail = "study is driven by displacement"
            elif node_class == "CrashSolverNode":
                if port_name == "impact" and count:
                    detail = impact_scope
                elif port_name == "constraints":
                    optional = support_optional
                    if support_optional and not count:
                        detail = "not used by this scenario"
            self._set_study_status_label(label, count, optional=optional, detail=detail)

    @staticmethod
    def _backend_status_label(ready, detail):
        label = QtWidgets.QLabel("Detected" if ready else "Not detected")
        label.setWordWrap(True)
        label.setStyleSheet("color:#72d38a;" if ready else "color:#ffb35c;")
        label.setToolTip(str(detail))
        return label

    def _has_driven_displacement(self, solver_node):
        """Whether a connected support prescribes a nonzero displacement."""
        for condition in self._connected_input_nodes(solver_node, "constraints"):
            try:
                if str(condition.get_property("constraint_type")) != "Displacement":
                    continue
                for axis in ("x", "y", "z"):
                    enabled = condition.get_property(f"displacement_{axis}_enabled")
                    if enabled is None:
                        enabled = True
                    value = float(condition.get_property(f"displacement_{axis}") or 0.0)
                    if bool(enabled) and abs(value) > 1e-15:
                        return True
            except (TypeError, ValueError):
                continue
        return False

    def _build_solver_study_definition(self, node, study_kind):
        """Build the in-node FEA/Crash setup card used by solver nodes."""
        setup_group = QtWidgets.QGroupBox("Study Definition")
        setup_layout = QtWidgets.QFormLayout(setup_group)
        mesh_status = self._study_status_label(
            len(self._connected_input_ports(node, "mesh"))
        )
        self._study_status_labels["mesh"] = mesh_status
        setup_layout.addRow("Mesh:", mesh_status)

        if study_kind == "fea":
            material_count = len(self._connected_input_ports(node, "material"))
            support_count = len(self._connected_input_ports(node, "constraints"))
            load_count = len(self._connected_input_ports(node, "loads"))
            driven_displacement = self._has_driven_displacement(node)
            material_status = self._study_status_label(material_count)
            support_status = self._study_status_label(support_count)
            load_status = self._study_status_label(
                load_count,
                optional=driven_displacement,
                detail="study is driven by displacement"
                if driven_displacement and not load_count
                else "",
            )
            self._study_status_labels.update(
                {
                    "material": material_status,
                    "constraints": support_status,
                    "loads": load_status,
                }
            )
            setup_layout.addRow("Material:", material_status)
            setup_layout.addRow("Supports:", support_status)
            setup_layout.addRow("Loads:", load_status)
            try:
                backend_ready, backend_detail = LibraryPanel._calculix_status()
            except Exception:
                backend_ready = False
                backend_detail = "CalculiX status unavailable"
            setup_layout.addRow(
                "CalculiX:",
                self._backend_status_label(backend_ready, backend_detail),
            )
            button_specs = [
                (
                    "Add Support",
                    "com.cad.sim.constraint",
                    "FEA Support",
                    "constraints",
                    "constraints",
                    "mesh",
                ),
                (
                    "Add Force",
                    "com.cad.sim.load",
                    "FEA Force",
                    "loads",
                    "loads",
                    "mesh",
                ),
                (
                    "Add Pressure",
                    "com.cad.sim.pressure_load",
                    "FEA Pressure",
                    "loads",
                    "loads",
                    "mesh",
                ),
            ]
            hint_text = (
                "Quick-add connects the new condition to this solver and reuses "
                "its mesh connection. Connect a Pick Geometry output to the new "
                "node's target_face input; gravity loads need no face."
            )
        else:
            material_count = len(self._connected_input_ports(node, "crash_material"))
            impact_nodes = self._connected_input_nodes(node, "impact")
            impact_count = len(impact_nodes)
            support_count = len(self._connected_input_ports(node, "constraints"))
            scope = ""
            if impact_nodes:
                try:
                    scope = str(impact_nodes[0].get_property("application_scope") or "")
                except Exception:
                    scope = ""
            normalized_scope = scope.strip().lower().replace("_", " ")
            support_optional = normalized_scope.startswith(
                ("moving body", "prescribed")
            )
            material_status = self._study_status_label(material_count)
            impact_status = self._study_status_label(
                impact_count, detail=scope if impact_count else ""
            )
            support_status = self._study_status_label(
                support_count,
                optional=support_optional,
                detail=(
                    "not used by this scenario"
                    if support_optional and not support_count
                    else ""
                ),
            )
            self._study_status_labels.update(
                {
                    "crash_material": material_status,
                    "impact": impact_status,
                    "constraints": support_status,
                }
            )
            setup_layout.addRow("Crash material:", material_status)
            setup_layout.addRow("Impact:", impact_status)
            setup_layout.addRow("Supports:", support_status)
            try:
                backend_ready, backend_detail = LibraryPanel._openradioss_status()
            except Exception:
                backend_ready = False
                backend_detail = "OpenRadioss status unavailable"
            setup_layout.addRow(
                "OpenRadioss:",
                self._backend_status_label(backend_ready, backend_detail),
            )
            button_specs = [
                (
                    "Add Impact",
                    "com.cad.sim.impact",
                    "Impact Condition",
                    "impact",
                    "impact",
                    None,
                ),
                (
                    "Add Support",
                    "com.cad.sim.constraint",
                    "Crash Support",
                    "constraints",
                    "constraints",
                    "mesh",
                ),
            ]
            hint_text = (
                "Quick-add wires the condition to this solver. Fixed-specimen "
                "and prescribed-wall scenarios also need a Pick Geometry output "
                "on the Impact node. A support is required only for the "
                "fixed-specimen moving-impactor scenario."
            )

        buttons = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        for index, spec in enumerate(button_specs):
            text, node_id, label, output_name, solver_input, mesh_input = spec
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(
                lambda _checked=False, values=spec[1:]: self._add_solver_setup_node(
                    node,
                    node_id=values[0],
                    label=values[1],
                    output_name=values[2],
                    solver_input=values[3],
                    mesh_input=values[4],
                )
            )
            button_layout.addWidget(button, index // 2, index % 2)
        setup_layout.addRow(buttons)

        hint = QtWidgets.QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8f98a5; font-size:10px;")
        setup_layout.addRow(hint)
        self.props_layout.addWidget(setup_group)

    def _build_fea_solver_ui(self, node):
        """FEA solver inspector with study setup next to solver settings."""
        self._build_solver_study_definition(node, "fea")
        self._build_generic_ui(node)

    def _build_crash_solver_ui(self, node):
        """Crash solver inspector with study setup next to solver settings."""
        self._build_solver_study_definition(node, "crash")
        self._build_generic_ui(node)
