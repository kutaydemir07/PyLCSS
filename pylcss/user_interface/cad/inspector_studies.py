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
            "Run the graph (CAD only — skips FEA/impact) and render this part."
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
        from pylcss.design_studio.crash.materials import CRASH_MATERIAL_PRESETS
        from pylcss.design_studio.fem._helpers import MATERIAL_DATABASE

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
                    "thermal_conductivity",
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
        if not is_impact:
            field_specs.append(
                (
                    "thermal_conductivity",
                    "k",
                    "Thermal conductivity (W/m K)",
                )
            )
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

    def _build_fea_solver_ui(self, node):
        """FEA solver inspector: solver settings only, wired up in the graph."""
        self._build_generic_ui(node)

    def _build_crash_solver_ui(self, node):
        """Impact solver inspector: solver settings only, wired up in the graph."""
        self._build_generic_ui(node)
