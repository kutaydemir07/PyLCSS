# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""TopologyInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging


import numpy as np
from PySide6 import QtCore, QtWidgets

from pylcss.design_studio.topology_optimization.configuration.presets import (
    INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
    INDUSTRIAL_WORKFLOW_MODES,
    industrial_topopt_defaults,
)

from .execution_workers import TopOptStepExportWorker

__all__ = ["TopologyInspectorMixin"]


class TopologyInspectorMixin:
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
            widget.setMinimumContentsLength(8)
            widget.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            item_texts = [str(item) for item in items]
            widget.addItems(item_texts)
            current = str(node.get_property(prop) or default or item_texts[0])
            idx = widget.findText(current)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            widget.currentTextChanged.connect(
                lambda v, p=prop: self.update_property(p, v)
            )
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

        def _refresh_topopt_later():
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        def _intent_combo(prop, items, default=None):
            widget = _combo(prop, items, default)

            def _changed(value, p=prop):
                self.update_property(p, value)
                # Guided mode translates engineering intent into conservative
                # numerical defaults. Expert mode must preserve the controls
                # the engineer explicitly set.
                if str(node.get_property("workflow_mode") or "Guided") != "Guided":
                    _refresh_topopt_later()
                    return
                goal = value if p == "design_goal" else node.get_property("design_goal")
                manufacturing = (
                    value
                    if p == "manufacturing_process"
                    else node.get_property("manufacturing_process")
                )
                settings = industrial_topopt_defaults(
                    goal,
                    "Automatic",
                    manufacturing,
                    nelx=node.get_property("nelx") or 30,
                    nely=node.get_property("nely") or 20,
                    nelz=node.get_property("nelz") or 10,
                )
                for key, setting in settings.items():
                    self.update_property(key, setting)
                _refresh_topopt_later()

            try:
                widget.currentTextChanged.disconnect()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            widget.currentTextChanged.connect(_changed)
            return widget

        intent_group = QtWidgets.QGroupBox("Design Intent")
        intent_layout = QtWidgets.QFormLayout()
        workflow_combo = _combo(
            "workflow_mode",
            list(INDUSTRIAL_WORKFLOW_MODES),
            "Guided",
        )

        def _workflow_changed(value):
            if str(value) == "Guided":
                settings = industrial_topopt_defaults(
                    node.get_property("design_goal"),
                    "Automatic",
                    node.get_property("manufacturing_process"),
                    nelx=node.get_property("nelx") or 30,
                    nely=node.get_property("nely") or 20,
                    nelz=node.get_property("nelz") or 10,
                )
                for key, setting in settings.items():
                    self.update_property(key, setting)
            _refresh_topopt_later()

        workflow_combo.currentTextChanged.connect(_workflow_changed)
        workflow_combo.setToolTip(
            "Guided derives grid/filter/numerical defaults from the selected "
            "engineering goal. Expert exposes those numerical controls."
        )
        intent_layout.addRow("Workflow:", workflow_combo)
        intent_layout.addRow(
            "Goal:",
            _intent_combo(
                "design_goal", INDUSTRIAL_DESIGN_GOALS, "Lightweight Stiffness"
            ),
        )
        intent_layout.addRow(
            "Manufacturing:",
            _intent_combo(
                "manufacturing_process", INDUSTRIAL_MANUFACTURING_PROCESSES, "None"
            ),
        )
        formulation_combo = _combo(
            "formulation",
            [
                "Density (SIMP)",
                "Level Set (Reaction-Diffusion)",
            ],
            "Density (SIMP)",
        )
        formulation_combo.currentTextChanged.connect(
            lambda _value: _refresh_topopt_later()
        )
        intent_layout.addRow("Formulation:", formulation_combo)
        if (
            str(node.get_property("formulation") or "Density (SIMP)")
            .lower()
            .startswith("level set")
        ):
            level_set_algorithm = QtWidgets.QLabel("Reaction-Diffusion")
            level_set_algorithm.setToolTip(
                "The level-set formulation evolves its signed interface with "
                "a volume-constrained reaction-diffusion update."
            )
            intent_layout.addRow("Interface Update:", level_set_algorithm)
        else:
            intent_layout.addRow(
                "Algorithm:",
                _combo(
                    "optimizer",
                    ["Auto", "OC", "MMA", "GCMMA", "Projected Gradient"],
                    "Auto",
                ),
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
        if (
            str(node.get_property("design_goal") or "").strip().lower()
            == "minimum mass under stress"
        ):
            intent_layout.addRow(
                "Allowable Stress (MPa):",
                _double(
                    "yield_stress",
                    250.0,
                    0.001,
                    1_000_000.0,
                    decimals=3,
                    step=10.0,
                ),
            )
        intent_layout.addRow("Material Budget:", material_container)

        intent_group.setLayout(intent_layout)
        self.props_layout.addWidget(intent_group)

        expert_mode = (
            str(node.get_property("workflow_mode") or "Guided").strip().lower()
            == "expert"
        )

        solve_group = QtWidgets.QGroupBox("Solve and Recovery")
        solve_layout = QtWidgets.QFormLayout()
        max_iter = _int("max_iter", 80, 1, 1000)
        max_iter.setToolTip(
            "Maximum, not requested, iteration count. The optimizer stops "
            "earlier when the convergence criterion remains satisfied."
        )
        solve_layout.addRow("Maximum Iterations:", max_iter)
        cutoff = _double("density_cutoff", 0.45, 0.01, 0.99, decimals=3, step=0.01)
        cutoff.setToolTip(
            "Manufacturing/recovery threshold. It does not change the solved "
            "density field; use the reported volume difference to judge it."
        )
        solve_layout.addRow("Recovery Cutoff:", cutoff)
        if expert_mode:
            solve_layout.addRow(
                "Relative Tolerance:",
                _double("tol", 0.01, 1.0e-6, 0.5, decimals=6, step=0.001),
            )
            solve_layout.addRow(
                "Convergence Patience:",
                _int("convergence_patience", 5, 1, 50),
            )
            grid = QtWidgets.QWidget()
            grid_layout = QtWidgets.QHBoxLayout(grid)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(4)
            for prop, default in (("nelx", 30), ("nely", 20), ("nelz", 10)):
                field = _int(prop, default, 1, 500)
                field.setPrefix(prop[-1].upper() + ": ")
                grid_layout.addWidget(field)
            grid.setToolTip(
                "Structured analysis cells along X, Y, and Z. Total cells and "
                "available memory limit practical resolution."
            )
            solve_layout.addRow("Voxel Grid:", grid)
            solve_layout.addRow(
                "Filter Radius (voxels):",
                _double("rmin", 1.5, 0.5, 20.0, decimals=2, step=0.1),
            )
            if (
                not str(node.get_property("formulation") or "Density (SIMP)")
                .lower()
                .startswith("level set")
            ):
                solve_layout.addRow(
                    "SIMP Penalization:",
                    _double("penal", 3.0, 1.0, 6.0, decimals=2, step=0.1),
                )
        solve_group.setToolTip(
            "A lattice or rib output is recovered after this solve and does "
            "not request a different optimization iteration count."
        )
        solve_group.setLayout(solve_layout)
        self.props_layout.addWidget(solve_group)

        def _connected_count(port_name):
            try:
                port = node.get_input(port_name)
                return len(port.connected_ports()) if port else 0
            except Exception:
                return 0

        setup_group = QtWidgets.QGroupBox("Study Definition")
        setup_layout = QtWidgets.QFormLayout(setup_group)
        setup_ports = [
            ("Design domain", "design_domain"),
            ("Material", "material"),
            ("Supports", "supports"),
            ("Forces", "loads"),
            ("Non-design regions", "non_design_regions"),
            ("Joints", "joints"),
            ("Operating cases", "load_cases"),
            ("Thermal sinks", "thermal_sinks"),
            ("Heat loads", "thermal_loads"),
        ]
        for label, port_name in setup_ports:
            count = _connected_count(port_name)
            status = QtWidgets.QLabel(
                f"{count} connected" if count else "Not connected"
            )
            status.setStyleSheet("color:#72d38a;" if count else "color:#ffb35c;")
            self._study_status_labels[port_name] = status
            setup_layout.addRow(label + ":", status)

        button_panel = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(button_panel)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_specs = [
            (
                "Add Support",
                "com.cad.topopt.support",
                "Topology Support",
                "supports",
                "supports",
            ),
            ("Add Force", "com.cad.topopt.load", "Topology Force", "loads", "loads"),
            (
                "Add Region",
                "com.cad.topopt.non_design_region",
                "Non-Design Region",
                "regions",
                "non_design_regions",
            ),
            ("Add Joint", "com.cad.topopt.joint", "Topology Joint", "joints", "joints"),
            (
                "Add Case",
                "com.cad.topopt.operating_case",
                "Operating Case",
                "load_case",
                "load_cases",
            ),
            (
                "Add Sink",
                "com.cad.topopt.thermal_sink",
                "Temperature Boundary",
                "thermal_sinks",
                "thermal_sinks",
            ),
            (
                "Add Heat",
                "com.cad.topopt.heat_load",
                "Heat Input",
                "thermal_loads",
                "thermal_loads",
            ),
        ]

        def _add_setup_node(spec):
            _button, node_id, label, output_name, input_name = spec
            app = self._get_main_app()
            if app is None:
                return
            app._batching_study_definition_edit = True
            try:
                x, y = node.pos()
            except Exception:
                x, y = (0.0, 0.0)
            created = app._spawn_node(
                node_id,
                label,
                x=float(x) - 330.0,
                y=float(y) + 120.0 * _connected_count(input_name),
            )
            if created is None:
                app._batching_study_definition_edit = False
                return
            try:
                created.get_output(output_name).connect_to(node.get_input(input_name))
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            self._finish_study_definition_edit(app, node)

        for index, spec in enumerate(button_specs):
            button = QtWidgets.QPushButton(spec[0])
            button.clicked.connect(
                lambda _checked=False, item=spec: _add_setup_node(item)
            )
            button_layout.addWidget(button, index // 2, index % 2)
        setup_layout.addRow(button_panel)
        setup_hint = QtWidgets.QLabel(
            "Select faces, edges, or vertices with Select Geometry "
            "(Interactive), then connect them "
            "to the support, force, joint, sink, or heat nodes. Connect a "
            "closed CAD solid to a Non-Design Region. A Joint uses two "
            "selected anchor regions. Operating Case nodes group "
            "independent load/support/joint combinations."
        )
        setup_hint.setWordWrap(True)
        setup_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
        setup_layout.addRow(setup_hint)
        self.props_layout.addWidget(setup_group)

        pipeline_group = QtWidgets.QGroupBox("Export")
        pipeline_layout = QtWidgets.QFormLayout()
        step_name = QtWidgets.QLineEdit(
            str(node.get_property("cad_export_filename") or "topology_optimized.step")
        )
        step_name.editingFinished.connect(
            lambda w=step_name: self.update_property("cad_export_filename", w.text())
        )
        pipeline_layout.addRow("STEP File:", step_name)
        btn_export_step = QtWidgets.QPushButton("Export STEP")
        btn_export_step.clicked.connect(lambda: self._export_topopt_step(node))
        pipeline_layout.addRow("Export:", btn_export_step)
        pipeline_group.setToolTip("Export the recovered topology shape after a run.")
        pipeline_group.setLayout(pipeline_layout)
        self.props_layout.addWidget(pipeline_group)

        mfg_group = QtWidgets.QGroupBox("Manufacturing")
        mfg_layout = QtWidgets.QFormLayout()
        is_level_set = (
            str(node.get_property("formulation") or "Density (SIMP)")
            .lower()
            .startswith("level set")
        )
        if not is_level_set and expert_mode:
            mfg_layout.addRow(
                "Symmetry:",
                _combo(
                    "symmetry", ["None", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"], "None"
                ),
            )
            mfg_layout.addRow(
                "Extrusion:", _combo("extrusion", ["None", "X", "Y", "Z"], "None")
            )
            mfg_layout.addRow(
                "Build Axis:",
                _combo(
                    "overhang_build_axis",
                    ["None", "+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                    "None",
                ),
            )
            mfg_layout.addRow(
                "Max Member Radius:",
                _double(
                    "max_member_size_voxels", 0.0, 0.0, 100.0, decimals=2, step=0.5
                ),
            )
            mfg_layout.addRow("Pattern Count:", _int("pattern_repeat", 1, 1, 64))
            mfg_layout.addRow(
                "Pattern Axis:", _combo("pattern_axis", ["X", "Y", "Z"], "Y")
            )
        structure_items = (
            ["Solid Envelope"]
            if is_level_set
            else [
                "Solid Envelope",
                "Topology-Following Ribs",
                "Gyroid Lattice",
                "Diamond Lattice",
                "Honeycomb Lattice",
                "Cubic Lattice",
                "Octet Truss Lattice",
            ]
        )
        structure_combo = _combo(
            "structure_mode",
            structure_items,
            "Solid Envelope",
        )
        structure_combo.currentTextChanged.connect(
            lambda _value: _refresh_topopt_later()
        )
        mfg_layout.addRow("Output Structure:", structure_combo)
        structure_mode = str(
            node.get_property("structure_mode") or "Solid Envelope"
        ).lower()
        if structure_mode != "solid envelope":
            mfg_layout.addRow(
                "Cell Size (voxels):",
                _double(
                    "structure_cell_size_voxels",
                    8.0,
                    3.0,
                    50.0,
                    decimals=2,
                    step=0.5,
                ),
            )
            mfg_layout.addRow(
                "Minimum Wall/Member:",
                _double(
                    "structure_member_thickness_voxels",
                    1.0,
                    0.25,
                    20.0,
                    decimals=2,
                    step=0.25,
                ),
            )
            mfg_layout.addRow(
                "Skin Thickness:",
                _double(
                    "structure_skin_thickness_voxels",
                    0.75,
                    0.0,
                    20.0,
                    decimals=2,
                    step=0.25,
                ),
            )
        if "lattice" in structure_mode:
            mfg_layout.addRow(
                "Optimize Local Density:",
                _check("lattice_variable_density"),
            )
            mfg_layout.addRow(
                "Minimum Relative Density:",
                _double(
                    "lattice_min_relative_density",
                    0.12,
                    0.01,
                    0.80,
                    decimals=3,
                    step=0.01,
                ),
            )
        print_ready = _check("print_ready_mesh")
        print_ready.stateChanged.connect(lambda _state: _refresh_topopt_later())
        mfg_layout.addRow("Repair / Smooth Mesh:", print_ready)
        if _get_bool("print_ready_mesh"):
            mfg_layout.addRow(
                "Retained Triangle Ratio:",
                _double(
                    "mesh_decimate_ratio",
                    1.0,
                    0.10,
                    1.0,
                    decimals=2,
                    step=0.05,
                ),
            )
            mfg_layout.addRow(
                "Maximum Relative Density:",
                _double(
                    "lattice_max_relative_density",
                    0.90,
                    0.20,
                    0.99,
                    decimals=3,
                    step=0.01,
                ),
            )
            mfg_layout.addRow(
                "Solid Transition:",
                _double(
                    "lattice_solid_transition_density",
                    0.92,
                    0.30,
                    1.0,
                    decimals=3,
                    step=0.01,
                ),
            )
        mfg_group.setToolTip(
            "Projection-style constraints applied after each density update: symmetry, extrusion, AM overhang, max member size, and repeated patterns."
        )
        mfg_group.setLayout(mfg_layout)
        self.props_layout.addWidget(mfg_group)

        view_group = QtWidgets.QGroupBox("Visualization")
        view_layout = QtWidgets.QFormLayout()

        validation_group = QtWidgets.QGroupBox("Validation and CAD")
        validation_layout = QtWidgets.QFormLayout()
        validate_check = _check("validate_after_optimize")
        validate_check.stateChanged.connect(lambda _state: _refresh_topopt_later())
        validation_layout.addRow("Validate after solve:", validate_check)
        if _get_bool("validate_after_optimize"):
            validation_layout.addRow(
                "Validation quality:",
                _combo(
                    "validation_quality", ["Standard", "Mesh Convergence"], "Standard"
                ),
            )
        validation_layout.addRow(
            "Build CAD after solve:", _check("generate_cad_after_optimize")
        )
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
        if not isinstance(result, dict) or result.get("density") is None:
            return result.get("recovered_shape") if isinstance(result, dict) else None
        try:
            import numpy as np
            from pylcss.design_studio.topology_optimization.geometry.surface_recovery import (
                _recover_voxel_shape,
            )
            from pylcss.design_studio.topology_optimization.manufacturing.structures import (
                structure_options_from_values,
            )

            bounds_payload = result.get("bounds")
            bounds = None
            if (
                isinstance(bounds_payload, dict)
                and "min" in bounds_payload
                and "max" in bounds_payload
            ):
                mins = np.asarray(bounds_payload["min"], dtype=float)
                maxs = np.asarray(bounds_payload["max"], dtype=float)
                if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                    bounds = (mins[:3], maxs[:3])

            passive_regions = result.get("passive_regions")
            if not isinstance(passive_regions, dict):
                passive_regions = {}
            structure_options = structure_options_from_values(
                node.get_property("structure_mode"),
                node.get_property("structure_cell_size_voxels"),
                node.get_property("structure_member_thickness_voxels"),
                node.get_property("structure_skin_thickness_voxels"),
                node.get_property("lattice_variable_density"),
                node.get_property("lattice_min_relative_density"),
                node.get_property("lattice_max_relative_density"),
                node.get_property("lattice_solid_transition_density"),
            )
            recovered = _recover_voxel_shape(
                np.asarray(result["density"], dtype=float),
                bounds,
                float(
                    result.get("density_cutoff")
                    or node.get_property("density_cutoff")
                    or 0.45
                ),
                print_ready=bool(node.get_property("print_ready_mesh")),
                decimate_ratio=float(node.get_property("mesh_decimate_ratio") or 1.0),
                solid_boxes=passive_regions.get("solid_boxes", ()),
                void_boxes=passive_regions.get("void_boxes", ()),
                solid_cylinders=passive_regions.get("solid_cylinders", ()),
                void_cylinders=passive_regions.get("void_cylinders", ()),
                joint_pin_cylinders=passive_regions.get("joint_pin_cylinders", ()),
                extrusion_axis=str(
                    result.get("extrusion_axis")
                    or node.get_property("extrusion")
                    or "none"
                ).lower(),
                source_mask=result.get("design_domain"),
                structure_options=structure_options,
                surface_backend=(
                    "legacy"
                    if str(node.get_property("surface_recovery_method") or "")
                    .lower()
                    .startswith("legacy")
                    else "vtk_sdf"
                ),
            )
            if recovered is not None and len(recovered.get("faces", [])) > 0:
                result["recovered_shape"] = recovered
                setattr(node, "_last_result", result)
                return recovered
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        return result.get("recovered_shape")

    def _export_topopt_step(self, node):
        """Reconstruct and export the topology result as a STEP body."""
        worker = getattr(self, "_topopt_step_export_worker", None)
        if worker is not None and worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("STEP export already running...")
            return

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") not in {"topopt_voxel"}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Topology Result",
                "Run topology optimisation before exporting STEP.",
            )
            return

        default_name = str(
            node.get_property("cad_export_filename") or "topology_optimized.step"
        )
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
                    "joint_pin_cylinders": list(getattr(bc, "joint_pin_cylinders", ())),
                }
            except Exception:
                passive_regions = {}

        node.set_property("cad_reconstruction_method", "Recovered Shape")
        cutoff = float(
            result.get("density_cutoff") or node.get_property("density_cutoff") or 0.45
        )
        extrusion_axis = (
            str(
                result.get("extrusion_axis") or node.get_property("extrusion") or "none"
            )
            .strip()
            .lower()
        )

        node.set_property("cad_export_filename", path)
        worker = TopOptStepExportWorker(
            result,
            path,
            density_cutoff=cutoff,
            extrusion_axis=extrusion_axis,
            passive_regions=passive_regions,
            parent=self,
        )
        self._topopt_step_export_worker = worker

        def _finish(export_path):
            self._topopt_step_export_worker = None
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported CAD STEP to {export_path}"
                )

        def _fail(message):
            self._topopt_step_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", message)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting CAD STEP in background...")
        worker.start()

    def _export_topopt_stl(self, node):
        """Export recovered shape from topology optimisation as binary STL."""
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "No Shape",
                "Run topology optimisation first — no recovered shape available.",
            )
            return
        recovered = self._refresh_topopt_recovered_shape(node, result)
        if recovered is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Shape",
                "Run topology optimisation first - no recovered shape available.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)"
        )
        if not path:
            return
        try:
            import numpy as np

            verts = np.asarray(recovered["vertices"], dtype=float)
            faces = np.asarray(recovered["faces"], dtype=int)
            try:
                import trimesh

                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
                tm.export(path, file_type="stl")
                if hasattr(self.window(), "statusBar") and self.window().statusBar():
                    self.window().statusBar().showMessage(
                        f"Exported {len(tm.faces)} triangles to {path}"
                    )
                return
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

            from stl import mesh as stl_mesh  # numpy-stl fallback

            stl_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    stl_obj.vectors[i][j] = verts[f[j]]
            stl_obj.save(path)
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported {len(faces)} triangles to {path}"
                )
        except ImportError:
            # Fallback: raw binary STL without numpy-stl
            self._write_binary_stl(path, recovered)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _write_binary_stl(self, path, shape_data):
        """Write binary STL without numpy-stl dependency."""
        import struct

        verts = shape_data["vertices"]
        faces = shape_data["faces"]
        with open(path, "wb") as f:
            f.write(b"\x00" * 80)  # header
            f.write(struct.pack("<I", len(faces)))
            for face in faces:
                v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                # compute normal
                e1 = np.array(v1) - np.array(v0)
                e2 = np.array(v2) - np.array(v0)
                n = np.cross(e1, e2)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n = n / norm
                f.write(struct.pack("<3f", *n))
                f.write(struct.pack("<3f", *v0))
                f.write(struct.pack("<3f", *v1))
                f.write(struct.pack("<3f", *v2))
                f.write(struct.pack("<H", 0))
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage(
                f"Exported {len(faces)} triangles to {path}"
            )
