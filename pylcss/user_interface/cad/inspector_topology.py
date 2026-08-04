# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""TopologyInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging


from PySide6 import QtCore, QtWidgets

from pylcss.design_studio.topology_optimization.manufacturing import (
    PUBLIC_FAMILY_KEYS,
    normalize_family_key,
)
from pylcss.design_studio.topology_optimization.manufacturing.structures import (
    PUBLIC_LATTICE_FAMILY_NAMES,
)
from pylcss.design_studio.topology_optimization.integration.lattice_infill_node import (
    AUTOMATIC_FINENESS_FRACTIONS,
)
from pylcss.design_studio.topology_optimization.integration.lattice_settings import (
    guided_lattice_voxel_dimensions,
    lattice_dimension_range_text,
    lattice_setup_is_guided,
    node_minimum_feature_size,
)
from pylcss.design_studio.topology_optimization.integration.voxelization import (
    voxel_size_from_bounds,
)
from pylcss.design_studio.topology_optimization.configuration.presets import (
    EXTRUDED_MANUFACTURING_PROCESSES as _EXTRUDED_PROCESSES,
    INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
    INDUSTRIAL_WORKFLOW_MODES,
    LATTICE_DESIGN_GOALS,
    industrial_topopt_defaults,
)

from .execution_workers import (
    TopOptCadPreviewWorker,
    TopOptMeshExportWorker,
    TopOptStepExportWorker,
)

__all__ = ["TopologyInspectorMixin"]


class TopologyInspectorMixin:
    def _build_topopt_definition_ui(self, node):
        """Engineer-facing inspector for TopOpt study-definition nodes.

        These nodes are intentionally small. Numerical penalty parameters stay
        program-controlled; the panel exposes physical intent, units, and
        connection status instead of raw backing-property names.
        """

        def _connected(port_name):
            try:
                port = node.get_input(port_name)
                return len(port.connected_ports()) if port is not None else 0
            except Exception:
                return 0

        def _connected_entity_types(port_name):
            kinds = []
            try:
                port = node.get_input(port_name)
                connected = list(port.connected_ports()) if port is not None else []
            except Exception:
                connected = []
            for connected_port in connected:
                try:
                    upstream = connected_port.node()
                    kind = str(
                        upstream.get_property("entity_type") or ""
                    ).title()
                except Exception:
                    continue
                if kind in {"Face", "Edge", "Vertex"} and kind not in kinds:
                    kinds.append(kind)
            return kinds

        def _combo(prop, items):
            widget = QtWidgets.QComboBox()
            widget.addItems(items)
            current = str(node.get_property(prop) or items[0])
            index = widget.findText(current)
            widget.setCurrentIndex(index if index >= 0 else 0)
            widget.currentTextChanged.connect(
                lambda value, name=prop: self.update_property(name, value)
            )
            return widget

        def _number(prop, default, *, minimum=-1.0e12, maximum=1.0e12):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(float(minimum), float(maximum))
            widget.setValue(float(node.get_property(prop) or default))
            widget.valueChanged.connect(
                lambda value, name=prop: self.update_property(name, value)
            )
            return widget

        def _text(prop, default):
            widget = QtWidgets.QLineEdit(str(node.get_property(prop) or default))
            widget.editingFinished.connect(
                lambda name=prop, field=widget: self.update_property(
                    name, field.text()
                )
            )
            return widget

        def _check(prop, label):
            widget = QtWidgets.QCheckBox(label)
            widget.setChecked(bool(node.get_property(prop)))
            widget.toggled.connect(
                lambda value, name=prop: self.update_property(name, bool(value))
            )
            return widget

        node_class = node.__class__.__name__
        definition = QtWidgets.QGroupBox("Setup")
        layout = QtWidgets.QFormLayout(definition)

        input_specs = {
            "TopologySupportNode": [("Selected interface", "target_region")],
            "TopologyLoadNode": [("Selected load region", "target_region")],
        }.get(node_class, [])
        for label, port_name in input_specs:
            count = _connected(port_name)
            entity_types = _connected_entity_types(port_name)
            kind_text = (
                " · " + "/".join(kind.lower() for kind in entity_types)
                if entity_types
                else ""
            )
            status = QtWidgets.QLabel(
                f"{count} connected{kind_text}" if count else "Not connected"
            )
            singular_topology_interface = (
                node_class in {"TopologySupportNode", "TopologyLoadNode"}
                and any(kind != "Face" for kind in entity_types)
            )
            status.setStyleSheet(
                "color:#ffb35c;"
                if not count or singular_topology_interface
                else "color:#72d38a;"
            )
            if singular_topology_interface:
                status.setToolTip(
                    "Edges and vertices have zero physical area. The voxel "
                    "solver must inflate them into a mesh-sized contact pad, "
                    "which can create artificial branches. Select the real "
                    "load/support face for a production topology study."
                )
            layout.addRow(label + ":", status)

        if node_class == "TopologySupportNode":
            layout.addRow(
                "Constraint:",
                _combo(
                    "support_type",
                    [
                        "Fixed",
                        "Block X Translation",
                        "Block Y Translation",
                        "Block Z Translation",
                        "Block XY Translation",
                        "Block YZ Translation",
                        "Block XZ Translation",
                    ],
                ),
            )
            note = (
                "Fixed locks X, Y, and Z. Use the physical attachment face. "
                "Edge/point supports are mesh-regularized and are intended only "
                "for academic benchmarks."
            )
        elif node_class == "TopologyLoadNode":
            for prop, label in (
                ("force_x", "Fx [N]"),
                ("force_y", "Fy [N]"),
                ("force_z", "Fz [N]"),
            ):
                layout.addRow(label + ":", _number(prop, 0.0))
            note = (
                "Total force is distributed across the selected face. Use the "
                "real load-introduction face; an edge or point is singular and "
                "can seed non-physical branching."
            )
        else:
            note = ""

        if note:
            hint = QtWidgets.QLabel(note)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#8f98a5; font-size:10px;")
            layout.addRow(hint)
        self.props_layout.addWidget(definition)

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
            widget.setMinimumWidth(0)
            widget.setMinimumContentsLength(1)
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
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _mapped_combo(prop, labels, default):
            widget = QtWidgets.QComboBox()
            widget.setMinimumWidth(0)
            widget.setMinimumContentsLength(1)
            widget.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            reverse = {display: value for value, display in labels.items()}
            widget.addItems(list(reverse))
            current = str(node.get_property(prop) or default)
            widget.setCurrentText(labels.get(current, current))
            widget.currentTextChanged.connect(
                lambda display, p=prop: self.update_property(
                    p,
                    reverse.get(display, display),
                )
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _check(prop, label=""):
            widget = QtWidgets.QCheckBox(label)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding if label else QtWidgets.QSizePolicy.Fixed,
                QtWidgets.QSizePolicy.Fixed,
            )
            widget.setChecked(_get_bool(prop))
            widget.stateChanged.connect(
                lambda state, p=prop: self.update_property(p, bool(state))
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _double(prop, default, lo, hi, decimals=3, step=0.1, suffix=""):
            widget = QtWidgets.QDoubleSpinBox()
            # A spin box formats through the system locale, so on a comma-
            # decimal machine "5.000" is drawn as "5,000" and reads as five
            # thousand. Millimetres and metres are three orders apart, which is
            # not a misreading anyone recovers from. Force the dot and state
            # the unit in the box itself.
            widget.setLocale(QtCore.QLocale(QtCore.QLocale.C))
            widget.setRange(float(lo), float(hi))
            widget.setDecimals(int(decimals))
            widget.setSingleStep(float(step))
            if suffix:
                widget.setSuffix(suffix)
            widget.setValue(_get_float(prop, default))
            widget.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _int(prop, default, lo, hi):
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(lo), int(hi))
            widget.setValue(_get_int(prop, default))
            widget.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _refresh_topopt_later():
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n) if self.current_node is n else None,
            )

        guided_preserved_properties = {
            "structure_mode",
            "lattice_settings_mode",
            "structure_cell_size_voxels",
            "structure_member_thickness_voxels",
            "structure_skin_thickness_voxels",
            # Manufacturing capabilities, not preset-derived numerics: a preset
            # change must not silently resize the lattice or the length scale.
            "lattice_cell_size_mm",
            "lattice_member_thickness_mm",
            "lattice_skin_thickness_mm",
            "minimum_member_size_mm",
            "minimum_void_size_mm",
            "maximum_member_size_mm",
            # The load envelope is an engineering decision about the load
            # cases, so a fidelity or process preset must not silently reset it.
            "load_aggregation",
            "lattice_target_relative_density",
            "lattice_variable_density",
            "lattice_min_relative_density",
            "lattice_max_relative_density",
            "lattice_solid_transition_density",
            "lattice_porosity",
            "optimize_lattice_members",
            "lattice_max_member_thickness_voxels",
            "lattice_member_sizing_iterations",
            "lattice_buckling_length_factor",
            "exclusion_scope",
            "exclusion_thickness_mode",
            "exclusion_thickness_mm",
        }

        def _apply_guided_settings(settings):
            for key, setting in settings.items():
                if key not in guided_preserved_properties:
                    self.update_property(key, setting)

        def _intent_combo(prop, items, default=None, display_labels=None):
            labels = display_labels or {}
            reverse_labels = {label: value for value, label in labels.items()}
            if labels:
                widget = QtWidgets.QComboBox()
                widget.setMinimumWidth(0)
                widget.setMinimumContentsLength(1)
                widget.setSizeAdjustPolicy(
                    QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
                )
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Fixed,
                )
                widget.addItems([labels.get(str(item), str(item)) for item in items])
                current = str(node.get_property(prop) or default or items[0])
                widget.setCurrentText(labels.get(current, current))
                tooltip = self._PROPERTY_TOOLTIPS.get(prop)
                if tooltip:
                    widget.setToolTip(str(tooltip))
            else:
                widget = _combo(prop, items, default)

            def _changed(value, p=prop):
                internal_value = reverse_labels.get(value, value)
                self.update_property(p, internal_value)
                # Guided mode translates engineering intent into conservative
                # numerical defaults. Expert mode must preserve the controls
                # the engineer explicitly set.
                if str(node.get_property("workflow_mode") or "Guided") != "Guided":
                    _refresh_topopt_later()
                    return
                goal = (
                    internal_value
                    if p == "design_goal"
                    else node.get_property("design_goal")
                )
                manufacturing = (
                    internal_value
                    if p == "manufacturing_process"
                    else node.get_property("manufacturing_process")
                )
                settings = industrial_topopt_defaults(
                    goal,
                    manufacturing,
                    nelx=node.get_property("nelx") or 30,
                    nely=node.get_property("nely") or 20,
                    nelz=node.get_property("nelz") or 10,
                )
                _apply_guided_settings(settings)
                _refresh_topopt_later()

            if not labels:
                try:
                    widget.currentTextChanged.disconnect()
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            widget.currentTextChanged.connect(_changed)
            return widget

        expert_mode = (
            str(node.get_property("workflow_mode") or "Guided").strip().lower()
            == "expert"
        )

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
                    node.get_property("manufacturing_process"),
                    nelx=node.get_property("nelx") or 30,
                    nely=node.get_property("nely") or 20,
                    nelz=node.get_property("nelz") or 10,
                )
                _apply_guided_settings(settings)
            _refresh_topopt_later()

        workflow_combo.currentTextChanged.connect(_workflow_changed)
        workflow_combo.setToolTip(
            "Guided derives the grid/filter/numerical controls from the "
            "selected fidelity and engineering goal. Expert exposes the raw "
            "numerical controls."
        )
        intent_layout.addRow("Workflow:", workflow_combo)
        # No fidelity row. The grid follows the study's physical feature sizes
        # and the convergence budget is fixed, so there is nothing here for a
        # user to trade off -- the one control that sets resolution is the
        # minimum member size, which is a manufacturing capability.
        lattice_study = normalize_family_key(
            node.get_property("structure_mode")
        ) not in {"", "solid"}
        design_goals = (
            LATTICE_DESIGN_GOALS if lattice_study else INDUSTRIAL_DESIGN_GOALS
        )
        intent_layout.addRow(
            "Objective:",
            _intent_combo(
                "design_goal",
                design_goals,
                "Lightweight Stiffness",
                {
                    "Lightweight Stiffness": "Stiffness / Weight",
                    "Minimum Mass Under Stress": "Stress-Limited Mass",
                },
            ),
        )
        goal_now = str(node.get_property("design_goal") or "").strip().lower()
        if goal_now:
            # The load envelope is its own decision, not a goal name. "Worst
            # Case" minimizes a smooth P-norm over the cases so no single case
            # exceeds the bound; "Weighted Sum" minimizes their average, where a
            # dominant case can erase a path another one needs.
            aggregation_row = _combo(
                "load_aggregation",
                ["Weighted Sum", "Worst Case"],
                "Weighted Sum",
            )
            aggregation_row.setToolTip(
                "How several load cases are combined into one objective. "
                "Weighted Sum minimizes the weighted average compliance. "
                "Worst Case minimizes a smooth maximum, so a rare severe case "
                "is bounded instead of averaged away."
            )
            intent_layout.addRow("Load Aggregation:", aggregation_row)
        intent_layout.addRow(
            "Process:",
            _intent_combo(
                "manufacturing_process",
                INDUSTRIAL_MANUFACTURING_PROCESSES,
                "None",
                {
                    "None": "Unrestricted",
                    "Additive": "Additive (3D Print)",
                    "Cast / Moulded": "Cast / Moulded",
                    "Extruded": "Extruded Profile",
                    "Symmetric": "Symmetry Only",
                    "Additive + Symmetric": "Additive + Symmetry",
                    "Extruded + Symmetric": "Extruded + Symmetry",
                },
            ),
        )
        process = str(node.get_property("manufacturing_process") or "None")
        process_explanations = {
            "None": (
                "Unrestricted shape: no process-specific direction or symmetry "
                "constraint is added."
            ),
            "Additive": (
                "Suppresses unsupported material relative to the selected print "
                "build direction."
            ),
            "Cast / Moulded": (
                "Removes undercuts so the mould can withdraw along the selected "
                "pull direction."
            ),
            "Extruded": (
                "Keeps one constant cross-section along the selected extrusion "
                "direction."
            ),
            "Symmetric": (
                "Mirrors the material layout across the selected global centre "
                "plane or planes."
            ),
            "Additive + Symmetric": (
                "Combines print overhang control with mirrored material layout."
            ),
            "Extruded + Symmetric": (
                "One constant cross-section plus a mirrored material layout. "
                "This is the combination that yields a symmetric part with an "
                "editable CAD body."
            ),
        }
        process_hint = QtWidgets.QLabel(process_explanations.get(process, ""))
        process_hint.setWordWrap(True)
        process_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
        intent_layout.addRow(process_hint)
        # Professional manufacturing-method pickers always ask for the
        # direction/plane that gives the method physical meaning.  Guided mode
        # keeps the numerical implementation program-controlled, but it must
        # not silently assume +Y printing, +Z mould withdrawal, or a Z profile.
        if not expert_mode and process in {"Additive", "Additive + Symmetric"}:
            intent_layout.addRow(
                "Build Direction:",
                _combo(
                    "overhang_build_axis",
                    ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                    "+Y",
                ),
            )
            intent_layout.addRow(
                "Self-Support Angle:",
                _double(
                    "overhang_angle_deg",
                    45.0,
                    1.0,
                    89.0,
                    decimals=1,
                    step=1.0,
                ),
            )
        if not expert_mode and process == "Cast / Moulded":
            intent_layout.addRow(
                "Mould Pull Direction:",
                _combo(
                    "pull_out_axis",
                    ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                    "+Z",
                ),
            )
        if not expert_mode:
            # Offered for every process, not only "Extruded". An extrusion axis
            # is what makes the body a swept profile, and a swept profile is the
            # only solid with an exact editable B-rep, so hiding this row was
            # hiding the one control that decides whether a study can produce
            # CAD at all. "None" stays available: an unconstrained load path is
            # a legitimate result, delivered as its recovered surface.
            extrusion_row = _combo(
                "extrusion",
                ["None", "X", "Y", "Z"],
                "Z" if process in _EXTRUDED_PROCESSES else "None",
            )
            extrusion_row.setToolTip(
                "Keeps one constant cross-section along this axis. Required for "
                "an editable CAD/STEP body: without it the optimized shape is "
                "delivered as its recovered surface (viewable, STL-exportable)."
            )
            intent_layout.addRow("Extrusion Direction:", extrusion_row)
        if not expert_mode and process in {
            "Symmetric",
            "Additive + Symmetric",
            "Extruded + Symmetric",
        }:
            intent_layout.addRow(
                "Symmetry Plane(s):",
                _mapped_combo(
                    "symmetry",
                    {
                        "X": "YZ (normal X)",
                        "Y": "XZ (normal Y)",
                        "Z": "XY (normal Z)",
                        "XY": "YZ + XZ",
                        "XZ": "YZ + XY",
                        "YZ": "XZ + XY",
                        "XYZ": "All three",
                    },
                    "Z",
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
        # Density-versus-level-set is a solver formulation, not design intent. No
        # commercial guided workflow asks for it, and the guided defaults already
        # pick the density formulation every preset was calibrated on.
        if expert_mode:
            intent_layout.addRow("Formulation:", formulation_combo)
        else:
            formulation_combo.setParent(None)
        # The update scheme is a solver choice, not design intent: no commercial
        # guided workflow offers OC against MMA against GCMMA, and the guided
        # defaults already pick the one each design goal was calibrated with.
        if expert_mode:
            intent_layout.addRow(
                "Algorithm:",
                _combo(
                    "optimizer",
                    ["Auto", "OC", "MMA", "GCMMA"],
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
        intent_layout.addRow("Target Volume:", material_container)
        # Physical length scale is a manufacturing requirement, not a numerical
        # expert knob. Industrial guided workflows expose minimum thickness in
        # model units and derive the filter/grid from it.
        minimum_member = _double(
            "minimum_member_size_mm",
            0.0,
            0.0,
            10_000.0,
            decimals=2,
            step=0.5,
            suffix=" mm",
        )
        minimum_member.setSpecialValueText("Auto (from grid)")
        minimum_member.setToolTip(
            "Thinnest rib or wall the design may contain, in millimetres.\n\n"
            "This is a manufacturing capability, not a solver setting: the "
            "thinnest section you can actually weld, cast or print. It also "
            "sets the analysis resolution, because the grid has to resolve it.\n\n"
            "Set it to 0 for Auto, which uses three grid cells -- a numerical "
            "floor, not a manufacturing one, so a released design should state "
            "the real value."
        )
        solid_envelope = (
            normalize_family_key(node.get_property("structure_mode")) == "solid"
        )
        if solid_envelope:
            intent_layout.addRow("Minimum Member Size:", minimum_member)
        minimum_void = _double(
            "minimum_void_size_mm",
            0.0,
            0.0,
            10_000.0,
            decimals=2,
            step=0.5,
            suffix=" mm",
        )
        minimum_void.setSpecialValueText("Same as member size")
        minimum_void.setToolTip(
            "Narrowest gap or hole the design may contain, in millimetres.\n\n"
            "The counterpart of the minimum member size: that one stops "
            "material getting too thin, this one stops the space between two "
            "members getting too narrow. They are separate because processes "
            "differ -- a cutter, a core, or a drain path needs room to reach "
            "in, and that clearance is rarely the same number as the wall "
            "thickness.\n\n"
            "Leave it at 0 to reuse the minimum member size."
        )
        if solid_envelope and expert_mode:
            intent_layout.addRow("Minimum Gap / Hole Size:", minimum_void)
        maximum_member = _double(
            "maximum_member_size_mm",
            0.0,
            0.0,
            10_000.0,
            decimals=2,
            step=0.5,
            suffix=" mm",
        )
        maximum_member.setSpecialValueText("No limit")
        maximum_member.setToolTip(
            "Thickest solid section the design may contain, in millimetres.\n\n"
            "Caps chunky blocks so material spreads into ribs instead. Useful "
            "for casting, where a thick section cools unevenly, and for "
            "welding, where plate stock has an upper gauge.\n\n"
            "Leave it at 0 for no limit."
        )
        if expert_mode:
            intent_layout.addRow("Maximum Member Size:", maximum_member)

        intent_group.setLayout(intent_layout)
        self.props_layout.addWidget(intent_group)

        resolution_group = QtWidgets.QGroupBox("Analysis Resolution")
        resolution_layout = QtWidgets.QFormLayout()
        # Guided and Expert workflows both keep iteration stopping and shape
        # recovery program-controlled; only analysis resolution is editable.
        physics = str(node.get_property("physics_mode") or "Structural").lower()
        # A reduced-order volumetric rejection term that explicitly does not
        # resolve wetted area or flow. Zero is the honest default for a guided
        # study; a non-zero value needs conjugate CFD to justify.
        if expert_mode and "thermal" in physics:
            convection = _double(
                "convection_coefficient", 0.0, 0.0, 1.0, decimals=6, step=1e-4
            )
            convection.setToolTip(
                "Reduced-order distributed heat rejection per unit material "
                "volume, in W/(mm³·K).\n\n"
                "Leave at 0 for pure conduction. This term does not resolve "
                "wetted surface area or fluid flow; validate physical heat "
                "sinks with conjugate CFD."
            )
            resolution_layout.addRow("Volumetric Heat Rejection:", convection)
        if expert_mode:
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
            resolution_layout.addRow("Voxel Grid:", grid)
            resolution_layout.addRow(
                "Filter Radius (voxels):",
                _double("rmin", 1.5, 0.5, 20.0, decimals=2, step=0.1),
            )
            if (
                not str(node.get_property("formulation") or "Density (SIMP)")
                .lower()
                .startswith("level set")
            ):
                resolution_layout.addRow(
                    "SIMP Penalization:",
                    _double("penal", 3.0, 1.0, 6.0, decimals=2, step=0.1),
                )
        if expert_mode:
            # A convergence study repeats the optimization on finer grids. It
            # is valuable verification, but it is an advanced solve policy—not
            # part of the short guided study definition.
            topology_convergence = _check(
                "topology_convergence_enabled",
                "Repeat optimization on successively finer grids",
            )
            topology_convergence.stateChanged.connect(
                lambda _state: _refresh_topopt_later()
            )
            resolution_layout.addRow(
                "Resolution convergence:", topology_convergence
            )
            if _get_bool("topology_convergence_enabled"):
                resolution_layout.addRow(
                    "Topology levels:",
                    _int("topology_convergence_levels", 3, 2, 3),
                )
        resolution_group.setToolTip(
            "Explicit discretization controls for expert studies."
        )
        resolution_group.setLayout(resolution_layout)
        if expert_mode:
            self.props_layout.addWidget(resolution_group)

        setup_group = QtWidgets.QGroupBox("Protected Interfaces")
        setup_layout = QtWidgets.QFormLayout(setup_group)
        exclusion_scope = _mapped_combo(
            "exclusion_scope",
            {
                "All Loads and Supports": "Loads + Supports",
                "Loads Only": "Forces",
                "Supports Only": "Supports",
                "None": "None",
            },
            "All Loads and Supports",
        )
        setup_layout.addRow("Keep Material At:", exclusion_scope)
        protected_hint = QtWidgets.QLabel(
            "Retains a program-controlled collar around selected load and support "
            "interfaces. Use a Preserved / Excluded Region node for any other "
            "solid or void that must remain unchanged."
        )
        protected_hint.setWordWrap(True)
        protected_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
        setup_layout.addRow(protected_hint)
        exclusion_mode = _mapped_combo(
            "exclusion_thickness_mode",
            {"Program Controlled": "Auto", "Manual": "Manual"},
            "Program Controlled",
        )
        exclusion_mode.currentTextChanged.connect(
            lambda _value: _refresh_topopt_later()
        )
        # Which interfaces to preserve is design intent and stays visible. How
        # thick the preserved collar is, is a voxel-derived number the program
        # already picks from the resolved grid.
        if expert_mode:
            setup_layout.addRow("Preservation Thickness:", exclusion_mode)
        if expert_mode and (
            str(
                node.get_property("exclusion_thickness_mode")
                or "Program Controlled"
            )
            == "Manual"
        ):
            setup_layout.addRow(
                "Manual Thickness (mm):",
                _double(
                    "exclusion_thickness_mm",
                    2.0,
                    0.001,
                    1_000_000.0,
                    decimals=3,
                    step=0.25,
                ),
            )
        elif expert_mode:
            last_result = getattr(node, "_last_result", None)
            passive = (
                last_result.get("passive_regions")
                if isinstance(last_result, dict)
                else None
            )
            effective = (
                passive.get("automatic_exclusion_thickness")
                if isinstance(passive, dict)
                else None
            )
            value_text = (
                f"{float(effective):.3g} mm on the last grid"
                if effective is not None
                else "2 average voxel lengths at run time"
            )
            effective_label = QtWidgets.QLabel(value_text)
            effective_label.setWordWrap(True)
            setup_layout.addRow("Effective Thickness:", effective_label)
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
        export_panel = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(export_panel)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(4)

        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)

        btn_export_step = QtWidgets.QPushButton("Export STEP")
        btn_export_step.setToolTip("Solid B-rep body, for CAD and CAM.")
        btn_export_step.clicked.connect(lambda: self._export_topopt_step(node))
        row1.addWidget(btn_export_step)

        btn_export_stl = QtWidgets.QPushButton("Export STL")
        btn_export_stl.setToolTip(
            "Solid envelope: tessellation of the same CAD body used by STEP. "
            "Lattice: triangulation of the manufactured mesh."
        )
        btn_export_stl.clicked.connect(lambda: self._export_topopt_stl(node))
        row1.addWidget(btn_export_stl)

        export_layout.addLayout(row1)

        btn_export_3mf = QtWidgets.QPushButton("Export 3MF Lattice")
        btn_export_3mf.setToolTip(
            "Strut lattices only.\n\n"
            "Writes the lattice as nodes and beams with per-end radii, in the "
            "3MF Beam Lattice format the AM tool chain reads natively. It "
            "stores the lattice as a lattice instead of as a tessellation of "
            "one, so it is exact, a few hundred kilobytes, and effectively "
            "instant — where the same body as STEP is megabytes and as STL is "
            "hundreds of thousands of triangles."
        )
        btn_export_3mf.clicked.connect(lambda: self._export_topopt_beam_3mf(node))
        export_layout.addWidget(btn_export_3mf)
        pipeline_layout.addRow("Shape:", export_panel)
        pipeline_group.setToolTip(
            "A solid envelope exports one reconstructed CAD body as STEP or "
            "as a tessellated STL. BCC and Octet are reconstructed exactly "
            "from their centrelines, so they export as STEP as well, and as "
            "native 3MF Beam Lattice preserving their nodes, beams and end "
            "radii. A minimal-surface or honeycomb result has no compact "
            "B-rep and exports as STL."
        )
        pipeline_group.setLayout(pipeline_layout)

        mfg_group = QtWidgets.QGroupBox("Manufacturing")
        mfg_layout = QtWidgets.QFormLayout()
        if expert_mode:
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
                "Overhang Angle:",
                _double(
                    "overhang_angle_deg",
                    45.0,
                    1.0,
                    89.0,
                    decimals=1,
                    step=1.0,
                ),
            )
            mfg_layout.addRow(
                "Pull-Out Direction:",
                _combo(
                    "pull_out_axis",
                    ["None", "+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                    "None",
                ),
            )
            mfg_layout.addRow(
                "Legacy Max Radius:",
                _double(
                    "max_member_size_voxels", 0.0, 0.0, 100.0, decimals=2, step=0.5
                ),
            )
            mfg_layout.addRow("Pattern Count:", _int("pattern_repeat", 1, 1, 64))
            mfg_layout.addRow(
                "Pattern Axis:", _combo("pattern_axis", ["X", "Y", "Z"], "Y")
            )
        from pylcss.design_studio.topology_optimization.manufacturing import (
            FAMILIES as _LATTICE_FAMILIES,
        )

        # Only the lattice study declares a cell family. The topology study
        # builds a solid envelope and has no structure to pick, so it does not
        # get a one-entry dropdown that reads like a choice.
        is_lattice_study = node.has_property("structure_mode")
        if is_lattice_study:
            # Keep the new-study picker focused on the production core. The
            # full registry remains supported by the builders for saved legacy
            # studies.
            structure_items = [
                _LATTICE_FAMILIES[key].display_name
                for key in PUBLIC_FAMILY_KEYS
                if key in _LATTICE_FAMILIES
            ]
            current_structure = str(node.get_property("structure_mode") or "")
            # Do not silently rewrite a saved study that used a retired
            # catalogue entry. Show its exact value only while that project is
            # open.
            if (
                current_structure
                and current_structure not in structure_items
                and normalize_family_key(current_structure) in _LATTICE_FAMILIES
            ):
                structure_items.append(current_structure)
            structure_combo = _combo(
                "structure_mode",
                structure_items,
                structure_items[0],
            )
            structure_combo.currentTextChanged.connect(
                lambda _value: _refresh_topopt_later()
            )
            structure_combo.setToolTip(
                "The unit cell the optimized load paths are built from.\n\n"
                "Sheet TPMS — Gyroid is the general-purpose default; Schwarz "
                "Primitive provides straight channels for flow and heat-"
                "transfer studies. Printability and powder evacuation still "
                "depend on wall thickness, orientation, process and part "
                "boundary.\n\n"
                "Strut lattices — BCC is compliant, widely printed and has no "
                "horizontal cell members; Octet is stretch dominated, close to "
                "isotropic and the structural reference. Octet can have member "
                "diameters sized against axial stress and buckling; both export "
                "as native 3MF beam lattices.\n\n"
                "Honeycomb — prismatic cells along one axis; stiff in that "
                "direction, compliant across it. Diamond and the other retired "
                "specialist cells remain loadable in projects created by "
                "earlier PyLCSS versions."
            )
            mfg_layout.addRow("Cell Family:", structure_combo)
        structure_mode = str(
            node.get_property("structure_mode") or "Solid Envelope"
        ).lower()
        # Ask the registry which family this is. Testing the display string for
        # the word "lattice" used to be enough; it stopped being enough the
        # moment a family was called "Gyroid Network (Skeletal)".
        structure_family = _LATTICE_FAMILIES.get(
            normalize_family_key(node.get_property("structure_mode"))
        )
        is_lattice_structure = structure_family is not None
        # A strut lattice is an exact B-rep -- one quadric per member, one ball
        # per joint -- so STEP is offered for it on the same footing as the
        # solid envelope. Only the families with no compact boundary
        # representation (minimal surfaces, honeycomb) lose the button.
        has_brep_export = (
            structure_family is None or structure_family.is_strut
        )
        btn_export_step.setEnabled(has_brep_export)
        step_name.setEnabled(has_brep_export)
        btn_export_step.setToolTip(
            "Solid B-rep body, for CAD and CAM."
            if not is_lattice_structure
            else (
                "Exact analytic B-rep built from the lattice centrelines: one "
                "cylinder or cone per member, one ball per node."
                if has_brep_export
                else
                "A minimal-surface or honeycomb cell has no compact B-rep. "
                "Release this result as STL."
            )
        )
        btn_export_3mf.setEnabled(
            structure_family is not None and structure_family.is_strut
        )
        is_lattice = structure_family is not None
        lattice_settings_guided = lattice_setup_is_guided(
            node.get_property("lattice_settings_mode")
        )
        if is_lattice:
            lattice_setup = _combo(
                "lattice_settings_mode",
                ["Guided", "Manual"],
                "Guided",
            )
            lattice_setup.currentTextChanged.connect(
                lambda _value: _refresh_topopt_later()
            )
            lattice_setup.setToolTip(
                "Guided chooses a family-safe cell resolution, wall/member "
                "size, skin, and density grading from the active grid. "
                "Manual exposes the manufacturing dimensions and their "
                "accepted range."
            )
            mfg_layout.addRow("Lattice Setup:", lattice_setup)
        # Cell pitch and the minimum printable wall/member are part of the
        # lattice definition, not numerical solver quality. Professional
        # lattice workflows expose them even in their short setup; zero keeps
        # a program-controlled value derived from the analysis grid. Skin is a
        # reconstruction detail and remains Expert-only.
        if (
            structure_mode != "solid envelope"
            and not lattice_settings_guided
        ):
            # Physical sizes first: a voxel is a different length on every part
            # and every quality preset, so a pitch in voxels silently rescales
            # the lattice when either changes. These override the voxel fields
            # below when non-zero.
            physical_fields = [
                (
                    "lattice_cell_size_mm",
                    "Cell Pitch:",
                    "Lattice unit-cell pitch in model units. This is the size "
                    "a printer's powder removal or minimum feature dictates.\n\n"
                    "Program Controlled derives it from the analysis grid, which "
                    "changes physical size whenever the grid is resized.",
                ),
                (
                    "lattice_member_thickness_mm",
                    "Min Wall / Member:",
                    "Thinnest lattice wall or strut, in model units.\n\n"
                    "Program Controlled derives it from the analysis grid.",
                ),
            ]
            if expert_mode:
                physical_fields.append((
                    "lattice_skin_thickness_mm",
                    "Skin Thickness:",
                    "Solid skin over the lattice, in model units.\n\n"
                    "Program Controlled derives it from the analysis grid.",
                ))
            for prop, label, tip in physical_fields:
                field = _double(prop, 0.0, 0.0, 10_000.0, decimals=3, step=0.5)
                field.setSpecialValueText("Program Controlled")
                field.setToolTip(tip)
                mfg_layout.addRow(label, field)
            if expert_mode:
                mfg_layout.addRow(
                    "Cell Pitch (analysis voxels):",
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
                    "Min Wall / Member (analysis voxels):",
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
                    "Skin Thickness (analysis voxels):",
                    _double(
                        "structure_skin_thickness_voxels",
                        0.75,
                        0.0,
                        20.0,
                        decimals=2,
                        step=0.25,
                    ),
                )
            resolved_voxel_size = getattr(
                node, "_resolved_lattice_voxel_size", None
            )
            last_result = getattr(node, "_last_result", None)
            if resolved_voxel_size is None and isinstance(last_result, dict):
                result_density = last_result.get("density")
                result_bounds = last_result.get("bounds")
                if (
                    result_density is not None
                    and isinstance(result_bounds, dict)
                    and "min" in result_bounds
                    and "max" in result_bounds
                ):
                    resolved_voxel_size = voxel_size_from_bounds(
                        tuple(int(v) for v in result_density.shape),
                        (result_bounds["min"], result_bounds["max"]),
                    )
            manual_pitch = _get_float("lattice_cell_size_mm", 0.0)
            if manual_pitch <= 0.0 and resolved_voxel_size is not None:
                manual_pitch = (
                    _get_float("structure_cell_size_voxels", 8.0)
                    * float(resolved_voxel_size)
                )
            range_hint = QtWidgets.QLabel(
                lattice_dimension_range_text(
                    node.get_property("structure_mode"),
                    resolved_voxel_size,
                    cell_size=manual_pitch,
                )
            )
            range_hint.setWordWrap(True)
            range_hint.setStyleSheet("color:#d6a85f; font-size:10px;")
            mfg_layout.addRow("Valid Range:", range_hint)
        if is_lattice and expert_mode and not lattice_settings_guided:
            # The mass budget, and the first thing an engineer specifying a
            # lattice actually knows. Cell pitch and member thickness are
            # printer capabilities that map onto mass unpredictably -- an 8 mm
            # octet cell with 1.6 mm struts measures 34% relative density, a
            # perforated solid rather than the open lattice those two numbers
            # suggest -- so the thickness is solved to hit this instead.
            target_density = _double(
                "lattice_target_relative_density",
                0.0,
                0.0,
                0.95,
                decimals=3,
                step=0.05,
            )
            target_density.setSpecialValueText("From member thickness")
            target_density.setToolTip(
                "Fraction of the optimized envelope the lattice should occupy — "
                "its mass budget.\n\n"
                "The member thickness is solved to reach it on the recovered "
                "grid, and the achieved value is reported in the log. When the "
                "cell pitch and analysis grid cannot resolve a member thin or "
                "thick enough, the log says what was reached instead.\n\n"
                "0 sizes the lattice from an explicit member thickness."
            )
            mfg_layout.addRow("Target Relative Density:", target_density)
            mfg_layout.addRow(
                "Part-Scale Stiffness Surrogate:",
                _combo(
                    "lattice_porosity",
                    [
                        "Conservative",
                        "Balanced (Concept)",
                        "Maximum Porosity (Concept)",
                    ],
                    "Conservative",
                ),
            )
            variable_density = _check("lattice_variable_density")
            variable_density.stateChanged.connect(
                lambda _state: _refresh_topopt_later()
            )
            mfg_layout.addRow("Grade Local Cell Density:", variable_density)
            # The grading band and the solid-transition level are numerics of the
            # density-to-thickness map, not manufacturing capabilities: nothing
            # a guided user knows tells them what to set here, and the preset
            # values are the ones every result was tuned against. Expert mode
            # keeps them.
            if expert_mode and _get_bool("lattice_variable_density", True):
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
            is_sized_strut_lattice = bool(
                structure_family is not None and structure_family.is_strut
            )
            if is_sized_strut_lattice:
                optimize_members = _check("optimize_lattice_members")
                optimize_members.stateChanged.connect(
                    lambda _state: _refresh_topopt_later()
                )
                mfg_layout.addRow(
                    "Optimize Member Sizes:",
                    optimize_members,
                )
                # Truss-sizing internals: a voxel-denominated diameter cap, an
                # iteration count, and an effective-length coefficient. The
                # defaults are the sizing loop's calibration, and nothing a
                # guided user knows about the part tells them what to move.
                if expert_mode and _get_bool("optimize_lattice_members", True):
                    mfg_layout.addRow(
                        "Max Diameter (analysis voxels):",
                        _double(
                            "lattice_max_member_thickness_voxels",
                            3.0,
                            0.25,
                            20.0,
                            decimals=2,
                            step=0.25,
                        ),
                    )
                    mfg_layout.addRow(
                        "Sizing Iterations:",
                        _int(
                            "lattice_member_sizing_iterations",
                            35,
                            1,
                            200,
                        ),
                    )
                    mfg_layout.addRow(
                        "Buckling Length Factor:",
                        _double(
                            "lattice_buckling_length_factor",
                            1.0,
                            0.1,
                            4.0,
                            decimals=2,
                            step=0.05,
                        ),
                    )
            homogenized = bool(
                structure_family is not None and structure_family.homogenized
            )
            lattice_hint = QtWidgets.QLabel(
                "Dimensions above are multiples of the topology-analysis voxel; "
                "use the reported voxel size to convert them to physical units. "
                + (
                    f"{structure_family.display_name} optimizes against its own "
                    "measured homogenized anisotropic tensor, so the continuum "
                    "exponent below is not used. A family with no shipped "
                    "table is homogenized once on first use, which takes about "
                    "a minute and is then cached. "
                    if homogenized
                    else
                    f"{structure_family.display_name} has no cubic homogenized "
                    "law, so it uses the isotropic continuum surrogate: "
                    "Conservative (p=1.8) is recommended, Balanced (p=1.25) "
                    "and Maximum Porosity (p=1.0) are optimistic concepts. "
                    if structure_family is not None
                    else ""
                )
                + "Cell orientation is fixed to the global axes. Octet can run "
                "a Phase 2 axial-truss sizing check for member stress, Euler "
                "buckling and displacement; BCC is bending-dominated. The "
                "strut families are reconstructed exactly from their "
                "centrelines and export as STEP, STL and native 3MF Beam "
                "Lattice; a minimal-surface or honeycomb result has no compact "
                "B-rep and exports as STL. Final beam/solid reanalysis "
                "of the explicit lattice remains mandatory."
            )
            lattice_hint.setWordWrap(True)
            lattice_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
            mfg_layout.addRow(lattice_hint)
        elif is_lattice and not lattice_settings_guided:
            minimum_density = _double(
                "lattice_min_relative_density",
                0.15,
                0.01,
                0.80,
                decimals=3,
                step=0.01,
            )
            minimum_density.setToolTip(
                "Lower printable relative density used when converting the "
                "optimized field into local wall or member thickness."
            )
            mfg_layout.addRow("Minimum Relative Density:", minimum_density)
            maximum_density = _double(
                "lattice_max_relative_density",
                0.60,
                0.20,
                0.99,
                decimals=3,
                step=0.01,
            )
            maximum_density.setToolTip(
                "Upper relative density used for an explicit lattice cell."
            )
            mfg_layout.addRow("Maximum Relative Density:", maximum_density)
            guided_lattice_hint = QtWidgets.QLabel(
                "Manual uses the dimensions and density band above. The "
                "accepted range is tied to the active analysis grid; rerun "
                "after changing quality or grid dimensions so it is checked "
                "before optimization. The explicit lattice still needs a "
                "beam or solid reanalysis before release."
            )
            guided_lattice_hint.setWordWrap(True)
            guided_lattice_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
            mfg_layout.addRow(guided_lattice_hint)
        elif is_lattice:
            resolved_voxel_size = getattr(
                node, "_resolved_lattice_voxel_size", None
            )
            last_result = getattr(node, "_last_result", None)
            if resolved_voxel_size is None and isinstance(last_result, dict):
                result_density = last_result.get("density")
                result_bounds = last_result.get("bounds")
                if (
                    result_density is not None
                    and isinstance(result_bounds, dict)
                    and "min" in result_bounds
                    and "max" in result_bounds
                ):
                    resolved_voxel_size = voxel_size_from_bounds(
                        tuple(int(v) for v in result_density.shape),
                        (result_bounds["min"], result_bounds["max"]),
                    )
            # Resolve the grid first: the printable floor is stated in model
            # units, so the guided dimensions depend on what a voxel measures.
            minimum_feature = node_minimum_feature_size(node)
            cell_voxels, member_voxels, skin_voxels = (
                guided_lattice_voxel_dimensions(
                    node.get_property("structure_mode"),
                    voxel_size=resolved_voxel_size,
                    minimum_feature_size=minimum_feature,
                )
            )
            if resolved_voxel_size is not None:
                automatic_sizes = (
                    f"On the active grid: {cell_voxels * resolved_voxel_size:.3g} "
                    f"cell pitch, {member_voxels * resolved_voxel_size:.3g} "
                    f"minimum wall/member, and "
                    f"{skin_voxels * resolved_voxel_size:.3g} skin thickness."
                )
            else:
                automatic_sizes = (
                    f"The resolved values are {cell_voxels:g} voxels per cell, "
                    f"{member_voxels:g} voxels minimum wall/member, and "
                    f"{skin_voxels:g} voxels of skin. Physical sizes are shown "
                    "after the grid is resolved."
                )
            if minimum_feature > 0.0:
                automatic_sizes += (
                    f" The wall/member is held at or above the "
                    f"{minimum_feature:.3g} minimum member size stated under "
                    "Design Intent."
                )
            else:
                automatic_sizes += (
                    " No minimum member size is stated under Design Intent, so "
                    "the wall/member follows the grid and is not checked "
                    "against a process capability."
                )
            guided_lattice_hint = QtWidgets.QLabel(
                "Guided owns all coupled lattice settings, so saved manual "
                "values cannot invalidate this run. It uses family-safe cell "
                "resolution and variable-density grading. "
                + automatic_sizes
                + " Final beam/solid reanalysis is still required."
            )
            guided_lattice_hint.setWordWrap(True)
            guided_lattice_hint.setStyleSheet(
                "color:#8f98a5; font-size:10px;"
            )
            mfg_layout.addRow("Automatic Values:", guided_lattice_hint)
        # Mesh repair, recovered-surface quality, and triangle retention are all
        # derived from the quality preset. Restating them here gave a guided user
        # three ways to contradict the preset they just chose, with no way to
        # judge the result until after a run.
        print_ready = _check("print_ready_mesh")
        print_ready.stateChanged.connect(lambda _state: _refresh_topopt_later())
        if expert_mode:
            mfg_layout.addRow("Repair / Smooth Mesh:", print_ready)
            mfg_layout.addRow(
                "Recovered Surface Quality:",
                _combo(
                    "surface_quality",
                    ["Standard", "High", "Professional"],
                    "Professional",
                ),
            )
        if expert_mode and _get_bool("print_ready_mesh"):
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
        mfg_group.setToolTip(
            "What the optimized load paths are built as. In guided mode that "
            "is the single Output Structure choice; the process constraints "
            "themselves (symmetry, extrusion, AM overhang, pull-out, max "
            "member size, repeated patterns) come from the Manufacturing "
            "process selected under Design Intent and are included in the "
            "differentiable optimization graph.\n\n"
            "Expert mode exposes those constraints directly, together with "
            "the lattice cell pitch, wall thickness, skin and mass budget."
        )
        mfg_group.setLayout(mfg_layout)
        self.props_layout.addWidget(mfg_group)

        view_group = QtWidgets.QGroupBox("Visualization")
        view_layout = QtWidgets.QFormLayout()

        visualization = QtWidgets.QComboBox()
        last_result = getattr(node, "_last_result", None)
        has_recovered_surface = bool(
            isinstance(last_result, dict)
            and isinstance(last_result.get("recovered_shape"), dict)
            and last_result["recovered_shape"].get("vertices") is not None
            and last_result["recovered_shape"].get("faces") is not None
        )
        visualization_modes = [
            "Manufactured Mesh" if is_lattice_structure else "CAD",
        ]
        # A persisted topology result can restore its solved density and
        # triangulated recovery without serializing the live OpenCASCADE B-rep.
        # Expose that saved surface directly instead of making "CAD" the only
        # geometry choice and then telling the user to run the solver again.
        if has_recovered_surface and not is_lattice_structure:
            visualization_modes.append("Surface")
        visualization_modes.append("Density")
        validation_result = (
            last_result.get("validation")
            if isinstance(last_result, dict)
            else None
        )
        if isinstance(validation_result, dict):
            if validation_result.get("stress") is not None:
                visualization_modes.append("Validated Von Mises Stress")
            if validation_result.get("displacement") is not None:
                visualization_modes.append("Validated Displacement")
        visualization.addItems(visualization_modes)
        current_view = str(
            (
                last_result.get("visualization_mode")
                if isinstance(last_result, dict)
                else None
            )
            or node.get_property("visualization")
            or "CAD"
        )
        geometry_view = "Manufactured Mesh" if is_lattice_structure else "CAD"
        recovered_view = (
            "Surface"
            if has_recovered_surface and not is_lattice_structure
            else geometry_view
        )
        current_view = {
            "Recovered Shape": recovered_view,
            "Recovered Surface (Mesh)": recovered_view,
            "Surface": recovered_view,
            "Reconstructed CAD (B-rep)": "CAD",
            "Voxel Density": "Density",
        }.get(current_view, current_view)
        if "density" in current_view.lower() or "voxel" in current_view.lower():
            current_view = "Density"
        view_index = visualization.findText(current_view)
        visualization.setCurrentIndex(view_index if view_index >= 0 else 0)
        visualization.currentTextChanged.connect(
            lambda v: self._set_topopt_visualization(node, v)
        )
        visualization.setToolTip(
            "CAD: automatically reconstructed solid envelope\n"
            "Surface: saved triangulated topology boundary\n"
            "Manufactured Mesh: explicit lattice, STL-ready\n"
            "Density: voxel design field"
        )
        view_layout.addRow("Mode:", visualization)
        if structure_mode != "solid envelope":
            structure_hint = QtWidgets.QLabel(
                "Manufactured Mesh shows the explicit lattice; Density shows "
                "its design field."
            )
            structure_hint.setWordWrap(True)
            structure_hint.setStyleSheet("color:#8f98a5; font-size:10px;")
            view_layout.addRow(structure_hint)

        # Report an eager or lazily cached CAD body when one exists.
        cad_report = (
            last_result.get("cad_reconstruction")
            if isinstance(last_result, dict)
            else None
        )
        if isinstance(cad_report, dict):
            method_name = str(cad_report.get("method") or "CAD B-rep")
            face_count = cad_report.get("cad_face_count_after_feature_healing")
            if face_count is None:
                face_count = cad_report.get("cad_face_count")
            edit_label = (
                ""
                if cad_report.get("editable") is True
                else "non-editable"
                if cad_report.get("editable") is False
                else "editability not reported"
            )
            count_label = f", {int(face_count)} faces" if face_count is not None else ""
            cad_status = QtWidgets.QLabel(
                f"{method_name}{count_label} · {edit_label}"
            )
            cad_status.setWordWrap(True)
            cad_status.setStyleSheet(
                "color:#66d17a; font-size:10px;"
                if cad_report.get("editable") is True
                else "color:#ffb74d; font-size:10px;"
            )
            view_layout.addRow(cad_status)

        view_group.setLayout(view_layout)
        self.props_layout.addWidget(view_group)
        self.props_layout.addWidget(pipeline_group)

    def _build_lattice_infill_ui(self, node):
        """Build the dedicated Inspector UI for the LatticeInfillNode."""
        def _get_int(prop, default):
            val = node.get_property(prop)
            try:
                return int(val) if val is not None else int(default)
            except (ValueError, TypeError):
                return int(default)

        def _get_float(prop, default):
            val = node.get_property(prop)
            try:
                return float(val) if val is not None else float(default)
            except (ValueError, TypeError):
                return float(default)

        def _combo(prop, items, default=None):
            widget = QtWidgets.QComboBox()
            widget.setMinimumWidth(0)
            widget.setMinimumContentsLength(1)
            widget.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            widget.addItems(list(items))
            current = str(node.get_property(prop) or default or (items[0] if items else ""))
            if current in items:
                widget.setCurrentText(current)
            elif items:
                widget.setCurrentText(items[0])
            widget.currentTextChanged.connect(
                lambda val, p=prop: self.update_property(p, str(val))
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _double(prop, default, lo, hi, decimals=3, step=0.1, suffix=""):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setLocale(QtCore.QLocale(QtCore.QLocale.C))
            widget.setDecimals(decimals)
            widget.setRange(lo, hi)
            widget.setSingleStep(step)
            if suffix:
                widget.setSuffix(suffix)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            widget.setValue(_get_float(prop, default))
            widget.valueChanged.connect(
                lambda val, p=prop: self.update_property(p, float(val))
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _int(prop, default, lo, hi):
            widget = QtWidgets.QSpinBox()
            widget.setRange(lo, hi)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            widget.setValue(_get_int(prop, default))
            widget.valueChanged.connect(
                lambda val, p=prop: self.update_property(p, int(val))
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        def _text(prop, default):
            widget = QtWidgets.QLineEdit(str(node.get_property(prop) or default))
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            widget.editingFinished.connect(
                lambda w=widget, p=prop: self.update_property(p, w.text())
            )
            tooltip = self._PROPERTY_TOOLTIPS.get(prop)
            if tooltip:
                widget.setToolTip(str(tooltip))
            return widget

        infill_group = QtWidgets.QGroupBox("Lattice Infill Settings")
        infill_layout = QtWidgets.QFormLayout()
        infill_layout.addRow("Lattice Pattern", _combo("structure_mode", list(PUBLIC_LATTICE_FAMILY_NAMES), "Gyroid Lattice"))
        infill_layout.addRow("Relative Density", _double("lattice_target_relative_density", 0.25, 0.05, 1.0, decimals=2, step=0.05))
        infill_layout.addRow("Cell Size Mode", _combo("infill_cell_size_mode", ["Automatic", "Manual"], "Automatic"))
        if str(node.get_property("infill_cell_size_mode") or "Automatic") == "Automatic":
            infill_layout.addRow("Cell Fineness", _combo("infill_fineness", list(AUTOMATIC_FINENESS_FRACTIONS.keys()), "Medium"))
        else:
            infill_layout.addRow("Cell Size (mm)", _double("infill_cell_size_mm", 10.0, 0.1, 10000.0, decimals=2, step=1.0, suffix=" mm"))
        infill_layout.addRow("Outer Skin", _double("infill_skin_thickness_mm", 0.0, 0.0, 100.0, decimals=2, step=0.5, suffix=" mm"))
        infill_group.setLayout(infill_layout)
        self.props_layout.addWidget(infill_group)

        grid_group = QtWidgets.QGroupBox("Grid & Display")
        grid_layout = QtWidgets.QFormLayout()
        grid_layout.addRow("Grid X (voxels)", _int("nelx", 40, 4, 300))
        grid_layout.addRow("Grid Y (voxels)", _int("nely", 40, 4, 300))
        grid_layout.addRow("Grid Z (voxels)", _int("nelz", 40, 4, 300))
        grid_layout.addRow("Visualization", _combo("visualization", ["Manufactured Mesh", "Density", "CAD"], "Manufactured Mesh"))
        grid_layout.addRow("Export Filename", _text("cad_export_filename", "lattice_infill.step"))
        grid_group.setLayout(grid_layout)
        self.props_layout.addWidget(grid_group)

    def _set_topopt_visualization(self, node, mode):
        """Switch result views and lazily build the requested CAD body."""
        mode = str(mode)
        self.update_property("visualization", mode)
        if mode != "CAD":
            return

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("recovered_shape") is None:
            window = self.window()
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage(
                    "Run Topology first.",
                    7000,
                )
            return

        settings = self._topopt_cad_settings(node, result)
        report = result.get("cad_reconstruction")
        cached_strategy = (
            str(report.get("requested_strategy") or "")
            if isinstance(report, dict)
            else ""
        )
        if result.get("cad_shape") is not None and cached_strategy == settings["strategy"]:
            result["visualization_mode"] = mode
            viewer = getattr(self.window(), "viewer", None)
            if viewer is not None:
                viewer.render_simulation(result)
            return
        self._preview_topopt_cad(node)

    def _topopt_cad_settings(self, node, result):
        """Collect reconstruction settings shared by preview and export."""
        passive_regions = result.get("passive_regions")
        if not isinstance(passive_regions, dict) and hasattr(node, "_build_bc"):
            try:
                bc = node._build_bc()
                passive_regions = {
                    "solid_boxes": list(getattr(bc, "solid_boxes", ())),
                    "void_boxes": list(getattr(bc, "void_boxes", ())),
                    "solid_cylinders": list(getattr(bc, "solid_cylinders", ())),
                    "void_cylinders": list(getattr(bc, "void_cylinders", ())),
                    "joint_pin_cylinders": list(
                        getattr(bc, "joint_pin_cylinders", ())
                    ),
                }
            except Exception:
                passive_regions = {}
        if not isinstance(passive_regions, dict):
            passive_regions = {}

        method = str(
            node.get_property("cad_reconstruction_method")
            or "Auto"
        ).strip().lower()
        strategy = "spline" if method.startswith("smooth") else "auto"
        cutoff = float(
            result.get("density_cutoff")
            or node.get_property("density_cutoff")
            or 0.30
        )
        extrusion_axis = (
            str(
                result.get("extrusion_axis")
                or node.get_property("extrusion")
                or "none"
            )
            .strip()
            .lower()
        )
        return {
            "passive_regions": passive_regions,
            "strategy": strategy,
            "cutoff": cutoff,
            "extrusion_axis": extrusion_axis,
        }

    def _preview_topopt_cad(self, node):
        """Build, cache, and display the actual STEP-ready TopOpt B-rep."""
        worker = getattr(self, "_topopt_cad_preview_worker", None)
        if worker is not None and worker.isRunning():
            window = self.window()
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage("CAD build already running.")
            return

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("recovered_shape") is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Result",
                "Run Topology first.",
            )
            return

        settings = self._topopt_cad_settings(node, result)
        if settings["extrusion_axis"] not in {"x", "y", "z"}:
            # Freeform patch fitting is gone: a general 3-D load path has no
            # exact B-rep, and approximating one produced a body that no longer
            # described the optimized structure. The recovered surface stays.
            QtWidgets.QMessageBox.information(
                self,
                "No Editable CAD Body",
                "This result has no exact B-rep. A solid CAD body is built "
                "only for an extrusion-constrained topology, whose surface is "
                "a profile swept along one axis.\n\n"
                "Set the manufacturing process to Extruded (or choose an "
                "explicit X, Y, or Z extrusion) and re-run to get an editable "
                "solid. Otherwise use the recovered surface, which carries "
                "this geometry exactly and exports as STL.",
            )
            return
        worker = TopOptCadPreviewWorker(
            result,
            density_cutoff=settings["cutoff"],
            extrusion_axis=settings["extrusion_axis"],
            passive_regions=settings["passive_regions"],
            reconstruction_strategy=settings["strategy"],
            parent=self,
        )
        self._topopt_cad_preview_worker = worker

        def _finish(shape, report):
            self._topopt_cad_preview_worker = None
            report = dict(report or {})
            result["cad_shape"] = shape
            result["shape"] = shape
            result["cad_reconstruction"] = report
            result["visualization_mode"] = "CAD"
            node._last_result = result
            node._last_cad_reconstruction_report = report

            if self.current_node is node:
                self.update_property(
                    "visualization",
                    "CAD",
                )
            else:
                node.set_property(
                    "visualization",
                    "CAD",
                )

            window = self.window()
            viewer = getattr(window, "viewer", None)
            if viewer is not None:
                viewer.render_simulation(result)
            results_panel = getattr(window, "results", None)
            if results_panel is not None:
                try:
                    results_panel.show_result(result)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Could not refresh TopOpt CAD result summary.",
                        exc_info=True,
                    )

            method_name = str(report.get("method") or "CAD B-rep")
            face_count = report.get("cad_face_count_after_feature_healing")
            count_text = (
                f", {int(face_count)} faces" if face_count is not None else ""
            )
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage(
                    f"CAD ready: {method_name}{count_text}",
                    12000,
                )
            QtCore.QTimer.singleShot(
                0,
                lambda: self.display_node(node) if self.current_node is node else None,
            )

        def _fail(message):
            self._topopt_cad_preview_worker = None
            from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
                lattice_cad_strategy,
            )

            if lattice_cad_strategy(result.get("structure_options")) == "isosurface":
                # Not an error: this family has no B-rep to build. Show the
                # recovered surface, which is the geometry that gets released,
                # and say why rather than raising a red dialog over a working
                # result. The message names the export formats that do carry it.
                result["visualization_mode"] = "Surface"
                viewer = getattr(self.window(), "viewer", None)
                if viewer is not None:
                    viewer.render_simulation(result)
                window = self.window()
                if hasattr(window, "statusBar") and window.statusBar():
                    window.statusBar().showMessage(
                        "Showing the manufactured mesh. This cell has no "
                        "compact B-rep, so it is released as STL; the strut "
                        "families export as STEP and 3MF.",
                        12000,
                    )
                return
            QtWidgets.QMessageBox.critical(
                self,
                "CAD Error",
                message,
            )

        worker.preview_finished.connect(_finish)
        worker.preview_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        window = self.window()
        if hasattr(window, "statusBar") and window.statusBar():
            window.statusBar().showMessage("Building CAD...")
        worker.start()

    def _refresh_topopt_recovered_shape(self, node, result):
        """Rebuild recovered_shape from the current density before export."""
        if not isinstance(result, dict) or result.get("density") is None:
            return result.get("recovered_shape") if isinstance(result, dict) else None
        try:
            import numpy as np
            from pylcss.design_studio.topology_optimization.geometry.surface_recovery import (
                _recover_voxel_shape,
            )
            from pylcss.design_studio.topology_optimization.integration.lattice_settings import (
                resolve_lattice_structure_options,
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
            density_array = np.asarray(result["density"], dtype=float)
            structure_options = resolve_lattice_structure_options(
                node,
                tuple(int(v) for v in density_array.shape),
                bounds,
            )
            recovered = _recover_voxel_shape(
                density_array,
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
                member_plan=getattr(node, "_last_member_plan", None),
                surface_backend=(
                    "legacy"
                    if str(node.get_property("surface_recovery_method") or "")
                    .lower()
                    .startswith("legacy")
                    else "vtk_sdf"
                ),
                surface_quality=str(
                    node.get_property("surface_quality") or "Professional"
                ),
            )
            if recovered is not None and len(recovered.get("faces", [])) > 0:
                result["recovered_shape"] = recovered
                result.pop("cad_shape", None)
                result.pop("shape", None)
                result.pop("cad_reconstruction", None)
                setattr(node, "_last_result", result)
                return recovered
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        return result.get("recovered_shape")

    def _export_topopt_step(self, node):
        """Export the automatically reconstructed solid CAD body as STEP."""
        from pylcss.design_studio.topology_optimization.manufacturing import (
            family_for,
        )

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
            QtWidgets.QMessageBox.warning(
                self,
                "No Result",
                "Run Topology first.",
            )
            return

        from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
            lattice_cad_strategy,
        )

        structure_options = result.get("structure_options")
        if structure_options is not None:
            strategy = lattice_cad_strategy(structure_options)
        else:
            family = family_for(node.get_property("structure_mode"))
            strategy = (
                "solid"
                if family is None
                else ("beam" if family.is_strut else "isosurface")
            )
        if strategy == "isosurface":
            QtWidgets.QMessageBox.information(
                self,
                "No B-rep for This Lattice",
                "A minimal-surface or honeycomb cell has no compact boundary "
                "representation, so there is no STEP body to write. Release "
                "this result as STL.\n\n"
                "The strut families (BCC, Octet Truss) are reconstructed "
                "exactly from their centrelines and do export as STEP.",
            )
            return

        cad_shape = result.get("cad_shape")
        if cad_shape is None:
            cad_shape = result.get("shape")
        if cad_shape is None and strategy == "beam":
            # A study saved before the solve reconstructed strut lattices, or
            # one whose reconstruction failed. The centrelines are in the
            # result either way, so build the body now rather than sending the
            # user back to re-run.
            from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
                reconstruct_topopt_cad,
            )

            try:
                cad_shape = reconstruct_topopt_cad(
                    result,
                    structure_options=structure_options,
                    member_plan=result.get("member_plan"),
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Lattice B-rep Unavailable",
                    f"The lattice centrelines did not build a solid: {exc}",
                )
                return
        if cad_shape is None:
            extrusion_axis = self._topopt_cad_settings(node, result)["extrusion_axis"]
            if extrusion_axis not in {"x", "y", "z"}:
                QtWidgets.QMessageBox.information(
                    self,
                    "No STEP for a General 3-D Result",
                    "A general three-dimensional load path has no exact "
                    "editable B-rep, so there is no STEP body to write.\n\n"
                    "Constrain the study to an extrusion axis and re-run for a "
                    "solid CAD body, or export this result as STL, which "
                    "carries the recovered surface exactly.",
                )
                return
            QtWidgets.QMessageBox.warning(
                self,
                "CAD Body Unavailable",
                "The solid result has no reconstructed CAD body. Rerun the "
                "topology study before exporting STEP or STL.",
            )
            return

        worker = getattr(self, "_topopt_step_export_worker", None)
        if worker is not None and worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("STEP export already running.")
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

        node.set_property("cad_export_filename", path)
        worker = TopOptStepExportWorker(
            result,
            path,
            cached_shape=cad_shape,
            parent=self,
        )
        self._topopt_step_export_worker = worker

        def _finish(export_path):
            self._topopt_step_export_worker = None
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"STEP exported: {export_path}"
                )

        def _fail(message):
            self._topopt_step_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", message)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting CAD body as STEP...")
        worker.start()

    def _export_topopt_stl(self, node):
        """Export solid CAD or lattice mesh as a binary STL."""
        from pylcss.design_studio.topology_optimization.manufacturing import (
            family_for,
        )

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "No Shape",
                "Run Topology first.",
            )
            return

        active_worker = getattr(self, "_topopt_mesh_export_worker", None)
        if active_worker is not None and active_worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("A mesh export is already running.")
            return

        structure_options = result.get("structure_options")
        is_solid_result = (
            getattr(structure_options, "mode", "solid") == "solid"
            if structure_options is not None
            else family_for(node.get_property("structure_mode")) is None
        )
        if is_solid_result:
            cad_shape = result.get("cad_shape")
            if cad_shape is None:
                cad_shape = result.get("shape")
            recovered = None
            if cad_shape is None:
                # A general 3-D solid never gets a B-rep, but its recovered
                # surface is the exact optimized geometry and is what STL
                # carries anyway.
                recovered = result.get("recovered_shape")
                if not isinstance(recovered, dict):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Shape",
                        "The solid result has neither a reconstructed CAD body "
                        "nor a recovered surface. Rerun the topology study "
                        "before exporting.",
                    )
                    return
        else:
            recovered = result.get("recovered_shape")
            cad_shape = None

        # A strut lattice is written from its centrelines; the worker falls
        # back to the recovered surface for every other family.
        if not is_solid_result and not isinstance(recovered, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "No Shape",
                "The topology result has no recovered surface. Rerun Topology first.",
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)"
        )
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"

        worker = TopOptMeshExportWorker(
            path,
            cad_shape=cad_shape,
            recovered_shape=recovered,
            topology_result=result,
            structure_options=structure_options,
            member_plan=result.get("member_plan"),
            export_format="stl",
            parent=self,
        )
        self._topopt_mesh_export_worker = worker

        def _finish(details):
            self._topopt_mesh_export_worker = None
            triangle_count = int(details.get("triangles", 0))
            source = str(details.get("source") or "")
            suffix = f" from the {source}" if source else ""
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported {triangle_count} STL triangles{suffix} "
                    f"to {details['path']}"
                )

        def _fail(message):
            self._topopt_mesh_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", message)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting topology as STL...")
        worker.start()

    def _export_topopt_beam_3mf(self, node):
        """Export a strut lattice as a 3MF beam lattice, from its centrelines."""
        from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
            lattice_cad_strategy,
        )

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
            QtWidgets.QMessageBox.warning(self, "No Result", "Run Topology first.")
            return

        options = result.get("structure_options")
        if lattice_cad_strategy(options) != "beam":
            QtWidgets.QMessageBox.information(
                self,
                "Not a Strut Lattice",
                "The 3MF beam lattice format describes cylindrical members "
                "between nodes. In the current catalogue it applies to BCC "
                "and Octet Truss; retired strut families remain supported in "
                "saved legacy projects.\n\n"
                "A surface or honeycomb lattice has no centrelines to write. "
                "Export it as STL instead.",
            )
            return

        active_worker = getattr(self, "_topopt_mesh_export_worker", None)
        if active_worker is not None and active_worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("A mesh export is already running.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export 3MF Beam Lattice", "topology_lattice.3mf", "3MF Files (*.3mf)"
        )
        if not path:
            return
        if not path.lower().endswith(".3mf"):
            path += ".3mf"

        worker = TopOptMeshExportWorker(
            path,
            topology_result=result,
            structure_options=options,
            member_plan=result.get("member_plan"),
            export_format="3mf",
            parent=self,
        )
        self._topopt_mesh_export_worker = worker

        def _finish(details):
            self._topopt_mesh_export_worker = None
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported {details['beams']} beams and "
                    f"{details['nodes']} nodes to {details['path']}"
                )

        def _fail(message):
            self._topopt_mesh_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", message)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting beam lattice as 3MF...")
        worker.start()
