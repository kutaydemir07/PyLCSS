# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""TopologyInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
from .result_presentation import result_state_rows
from pylcss.cae.design_studio.topology_optimization.geometry.lattice_cad import topopt_export_formats

from pylcss.cae.design_studio.topology_optimization.manufacturing.lattice_families import (
    PUBLIC_LATTICE_FAMILY_NAMES,
)
from pylcss.cae.design_studio.topology_optimization.integration.lattice_infill_node import (
    AUTOMATIC_FINENESS_FRACTIONS,
)
from pylcss.cae.design_studio.topology_optimization.configuration.presets import (
    INDUSTRIAL_DESIGN_GOALS,
    industrial_topopt_defaults,
)
from pylcss.cae.design_studio.topology_optimization.configuration.voxel_meshing import (
    DEFAULT_VOXEL_MESH_QUALITY,
    DEFAULT_MAX_VOXEL_ELEMENTS,
    VOXEL_MESH_ELEMENT_BUDGETS,
)

from .execution_workers import (
    TopOptCadPreviewWorker,
    TopOptLatticeWorker,
    TopOptMeshExportWorker,
    TopOptStepExportWorker,
)
from .inspector_controls import InspectorSection
from .inspector_non_design import NonDesignInspectorMixin
from .inspector_topopt_fea import TopOptFeaInspectorMixin
from .result_fields import available_result_fields

__all__ = ["TopologyInspectorMixin"]


class TopologyInspectorMixin(TopOptFeaInspectorMixin, NonDesignInspectorMixin):
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
                    "Edges and vertices apply line or point conditions without "
                    "a solid pad. Local stresses near these conditions depend "
                    "on mesh resolution. Select a face when the physical "
                    "load/support acts over an area."
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
            definition.setToolTip(note)
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

        guided_preserved_properties = {"load_aggregation"}

        def _apply_guided_settings(settings):
            import json

            overrides = set(json.loads(node.get_property("advanced_overrides") or "[]"))
            self._applying_topopt_defaults = True
            try:
                for key, setting in settings.items():
                    if key not in guided_preserved_properties | overrides and node.has_property(key):
                        self.update_property(key, setting)
            finally:
                self._applying_topopt_defaults = False

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
                goal = (
                    internal_value
                    if p == "design_goal"
                    else node.get_property("design_goal")
                )
                settings = industrial_topopt_defaults(goal)
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

        intent_group = QtWidgets.QGroupBox("Design Intent")
        intent_layout = QtWidgets.QFormLayout()
        design_goals = INDUSTRIAL_DESIGN_GOALS
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
        material_tooltip = (
            "Maximum optimized material volume as a percentage of the source "
            "design domain, including non-design material within that domain. "
            "Both objectives enforce this budget. Stress-Limited Mass may use "
            "less material; it must also satisfy the material's stress limit."
        )
        material_slider.setToolTip(material_tooltip)
        material_spin.setToolTip(material_tooltip)

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
        intent_layout.addRow("Target Volume:", material_container)

        intent_group.setLayout(intent_layout)
        self.props_layout.addWidget(intent_group)
        self._build_topopt_non_design_ui(node)

        mesh_group = QtWidgets.QGroupBox("Voxel Meshing")
        mesh_layout = QtWidgets.QFormLayout(mesh_group)
        automatic_grid = _check("automatic_grid")
        quality = _combo("voxel_mesh_quality", list(VOXEL_MESH_ELEMENT_BUDGETS), DEFAULT_VOXEL_MESH_QUALITY)
        quality.setToolTip(
            "Element budgets for Coarse / Medium / Fine / Very Fine: "
            + " / ".join(f"{budget:,}" for budget in VOXEL_MESH_ELEMENT_BUDGETS.values())
            + ". Automatic sizing finds the finest cubic grid within the preset "
            "budget, Maximum Elements and available RAM. The estimate reports "
            "the effective budget, grid and voxel size."
        )
        size = _double("voxel_size", 0.0, 0.0, 1e6, decimals=6, step=0.1, suffix=" mm")
        size.setSpecialValueText("Automatic")
        size.setToolTip(
            "0 uses the preset or manual element counts. A positive size "
            "overrides both and sets the exact cubic voxel edge in mm, on X/Y/Z. "
            "The grid box pads outward to retain that size; CAD geometry defines "
            "the material. Requests over Maximum Elements are rejected."
        )
        limit = _int("max_voxel_elements", DEFAULT_MAX_VOXEL_ELEMENTS, 0, 100_000_000)
        limit.setSpecialValueText("Unlimited")
        limit.setToolTip(
            "Default 1,000,000 grid elements. Automatic mode selects its pitch "
            "within the preset budget, this limit and available RAM. Explicit voxel sizes and manual "
            "counts are checked without changing them. 0 disables the element cap."
        )
        estimate = QtWidgets.QLabel()
        estimate.setWordWrap(True)
        estimate.setObjectName("TopOptVoxelMeshEstimate")
        refinement = _check("automatic_mesh_refinement")
        refinement.setToolTip(
            "Optional coarse-to-fine optimization, with up to three short "
            "warm-start stages before the final mesh. Each stage transfers "
            "its density design; the final stage keeps the full convergence "
            "budget. Small Coarse grids need only one solve."
        )
        self.property_widgets.update({
            "automatic_grid": automatic_grid, "voxel_mesh_quality": quality,
            "voxel_size": size, "automatic_mesh_refinement": refinement,
            "max_voxel_elements": limit,
        })

        manual_counts = {}
        for prop, label, default in (
            ("nelx", "Cells X:", 30),
            ("nely", "Cells Y:", 20),
            ("nelz", "Cells Z:", 10),
        ):
            widget = _int(prop, default, 1, 10000)
            manual_counts[prop] = (label, widget)
            self.property_widgets[prop] = widget

        def _mesh_controls():
            self._topopt_source_bounds = None
            explicit_size = size.value() > 0
            quality.setEnabled(automatic_grid.isChecked() and not explicit_size)
            automatic_grid.setEnabled(not explicit_size)
            manual = not automatic_grid.isChecked() and not explicit_size
            for label, widget in manual_counts.values():
                widget.setEnabled(manual)
                mesh_layout.setRowVisible(widget, manual)
            from pylcss.cae.design_studio.topology_optimization.integration.execution_setup import plan_topology_voxel_mesh
            from pylcss.cae.design_studio.topology_optimization.integration.geometry_mapping import _mesh_bounds
            try:
                # Inspecting a node must not use lazy input resolution: an
                # uncomputed Mesh/support/load can launch Netgen on the GUI
                # thread. Native CAD bounds also avoid display tessellation.
                source_bounds = _mesh_bounds(
                    node.get_cached_input_value("design_domain"),
                    allow_tessellation=False,
                )
                self._topopt_source_bounds = source_bounds
                if source_bounds is None:
                    estimate.setText("Run to estimate")
                    estimate.setToolTip("Run the upstream geometry or mesh to update this estimate.")
                    estimate.setStyleSheet("color:#8f98a5; font-size:10px;")
                    return
                plan = plan_topology_voxel_mesh(node, source_bounds)
                ram = plan.estimated_memory_bytes / (1024 ** 3)
                pitch_text = f"{plan.pitch:.6g} mm · " if plan.pitch is not None else ""
                estimate.setText(
                    f"{pitch_text}{' × '.join(map(str, plan.shape))} · {plan.elements:,} elements · {ram:.2f} GiB planning allowance"
                    + (f" · Budget {plan.automatic_element_budget:,}" if plan.automatic_element_budget is not None else "")
                    + (" · Resource limited" if plan.resource_limited else "")
                    + (" · Over limit" if plan.resource_error() else "")
                )
                detail = "Estimated grid size and RAM."
                if plan.automatic_element_budget is not None:
                    detail = (f"{plan.quality} preset: {plan.preset_element_budget:,} elements; "
                              f"effective budget: {plan.automatic_element_budget:,} elements "
                              "after Maximum Elements and available RAM. "
                              "Cubic grid counts are rounded for the solver, "
                              "so the actual count can be below the budget.")
                estimate.setToolTip(plan.resource_error() or detail)
                estimate.setStyleSheet("color:#ffb35c; font-size:10px;" if plan.resource_error() else "color:#8f98a5; font-size:10px;")
            except (TypeError, ValueError, OverflowError) as exc:
                estimate.setText(str(exc))
                estimate.setStyleSheet("color:#ffb35c; font-size:10px;")

        automatic_grid.toggled.connect(lambda _value: _mesh_controls())
        size.valueChanged.connect(lambda _value: _mesh_controls())
        quality.currentTextChanged.connect(lambda _value: _mesh_controls())
        limit.valueChanged.connect(lambda _value: _mesh_controls())
        self._topopt_mesh_estimate_refresh = _mesh_controls
        mesh_layout.addRow("Automatic Grid:", automatic_grid)
        mesh_layout.addRow("Final Resolution:", quality)
        mesh_layout.addRow("Voxel Size:", size)
        mesh_layout.addRow("Maximum Elements:", limit)
        for label, widget in manual_counts.values():
            mesh_layout.addRow(label, widget)
            widget.valueChanged.connect(lambda _value: _mesh_controls())
        mesh_layout.addRow("Estimate:", estimate)
        mesh_layout.addRow("Automatic Refinement:", refinement)
        _mesh_controls()
        previous_result = getattr(node, "_last_result", None)
        if isinstance(previous_result, dict) and previous_result.get("voxel_meshing"):
            solved_grid = previous_result.get("grid_shape") or previous_result["voxel_meshing"].get("grid")
            if solved_grid is not None:
                mesh_layout.addRow("Last Solved Grid:", QtWidgets.QLabel(" × ".join(map(str, solved_grid))))
        self.props_layout.addWidget(mesh_group)
        self._build_topopt_advanced_ui(node)

        previous_result = getattr(node, "_last_result", None)
        is_lattice_structure = bool(isinstance(previous_result, dict) and previous_result.get("generated_lattice"))

        view_group = QtWidgets.QGroupBox("Visualization")
        view_layout = QtWidgets.QFormLayout()

        visualization = QtWidgets.QComboBox()
        visualization.setObjectName("TopOptViewMode")
        last_result = getattr(node, "_last_result", None)
        has_recovered_surface = bool(
            isinstance(last_result, dict)
            and isinstance(last_result.get("recovered_shape"), dict)
            and last_result["recovered_shape"].get("vertices") is not None
            and last_result["recovered_shape"].get("faces") is not None
        )
        from .result_fields import topopt_view_mode
        geometry_view = "Lattice" if is_lattice_structure else "STL Shape"
        visualization_modes = available_result_fields(last_result)
        if visualization_modes == ["Default View"]:
            visualization_modes = ["Density"]
        visualization.addItems(visualization_modes)
        current_view = str(
            (
                last_result.get("visualization_mode")
                if isinstance(last_result, dict)
                else None
            )
            or node.get_property("visualization")
            or geometry_view
        )
        current_view = topopt_view_mode(last_result or {}, current_view)
        if "density" in current_view.lower() or "voxel" in current_view.lower():
            current_view = "Density"
        view_index = visualization.findText(current_view)
        visualization.setCurrentIndex(view_index if view_index >= 0 else 0)
        visualization.currentTextChanged.connect(
            lambda v: self._set_topopt_visualization(node, v)
        )
        visualization.setToolTip(
            "Density: solved voxel field\n"
            "STL Shape: recovered solid surface mesh\n"
            "CAD: completed CAD body\n"
            "FEA: completed independent analysis\n"
            "Lattice: generated cells with preserved solid regions"
        )
        view_layout.addRow("Mode:", visualization)

        # A recovered surface is useful while reconstruction runs, but is not
        # evidence that an editable CAD body has been built.
        has_cad_body = bool(
            isinstance(last_result, dict)
            and (last_result.get("cad_shape") is not None or last_result.get("shape") is not None)
        )
        if has_recovered_surface:
            triangle_count = len(last_result["recovered_shape"]["faces"])
            preview_status = QtWidgets.QLabel(
                f"{triangle_count:,} triangles"
            )
            preview_status.setWordWrap(True)
            preview_status.setStyleSheet("color:#ffb74d; font-size:10px;")
            view_layout.addRow("Surface:", preview_status)

        has_density = bool(
            isinstance(last_result, dict) and last_result.get("density") is not None
        )
        supports_cad = "STEP" in self._topopt_available_formats(node)
        cad_worker = getattr(self, "_topopt_cad_preview_worker", None)
        building_cad = cad_worker is not None and cad_worker.isRunning()
        build_cad = QtWidgets.QPushButton("Build CAD")
        build_cad.setObjectName("TopOptBuildCad")
        build_cad.setEnabled(
            supports_cad and (has_recovered_surface or has_density)
            and not has_cad_body and not building_cad
        )
        build_cad.clicked.connect(lambda _checked=False: self._preview_topopt_cad(node))
        if not (is_lattice_structure and not supports_cad):
            view_layout.addRow(build_cad)
        else:
            build_cad.hide()
        if not supports_cad:
            cad_hint = "Lattice mesh is ready. Export as STL."
        elif building_cad:
            cad_hint = "Building CAD in the background."
        elif has_cad_body:
            cad_hint = "CAD is ready. Select CAD above to view it or export as STEP."
        elif not (has_recovered_surface or has_density):
            cad_hint = "Run TopOpt first, then use Build CAD to create the CAD body."
        else:
            cad_hint = "Build CAD creates a CAD body from the solved result for STEP export."
        build_cad.setToolTip(cad_hint)
        cad_hint_label = QtWidgets.QLabel(cad_hint)
        cad_hint_label.setObjectName("TopOptCadHint")
        cad_hint_label.setWordWrap(True)
        view_layout.addRow(cad_hint_label)

        # Report a lazily cached CAD body or a failed reconstruction.
        cad_report = (
            last_result.get("cad_reconstruction")
            if isinstance(last_result, dict)
            else None
        )
        if isinstance(cad_report, dict) and not cad_report.get("skipped"):
            method_name = str(cad_report.get("method") or "CAD B-rep")
            face_count = cad_report.get("cad_face_count_after_feature_healing")
            if face_count is None:
                face_count = cad_report.get("cad_face_count")
            count_label = f", {int(face_count)} faces" if face_count is not None else ""
            cad_status = QtWidgets.QLabel(
                f"{method_name}{count_label}"
            )
            cad_status.setWordWrap(True)
            if cad_report.get("error"):
                cad_status.setToolTip(str(cad_report["error"]))
            cad_status.setStyleSheet(
                "color:#66d17a; font-size:10px;"
                if cad_report.get("editable") is True
                else "color:#ffb74d; font-size:10px;"
            )
            view_layout.addRow("CAD:", cad_status)

        view_group.setLayout(view_layout)
        self.props_layout.addWidget(view_group)
        self._build_post_topopt_lattice_ui(node, _combo, _double)
        self._build_topopt_fea_ui(node)
        pipeline_group = self._build_export_group(
            node,
            default_filename=(
                "topopt_lattice.stl"
                if is_lattice_structure
                else "topology_optimized.step"
            ),
            is_lattice=is_lattice_structure,
        )
        self.props_layout.addWidget(pipeline_group)

    def _build_topopt_advanced_ui(self, node):
        """Expose structural study controls within the single default workflow."""
        names = {
            "optimizer", "max_iter", "tol", "convergence_patience", "penal",
            "automatic_filter_radius", "rmin", "density_cutoff", "stress_constraint",
            "topology_convergence_enabled", "topology_convergence_levels",
            "topology_minimum_solid_size_mm", "topology_minimum_void_size_mm",
            "topology_objective_tolerance", "topology_overlap_tolerance", "voxel_ram_fraction",
        }
        names &= set(node.properties().get("custom", {}))
        self._topopt_advanced_property_names = names
        content = QtWidgets.QGroupBox()
        content_layout = QtWidgets.QVBoxLayout(content)
        panel_layout = self.props_layout
        self.props_layout = content_layout
        try:
            self._build_generic_ui(node, property_names=names)
        finally:
            self.props_layout = panel_layout
        feature_estimate = QtWidgets.QLabel()
        feature_estimate.setWordWrap(True)
        feature_estimate.setObjectName("TopOptPhysicalLengthEstimate")
        content_layout.addWidget(feature_estimate)
        def _manual_controls():
            from pylcss.input_values import as_bool
            automatic = as_bool(node.get_property("automatic_filter_radius"))
            self.property_widgets["rmin"].setEnabled(not automatic)
            for name in ("topology_minimum_solid_size_mm", "topology_minimum_void_size_mm"):
                self.property_widgets[name].setEnabled(automatic)
            if not automatic:
                feature_estimate.setText("Manual density filter; radius is stated in voxel units.")
                return
            bounds = getattr(self, "_topopt_source_bounds", None)
            if bounds is None:
                feature_estimate.setText("Run upstream geometry to estimate physical feature sizes.")
                return
            from pylcss.cae.design_studio.topology_optimization.configuration.length_scale import resolve_physical_length_scale
            from pylcss.cae.design_studio.topology_optimization.integration.execution_setup import plan_topology_voxel_mesh
            try:
                plan = plan_topology_voxel_mesh(node, bounds)
                scale = resolve_physical_length_scale(
                    plan.bounds, plan.shape,
                    node.get_property("topology_minimum_solid_size_mm"),
                    node.get_property("topology_minimum_void_size_mm"),
                )
                feature_estimate.setText(f"Planning sizes: solid {scale.minimum_solid_size:.4g} mm · void {scale.minimum_void_size:.4g} mm · filter radius {scale.filter_radius:.4g} mm")
            except (TypeError, ValueError) as exc:
                feature_estimate.setText(str(exc))

        self.property_widgets["automatic_filter_radius"].toggled.connect(lambda _value: _manual_controls())
        for name in ("topology_minimum_solid_size_mm", "topology_minimum_void_size_mm", "voxel_ram_fraction"):
            self.property_widgets[name].value_changed.connect(lambda _value: _manual_controls())
        self.property_widgets["voxel_ram_fraction"].value_changed.connect(lambda _value: self._topopt_mesh_estimate_refresh())
        _manual_controls()
        section = InspectorSection(
            "Advanced Settings", content, parent=self.props_widget,
        )
        self.props_layout.addWidget(section)

    def _build_post_topopt_lattice_ui(self, node, combo, number):
        group = QtWidgets.QGroupBox("Generate Lattice")
        layout = QtWidgets.QFormLayout(group)
        selector = combo("post_lattice_type", PUBLIC_LATTICE_FAMILY_NAMES)
        selector.setToolTip(
            "Octet Truss: stretch-dominated structural reference. "
            "BCC: compliant struts. Gyroid: continuous curved sheets. "
            "Schwarz Primitive: straight channels. Honeycomb: directional cells. "
            "Select a family for the intended loads and manufacturing process."
        )
        layout.addRow("Lattice Type:", selector)
        for prop, label in (
            ("post_lattice_cell_size_mm", "Cell Size:"),
            ("post_lattice_min_thickness_mm", "Minimum Wall / Strut:"),
            ("post_lattice_skin_thickness_mm", "Boundary Skin:"),
        ):
            widget = number(prop, 0.0, 0.0, 1e6, decimals=4, suffix=" mm")
            widget.setSpecialValueText("Auto fit" if prop == "post_lattice_cell_size_mm" else "Automatic")
            if prop == "post_lattice_skin_thickness_mm":
                from pylcss.cae.design_studio.topology_optimization.manufacturing import family_for
                def _skin_label(value, field=widget):
                    family = family_for(value)
                    field.setSpecialValueText("None" if family and family.is_strut else "Automatic")
                _skin_label(selector.currentText())
                selector.currentTextChanged.connect(_skin_label)
            if prop == "post_lattice_cell_size_mm":
                widget.setToolTip(
                    "Auto fit measures local thickness only in design material, "
                    "excluding non-design pads. It aims for two cells across the "
                    "remaining core, subject to printable thickness and build resolution. "
                    "Features too small for a complete cell stay solid. "
                    "A positive value sets a fixed pitch; boundary cells are trimmed."
                )
            layout.addRow(label, widget)
        for prop, label, default in (
            ("post_lattice_min_density", "Minimum Cell Density:", 0.15),
            ("post_lattice_max_density", "Maximum Cell Density:", 0.60),
        ):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(2.0, 80.0)
            widget.setDecimals(1)
            widget.setSuffix("%")
            widget.setValue(100.0 * float(node.get_property(prop) or default))
            widget.valueChanged.connect(lambda value, name=prop: self.update_property(name, value / 100.0))
            layout.addRow(label, widget)
        result = getattr(node, "_last_result", None)
        active = getattr(self, "_topopt_lattice_worker", None)
        running = active is not None and active.isRunning()
        button = QtWidgets.QPushButton("Generate Lattice")
        button.setEnabled(isinstance(result, dict) and result.get("density") is not None and not running)
        button.clicked.connect(lambda _checked=False: self._generate_topopt_lattice(node))
        layout.addRow(button)
        mesh_note = QtWidgets.QLabel(
            "Creates a surface mesh for STL. Non-design regions remain solid. "
            "STEP is optional for supported strut lattices."
        )
        mesh_note.setWordWrap(True)
        layout.addRow(mesh_note)
        if isinstance(result, dict) and result.get("generated_lattice"):
            report = result["generated_lattice"]
            label = QtWidgets.QLabel(f"{report['cell_family']} · {report['cell_size_mm']:.3g} mm cells")
            label.setWordWrap(True)
            layout.addRow("Generated:", label)
            fit = report.get("fit") or {}
            if fit:
                fit_label = QtWidgets.QLabel(
                    f"{fit.get('mode', 'Fixed pitch')} · "
                    f"{100 * float(fit.get('solid_feature_fraction') or 0):.1f}% "
                    "solid"
                )
                fit_label.setWordWrap(True)
                fit_label.setToolTip(
                    "Non-design regions remain solid. This percentage reports "
                    "additional design features kept solid because printable cells "
                    "cannot fit. Boundary cells are trimmed to the envelope."
                )
                if fit.get("mode") == "Manual" and fit.get("manual_unfit_fraction"):
                    fit_label.setText(f"Manual · {100 * float(fit['manual_unfit_fraction']):.1f}% incomplete cells")
                layout.addRow("Cell Fit:", fit_label)
            restore = QtWidgets.QPushButton("Show Solid TopOpt Result")
            restore.setEnabled(not running)
            restore.clicked.connect(lambda _checked=False: self._restore_topopt_solid(node))
            layout.addRow(restore)
        self.props_layout.addWidget(group)

    def _show_post_topopt_result(self, node, result):
        node._last_result = result
        node._last_structure_options = result.get("structure_options")
        node._last_member_plan = None
        node._last_cad_reconstruction_report = result.get("cad_reconstruction")
        from .result_fields import topopt_view_mode
        mode = "Lattice" if result.get("generated_lattice") else topopt_view_mode(result, result.get("visualization_mode") or "STL Shape")
        node.set_property("visualization", mode)
        result["visualization_mode"] = mode
        if result.get("generated_lattice"):
            node.set_property("cad_export_format", "STL")
            filename = node.get_property("cad_export_filename") or "topopt_lattice.stl"
            node.set_property("cad_export_filename", str(Path(filename).with_suffix(".stl")))
        window = self.window()
        window = getattr(window, "cad_widget", window)
        window._last_render_data = result
        window._last_rendered_node = node
        window._last_rendered_geom = None
        viewer = getattr(window, "viewer", None)
        render_result = getattr(window, "_render_result_in_viewer", None)
        if callable(render_result):
            render_result(result)
        elif viewer is not None:
            viewer.render_simulation(result)
        results_panel = getattr(window, "results", None)
        if results_panel is not None:
            results_panel.show_result(result)
        QtCore.QTimer.singleShot(0, lambda: self.display_node(node) if self.current_node is node else None)

    def _restore_topopt_solid(self, node):
        from pylcss.cae.design_studio.topology_optimization.integration.post_lattice import topology_source
        result = getattr(node, "_last_result", None)
        if isinstance(result, dict):
            self._show_post_topopt_result(node, topology_source(result))

    def _generate_topopt_lattice(self, node):
        from pylcss.cae.design_studio.topology_optimization.integration.post_lattice import POST_LATTICE_DEFAULTS
        window = self.window()
        window = getattr(window, "cad_widget", window)
        if callable(getattr(window, "_execution_is_active", None)) and window._execution_is_active():
            return
        for name in ("_topopt_lattice_worker", "_topopt_fea_worker", "_topopt_cad_preview_worker", "_topopt_step_export_worker", "_topopt_mesh_export_worker"):
            active = getattr(self, name, None)
            if active is not None and active.isRunning():
                return
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            return
        worker = TopOptLatticeWorker(result, {key: node.get_property(key) for key in POST_LATTICE_DEFAULTS}, parent=self)
        self._topopt_lattice_worker = worker
        status_bar = window.statusBar() if callable(getattr(window, "statusBar", None)) else None
        outcome = {"message": "Lattice generation cancelled."}
        clear_progress = getattr(window, "_clear_solve_progress", None)
        if callable(clear_progress):
            clear_progress()
        if status_bar is not None:
            status_bar.showMessage("Generating lattice mesh...")
        dialog = QtWidgets.QProgressDialog("Generating lattice cells...", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Generate Lattice")
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.canceled.connect(worker.requestInterruption)

        def finish():
            dialog.close()
            self._topopt_lattice_worker = None
            if callable(clear_progress):
                clear_progress()
            if status_bar is not None:
                status_bar.showMessage(outcome["message"])
            QtCore.QTimer.singleShot(0, lambda: self.display_node(node) if self.current_node is node else None)
            pending = getattr(window, "_start_pending_execute", None)
            if callable(pending):
                QtCore.QTimer.singleShot(0, pending)

        def completed(output):
            # A solve or project switch may have replaced this result meanwhile.
            outcome["message"] = "Lattice mesh is ready. Export as STL."
            if getattr(node, "_last_result", None) is result:
                self._show_post_topopt_result(node, output)

        def failed(message):
            outcome["message"] = "Lattice generation failed."
            QtWidgets.QMessageBox.warning(self, "Lattice Generation", message)

        def cancelled():
            outcome["message"] = "Lattice generation cancelled."

        def progress(info, _density, _step, _total):
            stage = str(info.get("stage") or "Generating lattice mesh...")
            dialog.setLabelText(stage)
            if status_bar is not None:
                status_bar.showMessage("Lattice: " + stage)

        queued = QtCore.Qt.QueuedConnection
        worker.lattice_finished.connect(completed, queued)
        worker.lattice_error.connect(failed, queued)
        worker.lattice_cancelled.connect(cancelled, queued)
        worker.lattice_progress.connect(progress, queued)
        worker.finished.connect(finish, queued)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _build_topopt_surface_quality_ui(self, result):
        """Surface quality UI omitted to keep the TopOpt inspector clean."""
        return

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

        def _rebuild_later():
            QtCore.QTimer.singleShot(
                0,
                lambda n=node: self.display_node(n)
                if self.current_node is n
                else None,
            )

        infill_group = QtWidgets.QGroupBox("Lattice Infill Settings")
        infill_layout = QtWidgets.QFormLayout()
        infill_layout.addRow("Lattice Pattern", _combo("structure_mode", list(PUBLIC_LATTICE_FAMILY_NAMES), "Gyroid Lattice"))
        infill_layout.addRow("Relative Density", _double("lattice_target_relative_density", 0.25, 0.0001, 1.0, decimals=4, step=0.05))
        infill_layout.addRow("Build Quality", _combo("infill_build_quality", ["Standard", "High", "Professional"], "High"))
        # Swapping the mode swaps the control below it, so the panel has to be
        # rebuilt; without this the fineness picker stayed on screen in Manual
        # mode and the pitch field never appeared.
        cell_mode = _combo("infill_cell_size_mode", ["Automatic", "Manual"], "Automatic")
        cell_mode.currentTextChanged.connect(lambda _value: _rebuild_later())
        infill_layout.addRow("Cell Size Mode", cell_mode)
        if str(node.get_property("infill_cell_size_mode") or "Automatic") == "Automatic":
            infill_layout.addRow("Cell Fineness", _combo("infill_fineness", list(AUTOMATIC_FINENESS_FRACTIONS.keys()), "Medium"))
        else:
            infill_layout.addRow("Cell Size (mm)", _double("infill_cell_size_mm", 10.0, 0.1, 10000.0, decimals=2, step=1.0, suffix=" mm"))
        if str(node.get_property("infill_cell_size_mode")) == "Manual":
            coarsening = QtWidgets.QCheckBox("Allow Larger Cells")
            coarsening.setObjectName("infill_allow_pitch_coarsening")
            coarsening.setChecked(bool(node.get_property("infill_allow_pitch_coarsening")))
            coarsening.setToolTip("Allow the build budget to increase the requested pitch. Otherwise preserve it and report insufficient resolution.")
            coarsening.toggled.connect(lambda value: self.update_property("infill_allow_pitch_coarsening", value))
            infill_layout.addRow(coarsening)
        skin_mode = _combo("infill_skin_mode", ["None", "Automatic", "Manual"], "None")
        skin_mode.currentTextChanged.connect(lambda _value: _rebuild_later())
        infill_layout.addRow("Outer Skin", skin_mode)
        if str(node.get_property("infill_skin_mode")) == "Manual":
            outer_skin = _double("infill_skin_thickness_mm", 0.0, 0.0, 10000.0, decimals=3, step=0.1, suffix=" mm")
            outer_skin.setToolTip("Requested thickness is preserved. The remaining lattice core depends on the part's local thickness.")
            infill_layout.addRow("Skin Thickness", outer_skin)
        repair = QtWidgets.QCheckBox("Repair Trimmed Connections")
        repair.setChecked(bool(node.get_property("repair_lattice_connections")))
        repair.toggled.connect(lambda value: self.update_property("repair_lattice_connections", value))
        infill_layout.addRow(repair)
        infill_group.setLayout(infill_layout)
        self.props_layout.addWidget(infill_group)

        self._build_lattice_infill_results(node)
        self._build_lattice_infill_export(node, _text)

    def _build_lattice_infill_results(self, node):
        """Report the delivered lattice, not the optimizer's study report.

        A lattice infill has no objective, no load case and no iteration, so
        the topology result rows -- material budget, compliance, objective
        improvement, iterations used, convergence -- are all undefined for it.
        What it does have is a requested relative density and a built one, and
        those were the two numbers with nowhere to appear.
        """
        result = getattr(node, "_last_result", None)
        summary = (
            result.get("lattice_infill") if isinstance(result, dict) else None
        )
        if not isinstance(summary, dict):
            return

        group = QtWidgets.QGroupBox("Result")
        layout = QtWidgets.QFormLayout()

        requested = float(summary.get("requested_relative_density") or 0.0)
        achieved = summary.get("achieved_relative_density")
        if achieved is not None:
            achieved_value = float(achieved)
            density_label = QtWidgets.QLabel(
                f"{achieved_value:.1%} built · {requested:.0%} requested"
            )
            # A search that lands outside a point of the request means the
            # thickness hit the raster floor or the cell ceiling, and the part
            # is not the mass that was asked for.
            on_target = abs(achieved_value - requested) <= 0.01
            density_label.setStyleSheet(
                "color:#66d17a; font-size:11px;"
                if on_target
                else "color:#ffb74d; font-size:11px;"
            )
            layout.addRow("Relative density", density_label)

        member = summary.get("member_thickness_model_units")
        if member is not None:
            layout.addRow(
                "Wall / strut", QtWidgets.QLabel(f"{float(member):.3g} mm")
            )
        built_pitch = summary.get("cell_size")
        requested_pitch = summary.get("requested_cell_size")
        if built_pitch:
            coarsened = bool(summary.get("pitch_coarsened")) and requested_pitch
            pitch_label = QtWidgets.QLabel(
                f"{float(built_pitch):.3g} mm "
                f"(asked {float(requested_pitch):.3g} mm)"
                if coarsened
                else f"{float(built_pitch):.3g} mm"
            )
            pitch_label.setWordWrap(True)
            if coarsened:
                pitch_label.setToolTip(
                    "The requested pitch needs a finer grid than this build "
                    "quality allows. A larger cell was built so the relative "
                    "density is still the one requested; raise Build Quality "
                    "for the pitch you asked for."
                )
                pitch_label.setStyleSheet("color:#ffb74d; font-size:11px;")
            layout.addRow("Cell pitch", pitch_label)
        reach = summary.get("envelope_reach")
        if reach is not None:
            reach_label = QtWidgets.QLabel(f"{float(reach):.0%} of the body")
            reach_label.setStyleSheet(
                "color:#8f98a5; font-size:10px;"
                if float(reach) >= 0.99
                else "color:#ffb74d; font-size:11px;"
            )
            layout.addRow("Filled", reach_label)

        manufacturing = (
            result.get("manufacturing") if isinstance(result, dict) else None
        )
        surface = (
            (manufacturing or {}).get("surface_quality")
            if isinstance(manufacturing, dict)
            else None
        )
        if isinstance(surface, dict):
            watertight = surface.get("is_watertight", surface.get("watertight"))
            faces = surface.get("faces")
            bodies = surface.get("connected_components")
            parts = []
            if faces:
                parts.append(f"{int(faces):,} triangles")
            if bodies:
                parts.append(f"{int(bodies)} body" if int(bodies) == 1 else f"{int(bodies)} bodies")
            if watertight is not None:
                parts.append("watertight" if watertight else "not watertight")
            if parts:
                mesh_label = QtWidgets.QLabel(" · ".join(parts))
                mesh_label.setWordWrap(True)
                mesh_label.setStyleSheet(
                    "color:#8f98a5; font-size:10px;"
                    if watertight is not False
                    else "color:#ff8a80; font-size:10px;"
                )
                layout.addRow("Mesh", mesh_label)

        grid_shape = [
            int(node.get_property(name) or 0) for name in ("nelx", "nely", "nelz")
        ]
        if all(value > 0 for value in grid_shape):
            grid_label = QtWidgets.QLabel(
                "{} × {} × {} voxels".format(*grid_shape)
            )
            grid_label.setStyleSheet("color:#8f98a5; font-size:10px;")
            layout.addRow("Build grid", grid_label)

        for label, value in result_state_rows(result):
            layout.addRow(label, QtWidgets.QLabel(value))

        case_compliances = (result or {}).get("case_compliances")
        if isinstance(case_compliances, dict) and len(case_compliances) > 1:
            cases_grp = QtWidgets.QGroupBox("Load Cases Performance")
            cases_lay = QtWidgets.QFormLayout(cases_grp)
            disps = (result or {}).get("case_max_displacements") or {}
            for cname, ccomp in case_compliances.items():
                cdisp = disps.get(cname)
                disp_str = f" · Max disp {cdisp:.4f} mm" if cdisp is not None else ""
                lbl = QtWidgets.QLabel(f"{float(ccomp):.3e} N·mm{disp_str}")
                lbl.setStyleSheet("font-size: 11px;")
                cases_lay.addRow(f"Case {cname}:", lbl)
            self.props_layout.addWidget(cases_grp)

        group.setLayout(layout)
        self.props_layout.addWidget(group)

    def _build_lattice_infill_export(self, node, _text):
        """Offer the same export routes the lattice study does.

        Nothing about export is specific to how the density field was produced,
        so an infill releases through exactly the same workers: an exact B-rep
        for the strut families, native 3MF Beam Lattice for their centrelines,
        and STL for the minimal-surface and honeycomb cells that have no
        compact boundary representation.
        """
        group = self._build_export_group(
            node, default_filename="lattice_infill.step", is_lattice=True
        )
        self.props_layout.addWidget(group)

    def _set_topopt_visualization(self, node, mode):
        """Switch between computed views without starting another operation."""
        mode = str(mode)
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or mode not in available_result_fields(result):
            return
        result["visualization_mode"] = mode
        self.update_property("visualization", mode)

    def _build_and_show_lattice_infill(self, node):
        window = self.window()
        window._last_rendered_node = node
        window._last_rendered_geom = None
        window._section_new_infill = node
        window._execute_graph(nodes=window._upstream_closure(node))

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

        strategy = 'auto'
        cutoff = float(
            result.get("density_cutoff")
            or node.get_property("density_cutoff")
            or 0.30
        )
        return {
            "passive_regions": passive_regions,
            "strategy": strategy,
            "cutoff": cutoff,
        }

    def _preview_topopt_cad(self, node):
        """Build, cache, and display the actual STEP-ready TopOpt B-rep."""
        if "STEP" not in self._topopt_available_formats(node):
            return
        window = self.window()
        window = getattr(window, "cad_widget", window)
        worker = getattr(self, "_topopt_cad_preview_worker", None)
        if worker is not None and worker.isRunning():
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage("CAD build already running.")
            return

        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or (
            result.get("recovered_shape") is None and result.get("density") is None
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "No Result",
                "Run Topology first.",
            )
            return

        settings = self._topopt_cad_settings(node, result)
        worker = TopOptCadPreviewWorker(
            result,
            density_cutoff=settings["cutoff"],
            passive_regions=settings["passive_regions"],
            reconstruction_strategy=settings["strategy"],
            parent=self,
        )
        self._topopt_cad_preview_worker = worker
        if self.current_node is node:
            button = self.props_widget.findChild(QtWidgets.QPushButton, "TopOptBuildCad")
            if button is not None:
                button.setEnabled(False)
            hint = self.props_widget.findChild(QtWidgets.QLabel, "TopOptCadHint")
            if hint is not None:
                hint.setText("Building CAD in the background.")

        def _refresh_cad_inspector():
            from pylcss.cae.design_studio.topology_optimization.integration.study_identity import (
                is_density_study_node,
            )

            current = self.current_node
            if current is not None and is_density_study_node(current):
                self.display_node(current)

        def _finish(shape, report):
            self._topopt_cad_preview_worker = None
            report = dict(report or {})
            layout = report.pop('_surface_layout', None)
            if layout is not None:
                result['surface_layout'] = layout
                if isinstance(result.get('recovered_shape'), dict):
                    result['recovered_shape']['surface_layout'] = layout
            result["cad_shape"] = shape
            if result.get("validation"):
                result.pop("validation", None)
                result.pop("validation_summary", None)
                result["fea_analysis_status"] = "geometry_changed"
                from pylcss.cae.design_studio.topology_optimization.verification.quality_gate import update_verification_status
                update_verification_status(result)
            result["shape"] = shape
            result["cad_reconstruction"] = report
            result.pop("cad_error", None)
            result["cad_preview_mesh"] = worker.preview_mesh
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

            clear_progress = getattr(window, "_clear_solve_progress", None)
            if callable(clear_progress):
                clear_progress()
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

            face_count = report.get(
                "cad_face_count", report.get("cad_face_count_after_feature_healing")
            )
            count_text = (
                f", {int(face_count)} faces" if face_count is not None else ""
            )
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage(
                    f"CAD ready{count_text}",
                    12000,
                )
            QtCore.QTimer.singleShot(0, _refresh_cad_inspector)

        def _fail(message):
            self._topopt_cad_preview_worker = None
            result["cad_error"] = message
            result["cad_reconstruction"] = {
                "method": "CAD reconstruction failed", "failed": True,
                "editable": False, "error": message,
            }
            mode = "Lattice" if result.get("generated_lattice") else "STL Shape"
            if mode not in available_result_fields(result):
                mode = "Density"
            result["visualization_mode"] = mode
            node.set_property("visualization", mode)
            viewer = getattr(window, "viewer", None)
            if viewer is not None:
                viewer.render_simulation(result)
            clear_progress = getattr(window, "_clear_solve_progress", None)
            if callable(clear_progress):
                clear_progress()
            QtWidgets.QMessageBox.critical(
                self,
                "CAD Error",
                message,
            )
            QtCore.QTimer.singleShot(0, _refresh_cad_inspector)

        def _cancelled():
            self._topopt_cad_preview_worker = None
            mode = "Lattice" if result.get("generated_lattice") else "STL Shape"
            if mode not in available_result_fields(result):
                mode = "Density"
            result["visualization_mode"] = mode
            node.set_property("visualization", mode)
            viewer = getattr(window, "viewer", None)
            if viewer is not None:
                viewer.render_simulation(result)
            clear_progress = getattr(window, "_clear_solve_progress", None)
            if callable(clear_progress):
                clear_progress()
            if hasattr(window, "statusBar") and window.statusBar():
                window.statusBar().showMessage("CAD reconstruction cancelled.")
            QtCore.QTimer.singleShot(0, _refresh_cad_inspector)

        worker.preview_finished.connect(_finish)
        worker.preview_error.connect(_fail)
        worker.preview_cancelled.connect(_cancelled)
        worker.finished.connect(worker.deleteLater)
        progress_slot = getattr(window, "_on_optimization_step", None)
        if callable(progress_slot):
            worker.preview_progress.connect(progress_slot)
        if hasattr(window, "statusBar") and window.statusBar():
            window.statusBar().showMessage("Building CAD...")
        worker.start()

    def _refresh_topopt_recovered_shape(self, node, result):
        """Rebuild recovered_shape from the current density before export."""
        if isinstance(result, dict) and result.get("generated_lattice"):
            return result.get("recovered_shape")
        if not isinstance(result, dict) or result.get("density") is None:
            return result.get("recovered_shape") if isinstance(result, dict) else None
        try:
            import numpy as np
            from pylcss.cae.design_studio.topology_optimization.geometry.surface_recovery import (
                _recover_voxel_shape,
            )
            from pylcss.cae.design_studio.topology_optimization.geometry.recovery_payload import (
                surface_recovery_constraints,
            )
            from pylcss.cae.design_studio.topology_optimization.integration.lattice_settings import (
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
                **surface_recovery_constraints(result),
                extrusion_axis="none",
                structure_options=structure_options,
                member_plan=getattr(node, "_last_member_plan", None),
            )
            if recovered is not None and len(recovered.get("faces", [])) > 0:
                result["recovered_shape"] = recovered
                result.pop("cad_shape", None)
                result.pop("shape", None)
                result.pop("cad_reconstruction", None)
                result.pop("cad_preview_mesh", None)
                setattr(node, "_last_result", result)
                return recovered
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        return result.get("recovered_shape")

    def _topopt_available_formats(self, node):
        result = getattr(node, "_last_result", None)
        if isinstance(result, dict):
            return topopt_export_formats(result)
        if callable(getattr(node, "resolve_infill_cell_size", None)):
            return ("STL",)
        return topopt_export_formats(node.get_property("structure_mode"))

    def _build_export_group(
        self, node, default_filename: str = "topology_optimized.step", is_lattice: bool = False
    ) -> QtWidgets.QGroupBox:
        """Create the unified Export panel with File Name, Browse, Export As, and a single Export button."""
        group = QtWidgets.QGroupBox("Export")
        layout = QtWidgets.QFormLayout()
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        formats = self._topopt_available_formats(node)
        cur_fmt = str(node.get_property("cad_export_format") or formats[0]).upper()
        if cur_fmt not in formats:
            cur_fmt = formats[0]
            node.set_property("cad_export_format", cur_fmt)
        ext_map = {"STEP": ".step", "STL": ".stl", "3MF": ".3mf"}
        cur_ext = ext_map.get(cur_fmt, ".step")

        cur_name = str(node.get_property("cad_export_filename") or default_filename)
        p_name = Path(cur_name)
        if p_name.suffix.lower() in [".step", ".stp", ".stl", ".3mf"]:
            cur_name = str(p_name.with_suffix(cur_ext))

        # 1) File Name row with line edit + Browse button
        file_container = QtWidgets.QWidget()
        file_layout = QtWidgets.QHBoxLayout(file_container)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(4)

        file_edit = QtWidgets.QLineEdit(cur_name)
        file_edit.setToolTip("Destination path and filename for the exported model.")
        file_edit.editingFinished.connect(
            lambda: self.update_property("cad_export_filename", file_edit.text().strip())
        )
        file_layout.addWidget(file_edit, 1)

        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.setToolTip("Choose destination directory and filename.")
        file_layout.addWidget(btn_browse, 0)

        layout.addRow("File Name:", file_container)

        # 2) Export As combo box
        format_combo = QtWidgets.QComboBox()
        format_labels = {"STEP": "STEP (*.step)", "STL": "STL (*.stl)", "3MF": "3MF Lattice (*.3mf)"}
        for fmt in formats:
            format_combo.addItem(format_labels[fmt], fmt)

        idx = format_combo.findData(cur_fmt)
        if idx >= 0:
            format_combo.setCurrentIndex(idx)

        def _on_format_changed(index):
            fmt_data = str(format_combo.itemData(index) or "STEP").upper()
            node.set_property("cad_export_format", fmt_data)
            new_ext = ext_map.get(fmt_data, ".step")
            txt = file_edit.text().strip()
            if txt:
                p = Path(txt)
                if p.suffix.lower() in [".step", ".stp", ".stl", ".3mf"]:
                    updated = str(p.with_suffix(new_ext))
                else:
                    updated = txt + new_ext
                file_edit.setText(updated)
                self.update_property("cad_export_filename", updated)

        format_combo.currentIndexChanged.connect(_on_format_changed)
        layout.addRow("Export As:", format_combo)

        # Browse handler
        def _on_browse():
            fmt_data = str(format_combo.currentData() or "STEP").upper()
            filters = {
                "STEP": "STEP Files (*.step *.stp)",
                "STL": "STL Files (*.stl)",
                "3MF": "3MF Files (*.3mf)",
            }
            filter_str = filters.get(fmt_data, "All Files (*.*)")
            cur_path = file_edit.text().strip()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, f"Export {fmt_data}", cur_path, filter_str
            )
            if path:
                req_ext = ext_map.get(fmt_data, "")
                if req_ext and not path.lower().endswith(req_ext):
                    path += req_ext
                file_edit.setText(path)
                self.update_property("cad_export_filename", path)

        btn_browse.clicked.connect(_on_browse)

        # 3) Single unified Export button
        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setStyleSheet("font-weight: bold; padding: 5px;")
        btn_export.setToolTip("Export the model using the selected format and filename.")
        btn_export.clicked.connect(lambda: self._export_topopt_unified(node, format_combo, file_edit))
        layout.addRow("", btn_export)

        group.setLayout(layout)
        return group

    def _ensure_recovered_surface(self, node, result: dict) -> dict | None:
        """Ensure result has a recovered surface mesh, computing it on demand if needed."""
        recovered = result.get("recovered_shape")
        if isinstance(recovered, dict) and len(recovered.get("vertices", [])) > 0:
            return recovered
        field = result.get("recovery_density") if result.get("recovery_density") is not None else result.get("density")
        if field is None:
            return None
        from pylcss.cae.design_studio.topology_optimization.geometry.cad_features import (
            _payload_bounds,
        )
        bounds = _payload_bounds(result)
        if bounds is None:
            bb_min = node.get_property("bounding_box_min") or [0.0, 0.0, 0.0]
            bb_max = node.get_property("bounding_box_max") or list(np.shape(field))
            bounds = (bb_min, bb_max)
        cutoff = float(result.get("recovery_cutoff") or result.get("density_cutoff") or 0.5)
        try:
            from pylcss.cae.design_studio.topology_optimization.geometry.surface_recovery import (
                _recover_voxel_shape,
            )
            from pylcss.cae.design_studio.topology_optimization.geometry.recovery_payload import (
                surface_recovery_constraints,
            )
            recovered = _recover_voxel_shape(
                np.asarray(field, dtype=float),
                bounds,
                cutoff,
                **surface_recovery_constraints(result),
                extrusion_axis=result.get("extrusion_axis") or "none",
                structure_options=result.get("structure_options"),
                member_plan=result.get("member_plan"),
            )
            if isinstance(recovered, dict):
                result["recovered_shape"] = recovered
                setattr(node, "_last_recovered_shape", recovered)
                return recovered
        except Exception as exc:
            logging.getLogger(__name__).warning("Auto surface recovery on export failed: %s", exc)
        return None

    def _export_topopt_unified(self, node, format_combo=None, file_edit=None):
        """Export geometry to the user's chosen file and format without failure."""
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
            QtWidgets.QMessageBox.warning(
                self,
                "No Result",
                "Run the study first before exporting.",
            )
            return

        fmt = "STEP"
        if format_combo is not None:
            fmt = str(format_combo.currentData() or "STEP").upper()
        else:
            fmt = str(node.get_property("cad_export_format") or "STEP").upper()

        path = ""
        if file_edit is not None:
            path = file_edit.text().strip()
        if not path:
            path = str(node.get_property("cad_export_filename") or "").strip()

        ext_map = {"STEP": ".step", "STL": ".stl", "3MF": ".3mf"}
        req_ext = ext_map.get(fmt, ".step")

        # If user did not specify an absolute path, prompt with file chooser
        if not path or not os.path.isabs(path):
            filters = {
                "STEP": "STEP Files (*.step *.stp)",
                "STL": "STL Files (*.stl)",
                "3MF": "3MF Files (*.3mf)",
            }
            filter_str = filters.get(fmt, "All Files (*.*)")
            default_name = path if path else f"topology_optimized{req_ext}"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, f"Export {fmt}", default_name, filter_str
            )
            if not path:
                return
            if not path.lower().endswith(req_ext):
                path += req_ext
            if file_edit is not None:
                file_edit.setText(path)
            self.update_property("cad_export_filename", path)

        if fmt == "STEP":
            self._execute_export_step(node, path)
        elif fmt == "STL":
            self._execute_export_stl(node, path)
        elif fmt == "3MF":
            self._execute_export_3mf(node, path)

    def _execute_export_step(self, node, export_path: str):
        if "STEP" not in self._topopt_available_formats(node):
            QtWidgets.QMessageBox.information(self, "STL Geometry", "Export this complete lattice body as STL. Member-only STEP would omit its outer skin.")
            return
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(self, "No Result", "Run the study first before exporting.")
            return

        worker = getattr(self, "_topopt_step_export_worker", None)
        if worker is not None and worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("STEP export already running.")
            return

        from pylcss.cae.design_studio.topology_optimization.geometry.cad_config import cached_cad_shape
        settings = self._topopt_cad_settings(node, result)
        cad_shape = cached_cad_shape(result, settings["strategy"])

        node.set_property("cad_export_filename", export_path)
        worker = TopOptStepExportWorker(
            result,
            export_path,
            density_cutoff=settings["cutoff"],
            passive_regions=settings["passive_regions"],
            reconstruction_strategy=settings["strategy"],
            cached_shape=cad_shape,
            parent=self,
        )
        self._topopt_step_export_worker = worker

        def _finish(out_path):
            self._topopt_step_export_worker = None
            # Only attach to the result this export used: a newer solve must
            # never receive a body prepared from an earlier density field.
            if worker.reconstruction_report is not None and getattr(node, "_last_result", None) is result:
                result["cad_shape"] = worker.cached_shape
                result["shape"] = worker.cached_shape
                result["cad_reconstruction"] = worker.reconstruction_report
                result["cad_preview_mesh"] = worker.preview_mesh
                node._last_cad_reconstruction_report = worker.reconstruction_report
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(f"STEP exported: {out_path}")

        def _fail(msg):
            self._topopt_step_export_worker = None
            QtWidgets.QMessageBox.critical(self, "Export Error", msg)

        worker.export_finished.connect(_finish)
        worker.export_error.connect(_fail)
        worker.finished.connect(worker.deleteLater)
        if hasattr(self.window(), "statusBar") and self.window().statusBar():
            self.window().statusBar().showMessage("Exporting CAD body as STEP...")
        worker.start()

    def _execute_export_stl(self, node, export_path: str):
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(self, "No Result", "Run the study first before exporting.")
            return

        active_worker = getattr(self, "_topopt_mesh_export_worker", None)
        if active_worker is not None and active_worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("A mesh export is already running.")
            return

        from pylcss.cae.design_studio.topology_optimization.geometry.cad_config import cached_cad_shape
        settings = self._topopt_cad_settings(node, result)
        structure_options = result.get("structure_options")
        cad_shape = cached_cad_shape(result, settings["strategy"])
        recovered = result.get("recovered_shape")

        # Recovery is expensive too; let the isolated export job do it. A
        # filename change or Export click must never rebuild the surface here.
        if (cad_shape is None and not isinstance(recovered, dict)
                and result.get("recovery_density") is None and result.get("density") is None):
            QtWidgets.QMessageBox.warning(
                self,
                "No Geometry",
                "Could not recover surface for STL export. Ensure the study produced a density field.",
            )
            return

        node.set_property("cad_export_filename", export_path)
        worker = TopOptMeshExportWorker(
            export_path,
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
            if isinstance(details.get("recovered_shape"), dict):
                result["recovered_shape"] = details["recovered_shape"]
            triangle_count = int(details.get("triangles", 0))
            source = str(details.get("source") or "")
            suffix = f" from {source}" if source else ""
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported {triangle_count} STL triangles{suffix} to {details['path']}"
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

    def _execute_export_3mf(self, node, export_path: str):
        result = getattr(node, "_last_result", None)
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(self, "No Result", "Run the study first before exporting.")
            return

        active_worker = getattr(self, "_topopt_mesh_export_worker", None)
        if active_worker is not None and active_worker.isRunning():
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage("An export is already running.")
            return

        structure_options = result.get("structure_options")
        from pylcss.cae.design_studio.topology_optimization.geometry.lattice_cad import lattice_cad_strategy
        if lattice_cad_strategy(structure_options) != "beam":
            QtWidgets.QMessageBox.information(
                self,
                "Not a Strut Lattice",
                "3MF Beam Lattice export is only available for strut lattices (BCC, Octet). "
                "For other shapes, export as STL or STEP.",
            )
            return

        node.set_property("cad_export_filename", export_path)
        worker = TopOptMeshExportWorker(
            export_path,
            topology_result=result,
            structure_options=structure_options,
            member_plan=result.get("member_plan"),
            export_format="3mf",
            parent=self,
        )
        self._topopt_mesh_export_worker = worker

        def _finish(details):
            self._topopt_mesh_export_worker = None
            beam_count = int(details.get("beams", 0))
            if hasattr(self.window(), "statusBar") and self.window().statusBar():
                self.window().statusBar().showMessage(
                    f"Exported {beam_count} 3MF beams to {details['path']}"
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

    def _export_topopt_step(self, node):
        """Legacy helper: exports STEP."""
        default_name = str(node.get_property("cad_export_filename") or "topology_optimized.step")
        if not default_name.lower().endswith((".step", ".stp")):
            default_name = str(Path(default_name).with_suffix(".step"))
        self._execute_export_step(node, default_name)

    def _export_topopt_stl(self, node):
        """Legacy helper: exports STL."""
        default_name = str(node.get_property("cad_export_filename") or "topology_optimized.stl")
        if not default_name.lower().endswith(".stl"):
            default_name = str(Path(default_name).with_suffix(".stl"))
        self._execute_export_stl(node, default_name)

    def _export_topopt_beam_3mf(self, node):
        """Legacy helper: exports 3MF."""
        default_name = str(node.get_property("cad_export_filename") or "topology_optimized.3mf")
        if not default_name.lower().endswith(".3mf"):
            default_name = str(Path(default_name).with_suffix(".3mf"))
        self._execute_export_3mf(node, default_name)

