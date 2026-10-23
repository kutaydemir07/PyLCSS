# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Analyse FEA action and results in the TopOpt inspector."""

import numpy as np
from PySide6 import QtCore, QtWidgets

from .topopt_fea_worker import TopOptFeaWorker
from pylcss.cae.design_studio.topology_optimization.configuration.fea_meshing import (
    FEA_ELEMENT_ORDERS,
    FEA_MESH_DEFAULTS,
    FEA_QUALITY_OPTIONS,
    fea_mesh_options_from_node,
)
from pylcss.cae.design_studio.topology_optimization.verification.fea_analysis import (
    topopt_fea_unavailable_reason,
)


from pylcss.cae.design_studio.topology_optimization.verification.quality_gate import (
    VERIFICATION_DEFAULTS, update_verification_status,
)


class TopOptFeaInspectorMixin:
    def _build_topopt_fea_ui(self, node):
        result = getattr(node, "_last_result", None)
        result = result if isinstance(result, dict) else {}
        worker = getattr(self, "_topopt_fea_worker", None)
        running = worker is not None and worker.isRunning()
        group = QtWidgets.QGroupBox("FEA Analysis")
        layout = QtWidgets.QFormLayout(group)

        def changed(name, value):
            self.update_property(name, value)
            if (result.get("validation") or {}).get("source") in {'topopt_cad_fea', 'topopt_lattice_fea'}:
                if name in VERIFICATION_DEFAULTS:
                    update_verification_status(
                        result,
                        safety_factor=node.get_property("validation_yield_safety_factor"),
                        displacement_limit=node.get_property("validation_displacement_limit_mm"),
                    )
                else:
                    result["fea_analysis_status"] = "settings_changed"
                    update_verification_status(result)
                    self._topopt_fea_status_label.setText("Settings changed — analyse again")
                self._topopt_verification_label.setText((result.get("quality_gate") or {}).get("status", "Not checked"))
                self._topopt_verification_label.setToolTip("\n".join(result.get("warnings") or []))

        mesh_group = QtWidgets.QGroupBox("FEA Meshing")
        mesh_layout = QtWidgets.QFormLayout(mesh_group)
        mesh_note = QtWidgets.QLabel(
            "Tetrahedral mesh of the current CAD solid or generated lattice material. "
            "TopOpt voxel sizing is separate."
        )
        mesh_note.setWordWrap(True)
        mesh_layout.addRow(mesh_note)

        def number(name, *, minimum=0.0, maximum=1e6, step=0.1, decimals=6, scale=1.0):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setObjectName(name)
            widget.setDecimals(decimals)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setKeyboardTracking(False)
            value = node.get_property(name)
            default = ({**FEA_MESH_DEFAULTS, **VERIFICATION_DEFAULTS})[name]
            widget.setValue(scale * (default if value is None else float(value)))
            widget.setEnabled(not running)
            widget.valueChanged.connect(lambda value: changed(name, value / scale))
            return widget

        maximum_size = number("fea_element_size_mm")
        maximum_size.setSuffix(" mm")
        maximum_size.setSpecialValueText("Automatic")
        maximum_size.setToolTip(
            "Global maximum tetrahedron size. Automatic uses the smaller of "
            "longest CAD dimension / 20 and shortest dimension / 2. "
            "This is independent of the original Mesh node."
        )
        mesh_layout.addRow("Maximum size:", maximum_size)
        minimum_size = number("fea_min_element_size_mm")
        minimum_size.setSuffix(" mm")
        minimum_size.setSpecialValueText("Mesher default")
        minimum_size.setToolTip(
            "Lower bound for local mesh sizing from edges and curvature. "
            "Must not exceed maximum size. Tiny CAD edges may still need repair."
        )
        mesh_layout.addRow("Minimum size:", minimum_size)
        surface_tolerance = number('fea_surface_tolerance_mm')
        surface_tolerance.setSuffix(' mm')
        surface_tolerance.setSpecialValueText('Automatic')
        surface_tolerance.setToolTip('Lattice FEA surface approximation only; the exported geometry is unchanged. '
            'Automatic limits movement to 2% of the minimum wall, 10% of contour spacing, and 5% of element size. '
            'The meshing report records the bound and volume change.')
        mesh_layout.addRow('Lattice surface tolerance:', surface_tolerance)
        backend = QtWidgets.QComboBox()
        backend.addItems(['Automatic', 'Sparse Direct', 'CalculiX'])
        backend.setCurrentText(node.get_property('fea_solver_backend') or 'Automatic')
        backend.setEnabled(not running)
        backend.setToolTip('Automatic uses sparse direct linear analysis when available, '
                           'and CalculiX for other supported models. Both analyse the current solid mesh.')
        backend.currentTextChanged.connect(lambda value: changed('fea_solver_backend', value))
        mesh_layout.addRow('Linear solver:', backend)
        grading = number(
            "fea_mesh_grading", minimum=0.01, maximum=1.0, step=0.05, decimals=2
        )
        grading.setToolTip(
            "How quickly element sizes change across the mesh. Smaller values "
            "give smoother transitions and can require more elements."
        )
        mesh_layout.addRow("Grading:", grading)
        order = QtWidgets.QComboBox()
        order.setObjectName("fea_element_order")
        for title, value in FEA_ELEMENT_ORDERS.items():
            order.addItem(title, value)
        order.setCurrentIndex(
            max(0, order.findData(node.get_property("fea_element_order") or "C3D10"))
        )
        order.setEnabled(not running)
        order.setToolTip(
            "Quadratic tetrahedra use ten nodes and are the default for stress "
            "analysis. Linear tetrahedra use four nodes and fewer solver unknowns."
        )
        order.currentIndexChanged.connect(
            lambda _index: changed("fea_element_order", order.currentData())
        )
        mesh_layout.addRow("Elements:", order)
        limit = QtWidgets.QSpinBox()
        limit.setObjectName("fea_max_elements")
        limit.setRange(1, 100_000_000)
        limit.setKeyboardTracking(False)
        value = node.get_property("fea_max_elements")
        limit.setValue(
            FEA_MESH_DEFAULTS["fea_max_elements"] if value is None else int(value)
        )
        limit.setEnabled(not running)
        limit.setToolTip(
            "Reject a completed mesh above this tetrahedron count before "
            "solving. Checked after Netgen finishes; this is not a meshing RAM limit."
        )
        limit.valueChanged.connect(lambda value: changed("fea_max_elements", value))
        mesh_layout.addRow("Maximum elements:", limit)
        layout.addRow(mesh_group)
        quality = QtWidgets.QComboBox()
        quality.setObjectName("fea_quality")
        quality.addItems(FEA_QUALITY_OPTIONS)
        quality.setCurrentText(node.get_property("fea_quality") or "Standard")
        quality.setEnabled(not running)
        quality.setToolTip(
            "Standard solves one mesh. Mesh Convergence refines the CAD mesh, or locally "
            "refines lattice stress variations and hotspots. Both CAD and lattice checks compare compliance and peak stress. "
            "Refinement stops on convergence, the element limit, or four levels."
        )
        quality.currentTextChanged.connect(lambda value: changed("fea_quality", value))
        layout.addRow("Analysis mode:", quality)
        tolerance = number("fea_convergence_tolerance", scale=100, step=1, decimals=2)
        tolerance.setSuffix(" %")
        tolerance.setToolTip(
            "Maximum relative change of compliance and peak stress between successive mesh levels. "
            "Used when Analysis mode is Mesh Convergence."
        )
        layout.addRow("Convergence tolerance:", tolerance)
        limits_group = QtWidgets.QGroupBox("Verification Limits")
        limits_layout = QtWidgets.QFormLayout(limits_group)
        factor = number("validation_yield_safety_factor", minimum=1.0, step=0.1, decimals=2)
        limits_layout.addRow("Yield safety factor:", factor)
        displacement = number("validation_displacement_limit_mm")
        displacement.setSuffix(" mm")
        displacement.setSpecialValueText("No limit")
        limits_layout.addRow("Displacement limit:", displacement)
        layout.addRow(limits_group)
        button = QtWidgets.QPushButton("Analysing FEA..." if running else "Analyse FEA")
        button.setObjectName("topopt_analyse_fea")
        unavailable = topopt_fea_unavailable_reason(result)
        button.setEnabled(not running and not unavailable)
        button.setToolTip(
            "Run CalculiX on the current CAD or actual lattice walls using the connected material, supports and loads."
        )
        button.clicked.connect(lambda _checked=False: self._analyse_topopt_fea(node))
        layout.addRow(button)
        if running:
            status = str(result.get("fea_analysis_stage") or "Running")
        elif result.get("fea_analysis_error"):
            status = "Failed"
        elif result.get("fea_analysis_status") == "cancelled":
            status = "Cancelled"
        elif result.get("fea_analysis_status") == "settings_changed":
            status = "Settings changed — analyse again"
        elif result.get("fea_analysis_status") == "geometry_changed":
            status = "Geometry changed — analyse again"
        elif unavailable:
            status = unavailable
        elif (result.get("validation") or {}).get("source") in {'topopt_cad_fea', 'topopt_lattice_fea'}:
            status = "Complete"
        else:
            status = "Ready"
        label = QtWidgets.QLabel(status)
        label.setWordWrap(True)
        label.setToolTip(str(result.get("fea_analysis_error") or ""))
        self._topopt_fea_status_label = label
        layout.addRow("Status:", label)
        verification = QtWidgets.QLabel((result.get("quality_gate") or {}).get("status", "Not checked"))
        verification.setToolTip("\n".join(result.get("warnings") or []))
        self._topopt_verification_label = verification
        layout.addRow("Checks:", verification)
        analysis = result.get("validation")
        if isinstance(analysis, dict) and analysis.get("source") in {'topopt_cad_fea', 'topopt_lattice_fea'}:
            study = analysis.get("convergence_study") or {}
            levels = study.get("levels") or []
            converged = study.get("converged")
            layout.addRow(
                "Mesh convergence:", QtWidgets.QLabel(
                    "Not assessed" if converged is None else "Converged" if converged else "Not converged"
                ),
            )
            if levels:
                final = levels[-1]
                layout.addRow(
                    "Mesh:",
                    QtWidgets.QLabel(
                        f"{final['n_elements']:,} elements · {final['n_nodes']:,} nodes"
                    ),
                )
                layout.addRow(
                    "Element type:",
                    QtWidgets.QLabel(str(analysis.get("element_order") or "C3D10")),
                )
            for key, title, unit in (
                ("stress", "Peak stress:", "MPa"),
                ("displacement", "Peak displacement:", "mm"),
            ):
                values = analysis.get(key)
                if values is not None and np.asarray(values).size:
                    values = np.asarray(values, dtype=float)
                    if key == "displacement":
                        values = np.linalg.norm(values.reshape(-1, 3), axis=1)
                    layout.addRow(
                        title, QtWidgets.QLabel(f"{float(np.max(values)):.4g} {unit}")
                    )
        self.props_layout.addWidget(group)

    def _analyse_topopt_fea(self, node):
        window = self.window()
        window = getattr(window, "cad_widget", window)
        for active in (
            getattr(window, "worker", None),
            getattr(self, "_topopt_fea_worker", None),
            getattr(self, "_topopt_lattice_worker", None),
            getattr(self, "_topopt_cad_preview_worker", None),
        ):
            if active is not None and active.isRunning():
                QtWidgets.QMessageBox.information(
                    self,
                    "Run Active",
                    "Wait for the current run to finish before analysing FEA.",
                )
                return
        result = getattr(node, "_last_result", None)
        unavailable = topopt_fea_unavailable_reason(result)
        if unavailable:
            QtWidgets.QMessageBox.warning(self, "CAD FEA", unavailable)
            return
        worker = TopOptFeaWorker(
            result,
            node.get_input_value("material"),
            node.get_input_list("supports"),
            node.get_input_list("loads"),
            quality=node.get_property("fea_quality") or "Standard",
            mesh_options=fea_mesh_options_from_node(node),
            parent=self,
        )
        self._topopt_fea_worker = worker
        result.pop("fea_analysis_error", None)
        result["fea_analysis_status"] = "running"
        update_verification_status(result)
        result["fea_analysis_stage"] = "Preparing current geometry mesh"

        def _current_result():
            return getattr(node, "_last_result", None) is result

        def _progress(message):
            if not _current_result():
                return
            result["fea_analysis_stage"] = message
            if self.current_node is node:
                self._topopt_fea_status_label.setText(message)
            if hasattr(window, "statusBar"):
                window.statusBar().showMessage(message)
            progress = getattr(window, "_set_solve_progress", None)
            if callable(progress):
                progress(message)

        def _finished(analysis):
            if not _current_result():
                return
            result["validation"] = analysis
            result.pop("validation_error", None)
            settings_current = (
                worker.mesh_options == fea_mesh_options_from_node(node)
                and worker.quality == (node.get_property("fea_quality") or "Standard")
            )
            result["fea_analysis_status"] = "completed" if settings_current else "settings_changed"
            update_verification_status(
                result, material=worker.material,
                safety_factor=node.get_property("validation_yield_safety_factor"),
                displacement_limit=node.get_property("validation_displacement_limit_mm"),
            )
            if self.current_node is node:
                mode = "FEA Lattice Von Mises Stress" if result.get("generated_lattice") else "FEA Von Mises Stress"
                result["visualization_mode"] = mode
                window._last_rendered_node = node
                window._last_rendered_geom = None
                self.update_property("visualization", mode)
                render = getattr(window, "_render_result_in_viewer", None)
                if callable(render):
                    render(result)
                else:
                    viewer = getattr(window, "viewer", None)
                    if viewer is not None:
                        viewer.render_simulation(result)
                show_results = getattr(window, "_show_results_panel", None)
                if callable(show_results):
                    show_results()
            if hasattr(window, "statusBar"):
                window.statusBar().showMessage("TopOpt FEA analysis complete.", 10000)

        def _failed(message):
            if _current_result():
                result["fea_analysis_status"] = "failed"
                result["fea_analysis_error"] = message
                update_verification_status(result)
            QtWidgets.QMessageBox.critical(self, "FEA Analysis Error", message)

        def _cancelled():
            if _current_result():
                result["fea_analysis_status"] = "cancelled"
                update_verification_status(result)
            if hasattr(window, "statusBar"):
                window.statusBar().showMessage("TopOpt FEA analysis cancelled.", 10000)

        def _cleanup():
            self._topopt_fea_worker = None
            clear_progress = getattr(window, "_clear_solve_progress", None)
            if callable(clear_progress):
                clear_progress()
            if self.current_node is node:
                self.display_node(node)
            pending = getattr(window, "_start_pending_execute", None)
            if callable(pending):
                QtCore.QTimer.singleShot(0, pending)

        queued = QtCore.Qt.QueuedConnection
        worker.analysis_progress.connect(_progress, queued)
        worker.analysis_finished.connect(_finished, queued)
        worker.analysis_error.connect(_failed, queued)
        worker.analysis_cancelled.connect(_cancelled, queued)
        worker.finished.connect(_cleanup, queued)
        worker.finished.connect(worker.deleteLater)
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage("Analysing TopOpt FEA with CalculiX...")
        progress = getattr(window, "_set_solve_progress", None)
        if callable(progress):
            progress("Analysing FEA...")
        worker.start()
        if self.current_node is node:
            self.display_node(node)
