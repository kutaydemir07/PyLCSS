# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Inspector-owned face selections and thicknesses for preserved material."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtWidgets

from pylcss.cae.design_studio.topology_optimization.geometry.non_design import (
    design_domain_shape,
    face_signature,
    read_non_design_regions,
)


class NonDesignInspectorMixin:
    def _build_topopt_non_design_ui(self, node):
        group = QtWidgets.QGroupBox("Non-Design Space")
        layout = QtWidgets.QVBoxLayout(group)
        enabled = QtWidgets.QCheckBox("Preserve selected face layers")
        enabled.setChecked(bool(node.get_property("non_design_enabled")))
        enabled.toggled.connect(
            lambda value: self.update_property("non_design_enabled", value)
        )
        layout.addWidget(enabled)
        enabled.setToolTip(
            "Select faces in the design domain, then set the inward thickness. "
            "Flat, circular and curved contours follow the selected surface. "
            "The cyan volume is kept as solid material during optimization."
        )
        try:
            regions = read_non_design_regions(
                node.get_property("non_design_regions") or "[]"
            )
        except ValueError as exc:
            regions = []
            layout.addWidget(QtWidgets.QLabel(str(exc)))
            reset = QtWidgets.QPushButton("Clear Invalid Selection")
            reset.clicked.connect(
                lambda: self._write_non_design_regions(node, [], rebuild=True)
            )
            layout.addWidget(reset)
        for index, region in enumerate(regions):
            row = QtWidgets.QGroupBox(
                f"Region {index + 1} · {len(region['face_indices'])} face(s)"
            )
            form = QtWidgets.QFormLayout(row)
            thickness = QtWidgets.QDoubleSpinBox()
            thickness.setLocale(QtCore.QLocale(QtCore.QLocale.C))
            thickness.setRange(0.001, 1_000_000.0)
            thickness.setDecimals(3)
            thickness.setSingleStep(0.5)
            thickness.setSuffix(" mm")
            thickness.setValue(float(region["thickness_mm"]))

            def changed(value, i=index):
                current = read_non_design_regions(
                    node.get_property("non_design_regions")
                )
                current[i]["thickness_mm"] = value
                self._write_non_design_regions(node, current)

            thickness.valueChanged.connect(changed)
            form.addRow("Inward Thickness:", thickness)
            faces = QtWidgets.QLabel(
                ", ".join(str(i + 1) for i in region["face_indices"])
            )
            faces.setWordWrap(True)
            form.addRow("Faces:", faces)
            buttons = QtWidgets.QWidget()
            buttons_layout = QtWidgets.QHBoxLayout(buttons)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            select = QtWidgets.QPushButton("Select Faces")
            select.clicked.connect(
                lambda _checked=False, i=index: self._pick_non_design_faces(node, i)
            )
            remove = QtWidgets.QPushButton("Remove")

            def deleted(_checked=False, i=index):
                current = read_non_design_regions(
                    node.get_property("non_design_regions")
                )
                current.pop(i)
                self._write_non_design_regions(node, current, rebuild=True)

            remove.clicked.connect(deleted)
            buttons_layout.addWidget(select)
            buttons_layout.addWidget(remove)
            form.addRow(buttons)
            layout.addWidget(row)
        add = QtWidgets.QPushButton("Select Faces / Add Region")
        add.clicked.connect(lambda: self._pick_non_design_faces(node))
        layout.addWidget(add)
        self._non_design_status = QtWidgets.QLabel()
        self._non_design_status.setWordWrap(True)
        layout.addWidget(self._non_design_status)
        self.props_layout.addWidget(group)

    def _write_non_design_regions(self, node, regions, *, rebuild=False):
        if self.current_node is not node:
            return
        self.update_property("non_design_regions", json.dumps(regions))
        if rebuild:
            self.display_node(node)

    def _pick_non_design_faces(self, node, region_index=None):
        app = self._get_main_app()
        if app is None:
            return
        if app._execution_is_active():
            app.statusBar().showMessage(
                "Wait for the current computation to finish before selecting faces."
            )
            return
        domain_input = app._cached_topopt_design_domain(node)
        if domain_input is None:
            port = node.get_input("design_domain")
            sources = (
                [connected.node() for connected in port.connected_ports()]
                if port
                else []
            )
            if not sources:
                app.statusBar().showMessage(
                    "Connect a CAD design domain before selecting non-design faces."
                )
                return
            upstream = self._upstream_nodes_for(sources[0])
            if any(
                getattr(source, "__identifier__", "")
                in {
                    "com.cad.sim.topopt_voxel",
                    "com.cad.sim.solver",
                    "com.cad.sim.crash_solver",
                }
                and getattr(source, "_last_result", None) is None
                for source in upstream
            ):
                app.statusBar().showMessage(
                    "Run the upstream solver before selecting its CAD faces."
                )
                return
            app._execute_graph(nodes=upstream, skip_simulation=True)
            worker = app.worker
            if worker is not None:
                worker.computation_finished.connect(
                    lambda _results: QtCore.QTimer.singleShot(
                        0,
                        lambda: self._pick_non_design_faces(node, region_index)
                        if self.current_node is node
                        and app._cached_topopt_design_domain(node) is not None
                        else None,
                    )
                )
            return
        try:
            domain = design_domain_shape(domain_input)
        except ValueError as exc:
            app.statusBar().showMessage(str(exc))
            return
        viewer = app.viewer
        active = self._active_pick_connections
        if active is not None:
            old_viewer, _old_done, _old_cancel = active
            old_viewer.picking_cancelled.emit()
            old_viewer.disable_picking_mode()
        app._render_result_in_viewer(domain)
        viewer.enable_picking_mode(multi_select=True)
        faces = domain.Faces()

        def disconnect():
            viewer.face_picked.disconnect(done)
            viewer.picking_cancelled.disconnect(cancel)
            self._active_pick_connections = None

        def done(selected):
            disconnect()
            if self.current_node is not node:
                return
            indices = sorted(
                {
                    i
                    for face in selected
                    for i, candidate in enumerate(faces)
                    if candidate.isSame(face)
                }
            )
            if not indices:
                app._preview_topopt_non_design(node, render_domain=True)
                return
            regions = read_non_design_regions(node.get_property("non_design_regions"))
            thickness = (
                regions[region_index]["thickness_mm"]
                if region_index is not None
                else 1.0
            )
            region = {
                "face_indices": indices,
                "thickness_mm": thickness,
                "face_signatures": [face_signature(faces[i]) for i in indices],
            }
            if region_index is None:
                regions.append(region)
            else:
                regions[region_index] = region
            self.update_property("non_design_enabled", True)
            self._write_non_design_regions(node, regions, rebuild=True)
            app._preview_topopt_non_design(node, render_domain=True)

        def cancel():
            disconnect()
            app._preview_topopt_non_design(node, render_domain=True)

        viewer.face_picked.connect(done)
        viewer.picking_cancelled.connect(cancel)
        self._active_pick_connections = (viewer, done, cancel)
        app.statusBar().showMessage(
            "Select non-design faces, then press Done in the viewer."
        )
