# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SelectionInspectorMixin behavior for the CAD property inspector."""

from __future__ import annotations

import logging


from PySide6 import QtWidgets


from .execution_workers import GraphExecutionWorker

__all__ = ["SelectionInspectorMixin"]


class SelectionInspectorMixin:
    def _build_select_face_ui(self, node):
        """Selector-aware UI for SelectFaceNode.

        SelectFaceNode is a swiss-army-knife with seven different selection
        strategies (bounding box, nearest point, face index, direction tag,
        range expression…).  The generic inspector shows *all* of their
        fields at once which is overwhelming.  Here we render only the
        fields relevant to the currently chosen ``selector_type``.
        """

        def _canonical_selector(value):
            aliases = {
                "direction": "Direction",
                "nearesttopoint": "NearestToPoint",
                "nearest point": "NearestToPoint",
                "nearest_point": "NearestToPoint",
                "index": "Index",
                "face index": "Index",
                "face_index": "Index",
                "largest area": "Largest Area",
                "largest_area": "Largest Area",
                "tag": "Tag",
                "box": "Box",
                "bounding box": "Box",
                "bounding_box": "Box",
                "coordinate range": "Coordinate Range",
                "range expression": "Coordinate Range",
                "range_expression": "Coordinate Range",
            }
            text = str(value or "Direction").strip()
            return aliases.get(text.lower(), text)

        def _canonical_direction(value):
            aliases = {
                "+X": ">X",
                "-X": "<X",
                "+Y": ">Y",
                "-Y": "<Y",
                "+Z": ">Z",
                "-Z": "<Z",
                "X+": ">X",
                "X-": "<X",
                "Y+": ">Y",
                "Y-": "<Y",
                "Z+": ">Z",
                "Z-": "<Z",
            }
            text = str(value or ">Z").strip().upper()
            return aliases.get(text, text)

        entity_type = str(node.get_property("entity_type") or "Face").title()
        if entity_type not in {"Face", "Edge", "Vertex"}:
            entity_type = "Face"

        # The selector type drives which field group is shown. The combo shows
        # friendly labels but stores the exact values SelectFaceNode executes.
        sel_type = _canonical_selector(node.get_property("selector_type"))
        type_options = [
            ("Direction", "Direction"),
            ("Nearest Point", "NearestToPoint"),
            (f"{entity_type} Index", "Index"),
            (
                "Largest Area" if entity_type == "Face" else "Longest Edge",
                "Largest Area",
            ),
            ("Bounding Box", "Box"),
            ("Range Expression", "Coordinate Range"),
            ("Tag", "Tag"),
        ]
        if entity_type == "Vertex":
            type_options = [
                option for option in type_options if option[1] != "Largest Area"
            ]
            if sel_type == "Largest Area":
                sel_type = "Direction"
                node.set_property("selector_type", sel_type)

        # ── 1. Selector type combo ──────────────────────────────────────
        grp_type = QtWidgets.QGroupBox("Selector")
        lay_type = QtWidgets.QFormLayout()
        entity_combo = QtWidgets.QComboBox()
        entity_combo.addItem("Face", "Face")
        entity_combo.addItem("Edge", "Edge")
        entity_combo.addItem("Vertex (Corner)", "Vertex")
        entity_combo.setCurrentIndex(max(0, entity_combo.findData(entity_type)))
        entity_combo.setToolTip(
            "Faces define physical distributed interfaces. Edges and vertices "
            "are available for general CAD/FEA selection, but topology "
            "optimization must regularize their zero-area line/point condition "
            "into a mesh-sized pad. Prefer a real attachment or load face."
        )

        def _change_entity(value):
            self.update_property("entity_type", value)
            self.display_node(node)

        entity_combo.currentIndexChanged.connect(
            lambda _index, combo=entity_combo: _change_entity(combo.currentData())
        )
        lay_type.addRow("Entity:", entity_combo)

        combo = QtWidgets.QComboBox()
        for label, value in type_options:
            combo.addItem(label, value)
        current_idx = combo.findData(sel_type)
        if current_idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(current_idx)
            combo.blockSignals(False)
        combo.setToolTip(
            "How this node picks faces:\n"
            "  Bounding Box     — every face whose centroid is inside the box\n"
            "  Nearest Point    — the single face closest to (near_x, near_y, near_z)\n"
            "  Direction        — every face whose normal is +X / −Z / …\n"
            "  Face Index       — pick by integer face id (zero-based, brittle)\n"
            "  Range Expression — Python boolean over x, y, z of the face centroid\n"
            "  Tag              — match a user-set tag string on the upstream node"
        )
        combo.setToolTip(
            f"Choose how to resolve {entity_type.lower()} geometry. Direction "
            "selects an X/Y/Z extreme; Nearest Point and Bounding Box use the "
            "entity center; Index is zero-based and can change after upstream "
            "CAD edits; Range Expression evaluates x, y, and z."
        )
        combo.currentIndexChanged.connect(
            lambda _i, c=combo: (
                self.update_property("selector_type", c.currentData()),
                self.display_node(node),
            )  # rebuild panel for the new type
        )
        lay_type.addRow("Type:", combo)
        grp_type.setLayout(lay_type)
        self.props_layout.addWidget(grp_type)

        # ── 2. Type-specific field group ────────────────────────────────
        grp = QtWidgets.QGroupBox("Parameters")
        lay = QtWidgets.QFormLayout()

        def _spin(prop, lo=-1e4, hi=1e4, dec=3):
            val = float(node.get_property(prop) or 0.0)
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setValue(val)
            w.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return w

        def _intspin(prop, lo=0, hi=10_000):
            try:
                val = int(node.get_property(prop) or 0)
            except Exception:
                val = 0
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            w.valueChanged.connect(lambda v, p=prop: self.update_property(p, v))
            return w

        def _line(prop, placeholder=""):
            val = node.get_property(prop) or ""
            w = QtWidgets.QLineEdit(str(val))
            w.setPlaceholderText(placeholder)
            w.editingFinished.connect(
                lambda p=prop, ww=w: self.update_property(p, ww.text())
            )
            return w

        if sel_type == "Box":
            lay.addRow("Min X:", _spin("box_min_x"))
            lay.addRow("Min Y:", _spin("box_min_y"))
            lay.addRow("Min Z:", _spin("box_min_z"))
            lay.addRow("Max X:", _spin("box_max_x"))
            lay.addRow("Max Y:", _spin("box_max_y"))
            lay.addRow("Max Z:", _spin("box_max_z"))
        elif sel_type == "NearestToPoint":
            lay.addRow("Near X:", _spin("near_x"))
            lay.addRow("Near Y:", _spin("near_y"))
            lay.addRow("Near Z:", _spin("near_z"))
        elif sel_type == "Direction":
            dir_combo = QtWidgets.QComboBox()
            direction_options = [
                ("+X (X max face)", ">X"),
                ("-X (X min face)", "<X"),
                ("+Y (Y max face)", ">Y"),
                ("-Y (Y min face)", "<Y"),
                ("+Z (Z max face)", ">Z"),
                ("-Z (Z min face)", "<Z"),
            ]
            for label, value in direction_options:
                dir_combo.addItem(label, value)
            dir_combo.setToolTip(
                "Pick every face whose outward normal points in this direction "
                "(within ~10° tolerance)."
            )
            cur = _canonical_direction(node.get_property("direction"))
            current_idx = dir_combo.findData(cur)
            if current_idx >= 0:
                dir_combo.setCurrentIndex(current_idx)
            dir_combo.currentIndexChanged.connect(
                lambda _i, c=dir_combo: self.update_property(
                    "direction", c.currentData()
                )
            )
            lay.addRow("Normal:", dir_combo)
        elif sel_type == "Index":
            w = _intspin("face_index", 0, 100_000)
            w.setToolTip(
                "Zero-based face index from CadQuery's `faces()` iteration order.\n"
                "Fragile — adding a fillet or boolean upstream renumbers faces."
            )
            lay.addRow("Index:", w)
        elif sel_type == "Coordinate Range":
            w = _line("range_expr", placeholder="e.g. z > 0.99 * z_max")
            w.setToolTip(
                "Python boolean over the face-centroid coordinates x, y, z.\n"
                "Face is picked when this expression is True for its centroid."
            )
            lay.addRow("Expr:", w)
        elif sel_type == "Tag":
            w = _line("tag", placeholder="e.g. top_face")
            w.setToolTip(
                "Match faces that were tagged with this string on the upstream node."
            )
            lay.addRow("Tag:", w)

        grp.setLayout(lay)
        self.props_layout.addWidget(grp)

        summary_group = QtWidgets.QGroupBox("Resolved Selection")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        result = getattr(node, "_last_result", None)
        summaries = (
            result.get("entity_summaries") or result.get("face_summaries")
            if isinstance(result, dict)
            else None
        )
        if summaries:
            count = int(
                result.get("entity_count") or result.get("face_count") or len(summaries)
            )
            summary_layout.addWidget(
                QtWidgets.QLabel(f"{count} {entity_type.lower()}(s) currently matched.")
            )
            for idx, info in enumerate(summaries[:6], start=1):
                center = info.get("center") or []
                bbox = info.get("bbox") or {}
                area = info.get("area")
                if len(center) != 3 or not bbox:
                    continue
                text = (
                    f"{entity_type} {idx}: center=({center[0]:.3g}, "
                    f"{center[1]:.3g}, {center[2]:.3g}), "
                    f"bbox X[{bbox.get('xmin', 0):.3g}, {bbox.get('xmax', 0):.3g}] "
                    f"Y[{bbox.get('ymin', 0):.3g}, {bbox.get('ymax', 0):.3g}] "
                    f"Z[{bbox.get('zmin', 0):.3g}, {bbox.get('zmax', 0):.3g}]"
                )
                if area is not None:
                    text += f", area={area:.3g}"
                if info.get("length") is not None:
                    text += f", length={info['length']:.3g}"
                label = QtWidgets.QLabel(text)
                label.setWordWrap(True)
                summary_layout.addWidget(label)
        else:
            label = QtWidgets.QLabel(
                "Run or preview the graph to list the matched geometry centers "
                "and bounds."
            )
            label.setWordWrap(True)
            label.setStyleSheet("color:#888; font-style:italic;")
            summary_layout.addWidget(label)

        btn_refresh = QtWidgets.QPushButton("Preview Selection")
        btn_refresh.setToolTip(
            "Execute the CAD-only graph preview and redraw the selected "
            "geometry overlay."
        )
        btn_refresh.clicked.connect(
            lambda _checked=False: (
                self._get_main_app()._execute_graph(skip_simulation=True)
                if self._get_main_app() is not None
                else None
            )
        )
        summary_layout.addWidget(btn_refresh)
        self.props_layout.addWidget(summary_group)

    def _build_interactive_select_ui(self, node):
        """Dedicated Properties Panel UI for InteractiveSelectFaceNode."""
        entity_type = str(node.get_property("entity_type") or "Face").title()
        if entity_type not in {"Face", "Edge", "Vertex"}:
            entity_type = "Face"

        entity_group = QtWidgets.QGroupBox("Geometry Type")
        entity_layout = QtWidgets.QFormLayout(entity_group)
        entity_combo = QtWidgets.QComboBox()
        entity_combo.addItem("Face", "Face")
        entity_combo.addItem("Edge", "Edge")
        entity_combo.addItem("Vertex (Corner)", "Vertex")
        entity_combo.setCurrentIndex(max(0, entity_combo.findData(entity_type)))

        def _set_entity(value):
            node.set_property("entity_type", value)
            if hasattr(node, "set_picked_entities"):
                node.set_picked_entities([])
            self.display_node(node)

        entity_combo.currentIndexChanged.connect(
            lambda _index, combo=entity_combo: _set_entity(combo.currentData())
        )
        entity_layout.addRow("Pick:", entity_combo)
        self.props_layout.addWidget(entity_group)

        # -- Status banner --
        sel_label = node.get_property("selection_label") or "No geometry selected"
        raw_indices = node.get_property("picked_face_indices") or ""
        face_indices = [
            int(t.strip()) for t in raw_indices.split(",") if t.strip().isdigit()
        ]

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
            group_list = QtWidgets.QGroupBox(
                f"Selected {entity_type}s ({len(face_indices)})"
            )
            vbox = QtWidgets.QVBoxLayout(group_list)
            for idx in face_indices:
                lbl = QtWidgets.QLabel(f"  {entity_type} index {idx}")
                lbl.setStyleSheet("color:#aad4ff; font-size:11px;")
                vbox.addWidget(lbl)
            self.props_layout.addWidget(group_list)

        # -- Pick button --
        btn_pick = QtWidgets.QPushButton(f"Pick {entity_type}s in 3D Viewer")
        btn_pick.setStyleSheet(
            "QPushButton { background:#1e5ab4; color:white; border-radius:5px;"
            "  padding:8px; font-weight:bold; font-size:13px; }"
            "QPushButton:hover { background:#2470d8; }"
        )
        btn_pick.setToolTip(
            f"Click to enter {entity_type.lower()}-picking mode.\n"
            f"Then click {entity_type.lower()}s on the 3D model. "
            "Ctrl+Click for multi-select."
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
            "<i>Note: The picker prepares cached upstream geometry when possible. "
            "If the upstream topology optimization has not run yet, run it first.</i>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:10px; margin-top:8px;")
        self.props_layout.addWidget(hint)

    @staticmethod
    def _viewer_has_pickable_faces(viewer):
        try:
            faces = list(getattr(viewer, "_all_occ_faces", []) or [])
            polydata = list(getattr(viewer, "_face_polydata_list", []) or [])
            return bool(faces) and any(pd is not None for pd in polydata)
        except Exception:
            return False

    @staticmethod
    def _upstream_nodes_for(node):
        ordered = []
        visited = set()

        def _walk(current):
            marker = id(current)
            if marker in visited:
                return
            visited.add(marker)
            try:
                ports = current.input_ports()
                if isinstance(ports, dict):
                    ports = list(ports.values())
                else:
                    ports = list(ports)
            except Exception:
                ports = []
            for port in ports:
                try:
                    connected = list(port.connected_ports())
                except Exception:
                    connected = []
                for conn_port in connected:
                    try:
                        _walk(conn_port.node())
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )
            ordered.append(current)

        _walk(node)
        return ordered

    def _prepare_viewer_for_picking(self, app, node):
        """Render or compute the nearest upstream geometry for face picking."""
        viewer = getattr(app, "viewer", None)
        if viewer is None:
            return False

        # Always prefer the selected picker node's own upstream geometry over
        # whatever happens to be displayed.  This prevents a Crash/FEA picker
        # from trying to split a stale dense TopOpt STL currently in the viewer.
        try:
            source_node, geometry = app._get_render_context_for_node(node)
        except Exception:
            source_node, geometry = None, None
        if geometry is not None:
            try:
                app._last_rendered_node = source_node or node
                app._render_result_in_viewer(geometry)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            if self._viewer_has_pickable_faces(viewer):
                return True
            if (
                hasattr(viewer, "ensure_mesh_face_picking")
                and viewer.ensure_mesh_face_picking()
            ):
                return True

        if self._viewer_has_pickable_faces(viewer):
            return True
        if (
            hasattr(viewer, "ensure_mesh_face_picking")
            and viewer.ensure_mesh_face_picking()
        ):
            return True

        if hasattr(app, "_execution_is_active") and app._execution_is_active():
            message = "Wait for the current graph run to finish before picking faces."
            try:
                app.statusBar().showMessage("Graph is busy.")
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            QtWidgets.QMessageBox.information(self, "Graph Running", message)
            return "pending"

        worker = getattr(self, "_pick_prepare_worker", None)
        try:
            worker_running = bool(worker is not None and worker.isRunning())
        except Exception:
            worker_running = False
        if worker_running:
            message = "Geometry is already being prepared for face picking."
            try:
                app.statusBar().showMessage("Preparing geometry...")
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            return "pending"

        upstream_nodes = [n for n in self._upstream_nodes_for(node) if n is not node]
        blocked_ids = {
            "com.cad.sim.solver",
            "com.cad.sim.crash_solver",
            "com.cad.sim.topopt_voxel",
            "com.cad.sim.lattice_voxel",
            "com.cad.sim.lattice_infill",
        }
        blocked = [
            n
            for n in upstream_nodes
            if getattr(n, "__identifier__", "") in blocked_ids
            and getattr(n, "_last_result", None) is None
        ]
        if blocked:
            message = (
                "Run the upstream topology/solver node once, then pick faces. "
                "The picker can prepare STL import and remesh geometry, but it "
                "will not silently start a new topology optimization."
            )
            try:
                app.statusBar().showMessage("Run the upstream solver first.")
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            QtWidgets.QMessageBox.information(self, "Run Upstream First", message)
            return "pending"

        if not upstream_nodes:
            if bool(getattr(viewer, "_mesh_picking_too_dense", False)):
                message = (
                    "This displayed topology/STL surface is too dense for direct "
                    "interactive patch picking. Pick from a face selector connected "
                    "after Remesh, so the picker can use the volume-mesh surface."
                )
                try:
                    app.statusBar().showMessage("Use a remeshed surface.")
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
                QtWidgets.QMessageBox.information(self, "Use Remeshed Surface", message)
                return "pending"
            return False

        try:
            try:
                app.statusBar().showMessage("Preparing geometry...")
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            worker = GraphExecutionWorker(
                upstream_nodes, skip_simulation=False, parent=self
            )
            self._pick_prepare_worker = worker

            def _finish(_results):
                try:
                    self._pick_prepare_worker = None
                    source_node, geometry = app._get_render_context_for_node(node)
                    if geometry is not None:
                        app._last_rendered_node = source_node or node
                        app._render_result_in_viewer(geometry)
                    ready = self._viewer_has_pickable_faces(viewer)
                    if not ready and hasattr(viewer, "ensure_mesh_face_picking"):
                        ready = viewer.ensure_mesh_face_picking()
                    if ready:
                        try:
                            app.statusBar().showMessage(
                                "Geometry ready. Pick faces in the viewer."
                            )
                        except Exception:
                            logging.getLogger(__name__).debug(
                                "Optional UI operation failed.", exc_info=True
                            )
                        self._start_picking_session(node)
                    else:
                        if bool(getattr(viewer, "_mesh_picking_too_dense", False)):
                            text = (
                                "The rendered surface is too dense for direct "
                                "interactive patch picking. Run Remesh first, "
                                "then pick on the remeshed surface."
                            )
                        else:
                            text = "The upstream geometry finished, but no pickable face patches were found."
                        QtWidgets.QMessageBox.information(
                            self,
                            "No Pickable Faces",
                            text,
                        )
                finally:
                    try:
                        worker.deleteLater()
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

            def _error(message):
                try:
                    self._pick_prepare_worker = None
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Could Not Prepare Picker",
                        f"Could not prepare upstream geometry for face picking:\n{message}",
                    )
                finally:
                    try:
                        worker.deleteLater()
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

            worker.computation_finished.connect(_finish)
            worker.computation_error.connect(_error)
            worker.start()
            return "pending"
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Could Not Prepare Picker",
                f"Could not prepare upstream geometry for face picking:\n{exc}",
            )
            return False

    def _start_picking_session(self, node):
        """Enable picking mode in the 3D viewer for this node."""
        # Walk up to find ProfessionalCadApp
        app = self._get_main_app()
        if app is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Viewer",
                "Cannot access the 3D viewer. Make sure the application is fully loaded.",
            )
            return

        viewer = getattr(app, "viewer", None)
        if viewer is None:
            QtWidgets.QMessageBox.warning(self, "No Viewer", "3D viewer not found.")
            return
        entity_type = str(node.get_property("entity_type") or "Face").title()
        if entity_type not in {"Face", "Edge", "Vertex"}:
            entity_type = "Face"

        prepared = self._prepare_viewer_for_picking(app, node)
        if prepared is not True:
            if prepared == "pending":
                return
            QtWidgets.QMessageBox.information(
                self,
                f"No Pickable {entity_type}s",
                f"No pickable geometry is available for {entity_type.lower()} "
                "selection yet. Run the upstream geometry/remesh or topology "
                "step, then try picking again.",
            )
            return

        # A completed pick used to leave its cancellation closure connected.
        # Repeated sessions therefore accumulated stale node references and a
        # later Cancel invoked every old callback. Tear down any prior session
        # before installing the new pair of signal handlers.
        active = self._active_pick_connections
        if active is not None:
            old_viewer, old_done, old_cancel = active
            old_signals = [
                old_viewer.face_picked,
                old_viewer.edge_picked,
                old_viewer.picking_cancelled,
            ]
            if hasattr(old_viewer, "vertex_picked"):
                old_signals.append(old_viewer.vertex_picked)
            for signal in old_signals:
                for handler in (old_done, old_cancel):
                    try:
                        signal.disconnect(handler)
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )
            self._active_pick_connections = None

        if entity_type == "Edge":
            picked_signal = viewer.edge_picked
            if not viewer.enable_edge_picking_mode(multi_select=True):
                QtWidgets.QMessageBox.information(
                    self,
                    "No Pickable Edges",
                    "The displayed geometry has no CAD edges. Edge selection "
                    "reads the B-rep wireframe, which a meshed or topology "
                    "result does not carry -- display the upstream CAD solid "
                    "before selecting edges.",
                )
                return
        elif entity_type == "Vertex":
            if not hasattr(viewer, "vertex_picked"):
                QtWidgets.QMessageBox.information(
                    self,
                    "Vertex Picking Unavailable",
                    "This viewer does not expose vertex picking for the "
                    "currently displayed geometry.",
                )
                return
            picked_signal = viewer.vertex_picked
            if not viewer.enable_vertex_picking_mode(multi_select=True):
                QtWidgets.QMessageBox.information(
                    self,
                    "No Pickable Vertices",
                    "The displayed geometry has no CAD vertices. Display the "
                    "upstream CAD solid before selecting corners.",
                )
                return
        else:
            picked_signal = viewer.face_picked
            viewer.enable_picking_mode(multi_select=True)

        def _disconnect_session():
            for signal, handler in (
                (picked_signal, _on_faces_picked),
                (viewer.picking_cancelled, _on_cancelled),
            ):
                try:
                    signal.disconnect(handler)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )
            if self._active_pick_connections is not None:
                active_viewer, active_done, _active_cancel = (
                    self._active_pick_connections
                )
                if active_viewer is viewer and active_done is _on_faces_picked:
                    self._active_pick_connections = None

        # Wire done signal
        def _on_faces_picked(occ_faces):
            _disconnect_session()
            # Map picked face objects → indices that the InteractiveSelectFace
            # node understands. Two flavours:
            #   * Mesh virtual-face dicts already carry their own 'stored_index'
            #     (a >=1000 patch id encoding direction+component). Read it
            #     directly — no identity check needed.
            #   * OCC face objects: locate them in viewer._all_occ_faces by
            #     hashCode equality so we can record their position.
            all_occ = getattr(
                viewer,
                {
                    "Face": "_all_occ_faces",
                    "Edge": "_all_occ_edges",
                    "Vertex": "_all_occ_vertices",
                }[entity_type],
                [],
            )
            picked_indices = []
            picked_labels = []
            for face in occ_faces:
                if isinstance(face, dict):
                    stored = face.get("stored_index")
                    if stored is None:
                        stored = face.get(
                            "node_id",
                            face.get("face_index", face.get("entity_index", 0)),
                        )
                    picked_indices.append(int(stored))
                    picked_labels.append(
                        face.get("label") or face.get("selector") or None
                    )
                    continue
                for i, f in enumerate(all_occ):
                    try:
                        if face.hashCode(10000) == f.hashCode(10000):
                            picked_indices.append(i)
                            picked_labels.append(None)
                            break
                    except Exception:
                        if face is f:
                            picked_indices.append(i)
                            picked_labels.append(None)
                            break

            # Suppress the graph's property-changed handler while we write
            # both properties; otherwise each set_property fires its own
            # display_node() and the panel rebuilds 3-4 times in a row,
            # leaving stale Clear-Selection buttons (red, deleteLater-pending)
            # stacked on top of each other until the event loop drains.
            app._suppress_graph_property_changed = True
            try:
                if hasattr(node, "set_picked_entities"):
                    node.set_picked_entities(picked_indices)
                elif hasattr(node, "set_picked_faces"):
                    node.set_picked_faces(picked_indices)
                else:
                    node.set_property(
                        "picked_face_indices", ",".join(str(i) for i in picked_indices)
                    )
                if picked_indices and any(picked_labels):
                    chunks = []
                    for idx, label in zip(picked_indices, picked_labels):
                        chunks.append(f"{idx} / {label}" if label else str(idx))
                    node.set_property(
                        "selection_label",
                        f"{len(picked_indices)} {entity_type.lower()}"
                        f"{'s' if len(picked_indices) != 1 else ''} selected  "
                        f"(idx: {', '.join(chunks)})",
                    )
                if hasattr(node, "_last_hash"):
                    node._last_hash = None
            finally:
                app._suppress_graph_property_changed = False

            # Single rebuild of the inspector with final state.
            self.display_node(node)

            if hasattr(app, "_execute_graph"):
                app._execute_graph(skip_simulation=True)

        def _on_cancelled():
            _disconnect_session()

        picked_signal.connect(_on_faces_picked)
        viewer.picking_cancelled.connect(_on_cancelled)
        self._active_pick_connections = (viewer, _on_faces_picked, _on_cancelled)

    def _clear_face_selection(self, node):
        """Clear all picked geometry from an interactive selector."""
        if hasattr(node, "set_picked_entities"):
            node.set_picked_entities([])
        elif hasattr(node, "set_picked_faces"):
            node.set_picked_faces([])
        else:
            node.set_property("picked_face_indices", "")
            node.set_property("selection_label", "No geometry selected")
        if hasattr(node, "_last_hash"):
            node._last_hash = None
        self.display_node(node)

    def _get_main_app(self):
        """Walk up the parent chain to find ProfessionalCadApp."""
        widget = self.parent()
        while widget is not None:
            if widget.__class__.__name__ == "ProfessionalCadApp":
                return widget
            widget = widget.parent() if hasattr(widget, "parent") else None
        return None
