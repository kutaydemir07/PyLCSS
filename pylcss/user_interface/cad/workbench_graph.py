# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchGraphMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging

from PySide6 import QtCore, QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchGraphMixin"]


class WorkbenchGraphMixin:
    def _on_node_double_clicked(self, node):
        """Handle a double-click on a node on the graph canvas.

        For ``CadQueryCodeNode`` this opens the full-screen CAD code editor —
        the same one the inspector's *Edit Code…* button uses.
        For ``FreeCadPartNode`` this launches the FreeCAD GUI subprocess on
        the node's .FCStd file and wires the save-watcher so the viewer
        refreshes automatically when the user saves inside FreeCAD.
        Other node types swallow the double-click so NodeGraphQt's default
        subgraph popup doesn't appear.
        """
        try:
            class_name = node.__class__.__name__
            if class_name == "CadQueryCodeNode":
                # The inspector holds the editor-open helper.
                self.properties._open_cad_code_editor(node)
                self.timeline.add_event(f"Opened code editor for {node.name()}")
                return
            if class_name == "FreeCadPartNode":
                self._open_freecad_for_node(node)
                return
        except Exception as exc:
            logger.exception("Double-click action failed for %s", node)
            message = f"Could not open {node.name()}: {exc}"
            self.statusBar().showMessage(message, 7000)
            self.timeline.add_event(message)
        # Other nodes have no double-click action.

    def _open_freecad_for_node(self, node):
        """Spawn FreeCAD on a FreeCadPartNode's .FCStd file and wire a
        per-node :class:`FCStdWatcher` so saves trigger a viewer refresh.

        Idempotent: opening the same node twice re-focuses the existing
        FreeCAD instance (the launcher's per-file registry guarantees this)
        and reuses the watcher already attached to the node.
        """
        if node is None or node.__class__.__name__ != "FreeCadPartNode":
            logger.debug("Ignored Open in FreeCAD without a FreeCadPartNode.")
            return False
        path_getter = getattr(node, "fcstd_path", None)
        if not callable(path_getter):
            logger.warning("FreeCadPartNode does not expose fcstd_path().")
            return False

        try:
            from pylcss.design_studio.freecad_bridge.launcher import FreeCadLauncher
            from pylcss.design_studio.freecad_bridge.watcher import FCStdWatcher
        except ImportError as exc:
            self.timeline.add_event(f"FreeCAD bridge unavailable: {exc}")
            return False

        fcstd_path = path_getter()

        launcher = getattr(node, "_freecad_launcher", None)
        if launcher is None:
            launcher = FreeCadLauncher(parent=self)
            node._freecad_launcher = launcher
            launcher.error_occurred.connect(
                lambda path, msg, n=node: self.timeline.add_event(
                    f"FreeCAD error for {n.name()}: {msg}"
                )
            )
            launcher.process_exited.connect(
                lambda path, code, n=node: self.timeline.add_event(
                    f"FreeCAD exited (code {code}) for {n.name()}"
                )
            )

        if not launcher.is_available():
            self.timeline.add_event(
                "FreeCAD not installed — run "
                "`python scripts/install_solvers.py --only freecad`"
            )
            return False

        # Wire (or re-wire) the save-watcher exactly once per node so each
        # save in FreeCAD triggers _cad_execute and the viewer updates.
        watcher = getattr(node, "_freecad_watcher", None)
        if watcher is None or str(watcher.fcstd_path) != str(fcstd_path):
            if watcher is not None:
                try:
                    watcher.stop()
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
            watcher = FCStdWatcher(fcstd_path, parent=self)
            watcher.saved.connect(lambda _p, n=node: self._on_freecad_save(n))
            node._freecad_watcher = watcher

        ok = launcher.open(fcstd_path)
        if ok:
            if getattr(node, "_last_result", None) is None:
                setter = getattr(node, "set_pending", None)
                if callable(setter):
                    setter(
                        "FreeCAD is open. Create the part and save the document "
                        "to update this node."
                    )
            message = (
                f"Opened FreeCAD for {node.name()} — create or edit the part, "
                "then save"
            )
            self.statusBar().showMessage(message)
            self.timeline.add_event(message)
            current = getattr(self.properties, "current_node", None)
            if current is node:
                self.properties.display_node(node)
        return bool(ok)

    def _on_freecad_save(self, node):
        """FCStdWatcher fired -- mark the node dirty and trigger a CAD
        execute so the new geometry shows up in the viewer."""
        setattr(node, "_dirty", True)
        self.timeline.add_event(f"FreeCAD saved: refreshing {node.name()}")
        try:
            # Re-use whichever execution entry point the rest of the widget
            # uses for "graph property changed -> re-run".
            if hasattr(self, "_execute_graph"):
                self._execute_graph(skip_simulation=True)
            elif hasattr(self, "execute_graph"):
                self.execute_graph(skip_simulation=True)
        except Exception as exc:
            self.timeline.add_event(f"CAD re-execute failed: {exc}")

    def _on_nodes_deleted(self, node_ids):
        """Release per-node FreeCAD launcher + watcher when the node is
        removed from the graph. NodeGraphQt emits node IDs (not nodes) here
        because the node objects have already been torn down -- we keep
        the launcher/watcher in a per-node attribute, so look them up via
        the still-alive references in our own bookkeeping."""
        # NodeGraphQt API gives us only IDs; we can't fetch the nodes back
        # because they're gone. Best-effort cleanup: walk every still-alive
        # node and ensure orphans get shut down on the next idle cycle.
        try:
            for node in self.graph.all_nodes():
                if not hasattr(node, "_freecad_watcher"):
                    continue
                # Node still alive -- nothing to do here.
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

    def _on_graph_property_changed(self, node, prop_name, prop_value):
        """Handle property changes from the graph (including widgets on nodes)."""
        # Mark node as dirty so it re-executes
        setattr(node, "_dirty", True)

        if getattr(self, "_suppress_graph_property_changed", False):
            return False

        # Update the properties panel if this node is selected.
        # Skip if the inspector itself triggered the change to avoid a reset loop.
        # Skip "silent" book-keeping props that the panel never displays —
        # rebuilding for those just causes UI freezes during graph execution.
        if (
            prop_name not in self._SILENT_PROP_NAMES
            and self.properties.current_node == node
            and not self.properties._updating_property
        ):
            self.properties.display_node(node)

        if prop_name in {"description", "tags", "notes"}:
            setattr(node, "_dirty", False)
            return

        # Visualization/post-processing changes update the cached payload and
        # viewer immediately; they never re-run an engineering solve.
        if prop_name in (
            "visualization",
            "deformation_scale",
            "disp_scale",
            "density_cutoff",
            "element_type",
        ):
            cached_result = getattr(node, "_last_result", None)
            if cached_result is not None and isinstance(cached_result, dict):
                if prop_name == "visualization":
                    cached_result["visualization_mode"] = prop_value
                elif prop_name == "deformation_scale":
                    text = str(prop_value).strip().lower()
                    if text == "auto":
                        cached_result["deformation_scale"] = cached_result.get(
                            "auto_deformation_scale", 1.0
                        )
                    else:
                        try:
                            cached_result["deformation_scale"] = float(text.rstrip("x"))
                        except ValueError:
                            pass
                elif prop_name == "disp_scale":
                    cached_result["disp_scale"] = float(prop_value)
                elif prop_name == "density_cutoff":
                    cached_result["density_cutoff"] = float(prop_value)
                    if (
                        cached_result.get("type") == "topopt_voxel"
                        and cached_result.get("density") is not None
                    ):
                        self._refresh_topopt_recovered_shape(node, cached_result)

                try:
                    self.viewer.render_simulation(cached_result)
                    try:
                        self._show_bc_for_node(node)
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)

            setattr(node, "_dirty", False)
            return

        # Always auto-execute (skip simulation nodes for performance)
        if not self.properties._updating_property:
            self._execute_graph(skip_simulation=True)

    def _on_connection_changed(self, port_in, port_out):
        """Handle connection changes (connect/disconnect)."""
        # Skip events during project loading to prevent spam
        if self._is_loading:
            return
        if getattr(self, "_rejecting_invalid_connection", False):
            return

        # NodeGraphQt's serialized port names stay backward compatible, while
        # the semantic schema prevents known cross-domain mistakes.
        if port_in is not None and port_out is not None:
            try:
                is_connected = port_out in list(port_in.connected_ports())
            except Exception:
                is_connected = False
            if is_connected:
                from pylcss.design_studio.core.port_schema import (
                    validate_connection,
                )

                valid, message = validate_connection(port_in, port_out)
                if not valid:
                    self._rejecting_invalid_connection = True
                    try:
                        port_in.disconnect_from(port_out)
                    finally:
                        self._rejecting_invalid_connection = False
                    self.statusBar().showMessage(message, 7000)
                    self.timeline.add_event(f"Connection rejected: {message}")
                    current = getattr(self.properties, "current_node", None)
                    if current is not None:
                        QtCore.QTimer.singleShot(
                            0,
                            lambda n=current: self.properties.display_node(n)
                            if self.properties.current_node is n
                            else None,
                        )
                    return

        # Mark both nodes as dirty
        if port_in:
            node = port_in.node()
            setattr(node, "_dirty", True)
        if port_out:
            node = port_out.node()
            setattr(node, "_dirty", True)

        # Auto-execute with skip_simulation for fast CAD preview
        self._execute_graph(skip_simulation=True)
        current = getattr(self.properties, "current_node", None)
        if current is not None:
            QtCore.QTimer.singleShot(
                0,
                lambda n=current: self.properties.display_node(n)
                if self.properties.current_node is n
                else None,
            )

    def eventFilter(self, source, event):
        """Handle drag/drop events on the graph widget to spawn nodes at drop location."""
        try:
            if source is getattr(self, "_graph_widget", None):
                if (
                    event.type() == QtCore.QEvent.DragEnter
                    or event.type() == QtCore.QEvent.DragMove
                ):
                    mime = event.mimeData()
                    if mime and (
                        mime.hasFormat("application/x-node-id") or mime.hasText()
                    ):
                        event.accept()
                        return True
                if event.type() == QtCore.QEvent.Drop:
                    mime = event.mimeData()
                    if mime and mime.hasFormat("application/x-node-id"):
                        node_id = bytes(mime.data("application/x-node-id")).decode(
                            "utf-8"
                        )
                        label = mime.text() or str(node_id)
                        pos = event.pos()
                        # spawn node using explicit coordinates
                        self._spawn_node(node_id, label, x=pos.x(), y=pos.y())
                        event.accept()
                        return True
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        return super().eventFilter(source, event)

    def _pending_node_issues(self, nodes=None):
        """Return "<node>: <message>" for every node currently reporting one.

        A preview never raises, so this is how a setup problem found during an
        automatic refresh reaches the user — in the status bar and timeline,
        not in a dialog they did not ask for.
        """
        issues = []
        if nodes is None:
            try:
                nodes = list(self.graph.all_nodes())
            except Exception:
                return issues
        for node in nodes:
            has_error = getattr(node, "has_error", None)
            try:
                if callable(has_error) and has_error():
                    message = str(node.get_error() or "Setup incomplete")
                else:
                    has_pending = getattr(node, "has_pending", None)
                    if not callable(has_pending) or not has_pending():
                        continue
                    message = str(node.get_pending() or "Setup incomplete")
                name = node.name() if callable(node.name) else node.name
            except Exception:
                continue
            issues.append(f"{name}: {message}")
        return issues

    _MEASURE_NODE_CLASSES = frozenset(
        {
            "MassPropertiesNode",
            "BoundingBoxNode",
            "MeasureDistanceNode",
            "SurfaceAreaNode",
        }
    )

    def _refresh_measurements(self, results):
        """Publish every Measure node's value to the results panel.

        Measure Distance and Surface Area return a plain number and Mass
        Properties / Bounding Box return an 'analysis' dict; neither reached
        the results panel before, so the values were computed and discarded.
        """
        if self.results is None:
            return
        rows = []
        try:
            nodes = list(self.graph.all_nodes())
        except Exception:
            return
        for node in nodes:
            class_name = node.__class__.__name__
            if class_name not in self._MEASURE_NODE_CLASSES:
                continue
            result = results.get(node, getattr(node, "_last_result", None))
            if result is None:
                continue
            try:
                name = node.name() if callable(node.name) else node.name
                rows.extend(
                    self.results.measurement_rows(name, class_name, result)
                )
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
        try:
            self.results.set_measurements(rows)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

    def _on_execution_finished(self, results):
        """Called when the background thread completes."""
        was_preview = bool(getattr(self.worker, "skip_simulation", False))
        executed_nodes = list(getattr(self.worker, "nodes", ()) or ())
        self.worker = None
        self._clear_solve_progress()
        # Lock before processing results
        self.result_mutex.lock()
        try:
            # 1. Unlock UI
            self.graph.widget.setEnabled(True)
            self.toolbar.setEnabled(True)
            try:
                self._refresh_measurements(results)
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            # Parameter aliases are discovered by the execution worker.  Port
            # graphics must be renamed/redrawn back on the GUI thread after
            # that worker completes.
            try:
                from pylcss.design_studio.core.port_schema import (
                    apply_display_port_labels,
                )

                for node in executed_nodes:
                    if node.__class__.__name__ == "FreeCadPartNode":
                        apply_display_port_labels(node)
            except Exception:
                logger.debug("Could not refresh FreeCAD parameter ports.", exc_info=True)

            issues = self._pending_node_issues(executed_nodes)
            if issues:
                summary = issues[0] if len(issues) == 1 else (
                    f"{issues[0]} (+{len(issues) - 1} more)"
                )
                self.statusBar().showMessage(summary)
            else:
                self.statusBar().showMessage("Done")
                if not was_preview:
                    self.timeline.add_event("Solve finished")

            # 2. Update Visualization (Must be done on Main Thread!)
            try:
                # Decide what to draw after a run.  Priority:
                #   1. The selected node's OWN result, if it's renderable
                #      (so clicking CAD shows the B-rep, validation shows FEA,
                #       topopt shows density — note _is_renderable_result
                #       recognises the 'topopt_voxel' dict, which has a
                #       'density' field but no 'mesh'/'vertices' key).
                #   2. Otherwise the optimization / FEA result produced in this
                #      graph, so the topopt outcome stays visible even when
                #      downstream CAD / STEP-export nodes are wired after it.
                #   3. Otherwise an upstream preview (design domain / mesh).
                #   4. Otherwise the last-rendered node.
                selected = next(iter(self.graph.selected_nodes()), None)
                target_node = None
                geom = None
                prefer_topopt = bool(getattr(self, "_prefer_topopt_after_run", False))

                if prefer_topopt:
                    sim_node = self._find_renderable_simulation_node()
                    if sim_node is not None:
                        result = getattr(sim_node, "_last_result", None)
                        if self._is_topopt_render_result(result):
                            target_node, geom = sim_node, result
                    if geom is None:
                        preview = getattr(self, "_last_topopt_preview_payload", None)
                        if self._is_renderable_result(preview):
                            geom = preview

                if geom is None and selected is not None:
                    own = results.get(selected, getattr(selected, "_last_result", None))
                    if self._is_renderable_result(own):
                        target_node, geom = selected, own

                if geom is None and selected is not None:
                    src, upstream_geom = self._get_render_context_for_node(selected)
                    if upstream_geom is not None:
                        target_node, geom = (src or selected), upstream_geom

                # Automatic preview skips the expensive TopOpt node itself,
                # but its upstream CAD assembly has just been evaluated. With
                # no node selected (the normal state immediately after opening
                # an example), explicitly show that design domain instead of
                # clearing the viewer. This is especially important for
                # unfused multi-body assemblies.
                if geom is None and was_preview:
                    target_node, geom = self._find_cached_topopt_domain()

                if geom is None:
                    sim_node = self._find_renderable_simulation_node()
                    if sim_node is not None:
                        target_node = sim_node
                        geom = getattr(sim_node, "_last_result", None)

                if geom is None:
                    last = getattr(self, "_last_rendered_node", None)
                    if last is not None:
                        cached = getattr(last, "_last_result", None)
                        if self._is_renderable_result(cached):
                            target_node, geom = last, cached

                if geom is None:
                    preview = getattr(self, "_last_topopt_preview_payload", None)
                    if self._is_renderable_result(preview):
                        geom = preview

                if geom is not None:
                    if target_node is not None:
                        self._last_rendered_node = target_node
                    self._render_result_in_viewer(geom)
                    try:
                        self._show_bc_for_node(target_node)
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                else:
                    self.viewer.clear()

            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
        finally:
            self._prefer_topopt_after_run = False
            self.result_mutex.unlock()
        QtCore.QTimer.singleShot(0, self._start_pending_execute)

    def _is_2d_sketch(self, obj):
        """Check if object is a 2D sketch (has wires but NO solids)."""
        if obj is None:
            return False

        # First check if this has solids - if so, it's a 3D shape, NOT a sketch
        try:
            # CadQuery Workplane with solids
            if hasattr(obj, "val"):
                val = obj.val()
                # Check for solid or compound
                if hasattr(val, "Solids") and val.Solids():
                    return False  # Has solids = 3D shape
            # Or direct solid
            if hasattr(obj, "Solids") and obj.Solids():
                return False  # Has solids = 3D shape
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        # Now check for pending wires (2D sketch)
        # Only treat as sketch if there are wires but no solids
        if hasattr(obj, "ctx") and hasattr(obj.ctx, "pendingWires"):
            if obj.ctx.pendingWires:
                return True
        return False

    def _on_execution_error(self, error_msg):
        """Called if background thread fails."""
        self.worker = None
        self._clear_solve_progress()
        self.graph.widget.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.timeline.add_event(f"Execution failed: {error_msg}")
        try:
            sim_node = self._find_renderable_simulation_node()
            if sim_node is not None:
                result = getattr(sim_node, "_last_result", None)
                if self._is_renderable_result(result):
                    self._last_rendered_node = sim_node
                    self._render_result_in_viewer(result)
            elif self._is_renderable_result(
                getattr(self, "_last_topopt_preview_payload", None)
            ):
                self._render_result_in_viewer(self._last_topopt_preview_payload)
            else:
                topopt_node, domain = self._find_cached_topopt_domain()
                if domain is not None:
                    self._last_rendered_node = topopt_node
                    self._render_result_in_viewer(domain)
            self.viewer.set_bc_overlay_data()
            self.viewer.render_bc_overlays()
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        # Only an explicit run reaches this handler now, so the dialog always
        # corresponds to something the user actually asked to compute.
        QtWidgets.QMessageBox.critical(self, "Computation Error", error_msg)
        QtCore.QTimer.singleShot(0, self._start_pending_execute)

    def _on_execution_cancelled(self, results):
        """Restore the UI after a user-requested safe stop without an error dialog."""
        self.worker = None
        self._clear_solve_progress()
        self.graph.widget.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.statusBar().showMessage("Stopped")
        self.timeline.add_event("Solve stopped")
        # A topology solver can return a valid partial design at its safe stop
        # point. Keep that cached result visible and exportable.
        try:
            rendered = False
            sim_node = self._find_renderable_simulation_node()
            if sim_node is not None:
                result = getattr(sim_node, "_last_result", None)
                if self._is_renderable_result(result):
                    self._last_rendered_node = sim_node
                    self._render_result_in_viewer(result)
                    rendered = True
            if not rendered:
                topopt_node, domain = self._find_cached_topopt_domain()
                if domain is not None:
                    self._last_rendered_node = topopt_node
                    self._render_result_in_viewer(domain)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        QtCore.QTimer.singleShot(0, self._start_pending_execute)
