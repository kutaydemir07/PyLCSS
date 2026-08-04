# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""CAD and Design Studio commands exposed through the assistant."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pylcss.design_studio.node_library import NODE_CLASS_MAPPING

logger = logging.getLogger(__name__)


class CadCommandsMixin:
    """Create, execute, inspect, save, and export Design Studio graphs."""

    main_window: Any
    _run_sync: Callable[..., Any]

    def _apply_layout(
        self,
        nodes: list[Any],
        connections: list[dict[str, Any]],
        start_x: int,
        start_y: int,
    ) -> None:
        """
        Simple layered layout algorithm (Left-to-Right).
        """
        if not nodes:
            return

        # Build local adjacency
        node_ids = {n.name(): n for n in nodes}  # Assumes name matches ID
        adj = {n.name(): [] for n in nodes}
        in_degree = {n.name(): 0 for n in nodes}

        for conn in connections:
            from_str = conn.get("from", "")
            to_str = conn.get("to", "")
            if "." not in from_str or "." not in to_str:
                continue

            u = from_str.split(".")[0]
            v = to_str.split(".")[0]

            if u in adj and v in in_degree:
                adj[u].append(v)
                in_degree[v] += 1

        # Assign ranks & topological sort roughly
        queue = [n_id for n_id, d in in_degree.items() if d == 0]
        ranks = {n_id: 0 for n_id in node_ids}

        processed = set()
        while queue:
            curr = queue.pop(0)
            processed.add(curr)

            curr_rank = ranks[curr]

            for neighbor in adj[curr]:
                ranks[neighbor] = max(ranks[neighbor], curr_rank + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle cycles/remaining nodes
        max_rank = max(ranks.values()) if ranks else 0

        for n_id in node_ids:
            if n_id not in processed:
                ranks[n_id] = max_rank + 1

        # Group by rank and assign positions
        rank_groups = {}
        for n_id, r in ranks.items():
            if r not in rank_groups:
                rank_groups[r] = []
            rank_groups[r].append(n_id)

        X_SPACING = 350
        Y_SPACING = 200

        for r in sorted(rank_groups.keys()):
            group = rank_groups[r]
            for i, n_id in enumerate(group):
                x = start_x + (r * X_SPACING)
                y = start_y + (i * Y_SPACING)
                if n_id in node_ids:
                    node_ids[n_id].set_pos(x, y)

    def _build_node_graph(self, data: dict[str, Any], sync: bool = False) -> Any:
        """
        Build a complete CAD node graph from LLM specification.
        Contains robust logic for node creation and property setting.
        """
        params = data.get("params", {})
        nodes_spec = params.get("nodes", [])
        conns_spec = params.get("connections", [])

        if not nodes_spec:
            logger.warning("Empty node spec for build_node_graph")
            return

        def run_tool() -> dict[str, Any]:
            graph = None
            try:
                if not hasattr(self.main_window, "cad_widget"):
                    raise RuntimeError("Design Studio is not available.")

                cad_widget = self.main_window.cad_widget
                graph = getattr(cad_widget, "graph", None)
                if not graph:
                    raise RuntimeError(
                        "Design Studio graph controller is not available."
                    )

                # Block graph property-changed / connection-changed signals
                # during programmatic build to prevent auto-update from
                # executing incomplete graphs.
                graph.blockSignals(True)

                created_info = []
                warnings = []
                id_to_node = {}
                for existing_node in graph.all_nodes():
                    # Copilot state exposes both the engineer-facing name and
                    # NodeGraphQt's stable ID.  Accept either on subsequent
                    # edits and connections.
                    id_to_node[str(existing_node.name())] = existing_node
                    stable_id = str(getattr(existing_node, "id", "") or "")
                    if stable_id:
                        id_to_node[stable_id] = existing_node

                # Determine start position
                start_x, start_y = 0, 0
                existing = graph.all_nodes()
                if existing:
                    start_y = max((n.pos()[1] for n in existing), default=0) + 200

                count = 0

                # Process Nodes
                for spec in nodes_spec:
                    spec_id = spec.get("id")
                    spec_type = spec.get("type")
                    spec_props = spec.get("properties", {})

                    if not spec_id:
                        continue

                    node = None
                    if spec_id in id_to_node:
                        # Update
                        node = id_to_node[spec_id]
                        for k, v in spec_props.items():
                            if hasattr(node, "set_property"):
                                issue = self._set_cad_property_safe(node, k, v)
                                if issue:
                                    warnings.append(f"{node.name()}.{k}: {issue}")
                        created_info.append(f"Updated {node.name()}")
                    else:
                        # Create
                        node_class = NODE_CLASS_MAPPING.get(spec_type)
                        if not node_class:
                            warnings.append(f"Unknown CAD node type: {spec_type}")
                            continue

                        node = node_class()
                        row = count // 4
                        col = count % 4
                        node.set_pos(start_x + (col * 250), start_y + (row * 150))
                        graph.add_node(node)

                        node.set_name(spec_id)
                        count += 1
                        id_to_node[spec_id] = node
                        stable_id = str(getattr(node, "id", "") or "")
                        if stable_id:
                            id_to_node[stable_id] = node

                        for k, v in spec_props.items():
                            issue = self._set_cad_property_safe(node, k, v)
                            if issue:
                                warnings.append(f"{node.name()}.{k}: {issue}")

                        created_info.append(f"Created {spec_id}")

                # Process Connections
                for conn in conns_spec:
                    try:
                        # Handle multiple connection schemas
                        from_str = conn.get("from")
                        to_str = conn.get("to")

                        # Fallback to alternate keys (LLM hallucination safety)
                        if not from_str:
                            f_node = conn.get("from_node", "")
                            f_port = conn.get("from_port", "shape")
                            if f_node:
                                from_str = f"{f_node}.{f_port}"

                        if not to_str:
                            t_node = conn.get("to_node", "")
                            t_port = conn.get("to_port", "shape")
                            if t_node:
                                to_str = f"{t_node}.{t_port}"

                        if (
                            not from_str
                            or not to_str
                            or "." not in from_str
                            or "." not in to_str
                        ):
                            continue

                        from_id, from_port = from_str.split(".", 1)
                        to_id, to_port = to_str.split(".", 1)

                        src = id_to_node.get(from_id)
                        dst = id_to_node.get(to_id)

                        if src and dst:
                            out_p = src.get_output(from_port)
                            # Fallback for common port names
                            if not out_p:
                                out_p = (
                                    src.get_output("result")
                                    or src.get_output("out")
                                    or src.get_output("shape")
                                )

                            in_p = dst.get_input(to_port)
                            # Fallback for common port names
                            if not in_p:
                                in_p = (
                                    dst.get_input("input_shape")
                                    or dst.get_input("target")
                                    or dst.get_input("shape")
                                )

                            if out_p and in_p:
                                out_p.connect_to(in_p)
                            else:
                                warnings.append(
                                    f"Could not connect {from_str} to {to_str}: "
                                    "one of the ports does not exist."
                                )
                        else:
                            warnings.append(
                                f"Could not connect {from_str} to {to_str}: "
                                "one of the nodes does not exist."
                            )
                    except Exception as e:
                        warnings.append(f"Connection failed: {e}")

                # Apply Layout
                batch_nodes = [
                    id_to_node[n.get("id")]
                    for n in nodes_spec
                    if n.get("id") in id_to_node
                ]
                self._apply_layout(batch_nodes, conns_spec, start_x, start_y)

                # Set the last created node as the render target so the
                # viewer shows the most recent geometry after execution.
                if batch_nodes:
                    cad_widget._last_rendered_node = batch_nodes[-1]

                logger.info(f"CAD Graph operation complete: {', '.join(created_info)}")
                return {
                    "success": not warnings,
                    "changes": created_info,
                    "warnings": warnings,
                    "node_count": len(graph.all_nodes()),
                }

            except Exception as e:
                logger.error(f"Failed to build CAD graph: {e}")
                if sync:
                    raise
                return {"success": False, "error": str(e)}
            finally:
                # Always re-enable signals so normal UI interaction works
                if graph:
                    graph.blockSignals(False)

        if sync:
            return self._run_sync(run_tool)
        else:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, run_tool)
            return {"success": True, "status": "scheduled"}

    def _set_cad_property_safe(self, node: Any, k: str, v: Any) -> str | None:
        """Set one native CAD property and report rejected copilot input."""
        if not hasattr(node, "has_property"):
            return "node does not expose editable properties"

        # Normalize Property Name
        final_prop_name = k
        if not node.has_property(k):
            custom_props = (
                node.properties().get("custom", {})
                if isinstance(node.properties(), dict)
                else {}
            )
            for p in custom_props:
                if p.lower() == k.lower():
                    final_prop_name = p
                    break

        if not node.has_property(final_prop_name):
            valid = sorted((node.properties().get("custom", {}) or {}).keys())
            return (
                f"unknown property; valid properties are {', '.join(valid)}"
                if valid
                else "unknown property"
            )

        # Normalize Value (Synonyms)
        final_val = v
        ENUM_MAP = {
            "operation": {
                "targets": ["Union", "Subtract", "Intersect"],
                "synonyms": {
                    "difference": "Subtract",
                    "cut": "Subtract",
                    "subtract": "Subtract",
                    "remove": "Subtract",
                    "add": "Union",
                    "combine": "Union",
                    "intersection": "Intersect",
                },
            }
        }

        prop_key_lower = final_prop_name.lower()
        map_entry = None
        for known_key, config in ENUM_MAP.items():
            if known_key in prop_key_lower:
                map_entry = config
                break

        if map_entry and isinstance(v, str):
            val_lower = v.lower()
            found = False
            for target in map_entry["targets"]:
                if target.lower() == val_lower:
                    final_val = target
                    found = True
                    break
            if not found and val_lower in map_entry["synonyms"]:
                final_val = map_entry["synonyms"][val_lower]

        attrs = getattr(getattr(node, "model", None), "_TEMP_property_attrs", {})
        items = (attrs.get(final_prop_name, {}) or {}).get("items")
        if items and isinstance(final_val, str):
            exact = next(
                (
                    candidate
                    for candidate in items
                    if str(candidate).casefold() == final_val.casefold()
                ),
                None,
            )
            if exact is None:
                return (
                    f"invalid value {final_val!r}; choose one of "
                    f"{', '.join(map(str, items))}"
                )
            final_val = exact

        current = node.get_property(final_prop_name)
        try:
            if isinstance(current, bool):
                if isinstance(final_val, str):
                    normalized = final_val.strip().casefold()
                    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
                        return "expected a boolean value"
                    final_val = normalized in {"true", "1", "yes"}
                else:
                    final_val = bool(final_val)
            elif isinstance(current, int) and not isinstance(current, bool):
                final_val = int(final_val)
            elif isinstance(current, float):
                final_val = float(final_val)
        except (TypeError, ValueError):
            return (
                f"value {final_val!r} is not compatible with {type(current).__name__}"
            )

        try:
            node.set_property(final_prop_name, final_val)
        except Exception as exc:
            return str(exc)
        return None

    def _add_cad_node(self, command: str) -> None:
        """Add a node in the Design Studio environment."""
        if not self.main_window:
            return
        if not hasattr(self.main_window, "cad_widget"):
            logger.warning("CAD widget not found")
            return

        widget = self.main_window.cad_widget

        # Map command to node type and display name
        node_map = {
            "cad_add_box": ("com.cad.geometry.box", "Box"),
            "cad_add_cylinder": ("com.cad.geometry.cylinder", "Cylinder"),
            "cad_add_fillet": ("com.cad.geometry.fillet", "Fillet"),
            "cad_add_boolean": ("com.cad.geometry.boolean", "Boolean"),
            "cad_add_union": ("com.cad.geometry.boolean", "Boolean Union"),
            "cad_add_cut": ("com.cad.geometry.boolean", "Boolean Subtract"),
        }

        node_info = node_map.get(command)
        if node_info:
            node_type, label = node_info
            # The CAD widget uses _spawn_node method
            if hasattr(widget, "_spawn_node"):
                try:
                    widget._spawn_node(node_type, label)
                    logger.info(f"Assistant: Added CAD node {label}")
                except Exception as e:
                    logger.error(f"Failed to create CAD node: {e}")
            else:
                logger.warning("CAD widget does not have _spawn_node method")
        else:
            logger.warning(f"Unknown CAD command: {command}")

    def _cad_execute(self, sync: bool = False) -> None:
        """Execute/run the CAD graph.

        When *sync=True* (used by the agentic tool handler), the engine
        is run **directly** on the main thread (via ``_run_sync``), the
        node results are updated inline, and the 3D viewer is refreshed
        immediately.  This avoids worker-thread signal-queueing issues
        that caused the viewer to show stale or empty geometry.
        """
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            logger.warning("CAD widget not found")
            if sync:
                raise RuntimeError("CAD widget not found")
            return
        widget = self.main_window.cad_widget

        if sync:
            # ---- synchronous path (agentic system) ---------------------
            def _run_engine_and_render() -> str:
                # Wait for any previous async worker first
                prev = getattr(widget, "worker", None)
                if prev and prev.isRunning():
                    logger.info("Waiting for previous CAD worker to finish…")
                    prev.wait()
                    # Drain its queued _on_execution_finished so it doesn't
                    # overwrite the render we're about to do.
                    from PySide6.QtWidgets import QApplication

                    QApplication.processEvents()

                from pylcss.design_studio.engine import execute_graph

                nodes = list(widget.graph.all_nodes())
                results = execute_graph(nodes)

                # Pick the render target (same logic as _on_execution_finished)
                target = getattr(widget, "_last_rendered_node", None)
                if target is None and nodes:
                    target = nodes[-1]
                geom = (
                    results.get(target, getattr(target, "_last_result", None))
                    if target
                    else None
                )

                if geom is not None:
                    widget._last_rendered_node = target
                    if isinstance(geom, dict) and (
                        "mesh" in geom or "displacement" in geom
                    ):
                        widget.viewer.render_simulation(geom)
                    elif widget._is_2d_sketch(geom):
                        widget.viewer.render_sketch(geom)
                    else:
                        widget.viewer.render_shape(geom)
                logger.info("Assistant: Executing CAD graph (sync)")
                return "CAD executed"

            self._run_sync(_run_engine_and_render)
        else:
            # ---- async path (UI button / assistant shortcut) -----------
            from PySide6.QtCore import QMetaObject, Qt

            if hasattr(widget, "execute_graph"):
                QMetaObject.invokeMethod(widget, "execute_graph", Qt.QueuedConnection)
                logger.info("Assistant: Executing CAD graph")
            elif hasattr(widget, "btn_execute"):
                QMetaObject.invokeMethod(
                    widget.btn_execute, "click", Qt.QueuedConnection
                )
                logger.info("Assistant: Running CAD")

    def _cad_export(self, sync: bool = False) -> None:
        """Export the CAD model to STL."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            return
        widget = self.main_window.cad_widget
        from PySide6.QtCore import QMetaObject, Qt

        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection

        if hasattr(widget, "btn_export"):
            QMetaObject.invokeMethod(widget.btn_export, "click", conn_type)
            logger.info("Assistant: Exporting CAD")

    def _cad_execute_scoped(
        self,
        terminal_node: str | None = None,
        *,
        preview: bool = False,
        sync: bool = False,
    ) -> Any:
        """Start a responsive, workflow-scoped Design Studio execution."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")
        widget = self.main_window.cad_widget

        def _start() -> dict[str, Any]:
            worker = getattr(widget, "worker", None)
            if worker is not None and worker.isRunning():
                raise RuntimeError(
                    "A Design Studio computation is already running. Stop it "
                    "or wait for it to finish."
                )

            all_nodes = list(widget.graph.all_nodes())
            terminal_ids = {
                "com.cad.sim.solver",
                "com.cad.sim.crash_solver",
                "com.cad.sim.topopt_voxel",
                "com.cad.sim.lattice_voxel",
            }
            terminals = [
                node
                for node in all_nodes
                if getattr(node, "__identifier__", "") in terminal_ids
            ]
            target = None
            if terminal_node:
                wanted = str(terminal_node).strip().casefold()
                matches = [
                    node
                    for node in all_nodes
                    if (
                        str(node.name()).casefold() == wanted
                        or str(getattr(node, "id", "")).casefold() == wanted
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Workflow terminal {terminal_node!r} was not found uniquely."
                    )
                target = matches[0]
            elif len(terminals) == 1:
                target = terminals[0]
            elif len(terminals) > 1 and not preview:
                names = ", ".join(str(node.name()) for node in terminals)
                raise ValueError(
                    "This graph contains multiple solver workflows. Pass "
                    f"terminal_node as one of: {names}."
                )

            nodes = (
                widget._upstream_closure(target) if target is not None else all_nodes
            )
            if target is not None:
                widget.graph.clear_selection()
                target.set_selected(True)
                widget._last_rendered_node = target
            widget._execute_graph(
                skip_simulation=bool(preview),
                nodes=nodes,
            )
            return {
                "success": True,
                "status": "started",
                "mode": "preview" if preview else "full",
                "terminal": str(target.name()) if target is not None else None,
                "node_count": len(nodes),
            }

        if sync:
            return self._run_sync(_start)
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, _start)
        return {"success": True, "status": "scheduled"}

    def _cad_stop(self, sync: bool = False) -> Any:
        """Request a safe stop of the active Design Studio computation."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")

        def _stop() -> dict[str, Any]:
            widget = self.main_window.cad_widget
            worker = getattr(widget, "worker", None)
            if worker is None or not worker.isRunning():
                return {"success": True, "status": "idle"}
            widget._cancel_execution()
            return {"success": True, "status": "stop_requested"}

        return self._run_sync(_stop) if sync else _stop()

    def _cad_select_node(self, node_ref: str, *, sync: bool = False) -> Any:
        """Select a Design Studio node by name or stable ID."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")
        wanted = str(node_ref or "").strip().casefold()
        if not wanted:
            raise ValueError("A node name or stable ID is required.")
        widget = self.main_window.cad_widget

        def _select() -> dict[str, Any]:
            matches = [
                node
                for node in widget.graph.all_nodes()
                if (
                    str(node.name()).casefold() == wanted
                    or str(getattr(node, "id", "") or "").casefold() == wanted
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Design Studio node {node_ref!r} was not found uniquely."
                )
            target = matches[0]
            widget.graph.clear_selection()
            target.set_selected(True)
            # NodeGraphQt normally emits node_selected here.  Calling the
            # handler explicitly makes the copilot result deterministic even
            # if a Qt signal is coalesced while a tool call is completing.
            widget._on_node_selected(target)
            return {
                "success": True,
                "id": str(getattr(target, "id", "") or ""),
                "name": str(target.name()),
                "has_cached_result": getattr(target, "_last_result", None) is not None,
            }

        return self._run_sync(_select) if sync else _select()

    def _cad_save_project_file(self, filename: str, *, sync: bool = False) -> Any:
        """Save the editable Design Studio graph and numerical result sidecar."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")
        if not str(filename or "").strip():
            raise ValueError("An explicit .cad filename is required.")
        widget = self.main_window.cad_widget

        def _save() -> dict[str, Any]:
            import os

            path = os.path.abspath(os.path.expanduser(str(filename)))
            if not path.lower().endswith(".cad"):
                path += ".cad"
            count = widget.save_project_file(path)
            return {
                "success": True,
                "path": path,
                "saved_result_sets": int(count),
                "result_sidecar": path + ".results.h5",
            }

        return self._run_sync(_save) if sync else _save()

    def _cad_load_project_file(self, filename: str, *, sync: bool = False) -> Any:
        """Load a Design Studio graph and its saved numerical results."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")
        if not str(filename or "").strip():
            raise ValueError("An explicit .cad filename is required.")
        widget = self.main_window.cad_widget

        def _load() -> dict[str, Any]:
            import os

            path = os.path.abspath(os.path.expanduser(str(filename)))
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            count = widget.load_project_file(path, preview=False)
            return {
                "success": True,
                "path": path,
                "restored_result_sets": int(count),
                "node_count": len(widget.graph.all_nodes()),
            }

        return self._run_sync(_load) if sync else _load()

    def _cad_export_file(
        self,
        file_format: str,
        filename: str,
        *,
        sync: bool = False,
    ) -> Any:
        """Export selected/cached CAD or a recovered topology shape."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            raise RuntimeError("Design Studio is not available.")
        fmt = str(file_format or "").strip().lower()
        if fmt not in {"step", "stl"}:
            raise ValueError("Export format must be 'step' or 'stl'.")
        if not filename:
            raise ValueError("Export requires an explicit filename.")
        widget = self.main_window.cad_widget

        def _export() -> dict[str, Any]:
            import os
            import cadquery as cq

            path = os.path.abspath(os.path.expanduser(str(filename)))
            if not os.path.splitext(path)[1]:
                path += ".step" if fmt == "step" else ".stl"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            selected = list(widget.graph.selected_nodes())
            target = (
                selected[0]
                if len(selected) == 1
                else getattr(widget, "_last_rendered_node", None)
            )
            payload = (
                getattr(target, "_last_result", None) if target is not None else None
            )
            if payload is None:
                raise RuntimeError(
                    "No cached geometry is available. Run or preview the target "
                    "node before exporting it."
                )

            if isinstance(payload, dict) and payload.get("type") == "topopt_voxel":
                if fmt == "step":
                    from pylcss.user_interface.cad.cad_widget import (
                        _external_write_cad_step,
                    )

                    _external_write_cad_step(payload, path)
                else:
                    recovered = payload.get("recovered_shape") or {}
                    vertices = recovered.get("vertices")
                    faces = recovered.get("faces")
                    if vertices is None or faces is None:
                        raise RuntimeError(
                            "The topology result has no recovered surface to export."
                        )
                    import trimesh

                    trimesh.Trimesh(
                        vertices=vertices, faces=faces, process=False
                    ).export(path)
            elif isinstance(payload, dict):
                raise RuntimeError(
                    "The selected cached result is analysis data, not exportable CAD."
                )
            else:
                cq.exporters.export(
                    payload,
                    path,
                    exportType="STEP" if fmt == "step" else "STL",
                )
            return {"success": True, "path": path, "format": fmt}

        return self._run_sync(_export) if sync else _export()

    def _get_cad_state(self, sync: bool = False) -> dict[str, Any]:
        """Return Design Studio nodes, connections, workflows, and result state."""
        if not self.main_window or not hasattr(self.main_window, "cad_widget"):
            return {"error": "Design Studio is not available"}
        widget = self.main_window.cad_widget

        def _state() -> dict[str, Any]:
            nodes = []
            workflows = []
            terminal_ids = {
                "com.cad.sim.solver",
                "com.cad.sim.crash_solver",
                "com.cad.sim.topopt_voxel",
                "com.cad.sim.lattice_voxel",
            }
            for node in widget.graph.all_nodes():
                node_type = str(getattr(node, "__identifier__", "") or node.type_)
                node_properties = node.properties()
                custom_properties = (
                    dict(node_properties.get("custom", {}) or {})
                    if isinstance(node_properties, dict)
                    else {}
                )
                result = getattr(node, "_last_result", None)
                result_summary = None
                if isinstance(result, dict):
                    useful_keys = (
                        "type",
                        "converged",
                        "message",
                        "iterations",
                        "max_iterations",
                        "compliance",
                        "peak_displacement",
                        "peak_stress",
                        "peak_stress_nodal",
                        "absorbed_energy",
                        "energy_balance_max_error",
                        "final_vol_frac",
                    )
                    result_summary = {
                        key: result[key]
                        for key in useful_keys
                        if key in result
                        and isinstance(result[key], (str, int, float, bool, type(None)))
                    }
                entry = {
                    "id": str(getattr(node, "id", "") or ""),
                    "name": str(node.name()),
                    "type": node_type,
                    "properties": custom_properties,
                    "has_cached_result": result is not None,
                    "result_summary": result_summary,
                    "error": str(getattr(node, "_error_message", "") or ""),
                }
                nodes.append(entry)
                if node_type in terminal_ids:
                    workflows.append(
                        {
                            "id": entry["id"],
                            "name": entry["name"],
                            "type": node_type,
                            "has_cached_result": entry["has_cached_result"],
                        }
                    )

            session = widget.graph.serialize_session()
            worker = getattr(widget, "worker", None)
            return {
                "node_count": len(nodes),
                "nodes": nodes,
                "connections": list(session.get("connections", [])),
                "workflows": workflows,
                "running": bool(worker is not None and worker.isRunning()),
                "project_file": widget.current_file,
            }

        return self._run_sync(_state) if sync else _state()
