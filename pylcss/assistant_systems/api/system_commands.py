# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""System-modeling graph commands exposed through the assistant."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pylcss.user_interface.system_modeling.node_registry import (
    SYSTEM_NODE_CLASS_MAPPING,
)

logger = logging.getLogger(__name__)


class SystemCommandsMixin:
    """Create, connect, inspect, and edit system-modeling graphs."""

    main_window: Any
    _run_sync: Callable[..., Any]

    def _build_system_graph(self, data: dict[str, Any], sync: bool = False) -> Any:
        """
        Build a complete System node graph from LLM specification.
        Parallel to _build_node_graph but for the Modeling environment.
        """
        params = data.get("params", {})
        nodes_spec = params.get("nodes", [])
        conns_spec = params.get("connections", [])

        if not nodes_spec:
            logger.warning("Empty node spec for build_system_graph")
            return

        if not self.main_window:
            logger.warning("Main window required for build_system_graph")
            if sync:
                raise RuntimeError("Main window required")
            return

        if not hasattr(self.main_window, "modeling_widget"):
            logger.error("Modeling widget not found")
            if sync:
                raise RuntimeError("Modeling widget not found")
            return
        modeling_widget = self.main_window.modeling_widget
        graph_controller = getattr(modeling_widget, "current_graph", None)

        if not graph_controller:
            logger.error("System Graph controller not found")
            if sync:
                raise RuntimeError("System Graph controller not found")
            return

        # 3. Execute Graph Building on GUI Thread
        def build_graph_safe() -> dict[str, Any]:
            try:
                tab_widget = self._get_tab_widget()
                if tab_widget and tab_widget.currentIndex() != 0:
                    tab_widget.setCurrentIndex(0)

                id_to_node = {}

                # A. Map Existing Nodes
                existing_nodes = graph_controller.all_nodes()
                existing_by_name = {n.name(): n for n in existing_nodes}
                id_to_node.update(existing_by_name)

                # Determine start position
                start_x, start_y = 0, 0
                if existing_nodes:
                    start_y = max((n.pos()[1] for n in existing_nodes), default=0) + 200

                # B. Process Nodes
                new_node_count = 0

                for node_def in nodes_spec:
                    node_id = node_def.get("id")
                    node_type = node_def.get("type")
                    props = node_def.get("properties", {})

                    if not node_id:
                        continue

                    if node_id in existing_by_name:
                        # Update
                        node = existing_by_name[node_id]
                        logger.info(f"Updating existing system node: {node_id}")
                        for prop_name, prop_val in props.items():
                            self._set_node_property_safe(
                                node, prop_name, prop_val, node_id
                            )
                        id_to_node[node_id] = node
                    else:
                        # Create
                        if not node_type:
                            continue

                        node_class = SYSTEM_NODE_CLASS_MAPPING.get(node_type)
                        if not node_class:
                            logger.warning(f"Unknown system node type: {node_type}")
                            continue

                        node = node_class()
                        node.set_name(str(node_id))

                        # Position
                        row = new_node_count // 3
                        col = new_node_count % 3
                        node.set_pos(start_x + (col * 300), start_y + (row * 150))
                        new_node_count += 1

                        graph_controller.add_node(node)
                        id_to_node[node_id] = node

                        for prop_name, prop_val in props.items():
                            self._set_node_property_safe(
                                node, prop_name, prop_val, node_id
                            )

                # C. Process Connections
                for conn in conns_spec:
                    try:
                        from_str = conn.get("from", "")
                        to_str = conn.get("to", "")

                        if "." not in from_str or "." not in to_str:
                            continue

                        from_id, from_port = from_str.split(".", 1)
                        to_id, to_port = to_str.split(".", 1)

                        src_node = id_to_node.get(from_id)
                        dst_node = id_to_node.get(to_id)

                        if src_node and dst_node:
                            out_port = src_node.get_output(from_port)
                            in_port = dst_node.get_input(to_port)
                            if out_port and in_port:
                                out_port.connect_to(in_port)
                    except Exception as e:
                        logger.warning(f"Connection failed: {e}")

                logger.info(
                    f"System Graph update complete. Processed {len(nodes_spec)} nodes."
                )
                return {
                    "success": True,
                    "processed_nodes": len(nodes_spec),
                }

            except Exception:
                logger.exception("System Graph build failed")
                if sync:
                    raise
                return {"success": False, "error": "System graph build failed"}

        if sync:
            return self._run_sync(build_graph_safe)
        else:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, build_graph_safe)
            return {"success": True, "status": "scheduled"}

    def _add_system(self) -> None:
        """Add a new system in modeling environment."""
        if not self.main_window:
            return
        if hasattr(self.main_window, "modeling_widget"):
            widget = self.main_window.modeling_widget
            if hasattr(widget, "system_manager") and hasattr(
                widget.system_manager, "add_system"
            ):
                from PySide6.QtCore import QMetaObject, Qt

                QMetaObject.invokeMethod(
                    widget.system_manager, "add_system", Qt.QueuedConnection
                )
                logger.info("Assistant: Adding new system")

    def _add_modeling_node(self, command: str) -> None:
        """Add a node in the modeling environment."""
        if not self.main_window:
            return
        if not hasattr(self.main_window, "modeling_widget"):
            logger.warning("Modeling widget not found")
            return

        widget = self.main_window.modeling_widget

        # Map command to method
        method_map = {
            "add_input": "add_input_node",
            "add_output": "add_output_node",
            "add_function": "add_function_node",
            "add_intermediate": "add_intermediate_node",
        }

        method_name = method_map.get(command)
        if method_name and hasattr(widget, method_name):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, method_name, Qt.QueuedConnection)
            logger.info(f"Assistant: {command.replace('_', ' ')}")
        else:
            logger.warning(f"Method {method_name} not found on modeling widget")

    def _validate_graph(self, sync: bool = False) -> None:
        """Validate the current graph."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        from PySide6.QtCore import QMetaObject, Qt

        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection

        if hasattr(widget, "validate_graph"):
            QMetaObject.invokeMethod(widget, "validate_graph", conn_type)
            logger.info("Assistant: Validating graph")

    def _remove_system(self) -> None:
        """Remove current system in modeling environment."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "system_manager") and hasattr(
            widget.system_manager, "btn_remove_system"
        ):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.system_manager.btn_remove_system, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Removing system")

    def _rename_system(self) -> None:
        """Rename current system in modeling environment."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "system_manager") and hasattr(
            widget.system_manager, "btn_rename_system"
        ):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.system_manager.btn_rename_system, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Renaming system")

    def _next_system(self) -> None:
        """Switch to next system."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "system_manager") and hasattr(
            widget.system_manager, "system_list"
        ):
            lst = widget.system_manager.system_list
            current = lst.currentRow()
            if current < lst.count() - 1:
                lst.setCurrentRow(current + 1)
            logger.info("Assistant: Next system")

    def _previous_system(self) -> None:
        """Switch to previous system."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "system_manager") and hasattr(
            widget.system_manager, "system_list"
        ):
            lst = widget.system_manager.system_list
            current = lst.currentRow()
            if current > 0:
                lst.setCurrentRow(current - 1)
            logger.info("Assistant: Previous system")

    def _auto_connect(self) -> None:
        """Auto-connect nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "auto_connect"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "auto_connect", Qt.QueuedConnection)
            logger.info("Assistant: Auto-connecting nodes")

    def _clear_graph(self, sync: bool = False) -> None:
        """Clear all nodes from the graph."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        from PySide6.QtCore import QMetaObject, Qt

        conn_type = Qt.BlockingQueuedConnection if sync else Qt.QueuedConnection
        if hasattr(widget, "current_graph") and hasattr(
            widget.current_graph, "clear_session"
        ):
            QMetaObject.invokeMethod(widget.current_graph, "clear_session", conn_type)
            logger.info("Assistant: Clearing graph")

    def _select_all_nodes(self) -> None:
        """Select all nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "current_graph"):
            graph = widget.current_graph
            if hasattr(graph, "select_all"):
                graph.select_all()
            logger.info("Assistant: Selecting all nodes")

    def _delete_selected(self) -> None:
        """Delete selected nodes in the graph."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            return
        widget = self.main_window.modeling_widget
        if hasattr(widget, "current_graph") and hasattr(
            widget.current_graph, "delete_selected"
        ):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.current_graph, "delete_selected", Qt.QueuedConnection
            )
            logger.info("Assistant: Deleting selected nodes")

    def _connect_nodes(self, data: dict[str, Any], sync: bool = False) -> Any:
        """Connect two nodes explicitly."""
        params = data.get("params", {})
        from_node_name = params.get("from_node")
        from_port_name = params.get("from_port", "shape")
        to_node_name = params.get("to_node")
        to_port_name = params.get("to_port")

        if not (from_node_name and to_node_name and to_port_name):
            logger.warning("Missing parameters for connect_nodes")
            if sync:
                raise ValueError("Missing parameters for connect_nodes")
            return

        if not self.main_window:
            if sync:
                raise RuntimeError("Main window required")
            return

        # Determine which environment we are in
        # Try CAD first, then Modeling
        target_graph = None
        if (
            hasattr(self.main_window, "cad_widget")
            and self.main_window.cad_widget.isVisible()
        ):
            target_graph = getattr(self.main_window.cad_widget, "graph", None)
        elif hasattr(self.main_window, "modeling_widget"):
            target_graph = getattr(
                self.main_window.modeling_widget, "current_graph", None
            )

        if not target_graph:
            logger.warning("No active graph found for connection")
            if sync:
                raise RuntimeError("No active graph found for connection")
            return

        def do_connect() -> dict[str, Any]:
            try:
                nodes = target_graph.all_nodes()

                def _matches(node: Any, requested: object) -> bool:
                    wanted = str(requested).casefold()
                    return (
                        str(node.name()).casefold() == wanted
                        or str(getattr(node, "id", "") or "").casefold() == wanted
                    )

                src_node = next((n for n in nodes if _matches(n, from_node_name)), None)
                dst_node = next((n for n in nodes if _matches(n, to_node_name)), None)

                if src_node and dst_node:
                    out = src_node.get_output(from_port_name)
                    inp = dst_node.get_input(to_port_name)
                    if out and inp:
                        out.connect_to(inp)
                        logger.info(
                            f"Connected {from_node_name}.{from_port_name} -> {to_node_name}.{to_port_name}"
                        )
                        return {
                            "success": True,
                            "connection": (
                                f"{src_node.name()}.{from_port_name} -> "
                                f"{dst_node.name()}.{to_port_name}"
                            ),
                        }
                    else:
                        msg = f"Ports not found: {from_port_name} -> {to_port_name}"
                        logger.warning(msg)
                        if sync:
                            raise ValueError(msg)
                        return {"success": False, "error": msg}
                else:
                    msg = f"Nodes not found: {from_node_name}, {to_node_name}"
                    logger.warning(msg)
                    if sync:
                        raise ValueError(msg)
                    return {"success": False, "error": msg}
            except Exception as e:
                logger.error(f"Connect failed: {e}")
                if sync:
                    raise
                return {"success": False, "error": str(e)}

        if sync:
            return self._run_sync(do_connect)
        else:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, do_connect)
            return {"success": True, "status": "scheduled"}

    def _modify_system_node(self, data: dict[str, Any], sync: bool = False) -> Any:
        """Wrapper to modify a single system node."""
        params = data.get("params", data)
        node_id = params.get("node_id")
        props = params.get("properties", {})

        if not node_id:
            logger.warning("modify_system_node missing node_id")
            return

        # Reuse build_system_graph which handles updates safely
        build_data = {"params": {"nodes": [{"id": node_id, "properties": props}]}}
        return self._build_system_graph(build_data, sync=sync)

    def _get_graph_state(self, sync: bool = False) -> dict[str, Any]:
        """Get the current graph state (nodes and connections)."""
        if not self.main_window:
            return {}

        # Determine active graph
        graph = None
        mode = "unknown"

        if (
            hasattr(self.main_window, "modeling_widget")
            and self.main_window.modeling_widget.isVisible()
        ):
            graph = getattr(self.main_window.modeling_widget, "current_graph", None)
            mode = "modeling"
        elif (
            hasattr(self.main_window, "cad_widget")
            and self.main_window.cad_widget.isVisible()
        ):
            graph = getattr(self.main_window.cad_widget, "graph", None)
            mode = "cad"

        if not graph:
            return {"error": "No active graph found"}

        def get_state_safe() -> dict[str, Any]:
            nodes_data = []
            for n in graph.all_nodes():
                # Filter useful properties
                props = n.properties()
                filtered_props = {
                    k: v for k, v in props.items() if not k.startswith("_")
                }

                n_data = {"id": n.name(), "type": n.type_, "properties": filtered_props}
                nodes_data.append(n_data)

            return {"mode": mode, "node_count": len(nodes_data), "nodes": nodes_data}

        if sync:
            return self._run_sync(get_state_safe)

        # Async not supported effectively for return values in this architecture
        return {}

    def _set_property(self, data: dict[str, Any]) -> None:
        """Set a property on a specific node."""
        params = data.get("params", {})
        node_name = params.get("node_name")
        prop_name = params.get("property")
        prop_value = params.get("value")

        if not (node_name and prop_name):
            logger.warning("Missing parameters for set_property")
            return

        if not self.main_window:
            return

        target_graph = None
        if (
            hasattr(self.main_window, "cad_widget")
            and self.main_window.cad_widget.isVisible()
        ):
            target_graph = getattr(self.main_window.cad_widget, "graph", None)
        elif hasattr(self.main_window, "modeling_widget"):
            target_graph = getattr(
                self.main_window.modeling_widget, "current_graph", None
            )

        if not target_graph:
            return

        def do_set() -> None:
            try:
                nodes = target_graph.all_nodes()
                node = next((n for n in nodes if n.name() == node_name), None)

                if node:
                    # Use safe setter
                    self._set_node_property_safe(node, prop_name, prop_value, node_name)
                    logger.info(f"Set {node_name}.{prop_name} = {prop_value}")
                else:
                    logger.warning(f"Node {node_name} not found")
            except Exception as e:
                logger.error(f"Set property failed: {e}")

        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, do_set)

    def _set_node_property_safe(
        self,
        node: Any,
        prop_name: str,
        prop_val: Any,
        node_identifier: str,
        conns_spec: list[dict] | None = None,
    ) -> None:
        """
        Safe, robust, case-insensitive property setter for both CAD and System nodes.
        Handles:
        1. Case-insensitive Name Matching (e.g. 'Operation' -> 'operation')
        2. Enum Synonyms (e.g. 'cut' -> 'Cut')
        3. Special Flags (e.g. 'use_surrogate')
        4. Port Auto-fixing (if conns_spec is provided)
        """
        if not hasattr(node, "set_property"):
            return

        try:
            # --- 0. Special Handling for Surrogate/Flags/Name ---
            if prop_name == "use_surrogate":
                node.set_property("use_surrogate", bool(prop_val))
                return

            if prop_name.lower() == "name":
                node.set_name(str(prop_val))
                logger.info(f"Renamed node {node_identifier} to {prop_val}")
                return

            # --- 1. Property Name Normalization ---
            actual_prop_name = prop_name
            node_props = node.properties()

            # --- 1.1 Property Alias Mapping (Handling LLM Hallucinations) ---
            # Map common generic names to specific internal names
            alias_map = {
                # Box
                "width": ("box_width",),
                "height": ("box_height",),
                "depth": ("box_depth",),
                "length": ("box_length",),
                # Cylinder / sphere
                "radius": ("cyl_radius", "sphere_radius"),
                # Cone
                "radius1": ("bottom_radius",),
                "bottom_radius": ("bottom_radius",),
                "radius2": ("top_radius",),
                "top_radius": ("top_radius",),
                # Torus
                "major_radius": ("major_radius",),
                "minor_radius": ("minor_radius",),
                # Generic Transforms
                "x": ("x_translate",),
                "y": ("y_translate",),
                "z": ("z_translate",),
            }

            for target_alias in alias_map.get(prop_name.lower(), ()):
                if target_alias in node_props:
                    actual_prop_name = target_alias
                    break

            # --- 1.2 Property Name Normalization (Case Insensitive) for remaining ---
            if actual_prop_name not in node_props:
                # Search case-insensitive
                for p in node_props:
                    if p.lower() == actual_prop_name.lower():
                        actual_prop_name = p
                        break

            # --- 2. Value Normalization (Enum/Combo mismatch & Synonyms) ---
            ENUM_MAP = {
                "operation": {
                    "targets": ["Union", "Cut", "Intersect"],
                    "synonyms": {
                        "difference": "Cut",
                        "subtract": "Cut",
                        "subtraction": "Cut",
                        "add": "Union",
                        "addition": "Union",
                        "intersection": "Intersect",
                    },
                },
                "selector_type": {
                    "targets": [
                        "Direction",
                        "NearestToPoint",
                        "Index",
                        "Largest Area",
                        "Tag",
                    ],
                    "synonyms": {},
                },
            }

            final_val = prop_val

            # If we found a matching property and it's in our ENUM map
            if actual_prop_name in ENUM_MAP and isinstance(prop_val, str):
                config = ENUM_MAP[actual_prop_name]
                val_lower = prop_val.lower()

                # Check direct match (case-insensitive)
                found = False
                for opt in config["targets"]:
                    if opt.lower() == val_lower:
                        final_val = opt
                        found = True
                        break

                # Check synonyms if not found
                if not found and val_lower in config["synonyms"]:
                    final_val = config["synonyms"][val_lower]

            # --- 3. Try Setting Property ---
            if actual_prop_name in node_props:
                node.set_property(actual_prop_name, final_val)
            else:
                # --- 4. Property Missing -> Check for Port Mismatch (Auto-fix) ---
                # Only if connection spec list is provided
                if conns_spec is not None:
                    # Use the original prop_name for checking input ports as they might differ
                    input_port = node.get_input(prop_name)
                    if input_port and isinstance(prop_val, str):
                        logger.info(
                            f"Auto-fixing LLM error: Converting property '{prop_name}'='{prop_val}' to connection"
                        )
                        src_port = "sketch" if prop_name == "sketch" else "shape"
                        conns_spec.append(
                            {
                                "from": f"{prop_val}.{src_port}",
                                "to": f"{node_identifier}.{prop_name}",
                            }
                        )
                    else:
                        logger.warning(
                            f"Property '{prop_name}' not found on {node_identifier} (and not a port)"
                        )
                else:
                    logger.warning(
                        f"Property '{prop_name}' not found on {node_identifier}"
                    )

        except Exception as e:
            logger.warning(
                f"Failed to set property {prop_name} on {node_identifier}: {e}"
            )
