# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchRenderingMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging
import os

from PySide6 import QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchRenderingMixin"]


class WorkbenchRenderingMixin:
    def _import_cad(self):
        """Prompt user for a geometry file, add an import node, and set its filepath."""
        try:
            from pylcss.io_manager.cad_io import CADImporter

            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import Geometry File", "", CADImporter.get_filter_string()
            )
            if not filepath:
                return

            ext = os.path.splitext(filepath)[1].lower()
            if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
                node = self._spawn_node(
                    "com.cad.import_step", f"Import {ext.upper()[1:]}"
                )
                if node:
                    node.set_property("filepath", filepath)
                    self._execute_graph()
            elif ext in (".stl", ".obj", ".3mf"):
                node = self._spawn_node(
                    "com.cad.import_stl", f"Import {ext.upper()[1:]}"
                )
                if node:
                    node.set_property("filepath", filepath)
                    self._execute_graph()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Unsupported Format",
                    f"Format {ext} not supported for direct node importing.",
                )
        except Exception as e:
            self.statusBar().showMessage(f"Error importing geometry: {e}")

    def _is_simulation_render_result(self, obj):
        """Return True for objects that should be drawn via render_simulation()."""
        if obj is None:
            return False
        if hasattr(obj, "p") and hasattr(obj, "t"):
            return True
        if isinstance(obj, dict):
            return (
                "mesh" in obj
                or ("vertices" in obj and "faces" in obj)
                or obj.get("type") == "topopt_voxel"
                or (obj.get("type") == "crash" and bool(obj.get("frames")))
            )
        return False

    def _is_renderable_result(self, obj):
        """Return True when the viewer can draw *obj* directly."""
        if obj is None:
            return False
        if self._is_simulation_render_result(obj):
            return True
        if (
            hasattr(obj, "tessellate")
            or hasattr(obj, "val")
            or hasattr(obj, "toCompound")
        ):
            return True
        if hasattr(obj, "ctx") and hasattr(obj.ctx, "pendingWires"):
            return bool(obj.ctx.pendingWires)
        return False

    def _render_result_in_viewer(self, obj):
        """Render a previously resolved object using the right viewer path."""
        if obj is None:
            return
        if self._is_simulation_render_result(obj):
            self.viewer.render_simulation(obj)
            if self.results is not None:
                try:
                    if isinstance(obj, dict):
                        self.results.show_result(obj)
                    elif hasattr(obj, "p") and hasattr(obj, "t"):
                        self.results.show_result(
                            {
                                "type": "mesh",
                                "backend": "Netgen",
                                "mesh": obj,
                                "mesh_quality": getattr(obj, "quality_report", None),
                            }
                        )
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
        elif self._is_2d_sketch(obj):
            self.viewer.render_sketch(obj)
        else:
            self.viewer.render_shape(obj)
        # Remember what's on screen so _on_node_selected can skip re-rendering
        # the same geometry when the user clicks a sibling node that resolves
        # to the same upstream payload.
        self._last_rendered_geom_id = id(obj)

    @staticmethod
    def _port_name(port):
        try:
            return port.name()
        except Exception:
            return ""

    def _preferred_render_ports(self, node):
        cls = node.__class__.__name__
        if cls in ("ConstraintNode", "LoadNode", "PressureLoadNode"):
            return ("target_face", "mesh", "shape")
        if cls == "ImpactConditionNode":
            return ("impact_face", "mesh", "shape")
        if cls in ("SelectFaceNode", "InteractiveSelectFaceNode", "MeshNode"):
            return ("shape",)
        return ()

    @staticmethod
    def _is_topopt_result_consumer(node):
        return False

    @staticmethod
    def _is_topopt_render_result(obj):
        return isinstance(obj, dict) and obj.get("type") in {"topopt_voxel"}

    def _ordered_input_ports(self, node, preferred_names=()):
        try:
            ports = node.input_ports()
            if isinstance(ports, dict):
                ports = list(ports.values())
            else:
                ports = list(ports)
        except Exception:
            return []
        preferred = []
        rest = []
        preferred_names = tuple(preferred_names or ())
        for port in ports:
            if self._port_name(port) in preferred_names:
                preferred.append(port)
            else:
                rest.append(port)
        preferred.sort(key=lambda p: preferred_names.index(self._port_name(p)))
        return preferred + rest

    def _find_upstream_renderable(self, node, visited=None, preferred_ports=()):
        """Walk upstream and return (source_node, renderable_result)."""
        if node is None:
            return None, None
        if visited is None:
            visited = set()
        marker = id(node)
        if marker in visited:
            return None, None
        visited.add(marker)

        result = getattr(node, "_last_result", None)
        if self._is_renderable_result(result):
            return node, result

        ports = self._ordered_input_ports(
            node,
            preferred_ports or self._preferred_render_ports(node),
        )
        for port in ports:
            try:
                connected_ports = list(port.connected_ports())
            except Exception:
                connected_ports = []
            for conn_port in connected_ports:
                upstream = conn_port.node()
                upstream_result = getattr(upstream, "_last_result", None)
                if self._is_renderable_result(upstream_result):
                    return upstream, upstream_result
                source, renderable = self._find_upstream_renderable(upstream, visited)
                if renderable is not None:
                    return source, renderable
        return None, None

    def _find_upstream_topopt_result(self, node, visited=None):
        """Return the nearest cached topology result feeding a topology consumer.

        Unlike _find_upstream_renderable, this intentionally does not keep
        walking into the design-domain mesh/shape when the TopOpt node has not
        run yet.  That prevents uncomputed Validation/CAD nodes from showing an
        unrelated box or base CAD body.
        """
        if node is None:
            return None, None
        if visited is None:
            visited = set()
        marker = id(node)
        if marker in visited:
            return None, None
        visited.add(marker)

        result = getattr(node, "_last_result", None)
        if self._is_topopt_render_result(result):
            return node, result

        try:
            port = node.get_input("topology_result")
        except Exception:
            port = None
        if port is None:
            return None, None

        try:
            connected_ports = list(port.connected_ports())
        except Exception:
            connected_ports = []

        for conn_port in connected_ports:
            upstream = conn_port.node()
            result = getattr(upstream, "_last_result", None)
            if self._is_topopt_render_result(result):
                return upstream, result
            if self._is_topopt_result_consumer(upstream):
                source, renderable = self._find_upstream_topopt_result(
                    upstream, visited
                )
                if renderable is not None:
                    return source, renderable
        return None, None

    def _get_render_context_for_node(self, node):
        """Return the best render target for a selected graph node."""
        result = getattr(node, "_last_result", None)
        if self._is_renderable_result(result):
            return node, result
        return self._find_upstream_renderable(
            node,
            preferred_ports=self._preferred_render_ports(node),
        )

    def _on_node_selected(self, node):
        """Handle node selection.

        NodeGraphQt can fire node_selected rapidly (rubber-band drag,
        keyboard arrow selection, programmatic selection from undo/redo),
        and each tick triggers a synchronous inspector rebuild + viewer
        re-render which causes a perceptible freeze for large meshes.
        Two cheap guards make this robust:
          * Skip everything when the node is already the current selection.
          * Skip the heavy viewer re-render when the geometry to be drawn
            is the same object as what's already on screen — overlays and
            highlights still update because they're cheap.
        """
        if not node:
            return
        # ``deserialize_session`` emits selection signals while nodes are
        # still being created and before their saved connections have all
        # been restored.  Executing a preview from one of those intermediate
        # selections reports bogus missing-input errors for otherwise valid
        # example graphs.
        if self._is_loading:
            return
        if getattr(self, "_batching_study_definition_edit", False):
            return
        if self.properties.current_node is node and self._last_rendered_node is node:
            return

        self.properties.display_node(node)
        self.statusBar().showMessage(f"Selected: {node.name}")

        if self._is_topopt_result_consumer(node):
            own = getattr(node, "_last_result", None)
            if self._is_renderable_result(own):
                source_node, geometry = node, own
            else:
                source_node, geometry = self._find_upstream_topopt_result(node)
        else:
            source_node, geometry = self._get_render_context_for_node(node)

        if geometry is not None:
            self._last_rendered_node = source_node or node
            if getattr(self, "_last_rendered_geom_id", None) != id(geometry):
                self._render_result_in_viewer(
                    geometry
                )  # also updates _last_rendered_geom_id

            # Re-apply face highlights if it's the interactive picker
            if node.__class__.__name__ == "InteractiveSelectFaceNode":
                raw = node.get_property("picked_face_indices") or ""
                idx_list = [
                    int(x.strip()) for x in raw.split(",") if x.strip().isdigit()
                ]
                entity_type = str(node.get_property("entity_type") or "Face").title()
                highlighter = {
                    "Face": "highlight_faces",
                    "Edge": "highlight_edges",
                    "Vertex": "highlight_vertices",
                }.get(entity_type, "highlight_faces")
                if idx_list and hasattr(self.viewer, highlighter):
                    getattr(self.viewer, highlighter)(idx_list)

            # Show BC overlays for this node (load/support highlights + arrows)
            try:
                self._show_bc_for_node(node)
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
        else:
            # Selection is observational. Adding/selecting an incomplete node
            # must never execute the graph or raise setup errors.
            self._last_rendered_node = node
            self.statusBar().showMessage(
                "Not run yet — connect inputs, then press Run."
            )
