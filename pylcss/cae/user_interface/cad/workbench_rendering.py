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
    _UPSTREAM_FIRST_RENDER_NODES = frozenset(
        {
            "ConstraintNode",
            "LoadNode",
            "PressureLoadNode",
            "ImpactConditionNode",
            "TopologySupportNode",
            "TopologyLoadNode",
        }
    )

    def _on_viewer_result_field_changed(self, field_name):
        """Keep toolbar field changes attached to the live cached result.

        The viewer intentionally renders a shallow payload copy. Without this
        bridge, its CAD choice never reaches the TopOpt node, cannot start an
        on-demand B-rep build, and leaves the result summary saying Density.
        """
        node = getattr(self, "_last_rendered_node", None)
        result = getattr(node, "_last_result", None) if node is not None else None
        if not isinstance(result, dict):
            node = self._find_renderable_simulation_node()
            result = (
                getattr(node, "_last_result", None)
                if node is not None
                else None
            )
        if not isinstance(result, dict):
            return

        field = str(field_name)
        result["visualization_mode"] = field
        try:
            self.results.show_result(result)
        except Exception:
            logger.debug(
                "Could not refresh the result summary after a field change.",
                exc_info=True,
            )

        if (
            field == "CAD"
            and result.get("type") == "topopt_voxel"
            and result.get("cad_shape") is None
            and result.get("shape") is None
            and isinstance(result.get("recovered_shape"), dict)
        ):
            # Only an extrusion-constrained result has an exact B-rep to build.
            # Anything else stays on its recovered surface rather than being
            # approximated by smooth patches.
            if str(result.get("extrusion_axis") or "none").strip().lower() not in {
                "x",
                "y",
                "z",
            }:
                self.statusBar().showMessage(
                    "No editable CAD body for this result: a general 3-D load "
                    "path has no exact B-rep. Showing the recovered surface. "
                    "Set the manufacturing process to Extruded for solid CAD."
                )
                return
            # The saved surface stays visible while reconstruction runs. This
            # is post-processing only and never executes the graph or solver.
            try:
                self.properties._preview_topopt_cad(node)
            except Exception:
                logger.exception("Could not start TopOpt CAD reconstruction.")

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
        if isinstance(obj, (list, tuple)) and obj:
            return any(self._is_simulation_render_result(item) for item in obj)
        if hasattr(obj, "p") and hasattr(obj, "t"):
            return True
        if isinstance(obj, dict):
            return (
                obj.get("mesh") is not None
                or (obj.get("vertices") is not None and obj.get("faces") is not None)
                or obj.get("type") in ("topopt_voxel", "fea_component")
                or (obj.get("type") == "crash" and bool(obj.get("frames")))
            )
        return False

    def _is_renderable_result(self, obj):
        """Return True when the viewer can draw *obj* directly."""
        if obj is None:
            return False
        if isinstance(obj, dict):
            if any(
                obj.get(name) is not None
                for name in (
                    "shape",
                    "components",
                    "recovered_shape",
                    "mesh",
                    "vertices",
                )
            ):
                return True
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
        # to the same upstream payload.  Keep the object itself rather than its
        # id(): a freed result's address can be handed to the next run's
        # payload, and the identity test then skipped a render that was needed.
        self._last_rendered_geom = obj

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
        if cls in (
            "TopologySupportNode",
            "TopologyLoadNode",
        ):
            return ("target_region",)
        if cls in (
            "SelectFaceNode",
            "InteractiveSelectFaceNode",
            "MeshNode",
        ):
            return ("shape",)
        if cls in (
            "SolverNode",
            "FEAComponentNode",
            "TopologyOptVoxelNode",
            "LatticeOptVoxelNode",
            "LatticeInfillNode",
            "AssemblyNode",
            "CrashSolverNode",
        ):
            return ("components", "mesh", "shape", "target_region", "design_domain", "constraints", "loads")
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

    def _find_upstream_renderable(
        self,
        node,
        visited=None,
        preferred_ports=(),
        *,
        include_self=True,
    ):
        """Walk upstream and return (source_node, renderable_result)."""
        if node is None:
            return None, None
        path_visited = set(visited) if visited is not None else set()
        marker = id(node)
        if marker in path_visited:
            return None, None
        path_visited.add(marker)

        upstream_first = node.__class__.__name__ in self._UPSTREAM_FIRST_RENDER_NODES
        if include_self and upstream_first:
            for port_name in self._preferred_render_ports(node):
                source, renderable = self._find_renderable_on_input(
                    node, port_name
                )
                if renderable is not None:
                    return source, renderable

        if include_self:
            result = getattr(node, "_last_result", None)
            if self._is_renderable_result(result):
                return node, result

        ports = self._ordered_input_ports(
            node,
            preferred_ports or self._preferred_render_ports(node),
        )
        collected_sources = []
        collected_results = []
        for port in ports:
            try:
                connected_ports = list(port.connected_ports())
            except Exception:
                connected_ports = []
            for conn_port in connected_ports:
                upstream = conn_port.node()
                source, renderable = self._find_upstream_renderable(
                    upstream, path_visited
                )
                if renderable is not None:
                    if isinstance(renderable, list):
                        collected_sources.extend([source] * len(renderable))
                        collected_results.extend(renderable)
                    else:
                        collected_sources.append(source)
                        collected_results.append(renderable)

        unique_sources = []
        unique_results = []
        seen_results = set()
        for source, renderable in zip(collected_sources, collected_results):
            marker = id(renderable)
            if marker in seen_results:
                continue
            seen_results.add(marker)
            unique_sources.append(source)
            unique_results.append(renderable)

        if not unique_results:
            return None, None
        if len(unique_results) == 1:
            return unique_sources[0], unique_results[0]
        return node, unique_results

    def _find_renderable_on_input(self, node, port_name):
        """Return renderable values reached through one named input port."""
        results = []
        sources = []
        for port in self._ordered_input_ports(node, (port_name,)):
            if self._port_name(port) != port_name:
                continue
            try:
                connections = list(port.connected_ports())
            except Exception:
                connections = []
            for connection in connections:
                source, renderable = self._find_upstream_renderable(
                    connection.node()
                )
                if renderable is None:
                    continue
                if isinstance(renderable, list):
                    sources.extend([source] * len(renderable))
                    results.extend(renderable)
                else:
                    sources.append(source)
                    results.append(renderable)
        if not results:
            return None, None
        if len(results) == 1:
            return sources[0], results[0]
        return node, results

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
        upstream_first = node.__class__.__name__ in self._UPSTREAM_FIRST_RENDER_NODES
        if self._is_renderable_result(result) and not upstream_first:
            return node, result
        if upstream_first:
            for port_name in self._preferred_render_ports(node):
                source, renderable = self._find_renderable_on_input(
                    node, port_name
                )
                if renderable is not None:
                    return source, renderable
        source, renderable = self._find_upstream_renderable(
            node,
            preferred_ports=self._preferred_render_ports(node),
            include_self=False,
        )
        if renderable is not None:
            return source, renderable
        if self._is_renderable_result(result):
            return node, result
        return None, None

    def _clear_selection_visuals(self):
        """Clear viewer-only state after the graph selection becomes empty."""
        self._last_rendered_node = None
        self._last_rendered_geom = None
        try:
            self.viewer.clear()
            self.viewer.clear_cached_results()
        except Exception:
            logger.debug("Could not clear the deselected viewer", exc_info=True)
        try:
            self.properties.display_node(None)
        except Exception:
            logger.debug("Could not clear the node inspector", exc_info=True)
        try:
            self.results.clear_results()
        except Exception:
            logger.debug("Could not clear the results panel", exc_info=True)

    def _on_node_selection_changed(self, selected, deselected):
        """Clear the 3-D view when the last selected graph node is unselected."""
        if self._is_loading or selected:
            return
        try:
            if self.graph.selected_nodes():
                return
        except Exception:
            logger.debug("Could not inspect graph selection", exc_info=True)
            return
        self._clear_selection_visuals()

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
        if self.properties.current_node is node and self._last_rendered_node is node:
            return

        self.properties.display_node(node)

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
            if getattr(self, "_last_rendered_geom", None) is not geometry:
                self._render_result_in_viewer(
                    geometry
                )  # also updates _last_rendered_geom

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
            try:
                pending = node.get_pending() if node.has_pending() else None
            except Exception:
                pending = None
            self.statusBar().showMessage(
                str(pending or "Not run. Connect inputs and press Run.")
            )
