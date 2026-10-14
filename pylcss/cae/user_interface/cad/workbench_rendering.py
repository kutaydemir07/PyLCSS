# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchRenderingMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging
import os

from PySide6 import QtWidgets

from pylcss.cae.design_studio.topology_optimization.integration.study_identity import (
    is_density_study_class,
)


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

        The viewer renders a shallow payload copy. Keep its selected field
        attached to the node so the result summary and inspector agree.
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
        from .result_fields import activate_modal_mode
        activate_modal_mode(result, field)
        result["visualization_mode"] = field
        try:
            self.results.show_result(result)
        except Exception:
            logger.debug(
                "Could not refresh the result summary after a field change.",
                exc_info=True,
            )

    def _on_results_view_mode_requested(self, field_name: str):
        """Handle 3-D view mode requested from ResultsPanel or popup dialog."""
        if getattr(self, "viewer", None) is not None:
            self.viewer.set_result_field(field_name)

    def _on_results_undeformed_requested(self, visible: bool):
        """Handle undeformed wireframe requested from ResultsPanel or popup dialog."""
        if getattr(self, "viewer", None) is not None:
            self.viewer.set_undeformed_overlay(visible)

    def _on_results_extrema_requested(self, visible: bool):
        """Handle min/max extrema markers requested from ResultsPanel or popup dialog."""
        if getattr(self, "viewer", None) is not None:
            self.viewer.set_result_extrema(visible)

    def _on_results_deform_scale_requested(self, scale: str):
        """Handle deformation exaggeration scale requested from ResultsPanel or popup dialog."""
        if getattr(self, "viewer", None) is not None:
            self.viewer.set_deformation_scale(scale)

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
        self.viewer._section_show_interior = False
        if isinstance(obj, dict) and obj.get('type') == 'topopt_voxel':
            from .result_fields import topopt_view_mode, topopt_view_result
            field = topopt_view_mode(obj, obj.get('visualization_mode') or 'STL Shape')
            obj['visualization_mode'] = field
            options = topopt_view_result(obj, field).get('structure_options')
            mode = options.get('mode', 'solid') if isinstance(options, dict) else getattr(options, 'mode', 'solid')
            self.viewer._section_show_interior = mode != 'solid'
        if self._is_simulation_render_result(obj):
            self.viewer.render_simulation(obj)
            if isinstance(obj, dict) and obj.get('type') == 'topopt_voxel':
                geometry_view = field in {'Lattice', 'STL Shape', 'CAD'}
                if mode != 'solid' and geometry_view and getattr(self, '_last_rendered_geom', None) is not obj:
                    self.viewer._section_axis.setCurrentText('Z')
                    self.viewer._section_slider.setValue(50)
                    self.viewer._btn_section.setChecked(True)
                    self.viewer._update_section_cut()
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
                    self._show_results_panel()
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
        selected = getattr(self.properties, "current_node", None)
        if selected is None or not selected.has_property("non_design_regions"):
            self.viewer.render_non_design_layers([])

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
            "ModalSolverNode",
            "FEAComponentNode",
            "TopologyOptVoxelNode",
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

    def _is_results_relevant_node(self, node) -> bool:
        """Return True if the node is a solver/study or measurement node."""
        if node is None:
            return False
        class_name = getattr(node, "__class__", type(node)).__name__
        if class_name in {
            "SolverNode",
            "ModalSolverNode",
            "CrashSolverNode",
            "TopologyOptVoxelNode",
            "LatticeInfillNode",
        }:
            return True
        if is_density_study_class(class_name):
            return True
        measure_classes = getattr(self, "_MEASURE_NODE_CLASSES", ())
        if class_name in measure_classes:
            return True
        last_res = getattr(node, "_last_result", None)
        if isinstance(last_res, dict) and last_res.get("type") in {
            "fea",
            "modal",
            "crash",
            "external_solver",
            "topopt_voxel",
            "lattice_infill",
        }:
            return True
        return False

    def _show_results_panel(self):
        """Reveal the 3D view mode selector panel upside in the inspector."""
        results = getattr(self, "results", None)
        if results is None:
            return
        if not results.isVisible():
            results.setVisible(True)
        if hasattr(results, "_header_bar"):
            results._header_bar.setVisible(True)

    def _update_results_panel_visibility(self, node=None):
        """Toggle results panel based on whether the selected node produces results."""
        results = getattr(self, "results", None)
        if results is None:
            return
        if node is not None and self._is_results_relevant_node(node):
            self._show_results_panel()
        else:
            results.setVisible(False)

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
        self._update_results_panel_visibility(None)

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
            is the same object as what's already on screen â€” overlays and
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
        self._update_results_panel_visibility(node)
        if not node.has_property("non_design_regions"):
            self.viewer.render_non_design_layers([])

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
                raw = node.get_property("picked_entity_indices") or ""
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
            if node.has_property("non_design_regions"):
                self._preview_topopt_non_design(node, cached_only=True)
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
